#include "gdn-fused.cuh"
#include "unary.cuh"

// ---------------------------------------------------------------------------------------------------------------------
// GGML_OP_GDN_CONV: one block per group of channels (a q or k head, or as many v channels) and sequence, one thread
// per channel. Each thread keeps its channel's window in registers: the history from the sequence's cache row, then
// one x value per token. For a normalized group the block sums the squares of each token's row the way rms_norm's
// vector path does for a row of up to 128 floats (a float4 per lane, then one warp sum), so a q or k head of 128
// comes out bit for bit as RMS_NORM+SCALE makes it.
// ---------------------------------------------------------------------------------------------------------------------

// the sum of squares of the group's row in ys, as rms_norm_f32<32, 1, true>; every thread gets it
static __device__ __forceinline__ float gdn_row_sum_sq(const float * ys, const int group, float * sum) {
    __syncthreads(); // ys written
    if (threadIdx.x < WARP_SIZE) {
        const float4 * y4  = (const float4 *) ys;
        float          tmp = 0.0f;
        for (int i = threadIdx.x; i < group / 4; i += WARP_SIZE) {
            const float4 v = y4[i];
            tmp += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
        }
        tmp = warp_reduce_sum(tmp);
        if (threadIdx.x == 0) {
            *sum = tmp;
        }
    }
    __syncthreads(); // sum written; ys and sum are rewritten only after the next call's first barrier
    return *sum;
}

template <int d_conv>
static __global__ void gdn_conv_f32(
        const float * __restrict__ cache, const int32_t * __restrict__ ids, const int64_t cache_s1,
        const float * __restrict__ x, const int64_t x_s1, const int64_t x_s2,
        const float * __restrict__ c, const int64_t c_s1,
        float * __restrict__ out, float * tails, const int64_t tail_seq_stride, const int64_t tail_slot_stride,
        const int channels, const int n_tokens, const int n_norm, const int group,
        const float norm_eps, const float norm_scale, const int K) {
    constexpr int n_hist = d_conv - 1;

    __shared__ __align__(16) float row[1024];
    __shared__ float sum;

    const int  ch     = blockIdx.x * group + threadIdx.x;
    const int  seq    = blockIdx.y;
    const bool active = ch < channels;
    const bool norm   = (blockIdx.x + 1) * group <= n_norm; // the same for the whole block

    float w[d_conv];
    float hist[n_hist]; // the history as read, for the slots that reach back into it
    float win[d_conv];  // win[0] the oldest column of the current window

    ggml_cuda_pdl_sync();
    if (active) {
        // read before anything is written: with the histories written into the cache (one sequence), the row may be
        // the one this thread writes, but only this thread reads or writes this channel of it
        const float * h = cache + (int64_t) ids[seq] * cache_s1 + (int64_t) ch * n_hist;
#pragma unroll
        for (int j = 0; j < n_hist; ++j) {
            hist[j] = h[j];
            win[j]  = hist[j];
        }
#pragma unroll
        for (int j = 0; j < d_conv; ++j) {
            w[j] = c[(int64_t) ch * c_s1 + j];
        }
    }

    const float * xs = x + (int64_t) seq * x_s2 + ch;
    float *       yo = out + (int64_t) seq * n_tokens * channels + ch;

    for (int t = 0; t < n_tokens; ++t) {
        float y = 0.0f;
        if (active) {
            win[n_hist] = xs[(int64_t) t * x_s1];
            float sumf = 0.0f;
#pragma unroll
            for (int j = 0; j < d_conv; ++j) {
                sumf += win[j] * w[j];
            }
            y = ggml_cuda_op_silu_single(sumf);
#pragma unroll
            for (int j = 0; j < n_hist; ++j) {
                win[j] = win[j + 1];
            }
        }
        if (norm) {
            // as rms_norm(y, eps/group) then scale(1/sqrt(group)): (scale * y) * scale_out
            row[threadIdx.x] = y;
            const float scale = rsqrtf(gdn_row_sum_sq(row, group, &sum) / group + norm_eps);
            y = scale * y;
            y *= norm_scale;
        }
        if (active) {
            yo[(int64_t) t * channels] = y;
        }
    }

    if (!active) {
        return;
    }

    // slot k is the window that ends k tokens before the end of the ubatch: columns start .. start + n_hist - 1 of
    // history + x, start = max(0, n_tokens - k)
    float * tail = tails + (int64_t) seq * tail_seq_stride + (int64_t) ch * n_hist;
    for (int k = 0; k < K; ++k) {
        const int start = max(0, n_tokens - k);
#pragma unroll
        for (int j = 0; j < n_hist; ++j) {
            const int col = start + j;
            float v = 0.0f;
            if (col < n_hist) {
#pragma unroll
                for (int i = 0; i < n_hist; ++i) {
                    v = i == col ? hist[i] : v;
                }
            } else {
                v = xs[(int64_t) (col - n_hist) * x_s1];
            }
            tail[(int64_t) k * tail_slot_stride + j] = v;
        }
    }
}

bool ggml_cuda_gdn_conv_supported(const ggml_tensor * op) {
    const ggml_tensor * cache = op->src[0];
    const ggml_tensor * x     = op->src[2];
    const ggml_tensor * c     = op->src[3];
    const int group = ggml_get_op_params_i32(op, 1);
    const int64_t d_conv = c->ne[0];
    return cache->nb[0] == sizeof(float) && x->nb[0] == sizeof(float) && c->nb[0] == sizeof(float) &&
           group % WARP_SIZE == 0 && group <= 1024 && d_conv >= 2 && d_conv <= 8 &&
           x->ne[1] <= 0x7fffffff && c->ne[1] <= 0x7fffffff;
}

void ggml_cuda_op_gdn_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gdn_conv_fused_cache * cache) {
    const ggml_tensor * src_cache = dst->src[0];
    const ggml_tensor * src_ids   = dst->src[1];
    const ggml_tensor * src_x     = dst->src[2];
    const ggml_tensor * src_c     = dst->src[3];

    const int   n_norm = ggml_get_op_params_i32(dst, 0);
    const int   group  = ggml_get_op_params_i32(dst, 1);
    const float eps    = ggml_get_op_params_f32(dst, 2);
    const int   K      = ggml_get_op_params_i32(dst, 3);

    const int64_t d_conv   = src_c->ne[0];
    const int64_t channels = src_c->ne[1];
    const int64_t n_tokens = src_x->ne[1];
    const int64_t n_seqs   = src_x->ne[2];
    const int64_t n_hist   = d_conv - 1;

    GGML_ASSERT(ggml_cuda_gdn_conv_supported(dst));

    float * out              = (float *) dst->data;
    float * tails            = out + channels * n_tokens * n_seqs;
    int64_t tail_seq_stride  = n_hist * channels;
    int64_t tail_slot_stride = n_hist * channels * n_seqs;
    if (cache != nullptr) {
        tails            = cache->data;
        tail_seq_stride  = cache->seq_stride;
        tail_slot_stride = cache->slot_stride;
    }

    // as build_gdn_l2_norm
    const float norm_eps   = eps / (float) group;
    const float norm_scale = 1.0f / sqrtf((float) group);

    const dim3 grid((channels + group - 1) / group, n_seqs, 1);
    const dim3 block(group, 1, 1);
    cudaStream_t stream = ctx.stream();

#define GDN_CONV_LAUNCH(D)                                                                                           \
    case D:                                                                                                          \
        gdn_conv_f32<D><<<grid, block, 0, stream>>>(                                                                 \
            (const float *) src_cache->data, (const int32_t *) src_ids->data, src_cache->nb[1] / sizeof(float),      \
            (const float *) src_x->data, src_x->nb[1] / sizeof(float), src_x->nb[2] / sizeof(float),                 \
            (const float *) src_c->data, src_c->nb[1] / sizeof(float),                                               \
            out, tails, tail_seq_stride, tail_slot_stride,                                                           \
            (int) channels, (int) n_tokens, n_norm, group, norm_eps, norm_scale, K);                                 \
        break;

    switch (d_conv) {
        GDN_CONV_LAUNCH(2)
        GDN_CONV_LAUNCH(3)
        GDN_CONV_LAUNCH(4)
        GDN_CONV_LAUNCH(5)
        GDN_CONV_LAUNCH(6)
        GDN_CONV_LAUNCH(7)
        GDN_CONV_LAUNCH(8)
        default:
            GGML_ABORT("unsupported d_conv %d", (int) d_conv);
    }
#undef GDN_CONV_LAUNCH
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------------------------------
// gated rms norm: one warp per row of up to 128 floats, a float4 per lane; the same sums and products, in the same
// order, as the RMS_NORM+MUL fusion's vector path (rms_norm_f32<32, 1, true>), the SIGMOID and the MUL it replaces,
// so the result is bit for bit theirs
// ---------------------------------------------------------------------------------------------------------------------

static __global__ void gated_rms_norm_f32(
        const float * __restrict__ x, const int64_t x_s1, const int64_t x_s2, const int64_t x_s3,
        const float * __restrict__ w,
        const float * __restrict__ z, const int64_t z_s1, const int64_t z_s2, const int64_t z_s3,
        float * __restrict__ dst, const int ncols, const int64_t ne1, const int64_t ne2, const int64_t nrows,
        const float eps) {
    const int64_t row = (int64_t) blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    if (row >= nrows) {
        return; // the whole warp
    }
    const int lane = threadIdx.x % WARP_SIZE;

    const int64_t i1 = row % ne1;
    const int64_t i2 = (row / ne1) % ne2;
    const int64_t i3 = row / (ne1 * ne2);

    const float4 * x4 = (const float4 *) (x + i1 * x_s1 + i2 * x_s2 + i3 * x_s3);
    const float4 * w4 = (const float4 *) w;
    const float *  zr = z + i1 * z_s1 + i2 * z_s2 + i3 * z_s3;
    float *        dr = dst + row * ncols;

    const int ncols4 = ncols / 4;

    ggml_cuda_pdl_sync();
    float4 xs  = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float  tmp = 0.0f;
    if (lane < ncols4) {
        xs = x4[lane];
        tmp += xs.x*xs.x + xs.y*xs.y + xs.z*xs.z + xs.w*xs.w;
    }
    tmp = warp_reduce_sum(tmp);
    const float scale = rsqrtf(tmp / ncols + eps);

    if (lane < ncols4) {
        float4 r = make_float4(scale*xs.x, scale*xs.y, scale*xs.z, scale*xs.w);
        const float4 m = w4[lane];
        r.x *= m.x;
        r.y *= m.y;
        r.z *= m.z;
        r.w *= m.w;
        const float * zc = zr + 4*lane;
        float *       dc = dr + 4*lane;
        dc[0] = r.x * (1.0f / (1.0f + expf(-zc[0])));
        dc[1] = r.y * (1.0f / (1.0f + expf(-zc[1])));
        dc[2] = r.z * (1.0f / (1.0f + expf(-zc[2])));
        dc[3] = r.w * (1.0f / (1.0f + expf(-zc[3])));
    }
}

bool ggml_cuda_gated_rms_norm_supported(const ggml_tensor * rms_norm, const ggml_tensor * mul_w,
                                        const ggml_tensor * sigmoid, const ggml_tensor * mul_gate) {
    const ggml_tensor * x = rms_norm->src[0];
    const ggml_tensor * w = mul_w->src[0] == rms_norm ? mul_w->src[1] : mul_w->src[0];
    const ggml_tensor * z = sigmoid->src[0];

    auto rows_f32 = [](const ggml_tensor * t) {
        return t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float);
    };
    // the vector path's float4 loads: 16-byte rows and bases (no data yet in graph_optimize: checked again at compute)
    auto rows_16 = [](const ggml_tensor * t) {
        return ((uintptr_t) t->data & 15) == 0 && t->nb[1] % 16 == 0 && t->nb[2] % 16 == 0 && t->nb[3] % 16 == 0;
    };

    return rows_f32(x) && rows_f32(w) && rows_f32(z) && rows_f32(mul_gate) &&
           ggml_is_contiguous(w) && ggml_nrows(w) == 1 && w->ne[0] == x->ne[0] &&
           ggml_are_same_shape(x, z) && ggml_are_same_shape(x, mul_gate) && ggml_is_contiguous(mul_gate) &&
           x->ne[0] % 4 == 0 && x->ne[0] <= 4 * WARP_SIZE &&
           rows_16(x) && rows_16(w) && rows_16(mul_gate);
}

void ggml_cuda_op_gated_rms_norm(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm,
                                 const ggml_tensor * mul_w, const ggml_tensor * sigmoid, ggml_tensor * mul_gate) {
    const ggml_tensor * x = rms_norm->src[0];
    const ggml_tensor * w = mul_w->src[0] == rms_norm ? mul_w->src[1] : mul_w->src[0];
    const ggml_tensor * z = sigmoid->src[0];

    float eps;
    memcpy(&eps, rms_norm->op_params, sizeof(float));

    const int64_t ncols = x->ne[0];
    const int64_t nrows = ggml_nrows(x);

    constexpr int rows_per_block = 4;
    const dim3 grid((nrows + rows_per_block - 1) / rows_per_block, 1, 1);
    const dim3 block(rows_per_block * WARP_SIZE, 1, 1);

    gated_rms_norm_f32<<<grid, block, 0, ctx.stream()>>>(
        (const float *) x->data, x->nb[1] / sizeof(float), x->nb[2] / sizeof(float), x->nb[3] / sizeof(float),
        (const float *) w->data,
        (const float *) z->data, z->nb[1] / sizeof(float), z->nb[2] / sizeof(float), z->nb[3] / sizeof(float),
        (float *) mul_gate->data, (int) ncols, x->ne[1], x->ne[2], nrows, eps);
    CUDA_CHECK(cudaGetLastError());
}
