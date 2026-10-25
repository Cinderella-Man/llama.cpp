#include "fattn-pascal.cuh"
#include "fattn-common.cuh"

#include <cstdlib>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

namespace {

constexpr int FAP_D    = 128; // head size of K and V
constexpr int FAP_G    = 8;   // query heads per KV head
constexpr int FAP_NW   = 4;   // warps per block, one KV head each
constexpr int FAP_TILE = 32;  // keys per softmax tile (one mask value per lane)
constexpr int FAP_MAX_COUNTERS = 64;
constexpr int FAP_PART   = FAP_D + 4; // floats per partial: FAP_D outputs, the maximum and the sum, padded to 16 bytes
constexpr float FAP_LOG2_E = 1.4426950408889634f;

// Byte strides; the query, K, V and the mask row are those of token 0 and sequence 0.
struct fap_args {
    const char  * Q;
    int64_t       q_nb2;          // query head stride
    const char  * K;
    int64_t       k_nb1, k_nb2;   // key stride, KV head stride
    const char  * V;
    int64_t       v_nb1, v_nb2;
    const half  * mask;           // n_kv values, or nullptr
    const float * sinks;          // one per query head, or nullptr
    float       * dst;            // FAP_D values per query head
    float       * part;           // per (block x, KV head, query head): FAP_D partial outputs, then the maximum and sum
    unsigned    * counter;        // one per block column (blockIdx.y), zero between launches
    int           n_kv;
    int           keys_per_block; // a multiple of FAP_TILE
    float         scale_log2;     // the softmax scale times log2(e)
    int           fault;          // GGML_CUDA_FATTN_PASCAL_FAULT, for testing the tests: 1 output x 1.01, 2 sinks dropped,
                                  // 3 every block's last key tile dropped
};

// 4 halves at p (8-byte aligned) as floats
static __device__ __forceinline__ float4 fap_load_h4(const char * p) {
    const uint2 u = __ldg((const uint2 *) p);
    const float2 a = __half22float2(*(const half2 *) &u.x);
    const float2 b = __half22float2(*(const half2 *) &u.y);
    return make_float4(a.x, a.y, b.x, b.y);
}

// v[32] holds each lane's partial sums of 32 values; afterwards v[0] of lane l is the sum over all lanes of value l.
static __device__ __forceinline__ float fap_transpose_reduce(float v[32]) {
    const int lane = threadIdx.x;
#pragma unroll
    for (int o = 16, n = 32; o > 0; o >>= 1, n >>= 1) {
        const bool upper = lane & o;
#pragma unroll
        for (int i = 0; i < n/2; ++i) {
            const float send = upper ? v[i] : v[i + n/2];
            const float keep = upper ? v[i + n/2] : v[i];
            v[i] = keep + __shfl_xor_sync(0xFFFFFFFF, send, o, WARP_SIZE);
        }
    }
    return v[0];
}

__launch_bounds__(FAP_NW*WARP_SIZE, 3)
static __global__ void flash_attn_ext_pascal_f16(const fap_args a) {
    const int lane = threadIdx.x;
    const int g    = blockIdx.y*FAP_NW + threadIdx.y; // KV head
    const int hl   = lane & (FAP_G - 1);             // the query head (within g) whose scores this lane keeps

    __shared__ float p_sm[FAP_NW][FAP_TILE*FAP_G];
    __shared__ bool  last_block;

    // the query of the 8 heads, dimensions 4*lane .. 4*lane + 3
    float4 q[FAP_G];
#pragma unroll
    for (int h = 0; h < FAP_G; ++h) {
        const float4 t = *(const float4 *) (a.Q + (int64_t) (g*FAP_G + h)*a.q_nb2 + 16*lane);
        q[h] = make_float4(t.x*a.scale_log2, t.y*a.scale_log2, t.z*a.scale_log2, t.w*a.scale_log2);
    }

    float4 acc[FAP_G];
#pragma unroll
    for (int h = 0; h < FAP_G; ++h) {
        acc[h] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    float M = -FLT_MAX/2.0f; // running maximum (base 2) of head hl
    float L = 0.0f;          // running sum of head hl

    const int k_begin = blockIdx.x*a.keys_per_block;
    const int k_end   = min(k_begin + a.keys_per_block, a.n_kv);

    const char * Kg = a.K + (int64_t) g*a.k_nb2 + 8*lane;
    const char * Vg = a.V + (int64_t) g*a.v_nb2 + 8*lane;

    for (int k0 = k_begin; k0 < k_end - (a.fault == 3 ? FAP_TILE : 0); k0 += FAP_TILE) {
        // one mask value per lane, in base 2; a tile masked for every key is skipped
        float mk = 0.0f;
        if (a.mask) {
            mk = __half2float(a.mask[k0 + lane]);
            if (__all_sync(0xFFFFFFFF, mk == -INFINITY)) {
                continue;
            }
            mk *= FAP_LOG2_E;
        }

        // scores: s[j4] of lane l belongs to key k0 + 4*j4 + l/8 and query head l%8
        float s[FAP_TILE/4];
#pragma unroll
        for (int j4 = 0; j4 < FAP_TILE/4; ++j4) {
            float v[32];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float4 kf = fap_load_h4(Kg + (int64_t) (k0 + 4*j4 + j)*a.k_nb1);
#pragma unroll
                for (int h = 0; h < FAP_G; ++h) {
                    float t = q[h].x*kf.x;
                    t = fmaf(q[h].y, kf.y, t);
                    t = fmaf(q[h].z, kf.z, t);
                    t = fmaf(q[h].w, kf.w, t);
                    v[j*FAP_G + h] = t;
                }
            }
            s[j4] = fap_transpose_reduce(v) + __shfl_sync(0xFFFFFFFF, mk, 4*j4 + lane/FAP_G, WARP_SIZE);
        }

        // online softmax of head hl over the tile: lanes l, l^8, l^16, l^24 share a head
        float m_tile = s[0];
#pragma unroll
        for (int j4 = 1; j4 < FAP_TILE/4; ++j4) {
            m_tile = fmaxf(m_tile, s[j4]);
        }
        m_tile = fmaxf(m_tile, __shfl_xor_sync(0xFFFFFFFF, m_tile,  8, WARP_SIZE));
        m_tile = fmaxf(m_tile, __shfl_xor_sync(0xFFFFFFFF, m_tile, 16, WARP_SIZE));
        const float M_new = fmaxf(M, m_tile);
        const float scale_old = exp2f(M - M_new);
        M = M_new;

        float l_tile = 0.0f;
#pragma unroll
        for (int j4 = 0; j4 < FAP_TILE/4; ++j4) {
            const float p = exp2f(s[j4] - M);
            l_tile += p;
            p_sm[threadIdx.y][(4*j4 + lane/FAP_G)*FAP_G + hl] = p;
        }
        l_tile += __shfl_xor_sync(0xFFFFFFFF, l_tile,  8, WARP_SIZE);
        l_tile += __shfl_xor_sync(0xFFFFFFFF, l_tile, 16, WARP_SIZE);
        L = L*scale_old + l_tile;

        // the accumulators of every head, rescaled by their head's factor (held by lane h)
#pragma unroll
        for (int h = 0; h < FAP_G; ++h) {
            const float f = __shfl_sync(0xFFFFFFFF, scale_old, h, WARP_SIZE);
            acc[h].x *= f;
            acc[h].y *= f;
            acc[h].z *= f;
            acc[h].w *= f;
        }
        __syncwarp();

        // P·V: each V row once, for the 8 heads
#pragma unroll 8
        for (int kk = 0; kk < FAP_TILE; ++kk) {
            const float4 vf = fap_load_h4(Vg + (int64_t) (k0 + kk)*a.v_nb1);
            const float4 p0 = *(const float4 *) &p_sm[threadIdx.y][kk*FAP_G + 0];
            const float4 p1 = *(const float4 *) &p_sm[threadIdx.y][kk*FAP_G + 4];
            const float p[FAP_G] = { p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w };
#pragma unroll
            for (int h = 0; h < FAP_G; ++h) {
                acc[h].x = fmaf(p[h], vf.x, acc[h].x);
                acc[h].y = fmaf(p[h], vf.y, acc[h].y);
                acc[h].z = fmaf(p[h], vf.z, acc[h].z);
                acc[h].w = fmaf(p[h], vf.w, acc[h].w);
            }
        }
        __syncwarp();
    }

    // this block's partial outputs, maxima and sums
    const int n_head_kv = gridDim.y*FAP_NW;
    const int64_t part_stride = FAP_PART;
    float * part = a.part + ((int64_t) blockIdx.x*n_head_kv + g)*FAP_G*part_stride;
#pragma unroll
    for (int h = 0; h < FAP_G; ++h) {
        *(float4 *) (part + h*part_stride + 4*lane) = acc[h];
    }
    if (lane < FAP_G) {
        part[lane*part_stride + FAP_D + 0] = M;
        part[lane*part_stride + FAP_D + 1] = L;
    }

    // the last block of this column merges every block's partials
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        last_block = atomicAdd(&a.counter[blockIdx.y], 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last_block) {
        return;
    }
    __threadfence();

    for (int h = 0; h < FAP_G; ++h) {
        const float * ph = a.part + ((int64_t) g*FAP_G + h)*part_stride;
        const int64_t s_stride = (int64_t) n_head_kv*FAP_G*part_stride;

        float m_all = -FLT_MAX/2.0f;
        for (int b = 0; b < (int) gridDim.x; ++b) {
            m_all = fmaxf(m_all, __ldcg(ph + b*s_stride + FAP_D));
        }
        float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float  l = 0.0f;
        for (int b = 0; b < (int) gridDim.x; ++b) {
            const float f = exp2f(__ldcg(ph + b*s_stride + FAP_D) - m_all);
            const float4 pb = __ldcg((const float4 *) (ph + b*s_stride + 4*lane));
            o.x = fmaf(f, pb.x, o.x);
            o.y = fmaf(f, pb.y, o.y);
            o.z = fmaf(f, pb.z, o.z);
            o.w = fmaf(f, pb.w, o.w);
            l = fmaf(f, __ldcg(ph + b*s_stride + FAP_D + 1), l);
        }
        if (a.sinks && a.fault != 2) {
            const float sink = a.sinks[g*FAP_G + h]*FAP_LOG2_E;
            const float m2 = fmaxf(m_all, sink);
            const float f  = exp2f(m_all - m2);
            l = l*f + exp2f(sink - m2);
            o.x *= f;
            o.y *= f;
            o.z *= f;
            o.w *= f;
        }
        const float inv = (a.fault == 1 ? 1.01f : 1.0f)/l;
        *(float4 *) (a.dst + (int64_t) (g*FAP_G + h)*FAP_D + 4*lane) = make_float4(o.x*inv, o.y*inv, o.z*inv, o.w*inv);
    }

    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        a.counter[blockIdx.y] = 0;
    }
}

static int fap_mode() {
    static const int mode = [] {
        const char * env = getenv("GGML_CUDA_FATTN_PASCAL");
        return env ? atoi(env) : 0;
    }();
    return mode;
}

static int fap_fault() {
    static const int fault = [] {
        const char * env = getenv("GGML_CUDA_FATTN_PASCAL_FAULT");
        return env ? atoi(env) : 0;
    }();
    return fault;
}

static bool fap_aligned(const void * p, const int64_t al) {
    return (uintptr_t) p % al == 0;
}

} // namespace

bool ggml_cuda_flash_attn_ext_pascal(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int mode = fap_mode();
    const int cc   = ggml_cuda_info().devices[ctx.device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || !(mode == 2 || (mode == 1 && cc == GGML_CUDA_CC_DP4A))) {
        return false;
    }

    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);

    // one token of one sequence, head size 128, F16 K and V, 8 query heads per KV head, in groups of 4 KV heads
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F32 ||
            Q->ne[0] != FAP_D || K->ne[0] != FAP_D || V->ne[0] != FAP_D || Q->ne[1] != 1 || Q->ne[3] != 1 ||
            K->ne[3] != 1 || V->ne[3] != 1 || K->ne[2] % FAP_NW != 0 || K->ne[2]/FAP_NW > FAP_MAX_COUNTERS ||
            Q->ne[2] != K->ne[2]*FAP_G || V->ne[2] != K->ne[2] || V->ne[1] != K->ne[1] ||
            K->ne[1] % FATTN_KQ_STRIDE != 0 || max_bias != 0.0f || logit_softcap != 0.0f || n_kv_max != 0 ||
            !ggml_is_contiguous(dst)) {
        return false;
    }
    if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[0] < K->ne[1] || mask->ne[2] != 1 || mask->ne[3] != 1)) {
        return false;
    }
    if (sinks && (sinks->type != GGML_TYPE_F32 || ggml_nelements(sinks) != Q->ne[2])) {
        return false;
    }
    // rows of 8 bytes per lane, the query as float4
    if (Q->nb[0] != sizeof(float) || K->nb[0] != sizeof(half) || V->nb[0] != sizeof(half) ||
            Q->nb[2] % 16 != 0 || !fap_aligned(Q->data, 16) ||
            K->nb[1] % 8 != 0 || K->nb[2] % 8 != 0 || !fap_aligned(K->data, 8) ||
            V->nb[1] % 8 != 0 || V->nb[2] % 8 != 0 || !fap_aligned(V->data, 8) || !fap_aligned(dst->data, 16)) {
        return false;
    }

    if (ctx.fattn_pascal.counters == nullptr) {
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaMalloc(&ctx.fattn_pascal.counters, FAP_MAX_COUNTERS*sizeof(unsigned)));
        CUDA_CHECK(cudaMemset(ctx.fattn_pascal.counters, 0, FAP_MAX_COUNTERS*sizeof(unsigned)));
    }

    // key splits: about 512 keys per block, 5 to 15 blocks per KV-head group (3 blocks per SM on a 5-SM card)
    const int n_kv   = (int) K->ne[1];
    const int splits = std::max(5, std::min(15, n_kv/512));
    const int kpb    = GGML_PAD((n_kv + splits - 1)/splits, FAP_TILE);
    const int nblk   = (n_kv + kpb - 1)/kpb;
    const int n_head_kv = (int) K->ne[2];

    ggml_cuda_pool_alloc<float> part(ctx.pool(), (size_t) nblk*n_head_kv*FAP_G*FAP_PART);

    fap_args a;
    a.Q              = (const char *) Q->data;
    a.q_nb2          = Q->nb[2];
    a.K              = (const char *) K->data;
    a.k_nb1          = K->nb[1];
    a.k_nb2          = K->nb[2];
    a.V              = (const char *) V->data;
    a.v_nb1          = V->nb[1];
    a.v_nb2          = V->nb[2];
    a.mask           = mask ? (const half *) mask->data : nullptr;
    a.sinks          = sinks ? (const float *) sinks->data : nullptr;
    a.dst            = (float *) dst->data;
    a.part           = part.get();
    a.counter        = ctx.fattn_pascal.counters;
    a.n_kv           = n_kv;
    a.keys_per_block = kpb;
    a.scale_log2     = scale*FAP_LOG2_E;
    a.fault          = fap_fault();

    const dim3 grid(nblk, n_head_kv/FAP_NW, 1);
    const dim3 block(WARP_SIZE, FAP_NW, 1);
    flash_attn_ext_pascal_f16<<<grid, block, 0, ctx.stream()>>>(a);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#else

bool ggml_cuda_flash_attn_ext_pascal(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED_VARS(ctx, dst);
    return false;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
