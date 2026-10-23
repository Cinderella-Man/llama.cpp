#include "mmvq-pascal.cuh"
#include "unary.cuh"

#include <climits>
#include <cstdlib>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// How the kernels split the work (one token, ncols_dst == 1):
//   - A warp is cut into lane groups of lpr lanes, lpr a power of two chosen from the row width. A lane group owns R
//     consecutive rows (twice that with a fused gate), which share every load of the activations.
//   - Each lane takes C consecutive weight blocks per iteration. They are loaded either directly with the widest loads
//     their alignment allows, or (staged) by the whole lane group with coalesced 128-bit loads into shared memory, from
//     where each lane reads its own blocks. Staging matters for 2-byte-aligned blocks (Q8_0, Q4_0), whose direct loads
//     are 32 bits wide and strided.
//   - 4-bit codes are unpacked with masks only: dp4a on (q & 0x0F0F0F0F) and an unsigned-by-signed dp4a on
//     (q & 0xF0F0F0F0), combined as (lo << 4) + hi, with the factor 1/16 moved into the activation scale.
//   - Row pointers are computed once and advanced by a constant; partial sums are reduced with warp shuffles only.
// The expert path (ids), the routed + shared expert launch (fusion.shared_up), biases and the GLU epilogue follow
// mul_mat_vec_q exactly.

#define MMVQ_PASCAL_NWARPS 4

namespace {

template <ggml_type type> struct mmvqp_traits;
template <> struct mmvqp_traits<GGML_TYPE_Q4_0> { static constexpr int bs = sizeof(block_q4_0); };
template <> struct mmvqp_traits<GGML_TYPE_Q4_1> { static constexpr int bs = sizeof(block_q4_1); };
template <> struct mmvqp_traits<GGML_TYPE_Q8_0> { static constexpr int bs = sizeof(block_q8_0); };

constexpr __host__ __device__ int mmvqp_gcd(const int a, const int b) {
    return b == 0 ? a : mmvqp_gcd(b, a % b);
}

// a: 4 unsigned bytes, b: 4 signed bytes
static __device__ __forceinline__ int mmvqp_dp4a_us(const uint32_t a, const int b, const int c) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A
    int d;
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
#else
    const int8_t * b8 = (const int8_t *) &b;
    return c + (int) (a & 0xff) * b8[0] + (int) ((a >> 8) & 0xff) * b8[1] + (int) ((a >> 16) & 0xff) * b8[2] + (int) (a >> 24) * b8[3];
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A
}

static __device__ __forceinline__ float mmvqp_h2f_lo(const uint32_t w) {
    return __half2float(__ushort_as_half((unsigned short) (w & 0xffff)));
}

static __device__ __forceinline__ float mmvqp_h2f_hi(const uint32_t w) {
    return __half2float(__ushort_as_half((unsigned short) (w >> 16)));
}

// NB bytes (a multiple of 4) from global memory aligned to AL bytes, through the read-only path.
template <int NB, int AL>
static __device__ __forceinline__ void mmvqp_ldg_words(uint32_t * w, const char * p) {
    static_assert(NB % 4 == 0 && AL >= 4, "chunk must be 4-byte aligned");
    if constexpr (AL >= 16 && NB % 16 == 0) {
#pragma unroll
        for (int i = 0; i < NB/16; ++i) {
            const uint4 v = __ldg((const uint4 *) p + i);
            w[4*i + 0] = v.x; w[4*i + 1] = v.y; w[4*i + 2] = v.z; w[4*i + 3] = v.w;
        }
    } else if constexpr (AL >= 8 && NB % 8 == 0) {
#pragma unroll
        for (int i = 0; i < NB/8; ++i) {
            const uint2 v = __ldg((const uint2 *) p + i);
            w[2*i + 0] = v.x; w[2*i + 1] = v.y;
        }
    } else {
#pragma unroll
        for (int i = 0; i < NB/4; ++i) {
            w[i] = __ldg((const unsigned int *) p + i);
        }
    }
}

// NB bytes (a multiple of 4) from shared memory aligned to AL bytes.
template <int NB, int AL>
static __device__ __forceinline__ void mmvqp_lds_words(uint32_t * w, const char * p) {
    if constexpr (AL >= 16 && NB % 16 == 0) {
#pragma unroll
        for (int i = 0; i < NB/16; ++i) {
            const uint4 v = ((const uint4 *) p)[i];
            w[4*i + 0] = v.x; w[4*i + 1] = v.y; w[4*i + 2] = v.z; w[4*i + 3] = v.w;
        }
    } else if constexpr (AL >= 8 && NB % 8 == 0) {
#pragma unroll
        for (int i = 0; i < NB/8; ++i) {
            const uint2 v = ((const uint2 *) p)[i];
            w[2*i + 0] = v.x; w[2*i + 1] = v.y;
        }
    } else {
#pragma unroll
        for (int i = 0; i < NB/4; ++i) {
            w[i] = ((const uint32_t *) p)[i];
        }
    }
}

// 32-bit word / 16-bit value at an even byte offset of a word array (offsets are compile-time constants after unrolling).
static __device__ __forceinline__ uint32_t mmvqp_word_at(const uint32_t * w, const int off) {
    return (off & 3) == 0 ? w[off >> 2] : __byte_perm(w[off >> 2], w[(off >> 2) + 1], 0x5432);
}

static __device__ __forceinline__ uint32_t mmvqp_u16_at(const uint32_t * w, const int off) {
    return (off & 3) == 0 ? (w[off >> 2] & 0xffff) : (w[off >> 2] >> 16);
}

// NA Q8_1 activation blocks, split into quants and float scales.
template <int NA>
struct mmvqp_act {
    int   u[NA][8];
    float d[NA];  // d8 (Q8_0) or d8 / 16 (4-bit types)
    float s[NA];  // s8 (Q4_1) or -8 * s8 (Q4_0)
};

template <ggml_type type, int NA>
static __device__ __forceinline__ void mmvqp_load_act(mmvqp_act<NA> & a, const char * p) {
    constexpr int NB = NA * sizeof(block_q8_1);
    uint32_t w[NB / 4];
    mmvqp_ldg_words<NB, mmvqp_gcd(NB, 16)>(w, p);
#pragma unroll
    for (int b = 0; b < NA; ++b) {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            a.u[b][k] = (int) w[9*b + 1 + k];
        }
        const float d8 = mmvqp_h2f_lo(w[9*b]);
        const float s8 = mmvqp_h2f_hi(w[9*b]);
        if constexpr (type == GGML_TYPE_Q8_0) {
            a.d[b] = d8;
            a.s[b] = 0.0f;
        } else if constexpr (type == GGML_TYPE_Q4_0) {
            a.d[b] = d8 * 0.0625f;
            a.s[b] = -8.0f * s8;
        } else {
            a.d[b] = d8 * 0.0625f;
            a.s[b] = s8;
        }
    }
}

// 16 * sum(code * activation) over one 32-weight block of 4-bit codes q0..q3.
static __device__ __forceinline__ int mmvqp_q4_isum16(
        const uint32_t q0, const uint32_t q1, const uint32_t q2, const uint32_t q3, const int * u) {
    int lo = 0;
    int hi = 0;
    lo = ggml_cuda_dp4a((int) (q0 & 0x0F0F0F0F), u[0], lo);
    hi = mmvqp_dp4a_us(q0 & 0xF0F0F0F0, u[4], hi);
    lo = ggml_cuda_dp4a((int) (q1 & 0x0F0F0F0F), u[1], lo);
    hi = mmvqp_dp4a_us(q1 & 0xF0F0F0F0, u[5], hi);
    lo = ggml_cuda_dp4a((int) (q2 & 0x0F0F0F0F), u[2], lo);
    hi = mmvqp_dp4a_us(q2 & 0xF0F0F0F0, u[6], hi);
    lo = ggml_cuda_dp4a((int) (q3 & 0x0F0F0F0F), u[3], lo);
    hi = mmvqp_dp4a_us(q3 & 0xF0F0F0F0, u[7], hi);
    return (lo << 4) + hi;
}

// C weight blocks (raw words) times the matching activation blocks, added to acc.
template <ggml_type type, int C>
static __device__ __forceinline__ float mmvqp_dot(const uint32_t * w, const mmvqp_act<C> & a, float acc) {
    constexpr int BS = mmvqp_traits<type>::bs;
#pragma unroll
    for (int b = 0; b < C; ++b) {
        const int o = b * BS;
        if constexpr (type == GGML_TYPE_Q4_1) {
            const uint32_t dm = mmvqp_word_at(w, o);
            const int isum = mmvqp_q4_isum16(mmvqp_word_at(w, o + 4), mmvqp_word_at(w, o + 8), mmvqp_word_at(w, o + 12),
                                             mmvqp_word_at(w, o + 16), a.u[b]);
            acc = fmaf((float) isum, mmvqp_h2f_lo(dm) * a.d[b], acc);
            acc = fmaf(mmvqp_h2f_hi(dm), a.s[b], acc);
        } else if constexpr (type == GGML_TYPE_Q4_0) {
            const float d = mmvqp_h2f_lo(mmvqp_u16_at(w, o));
            const int isum = mmvqp_q4_isum16(mmvqp_word_at(w, o + 2), mmvqp_word_at(w, o + 6), mmvqp_word_at(w, o + 10),
                                             mmvqp_word_at(w, o + 14), a.u[b]);
            acc = fmaf(d, fmaf((float) isum, a.d[b], a.s[b]), acc);
        } else { // GGML_TYPE_Q8_0
            const float d = mmvqp_h2f_lo(mmvqp_u16_at(w, o));
            int isum = 0;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                isum = ggml_cuda_dp4a((int) mmvqp_word_at(w, o + 2 + 4*k), a.u[b][k], isum);
            }
            acc = fmaf((float) isum, d * a.d[b], acc);
        }
    }
    return acc;
}

// Byte strides for x and the activations, float strides for dst.
struct mmvqp_args {
    const char    * vx;
    const char    * vy;
    const int32_t * ids;
    float         * dst;
    int             nrows;
    int             row_bytes;
    int             lpr_log2;  // lanes per row = 1 << lpr_log2
    int             niter;     // iterations per lane
    uint3           nchannels_y;
    uint3           channel_ratio;
    uint3           sample_ratio;
    int64_t         channel_bytes_x;
    int64_t         sample_bytes_x;
    int64_t         channel_bytes_y;
    int64_t         sample_bytes_y;
    int64_t         stride_channel_dst;
    int64_t         stride_sample_dst;
};

enum mmvqp_fusion { MMVQP_PLAIN = 0, MMVQP_BIAS = 1, MMVQP_GATE = 2 };

template <ggml_type type, int R, int C, bool staged, bool gate>
constexpr bool mmvqp_fits() {
    return !staged || MMVQ_PASCAL_NWARPS * WARP_SIZE * R * (gate ? 2 : 1) * C * mmvqp_traits<type>::bs <= 48 * 1024;
}

template <ggml_type type, int R, int C, bool staged, int fk>
__launch_bounds__(MMVQ_PASCAL_NWARPS*WARP_SIZE, 1)
static __global__ void mul_mat_vec_q_pascal(const mmvqp_args p, const ggml_cuda_mm_fusion_args_device fusion) {
    constexpr int NW = MMVQ_PASCAL_NWARPS;
    constexpr int BS = mmvqp_traits<type>::bs;
    constexpr int WB = C * BS;                     // weight bytes per lane and iteration
    constexpr int YB = C * sizeof(block_q8_1);     // activation bytes per lane and iteration
    constexpr int M  = fk == MMVQP_GATE ? 2 : 1;   // matrices: up (or plain), gate
    constexpr int NU = (WB + 15) / 16;             // staged: 16-byte units per lane and row, the last one predicated
    __shared__ uint4 sm[staged ? NW : 1][staged ? WARP_SIZE * R * M * WB / 16 : 1];

    const int lane     = threadIdx.x;
    const int lpr_log2 = p.lpr_log2;
    const int lpr      = 1 << lpr_log2;
    const int sl       = lane & (lpr - 1);
    const int grp      = lane >> lpr_log2;
    const int row0     = (((blockIdx.x * NW + threadIdx.y) << (5 - lpr_log2)) + grp) * R;

    const bool     shared_expert = fk == MMVQP_GATE && fusion.shared_up && blockIdx.y == gridDim.y - 1;
    const uint32_t channel_dst   = shared_expert ? 0 : blockIdx.y;
    const uint32_t sample_dst    = blockIdx.z;

    ggml_cuda_pdl_sync();
    const uint32_t channel_x = shared_expert ? 0 : p.ids ? p.ids[channel_dst] : fastdiv(channel_dst, p.channel_ratio);
    const uint32_t channel_y = p.ids ? fastmodulo(channel_dst, p.nchannels_y) : channel_dst;
    const uint32_t sample_x  = fastdiv(sample_dst, p.sample_ratio);

    const int64_t xoff = sample_x*p.sample_bytes_x + channel_x*p.channel_bytes_x;
    const char * xm[M];
    xm[0] = (shared_expert ? (const char *) fusion.shared_up : p.vx) + xoff;
    if constexpr (M == 2) {
        xm[1] = (const char *) (shared_expert ? fusion.shared_gate : fusion.gate) + xoff;
    }

    const char * rw[M][R];
#pragma unroll
    for (int m = 0; m < M; ++m) {
#pragma unroll
        for (int i = 0; i < R; ++i) {
            rw[m][i] = xm[m] + (int64_t) min(row0 + i, p.nrows - 1) * p.row_bytes + (staged ? 0 : sl * WB);
        }
    }
    const char * yp = p.vy + sample_dst*p.sample_bytes_y + channel_y*p.channel_bytes_y + sl * YB;
    const int wstep = WB << lpr_log2;
    const int ystep = YB << lpr_log2;
    [[maybe_unused]] char * smw = (char *) sm[staged ? threadIdx.y : 0] + (staged ? grp * R * M * WB << lpr_log2 : 0);
    [[maybe_unused]] const int seg_units = (WB << lpr_log2) >> 4;

    float acc[M][R];
#pragma unroll
    for (int m = 0; m < M; ++m) {
#pragma unroll
        for (int i = 0; i < R; ++i) {
            acc[m][i] = 0.0f;
        }
    }

    for (int it = 0; it < p.niter; ++it) {
        if constexpr (staged) {
#pragma unroll
            for (int m = 0; m < M; ++m) {
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    uint4       * dsm = (uint4 *) (smw + ((m*R + i) * WB << lpr_log2));
                    const uint4 * src = (const uint4 *) rw[m][i];
#pragma unroll
                    for (int j = 0; j < NU; ++j) {
                        const int u = sl + (j << lpr_log2);
                        if (j < NU - 1 || u < seg_units) {
                            dsm[u] = __ldg(src + u);
                        }
                    }
                }
            }
            __syncwarp();
        }
        mmvqp_act<C> a;
        mmvqp_load_act<type, C>(a, yp);
#pragma unroll
        for (int m = 0; m < M; ++m) {
#pragma unroll
            for (int i = 0; i < R; ++i) {
                uint32_t w[WB / 4];
                if constexpr (staged) {
                    mmvqp_lds_words<WB, mmvqp_gcd(WB, 16)>(w, smw + ((m*R + i) * WB << lpr_log2) + sl * WB);
                } else {
                    mmvqp_ldg_words<WB, mmvqp_gcd(WB, 16)>(w, rw[m][i]);
                }
                acc[m][i] = mmvqp_dot<type, C>(w, a, acc[m][i]);
                rw[m][i] += wstep;
            }
        }
        if constexpr (staged) {
            __syncwarp();
        }
        yp += ystep;
    }

#pragma unroll
    for (int i = 0; i < R; ++i) {
#pragma unroll
        for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
            if (off < lpr) {
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    acc[m][i] += __shfl_xor_sync(0xFFFFFFFF, acc[m][i], off, WARP_SIZE);
                }
            }
        }
    }

    float * dst = (shared_expert ? fusion.shared_dst : p.dst) + sample_dst*p.stride_sample_dst + channel_dst*p.stride_channel_dst;
    [[maybe_unused]] const int64_t bias_off = sample_dst*p.stride_sample_dst + (p.ids ? channel_x : channel_dst)*p.stride_channel_dst;
#pragma unroll
    for (int i = 0; i < R; ++i) {
        const int row = row0 + i;
        if ((i & (lpr - 1)) != sl || row >= p.nrows) {
            continue;
        }
        float result = acc[0][i];
        if constexpr (fk != MMVQP_PLAIN) {
            if (fusion.x_bias) {
                result += ((const float *) fusion.x_bias)[bias_off + row];
            }
        }
        if constexpr (fk == MMVQP_GATE) {
            float gate_value = acc[M - 1][i];
            if (fusion.gate_bias) {
                gate_value += ((const float *) fusion.gate_bias)[bias_off + row];
            }
            switch (fusion.glu_op) {
                case GGML_GLU_OP_SWIGLU:
                    result *= ggml_cuda_op_silu_single(gate_value);
                    break;
                case GGML_GLU_OP_GEGLU:
                    result *= ggml_cuda_op_gelu_single(gate_value);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                    break;
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, fusion.glu_limit);
                    break;
                default:
                    result = result * gate_value;
                    break;
            }
        }
        dst[row] = result;
    }
}

// The launch shape per type: rows per lane group (R), blocks per lane (C), staged or direct loads.
struct mmvqp_shape {
    int  R;
    int  C;
    bool staged;
};

// Measured on a P106-090 at the shapes of Flint (K = 512 to 4096) with bench/mmvq_pascal of
// github.com/Cinderella-Man/llms-on-crypto-mining-rigs: staged loads win for every type there; the direct-load shape is
// the fallback where a row segment can't be staged.
static mmvqp_shape mmvqp_choose(const ggml_type type, const bool gate) {
    switch (type) {
        case GGML_TYPE_Q4_1:
            return gate ? mmvqp_shape{ 2, 2, true } : mmvqp_shape{ 2, 4, true };
        case GGML_TYPE_Q4_0:
            return { 2, 2, true };
        case GGML_TYPE_Q8_0:
            return gate ? mmvqp_shape{ 1, 2, true } : mmvqp_shape{ 2, 2, true };
        default:
            GGML_ABORT("fatal error");
    }
}

template <ggml_type type, int R, int C, bool staged>
static void mmvqp_launch(const mmvqp_args & p, const ggml_cuda_mm_fusion_args_device & fusion, const int fk,
                         const dim3 grid, cudaStream_t stream) {
    const dim3 block(WARP_SIZE, MMVQ_PASCAL_NWARPS, 1);
    switch (fk) {
        case MMVQP_PLAIN:
            if constexpr (mmvqp_fits<type, R, C, staged, false>()) {
                mul_mat_vec_q_pascal<type, R, C, staged, MMVQP_PLAIN><<<grid, block, 0, stream>>>(p, fusion);
            } else {
                GGML_ABORT("fatal error");
            }
            break;
        case MMVQP_BIAS:
            if constexpr (mmvqp_fits<type, R, C, staged, false>()) {
                mul_mat_vec_q_pascal<type, R, C, staged, MMVQP_BIAS><<<grid, block, 0, stream>>>(p, fusion);
            } else {
                GGML_ABORT("fatal error");
            }
            break;
        default:
            if constexpr (mmvqp_fits<type, R, C, staged, true>()) {
                mul_mat_vec_q_pascal<type, R, C, staged, MMVQP_GATE><<<grid, block, 0, stream>>>(p, fusion);
            } else {
                GGML_ABORT("fatal error");
            }
            break;
    }
}

template <ggml_type type>
static void mmvqp_launch_shape(const mmvqp_shape s, const mmvqp_args & p, const ggml_cuda_mm_fusion_args_device & fusion,
                               const int fk, const dim3 grid, cudaStream_t stream) {
#define MMVQP_CASE(R_, C_, S_)                                              \
    if (s.R == R_ && s.C == C_ && s.staged == S_) {                         \
        mmvqp_launch<type, R_, C_, S_>(p, fusion, fk, grid, stream);        \
        return;                                                             \
    }
    if constexpr (type == GGML_TYPE_Q4_1) {
        MMVQP_CASE(2, 4, true)
        MMVQP_CASE(2, 4, false)
        MMVQP_CASE(2, 2, true)
        MMVQP_CASE(2, 2, false)
    } else if constexpr (type == GGML_TYPE_Q4_0) {
        MMVQP_CASE(2, 2, true)
        MMVQP_CASE(2, 2, false)
    } else {
        MMVQP_CASE(1, 2, true)
        MMVQP_CASE(1, 2, false)
        MMVQP_CASE(2, 2, true)
        MMVQP_CASE(2, 2, false)
    }
#undef MMVQP_CASE
    GGML_ABORT("fatal error");
}

static int mmvqp_mode() {
    static const int mode = [] {
        const char * env = getenv("GGML_CUDA_MMVQ_PASCAL");
        return env ? atoi(env) : 1;
    }();
    return mode;
}

static bool mmvqp_aligned(const void * ptr, const int64_t al) {
    return ptr == nullptr || (uintptr_t) ptr % al == 0;
}

} // namespace

bool ggml_cuda_mmvq_pascal(
        const void * vx, ggml_type type, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device & fusion,
        float * dst, int64_t ncols_x, int64_t nrows_x, int64_t ncols_dst, int64_t stride_row_x,
        int64_t nchannels_x, int64_t nchannels_y, int64_t nchannels_dst, int64_t stride_channel_x, int64_t stride_channel_y,
        int64_t stride_channel_dst, int64_t nsamples_x, int64_t nsamples_dst, int64_t stride_sample_x, int64_t stride_sample_y,
        int64_t stride_sample_dst, cudaStream_t stream) {
    const int mode = mmvqp_mode();
    if (mode == 0 || ncols_dst != 1) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || (mode == 1 && cc != GGML_CUDA_CC_DP4A)) {
        return false;
    }
    if (type != GGML_TYPE_Q4_0 && type != GGML_TYPE_Q4_1 && type != GGML_TYPE_Q8_0) {
        return false;
    }
    if (fusion.x_scale || fusion.gate_scale) {
        return false;
    }

    const bool gate = fusion.gate != nullptr;
    mmvqp_shape s = mmvqp_choose(type, gate);

    const int64_t bs     = ggml_type_size(type);
    const int64_t blocks = ncols_x / ggml_blck_size(type);
    if (blocks % s.C != 0) {
        return false;
    }
    const int64_t chunks = blocks / s.C;
    int lpr_log2 = 0;
    while (lpr_log2 < 5 && chunks % (2 << lpr_log2) == 0) {
        lpr_log2++;
    }

    // every chunk load must be aligned as the kernel assumes: staged segments to 16 bytes, direct chunks to their size
    const int64_t row_bytes = stride_row_x * bs;
    const auto x_aligned = [&](const int64_t al) {
        return mmvqp_aligned(vx, al) && mmvqp_aligned(fusion.gate, al) && mmvqp_aligned(fusion.shared_up, al) &&
            mmvqp_aligned(fusion.shared_gate, al) && row_bytes % al == 0 && (stride_channel_x * bs) % al == 0 &&
            (stride_sample_x * bs) % al == 0;
    };
    if (s.staged && (((s.C * bs) << lpr_log2) % 16 != 0 || !x_aligned(16))) {
        s.staged = false;
    }
    if (row_bytes > INT_MAX || (!s.staged && !x_aligned(mmvqp_gcd((int) (s.C * bs), 16)))) {
        return false;
    }
    const int64_t aly = mmvqp_gcd((int) (s.C * sizeof(block_q8_1)), 16);
    if (!mmvqp_aligned(vy, aly) || (stride_channel_y * (int64_t) sizeof(block_q8_1)) % aly != 0 ||
            (stride_sample_y * (int64_t) sizeof(block_q8_1)) % aly != 0) {
        return false;
    }

    mmvqp_args p;
    p.vx                 = (const char *) vx;
    p.vy                 = (const char *) vy;
    p.ids                = ids;
    p.dst                = dst;
    p.nrows              = (int) nrows_x;
    p.row_bytes          = (int) row_bytes;
    p.lpr_log2           = lpr_log2;
    p.niter              = (int) (chunks >> lpr_log2);
    p.nchannels_y        = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    p.channel_ratio      = ids ? make_uint3(0, 0, 0) : init_fastdiv_values(nchannels_dst / nchannels_x);
    p.sample_ratio       = init_fastdiv_values(nsamples_dst / nsamples_x);
    p.channel_bytes_x    = stride_channel_x * bs;
    p.sample_bytes_x     = stride_sample_x * bs;
    p.channel_bytes_y    = stride_channel_y * (int64_t) sizeof(block_q8_1);
    p.sample_bytes_y     = stride_sample_y * (int64_t) sizeof(block_q8_1);
    p.stride_channel_dst = stride_channel_dst;
    p.stride_sample_dst  = stride_sample_dst;

    const int  fk             = gate ? MMVQP_GATE : (fusion.x_bias ? MMVQP_BIAS : MMVQP_PLAIN);
    const int  rows_per_block = MMVQ_PASCAL_NWARPS * (WARP_SIZE >> lpr_log2) * s.R;
    const dim3 grid((unsigned) ((nrows_x + rows_per_block - 1) / rows_per_block),
                    (unsigned) (nchannels_dst + (fusion.shared_up ? 1 : 0)), (unsigned) nsamples_dst);

    switch (type) {
        case GGML_TYPE_Q4_0: mmvqp_launch_shape<GGML_TYPE_Q4_0>(s, p, fusion, fk, grid, stream); break;
        case GGML_TYPE_Q4_1: mmvqp_launch_shape<GGML_TYPE_Q4_1>(s, p, fusion, fk, grid, stream); break;
        case GGML_TYPE_Q8_0: mmvqp_launch_shape<GGML_TYPE_Q8_0>(s, p, fusion, fk, grid, stream); break;
        default:             GGML_ABORT("fatal error");
    }
    return true;
}

#else

bool ggml_cuda_mmvq_pascal(
        const void * vx, ggml_type type, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device & fusion,
        float * dst, int64_t ncols_x, int64_t nrows_x, int64_t ncols_dst, int64_t stride_row_x,
        int64_t nchannels_x, int64_t nchannels_y, int64_t nchannels_dst, int64_t stride_channel_x, int64_t stride_channel_y,
        int64_t stride_channel_dst, int64_t nsamples_x, int64_t nsamples_dst, int64_t stride_sample_x, int64_t stride_sample_y,
        int64_t stride_sample_dst, cudaStream_t stream) {
    GGML_UNUSED_VARS(vx, type, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, nchannels_x, nchannels_y,
                     nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst,
                     stride_sample_x, stride_sample_y, stride_sample_dst, stream);
    return false;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
