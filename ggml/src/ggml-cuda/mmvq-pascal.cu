#include "mmvq-pascal.cuh"
#include "unary.cuh"

#include <algorithm>
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
//   - The activation block sums that the offsets and mins need (Q4_0, Q4_1, Q5_0, Q5_K) are computed in FP32 from the
//     int8 values once per activation load and shared by all rows; block_q8_1's FP16 sum is never read, because it
//     overflows on large activations.
//   - Q5_K and IQ4_XS take one 256-weight super-block per lane. IQ4_XS looks its 4-bit indices up with __byte_perm in a
//     table held in registers, with the activations pre-permuted into the order the lookup produces.
//   - Q5_0's fifth bits stay where they are: a block's sum splits into sum(nibble * a) + 16 * sum(bit * a), and
//     (qh >> j) & 0x01010101 holds the fifth bits of weights j, 8 + j, 16 + j and 24 + j, so the activations are
//     regrouped the same way once per block (two 4 x 4 byte transposes), shared by all rows.
//   - Row pointers are computed once and advanced by a constant; partial sums are reduced with warp shuffles only.
// The expert path (ids), the routed + shared expert launch (fusion.shared_up), biases and the GLU epilogue follow
// mul_mat_vec_q exactly.
//
// 2 to MMVQP_MAX_COLS tokens (ncols_dst > 1) run mul_mat_vec_q_pascal_mc, which keeps the lane groups, launch shapes
// and staging of the single-token kernel and adds a loop over the tokens inside each iteration: a lane group stages its
// weight chunks once, then loads one token's activations at a time and multiplies them by every row's chunk, so weights
// are read once per step and activations once per token, shared by all rows of the lane group.
//   - MUL_MAT: every block serves all tokens of its channel.
//   - MUL_MAT_ID: one block per (token, slot) entry of ids. The first entry of each distinct expert serves the later
//     entries that picked the same expert (up to MMVQP_MAX_COLS of them; a further entry starts a new group), and the
//     other blocks exit at once, so every distinct expert's rows are read once per step instead of once per token.
//   - The routed + shared expert launch adds one block that serves all tokens with the shared expert.
//   - GGML_CUDA_MMVQ_PASCAL_STAGE_ACT=1 also stages each token's activations through shared memory (where they fit
//     next to the weights: all types but Q5_K and the staged IQ4_XS shape), so that they are read with coalesced loads
//     instead of one strided load per lane and token. Off by default until it is measured on a P106.

#define MMVQ_PASCAL_NWARPS 4
#define MMVQP_MAX_COLS 4

namespace {

// Bytes and Q8_1 activation blocks per weight block.
template <ggml_type type> struct mmvqp_traits;
template <> struct mmvqp_traits<GGML_TYPE_Q4_0>   { static constexpr int bs = sizeof(block_q4_0);   static constexpr int na = 1; };
template <> struct mmvqp_traits<GGML_TYPE_Q4_1>   { static constexpr int bs = sizeof(block_q4_1);   static constexpr int na = 1; };
template <> struct mmvqp_traits<GGML_TYPE_Q5_0>   { static constexpr int bs = sizeof(block_q5_0);   static constexpr int na = 1; };
template <> struct mmvqp_traits<GGML_TYPE_Q8_0>   { static constexpr int bs = sizeof(block_q8_0);   static constexpr int na = 1; };
template <> struct mmvqp_traits<GGML_TYPE_Q5_K>   { static constexpr int bs = sizeof(block_q5_K);   static constexpr int na = QK_K / QK8_1; };
template <> struct mmvqp_traits<GGML_TYPE_IQ4_XS> { static constexpr int bs = sizeof(block_iq4_xs); static constexpr int na = QK_K / QK8_1; };

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
    int   t[NA][8];  // Q5_0 only: t[j] = bytes a[j], a[8 + j], a[16 + j], a[24 + j] of the block
    float d[NA];     // d8 (Q8_0, Q5_K, IQ4_XS) or d8 / 16 (4- and 5-bit legacy types)
    float s[NA];     // with S = d8 * the sum of the int8 values: S (Q4_1, Q5_K), -8 * S (Q4_0), -16 * S (Q5_0)
};

// 4 x 4 byte transpose: out[j] = bytes j of x0, x1, x2, x3.
static __device__ __forceinline__ void mmvqp_transpose4(
        const uint32_t x0, const uint32_t x1, const uint32_t x2, const uint32_t x3, int * out) {
    const uint32_t a = __byte_perm(x0, x1, 0x5140);
    const uint32_t b = __byte_perm(x0, x1, 0x7362);
    const uint32_t c = __byte_perm(x2, x3, 0x5140);
    const uint32_t d = __byte_perm(x2, x3, 0x7362);
    out[0] = (int) __byte_perm(a, c, 0x5410);
    out[1] = (int) __byte_perm(a, c, 0x7632);
    out[2] = (int) __byte_perm(b, d, 0x5410);
    out[3] = (int) __byte_perm(b, d, 0x7632);
}

// From global memory, or (from_smem) from activations staged in shared memory.
template <ggml_type type, int NA, bool from_smem = false>
static __device__ __forceinline__ void mmvqp_load_act(mmvqp_act<NA> & a, const char * p) {
    constexpr int NB = NA * sizeof(block_q8_1);
    uint32_t w[NB / 4];
    if constexpr (from_smem) {
        mmvqp_lds_words<NB, mmvqp_gcd(NB, 16)>(w, p);
    } else {
        mmvqp_ldg_words<NB, mmvqp_gcd(NB, 16)>(w, p);
    }
#pragma unroll
    for (int b = 0; b < NA; ++b) {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            a.u[b][k] = (int) w[9*b + 1 + k];
        }
        const float d8 = mmvqp_h2f_lo(w[9*b]);
        // The offsets (Q4_0, Q5_0) and mins (Q4_1, Q5_K) need the block's sum. It is taken from the int8 values, in
        // FP32, not from the FP16 sum that block_q8_1 stores next to d8: that one becomes infinite once the block's
        // activations add up past 65504 (Flint's expert down-projections in layers 46-48), while d8 * sum(int8) is
        // finite whenever d8 is. 8 dp4a per activation block, shared by all rows of the lane group.
        float S = 0.0f;
        if constexpr (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_K) {
            int sum = 0;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                sum = ggml_cuda_dp4a(0x01010101, a.u[b][k], sum);
            }
            S = d8 * (float) sum;
        }
        if constexpr (type == GGML_TYPE_Q8_0) {
            a.d[b] = d8;
            a.s[b] = 0.0f;
        } else if constexpr (type == GGML_TYPE_Q4_0) {
            a.d[b] = d8 * 0.0625f;
            a.s[b] = -8.0f * S;
        } else if constexpr (type == GGML_TYPE_Q5_0) {
            a.d[b] = d8 * 0.0625f;
            a.s[b] = -16.0f * S;
            mmvqp_transpose4(a.u[b][0], a.u[b][2], a.u[b][4], a.u[b][6], a.t[b] + 0);
            mmvqp_transpose4(a.u[b][1], a.u[b][3], a.u[b][5], a.u[b][7], a.t[b] + 4);
        } else if constexpr (type == GGML_TYPE_Q5_K) {
            a.d[b] = d8;
            a.s[b] = S;
        } else if constexpr (type == GGML_TYPE_IQ4_XS) {
            // word k of the codes holds weights 4k..4k+3 (low nibbles) and 4k+16..4k+19 (high nibbles); the table
            // lookup returns them as (4k, 4k+16, 4k+1, 4k+17) and (4k+2, 4k+18, 4k+3, 4k+19)
            int p[8];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                p[2*k + 0] = (int) __byte_perm(a.u[b][k], a.u[b][k + 4], 0x5140);
                p[2*k + 1] = (int) __byte_perm(a.u[b][k], a.u[b][k + 4], 0x7362);
            }
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                a.u[b][k] = p[k];
            }
            a.d[b] = d8;
            a.s[b] = 0.0f;
        } else {
            a.d[b] = d8 * 0.0625f;
            a.s[b] = S;
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

// One Q5_K super-block as 44 words: dm, 3 words of scales, 8 words qh, 32 words qs; activation blocks a0..a0+7.
template <int NA>
static __device__ __forceinline__ float mmvqp_dot_q5_K(const uint32_t * w, const mmvqp_act<NA> & a, const int a0, float acc) {
    const uint32_t sc_lo = w[1] & 0x3f3f3f3f;                                   // scales of sub-blocks 0..3
    const uint32_t m_lo  = w[2] & 0x3f3f3f3f;                                   // mins of sub-blocks 0..3
    const uint32_t sc_hi = (w[3] & 0x0f0f0f0f) | ((w[1] >> 2) & 0x30303030);   // scales of 4..7
    const uint32_t m_hi  = ((w[3] >> 4) & 0x0f0f0f0f) | ((w[2] >> 2) & 0x30303030);
    float sumd = 0.0f;
    float summ = 0.0f;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        int isum = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const uint32_t q   = w[12 + 8*(s/2) + k];
            const uint32_t nib = (s & 1) ? (q >> 4) & 0x0F0F0F0F : q & 0x0F0F0F0F;
            const uint32_t hb  = s <= 4 ? (w[4 + k] << (4 - s)) & 0x10101010 : (w[4 + k] >> (s - 4)) & 0x10101010;
            isum = ggml_cuda_dp4a((int) (nib | hb), a.u[a0 + s][k], isum);
        }
        const int sc = (int) (((s < 4 ? sc_lo : sc_hi) >> (8 * (s & 3))) & 0xff);
        const int m  = (int) (((s < 4 ? m_lo  : m_hi ) >> (8 * (s & 3))) & 0xff);
        sumd = fmaf(a.d[a0 + s], (float) (isum * sc), sumd);
        summ = fmaf((float) m, a.s[a0 + s], summ);
    }
    return acc + mmvqp_h2f_lo(w[0]) * sumd - mmvqp_h2f_hi(w[0]) * summ;
}

// One IQ4_XS super-block as 34 words: d | scales_h << 16, scales_l, 32 words qs; activation blocks a0..a0+7.
template <int NA>
static __device__ __forceinline__ float mmvqp_dot_iq4_xs(const uint32_t * w, const mmvqp_act<NA> & a, const int a0, float acc) {
    // kvalues_iq4nl as four words
    constexpr uint32_t t0 = 0xBFAD9881u, t1 = 0xF6EADDCFu, t2 = 0x26190D01u, t3 = 0x71594535u;
    const uint32_t sh = w[0] >> 16;
    const uint32_t sl = w[1];
    float sumf = 0.0f;
#pragma unroll
    for (int ib = 0; ib < 8; ++ib) {
        int isum = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t q4  = w[2 + 4*ib + j];
            const uint32_t sel = 0x32103210 | ((q4 & 0x88888888) >> 1);
            const uint32_t r0  = __byte_perm(__byte_perm(t0, t1, q4),       __byte_perm(t2, t3, q4),       sel);
            const uint32_t r1  = __byte_perm(__byte_perm(t0, t1, q4 >> 16), __byte_perm(t2, t3, q4 >> 16), sel >> 16);
            isum = ggml_cuda_dp4a((int) r0, a.u[a0 + ib][2*j + 0], isum);
            isum = ggml_cuda_dp4a((int) r1, a.u[a0 + ib][2*j + 1], isum);
        }
        const int ls = (int) (((sl >> (4 * ib)) & 0xF) | (((sh >> (2 * ib)) & 3) << 4));
        sumf = fmaf(a.d[a0 + ib], (float) (isum * (ls - 32)), sumf);
    }
    return fmaf(mmvqp_h2f_lo(w[0]), sumf, acc);
}

// C weight blocks (raw words) times the matching activation blocks, added to acc.
template <ggml_type type, int C>
static __device__ __forceinline__ float mmvqp_dot(const uint32_t * w, const mmvqp_act<C * mmvqp_traits<type>::na> & a, float acc) {
    constexpr int BS = mmvqp_traits<type>::bs;
#pragma unroll
    for (int b = 0; b < C; ++b) {
        const int o = b * BS;
        if constexpr (type == GGML_TYPE_Q5_K) {
            acc = mmvqp_dot_q5_K(w + o/4, a, 8*b, acc);
        } else if constexpr (type == GGML_TYPE_IQ4_XS) {
            acc = mmvqp_dot_iq4_xs(w + o/4, a, 8*b, acc);
        } else if constexpr (type == GGML_TYPE_Q4_1) {
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
        } else if constexpr (type == GGML_TYPE_Q5_0) {
            const float    d  = mmvqp_h2f_lo(mmvqp_u16_at(w, o));
            const uint32_t qh = mmvqp_word_at(w, o + 2);
            int hsum = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                hsum = ggml_cuda_dp4a((int) ((qh >> j) & 0x01010101), a.t[b][j], hsum);
            }
            const int isum = mmvqp_q4_isum16(mmvqp_word_at(w, o + 6), mmvqp_word_at(w, o + 10), mmvqp_word_at(w, o + 14),
                                             mmvqp_word_at(w, o + 18), a.u[b]) + (hsum << 8);
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

// Several tokens (mul_mat_vec_q_pascal_mc): token strides and the shape of ids.
struct mmvqp_args_mc : mmvqp_args {
    int64_t col_bytes_y;
    int64_t stride_col_dst;
    int     ncols;      // tokens
    int     nslots;     // ids: entries per token (experts used)
    int     ids_stride;
    bool    stage_act;  // stage each token's activations through shared memory where they fit
};

enum mmvqp_fusion { MMVQP_PLAIN = 0, MMVQP_BIAS = 1, MMVQP_GATE = 2 };

template <ggml_type type, int R, int C, bool staged, bool gate>
constexpr bool mmvqp_fits() {
    return !staged || MMVQ_PASCAL_NWARPS * WARP_SIZE * R * (gate ? 2 : 1) * C * mmvqp_traits<type>::bs <= 48 * 1024;
}

// the multi-token kernel with each token's activations staged as well (sa)
template <ggml_type type, int R, int C, bool staged, bool gate, bool sa>
constexpr bool mmvqp_fits_mc() {
    constexpr int wb = staged ? R * (gate ? 2 : 1) * C * mmvqp_traits<type>::bs : 0;
    constexpr int yb = sa ? C * mmvqp_traits<type>::na * (int) sizeof(block_q8_1) : 0;
    return MMVQ_PASCAL_NWARPS * WARP_SIZE * (wb + yb) <= 48 * 1024;
}

template <ggml_type type, int R, int C, bool staged, int fk>
__launch_bounds__(MMVQ_PASCAL_NWARPS*WARP_SIZE, 1)
static __global__ void mul_mat_vec_q_pascal(const mmvqp_args p, const ggml_cuda_mm_fusion_args_device fusion) {
    constexpr int NW = MMVQ_PASCAL_NWARPS;
    constexpr int BS = mmvqp_traits<type>::bs;
    constexpr int WB = C * BS;                     // weight bytes per lane and iteration
    constexpr int NA = C * mmvqp_traits<type>::na; // activation blocks per lane and iteration
    constexpr int YB = NA * sizeof(block_q8_1);    // activation bytes per lane and iteration
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
        mmvqp_act<NA> a;
        mmvqp_load_act<type, NA>(a, yp);
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

// 2 to MMVQP_MAX_COLS tokens. Grid: (row blocks, channels, samples) for MUL_MAT; (row blocks, tokens * slots + shared
// expert, samples) for MUL_MAT_ID, where block y = t * nslots + k is entry (token t, slot k) of ids.
// sa: each token's activation segment is copied into shared memory by the whole warp with coalesced 128-bit loads, and
// every lane reads its blocks from there, instead of loading its own blocks from global memory at a stride of YB bytes.
template <ggml_type type, int R, int C, bool staged, int fk, bool sa>
__launch_bounds__(MMVQ_PASCAL_NWARPS*WARP_SIZE, 1)
static __global__ void mul_mat_vec_q_pascal_mc(const mmvqp_args_mc p, const ggml_cuda_mm_fusion_args_device fusion) {
    constexpr int NW = MMVQ_PASCAL_NWARPS;
    constexpr int NC = MMVQP_MAX_COLS;
    constexpr int BS = mmvqp_traits<type>::bs;
    constexpr int WB = C * BS;                     // weight bytes per lane and iteration
    constexpr int NA = C * mmvqp_traits<type>::na; // activation blocks per lane and iteration
    constexpr int YB = NA * sizeof(block_q8_1);    // activation bytes per lane, iteration and token
    constexpr int M  = fk == MMVQP_GATE ? 2 : 1;   // matrices: up (or plain), gate
    constexpr int NU = (WB + 15) / 16;             // staged: 16-byte units per lane and row, the last one predicated
    constexpr int NY = (WARP_SIZE * YB / 16 + WARP_SIZE - 1) / WARP_SIZE; // sa: 16-byte units per lane and token
    __shared__ uint4 sm[staged ? NW : 1][staged ? WARP_SIZE * R * M * WB / 16 : 1];
    __shared__ uint4 sm_y[sa ? NW : 1][sa ? WARP_SIZE * YB / 16 : 1];

    const int lane     = threadIdx.x;
    const int lpr_log2 = p.lpr_log2;
    const int lpr      = 1 << lpr_log2;
    const int sl       = lane & (lpr - 1);
    const int grp      = lane >> lpr_log2;
    const int row0     = (((blockIdx.x * NW + threadIdx.y) << (5 - lpr_log2)) + grp) * R;

    const bool     shared_expert = fk == MMVQP_GATE && fusion.shared_up && blockIdx.y == gridDim.y - 1;
    const uint32_t sample_dst    = blockIdx.z;
    const uint32_t sample_x      = fastdiv(sample_dst, p.sample_ratio);

    // the tokens this block serves: offsets of their activations (bytes from vy) and outputs (floats from dst), which
    // the host checked to fit in an int
    int      yc[NC];
    int      dc[NC];
    int      nt = 0;
    uint32_t channel_x;
    uint32_t channel_bias;

    ggml_cuda_pdl_sync();
    if (p.ids && !shared_expert) {
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            yc[j] = 0;
            dc[j] = 0;
        }
        const int     t0 = blockIdx.y / p.nslots;
        const int     k0 = blockIdx.y - t0 * p.nslots;
        const int32_t e  = p.ids[k0 + t0 * p.ids_stride];
        // entries before this one with the same expert: only every NC-th of them starts a group
        int before = 0;
        for (int t = 0; t <= t0; ++t) {
            const int kend = t < t0 ? p.nslots : k0;
            for (int k = 0; k < kend; ++k) {
                before += p.ids[k + t * p.ids_stride] == e;
            }
        }
        if (before % NC != 0) {
            return;
        }
        for (int t = t0, k = k0; t < p.ncols && nt < NC; ) {
            if (p.ids[k + t * p.ids_stride] == e) {
                const uint32_t channel_y = fastmodulo((uint32_t) k, p.nchannels_y);
#pragma unroll
                for (int j = 0; j < NC; ++j) {
                    if (j == nt) {
                        yc[j] = (int) (sample_dst*p.sample_bytes_y + channel_y*p.channel_bytes_y + t*p.col_bytes_y);
                        dc[j] = (int) (sample_dst*p.stride_sample_dst + k*p.stride_channel_dst + t*p.stride_col_dst);
                    }
                }
                nt++;
            }
            if (++k == p.nslots) {
                k = 0;
                ++t;
            }
        }
        channel_x    = (uint32_t) e;
        channel_bias = channel_x;
    } else {
        const uint32_t channel_dst = shared_expert ? 0 : blockIdx.y;
        const uint32_t channel_y   = p.ids ? 0 : channel_dst;
        const int64_t  scol_dst    = shared_expert ? fusion.shared_stride_col_dst : p.stride_col_dst;
        nt = p.ncols;
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            yc[j] = (int) (sample_dst*p.sample_bytes_y + channel_y*p.channel_bytes_y + j*p.col_bytes_y);
            dc[j] = (int) (sample_dst*p.stride_sample_dst + channel_dst*p.stride_channel_dst + j*scol_dst);
        }
        channel_x    = shared_expert ? 0 : fastdiv(channel_dst, p.channel_ratio);
        channel_bias = channel_dst;
    }

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
    const int wstep = WB << lpr_log2;
    const int ystep = YB << lpr_log2;
    [[maybe_unused]] char * smw = (char *) sm[staged ? threadIdx.y : 0] + (staged ? grp * R * M * WB << lpr_log2 : 0);
    [[maybe_unused]] const int seg_units = (WB << lpr_log2) >> 4;

    float acc[NC][M][R];
#pragma unroll
    for (int j = 0; j < NC; ++j) {
#pragma unroll
        for (int m = 0; m < M; ++m) {
#pragma unroll
            for (int i = 0; i < R; ++i) {
                acc[j][m][i] = 0.0f;
            }
        }
    }

    const char * yp = p.vy + sl * YB;
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
        // sa: every row's chunk goes into registers before the first token, because the warp barriers between tokens
        // keep the compiler from reusing the chunks it loaded for the previous token (without sa it reuses them)
        [[maybe_unused]] uint32_t wr[sa ? M : 1][sa ? R : 1][WB / 4];
        if constexpr (sa) {
#pragma unroll
            for (int m = 0; m < M; ++m) {
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    if constexpr (staged) {
                        mmvqp_lds_words<WB, mmvqp_gcd(WB, 16)>(wr[m][i], smw + ((m*R + i) * WB << lpr_log2) + sl * WB);
                    } else {
                        mmvqp_ldg_words<WB, mmvqp_gcd(WB, 16)>(wr[m][i], rw[m][i]);
                    }
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            if (j >= nt) {
                break;
            }
            mmvqp_act<NA> a;
            if constexpr (sa) {
                // all lane groups of the warp read the same segment of lpr * YB bytes
                const uint4 * src  = (const uint4 *) (yp - sl * YB + yc[j]);
                const int     segy = (YB << lpr_log2) >> 4;
                __syncwarp();
#pragma unroll
                for (int u0 = 0; u0 < NY; ++u0) {
                    const int u = lane + u0 * WARP_SIZE;
                    if (u < segy) {
                        sm_y[threadIdx.y][u] = __ldg(src + u);
                    }
                }
                __syncwarp();
                mmvqp_load_act<type, NA, true>(a, (const char *) sm_y[threadIdx.y] + sl * YB);
            } else {
                mmvqp_load_act<type, NA>(a, yp + yc[j]);
            }
#pragma unroll
            for (int m = 0; m < M; ++m) {
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    if constexpr (sa) {
                        acc[j][m][i] = mmvqp_dot<type, C>(wr[m][i], a, acc[j][m][i]);
                    } else {
                        uint32_t w[WB / 4];
                        if constexpr (staged) {
                            mmvqp_lds_words<WB, mmvqp_gcd(WB, 16)>(w, smw + ((m*R + i) * WB << lpr_log2) + sl * WB);
                        } else {
                            mmvqp_ldg_words<WB, mmvqp_gcd(WB, 16)>(w, rw[m][i]);
                        }
                        acc[j][m][i] = mmvqp_dot<type, C>(w, a, acc[j][m][i]);
                    }
                }
            }
        }
        if constexpr (staged) {
            __syncwarp();
        }
#pragma unroll
        for (int m = 0; m < M; ++m) {
#pragma unroll
            for (int i = 0; i < R; ++i) {
                rw[m][i] += wstep;
            }
        }
        yp += ystep;
    }

#pragma unroll
    for (int j = 0; j < NC; ++j) {
        if (j >= nt) {
            break;
        }
#pragma unroll
        for (int i = 0; i < R; ++i) {
#pragma unroll
            for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
                if (off < lpr) {
#pragma unroll
                    for (int m = 0; m < M; ++m) {
                        acc[j][m][i] += __shfl_xor_sync(0xFFFFFFFF, acc[j][m][i], off, WARP_SIZE);
                    }
                }
            }
        }
    }

    float * dst = shared_expert ? fusion.shared_dst : p.dst;
    [[maybe_unused]] const int64_t bias_off = sample_dst*p.stride_sample_dst + channel_bias*p.stride_channel_dst;
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        if (j >= nt) {
            break;
        }
#pragma unroll
        for (int i = 0; i < R; ++i) {
            const int row = row0 + i;
            if ((i & (lpr - 1)) != sl || row >= p.nrows) {
                continue;
            }
            float result = acc[j][0][i];
            if constexpr (fk != MMVQP_PLAIN) {
                if (fusion.x_bias) {
                    result += ((const float *) fusion.x_bias)[bias_off + row];
                }
            }
            if constexpr (fk == MMVQP_GATE) {
                float gate_value = acc[j][M - 1][i];
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
            dst[dc[j] + row] = result;
        }
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
        case GGML_TYPE_Q5_0:
            return gate ? mmvqp_shape{ 1, 2, true } : mmvqp_shape{ 2, 2, true };
        case GGML_TYPE_Q8_0:
            return gate ? mmvqp_shape{ 1, 2, true } : mmvqp_shape{ 2, 2, true };
        case GGML_TYPE_Q5_K:
            return { 1, 1, true };
        case GGML_TYPE_IQ4_XS:
            return gate ? mmvqp_shape{ 1, 1, false } : mmvqp_shape{ 1, 1, true };
        default:
            GGML_ABORT("fatal error");
    }
}

// The single-token kernel for one token, the multi-token one for more.
template <ggml_type type, int R, int C, bool staged, int fk>
static void mmvqp_launch_kernel(const mmvqp_args_mc & p, const ggml_cuda_mm_fusion_args_device & fusion, const dim3 grid,
                                cudaStream_t stream) {
    const dim3 block(WARP_SIZE, MMVQ_PASCAL_NWARPS, 1);
    if constexpr (mmvqp_fits<type, R, C, staged, fk == MMVQP_GATE>()) {
        if (p.ncols == 1) {
            mul_mat_vec_q_pascal<type, R, C, staged, fk><<<grid, block, 0, stream>>>(p, fusion);
            return;
        }
        if constexpr (mmvqp_fits_mc<type, R, C, staged, fk == MMVQP_GATE, true>()) {
            if (p.stage_act) {
                mul_mat_vec_q_pascal_mc<type, R, C, staged, fk, true><<<grid, block, 0, stream>>>(p, fusion);
                return;
            }
        }
        mul_mat_vec_q_pascal_mc<type, R, C, staged, fk, false><<<grid, block, 0, stream>>>(p, fusion);
    } else {
        GGML_ABORT("fatal error");
    }
}

template <ggml_type type, int R, int C, bool staged>
static void mmvqp_launch(const mmvqp_args_mc & p, const ggml_cuda_mm_fusion_args_device & fusion, const int fk,
                         const dim3 grid, cudaStream_t stream) {
    switch (fk) {
        case MMVQP_PLAIN:
            mmvqp_launch_kernel<type, R, C, staged, MMVQP_PLAIN>(p, fusion, grid, stream);
            break;
        case MMVQP_BIAS:
            mmvqp_launch_kernel<type, R, C, staged, MMVQP_BIAS>(p, fusion, grid, stream);
            break;
        default:
            mmvqp_launch_kernel<type, R, C, staged, MMVQP_GATE>(p, fusion, grid, stream);
            break;
    }
}

template <ggml_type type>
static void mmvqp_launch_shape(const mmvqp_shape s, const mmvqp_args_mc & p, const ggml_cuda_mm_fusion_args_device & fusion,
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
    } else if constexpr (type == GGML_TYPE_Q5_0) {
        MMVQP_CASE(1, 2, true)
        MMVQP_CASE(1, 2, false)
        MMVQP_CASE(2, 2, true)
        MMVQP_CASE(2, 2, false)
    } else if constexpr (type == GGML_TYPE_Q5_K || type == GGML_TYPE_IQ4_XS) {
        MMVQP_CASE(1, 1, true)
        MMVQP_CASE(1, 1, false)
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

// GGML_CUDA_MMVQ_PASCAL_MC=0: 2 to MMVQP_MAX_COLS tokens go to llama.cpp's kernels, as before the multi-token kernels;
// one token keeps the Pascal kernels
static bool mmvqp_multi_col() {
    static const bool mc = [] {
        const char * env = getenv("GGML_CUDA_MMVQ_PASCAL_MC");
        return env == nullptr || atoi(env) != 0;
    }();
    return mc;
}

// GGML_CUDA_MMVQ_PASCAL_STAGE_ACT=1: the multi-token kernels stage each token's activations through shared memory
static bool mmvqp_stage_act() {
    static const bool stage = [] {
        const char * env = getenv("GGML_CUDA_MMVQ_PASCAL_STAGE_ACT");
        return env && atoi(env) != 0;
    }();
    return stage;
}

static bool mmvqp_aligned(const void * ptr, const int64_t al) {
    return ptr == nullptr || (uintptr_t) ptr % al == 0;
}

} // namespace

bool ggml_cuda_mmvq_pascal_active(const int cc) {
    const int mode = mmvqp_mode();
    return mode != 0 && GGML_CUDA_CC_IS_NVIDIA(cc) && (mode != 1 || cc == GGML_CUDA_CC_DP4A);
}

bool ggml_cuda_mmvq_pascal(
        const void * vx, ggml_type type, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device & fusion,
        float * dst, int64_t ncols_x, int64_t nrows_x, int64_t ncols_dst, int64_t stride_row_x, int64_t stride_col_y,
        int64_t stride_col_dst, int64_t nchannels_x, int64_t nchannels_y, int64_t nchannels_dst, int64_t stride_channel_x,
        int64_t stride_channel_y, int64_t stride_channel_dst, int64_t nsamples_x, int64_t nsamples_dst, int64_t stride_sample_x,
        int64_t stride_sample_y, int64_t stride_sample_dst, int64_t ids_stride, cudaStream_t stream) {
    if (ncols_dst < 1 || ncols_dst > MMVQP_MAX_COLS || (ncols_dst > 1 && !mmvqp_multi_col()) ||
            !ggml_cuda_mmvq_pascal_active(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
        return false;
    }
    if (type != GGML_TYPE_Q4_0 && type != GGML_TYPE_Q4_1 && type != GGML_TYPE_Q5_0 && type != GGML_TYPE_Q8_0 &&
            type != GGML_TYPE_Q5_K && type != GGML_TYPE_IQ4_XS) {
        return false;
    }
    if (fusion.x_scale || fusion.gate_scale) {
        return false;
    }
    // llama.cpp fuses several tokens' products only for MUL_MAT_ID
    if (ncols_dst > 1 && !ids && (fusion.gate || fusion.x_bias)) {
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
    const int64_t aly = mmvqp_gcd((int) (s.C * (ggml_blck_size(type) / QK8_1) * sizeof(block_q8_1)), 16);
    if (!mmvqp_aligned(vy, aly) || (stride_channel_y * (int64_t) sizeof(block_q8_1)) % aly != 0 ||
            (stride_sample_y * (int64_t) sizeof(block_q8_1)) % aly != 0 ||
            (ncols_dst > 1 && (stride_col_y * (int64_t) sizeof(block_q8_1)) % aly != 0)) {
        return false;
    }

    // the multi-token kernel keeps the offsets of each token's activations and outputs in an int
    if (ncols_dst > 1) {
        const int64_t y_extent = (nsamples_dst - 1) * stride_sample_y + (std::max(nchannels_y, nchannels_dst) - 1) *
            stride_channel_y + (ncols_dst - 1) * stride_col_y + blocks * (ggml_blck_size(type) / QK8_1);
        const int64_t d_extent = (nsamples_dst - 1) * stride_sample_dst + (nchannels_dst - 1) * stride_channel_dst +
            (ncols_dst - 1) * std::max(stride_col_dst, (int64_t) fusion.shared_stride_col_dst) + nrows_x;
        if (y_extent * (int64_t) sizeof(block_q8_1) > INT_MAX || d_extent > INT_MAX) {
            return false;
        }
    }

    mmvqp_args_mc p;
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
    p.col_bytes_y        = stride_col_y * (int64_t) sizeof(block_q8_1);
    p.stride_col_dst     = stride_col_dst;
    p.ncols              = (int) ncols_dst;
    p.nslots             = (int) nchannels_dst;
    p.ids_stride         = (int) ids_stride;
    // staged activation segments start at 16-byte boundaries
    const int64_t yb_seg = (s.C * (ggml_blck_size(type) / QK8_1) * (int64_t) sizeof(block_q8_1)) << lpr_log2;
    p.stage_act          = ncols_dst > 1 && mmvqp_stage_act() && yb_seg % 16 == 0 && mmvqp_aligned(vy, 16) &&
        (stride_col_y * (int64_t) sizeof(block_q8_1)) % 16 == 0 &&
        (stride_channel_y * (int64_t) sizeof(block_q8_1)) % 16 == 0 &&
        (stride_sample_y * (int64_t) sizeof(block_q8_1)) % 16 == 0;

    // several tokens with ids: one block per (token, slot) entry
    const int64_t nblocks_y = ids && ncols_dst > 1 ? nchannels_dst * ncols_dst : nchannels_dst;

    const int  fk             = gate ? MMVQP_GATE : (fusion.x_bias ? MMVQP_BIAS : MMVQP_PLAIN);
    const int  rows_per_block = MMVQ_PASCAL_NWARPS * (WARP_SIZE >> lpr_log2) * s.R;
    const dim3 grid((unsigned) ((nrows_x + rows_per_block - 1) / rows_per_block),
                    (unsigned) (nblocks_y + (fusion.shared_up ? 1 : 0)), (unsigned) nsamples_dst);

    switch (type) {
        case GGML_TYPE_Q4_0: mmvqp_launch_shape<GGML_TYPE_Q4_0>(s, p, fusion, fk, grid, stream); break;
        case GGML_TYPE_Q4_1: mmvqp_launch_shape<GGML_TYPE_Q4_1>(s, p, fusion, fk, grid, stream); break;
        case GGML_TYPE_Q5_0: mmvqp_launch_shape<GGML_TYPE_Q5_0>(s, p, fusion, fk, grid, stream); break;
        case GGML_TYPE_Q5_K: mmvqp_launch_shape<GGML_TYPE_Q5_K>(s, p, fusion, fk, grid, stream); break;
        case GGML_TYPE_IQ4_XS: mmvqp_launch_shape<GGML_TYPE_IQ4_XS>(s, p, fusion, fk, grid, stream); break;
        case GGML_TYPE_Q8_0: mmvqp_launch_shape<GGML_TYPE_Q8_0>(s, p, fusion, fk, grid, stream); break;
        default:             GGML_ABORT("fatal error");
    }
    return true;
}

#else

bool ggml_cuda_mmvq_pascal_active(const int cc) {
    GGML_UNUSED(cc);
    return false;
}

bool ggml_cuda_mmvq_pascal(
        const void * vx, ggml_type type, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device & fusion,
        float * dst, int64_t ncols_x, int64_t nrows_x, int64_t ncols_dst, int64_t stride_row_x, int64_t stride_col_y,
        int64_t stride_col_dst, int64_t nchannels_x, int64_t nchannels_y, int64_t nchannels_dst, int64_t stride_channel_x,
        int64_t stride_channel_y, int64_t stride_channel_dst, int64_t nsamples_x, int64_t nsamples_dst, int64_t stride_sample_x,
        int64_t stride_sample_y, int64_t stride_sample_dst, int64_t ids_stride, cudaStream_t stream) {
    GGML_UNUSED_VARS(vx, type, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                     nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                     nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
    return false;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
