#pragma once

#include "common.cuh"

// The device code of the Pascal matrix-vector products (mmvq-pascal.cu) that the persistent MoE kernel
// (moe-persistent.cu) shares, so that both compile the same arithmetic: weight block layouts, loads, activation blocks
// with FP32 sums, the dot products per type, and the launch shape per type.

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// Bytes and Q8_1 activation blocks per weight block.
template <ggml_type type> struct mmvqp_traits;
template <> struct mmvqp_traits<GGML_TYPE_Q4_0>   { static constexpr int bs = sizeof(block_q4_0);   static constexpr int na = 1; };
template <> struct mmvqp_traits<GGML_TYPE_Q4_1>   { static constexpr int bs = sizeof(block_q4_1);   static constexpr int na = 1; };
template <> struct mmvqp_traits<GGML_TYPE_Q5_0>   { static constexpr int bs = sizeof(block_q5_0);   static constexpr int na = 1; };
template <> struct mmvqp_traits<GGML_TYPE_Q8_0>   { static constexpr int bs = sizeof(block_q8_0);   static constexpr int na = 1; };
template <> struct mmvqp_traits<GGML_TYPE_Q4_K>   { static constexpr int bs = sizeof(block_q4_K);   static constexpr int na = QK_K / QK8_1; };
template <> struct mmvqp_traits<GGML_TYPE_Q5_K>   { static constexpr int bs = sizeof(block_q5_K);   static constexpr int na = QK_K / QK8_1; };
template <> struct mmvqp_traits<GGML_TYPE_IQ4_XS> { static constexpr int bs = sizeof(block_iq4_xs); static constexpr int na = QK_K / QK8_1; };
template <> struct mmvqp_traits<GGML_TYPE_Q6_K>   { static constexpr int bs = sizeof(block_q6_K);   static constexpr int na = QK_K / QK8_1; };
template <> struct mmvqp_traits<GGML_TYPE_IQ2_S>   { static constexpr int bs = sizeof(block_iq2_s);   static constexpr int na = QK_K / QK8_1; };
template <> struct mmvqp_traits<GGML_TYPE_IQ3_XXS> { static constexpr int bs = sizeof(block_iq3_xxs); static constexpr int na = QK_K / QK8_1; };
template <> struct mmvqp_traits<GGML_TYPE_N2P>     { static constexpr int bs = sizeof(block_n2p);     static constexpr int na = QK_K / QK8_1; };
template <> struct mmvqp_traits<GGML_TYPE_N3P>     { static constexpr int bs = sizeof(block_n3p);     static constexpr int na = QK_K / QK8_1; };

// Tuning switches of the grid types' kernels, set at compile time (-D); the defaults are the fastest of 11 variants on a
// P106 (one-token expert products at Ornith-1.5-122B's shapes, 2026-10-27):
//   MMVQP_IQ_SIGN_LUT    1: a weight group's sign masks come from a shared-memory table; 0: from its sign bits, in registers
//   MMVQP_IQ_MIN_BLOCKS  blocks per SM the compiler keeps registers for (__launch_bounds__), 1: no limit
//   MMVQP_IQ_R           rows per lane group
//   MMVQP_IQ_PERSIST     one token: blocks per SM, each looping over row blocks, so a block fills its tables once for
//                        many rows; 0: one block per row block
#ifndef MMVQP_IQ_PERSIST
#define MMVQP_IQ_PERSIST 0
#endif
#ifndef MMVQP_IQ_SIGN_LUT
#define MMVQP_IQ_SIGN_LUT 0
#endif
#ifndef MMVQP_IQ_MIN_BLOCKS
#define MMVQP_IQ_MIN_BLOCKS 3
#endif
#ifndef MMVQP_IQ_R
#define MMVQP_IQ_R 1
#endif
template <ggml_type type> constexpr __host__ __device__ bool mmvqp_grid_type() {
    return type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_XXS;
}
template <ggml_type type> constexpr __host__ __device__ int mmvqp_min_blocks() {
    return mmvqp_grid_type<type>() ? MMVQP_IQ_MIN_BLOCKS : 1;
}

// Lookup tables of the grid types, copied into each block's shared memory by mmvqp_init_tabs (32-bit words): the grid
// (IQ2_S: 1024 entries of 8 bytes, IQ3_XXS: 256 entries of 4 bytes), then for every sign pattern two words of byte
// masks, 0xFF where a weight is negative, for its first and its last four weights (IQ2_S: the 256 sign bytes; IQ3_XXS:
// the 128 7-bit patterns, whose eighth sign is their parity).
template <ggml_type type> constexpr __host__ __device__ int mmvqp_tab_grid_words() {
    return type == GGML_TYPE_IQ2_S ? 2*1024 : type == GGML_TYPE_IQ3_XXS ? 256 : 0;
}
template <ggml_type type> constexpr __host__ __device__ int mmvqp_tab_words() {
    return type == GGML_TYPE_IQ2_S ? 2*1024 + 2*256 : type == GGML_TYPE_IQ3_XXS ? 256 + 2*128 : 0;
}

// 4 sign bits -> 4 byte masks (bit j -> 0xFF in byte j)
static __device__ __forceinline__ uint32_t mmvqp_sign_bytes(const uint32_t s4) {
    return ((s4 * 0x00204081u) & 0x01010101u) * 0xFFu;
}

// Fills the tables of a grid type in shared memory, with all threads of the block; the caller synchronizes the block.
template <ggml_type type>
static __device__ __forceinline__ void mmvqp_init_tabs(uint32_t * tab) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int nth = blockDim.x * blockDim.y;
    constexpr int G = mmvqp_tab_grid_words<type>();
    if constexpr (type == GGML_TYPE_IQ2_S) {
        const uint32_t * grid = (const uint32_t *) iq2s_grid;
        for (int i = tid; i < G; i += nth) {
            tab[i] = grid[i];
        }
        for (int s = tid; MMVQP_IQ_SIGN_LUT && s < 256; s += nth) {
            tab[G + 2*s + 0] = mmvqp_sign_bytes(s & 0xF);
            tab[G + 2*s + 1] = mmvqp_sign_bytes(s >> 4);
        }
    } else if constexpr (type == GGML_TYPE_IQ3_XXS) {
        for (int i = tid; i < G; i += nth) {
            tab[i] = iq3xxs_grid[i];
        }
        for (int s = tid; MMVQP_IQ_SIGN_LUT && s < 128; s += nth) {
            const uint32_t s8 = s | ((__popc(s) & 1) << 7); // ksigns_iq2xs[s]
            tab[G + 2*s + 0] = mmvqp_sign_bytes(s8 & 0xF);
            tab[G + 2*s + 1] = mmvqp_sign_bytes(s8 >> 4);
        }
    }
    GGML_UNUSED_VARS(tab, tid, nth);
}

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

// NB bytes (a multiple of 4) from global memory aligned to AL bytes, through L2 only.
template <int NB, int AL>
static __device__ __forceinline__ void mmvqp_ldcg_words(uint32_t * w, const char * p) {
    static_assert(NB % 4 == 0 && AL >= 4, "chunk must be 4-byte aligned");
    if constexpr (AL >= 16 && NB % 16 == 0) {
#pragma unroll
        for (int i = 0; i < NB/16; ++i) {
            const uint4 v = __ldcg((const uint4 *) p + i);
            w[4*i + 0] = v.x; w[4*i + 1] = v.y; w[4*i + 2] = v.z; w[4*i + 3] = v.w;
        }
    } else if constexpr (AL >= 8 && NB % 8 == 0) {
#pragma unroll
        for (int i = 0; i < NB/8; ++i) {
            const uint2 v = __ldcg((const uint2 *) p + i);
            w[2*i + 0] = v.x; w[2*i + 1] = v.y;
        }
    } else {
#pragma unroll
        for (int i = 0; i < NB/4; ++i) {
            w[i] = __ldcg((const unsigned int *) p + i);
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
    int   t[NA][8];  // Q5_0: t[j] = bytes a[j], a[8 + j], a[16 + j], a[24 + j]; Q6_K: t[0], t[1] = 32 x the sums of a[0..15], a[16..31]
    float d[NA];     // d8 (Q8_0, K-quants, IQ4_XS) or d8 / 16 (4- and 5-bit legacy types)
    float s[NA];     // with S = d8 * the sum of the int8 values: S (Q4_1, Q4_K, Q5_K), -8 * S (Q4_0), -16 * S (Q5_0)
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

// Where activations are read from: global memory through the read-only path, activations staged in shared memory, or
// global memory through L2 only (data written earlier in the same launch, which the read-only path may not see).
enum mmvqp_src { MMVQP_SRC_LDG = 0, MMVQP_SRC_SMEM = 1, MMVQP_SRC_LDCG = 2 };

template <ggml_type type, int NA, int src = MMVQP_SRC_LDG>
static __device__ __forceinline__ void mmvqp_load_act(mmvqp_act<NA> & a, const char * p) {
    constexpr int NB = NA * sizeof(block_q8_1);
    uint32_t w[NB / 4];
    if constexpr (src == MMVQP_SRC_SMEM) {
        mmvqp_lds_words<NB, mmvqp_gcd(NB, 16)>(w, p);
    } else if constexpr (src == MMVQP_SRC_LDCG) {
        mmvqp_ldcg_words<NB, mmvqp_gcd(NB, 16)>(w, p);
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
        // The offsets (Q4_0, Q5_0) and mins (Q4_1, Q4_K, Q5_K) need the block's sum. It is taken from the int8 values,
        // in FP32, not from the FP16 sum that block_q8_1 stores next to d8: that one becomes infinite once the block's
        // activations add up past 65504 (Flint's expert down-projections in layers 46-48), while d8 * sum(int8) is
        // finite whenever d8 is. 8 dp4a per activation block, shared by all rows of the lane group.
        float S = 0.0f;
        if constexpr (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q4_K ||
                      type == GGML_TYPE_Q5_K) {
            int sum = 0;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                sum = ggml_cuda_dp4a(0x01010101, a.u[b][k], sum);
            }
            S = d8 * (float) sum;
        }
        if constexpr (type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_XXS ||
                      type == GGML_TYPE_N2P || type == GGML_TYPE_N3P) {
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
        } else if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
            a.d[b] = d8;
            a.s[b] = S;
        } else if constexpr (type == GGML_TYPE_Q6_K) {
            int s0 = 0;
            int s1 = 0;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                s0 = ggml_cuda_dp4a(0x01010101, a.u[b][k], s0);
                s1 = ggml_cuda_dp4a(0x01010101, a.u[b][k + 4], s1);
            }
            a.t[b][0] = s0 << 5;
            a.t[b][1] = s1 << 5;
            a.d[b] = d8;
            a.s[b] = 0.0f;
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

// One Q4_K super-block as 36 words: dm, 3 words of scales, 32 words qs; activation blocks a0..a0+7. Sub-blocks 2j and
// 2j + 1 take the low and high nibbles of qs words 4 + 8j .. 4 + 8j + 7.
template <int NA>
static __device__ __forceinline__ float mmvqp_dot_q4_K(const uint32_t * w, const mmvqp_act<NA> & a, const int a0, float acc) {
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
            const uint32_t q = w[4 + 8*(s/2) + k];
            if (s & 1) {
                isum = mmvqp_dp4a_us(q & 0xF0F0F0F0, a.u[a0 + s][k], isum);
            } else {
                isum = ggml_cuda_dp4a((int) (q & 0x0F0F0F0F), a.u[a0 + s][k], isum);
            }
        }
        if (s & 1) {
            isum >>= 4; // the high nibbles' sum is exactly 16 times too large
        }
        const int sc = (int) (((s < 4 ? sc_lo : sc_hi) >> (8 * (s & 3))) & 0xff);
        const int m  = (int) (((s < 4 ? m_lo  : m_hi ) >> (8 * (s & 3))) & 0xff);
        sumd = fmaf(a.d[a0 + s], (float) (isum * sc), sumd);
        summ = fmaf((float) m, a.s[a0 + s], summ);
    }
    return acc + mmvqp_h2f_lo(w[0]) * sumd - mmvqp_h2f_hi(w[0]) * summ;
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

// One Q6_K super-block as 53 words (4-byte aligned here): ql words 0..31, qh words 32..47, scale words 48..51, d in the
// low half of word 52; activation blocks a0..a0+7. Sub-block s = 4h + k takes the low (k < 2) or high nibbles of ql
// words 16h + 8(k & 1) + j and bits 2k, 2k + 1 of qh words 32 + 8h + j, with scales[2s] for its first 16 weights and
// scales[2s + 1] for the others.
template <int NA>
static __device__ __forceinline__ float mmvqp_dot_q6_K(const uint32_t * w, const mmvqp_act<NA> & a, const int a0, float acc) {
    float sumf = 0.0f;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        const int h = s >> 2;
        const int k = s & 3;
        int isum0 = 0;
        int isum1 = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint32_t ql = w[16*h + 8*(k & 1) + j];
            const uint32_t qh = w[32 + 8*h + j];
            const uint32_t lo = k < 2 ? ql & 0x0F0F0F0F : (ql >> 4) & 0x0F0F0F0F;
            const uint32_t hi = k == 0 ? (qh << 4) & 0x30303030 : k == 1 ? (qh << 2) & 0x30303030
                              : k == 2 ? qh & 0x30303030 : (qh >> 2) & 0x30303030;
            if (j < 4) {
                isum0 = ggml_cuda_dp4a((int) (lo | hi), a.u[a0 + s][j], isum0);
            } else {
                isum1 = ggml_cuda_dp4a((int) (lo | hi), a.u[a0 + s][j], isum1);
            }
        }
        const uint32_t sw  = w[48 + (s >> 1)];
        const int      sc0 = (int) (int8_t) (sw >> (16 * (s & 1)));
        const int      sc1 = (int) (int8_t) (sw >> (16 * (s & 1) + 8));
        sumf = fmaf(a.d[a0 + s], (float) (sc0 * (isum0 - a.t[a0 + s][0]) + sc1 * (isum1 - a.t[a0 + s][1])), sumf);
    }
    return fmaf(mmvqp_h2f_lo(w[52]), sumf, acc);
}

// The grid types: the grid entries are unsigned (IQ2_S 8/25/43, IQ3_XXS 4..62), and a weight's sign is applied as
// sum(s * g * u) = sum(g * u) - 2 * sum((g & m) * u) with m the byte mask of the negative weights, two dp4a and one AND
// per 4 weights. Each 16- or 32-weight scale multiplies its integer sum once, in FP32: no integer division.

// The byte masks of a group of 8 weights from its sign pattern (IQ2_S: a sign byte; IQ3_XXS: 7 bits, the eighth sign
// being their parity): from the shared-memory table, or computed in registers (MMVQP_IQ_SIGN_LUT=0).
template <ggml_type type>
static __device__ __forceinline__ uint2 mmvqp_iq_masks(const uint2 * masks, const uint32_t s) {
#if MMVQP_IQ_SIGN_LUT
    return masks[s];
#else
    uint32_t s8 = s;
    if constexpr (type == GGML_TYPE_IQ3_XXS) {
        s8 |= (__popc(s) & 1) << 7;
    }
    GGML_UNUSED(masks);
    return make_uint2(mmvqp_sign_bytes(s8 & 0xF), mmvqp_sign_bytes(s8 >> 4));
#endif
}

// One IQ2_S super-block as 21 words (bytes: d 0-1; qs 2-65: 32 index bytes, then 32 sign bytes; qh 66-73: the 2 high
// index bits of each group of 8 weights; scales 74-81: a 4-bit scale per 16 weights); activation blocks a0..a0+7.
// Weights = d * (0.5 + scale) / 4 * grid * sign.
template <int NA>
static __device__ __forceinline__ float mmvqp_dot_iq2_s(const uint32_t * w, const mmvqp_act<NA> & a, const int a0,
                                                        const uint32_t * tab, float acc) {
    const uint2 * grid  = (const uint2 *) tab;
    const uint2 * masks = (const uint2 *) (tab + mmvqp_tab_grid_words<GGML_TYPE_IQ2_S>());
    float sumf = 0.0f;
#pragma unroll
    for (int ib = 0; ib < 8; ++ib) {
        const uint32_t qs = mmvqp_word_at(w, 2 + 4*ib);
        const uint32_t sg = mmvqp_word_at(w, 34 + 4*ib);
        const uint32_t qh = (mmvqp_word_at(w, 66 + 4*(ib/4)) >> (8*(ib%4))) & 0xff;
        const uint32_t sc = (mmvqp_word_at(w, 74 + 4*(ib/4)) >> (8*(ib%4))) & 0xff;
        int p[2] = {0, 0};
        int n[2] = {0, 0};
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 g = grid[((qs >> (8*l)) & 0xff) | (((qh >> (2*l)) & 3) << 8)];
            const uint2 m = mmvqp_iq_masks<GGML_TYPE_IQ2_S>(masks, (sg >> (8*l)) & 0xff);
            const int  u0 = a.u[a0 + ib][2*l + 0];
            const int  u1 = a.u[a0 + ib][2*l + 1];
            p[l/2] = ggml_cuda_dp4a((int) g.x, u0, p[l/2]);
            p[l/2] = ggml_cuda_dp4a((int) g.y, u1, p[l/2]);
            n[l/2] = ggml_cuda_dp4a((int) (g.x & m.x), u0, n[l/2]);
            n[l/2] = ggml_cuda_dp4a((int) (g.y & m.y), u1, n[l/2]);
        }
        const int isum = (2*(int) (sc & 0xF) + 1) * (p[0] - 2*n[0]) + (2*(int) (sc >> 4) + 1) * (p[1] - 2*n[1]);
        sumf = fmaf(a.d[a0 + ib], (float) isum, sumf);
    }
    return fmaf(0.125f * mmvqp_h2f_lo(w[0]), sumf, acc);
}

// One IQ3_XXS super-block as 25 words (bytes: d 0-1; 64 index bytes 2-65, one per 4 weights; 66-97: per 32 weights a
// word of four 7-bit sign patterns and a 4-bit scale on top); activation blocks a0..a0+7.
// Weights = d * (0.5 + scale) / 2 * grid * sign.
template <int NA>
static __device__ __forceinline__ float mmvqp_dot_iq3_xxs(const uint32_t * w, const mmvqp_act<NA> & a, const int a0,
                                                          const uint32_t * tab, float acc) {
    const uint2 * masks = (const uint2 *) (tab + mmvqp_tab_grid_words<GGML_TYPE_IQ3_XXS>());
    float sumf = 0.0f;
#pragma unroll
    for (int ib = 0; ib < 8; ++ib) {
        const uint32_t q0  = mmvqp_word_at(w, 2 + 8*ib);
        const uint32_t q1  = mmvqp_word_at(w, 6 + 8*ib);
        const uint32_t aux = mmvqp_word_at(w, 66 + 4*ib);
        int p = 0;
        int n = 0;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t q  = l < 2 ? q0 : q1;
            const uint32_t g0 = tab[(q >> (16*(l & 1)))     & 0xff];
            const uint32_t g1 = tab[(q >> (16*(l & 1) + 8)) & 0xff];
            const uint2    m  = mmvqp_iq_masks<GGML_TYPE_IQ3_XXS>(masks, (aux >> (7*l)) & 0x7f);
            const int      u0 = a.u[a0 + ib][2*l + 0];
            const int      u1 = a.u[a0 + ib][2*l + 1];
            p = ggml_cuda_dp4a((int) g0, u0, p);
            p = ggml_cuda_dp4a((int) g1, u1, p);
            n = ggml_cuda_dp4a((int) (g0 & m.x), u0, n);
            n = ggml_cuda_dp4a((int) (g1 & m.y), u1, n);
        }
        sumf = fmaf(a.d[a0 + ib], (float) ((2*(int) (aux >> 28) + 1) * (p - 2*n)), sumf);
    }
    return fmaf(0.25f * mmvqp_h2f_lo(w[0]), sumf, acc);
}

// N2P and N3P decode their codes with __byte_perm from level tables held in registers (common.cuh: N2P_TAB*, N3P_TAB*);
// each sub-block's shift bit selects the table registers once, so the inner loops are a shift, a permute and a dp4a per
// 4 weights (N3P: a second shift and a LOP3 to merge its third bits). Their levels carry their own offset: no activation
// sums.

// One N2P super-block as 22 words (4-byte aligned here): qs words 0..15 (word k: weights 16k..16k+15), scale words
// 16..19 (a nibble per 8 weights), shift bits word 20, d in the low half of word 21; activation blocks a0..a0+7.
template <int NA>
static __device__ __forceinline__ float mmvqp_dot_n2p(const uint32_t * w, const mmvqp_act<NA> & a, const int a0, float acc) {
    const uint32_t ex = w[20];
    float sumf = 0.0f;
#pragma unroll
    for (int ib = 0; ib < 8; ++ib) {
        const uint32_t sc = w[16 + ib/2] >> 16*(ib%2); // scales of sub-blocks 4ib..4ib+3
        int isum = 0;
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const uint32_t q = w[2*ib + k];
#pragma unroll
            for (int sb = 0; sb < 2; ++sb) {
                const uint32_t T = (ex >> (4*ib + 2*k + sb)) & 1 ? N2P_TAB1 : N2P_TAB0;
                int sumb = ggml_cuda_dp4a((int) __byte_perm(T, T, q >> (16*sb)),     a.u[a0 + ib][4*k + 2*sb],     0);
                sumb     = ggml_cuda_dp4a((int) __byte_perm(T, T, q >> (16*sb + 2)), a.u[a0 + ib][4*k + 2*sb + 1], sumb);
                isum += sumb * ((int) ((sc >> 4*(2*k + sb)) & 0xF) - 8);
            }
        }
        sumf = fmaf(a.d[a0 + ib], (float) isum, sumf);
    }
    return fmaf(mmvqp_h2f_lo(w[21]), sumf, acc);
}

// One N3P super-block as 26 words: d | extra << 16 | scales_h << 24, scales_l, qs words 2..17 (two per 32 weights), qh
// words 18..25 (one per 32 weights); activation blocks a0..a0+7. The selector of weights 4g..4g+3 of a sub-block takes
// the low code bits from qs shifted by {0, 2, 16, 18}[g%4] and the third bits from qh shifted by 16*(g/4) + g%4 - 2;
// bit 3 of each selector nibble, which the merge fills with a neighbour's third bit, is ignored by __byte_perm.
template <int NA>
static __device__ __forceinline__ float mmvqp_dot_n3p(const uint32_t * w, const mmvqp_act<NA> & a, const int a0, float acc) {
    const uint32_t w0 = w[0];
    const uint32_t sl = w[1];
    float sumf = 0.0f;
#pragma unroll
    for (int ib = 0; ib < 8; ++ib) {
        const bool     sh = (w0 >> (16 + ib)) & 1;
        const uint32_t ta = sh ? N3P_TAB1A : N3P_TAB0A;
        const uint32_t tb = sh ? N3P_TAB1B : N3P_TAB0B;
        const uint32_t hi = w[18 + ib];
        int isum = 0;
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const uint32_t lo  = w[2 + 2*ib + g/4];
            const int      t   = 16*((g%4)/2) + 2*(g%2);
            const int      s   = 16*(g/4) + g%4 - 2;
            const uint32_t hs  = s >= 0 ? hi >> s : hi << -s;
            const uint32_t sel = ((lo >> t) & 0x3333) | (hs & 0xCCCC);
            isum = ggml_cuda_dp4a((int) __byte_perm(ta, tb, sel), a.u[a0 + ib][g], isum);
        }
        const int ls = (int) (((sl >> 4*ib) & 0xF) | (((w0 >> (24 + ib)) & 1) << 4)) - 16;
        sumf = fmaf(a.d[a0 + ib], (float) (isum * ls), sumf);
    }
    return fmaf(mmvqp_h2f_lo(w0), sumf, acc);
}

// C weight blocks (raw words) times the matching activation blocks, added to acc. tab: the shared-memory tables of the
// grid types (mmvqp_init_tabs).
template <ggml_type type, int C>
static __device__ __forceinline__ float mmvqp_dot(const uint32_t * w, const mmvqp_act<C * mmvqp_traits<type>::na> & a, float acc,
                                                  const uint32_t * tab = nullptr) {
    constexpr int BS = mmvqp_traits<type>::bs;
#pragma unroll
    for (int b = 0; b < C; ++b) {
        const int o = b * BS;
        if constexpr (type == GGML_TYPE_IQ2_S) {
            static_assert(C == 1, "IQ2_S: one super-block per lane");
            acc = mmvqp_dot_iq2_s(w, a, 8*b, tab, acc);
        } else if constexpr (type == GGML_TYPE_IQ3_XXS) {
            static_assert(C == 1, "IQ3_XXS: one super-block per lane");
            acc = mmvqp_dot_iq3_xxs(w, a, 8*b, tab, acc);
        } else if constexpr (type == GGML_TYPE_Q4_K) {
            acc = mmvqp_dot_q4_K(w + o/4, a, 8*b, acc);
        } else if constexpr (type == GGML_TYPE_Q5_K) {
            acc = mmvqp_dot_q5_K(w + o/4, a, 8*b, acc);
        } else if constexpr (type == GGML_TYPE_IQ4_XS) {
            acc = mmvqp_dot_iq4_xs(w + o/4, a, 8*b, acc);
        } else if constexpr (type == GGML_TYPE_Q6_K) {
            acc = mmvqp_dot_q6_K(w, a, 8*b, acc);
        } else if constexpr (type == GGML_TYPE_N2P) {
            static_assert(C == 1, "N2P: one super-block per lane");
            acc = mmvqp_dot_n2p(w, a, 8*b, acc);
        } else if constexpr (type == GGML_TYPE_N3P) {
            acc = mmvqp_dot_n3p(w + o/4, a, 8*b, acc);
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

// Several tokens (mul_mat_vec_q_pascal_mt): one 32-weight sub-block of a lane's chunk is decoded once into signed int8
// words in activation order, with its integer scales, then multiplied by each token's activations. The integer sums and
// the FP32 operations on them are those of mmvqp_dot, so every token's result is bit-identical to the one-token
// kernel's. The grid types' signs go into the words: (g ^ m) + (m & 0x01010101) negates the bytes under the mask m
// without carries, since every grid byte is at least 1, so a token costs one dp4a per 4 weights instead of two. Q5_0's
// fifth bits are spread into one byte per weight in natural order, which the plain activations multiply (the one-token
// kernel transposes the activations instead).
template <ggml_type type> constexpr __host__ __device__ bool mmvqp_mt_type() {
    return type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_N2P || type == GGML_TYPE_N3P ||
        type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q5_0;
}

struct mmvqp_sub {
    int   q[8];  // weight bytes, activation word k = weights 4k..4k+3 (Q5_0: low nibbles in 0-3, high nibbles in place in 4-7)
    int   h[8];  // Q5_0: the fifth bits, a 0/1 byte per weight
    int   sc[4]; // integer scales: N2P one per 8 weights, IQ2_S one per 16, IQ3_XXS and N3P one per 32
    float d;     // Q8_0, Q5_0: the block's scale
};

// 4 bits -> 4 bytes of 0 or 1 (bit j -> byte j)
static __device__ __forceinline__ uint32_t mmvqp_bit_bytes(const uint32_t s4) {
    return (s4 * 0x00204081u) & 0x01010101u;
}

// grid bytes g with the negative ones (byte mask m) negated
static __device__ __forceinline__ int mmvqp_signed_grid(const uint32_t g, const uint32_t m) {
    return (int) ((g ^ m) + (m & 0x01010101u));
}

// Sub-block ib of weight block b of a lane's chunk (raw words w, as mmvqp_dot reads them).
template <ggml_type type>
static __device__ __forceinline__ void mmvqp_sub_decode(const uint32_t * w, const int b, const int ib, const uint32_t * tab,
                                                        mmvqp_sub & s) {
    constexpr int BS = mmvqp_traits<type>::bs;
    const int o = b * BS;
    if constexpr (type == GGML_TYPE_IQ2_S) {
        const uint2 * grid  = (const uint2 *) tab;
        const uint2 * masks = (const uint2 *) (tab + mmvqp_tab_grid_words<GGML_TYPE_IQ2_S>());
        const uint32_t qs = mmvqp_word_at(w, 2 + 4*ib);
        const uint32_t sg = mmvqp_word_at(w, 34 + 4*ib);
        const uint32_t qh = (mmvqp_word_at(w, 66 + 4*(ib/4)) >> (8*(ib%4))) & 0xff;
        const uint32_t sc = (mmvqp_word_at(w, 74 + 4*(ib/4)) >> (8*(ib%4))) & 0xff;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 g = grid[((qs >> (8*l)) & 0xff) | (((qh >> (2*l)) & 3) << 8)];
            const uint2 m = mmvqp_iq_masks<GGML_TYPE_IQ2_S>(masks, (sg >> (8*l)) & 0xff);
            s.q[2*l + 0] = mmvqp_signed_grid(g.x, m.x);
            s.q[2*l + 1] = mmvqp_signed_grid(g.y, m.y);
        }
        s.sc[0] = 2*(int) (sc & 0xF) + 1;
        s.sc[1] = 2*(int) (sc >> 4) + 1;
    } else if constexpr (type == GGML_TYPE_IQ3_XXS) {
        const uint2 * masks = (const uint2 *) (tab + mmvqp_tab_grid_words<GGML_TYPE_IQ3_XXS>());
        const uint32_t q0  = mmvqp_word_at(w, 2 + 8*ib);
        const uint32_t q1  = mmvqp_word_at(w, 6 + 8*ib);
        const uint32_t aux = mmvqp_word_at(w, 66 + 4*ib);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t q  = l < 2 ? q0 : q1;
            const uint32_t g0 = tab[(q >> (16*(l & 1)))     & 0xff];
            const uint32_t g1 = tab[(q >> (16*(l & 1) + 8)) & 0xff];
            const uint2    m  = mmvqp_iq_masks<GGML_TYPE_IQ3_XXS>(masks, (aux >> (7*l)) & 0x7f);
            s.q[2*l + 0] = mmvqp_signed_grid(g0, m.x);
            s.q[2*l + 1] = mmvqp_signed_grid(g1, m.y);
        }
        s.sc[0] = 2*(int) (aux >> 28) + 1;
    } else if constexpr (type == GGML_TYPE_N2P) {
        const uint32_t ex = w[20];
        const uint32_t sc = w[16 + ib/2] >> 16*(ib%2);
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const uint32_t q = w[2*ib + k];
#pragma unroll
            for (int sb = 0; sb < 2; ++sb) {
                const uint32_t T = (ex >> (4*ib + 2*k + sb)) & 1 ? N2P_TAB1 : N2P_TAB0;
                s.q[4*k + 2*sb + 0] = (int) __byte_perm(T, T, q >> (16*sb));
                s.q[4*k + 2*sb + 1] = (int) __byte_perm(T, T, q >> (16*sb + 2));
                s.sc[2*k + sb] = (int) ((sc >> 4*(2*k + sb)) & 0xF) - 8;
            }
        }
    } else if constexpr (type == GGML_TYPE_N3P) {
        const uint32_t * wb = w + o/4;
        const uint32_t w0 = wb[0];
        const bool     sh = (w0 >> (16 + ib)) & 1;
        const uint32_t ta = sh ? N3P_TAB1A : N3P_TAB0A;
        const uint32_t tb = sh ? N3P_TAB1B : N3P_TAB0B;
        const uint32_t hi = wb[18 + ib];
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const uint32_t lo  = wb[2 + 2*ib + g/4];
            const int      t   = 16*((g%4)/2) + 2*(g%2);
            const int      sf  = 16*(g/4) + g%4 - 2;
            const uint32_t hs  = sf >= 0 ? hi >> sf : hi << -sf;
            const uint32_t sel = ((lo >> t) & 0x3333) | (hs & 0xCCCC);
            s.q[g] = (int) __byte_perm(ta, tb, sel);
        }
        s.sc[0] = (int) (((wb[1] >> 4*ib) & 0xF) | (((w0 >> (24 + ib)) & 1) << 4)) - 16;
    } else if constexpr (type == GGML_TYPE_Q5_0) {
        s.d = mmvqp_h2f_lo(mmvqp_u16_at(w, o));
        const uint32_t qh = mmvqp_word_at(w, o + 2);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t q = mmvqp_word_at(w, o + 6 + 4*k);
            s.q[k]     = (int) (q & 0x0F0F0F0F);
            s.q[4 + k] = (int) (q & 0xF0F0F0F0);
        }
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            s.h[k] = (int) mmvqp_bit_bytes((qh >> (4*k)) & 0xF);
        }
    } else { // GGML_TYPE_Q8_0
        s.d = mmvqp_h2f_lo(mmvqp_u16_at(w, o));
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            s.q[k] = (int) mmvqp_word_at(w, o + 2 + 4*k);
        }
    }
    GGML_UNUSED_VARS(tab, o, ib);
}

// The integer sum of a decoded sub-block with one token's activation words u, as mmvqp_dot forms it.
template <ggml_type type>
static __device__ __forceinline__ int mmvqp_sub_isum(const mmvqp_sub & s, const int * u) {
    if constexpr (type == GGML_TYPE_IQ2_S) {
        int p0 = 0;
        int p1 = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            p0 = ggml_cuda_dp4a(s.q[k],     u[k],     p0);
            p1 = ggml_cuda_dp4a(s.q[4 + k], u[4 + k], p1);
        }
        return s.sc[0] * p0 + s.sc[1] * p1;
    } else if constexpr (type == GGML_TYPE_N2P) {
        int isum = 0;
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            int sumb = ggml_cuda_dp4a(s.q[2*g], u[2*g], 0);
            sumb     = ggml_cuda_dp4a(s.q[2*g + 1], u[2*g + 1], sumb);
            isum += sumb * s.sc[g];
        }
        return isum;
    } else if constexpr (type == GGML_TYPE_Q5_0) {
        int lo = 0;
        int hi = 0;
        int hs = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            lo = ggml_cuda_dp4a(s.q[k], u[k], lo);
            hi = mmvqp_dp4a_us((uint32_t) s.q[4 + k], u[4 + k], hi);
        }
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            hs = ggml_cuda_dp4a(s.h[k], u[k], hs);
        }
        return (lo << 4) + hi + (hs << 8);
    } else { // IQ3_XXS, N3P: one scale per sub-block; Q8_0: none
        int isum = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            isum = ggml_cuda_dp4a(s.q[k], u[k], isum);
        }
        if constexpr (type == GGML_TYPE_Q8_0) {
            return isum;
        } else {
            return isum * s.sc[0];
        }
    }
}

// Adds a token's sub-block sum: the super-block types into their super-block sum sumf (finished by mmvqp_sub_finish),
// Q8_0 and Q5_0 into acc. ad, as: the token's activation scale and (Q5_0) offset term, as mmvqp_load_act makes them.
template <ggml_type type>
static __device__ __forceinline__ void mmvqp_sub_add(const mmvqp_sub & s, const int isum, const float ad, const float as,
                                                     float & sumf, float & acc) {
    if constexpr (type == GGML_TYPE_Q8_0) {
        acc = fmaf((float) isum, s.d * ad, acc);
    } else if constexpr (type == GGML_TYPE_Q5_0) {
        acc = fmaf(s.d, fmaf((float) isum, ad, as), acc);
    } else {
        sumf = fmaf(ad, (float) isum, sumf);
    }
    GGML_UNUSED(as);
}

// A super-block's sum times its scale, added to acc (Q8_0 and Q5_0: nothing to do).
template <ggml_type type>
static __device__ __forceinline__ float mmvqp_sub_finish(const uint32_t * w, const int b, const float sumf, const float acc) {
    if constexpr (type == GGML_TYPE_IQ2_S) {
        return fmaf(0.125f * mmvqp_h2f_lo(w[0]), sumf, acc);
    } else if constexpr (type == GGML_TYPE_IQ3_XXS) {
        return fmaf(0.25f * mmvqp_h2f_lo(w[0]), sumf, acc);
    } else if constexpr (type == GGML_TYPE_N2P) {
        return fmaf(mmvqp_h2f_lo(w[21]), sumf, acc);
    } else if constexpr (type == GGML_TYPE_N3P) {
        return fmaf(mmvqp_h2f_lo(w[b * mmvqp_traits<type>::bs / 4]), sumf, acc);
    } else {
        GGML_UNUSED_VARS(w, b, sumf);
        return acc;
    }
}

// The launch shape per type: rows per lane group (R), blocks per lane (C), staged or direct loads. A row's summation
// order depends only on C (blocks per lane) and the row width, which set the lanes per row; R and staging don't change
// any row's sum.
struct mmvqp_shape {
    int  R;
    int  C;
    bool staged;
};

// Measured on a P106-090 at the shapes of Flint (K = 512 to 4096) with bench/mmvq_pascal of
// github.com/Cinderella-Man/llms-on-crypto-mining-rigs: staged loads win for every type there; the direct-load shape is
// the fallback where a row segment can't be staged.
static constexpr mmvqp_shape mmvqp_choose(const ggml_type type, const bool gate) {
    switch (type) {
        case GGML_TYPE_Q4_1:
            return gate ? mmvqp_shape{ 2, 2, true } : mmvqp_shape{ 2, 4, true };
        case GGML_TYPE_Q4_0:
            return { 2, 2, true };
        case GGML_TYPE_Q5_0:
            return gate ? mmvqp_shape{ 1, 2, true } : mmvqp_shape{ 2, 2, true };
        case GGML_TYPE_Q8_0:
            return gate ? mmvqp_shape{ 1, 2, true } : mmvqp_shape{ 2, 2, true };
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
            return { 1, 1, true };
        case GGML_TYPE_IQ4_XS:
            return gate ? mmvqp_shape{ 1, 1, false } : mmvqp_shape{ 1, 1, true };
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_N2P:
        case GGML_TYPE_N3P:
            return { 1, 1, true };
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
            return { MMVQP_IQ_R, 1, true };
        default:
            GGML_ABORT("fatal error");
    }
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
