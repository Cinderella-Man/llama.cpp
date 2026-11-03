#pragma once

// The decoding of IQ2_XXS, IQ2_XS and IQ3_S in the Pascal matrix-vector kernels (mmvq-pascal.cu): one 256-weight
// super-block, read as 32-bit words, times its eight Q8_1 activation blocks. Kept free of ggml's headers so that
// bench/mmvq_pascal/verify/iq_unit_check.cu of github.com/Cinderella-Man/llms-on-crypto-mining-rigs can build it for the
// host (MMVQP_IQ_HD) and check it against ggml's own dequantization on a CPU. The kernels call these functions from
// mmvqp_dot (mmvq-pascal-impl.cuh), with the grid of the type in shared memory.
//
// As for IQ2_S and IQ3_XXS: the grid entries are unsigned bytes (IQ2_XXS and IQ2_XS 8, 25, 43; IQ3_S 1 to 15), a
// weight's sign is applied as sum(s * g * u) = sum(g * u) - 2 * sum((g & m) * u), with m the byte mask of the negative
// weights (two dp4a and one AND per 4 weights), and each 16- or 32-weight scale multiplies its integer sum once, in
// FP32. The byte masks are computed from the sign bits in registers, the faster of the two ways on a P106 for the older
// grid types (MMVQP_IQ_SIGN_LUT=0); these types keep no table of masks.
//
// Block layouts (ggml-common.h), byte offsets in the block:
//   IQ2_XXS, 66 bytes: d 0-1; then per 32 weights 8 bytes at 2 + 8 ib: four 8-bit grid indices (256 entries of 8
//            bytes), then a word of four 7-bit sign patterns (bits 0-27) and a 4-bit scale (bits 28-31).
//            Weights = d * (0.5 + scale) / 4 * grid * sign.
//   IQ2_XS,  74 bytes: d 0-1; 32 16-bit entries at 2-65, each a 9-bit grid index (512 entries of 8 bytes) and a 7-bit
//            sign pattern on top, four per 32 weights; scales 66-73, two 4-bit scales per 32 weights (one per 16).
//            Weights = d * (0.5 + scale) / 4 * grid * sign.
//   IQ3_S,  110 bytes: d 0-1; 64 low index bytes at 2-65, one per 4 weights; qh 66-73, per 32 weights a byte whose bit
//            g is the ninth index bit of its group g of 4 weights (512 grid entries of 4 bytes); signs 74-105, a byte
//            per 8 weights; scales 106-109, a 4-bit scale per 32 weights. Weights = d * (1 + 2 scale) * grid * sign.
// The 7-bit sign patterns of IQ2_XXS and IQ2_XS give the eighth sign as their parity (ggml's ksigns_iq2xs).

#include <cmath>
#include <cstdint>
#include <cstring>
#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif

#ifndef MMVQP_IQ_HD
#define MMVQP_IQ_HD __device__
#endif

namespace mmvqp_iq {

// bytes per block, and the 32-bit words of each grid in shared memory
constexpr int BS_IQ2_XXS = 66;
constexpr int BS_IQ2_XS  = 74;
constexpr int BS_IQ3_S   = 110;
constexpr int GRID_WORDS_IQ2_XXS = 2*256;
constexpr int GRID_WORDS_IQ2_XS  = 2*512;
constexpr int GRID_WORDS_IQ3_S   = 512;

static MMVQP_IQ_HD __forceinline__ uint32_t prmt(const uint32_t a, const uint32_t b, const uint32_t sel) {
#ifdef __CUDA_ARCH__
    return __byte_perm(a, b, sel);
#else
    const uint64_t x = (uint64_t) a | ((uint64_t) b << 32);
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) {
        r |= (uint32_t) ((x >> (8 * ((sel >> (4 * i)) & 7))) & 0xff) << (8 * i);
    }
    return r;
#endif
}

static MMVQP_IQ_HD __forceinline__ int dp4a(const int a, const int b, const int c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    return c + a8[0] * b8[0] + a8[1] * b8[1] + a8[2] * b8[2] + a8[3] * b8[3];
#endif
}

static MMVQP_IQ_HD __forceinline__ uint32_t popc(const uint32_t x) {
#ifdef __CUDA_ARCH__
    return __popc(x);
#else
    return (uint32_t) __builtin_popcount(x);
#endif
}

static MMVQP_IQ_HD __forceinline__ float h2f_lo(const uint32_t w) {
    const unsigned short h = (unsigned short) (w & 0xffff);
#ifdef __CUDA_ARCH__
    return __half2float(__ushort_as_half(h));
#else
    const int e = (h >> 10) & 0x1f, m = h & 0x3ff;
    const float v = e == 0 ? m * (1.0f / 16777216.0f) : e == 31 ? (m ? NAN : INFINITY) : ldexpf((float) (1024 + m), e - 25);
    return h & 0x8000 ? -v : v;
#endif
}

// The 32-bit word at an even byte offset of a block held as words (offsets are constants after unrolling).
static MMVQP_IQ_HD __forceinline__ uint32_t word_at(const uint32_t * w, const int off) {
    return (off & 3) == 0 ? w[off >> 2] : prmt(w[off >> 2], w[(off >> 2) + 1], 0x5432);
}

// 4 sign bits -> 4 byte masks (bit j -> 0xFF in byte j)
static MMVQP_IQ_HD __forceinline__ uint32_t sign_bytes(const uint32_t s4) {
    return ((s4 * 0x00204081u) & 0x01010101u) * 0xFFu;
}

// The masks of a group of 8 weights from 8 sign bits, or from 7 with the eighth their parity.
static MMVQP_IQ_HD __forceinline__ void masks8(const uint32_t s8, uint32_t & m0, uint32_t & m1) {
    m0 = sign_bytes(s8 & 0xF);
    m1 = sign_bytes(s8 >> 4);
}
static MMVQP_IQ_HD __forceinline__ void masks7(const uint32_t s7, uint32_t & m0, uint32_t & m1) {
    masks8(s7 | ((popc(s7) & 1) << 7), m0, m1);
}

// Entry i of a grid of 8-byte entries (IQ2_XXS, IQ2_XS) as two words: one 64-bit shared-memory load on the device.
static MMVQP_IQ_HD __forceinline__ void grid8(const uint32_t * grid, const uint32_t i, uint32_t & g0, uint32_t & g1) {
#ifdef __CUDA_ARCH__
    const uint2 v = ((const uint2 *) grid)[i];
    g0 = v.x;
    g1 = v.y;
#else
    g0 = grid[2*i];
    g1 = grid[2*i + 1];
#endif
}

// One group of 8 weights (grid words g0, g1, sign masks m0, m1) times activations u0, u1: p += sum(g * u),
// n += sum((g & m) * u).
static MMVQP_IQ_HD __forceinline__ void group8(const uint32_t g0, const uint32_t g1, const uint32_t m0, const uint32_t m1,
                                               const int u0, const int u1, int & p, int & n) {
    p = dp4a((int) g0, u0, p);
    p = dp4a((int) g1, u1, p);
    n = dp4a((int) (g0 & m0), u0, n);
    n = dp4a((int) (g1 & m1), u1, n);
}

// One super-block as words w (17 for IQ2_XXS, 19 for IQ2_XS, 28 for IQ3_S, starting at the block's first byte), its
// activation blocks' quants u[ib][0..7] and scales d8[ib], the type's grid as words; returns acc + the dot product.
static MMVQP_IQ_HD __forceinline__ float dot_iq2_xxs(const uint32_t * w, const int (*u)[8], const float * d8,
                                                     const uint32_t * grid, float acc) {
    float sumf = 0.0f;
#pragma unroll
    for (int ib = 0; ib < 8; ++ib) {
        const uint32_t q   = word_at(w, 2 + 8*ib);
        const uint32_t aux = word_at(w, 6 + 8*ib);
        int p = 0;
        int n = 0;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            uint32_t g0, g1, m0, m1;
            grid8(grid, (q >> (8*l)) & 0xff, g0, g1);
            masks7((aux >> (7*l)) & 0x7f, m0, m1);
            group8(g0, g1, m0, m1, u[ib][2*l + 0], u[ib][2*l + 1], p, n);
        }
        sumf = fmaf(d8[ib], (float) ((2*(int) (aux >> 28) + 1) * (p - 2*n)), sumf);
    }
    return fmaf(0.125f * h2f_lo(w[0]), sumf, acc);
}

static MMVQP_IQ_HD __forceinline__ float dot_iq2_xs(const uint32_t * w, const int (*u)[8], const float * d8,
                                                    const uint32_t * grid, float acc) {
    float sumf = 0.0f;
#pragma unroll
    for (int ib = 0; ib < 8; ++ib) {
        const uint32_t q0 = word_at(w, 2 + 8*ib); // entries 4 ib, 4 ib + 1
        const uint32_t q1 = word_at(w, 6 + 8*ib); // entries 4 ib + 2, 4 ib + 3
        const uint32_t sc = (word_at(w, 66 + 4*(ib/4)) >> (8*(ib%4))) & 0xff;
        int p[2] = {0, 0};
        int n[2] = {0, 0};
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t e  = ((l < 2 ? q0 : q1) >> (16*(l & 1))) & 0xffff;
            uint32_t g0, g1, m0, m1;
            grid8(grid, e & 511, g0, g1);
            masks7(e >> 9, m0, m1);
            group8(g0, g1, m0, m1, u[ib][2*l + 0], u[ib][2*l + 1], p[l/2], n[l/2]);
        }
        const int isum = (2*(int) (sc & 0xF) + 1) * (p[0] - 2*n[0]) + (2*(int) (sc >> 4) + 1) * (p[1] - 2*n[1]);
        sumf = fmaf(d8[ib], (float) isum, sumf);
    }
    return fmaf(0.125f * h2f_lo(w[0]), sumf, acc);
}

static MMVQP_IQ_HD __forceinline__ float dot_iq3_s(const uint32_t * w, const int (*u)[8], const float * d8,
                                                   const uint32_t * grid, float acc) {
    const uint32_t scales = word_at(w, 106); // nibble ib: the scale of weights 32 ib .. 32 ib + 31
    float sumf = 0.0f;
#pragma unroll
    for (int ib = 0; ib < 8; ++ib) {
        const uint32_t q0 = word_at(w, 2 + 8*ib); // low index bytes of groups 0..3 (two per 8 weights)
        const uint32_t q1 = word_at(w, 6 + 8*ib); // groups 4..7
        const uint32_t qh = (word_at(w, 66 + 4*(ib/4)) >> (8*(ib%4))) & 0xff;
        const uint32_t sg = word_at(w, 74 + 4*ib); // a sign byte per 8 weights
        int p = 0;
        int n = 0;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t q  = (l < 2 ? q0 : q1) >> (16*(l & 1));
            const uint32_t i0 = (q & 0xff)        | (((qh >> (2*l))     & 1) << 8);
            const uint32_t i1 = ((q >> 8) & 0xff) | (((qh >> (2*l + 1)) & 1) << 8);
            uint32_t m0, m1;
            masks8((sg >> (8*l)) & 0xff, m0, m1);
            group8(grid[i0], grid[i1], m0, m1, u[ib][2*l + 0], u[ib][2*l + 1], p, n);
        }
        sumf = fmaf(d8[ib], (float) ((2*(int) ((scales >> (4*ib)) & 0xF) + 1) * (p - 2*n)), sumf);
    }
    return fmaf(h2f_lo(w[0]), sumf, acc);
}

} // namespace mmvqp_iq
