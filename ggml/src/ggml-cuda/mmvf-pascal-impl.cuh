#pragma once

// The kernel of mmvf-pascal.cu without ggml's headers, so that a standalone benchmark can time the same code
// (bench/mmvq_pascal --bf16 in llms-on-crypto-mining-rigs carries a copy).

#include <cstdint>

#define MMVF_PASCAL_NT 128 // threads per block (4 warps)

// 8 BF16 weights (one 16-byte load) times 8 FP32 activations, added to acc one weight at a time in column order. A
// BF16 value is the upper half of an FP32 one, so moving it into place converts it exactly.
static __device__ __forceinline__ float mmvfp_dot8(const uint4 w, const float4 a, const float4 b, float acc) {
    acc = fmaf(__uint_as_float(w.x << 16),         a.x, acc);
    acc = fmaf(__uint_as_float(w.x & 0xFFFF0000u), a.y, acc);
    acc = fmaf(__uint_as_float(w.y << 16),         a.z, acc);
    acc = fmaf(__uint_as_float(w.y & 0xFFFF0000u), a.w, acc);
    acc = fmaf(__uint_as_float(w.z << 16),         b.x, acc);
    acc = fmaf(__uint_as_float(w.z & 0xFFFF0000u), b.y, acc);
    acc = fmaf(__uint_as_float(w.w << 16),         b.z, acc);
    acc = fmaf(__uint_as_float(w.w & 0xFFFF0000u), b.w, acc);
    return acc;
}

// G lanes per row group (8, 32, or 128, which spans the block's 4 warps), R rows per group, T tokens.
// x: nrows rows of nchunks 16-byte chunks (8 weights each), stride_row chunks apart. y: T columns of FP32 activations,
// stride_col_y4 float4 apart. dst, and bias if has_bias: T columns, stride_col_dst floats apart.
// Lane sl of a group takes chunks sl, sl + G, ... of each of its R rows, so every weight load of a warp is a coalesced
// 128-bit load; the activations of a chunk are loaded once for all R rows. Partial sums are reduced with shuffles in
// the warp, then (G = 128) over the warps through shared memory, always in the same order; after the shuffles every lane
// of a group holds all T * R sums, and lane lg writes those whose index t * R + i is lg modulo the group's lanes in the
// warp, so a lane writes several outputs where T * R exceeds them.
template <int G, int R, int T, bool has_bias>
__launch_bounds__(MMVF_PASCAL_NT)
static __global__ void mul_mat_vec_bf16_pascal(
        const uint4 * __restrict__ x, const float4 * __restrict__ y, const float * __restrict__ bias, float * __restrict__ dst,
        const int nrows, const int nchunks, const int stride_row, const int stride_col_y4, const int stride_col_dst) {
    constexpr int warp = 32;
    constexpr int NW   = MMVF_PASCAL_NT / warp;
    constexpr int GW   = G < warp ? G : warp;     // lanes of a group within one warp
    constexpr int WPG  = G > warp ? G / warp : 1; // warps per group
    static_assert(G == 8 || G == 32 || G == 128, "unsupported group size");

    const int tid  = threadIdx.x;
    const int sl   = tid % G;
    const int lg   = tid % GW;
    const int row0 = (blockIdx.x * (MMVF_PASCAL_NT / G) + tid / G) * R;

    const uint4 * xr[R];
#pragma unroll
    for (int i = 0; i < R; ++i) {
        xr[i] = x + (int64_t) min(row0 + i, nrows - 1) * stride_row + sl;
    }
    const float4 * yp = y + 2 * sl;

    float acc[T][R];
#pragma unroll
    for (int t = 0; t < T; ++t) {
#pragma unroll
        for (int i = 0; i < R; ++i) {
            acc[t][i] = 0.0f;
        }
    }

#pragma unroll 2
    for (int c = sl; c < nchunks; c += G) {
        uint4 w[R];
#pragma unroll
        for (int i = 0; i < R; ++i) {
            w[i] = __ldg(xr[i]);
            xr[i] += G;
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float4 a = __ldg(yp + t * stride_col_y4);
            const float4 b = __ldg(yp + t * stride_col_y4 + 1);
#pragma unroll
            for (int i = 0; i < R; ++i) {
                acc[t][i] = mmvfp_dot8(w[i], a, b, acc[t][i]);
            }
        }
        yp += 2 * G;
    }

#pragma unroll
    for (int t = 0; t < T; ++t) {
#pragma unroll
        for (int i = 0; i < R; ++i) {
#pragma unroll
            for (int off = GW / 2; off > 0; off >>= 1) {
                acc[t][i] += __shfl_xor_sync(0xFFFFFFFF, acc[t][i], off, warp);
            }
        }
    }

    if constexpr (WPG > 1) {
        __shared__ float buf[NW][T * R];
        const int wid = tid / warp;
#pragma unroll
        for (int t = 0; t < T; ++t) {
#pragma unroll
            for (int i = 0; i < R; ++i) {
                if ((t * R + i) % GW == lg) {
                    buf[wid][t * R + i] = acc[t][i];
                }
            }
        }
        __syncthreads();
        if (wid % WPG != 0) {
            return;
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
#pragma unroll
            for (int i = 0; i < R; ++i) {
                if ((t * R + i) % GW == lg) {
#pragma unroll
                    for (int k = 1; k < WPG; ++k) {
                        acc[t][i] += buf[wid + k][t * R + i];
                    }
                }
            }
        }
    }

#pragma unroll
    for (int t = 0; t < T; ++t) {
#pragma unroll
        for (int i = 0; i < R; ++i) {
            const int row = row0 + i;
            if ((t * R + i) % GW != lg || row >= nrows) {
                continue;
            }
            float v = acc[t][i];
            if constexpr (has_bias) {
                v += bias[t * stride_col_dst + row];
            }
            dst[t * stride_col_dst + row] = v;
        }
    }
}
