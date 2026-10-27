#include "mmvq-pascal.cuh"
#include "mmvq-pascal-impl.cuh"
#include "unary.cuh"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

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
//   - The activation block sums that the offsets and mins need (Q4_0, Q4_1, Q5_0, Q4_K, Q5_K) are computed in FP32 from
//     the int8 values once per activation load and shared by all rows; block_q8_1's FP16 sum is never read, because it
//     overflows on large activations.
//   - Q4_K, Q5_K and IQ4_XS take one 256-weight super-block per lane. IQ4_XS looks its 4-bit indices up with __byte_perm
//     in a table held in registers, with the activations pre-permuted into the order the lookup produces.
//   - Q4_K is Q5_K without the fifth bit: sub-blocks 2j and 2j + 1 take the low and high nibbles of the same 8 code
//     words. The high nibbles stay in place (q & 0xF0F0F0F0, unsigned-by-signed dp4a), and the sub-block's sum, which is
//     then exactly 16 times too large, is shifted right by 4 once, instead of shifting every word.
//   - Q6_K takes one super-block per lane too, staged only: its 210-byte super-blocks are 2-byte aligned, so lanes whose
//     block starts at an odd half-word shift the words they read from shared memory by 2 bytes, rows that are only 8- or
//     4-byte aligned (K = 1024, 512) are staged with 8- or 4-byte loads, and a fused gate/up stages its two matrices one
//     after the other in the same buffer (27 KB per block instead of 54 KB). The 6-bit codes are nibbles of ql OR-ed with
//     a 2-bit field of qh, one dp4a per 4 weights, and the offset -32 comes from the activations' half-block sums,
//     shared by all rows.
//   - Q5_0's fifth bits stay where they are: a block's sum splits into sum(nibble * a) + 16 * sum(bit * a), and
//     (qh >> j) & 0x01010101 holds the fifth bits of weights j, 8 + j, 16 + j and 24 + j, so the activations are
//     regrouped the same way once per block (two 4 x 4 byte transposes), shared by all rows.
//   - Q2_0 (64 weights, an FP16 scale and 16 bytes of 2-bit codes, as Q4_0's 18 bytes) takes two blocks per lane, so
//     that chunks stay 4-byte aligned. Its codes are looked up with __byte_perm in a 4-byte table (0x020100FF: codes 0..3
//     -> -1..2) with the code word itself as the selector, 4 lookups and 3 shifts per 16 weights; the lookups return
//     weights (0, 2, 4, 6) and (1, 3, 5, 7) of each run of 8, so the activations are split into even and odd bytes once
//     per load, shared by all rows. No masks, offsets or activation sums.
//   - Tile staging (Q2_0 without a fused gate, where a lane group's segment isn't a multiple of 16 bytes, as for
//     Qwen3.8-Flash-Next's 640-wide down projection: 180-byte rows, one lane per row, 36 bytes per iteration): each
//     warp copies all its rows, which are contiguous, into dynamic shared memory once with coalesced 128-bit loads,
//     then every lane reads its chunks from there, instead of each lane loading its own row 4 bytes at a time.
//   - The grid types (IQ2_S, IQ3_XXS, IQ2_XXS, IQ2_XS, IQ3_S) take one super-block per lane, staged as Q6_K. Each block
//     copies its type's grid into shared memory first; a weight's sign is applied as sum(g * u) - 2 * sum((g & m) * u)
//     with m the byte mask of the negative weights, and every scale multiplies its integer sum in FP32. The decoding of
//     IQ2_XXS, IQ2_XS and IQ3_S is in mmvq-pascal-iq.cuh, free of ggml's headers so that it can be checked on a CPU.
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
//     next to the weights: all types but Q4_K, Q5_K and the staged IQ4_XS shape), so that they are read with coalesced
//     loads instead of one strided load per lane and token. Off by default until it is measured on a P106.

#define MMVQ_PASCAL_NWARPS 4
#define MMVQP_MAX_COLS 4
#define MMVQP_TILE_MAX_BYTES (48 * 1024) // a block's shared memory without opting in to more

namespace {

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
    int             nrowblocks; // row blocks (MMVQP_IQ_PERSIST: fewer blocks than row blocks, each looping over them)
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

// Types whose blocks start at 2-byte boundaries (Q6_K 210 bytes, IQ2_S 82, IQ3_XXS 98, IQ2_XXS 66, IQ2_XS 74, IQ3_S
// 110, N2P 86): staged only, one super-block per lane, read from shared memory with mmvqp_lds_h.
template <ggml_type type> constexpr __host__ __device__ bool mmvqp_halfword() {
    return type == GGML_TYPE_Q6_K || mmvqp_grid_type<type>() || type == GGML_TYPE_N2P;
}
static bool mmvqp_grid_type_of(const ggml_type type) {
    return type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS ||
        type == GGML_TYPE_IQ3_S;
}
static bool mmvqp_halfword_type(const ggml_type type) {
    return type == GGML_TYPE_Q6_K || mmvqp_grid_type_of(type) || type == GGML_TYPE_N2P;
}

// Staged weights: matrices staged at a time (the half-word types' gate and up one after the other: half the staging
// memory and registers), and bytes of padding behind each warp's buffer (mmvqp_lds_h reads 2 bytes past a block).
template <ggml_type type, int M> constexpr __host__ __device__ int mmvqp_staged_mats() { return mmvqp_halfword<type>() ? 1 : M; }
template <ggml_type type> constexpr __host__ __device__ int mmvqp_staged_pad() { return mmvqp_halfword<type>() ? 16 : 0; }

template <ggml_type type, int R, int C, bool staged, bool gate>
constexpr bool mmvqp_fits() {
    return !staged || MMVQ_PASCAL_NWARPS * (WARP_SIZE * R * mmvqp_staged_mats<type, gate ? 2 : 1>() * C * mmvqp_traits<type>::bs +
        mmvqp_staged_pad<type>()) + 4 * mmvqp_tab_words<type>() <= 48 * 1024;
}

// the multi-token kernel with each token's activations staged as well (sa)
template <ggml_type type, int R, int C, bool staged, bool gate, bool sa>
constexpr bool mmvqp_fits_mc() {
    constexpr int wb = staged ? WARP_SIZE * R * mmvqp_staged_mats<type, gate ? 2 : 1>() * C * mmvqp_traits<type>::bs +
        mmvqp_staged_pad<type>() : 0;
    constexpr int yb = sa ? WARP_SIZE * C * mmvqp_traits<type>::na * (int) sizeof(block_q8_1) : 0;
    return MMVQ_PASCAL_NWARPS * (wb + yb) + 4 * mmvqp_tab_words<type>() <= 48 * 1024;
}

// The grid types' tables in shared memory, filled by the whole block before its first product.
template <ggml_type type>
static __device__ __forceinline__ void mmvqp_tabs_ready(uint32_t * tab) {
    if constexpr (mmvqp_tab_words<type>() > 0) {
        mmvqp_init_tabs<type>(tab);
        __syncthreads();
    }
    GGML_UNUSED(tab);
}

// Loads that stage weight segments: 16 bytes, or 8 and 4 where a half-word type's rows are only 8- or 4-byte aligned.
template <int SU> struct mmvqp_stage_unit;
template <> struct mmvqp_stage_unit<16> { using t = uint4; };
template <> struct mmvqp_stage_unit<8>  { using t = uint2; };
template <> struct mmvqp_stage_unit<4>  { using t = unsigned int; };

// One lane's super-block of NB bytes (NB % 4 == 2) at a 2-byte boundary in shared memory as NB/4 + 1 4-byte-aligned
// words: the words around it, shifted by 2 bytes where it starts at an odd half-word.
template <int NB>
static __device__ __forceinline__ void mmvqp_lds_h(uint32_t * w, const char * p) {
    static_assert(NB % 4 == 2, "a block of an odd number of half-words");
    const int        mis = (int) ((uintptr_t) p & 2);
    const uint32_t * aw  = (const uint32_t *) (p - mis); // pointer arithmetic keeps the shared address space
    const uint32_t   sel = mis ? 0x5432 : 0x3210;
    uint32_t cur = aw[0];
#pragma unroll
    for (int i = 0; i < NB/4; ++i) {
        const uint32_t nxt = aw[i + 1];
        w[i] = __byte_perm(cur, nxt, sel);
        cur = nxt;
    }
    w[NB/4] = __byte_perm(cur, 0, sel);
}

// A lane's staged weight chunk of WB bytes from shared memory.
template <ggml_type type, int WB>
static __device__ __forceinline__ void mmvqp_lds_chunk(uint32_t * w, const char * p) {
    if constexpr (mmvqp_halfword<type>()) {
        static_assert(WB == mmvqp_traits<type>::bs, "half-word types: one super-block per lane");
        mmvqp_lds_h<WB>(w, p);
    } else {
        mmvqp_lds_words<WB, mmvqp_gcd(WB, 16)>(w, p);
    }
}

// Tile staging (mmvqp_shape::tile): rows [wrow0, wrow0 + (WARP_SIZE >> lpr_log2) * R) of matrix x, the warp's rows, which
// lie one after the other in memory, copied once into the warp's part of the dynamic shared memory with coalesced 128-bit
// loads, for rows whose per-iteration segments aren't 16-byte aligned (Q2_0 at K = 640: 180-byte rows). Returns the
// warp's copy; row r - wrow0 starts at (r - wrow0) * row_bytes in it, as in global memory. The host checked that the rows
// are contiguous, that a warp's tile is a whole number of 16-byte units and that each matrix ends at a 16-byte boundary.
template <int R>
static __device__ __forceinline__ const char * mmvqp_stage_tile(const char * x, const mmvqp_args & p, const uint32_t bx) {
    extern __shared__ uint4 mmvqp_tile_sm[];
    constexpr int UN = 4; // loads in flight per lane
    const int wrows  = (WARP_SIZE >> p.lpr_log2) * R;
    const int wrow0  = (int) (bx * MMVQ_PASCAL_NWARPS + threadIdx.y) * wrows;
    const int wbytes = wrows * p.row_bytes;
    uint4       * dsm = (uint4 *) ((char *) mmvqp_tile_sm + threadIdx.y * wbytes);
    const uint4 * src = (const uint4 *) (x + (int64_t) wrow0 * p.row_bytes);
    const int     nu  = (max(0, min(wrows, p.nrows - wrow0)) * p.row_bytes) >> 4;
    for (int u0 = threadIdx.x; u0 < nu; u0 += UN * WARP_SIZE) {
        uint4 v[UN];
#pragma unroll
        for (int j = 0; j < UN; ++j) {
            if (u0 + j * WARP_SIZE < nu) {
                v[j] = __ldg(src + u0 + j * WARP_SIZE);
            }
        }
#pragma unroll
        for (int j = 0; j < UN; ++j) {
            if (u0 + j * WARP_SIZE < nu) {
                dsm[u0 + j * WARP_SIZE] = v[j];
            }
        }
    }
    __syncwarp();
    return (const char *) dsm;
}

// One token: the rows of row block bx (blockIdx.x of mul_mat_vec_q_pascal). Shared by mul_mat_vec_q_pascal and the
// Q/K/V launch (mul_mat_vec_q_pascal_qkv), where each type's instantiation has its own staging memory.
template <ggml_type type, int R, int C, bool staged, int fk, int SU = 16, bool tile = false>
static __device__ __forceinline__ void mmvqp_rows_1t(const mmvqp_args & p, const ggml_cuda_mm_fusion_args_device & fusion,
                                                     const uint32_t bx) {
    using unit_t = typename mmvqp_stage_unit<SU>::t;
    constexpr int NW = MMVQ_PASCAL_NWARPS;
    constexpr int BS = mmvqp_traits<type>::bs;
    constexpr int WB = C * BS;                     // weight bytes per lane and iteration
    constexpr int NA = C * mmvqp_traits<type>::na; // activation blocks per lane and iteration
    constexpr int YB = NA * sizeof(block_q8_1);    // activation bytes per lane and iteration
    constexpr int M  = fk == MMVQP_GATE ? 2 : 1;   // matrices: up (or plain), gate
    constexpr int MS = mmvqp_staged_mats<type, M>(); // matrices staged at a time
    constexpr int NU = (WB + SU - 1) / SU;         // staged: units per lane and row, the last one predicated
    constexpr bool seg = staged && !tile;          // staged segment by segment (tile: whole rows, once)
    static_assert(!tile || (staged && M == 1 && SU == 16), "tile staging: one matrix, 16-byte units");
    constexpr int TW = mmvqp_tab_words<type>();    // grid types: table words
    __shared__ uint4 sm[seg ? NW : 1][seg ? (WARP_SIZE * R * MS * WB + mmvqp_staged_pad<type>()) / 16 : 1];
    __shared__ uint2 tab[TW > 0 ? TW / 2 : 1];
    if (bx == blockIdx.x) { // a block's first row block (MMVQP_IQ_PERSIST: its later ones keep the tables)
        mmvqp_tabs_ready<type>((uint32_t *) tab);
    }

    const int lane     = threadIdx.x;
    const int lpr_log2 = p.lpr_log2;
    const int lpr      = 1 << lpr_log2;
    const int sl       = lane & (lpr - 1);
    const int grp      = lane >> lpr_log2;
    const int row0     = (((bx * NW + threadIdx.y) << (5 - lpr_log2)) + grp) * R;

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
    [[maybe_unused]] char * smw = (char *) sm[seg ? threadIdx.y : 0] + (seg ? grp * R * MS * WB << lpr_log2 : 0);
    [[maybe_unused]] const int seg_units = (WB << lpr_log2) >> (SU == 16 ? 4 : SU == 8 ? 3 : 2);
    [[maybe_unused]] const char * smt = nullptr; // tile: this lane's chunk of the lane group's first row
    if constexpr (tile) {
        smt = mmvqp_stage_tile<R>(xm[0], p, bx) + grp * R * p.row_bytes + sl * WB;
    }

    float acc[M][R];
#pragma unroll
    for (int m = 0; m < M; ++m) {
#pragma unroll
        for (int i = 0; i < R; ++i) {
            acc[m][i] = 0.0f;
        }
    }

    for (int it = 0; it < p.niter; ++it) {
        mmvqp_act<NA> a;
#pragma unroll
        for (int g = 0; g < M / MS; ++g) {
            if constexpr (seg) {
#pragma unroll
                for (int ms = 0; ms < MS; ++ms) {
#pragma unroll
                    for (int i = 0; i < R; ++i) {
                        unit_t       * dsm = (unit_t *) (smw + ((ms*R + i) * WB << lpr_log2));
                        const unit_t * src = (const unit_t *) rw[g*MS + ms][i];
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
            if (g == 0) {
                mmvqp_load_act<type, NA>(a, yp);
            }
#pragma unroll
            for (int ms = 0; ms < MS; ++ms) {
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    const int m = g*MS + ms;
                    uint32_t w[(WB + 3) / 4];
                    if constexpr (tile) {
                        mmvqp_lds_chunk<type, WB>(w, smt + i * p.row_bytes);
                    } else if constexpr (staged) {
                        mmvqp_lds_chunk<type, WB>(w, smw + ((ms*R + i) * WB << lpr_log2) + sl * WB);
                    } else {
                        mmvqp_ldg_words<WB, mmvqp_gcd(WB, 16)>(w, rw[m][i]);
                    }
                    acc[m][i] = mmvqp_dot<type, C>(w, a, acc[m][i], (const uint32_t *) tab);
                    rw[m][i] += wstep;
                }
            }
            if constexpr (seg) {
                __syncwarp();
            }
        }
        if constexpr (tile) {
            smt += wstep;
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

template <ggml_type type, int R, int C, bool staged, int fk, int SU = 16, bool tile = false>
__launch_bounds__(MMVQ_PASCAL_NWARPS*WARP_SIZE, mmvqp_min_blocks<type>())
static __global__ void mul_mat_vec_q_pascal(const mmvqp_args p, const ggml_cuda_mm_fusion_args_device fusion) {
    if constexpr (mmvqp_grid_type<type>() && MMVQP_IQ_PERSIST > 0) {
        for (uint32_t bx = blockIdx.x; bx < (uint32_t) p.nrowblocks; bx += gridDim.x) {
            mmvqp_rows_1t<type, R, C, staged, fk, SU, tile>(p, fusion, bx);
        }
    } else {
        mmvqp_rows_1t<type, R, C, staged, fk, SU, tile>(p, fusion, blockIdx.x);
    }
}

// One token, three products of the same activations in one launch (an attention block's Q, K and V): row blocks
// [0, nbq) compute Q, [nbq, nbq + nbk) K, the rest V, each exactly as its own mul_mat_vec_q_pascal launch would (staged
// shapes only). K and V share a type.
template <ggml_type tq, int Rq, int Cq, ggml_type tkv, int Rkv, int Ckv>
__launch_bounds__(MMVQ_PASCAL_NWARPS*WARP_SIZE, 1)
static __global__ void mul_mat_vec_q_pascal_qkv(const mmvqp_args pq, const mmvqp_args pk, const mmvqp_args pv,
                                                const int nbq, const int nbk) {
    const ggml_cuda_mm_fusion_args_device fusion = {};
    const uint32_t bx = blockIdx.x;
    if (bx < (uint32_t) nbq) {
        mmvqp_rows_1t<tq, Rq, Cq, true, MMVQP_PLAIN>(pq, fusion, bx);
    } else if (bx < (uint32_t) (nbq + nbk)) {
        mmvqp_rows_1t<tkv, Rkv, Ckv, true, MMVQP_PLAIN>(pk, fusion, bx - nbq);
    } else {
        mmvqp_rows_1t<tkv, Rkv, Ckv, true, MMVQP_PLAIN>(pv, fusion, bx - nbq - nbk);
    }
}

// 2 to MMVQP_MAX_COLS tokens. Grid: (row blocks, channels, samples) for MUL_MAT; (row blocks, tokens * slots + shared
// expert, samples) for MUL_MAT_ID, where block y = t * nslots + k is entry (token t, slot k) of ids.
// sa: each token's activation segment is copied into shared memory by the whole warp with coalesced 128-bit loads, and
// every lane reads its blocks from there, instead of loading its own blocks from global memory at a stride of YB bytes.
template <ggml_type type, int R, int C, bool staged, int fk, bool sa, int SU = 16, bool tile = false>
__launch_bounds__(MMVQ_PASCAL_NWARPS*WARP_SIZE, mmvqp_min_blocks_mc<type>())
static __global__ void mul_mat_vec_q_pascal_mc(const mmvqp_args_mc p, const ggml_cuda_mm_fusion_args_device fusion) {
    using unit_t = typename mmvqp_stage_unit<SU>::t;
    constexpr int NW = MMVQ_PASCAL_NWARPS;
    constexpr int NC = MMVQP_MAX_COLS;
    constexpr int BS = mmvqp_traits<type>::bs;
    constexpr int WB = C * BS;                     // weight bytes per lane and iteration
    constexpr int NA = C * mmvqp_traits<type>::na; // activation blocks per lane and iteration
    constexpr int YB = NA * sizeof(block_q8_1);    // activation bytes per lane, iteration and token
    constexpr int M  = fk == MMVQP_GATE ? 2 : 1;   // matrices: up (or plain), gate
    constexpr int MS = mmvqp_staged_mats<type, M>(); // matrices staged at a time
    constexpr int NU = (WB + SU - 1) / SU;         // staged: units per lane and row, the last one predicated
    constexpr int NY = (WARP_SIZE * YB / 16 + WARP_SIZE - 1) / WARP_SIZE; // sa: 16-byte units per lane and token
    constexpr bool seg = staged && !tile;          // staged segment by segment (tile: whole rows, once)
    static_assert(!tile || (staged && M == 1 && SU == 16), "tile staging: one matrix, 16-byte units");
    constexpr int TW = mmvqp_tab_words<type>();    // grid types: table words
    __shared__ uint4 sm[seg ? NW : 1][seg ? (WARP_SIZE * R * MS * WB + mmvqp_staged_pad<type>()) / 16 : 1];
    __shared__ uint4 sm_y[sa ? NW : 1][sa ? WARP_SIZE * YB / 16 : 1];
    __shared__ uint2 tab[TW > 0 ? TW / 2 : 1];
    // before any block leaves early (a later entry of an expert another block serves)
    mmvqp_tabs_ready<type>((uint32_t *) tab);

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
    [[maybe_unused]] char * smw = (char *) sm[seg ? threadIdx.y : 0] + (seg ? grp * R * MS * WB << lpr_log2 : 0);
    [[maybe_unused]] const int seg_units = (WB << lpr_log2) >> (SU == 16 ? 4 : SU == 8 ? 3 : 2);
    [[maybe_unused]] const char * smt = nullptr; // tile: this lane's chunk of the lane group's first row
    if constexpr (tile) {
        smt = mmvqp_stage_tile<R>(xm[0], p, blockIdx.x) + grp * R * p.row_bytes + sl * WB;
    }

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
#pragma unroll
        for (int g = 0; g < M / MS; ++g) {
            if constexpr (seg) {
#pragma unroll
                for (int ms = 0; ms < MS; ++ms) {
#pragma unroll
                    for (int i = 0; i < R; ++i) {
                        unit_t       * dsm = (unit_t *) (smw + ((ms*R + i) * WB << lpr_log2));
                        const unit_t * src = (const unit_t *) rw[g*MS + ms][i];
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
            // sa: every row's chunk goes into registers before the first token, because the warp barriers between
            // tokens keep the compiler from reusing the chunks it loaded for the previous token (without sa it reuses
            // them)
            [[maybe_unused]] uint32_t wr[sa ? MS : 1][sa ? R : 1][(WB + 3) / 4];
            if constexpr (sa) {
#pragma unroll
                for (int ms = 0; ms < MS; ++ms) {
#pragma unroll
                    for (int i = 0; i < R; ++i) {
                        if constexpr (tile) {
                            mmvqp_lds_chunk<type, WB>(wr[ms][i], smt + i * p.row_bytes);
                        } else if constexpr (staged) {
                            mmvqp_lds_chunk<type, WB>(wr[ms][i], smw + ((ms*R + i) * WB << lpr_log2) + sl * WB);
                        } else {
                            mmvqp_ldg_words<WB, mmvqp_gcd(WB, 16)>(wr[ms][i], rw[g*MS + ms][i]);
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
                    mmvqp_load_act<type, NA, MMVQP_SRC_SMEM>(a, (const char *) sm_y[threadIdx.y] + sl * YB);
                } else {
                    mmvqp_load_act<type, NA>(a, yp + yc[j]);
                }
#pragma unroll
                for (int ms = 0; ms < MS; ++ms) {
#pragma unroll
                    for (int i = 0; i < R; ++i) {
                        const int m = g*MS + ms;
                        if constexpr (sa) {
                            acc[j][m][i] = mmvqp_dot<type, C>(wr[ms][i], a, acc[j][m][i], (const uint32_t *) tab);
                        } else {
                            uint32_t w[(WB + 3) / 4];
                            if constexpr (tile) {
                                mmvqp_lds_chunk<type, WB>(w, smt + i * p.row_bytes);
                            } else if constexpr (staged) {
                                mmvqp_lds_chunk<type, WB>(w, smw + ((ms*R + i) * WB << lpr_log2) + sl * WB);
                            } else {
                                mmvqp_ldg_words<WB, mmvqp_gcd(WB, 16)>(w, rw[m][i]);
                            }
                            acc[j][m][i] = mmvqp_dot<type, C>(w, a, acc[j][m][i], (const uint32_t *) tab);
                        }
                    }
                }
            }
            if constexpr (seg) {
                __syncwarp();
            }
        }
#pragma unroll
        for (int m = 0; m < M; ++m) {
#pragma unroll
            for (int i = 0; i < R; ++i) {
                rw[m][i] += wstep;
            }
        }
        if constexpr (tile) {
            smt += wstep;
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

// The single-token kernel for one token, the multi-token one for more. tile: smem bytes of dynamic shared memory hold the
// block's warp tiles (mmvqp_stage_tile).
template <ggml_type type, int R, int C, bool staged, int fk, int SU = 16, bool tile = false>
static void mmvqp_launch_kernel(const mmvqp_args_mc & p, const ggml_cuda_mm_fusion_args_device & fusion, const dim3 grid,
                                cudaStream_t stream, const int smem = 0) {
    const dim3 block(WARP_SIZE, MMVQ_PASCAL_NWARPS, 1);
    if constexpr (mmvqp_fits<type, R, C, staged && !tile, fk == MMVQP_GATE>()) {
        if (p.ncols == 1) {
            mul_mat_vec_q_pascal<type, R, C, staged, fk, SU, tile><<<grid, block, smem, stream>>>(p, fusion);
            return;
        }
        if constexpr (mmvqp_fits_mc<type, R, C, staged && !tile, fk == MMVQP_GATE, true>()) {
            if (p.stage_act) {
                mul_mat_vec_q_pascal_mc<type, R, C, staged, fk, true, SU, tile><<<grid, block, smem, stream>>>(p, fusion);
                return;
            }
        }
        mul_mat_vec_q_pascal_mc<type, R, C, staged, fk, false, SU, tile><<<grid, block, smem, stream>>>(p, fusion);
    } else {
        GGML_ABORT("fatal error");
    }
}

template <ggml_type type, int R, int C, bool staged, int SU = 16>
static void mmvqp_launch(const mmvqp_args_mc & p, const ggml_cuda_mm_fusion_args_device & fusion, const int fk,
                         const dim3 grid, cudaStream_t stream) {
    switch (fk) {
        case MMVQP_PLAIN:
            mmvqp_launch_kernel<type, R, C, staged, MMVQP_PLAIN, SU>(p, fusion, grid, stream);
            break;
        case MMVQP_BIAS:
            mmvqp_launch_kernel<type, R, C, staged, MMVQP_BIAS, SU>(p, fusion, grid, stream);
            break;
        default:
            mmvqp_launch_kernel<type, R, C, staged, MMVQP_GATE, SU>(p, fusion, grid, stream);
            break;
    }
}

// Tile-staged rows (no fused gate: a tile holds one matrix).
template <ggml_type type, int R, int C>
static void mmvqp_launch_tile(const mmvqp_args_mc & p, const ggml_cuda_mm_fusion_args_device & fusion, const int fk,
                              const dim3 grid, cudaStream_t stream, const int smem) {
    switch (fk) {
        case MMVQP_PLAIN:
            mmvqp_launch_kernel<type, R, C, true, MMVQP_PLAIN, 16, true>(p, fusion, grid, stream, smem);
            break;
        case MMVQP_BIAS:
            mmvqp_launch_kernel<type, R, C, true, MMVQP_BIAS, 16, true>(p, fusion, grid, stream, smem);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

// su: bytes per staging load (16; the half-word types also 8 and 4); smem: dynamic shared memory of a tile-staged shape
template <ggml_type type>
static void mmvqp_launch_shape(const mmvqp_shape s, const mmvqp_args_mc & p, const ggml_cuda_mm_fusion_args_device & fusion,
                               const int fk, const dim3 grid, cudaStream_t stream, const int su, const int smem) {
#define MMVQP_CASE(R_, C_, S_)                                              \
    if (!s.tile && s.R == R_ && s.C == C_ && s.staged == S_) {              \
        mmvqp_launch<type, R_, C_, S_>(p, fusion, fk, grid, stream);        \
        return;                                                             \
    }
#define MMVQP_TILE_CASE(R_, C_)                                             \
    if (s.tile && s.R == R_ && s.C == C_) {                                 \
        mmvqp_launch_tile<type, R_, C_>(p, fusion, fk, grid, stream, smem); \
        return;                                                             \
    }
    if constexpr (type == GGML_TYPE_Q2_0) {
        MMVQP_TILE_CASE(1, 2)
        MMVQP_TILE_CASE(2, 2)
        MMVQP_CASE(1, 2, true)
        MMVQP_CASE(1, 2, false)
        MMVQP_CASE(2, 2, true)
        MMVQP_CASE(2, 2, false)
    } else if constexpr (mmvqp_halfword<type>()) {
        if (s.R == 1 && s.C == 1 && s.staged) {
            switch (su) {
                case 16: mmvqp_launch<type, 1, 1, true, 16>(p, fusion, fk, grid, stream); return;
                case 8:  mmvqp_launch<type, 1, 1, true, 8>(p, fusion, fk, grid, stream);  return;
                default: mmvqp_launch<type, 1, 1, true, 4>(p, fusion, fk, grid, stream);  return;
            }
        }
        if constexpr (mmvqp_grid_type<type>() && MMVQP_IQ_R == 2) {
            if (s.R == 2 && s.C == 1 && s.staged) {
                switch (su) {
                    case 16: mmvqp_launch<type, 2, 1, true, 16>(p, fusion, fk, grid, stream); return;
                    case 8:  mmvqp_launch<type, 2, 1, true, 8>(p, fusion, fk, grid, stream);  return;
                    default: mmvqp_launch<type, 2, 1, true, 4>(p, fusion, fk, grid, stream);  return;
                }
            }
        }
    } else if constexpr (type == GGML_TYPE_Q4_1) {
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
    } else if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_N3P) {
        MMVQP_CASE(1, 1, true)
        MMVQP_CASE(1, 1, false)
    } else {
        MMVQP_CASE(1, 2, true)
        MMVQP_CASE(1, 2, false)
        MMVQP_CASE(2, 2, true)
        MMVQP_CASE(2, 2, false)
    }
#undef MMVQP_CASE
#undef MMVQP_TILE_CASE
    GGML_UNUSED_VARS(su, smem);
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

// GGML_CUDA_MMVQ_PASCAL_Q2_0=G,P,T: Q2_0's launch shape, so that one binary measures the alternatives on a P106: G and P
// rows per lane group (1 or 2) with and without a fused gate, and T where rows go whose segments can't be staged 16 bytes
// at a time (K = 640): 0 direct loads, 1 tile staging (default), 2 tile staging wherever it fits. 0 or a missing field
// keeps the default (mmvqp_choose's value; with a fused gate, R = 1 for one token where R = 2 leaves fewer than 4 blocks
// per SM).
struct mmvqp_q2_0_cfg {
    int r_gate  = 0;
    int r_plain = 0;
    int tile    = 1;
};

static const mmvqp_q2_0_cfg & mmvqp_q2_0_config() {
    static const mmvqp_q2_0_cfg cfg = [] {
        mmvqp_q2_0_cfg c;
        const char * env = getenv("GGML_CUDA_MMVQ_PASCAL_Q2_0");
        int v[3] = { 0, 0, 1 };
        const int n = env ? sscanf(env, "%d,%d,%d", &v[0], &v[1], &v[2]) : 0;
        c.r_gate  = n >= 1 && (v[0] == 1 || v[0] == 2) ? v[0] : 0;
        c.r_plain = n >= 2 && (v[1] == 1 || v[1] == 2) ? v[1] : 0;
        c.tile    = n >= 3 && v[2] >= 0 && v[2] <= 2 ? v[2] : 1;
        return c;
    }();
    return cfg;
}

// GGML_CUDA_MMVQ_PASCAL_NSM=n: the SM count Q2_0's gate/up launch shape is chosen for (0: the device's), e.g. 5 to run
// the P106-090's shapes on another GPU
static int mmvqp_nsm() {
    static const int nsm = [] {
        const char * env = getenv("GGML_CUDA_MMVQ_PASCAL_NSM");
        return env ? atoi(env) : 0;
    }();
    return nsm > 0 ? nsm : ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
}

// GGML_CUDA_MMVQ_PASCAL_SKIP=q4_K,q6_K: weight types (ggml type names, comma-separated) left to mul_mat_vec_q, so that
// one binary measures each type's kernel on and off while the other types keep theirs
static bool mmvqp_skip_type(const ggml_type type) {
    static const std::vector<ggml_type> skip = [] {
        std::vector<ggml_type> types;
        const char * env = getenv("GGML_CUDA_MMVQ_PASCAL_SKIP");
        const std::string list = env ? env : "";
        size_t a = 0;
        while (a < list.size()) {
            size_t b = list.find(',', a);
            if (b == std::string::npos) {
                b = list.size();
            }
            const std::string name = list.substr(a, b - a);
            for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
                const char * tn = ggml_type_name((ggml_type) t);
                if (tn && name == tn) {
                    types.push_back((ggml_type) t);
                }
            }
            a = b + 1;
        }
        return types;
    }();
    return std::find(skip.begin(), skip.end(), type) != skip.end();
}

static bool mmvqp_aligned(const void * ptr, const int64_t al) {
    return ptr == nullptr || (uintptr_t) ptr % al == 0;
}

} // namespace

// 1 (default): compute capability 6.1 only; 2: any NVIDIA GPU (testing)
bool ggml_cuda_mmvq_pascal_active(const int cc) {
    const int mode = mmvqp_mode();
    return GGML_CUDA_CC_IS_NVIDIA(cc) && (mode == 2 || (mode == 1 && cc == GGML_CUDA_CC_DP4A));
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
            type != GGML_TYPE_Q4_K && type != GGML_TYPE_Q5_K && type != GGML_TYPE_IQ4_XS && type != GGML_TYPE_Q6_K &&
            type != GGML_TYPE_Q2_0 && !mmvqp_grid_type_of(type) && type != GGML_TYPE_N2P && type != GGML_TYPE_N3P) {
        return false;
    }
    if (mmvqp_skip_type(type)) {
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
    if (type == GGML_TYPE_Q2_0) {
        const int r = gate ? mmvqp_q2_0_config().r_gate : mmvqp_q2_0_config().r_plain;
        s.R = r > 0 ? r : s.R;
    }

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
    // Q2_0 with a fused gate, one token: R = 1 where R = 2 leaves fewer than 4 blocks per SM (Q612; the shared expert's
    // 2560 -> 640 gate/up in its own launch, 10 blocks on a P106: 7.78 against 10.98 us in R16)
    if (type == GGML_TYPE_Q2_0 && gate && ncols_dst == 1 && s.R == 2 && mmvqp_q2_0_config().r_gate == 0) {
        const int64_t rows_per_block = MMVQ_PASCAL_NWARPS * (WARP_SIZE >> lpr_log2) * 2;
        const int64_t nblocks = (nrows_x + rows_per_block - 1) / rows_per_block * (nchannels_dst + (fusion.shared_up ? 1 : 0)) *
            nsamples_dst;
        if (nblocks < 4 * mmvqp_nsm()) {
            s.R = 1;
        }
    }

    // every chunk load must be aligned as the kernel assumes: staged segments to 16 bytes, direct chunks to their size
    const int64_t row_bytes = stride_row_x * bs;
    const auto x_aligned = [&](const int64_t al) {
        return mmvqp_aligned(vx, al) && mmvqp_aligned(fusion.gate, al) && mmvqp_aligned(fusion.shared_up, al) &&
            mmvqp_aligned(fusion.shared_gate, al) && row_bytes % al == 0 && (stride_channel_x * bs) % al == 0 &&
            (stride_sample_x * bs) % al == 0;
    };
    int su   = 16;
    int smem = 0; // tile staging: the block's warp tiles
    if (mmvqp_halfword_type(type)) {
        // staged only (2-byte-aligned blocks), with the widest loads the rows and the segment allow
        su = 0;
        for (const int u : { 16, 8, 4 }) {
            if (((s.C * bs) << lpr_log2) % u == 0 && x_aligned(u)) {
                su = u;
                break;
            }
        }
        if (su == 0 || row_bytes > INT_MAX) {
            return false;
        }
    } else {
        const bool seg_ok = ((s.C * bs) << lpr_log2) % 16 == 0 && x_aligned(16);
        // tile staging (shapes that allow it, no fused gate): the warp's rows must be contiguous, whole 16-byte units and
        // fit, and every matrix must end at a 16-byte boundary so that the last tile reads nothing past it
        const int tile_mode = type == GGML_TYPE_Q2_0 ? mmvqp_q2_0_config().tile : 0;
        if (s.staged && s.tile && !gate && (tile_mode == 2 || (tile_mode == 1 && !seg_ok))) {
            const int64_t wbytes = (int64_t) (WARP_SIZE >> lpr_log2) * s.R * row_bytes;
            if (stride_row_x == blocks && wbytes % 16 == 0 && (nrows_x * row_bytes) % 16 == 0 && mmvqp_aligned(vx, 16) &&
                    (stride_channel_x * bs) % 16 == 0 && (stride_sample_x * bs) % 16 == 0 &&
                    MMVQ_PASCAL_NWARPS * wbytes <= MMVQP_TILE_MAX_BYTES) {
                smem = (int) (MMVQ_PASCAL_NWARPS * wbytes);
            }
        }
        s.tile = smem > 0;
        if (s.staged && !s.tile && !seg_ok) {
            s.staged = false;
        }
        if (row_bytes > INT_MAX || (!s.staged && !x_aligned(mmvqp_gcd((int) (s.C * bs), 16)))) {
            return false;
        }
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
    // the staged activations' static shared memory comes on top of a tile's
    if (s.tile && p.stage_act && smem + MMVQ_PASCAL_NWARPS * WARP_SIZE * (yb_seg >> lpr_log2) > MMVQP_TILE_MAX_BYTES) {
        p.stage_act = false;
    }

    // several tokens with ids: one block per (token, slot) entry
    const int64_t nblocks_y = ids && ncols_dst > 1 ? nchannels_dst * ncols_dst : nchannels_dst;

    const int  fk             = gate ? MMVQP_GATE : (fusion.x_bias ? MMVQP_BIAS : MMVQP_PLAIN);
    const int  rows_per_block = MMVQ_PASCAL_NWARPS * (WARP_SIZE >> lpr_log2) * s.R;
    dim3 grid((unsigned) ((nrows_x + rows_per_block - 1) / rows_per_block),
              (unsigned) (nblocks_y + (fusion.shared_up ? 1 : 0)), (unsigned) nsamples_dst);
    p.nrowblocks = (int) grid.x;
    if (MMVQP_IQ_PERSIST > 0 && ncols_dst == 1 && mmvqp_grid_type_of(type)) {
        // MMVQP_IQ_PERSIST blocks per SM in all, over the experts (grid.y) and samples (grid.z)
        const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
        const int per = (int) (grid.y * grid.z);
        grid.x = std::min(grid.x, (unsigned) std::max(1, (MMVQP_IQ_PERSIST * nsm + per - 1) / per));
    }

    switch (type) {
        case GGML_TYPE_Q4_0: mmvqp_launch_shape<GGML_TYPE_Q4_0>(s, p, fusion, fk, grid, stream, su, smem); break;
        case GGML_TYPE_Q4_1: mmvqp_launch_shape<GGML_TYPE_Q4_1>(s, p, fusion, fk, grid, stream, su, smem); break;
        case GGML_TYPE_Q5_0: mmvqp_launch_shape<GGML_TYPE_Q5_0>(s, p, fusion, fk, grid, stream, su, smem); break;
        case GGML_TYPE_Q4_K: mmvqp_launch_shape<GGML_TYPE_Q4_K>(s, p, fusion, fk, grid, stream, su, smem); break;
        case GGML_TYPE_Q5_K: mmvqp_launch_shape<GGML_TYPE_Q5_K>(s, p, fusion, fk, grid, stream, su, smem); break;
        case GGML_TYPE_IQ4_XS: mmvqp_launch_shape<GGML_TYPE_IQ4_XS>(s, p, fusion, fk, grid, stream, su, smem); break;
        case GGML_TYPE_Q8_0: mmvqp_launch_shape<GGML_TYPE_Q8_0>(s, p, fusion, fk, grid, stream, su, smem); break;
        case GGML_TYPE_Q6_K: mmvqp_launch_shape<GGML_TYPE_Q6_K>(s, p, fusion, fk, grid, stream, su, smem); break;
        case GGML_TYPE_Q2_0: mmvqp_launch_shape<GGML_TYPE_Q2_0>(s, p, fusion, fk, grid, stream, su, smem); break;
        case GGML_TYPE_IQ2_S: mmvqp_launch_shape<GGML_TYPE_IQ2_S>(s, p, fusion, fk, grid, stream, su, smem); break;
        case GGML_TYPE_IQ3_XXS: mmvqp_launch_shape<GGML_TYPE_IQ3_XXS>(s, p, fusion, fk, grid, stream, su, smem); break;
        case GGML_TYPE_IQ2_XXS: mmvqp_launch_shape<GGML_TYPE_IQ2_XXS>(s, p, fusion, fk, grid, stream, su, smem); break;
        case GGML_TYPE_IQ2_XS:  mmvqp_launch_shape<GGML_TYPE_IQ2_XS>(s, p, fusion, fk, grid, stream, su, smem);  break;
        case GGML_TYPE_IQ3_S:   mmvqp_launch_shape<GGML_TYPE_IQ3_S>(s, p, fusion, fk, grid, stream, su, smem);   break;
        case GGML_TYPE_N2P:  mmvqp_launch_shape<GGML_TYPE_N2P>(s, p, fusion, fk, grid, stream, su, smem); break;
        case GGML_TYPE_N3P:  mmvqp_launch_shape<GGML_TYPE_N3P>(s, p, fusion, fk, grid, stream, su, smem); break;
        default:             GGML_ABORT("fatal error");
    }
    return true;
}

// One token, a plain product of a contiguous matrix: the arguments mul_mat_vec_q_pascal runs with in
// ggml_cuda_mmvq_pascal, if that is a staged shape (as in that function, with no channels, samples or fusion)
static bool mmvqp_prepare_plain_staged(const void * vx, const ggml_type type, const void * vy, float * dst,
                                       const int64_t ncols_x, const int64_t nrows_x, mmvqp_args & p, int & nblocks) {
    if (type != GGML_TYPE_Q4_0 && type != GGML_TYPE_Q4_1 && type != GGML_TYPE_Q5_0 && type != GGML_TYPE_Q8_0 &&
            type != GGML_TYPE_Q5_K && type != GGML_TYPE_IQ4_XS) {
        return false;
    }
    if (mmvqp_skip_type(type)) {
        return false;
    }
    const mmvqp_shape s = mmvqp_choose(type, false);

    const int64_t bs     = ggml_type_size(type);
    const int64_t blocks = ncols_x / ggml_blck_size(type);
    if (!s.staged || ncols_x % ggml_blck_size(type) != 0 || blocks % s.C != 0) {
        return false;
    }
    const int64_t chunks = blocks / s.C;
    int lpr_log2 = 0;
    while (lpr_log2 < 5 && chunks % (2 << lpr_log2) == 0) {
        lpr_log2++;
    }
    const int64_t row_bytes = blocks * bs;
    if (((s.C * bs) << lpr_log2) % 16 != 0 || !mmvqp_aligned(vx, 16) || row_bytes % 16 != 0 || row_bytes > INT_MAX) {
        return false;
    }
    const int64_t aly = mmvqp_gcd((int) (s.C * (ggml_blck_size(type) / QK8_1) * sizeof(block_q8_1)), 16);
    if (!mmvqp_aligned(vy, aly)) {
        return false;
    }

    p                    = {};
    p.vx                 = (const char *) vx;
    p.vy                 = (const char *) vy;
    p.ids                = nullptr;
    p.dst                = dst;
    p.nrows              = (int) nrows_x;
    p.row_bytes          = (int) row_bytes;
    p.lpr_log2           = lpr_log2;
    p.niter              = (int) (chunks >> lpr_log2);
    p.nchannels_y        = make_uint3(0, 0, 0);
    p.channel_ratio      = init_fastdiv_values(1);
    p.sample_ratio       = init_fastdiv_values(1);
    p.channel_bytes_x    = row_bytes * nrows_x;
    p.sample_bytes_x     = row_bytes * nrows_x;

    const int64_t rows_per_block = MMVQ_PASCAL_NWARPS * (WARP_SIZE >> lpr_log2) * s.R;
    nblocks = (int) ((nrows_x + rows_per_block - 1) / rows_per_block);
    return true;
}

bool ggml_cuda_mmvq_pascal_qkv(const ggml_cuda_mmvq_pascal_qkv_args & a, cudaStream_t stream) {
    if (!ggml_cuda_mmvq_pascal_active(ggml_cuda_info().devices[ggml_cuda_get_device()].cc) || a.type[1] != a.type[2]) {
        return false;
    }
    mmvqp_args p[3];
    int        nb[3];
    for (int j = 0; j < 3; ++j) {
        if (!mmvqp_prepare_plain_staged(a.vx[j], a.type[j], a.vy, a.dst[j], a.ncols, a.nrows[j], p[j], nb[j])) {
            return false;
        }
    }
    const dim3 grid(nb[0] + nb[1] + nb[2], 1, 1);
    const dim3 block(WARP_SIZE, MMVQ_PASCAL_NWARPS, 1);

#define MMVQP_QKV_CASE(TQ, TKV)                                                                                        \
    if (a.type[0] == TQ && a.type[1] == TKV) {                                                                        \
        [[maybe_unused]] constexpr mmvqp_shape sq  = mmvqp_choose(TQ,  false);                                        \
        [[maybe_unused]] constexpr mmvqp_shape skv = mmvqp_choose(TKV, false);                                        \
        static_assert(mmvqp_fits<TQ, sq.R, sq.C, true, false>() && mmvqp_fits<TKV, skv.R, skv.C, true, false>(), "");  \
        mul_mat_vec_q_pascal_qkv<TQ, sq.R, sq.C, TKV, skv.R, skv.C><<<grid, block, 0, stream>>>(p[0], p[1], p[2],      \
                                                                                               nb[0], nb[1]);         \
        return true;                                                                                                  \
    }
    // the attention types of the Flint files this fork targets (K2h-r16, G3: Q5_0/Q8_0 mixes; UD-Q4_K_XL: Q8_0, Q5_K)
    MMVQP_QKV_CASE(GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    MMVQP_QKV_CASE(GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    MMVQP_QKV_CASE(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    MMVQP_QKV_CASE(GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    MMVQP_QKV_CASE(GGML_TYPE_Q5_K, GGML_TYPE_Q5_K)
#undef MMVQP_QKV_CASE
    return false;
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

bool ggml_cuda_mmvq_pascal_qkv(const ggml_cuda_mmvq_pascal_qkv_args & args, cudaStream_t stream) {
    GGML_UNUSED_VARS(args, stream);
    return false;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
