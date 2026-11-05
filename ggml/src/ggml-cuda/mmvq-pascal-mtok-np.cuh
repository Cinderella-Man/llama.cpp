#pragma once

// The multi-token expert kernels of mmvq-pascal-mtok.cu for N2P, N3P, N2P-128 and N3P-128 (Q928): MUL_MAT_ID of 2 to 8
// tokens, each distinct expert of the step read once for all the tokens that picked it, as for the grid types
// (mmvq-pascal-mtok-iq.cuh, whose groups, members and arguments these share). The routed experts of Qwen3.8-Flash-Next
// in N2P (gate/up, K = 2560) and N2P-128 or N3P-128 (down, K = 640).
//   - The block stages its members' activations in shared memory once, for all its row blocks: per member and block of
//     the row (256 or 128 weights) the int8 values and 16 bytes of padding, so that neighbouring lanes hit different
//     banks; the FP32 scales of the row's Q8_1 blocks after them.
//   - Weights: one block per lane and iteration, lanes per row from the row width. A lane group's row segment is staged
//     through shared memory with the widest loads the rows allow (16, 8 or 4 bytes), the gate and up matrices one after
//     the other; where the segments are not whole 16-byte units and there is no fused gate (the down projection, K = 640:
//     N2P-128's 220-byte and N3P-128's 270-byte rows, one lane per row), each warp stages its rows at once as a tile
//     with 16-byte loads, as the Q2_0 kernel does (MTOK_TILE). One row per lane group.
//   - Each 32 weights are decoded once (__byte_perm from the register tables, as mmvqp_dot_n2p and mmvqp_dot_n3p do) and
//     multiplied by every member: 8 dp4a per member.
// Each token's sum is formed as in the one-token kernel: the same integer sums per 32 weights, the same FP32 operations in
// the same order, the same reduction over a row's lanes, so every result is bit-identical to mul_mat_vec_q_pascal's.

#include "mmvq-pascal-mtok-iq.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

namespace mtok_np {

// NB: 32-weight blocks per block of the type; N3: the 3-bit layout (one scale per 32 weights)
template <ggml_type type> constexpr __host__ __device__ int nb() {
    return type == GGML_TYPE_N2P || type == GGML_TYPE_N3P ? 8 : 4;
}
template <ggml_type type> constexpr __host__ __device__ bool n3() {
    return type == GGML_TYPE_N3P || type == GGML_TYPE_N3P_128;
}
// a member's staged int8 values per block of the row: the block's weights and 16 bytes of padding
constexpr __host__ __device__ int act_run(const int nb) {
    return 32*nb + 16;
}

// how a lane gets its blocks: staged row segments or staged warp tiles of whole rows
enum staging { SEG = 0, TILE = 1 };

struct args : mtok_iq::args {
    int tile_bytes; // TILE: the warp tiles, at the start of the dynamic shared memory
};

// The int8 weights of 32-weight block ib of a lane's block (its words w, laid out as mmvqp_dot_n2p / mmvqp_dot_n3p read
// them) and its integer scales: N2P and N2P-128 one per 8 weights (ls[0..3]), N3P and N3P-128 one (ls[0]).
template <ggml_type type>
static __device__ __forceinline__ void decode32(const uint32_t * w, const int ib, int * q, int * ls) {
    constexpr int NB = nb<type>();
    if constexpr (!n3<type>()) {
        const uint32_t ex = w[5*NB/2];
        const uint32_t sc = w[2*NB + ib/2] >> 16*(ib%2);
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const uint32_t qw = w[2*ib + k];
#pragma unroll
            for (int sb = 0; sb < 2; ++sb) {
                const uint32_t T = (ex >> (4*ib + 2*k + sb)) & 1 ? N2P_TAB1 : N2P_TAB0;
                q[4*k + 2*sb]     = (int) __byte_perm(T, T, qw >> (16*sb));
                q[4*k + 2*sb + 1] = (int) __byte_perm(T, T, qw >> (16*sb + 2));
                ls[2*k + sb] = (int) ((sc >> 4*(2*k + sb)) & 0xF) - 8;
            }
        }
    } else {
        constexpr int QS = NB == 8 ?  2 :  0;
        constexpr int QH = NB == 8 ? 18 :  8;
        constexpr int W0 = NB == 8 ?  0 : 12;
        constexpr int SL = NB == 8 ?  1 : 13;
        const uint32_t w0 = w[W0];
        const bool     sh = (w0 >> (16 + ib)) & 1;
        const uint32_t ta = sh ? N3P_TAB1A : N3P_TAB0A;
        const uint32_t tb = sh ? N3P_TAB1B : N3P_TAB0B;
        const uint32_t hi = w[QH + ib];
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const uint32_t lo  = w[QS + 2*ib + g/4];
            const int      t   = 16*((g%4)/2) + 2*(g%2);
            const int      s   = 16*(g/4) + g%4 - 2;
            const uint32_t hs  = s >= 0 ? hi >> s : hi << -s;
            const uint32_t sel = ((lo >> t) & 0x3333) | (hs & 0xCCCC);
            q[g] = (int) __byte_perm(ta, tb, sel);
        }
        ls[0] = (int) (((w[SL] >> 4*ib) & 0xF) | (((w0 >> (24 + ib)) & 1) << 4)) - 16;
    }
}

// the block's FP16 scale
template <ggml_type type>
static __device__ __forceinline__ float block_scale(const uint32_t * w) {
    constexpr int NB = nb<type>();
    if constexpr (!n3<type>()) {
        return NB == 8 ? mmvqp_h2f_lo(w[21]) : mmvqp_h2f_hi(w[10]);
    } else {
        return mmvqp_h2f_lo(w[NB == 8 ? 0 : 12]);
    }
}

// A lane's block of BS bytes from shared memory: 2-byte-aligned blocks with mmvqp_lds_h's shifts, others as words.
template <int BS>
static __device__ __forceinline__ void lds_block(uint32_t * w, const char * p) {
    if constexpr (BS % 4 == 2) {
        mtok_iq::lds_h<BS>(w, p);
    } else {
        mmvqp_lds_words<BS, mmvqp_gcd(BS, 16)>(w, p);
    }
}

// TILE: the warp's rows of row block rblk of matrix x (contiguous in memory) copied into its part of the tiles with
// coalesced 128-bit loads (mtok_stage_tile of mmvq-pascal-mtok.cu). Returns the warp's copy.
static __device__ __forceinline__ const char * stage_tile(const char * x, const args & p, const int rblk, char * tiles) {
    constexpr int UN = 4; // loads in flight per lane
    const int wrows  = WARP_SIZE >> p.lpr_log2;
    const int wrow0  = (rblk * MMVQP_MTOK_IQ_NWARPS + (int) threadIdx.y) * wrows;
    const int wbytes = wrows * p.row_bytes;
    uint4       * dsm = (uint4 *) (tiles + threadIdx.y * wbytes);
    const uint4 * src = (const uint4 *) (x + (int64_t) wrow0 * p.row_bytes);
    const int     nu  = (max(0, min(wrows, p.nrows - wrow0)) * p.row_bytes) >> 4;
    __syncwarp(); // every lane is done with the warp's previous tile
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

// The block's rows for its NT members: xu and xg the up (or only) and gate matrices of its expert, smw the warp's segment
// buffer (SEG), tiles the warp tiles (TILE), act the members' staged activations, doff (lane j) member j's output offset.
template <ggml_type type, bool gate, int st, int SU, int NT>
static __device__ __forceinline__ void rows(const args & p, const ggml_cuda_mm_fusion_args_device & fusion,
                                            const char * xu, const char * xg, char * smw, char * tiles,
                                            const char * act, const int doff, float * dst, const int64_t bias_off) {
    using unit_t = typename mtok_iq::stage_unit<SU>::t;
    constexpr int NB  = nb<type>();
    constexpr int RUN = act_run(NB);
    constexpr int BS  = mmvqp_traits<type>::bs;
    constexpr int NWD = (BS + 3) / 4;            // words of a block in registers
    constexpr int NW  = MMVQP_MTOK_IQ_NWARPS;
    constexpr int M   = gate ? 2 : 1;            // matrices: up (or plain), gate
    constexpr int NU  = (BS + SU - 1) / SU;      // SEG: staging units per lane and row, the last one predicated
    constexpr int NL  = n3<type>() ? 1 : 4;      // integer scales per 32 weights
    static_assert(st == SEG || !gate, "tile staging: one matrix");

    const int lane     = threadIdx.x;
    const int lpr_log2 = p.lpr_log2;
    const int lpr      = 1 << lpr_log2;
    const int sl       = lane & (lpr - 1);
    const int grp      = lane >> lpr_log2;
    const int wstep    = BS << lpr_log2;        // a row segment's bytes per iteration
    [[maybe_unused]] const int su_units = wstep / SU;
    GGML_UNUSED_VARS(xg, smw, tiles);

    int dc[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        dc[j] = __shfl_sync(0xFFFFFFFF, doff, j);
    }
    const float * ad0 = (const float *) (act + p.nchunk * RUN); // member 0's scales
    const int     ast = p.act_tok / 4;                          // floats between members

    for (int rbi = 0; rbi < p.rb; ++rbi) {
        const int rblk = (int) blockIdx.x * p.rb + rbi;
        if (rblk >= p.nrowblocks) {
            break;
        }
        const int row0 = ((rblk * NW + (int) threadIdx.y) << (5 - lpr_log2)) + grp;

        const char * rw[M];
        rw[0] = xu + (int64_t) min(row0, p.nrows - 1) * p.row_bytes;
        if constexpr (gate) {
            rw[1] = xg + (int64_t) min(row0, p.nrows - 1) * p.row_bytes;
        }
        [[maybe_unused]] const char * smt = nullptr; // TILE: this lane's block of its row
        if constexpr (st == TILE) {
            smt = stage_tile(xu, p, rblk, tiles) + grp * p.row_bytes + sl * BS;
        }

        float acc[NT][M];
#pragma unroll
        for (int j = 0; j < NT; ++j) {
#pragma unroll
            for (int m = 0; m < M; ++m) {
                acc[j][m] = 0.0f;
            }
        }

        for (int it = 0; it < p.niter; ++it) {
            const int     c  = (it << lpr_log2) + sl; // the lane's block of the row
            const char  * au = act + c * RUN;         // its int8 activations, member 0
            const float * ad = ad0 + NB * c;          // its activation scales, member 0
#pragma unroll
            for (int m = 0; m < M; ++m) {
                uint32_t w[NWD];
                if constexpr (st == SEG) {
                    // the lane group's segment of matrix m, then the lane's block from it
                    unit_t       * dsm = (unit_t *) (smw + grp * wstep);
                    const unit_t * src = (const unit_t *) rw[m];
#pragma unroll
                    for (int j = 0; j < NU; ++j) {
                        const int u = sl + (j << lpr_log2);
                        if (j < NU - 1 || u < su_units) {
                            dsm[u] = __ldg(src + u);
                        }
                    }
                    rw[m] += wstep;
                    __syncwarp();
                    lds_block<BS>(w, smw + lane * BS);
                    __syncwarp(); // the block is in registers: the next segment may overwrite it
                } else {
                    lds_block<BS>(w, smt);
                    smt += wstep;
                }

                const float dd = block_scale<type>(w);
                float sumf[NT];
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    sumf[j] = 0.0f;
                }
#pragma unroll
                for (int ib = 0; ib < NB; ++ib) {
                    int q[8];
                    int ls[4];
                    decode32<type>(w, ib, q, ls);
#pragma unroll
                    for (int j = 0; j < NT; ++j) {
                        const uint4 * pu = (const uint4 *) (au + j * p.act_tok + 32 * ib);
                        const uint4   u0 = pu[0];
                        const uint4   u1 = pu[1];
                        const int     u[8] = { (int) u0.x, (int) u0.y, (int) u0.z, (int) u0.w,
                                               (int) u1.x, (int) u1.y, (int) u1.z, (int) u1.w };
                        int isum = 0;
                        if constexpr (NL == 4) {
#pragma unroll
                            for (int s = 0; s < 4; ++s) {
                                int sumb = ggml_cuda_dp4a(q[2*s], u[2*s], 0);
                                sumb     = ggml_cuda_dp4a(q[2*s + 1], u[2*s + 1], sumb);
                                isum += sumb * ls[s];
                            }
                        } else {
#pragma unroll
                            for (int g = 0; g < 8; ++g) {
                                isum = ggml_cuda_dp4a(q[g], u[g], isum);
                            }
                            isum *= ls[0];
                        }
                        // mmvqp_dot_n2p / mmvqp_dot_n3p: fmaf(d8, isum, sumf) per 32 weights, then fmaf(d, sumf, acc)
                        sumf[j] = fmaf(ad[j * ast + ib], (float) isum, sumf[j]);
                    }
                }
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    acc[j][m] = fmaf(dd, sumf[j], acc[j][m]);
                }
            }
        }

#pragma unroll
        for (int j = 0; j < NT; ++j) {
#pragma unroll
            for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
                if (off < lpr) {
#pragma unroll
                    for (int m = 0; m < M; ++m) {
                        acc[j][m] += __shfl_xor_sync(0xFFFFFFFF, acc[j][m], off, WARP_SIZE);
                    }
                }
            }
        }

        const int row = row0;
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            if (sl != 0 || row >= p.nrows) {
                break;
            }
            float result = acc[j][0];
            if (fusion.x_bias) {
                result += ((const float *) fusion.x_bias)[bias_off + row];
            }
            if constexpr (gate) {
                float gate_value = acc[j][M - 1];
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

// The member count nt (1..NTM) as a constant.
template <ggml_type type, bool gate, int st, int SU, int NT, int NTM>
static __device__ __forceinline__ void rows_nt(const int nt, const args & p, const ggml_cuda_mm_fusion_args_device & fusion,
                                               const char * xu, const char * xg, char * smw, char * tiles,
                                               const char * act, const int doff, float * dst, const int64_t bias_off) {
    if constexpr (NT == NTM) {
        rows<type, gate, st, SU, NT>(p, fusion, xu, xg, smw, tiles, act, doff, dst, bias_off);
    } else {
        if (nt == NT) {
            rows<type, gate, st, SU, NT>(p, fusion, xu, xg, smw, tiles, act, doff, dst, bias_off);
        } else {
            rows_nt<type, gate, st, SU, NT + 1, NTM>(nt, p, fusion, xu, xg, smw, tiles, act, doff, dst, bias_off);
        }
    }
}

// Static shared memory of a SEG kernel: the warps' staging buffers (one matrix's segments, 16 bytes of padding for
// lds_h's read past a block).
template <ggml_type type> constexpr int static_smem() {
    return MMVQP_MTOK_IQ_NWARPS * (WARP_SIZE * mmvqp_traits<type>::bs + 16);
}

// Grid: (row blocks / rb, tokens * slots + shared expert, samples). Dynamic shared memory: the warp tiles (TILE), then
// the staged activations of up to NTM members, act_tok bytes each.
template <ggml_type type, bool gate, int st, int SU, int NTM>
__launch_bounds__(MMVQP_MTOK_IQ_NWARPS*WARP_SIZE, NTM >= 8 ? 2 : MMVQP_MTOK_IQ_MIN_BLOCKS)
static __global__ void mul_mat_vec_np_pascal_mtok(const args p, const ggml_cuda_mm_fusion_args_device fusion) {
    constexpr int NB  = nb<type>();
    constexpr int RUN = act_run(NB);
    constexpr int BS  = mmvqp_traits<type>::bs;
    constexpr int NW  = MMVQP_MTOK_IQ_NWARPS;
    __shared__ uint4 sm[st == SEG ? NW : 1][st == SEG ? (WARP_SIZE * BS + 16 + 15) / 16 : 1];
    extern __shared__ uint4 mtok_np_dyn[];

    const int      lane          = threadIdx.x;
    const bool     shared_expert = gate && fusion.shared_up && blockIdx.y == gridDim.y - 1;
    const uint32_t sample_dst    = blockIdx.z;
    const uint32_t sample_x      = fastdiv(sample_dst, p.sample_ratio);

    int      nt   = 0;
    int      yoff = 0;
    int      doff = 0;
    uint32_t channel_x;
    ggml_cuda_pdl_sync();
    if (!shared_expert) {
        if (!mtok_iq::group(p, sample_dst, nt, yoff, doff, channel_x)) {
            return;
        }
    } else {
        nt = p.ncols;
        if (lane < nt) {
            yoff = (int) (sample_dst*p.sample_bytes_y + lane*p.col_bytes_y);
            doff = (int) (sample_dst*p.stride_sample_dst + lane*fusion.shared_stride_col_dst);
        }
        channel_x = 0;
    }

    // the members' activations, staged by the whole block
    const int tid   = threadIdx.y * WARP_SIZE + lane;
    char    * tiles = (char *) mtok_np_dyn;
    char    * act   = tiles + p.tile_bytes;
    const int nq    = NB * p.nchunk;
    for (int j = 0; j < nt; ++j) {
        const char * yj = p.vy + __shfl_sync(0xFFFFFFFF, yoff, j);
        char       * aj = act + j * p.act_tok;
        for (int q = tid; q < nq; q += NW * WARP_SIZE) {
            const unsigned int * ys = (const unsigned int *) (yj + q * (int) sizeof(block_q8_1));
            uint32_t v[9];
#pragma unroll
            for (int k = 0; k < 9; ++k) {
                v[k] = __ldg(ys + k);
            }
            uint4 * du = (uint4 *) (aj + (q / NB) * RUN + (q % NB) * 32);
            du[0] = make_uint4(v[1], v[2], v[3], v[4]);
            du[1] = make_uint4(v[5], v[6], v[7], v[8]);
            ((float *) (aj + p.nchunk * RUN))[q] = mmvqp_h2f_lo(v[0]);
        }
    }
    __syncthreads();

    const int64_t xoff     = sample_x*p.sample_bytes_x + channel_x*p.channel_bytes_x;
    const char *  xu       = (shared_expert ? (const char *) fusion.shared_up : p.vx) + xoff;
    const char *  xg       = gate ? (const char *) (shared_expert ? fusion.shared_gate : fusion.gate) + xoff : nullptr;
    float       * dst      = shared_expert ? fusion.shared_dst : p.dst;
    const int64_t bias_off = sample_dst*p.stride_sample_dst + channel_x*p.stride_channel_dst;
    rows_nt<type, gate, st, SU, 1, NTM>(nt, p, fusion, xu, xg, (char *) sm[st == SEG ? threadIdx.y : 0], tiles, act, doff,
                                        dst, bias_off);
}

template <ggml_type type, bool gate, int st, int SU>
static void launch_ntm(const int ntm, const args & p, const ggml_cuda_mm_fusion_args_device & fusion, const dim3 grid,
                       const int smem, cudaStream_t stream) {
    const dim3 block(WARP_SIZE, MMVQP_MTOK_IQ_NWARPS, 1);
    if (ntm <= 2) {
        mul_mat_vec_np_pascal_mtok<type, gate, st, SU, 2><<<grid, block, smem, stream>>>(p, fusion);
    } else if (ntm <= 4) {
        mul_mat_vec_np_pascal_mtok<type, gate, st, SU, 4><<<grid, block, smem, stream>>>(p, fusion);
    } else {
        mul_mat_vec_np_pascal_mtok<type, gate, st, SU, 8><<<grid, block, smem, stream>>>(p, fusion);
    }
}

// su: bytes per staging load (16, 8 or 4; 0: warp tiles, no gate); ntm: the most members a group can have
template <ggml_type type>
static void launch(const args & p, const ggml_cuda_mm_fusion_args_device & fusion, const dim3 grid, const int smem,
                   const bool gate, const int su, const int ntm, cudaStream_t stream) {
    if (gate) {
        switch (su) {
            case 16: launch_ntm<type, true, SEG, 16>(ntm, p, fusion, grid, smem, stream); break;
            case 8:  launch_ntm<type, true, SEG, 8>(ntm, p, fusion, grid, smem, stream);  break;
            default: launch_ntm<type, true, SEG, 4>(ntm, p, fusion, grid, smem, stream);  break;
        }
    } else {
        switch (su) {
            case 16: launch_ntm<type, false, SEG, 16>(ntm, p, fusion, grid, smem, stream); break;
            case 8:  launch_ntm<type, false, SEG, 8>(ntm, p, fusion, grid, smem, stream);  break;
            case 4:  launch_ntm<type, false, SEG, 4>(ntm, p, fusion, grid, smem, stream);  break;
            default: launch_ntm<type, false, TILE, 16>(ntm, p, fusion, grid, smem, stream); break;
        }
    }
}

} // namespace mtok_np

// The launchers per type (mmvq-pascal-mtok-<type>.cu, one file per type so that they compile in parallel).
#define MMVQP_MTOK_NP_DECL(name) \
    void ggml_cuda_mtok_np_launch_##name(const mtok_np::args & p, const ggml_cuda_mm_fusion_args_device & fusion, \
                                         const dim3 grid, const int smem, const bool gate, const int su, const int ntm, \
                                         cudaStream_t stream)
MMVQP_MTOK_NP_DECL(n2p);
MMVQP_MTOK_NP_DECL(n3p);
MMVQP_MTOK_NP_DECL(n2p_128);
MMVQP_MTOK_NP_DECL(n3p_128);

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
