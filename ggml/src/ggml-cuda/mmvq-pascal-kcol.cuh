#pragma once

#include "common.cuh"

// Matrix products of 1 to 8 tokens (columns) for compute capability 6.1 (Pascal), tuned for the 5-SM P106-090: each
// weight is read once and decoded once per launch for all columns, and the activations are quantized into records that
// every block stages in shared memory once (the design is at the top of mmvq-pascal-kcol-impl.cuh). They are for the
// k-token passes of speculative decoding, where kernels that loop over the columns cost 2.3x (BF16) to 2.8x (Q3_K,
// IQ4_XS) a token at 4 tokens on Qwen3.8-Flash-Next's dense products. Types: Q8_0, Q2_0, Q3_K, Q4_K, Q5_K, Q6_K,
// IQ4_XS, and Q4_0, Q5_0 and IQ4_NL in rows of whole 64-weight pairs of blocks (the shared expert's down projections);
// since Q713 the grid types IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S and IQ1_M (Swift's IQ attention, shared experts and
// ssm_out at 2 to 8 tokens; one token keeps their one-token Pascal kernels, IQ1_M llama.cpp's);
// plain 2-D products without fusion (MUL_MAT; the experts' MUL_MAT_ID stays with mmvq-pascal.cu).
//
// GGML_CUDA_MMVQ_PASCAL_KCOL: 0 off; 1 (default) 2 to 8 columns, and 1 column for Q3_K and IQ4_NL (which have no other
// Pascal kernel) and for Q8_0 rows whose 64-weight chunks aren't a multiple of 4 (the one-token kernel loads those 32
// bits at a time); 2 every product of 1 to 8 columns of these types. GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=q4_K,q6_K: types
// left to the other kernels. GGML_CUDA_MMVQ_PASCAL_KCOL_R=1|2 rows per lane group, _KCOL_ITT=n iterations per activation
// tile, _KCOL_SMEM=bytes of shared memory per block (at most 49152), _KCOL_NSM=n the SM count the shape is chosen for:
// launch shapes for tuning. GGML_CUDA_MMVQ_PASCAL=0 turns these kernels off with the other Pascal kernels, =2 runs them
// on any NVIDIA GPU (testing).

// Runs dst = src0 * src1 if these kernels take it; returns false (nothing launched) otherwise.
bool ggml_cuda_mmvq_pascal_kcol(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                ggml_tensor * dst);

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
namespace kcol { struct params; }

// The launchers per type (mmvq-pascal-kcol-<type>.cu, one file per type so that they compile in parallel).
void ggml_cuda_kcol_launch_q8_0(const kcol::params & p, cudaStream_t stream);
void ggml_cuda_kcol_launch_q2_0(const kcol::params & p, cudaStream_t stream);
void ggml_cuda_kcol_launch_q3_K(const kcol::params & p, cudaStream_t stream);
void ggml_cuda_kcol_launch_q4_K(const kcol::params & p, cudaStream_t stream);
void ggml_cuda_kcol_launch_q5_K(const kcol::params & p, cudaStream_t stream);
void ggml_cuda_kcol_launch_q6_K(const kcol::params & p, cudaStream_t stream);
void ggml_cuda_kcol_launch_iq4_xs(const kcol::params & p, cudaStream_t stream);
void ggml_cuda_kcol_launch_q4_0(const kcol::params & p, cudaStream_t stream);
void ggml_cuda_kcol_launch_q5_0(const kcol::params & p, cudaStream_t stream);
void ggml_cuda_kcol_launch_iq4_nl(const kcol::params & p, cudaStream_t stream);
void ggml_cuda_kcol_launch_iq2_xxs(const kcol::params & p, cudaStream_t stream);
void ggml_cuda_kcol_launch_iq2_xs(const kcol::params & p, cudaStream_t stream);
void ggml_cuda_kcol_launch_iq2_s(const kcol::params & p, cudaStream_t stream);
void ggml_cuda_kcol_launch_iq3_xxs(const kcol::params & p, cudaStream_t stream);
void ggml_cuda_kcol_launch_iq3_s(const kcol::params & p, cudaStream_t stream);
void ggml_cuda_kcol_launch_iq1_m(const kcol::params & p, cudaStream_t stream);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
