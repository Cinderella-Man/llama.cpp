#pragma once

#include "common.cuh"

// Q2_0 MUL_MAT_ID of 2 to 8 tokens tuned for compute capability 6.1: the routed experts of a k-token pass of
// Qwen3.8-Flash-Next (a speculative verify, pipelined guesses), where the k tokens pick 10 of 512 experts each. Each
// distinct expert of the step is read once and applied to every token that picked it, each weight decoded once for all
// of them, so the cost follows the number of distinct experts the step touches instead of k x 10. Every token's result
// is bit-identical to the one-token Pascal kernel's (ggml_cuda_mmvq_pascal), with or without a fused gate, bias, GLU or
// the routed + shared expert launch.
//
// Takes the arguments of ggml_cuda_mmvq_pascal and returns true if it ran the product; false leaves it to
// ggml_cuda_mmvq_pascal and mul_mat_vec_q. It runs where the Pascal kernels are active (GGML_CUDA_MMVQ_PASCAL) and Q2_0
// is not skipped (GGML_CUDA_MMVQ_PASCAL_SKIP).
// GGML_CUDA_MMVQ_PASCAL_MTOK=E,CAP,RB,G (missing fields and 0 keep the defaults):
//   E   1 on (default), 0 off: 2-4 tokens go to mul_mat_vec_q_pascal_mc, 5-8 to llama.cpp's mul_mat_vec_q_moe
//   CAP entries per group, 1-8 (default: the token count); 1 reads each expert once per (token, slot) entry, as the
//       other kernels do, so that one binary measures what the sharing gains
//   RB  row blocks per block (default 4 where an expert has 8 or more of them, else 1)
//   G   1: a fused gate/up loads its weights directly instead of staging them through shared memory (default 0)
bool ggml_cuda_mmvq_pascal_mtok(
        const void * vx, ggml_type type, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device & fusion,
        float * dst, int64_t ncols_x, int64_t nrows_x, int64_t ncols_dst, int64_t stride_row_x, int64_t stride_col_y,
        int64_t stride_col_dst, int64_t nchannels_x, int64_t nchannels_y, int64_t nchannels_dst, int64_t stride_channel_x,
        int64_t stride_channel_y, int64_t stride_channel_dst, int64_t nsamples_x, int64_t nsamples_dst, int64_t stride_sample_x,
        int64_t stride_sample_y, int64_t stride_sample_dst, int64_t ids_stride, cudaStream_t stream);
