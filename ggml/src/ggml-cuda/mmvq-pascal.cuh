#pragma once

#include "common.cuh"

// Quantized matrix-vector products of 1 to 4 tokens (Q4_0, Q4_1, Q5_0, Q8_0, Q5_K, IQ4_XS, Q6_K) tuned for compute
// capability 6.1 (Pascal with dp4a). On small Pascal cards such as the 5-SM P106-090, mul_mat_vec_q runs out of
// instruction issue and load throughput long before memory bandwidth; these kernels give each row a lane group sized to
// the row width, let several rows share one load of the activations, load several weight blocks per lane with coalesced
// wide loads and keep the addressing out of the loop. They compute the activation block sums in FP32 from the int8
// values instead of reading the FP16 sum of block_q8_1, so activations whose blocks add up past FP16's range stay
// finite. With 2 to 4 tokens each weight row is read once for all of them, and for MUL_MAT_ID each distinct expert once
// for all tokens that picked it.
//
// Takes the same arguments as mul_mat_vec_q_switch_type (strides in blocks, Q8_1 activations) and returns true if it
// ran the product; false leaves it to mul_mat_vec_q. GGML_CUDA_MMVQ_PASCAL: 0 off, 1 compute capability 6.1 only
// (default), 2 any NVIDIA GPU (for testing). GGML_CUDA_MMVQ_PASCAL_STAGE_ACT=1: with 2 to 4 tokens, stage each token's
// activations through shared memory too (same results, a different load pattern).
bool ggml_cuda_mmvq_pascal(
        const void * vx, ggml_type type, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device & fusion,
        float * dst, int64_t ncols_x, int64_t nrows_x, int64_t ncols_dst, int64_t stride_row_x, int64_t stride_col_y,
        int64_t stride_col_dst, int64_t nchannels_x, int64_t nchannels_y, int64_t nchannels_dst, int64_t stride_channel_x,
        int64_t stride_channel_y, int64_t stride_channel_dst, int64_t nsamples_x, int64_t nsamples_dst, int64_t stride_sample_x,
        int64_t stride_sample_y, int64_t stride_sample_dst, int64_t ids_stride, cudaStream_t stream);

// Whether the kernels run on a device of compute capability cc under GGML_CUDA_MMVQ_PASCAL.
bool ggml_cuda_mmvq_pascal_active(int cc);
