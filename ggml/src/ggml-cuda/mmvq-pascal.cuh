#pragma once

#include "common.cuh"

// Single-token quantized matrix-vector products (Q4_0, Q4_1, Q5_0, Q8_0) tuned for compute capability 6.1 (Pascal
// with dp4a). On small Pascal cards such as the 5-SM P106-090, mul_mat_vec_q runs out of instruction issue and load
// throughput long before memory bandwidth; these kernels give each row a lane group sized to the row width, let several
// rows share one load of the activations, load several weight blocks per lane with coalesced wide loads and keep the
// addressing out of the loop.
//
// Takes the same arguments as mul_mat_vec_q_switch_type (strides in blocks, Q8_1 activations) and returns true if it
// ran the product; false leaves it to mul_mat_vec_q. GGML_CUDA_MMVQ_PASCAL: 0 off, 1 compute capability 6.1 only
// (default), 2 any NVIDIA GPU (for testing).
bool ggml_cuda_mmvq_pascal(
        const void * vx, ggml_type type, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device & fusion,
        float * dst, int64_t ncols_x, int64_t nrows_x, int64_t ncols_dst, int64_t stride_row_x,
        int64_t nchannels_x, int64_t nchannels_y, int64_t nchannels_dst, int64_t stride_channel_x, int64_t stride_channel_y,
        int64_t stride_channel_dst, int64_t nsamples_x, int64_t nsamples_dst, int64_t stride_sample_x, int64_t stride_sample_y,
        int64_t stride_sample_dst, cudaStream_t stream);
