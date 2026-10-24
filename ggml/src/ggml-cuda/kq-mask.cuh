#include "common.cuh"

#define CUDA_KQ_MASK_BLOCK_SIZE 256

void ggml_cuda_op_kq_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
