#include "common.cuh"

#define CUDA_ROPE_BLOCK_SIZE 256

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * set_rows);

void ggml_cuda_op_rms_norm_mul_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * rms_norm, ggml_tensor * mul, ggml_tensor * rope, ggml_tensor * set_rows);

// One token of an attention block after its Q/K/V products, in one launch: Q's RMS_NORM + MUL + ROPE, K's RMS_NORM +
// MUL + ROPE + VIEW + SET_ROWS (into an F16 cache) and V's VIEW + SET_ROWS, each value as
// ggml_cuda_op_rms_norm_mul_rope_fused and the SET_ROWS op compute it. The products are read from q, k and v (each
// laid out as its tensor, contiguous), not from the graph's tensors. ggml_cuda_attn_epilogue_supported: whether it covers
// these nodes (ggml_cuda_op_attn_epilogue asserts it).
bool ggml_cuda_attn_epilogue_supported(
        const ggml_tensor * q_norm, const ggml_tensor * q_mul, const ggml_tensor * q_rope,
        const ggml_tensor * k_norm, const ggml_tensor * k_mul, const ggml_tensor * k_rope, const ggml_tensor * k_set_rows,
        const ggml_tensor * v_set_rows);
void ggml_cuda_op_attn_epilogue(ggml_backend_cuda_context & ctx,
        const ggml_tensor * q_norm, const ggml_tensor * q_mul, const ggml_tensor * q_rope,
        const ggml_tensor * k_norm, const ggml_tensor * k_mul, const ggml_tensor * k_rope, const ggml_tensor * k_set_rows,
        const ggml_tensor * v_set_rows, const float * q, const float * k, const float * v);
