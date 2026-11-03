#pragma once

#include "common.cuh"

// Gated DeltaNet launches that replace short chains of ops (C20, Q438; ports of Strata's gdn_conv_l2_kernel and
// gdn_step_norm_kernel, MIT): the conv window as one op, and the gated output norm as one fusion.

// where GGML_OP_GDN_CONV writes its new histories when the backend fuses the cpy into the cache that follows it;
// strides in elements
struct ggml_cuda_gdn_conv_fused_cache {
    float * data;        // slot 0, sequence 0
    int64_t seq_stride;  // between sequences
    int64_t slot_stride; // between rollback slots
};

bool ggml_cuda_gdn_conv_supported(const ggml_tensor * op);

// GGML_OP_GDN_CONV; with cache the histories go there instead of the tail of dst
void ggml_cuda_op_gdn_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                           const ggml_cuda_gdn_conv_fused_cache * cache = nullptr);

// rms_norm(x) * w * sigmoid(z) over rows of x, as RMS_NORM, MUL, UNARY(SIGMOID), MUL: the GDN output norm
// (llama_model_qwen4exp::graph::build_norm_gated). x, w and dst have contiguous rows; z any row strides.
bool ggml_cuda_gated_rms_norm_supported(const ggml_tensor * rms_norm, const ggml_tensor * mul_w,
                                        const ggml_tensor * sigmoid, const ggml_tensor * mul_gate);
void ggml_cuda_op_gated_rms_norm(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm,
                                 const ggml_tensor * mul_w, const ggml_tensor * sigmoid, ggml_tensor * mul_gate);
