#pragma once

#include "common.cuh"

// GGML_OP_MOE_TAIL (ggml_moe_tail, ggml_moe_tail_ids): the low-weight tail of each token's routed experts, cut in the
// arithmetic of the CPU backend (a double sum, IEEE divisions, a float running sum), so that both backends drop the same
// slots. The fused router kernel (topk-moe.cu) computes the same cut in registers.
//
// A dropped slot's id is -1 for up to n_skip tokens: the expert kernels with ids (mmvq.cu, mmvq-pascal.cu,
// mmvq-pascal-mtok.cu) write zeros for it and never read its rows; the others (MMQ, MMF, MMVF and the host fallback of
// ggml_cuda_mul_mat_id) run on a copy of the ids in which the slot names an expert its token does not, and then zero the
// slot's rows (ggml_cuda_skip_ids). With repeat the slot names its token's last kept expert instead, which the fork's
// expert kernels take; the copy then names an unused expert for it too, since llama.cpp's MMQ cannot take a repeated
// expert (Q875). Past n_skip tokens every slot keeps its own expert at weight 0.

void ggml_cuda_op_moe_tail(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// One slot of the cut, as the CPU backend computes it (ggml-cpu/ops.cpp, ggml_moe_tail_keep): share = weight / total,
// before = (running sum of the shares up to it) - share.
static __device__ __forceinline__ bool ggml_cuda_moe_tail_keep(const float wt, const float total, float & cum, const int j,
                                                               const float p, const float w) {
    const float share  = __fdiv_rn(wt, total);
    cum                = j == 0 ? share : __fadd_rn(cum, share);
    const float before = __fsub_rn(cum, share);
    return __fsub_rn(p, before) > 0.0f && (w <= 0.0f || __fsub_rn(share, w) > 0.0f);
}

// what ids may hold besides the router's distinct experts: nothing (ARGSORT, TOP_K, a MOE_TAIL past n_skip tokens),
// -1 (anything else, e.g. a test's input), or -1 and repeats (a MOE_TAIL with repeat)
enum ggml_cuda_skip_mode {
    GGML_CUDA_SKIP_NONE,
    GGML_CUDA_SKIP_NEG,
    GGML_CUDA_SKIP_REPEAT,
};
ggml_cuda_skip_mode ggml_cuda_ids_skip_mode(const ggml_tensor * ids);

// zeros in dst's rows [ne0] of the slots whose id is -1 (dst [ne0, slots, tokens], ids [slots, tokens])
void ggml_cuda_ids_zero_rows(ggml_backend_cuda_context & ctx, const ggml_tensor * ids, ggml_tensor * dst);

// For a kernel that reads every slot's expert, of n_expert experts: ids() is ids, or (when they may hold -1 or repeats)
// a copy in which each such slot names an expert its token does not; zero_rows() then writes zeros to dst's rows [ne0]
// of the slots that were -1 (dst [ne0, slots, tokens]).
struct ggml_cuda_skip_ids {
    ggml_cuda_skip_ids(ggml_backend_cuda_context & ctx, const ggml_tensor * ids, int64_t n_expert);

    const ggml_tensor * ids() const { return active ? &fixed : orig; }
    void zero_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) const;

    const ggml_tensor *        orig;
    ggml_tensor                fixed;
    ggml_cuda_pool_alloc<char> buf;
    bool                       active = false;
};
