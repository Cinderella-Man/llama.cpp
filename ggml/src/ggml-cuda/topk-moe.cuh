#include "common.cuh"
#include "ggml.h"

#include <initializer_list>

// Rows that one CUDA block handles.
#define TOPK_MOE_ROWS_PER_BLOCK 8

struct ggml_cuda_topk_moe_args {
    bool sigmoid{};
    bool sqrt_softplus{};
    bool softmax{};
    bool delayed_softmax{};
    bool prob_bias{};
    bool norm{};
    bool scale{};
    int  n_groups{};     // > 0: pick experts only from the top n_group_used expert groups
    int  n_group_used{};
    // the routed experts' tail (GGML_OP_MOE_TAIL, at most WARP_SIZE experts used) on the final weights: weights and ids
    // are then the two MOE_TAIL nodes' outputs
    bool  tail{};
    float tail_p{};
    float tail_w{};
    bool  tail_renorm{};
    bool  tail_skip{};   // up to n_skip tokens: a dropped slot's id is -1 (or with tail_repeat its token's last kept
    bool  tail_repeat{}; // expert); past them every slot keeps its own expert
};

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             clamp,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args);

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * gating_op,
                                   const ggml_tensor * weights,
                                   const ggml_tensor * logits,
                                   const ggml_tensor * ids);
