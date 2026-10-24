#pragma once

#include "common.cuh"

// One MoE layer of a Flint-shaped model at one token as a single persistent kernel (GGML_CUDA_PERSISTENT_MOE).
//
// Shapes: hidden size 2048, 256 routed experts of width 512 of which 6 are used, one shared expert of width 1024,
// sigmoid routing with normalized weights. The kernel computes the layer from its input x (the residual stream) to
// ffn_out: RMS norm, router, top-6, the six routed experts' gate/up + SwiGLU and down projections, the shared expert
// and the weighted sum. That is the 35 graph nodes for which the per-op path makes 10 launches (norm, router, top-k,
// routed gate/up, quantization + routed down, weighted sum, shared gate/up, quantization + shared down).
//
// Work list: the layer is cut into warp-sized items numbered so that every dependency points to a smaller number, and
// the warps take them from a counter. Dependencies are counters, not barriers, so the kernel completes at any grid
// size and needs no cooperative launch. Every product, reduction and quantization is the per-op path's device code in
// the same order, so ffn_out is bit-identical to it. bench/persistent_pascal of
// github.com/Cinderella-Man/llms-on-crypto-mining-rigs has the prototype and the argument why it cannot deadlock.
//
// GGML_CUDA_PERSISTENT_MOE:       0 off (default), 1 compute capability 6.1, 2 any NVIDIA GPU (for testing)
// GGML_CUDA_PERSISTENT_MOE_CHECK: 1 runs the per-op path as well and compares ffn_out bit for bit on the device; the
//                                 counts are logged per graph evaluation (CUDA graphs are off in this mode)
// GGML_CUDA_PERSISTENT_MOE_GRID:  N blocks per launch instead of the resident maximum (for testing)
// GGML_CUDA_PERSISTENT_MOE_FAULT: a deliberate error, to show the tests catch one: 1 a wait that times out, 2 routing
//                                 weights off by 1e-4, 3 the shared expert missing from ffn_out

struct ggml_cuda_moe_layer {
    const ggml_tensor * x         = nullptr; // layer input (residual stream), F32 [n_embd]
    const ggml_tensor * norm_w    = nullptr; // F32 [n_embd]
    const ggml_tensor * router    = nullptr; // F16, BF16 or F32 [n_embd, n_expert]
    const ggml_tensor * gate_exps = nullptr; // [n_embd, n_ff_exp, n_expert]
    const ggml_tensor * up_exps   = nullptr;
    const ggml_tensor * down_exps = nullptr; // [n_ff_exp, n_embd, n_expert]
    const ggml_tensor * gate_sh   = nullptr; // [n_embd, n_ff_shexp]
    const ggml_tensor * up_sh     = nullptr;
    const ggml_tensor * down_sh   = nullptr; // [n_ff_shexp, n_embd]
    ggml_tensor *       dst       = nullptr; // ffn_out: the shared expert plus the routed experts' weighted sum
    float               eps       = 0.0f;
    float               clamp_val = 0.0f;    // lower bound of the routing weights' sum
    float               scale_val = 1.0f;    // routing weight scale
    int                 n_nodes   = 0;       // graph nodes from the RMS norm to dst, foreign view nodes included
};

// whether the kernel is switched on for a device of compute capability cc
bool ggml_cuda_moe_persistent_enabled(int cc);

// GGML_CUDA_PERSISTENT_MOE_CHECK=1
bool ggml_cuda_moe_persistent_check_mode();

// whether a kernel exists for the layer's shapes, types and alignment
bool ggml_cuda_moe_persistent_supported(const ggml_cuda_moe_layer & layer);

// Runs the layer into layer.dst. In check mode it writes a scratch output instead and remembers layer.dst; the per-op
// path then computes dst as usual and ggml_cuda_moe_persistent_check_node compares the two once dst is written.
void ggml_cuda_op_moe_persistent(ggml_backend_cuda_context & ctx, const ggml_cuda_moe_layer & layer);

// check mode: after node was computed, compare it with the persistent kernel's output if it is the remembered dst
void ggml_cuda_moe_persistent_check_node(ggml_backend_cuda_context & ctx, const ggml_tensor * node);

// At the start of every graph compute (CUDA graph replays included): aborts if an earlier launch timed out (a bug: the
// kernel cannot deadlock), and resets the per-evaluation counts.
void ggml_cuda_moe_persistent_begin_graph(ggml_backend_cuda_context & ctx);

// After the nodes of a graph evaluation were launched: in check mode, waits for them and logs how many layers the kernel
// ran and how many of their outputs differed from the per-op path's.
void ggml_cuda_moe_persistent_end_graph(ggml_backend_cuda_context & ctx);

// frees the context's counters and scratch
void ggml_cuda_moe_persistent_free(ggml_backend_cuda_context & ctx);
