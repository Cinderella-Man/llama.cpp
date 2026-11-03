#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

#define LLAMA_MAX_SEQ 256

struct llama_head_shortlist;

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;        // number of recurrent-state snapshots per seq for rollback
    uint32_t n_outputs_max;   // max outputs supported by the context
    uint32_t n_outputs_max_per_seq;
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    int32_t  nextn_layer_offset = 0;

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool embeddings_nextn;        // also extract the hidden state before the final output norm
    bool embeddings_nextn_masked; // extract for only rows where batch.logits != 0
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool fused_lid;          // use fused lightning indexer
    bool auto_flid;
    bool fused_dsv4_hc_pre;
    bool fused_dsv4_hc_comb;
    bool fused_dsv4_hc_post;
    bool auto_fhc;
    bool fused_top_k_mask;   // use fused top-k sparse attention mask
    bool auto_ftkm;
    bool fused_kv_rows;      // unified cache: attend to the cells of each sequence only (ggml_flash_attn_ext_rows)
    bool auto_fkvr;
    bool no_perf;
    bool warmup;             // TODO: remove [TAG_LLAMA_GRAPH_NO_WARMUP]
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;
    bool training;           // set by llama_opt_init()
    bool moe_down_rescale_tg; // LLAMA_NO_MOE_DOWN_RESCALE unset: as "tg" (every device of compute capability 6.1)

    bool     screen_head;    // screened output head: exact logits only for the candidates of the model's output_screen
    uint32_t screen_head_m;  // number of candidates; 0 = the model's output_screen.top_m

    // a draft's head over a shortlist of tokens (llama_head_shortlist_*), used while head_short_on; head_short_n is the
    // number of further ids at the last change, so that a change rebuilds the graph
    const llama_head_shortlist * head_short    = nullptr;
    bool                         head_short_on = false;
    uint32_t                     head_short_n  = 0;

    // fewer routed experts on confident tokens (Q618, llama_set_moe_tail, LLAMA_MOE_TAIL): after the router's top-k a
    // slot's weight becomes 0 when the shares of the slots ranked above it already sum to moe_tail_p, or when its own
    // share is below moe_tail_w (shares: the slot's weight over the token's sum of its k weights); moe_tail_renorm
    // rescales the kept weights to that sum; moe_tail_hist makes the graph output the experts kept per token and layer
    bool  moe_tail        = false;
    float moe_tail_p      = 1.0f;
    float moe_tail_w      = 0.0f;
    bool  moe_tail_renorm = false;
    bool  moe_tail_hist   = false;

    // the main model's head over the shortlist head_short with an exact bound on the other logits (Q619,
    // llama_head_bound_*): while head_bound_on, a decode with one output computes the shortlist's logits and keeps
    // the head's input row for the bound and the full-head fallback
    bool head_bound_on = false;

    size_t moe_cache_size;

    std::vector<bool> embeddings_layer_inp; // [n_layer()] extract input embeddings for layer
    bool embeddings_layer_inp_dev; // write enabled layer inputs to a device buffer instead of host

    // staircase drafter (llama_pipe_draft_*): layer inputs the pipe evaluations copy out (kept as graph outputs), and
    // whether a micro-batch with fewer outputs than tokens also computes every position's final hidden state
    std::vector<bool> pipe_tap_layer; // [n_layer() + 1]
    bool              pipe_embd_all = false;

    enum llama_context_type ctx_type;
    enum llama_rope_scaling_type rope_scaling_type;
    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;

    llama_context * ctx_other;
};
