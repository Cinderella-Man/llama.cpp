#pragma once

#include "llama.h"

#include <cstdint>

enum diffusion_algorithm {
    DIFFUSION_ALGORITHM_ORIGIN           = 0,
    DIFFUSION_ALGORITHM_ENTROPY_BASED    = 1,
    DIFFUSION_ALGORITHM_MARGIN_BASED     = 2,
    DIFFUSION_ALGORITHM_RANDOM           = 3,
    DIFFUSION_ALGORITHM_CONFIDENCE_BASED = 4,
};

// Unified transfer scheduling methods
enum diffusion_transfer_schedule {
    DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED = 0,  // Dream-style: (1.0 - s/t) * remaining
    DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED    = 1,  // LLaDA-style: process in blocks with get_num_transfer_tokens
};

typedef bool (*diffusion_step_callback_t)(int32_t             step,
                                          int32_t             total_steps,
                                          const llama_token * tokens,
                                          int32_t             n_tokens,
                                          void *              user_data);

struct diffusion_params {
    int32_t                   steps                   = 0;
    float                     temperature             = 0;
    llama_token               mask_token_id           = LLAMA_TOKEN_NULL;
    diffusion_step_callback_t step_callback           = nullptr;
    void *                    step_callback_user_data = nullptr;
    int32_t                   seed                    = 0;
    bool                      visual_mode             = false;
    bool                      shift_logits            = false;  // Shift logits by -1 after decode
    bool                      suppress_mask_token     = false;  // forbid revealing a position as the mask token
                                                                // (masked-diffusion models that can emit it)
    bool                      self_conditioning       = false;  // feed each step's canvas logits back into the
                                                                // next step (DiffusionGemma; no-op for others)

    float   top_p = 0.;
    int32_t top_k = 0.;

    diffusion_algorithm         algorithm = DIFFUSION_ALGORITHM_CONFIDENCE_BASED;
    diffusion_transfer_schedule schedule  = DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED;

    float   cfg_scale        = 0.;     // Config scale for classifier-free guidance
    float   eps              = 0.;     // Timestep scheduling
    int32_t block_length     = 0;      // Block size (for block scheduling)
    float   alg_temp         = 0;      // algorithm temperature (0.0 = deterministic)
    bool    add_gumbel_noise = false;  // Add gumbel noise to the logits if temp > 0.0
    bool    backend_sampling = false;  // Sample on the backend (GPU) instead of the CPU
    bool    infill           = false;  // Input is a canvas: only the mask tokens it contains are generated

    float   conf_threshold   = 0.0f;   // Commit all tokens with confidence >= threshold per step (0 = use the transfer schedule)

    int32_t kv_prefix        = 0;      // Layer A prefix cache: block size (0 = off). Per block: one full
                                       // warm forward, then suffix-only steps vs the cached prefix.
    int32_t kv_block         = 0;      // Layer A dual cache: block size (0 = off). Per block: one full
                                       // warm forward, then block-only steps vs the frozen full canvas.
                                       // Requires conf_threshold > 0, timestep schedule, no infill/cfg/sc.

    int32_t kv_rewarm        = 6;      // re-warm after this many cached steps (drift guard; swept 2026-06-12)
    int32_t kv_rewarm_commits = 0;     // re-warm after this many commits since last warm (0 = off;
                                       // drift tracks canvas CHANGES, not steps - often the better trigger)
    float   tau_alpha        = 0.0f;   // Layer B1 adaptive threshold, BLOCK-SCOPED (the reference
                                       // semantics): tau decays only INSIDE a 32-token window from the
                                       // first remaining mask; positions beyond keep conf_threshold.
                                       // tau_eff = conf_threshold * (1 - alpha*(1 - r_mask_window)).
                                       // Global decay was refuted by bench (13/45). 0 = fixed.
    float   tau_floor        = 0.0f;   // absolute lower bound for the decayed tau (0 = none)
    float   kv_span          = 0.0f;   // Layer B3 (SlowFast): dynamic kv-block sizing - at warm
                                       // steps, the next block extends over the contiguous
                                       // confident span (confidence >= kv_span) from the block
                                       // start, clamped to [8, 64]. Requires kv_block. 0 = fixed.
    float   remask_margin    = 0.0f;   // Layer B5 (ReMDM-inspired): on exact steps, a previously
                                       // committed token whose row now prefers a DIFFERENT token by
                                       // this prob margin gets re-masked (budget-capped). 0 = off.
    int32_t remask_budget    = 2;      // max remasks per step (total per run capped at 16)

    float   early_commit     = 0.0f;   // Layer B2 (Prophet): when EVERY remaining masked position has
                                       // top1-top2 prob gap >= this, commit all and finish. Only fires
                                       // on exact (uncached/warm) steps. 0 = off.

    int32_t window           = 0;      // Layer C1a: contiguous suffix window on the SQUARE (no-kv)
                                       // path - batch ends at max(first_mask + W, last committed
                                       // non-EOG + 1); distant uncommitted masks are not decoded.
                                       // Commits scan the whole batch (no step inflation). 0 = off.

    int32_t gen_initial      = 0;      // Layer C4 in-run canvas growth (square path): start the
                                       // active canvas at n_input + N and grow by +64 while fewer
                                       // than 8 masks remain and no EOG is committed; max_length
                                       // stays the allocation. 0 = off (start at max_length).

    int32_t sub_block        = 0;      // Fast-dLLM v2 block-AR decode: sub-block size for the
                                       // left-to-right commit schedule (0 = 8, the reference).

    float   block_eb         = 0.0f;   // E5c: block-AR alternative accept rule - per step, commit
                                       // masked positions by ascending entropy while the cumulative
                                       // entropy stays <= this bound (DG-style budget; replaces the
                                       // conf_threshold rule). 0 = off.

    bool    block_kv         = false;  // Fast-dLLM v2 block-AR decode (E3): cache committed blocks'
                                       // KV in the model pkv store; forward only the active block

    int32_t kv_anchor        = 3;      // Layer B4: with kv_window in PREFIX mode, always include the
                                       // last N canvas rows in the batch (the "final anchor" - the
                                       // model needs to see that the end exists; streaming-dllm
                                       // sampler.py tail_keep). BLOCK mode unsupported (store-write
                                       // offsets assume a contiguous batch). 0 = off.
    int32_t kv_window        = 0;      // suffix lookahead window beyond the active block (0 = mode default:
                                       // prefix decodes the whole suffix, dual decodes the block only).
                                       // With W: decode/commit only [block_start, block_end + W) - distant
                                       // masks are dropped from the batch entirely (DPad-style)

    float * output_confidences = nullptr;  // [out, optional, size max_length] confidence at commit time per
                                           // position; -1 = prompt/uncommitted/ORIGIN (records no confidence)

    bool *  out_degenerate = nullptr;  // [out, optional] set when threshold decoding aborted because end
                                       // tokens flooded the canvas while masked positions remained

    int32_t max_length = 0;            // Maximum sequence length
};

void diffusion_generate(llama_context *          ctx,
                        const llama_token *      input_tokens,
                        llama_token *            output_tokens,
                        int32_t                  n_input,
                        const diffusion_params & params,
                        int32_t &                n_generated);

// Fast-dLLM v2 (arXiv:2509.26328) block-AR decode: the canvas grows one block_length
// block at a time (semi-autoregressive); within a block, sub_block spans commit left
// to right by confidence threshold (always at least the argmax). Attention is
// block-causal (built into the fast-dllm graph); logits are token-shifted like Dream.
// v1 is UNCACHED: every step re-forwards [0 .. block_end) on the square path.
// Uses: max_length (allocation), block_length (0 = 32), sub_block (0 = 8),
// conf_threshold (0 = 0.9), steps (cap), mask_token_id, top_p/temperature (v1:
// reference semantics - temperature 0 = argmax; sampling ignored otherwise).
void diffusion_generate_block_ar(llama_context *          ctx,
                                 const llama_token *      input_tokens,
                                 llama_token *            output_tokens,
                                 int32_t                  n_input,
                                 const diffusion_params & params,
                                 int32_t &                n_generated);

// Entropy-bound denoiser for block-diffusion canvas models (DiffusionGemma). Unlike the masked path, the
// canvas is random-initialized and non-accepted positions are renoised each step; tokens are accepted by a
// per-position entropy (mutual-information) bound, under a linear temperature schedule, with adaptive
// stopping. Writes the final argmax canvas into output_tokens[n_input .. max_length).
struct diffusion_eb_params {
    int32_t max_denoising_steps  = 48;
    float   t_min                = 0.4f;   // temperature at the last step
    float   t_max                = 0.8f;   // temperature at the first step
    float   entropy_bound        = 0.1f;   // accept lowest-entropy tokens within this MI bound
    int32_t stability_threshold  = 1;      // steps the argmax canvas must hold to count as stable
    float   confidence_threshold = 0.005f; // stop once mean canvas entropy drops below this
    int32_t seed                 = 0;
    int32_t max_length           = 0;      // n_input + canvas_length
    bool    kv_cache             = false;  // prefix-KV-cache the prompt (PREFILL once, decode canvas-only
                                           // per step) instead of re-decoding [prompt|canvas] every step

    diffusion_step_callback_t step_callback           = nullptr;
    void *                    step_callback_user_data = nullptr;
    bool                      visual_mode             = false;

    float * output_confidences = nullptr;  // [out, optional, size max_length] final-step ENTROPY per canvas
                                           // position (lower = more confident); -1 outside the canvas
};

void diffusion_generate_entropy_bound(llama_context *             ctx,
                                      const llama_token *         input_tokens,
                                      llama_token *               output_tokens,
                                      int32_t                     n_input,
                                      const diffusion_eb_params & params,
                                      int32_t &                   n_generated);
