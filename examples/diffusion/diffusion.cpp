#include "diffusion.h"

#include "log.h"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <thread>
#include <utility>
#include <vector>

static float calculate_confidence(const llama_token_data_array & cur_p,
                                  diffusion_algorithm            algorithm,
                                  std::mt19937 &                 rng) {
    switch (algorithm) {
        case DIFFUSION_ALGORITHM_CONFIDENCE_BASED:
            return cur_p.data[cur_p.selected].p;  // Selected token probability

        case DIFFUSION_ALGORITHM_ENTROPY_BASED:
            {
                float       entropy = 0.0f;
                const float epsilon = 1e-10f;
                for (size_t i = 0; i < cur_p.size; i++) {
                    float prob = cur_p.data[i].p;
                    entropy += prob * logf(prob + epsilon);
                }
                return -entropy;  // Higher entropy = lower confidence
            }

        case DIFFUSION_ALGORITHM_MARGIN_BASED:
            return (cur_p.size > 1) ? cur_p.data[0].p - cur_p.data[1].p : cur_p.data[0].p;

        case DIFFUSION_ALGORITHM_RANDOM:
            {
                std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
                return uniform(rng);  // Random confidence
            }

        case DIFFUSION_ALGORITHM_ORIGIN:
            return cur_p.data[cur_p.selected].p;

        default:
            return 0.0f;
    }
}

// Confidence against an absolute threshold must not depend on --temp: undo the
// temperature sharpening with q_i ~ p_i^temp (renormalized). The map is monotonic,
// so rankings are unchanged - only the absolute values move back to the raw scale.
static float calculate_confidence_detempered(
        const float *       probs,
        uint32_t            n,
        uint32_t            sel,
        diffusion_algorithm algorithm,
        float               temp,
        std::mt19937 &      rng) {
    if (algorithm == DIFFUSION_ALGORITHM_RANDOM) {
        return std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
    }

    double sum = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        sum += pow(probs[i], temp);
    }
    if (sum <= 0.0) {
        return 0.0f;
    }

    switch (algorithm) {
        case DIFFUSION_ALGORITHM_ENTROPY_BASED:
            {
                double entropy = 0.0;
                const double epsilon = 1e-10;
                for (uint32_t i = 0; i < n; i++) {
                    const double q = pow(probs[i], temp) / sum;
                    entropy += q * log(q + epsilon);
                }
                return (float) -entropy;
            }

        case DIFFUSION_ALGORITHM_MARGIN_BASED:
            {
                double q1 = 0.0;
                double q2 = 0.0;
                for (uint32_t i = 0; i < n; i++) {
                    const double q = pow(probs[i], temp) / sum;
                    if (q > q1) {
                        q2 = q1;
                        q1 = q;
                    } else if (q > q2) {
                        q2 = q;
                    }
                }
                return (float) (q1 - q2);
            }

        default:  // probability of the selected token
            return (float) (pow(probs[sel], temp) / sum);
    }
}

// Unified transfer count calculation function
static int32_t calculate_transfer_count(int32_t                      step,
                                        int32_t                      total_steps,
                                        int32_t                      remaining_masked,
                                        diffusion_transfer_schedule  schedule,
                                        float                        eps,
                                        const std::vector<int32_t> & num_transfer_tokens = {}) {
    switch (schedule) {
        case DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED:
            {
                float t          = 1.0f - (float) step / total_steps * (1.0f - eps);
                float s          = 1.0f - (float) (step + 1) / total_steps * (1.0f - eps);
                float p_transfer = (step < total_steps - 1) ? (1.0f - s / t) : 1.0f;
                return (int32_t) (remaining_masked * p_transfer);
            }

        case DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED:
            if (!num_transfer_tokens.empty() && step < (int32_t) num_transfer_tokens.size()) {
                return num_transfer_tokens[step];
            }
            return remaining_masked / (total_steps - step);  // Fallback

        default:
            return remaining_masked / (total_steps - step);
    }
}

static void add_gumbel_noise(float * logits, int32_t n_vocab, float temperature, std::mt19937 & rng) {
    if (temperature == 0.0f) {
        return;
    }

    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    for (int32_t i = 0; i < n_vocab; i++) {
        double noise        = uniform(rng);
        // Prevent log(0)
        noise               = std::max(noise, 1e-20);
        double gumbel_noise = std::pow(-std::log(noise), temperature);
        logits[i]           = std::exp(logits[i]) / gumbel_noise;
    }
}

static std::vector<int32_t> get_num_transfer_tokens(int32_t mask_count, int32_t steps) {
    std::vector<int32_t> num_transfer_tokens(steps);

    int32_t base      = mask_count / steps;
    int32_t remainder = mask_count % steps;

    for (int32_t i = 0; i < steps; i++) {
        num_transfer_tokens[i] = base + (i < remainder ? 1 : 0);
    }

    return num_transfer_tokens;
}

void diffusion_generate(llama_context *          ctx,
                        const llama_token *      input_tokens,
                        llama_token *            output_tokens,
                        int32_t                  n_input,
                        const diffusion_params & params,
                        int32_t &                n_generated) {
    n_generated = 0;
    // an infill canvas is allowed to fill the whole sequence (no generation tail)
    if (!ctx || !input_tokens || !output_tokens || n_input <= 0 ||
            (params.infill ? params.max_length < n_input : params.max_length <= n_input)) {
        return;
    }

    if (params.infill) {
        if (params.schedule != DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED) {
            LOG_ERR("%s: infill requires the timestep schedule (--diffusion-eps)\n", __func__);
            return;
        }
        if (std::find(input_tokens, input_tokens + n_input, params.mask_token_id) == input_tokens + n_input) {
            LOG_ERR("%s: infill input contains no mask tokens\n", __func__);
            return;
        }
    }

    const llama_model * model = llama_get_model(ctx);

    // Initialize with input and pad with mask tokens
    std::copy(input_tokens, input_tokens + n_input, output_tokens);
    std::fill(output_tokens + n_input, output_tokens + params.max_length, params.mask_token_id);

    if (params.output_confidences) {
        std::fill(params.output_confidences, params.output_confidences + params.max_length, -1.0f);
    }
    if (params.out_degenerate) {
        *params.out_degenerate = false;
    }

    std::mt19937 rng(params.seed);

    llama_set_causal_attn(ctx, false);

    int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::vector<llama_token_data> candidates(n_vocab);
    std::vector<llama_token_data> conf_candidates;
    conf_candidates.reserve(params.max_length);
    std::vector<float> conf_probs;
    std::vector<int32_t> mask_positions;
    mask_positions.reserve(params.max_length);

    // Setup sampler chain
    struct llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (params.top_k > 0) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(params.top_k));
    }
    if (params.top_p < 1.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(params.top_p, 1));
    }
    if (params.temperature > 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(params.temperature));
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(params.seed));

    struct llama_sampler * dist_sampler = llama_sampler_init_dist(params.seed);

    if (params.conf_threshold > 0.0f &&
            params.algorithm != DIFFUSION_ALGORITHM_CONFIDENCE_BASED &&
            params.algorithm != DIFFUSION_ALGORITHM_MARGIN_BASED) {
        LOG_WRN("diffusion: --diffusion-conf-threshold expects a confidence in [0, 1] - "
                "use the confidence or margin algorithm\n");
    }

    // Backend (GPU) sampling: attach a dedicated chain so that every output row is
    // sampled in the compute graph and only tokens/candidate probs reach the host.
    // Not compatible with CFG (host-side logit blending) or gumbel noise.
    // self-conditioning needs raw logits on the host each step, and the backend chain has no
    // way to suppress the mask token - both fall back to CPU sampling
    bool use_backend = params.backend_sampling && params.cfg_scale <= 0.0f && !params.add_gumbel_noise &&
                       !params.self_conditioning && !params.suppress_mask_token;

    if (params.backend_sampling && !use_backend) {
        LOG_WRN("diffusion: backend sampling is not supported with cfg_scale or gumbel noise, using CPU sampling\n");
    }

    // without top_k the in-graph softmax/cumsum would run over the full vocabulary
    // for every output row
    if (use_backend && params.top_k <= 0) {
        LOG_WRN("diffusion: backend sampling requires --top-k > 0, using CPU sampling\n");
        use_backend = false;
    }

    struct llama_sampler * backend_sampler = nullptr;
    if (use_backend) {
        backend_sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(backend_sampler, llama_sampler_init_top_k(params.top_k));
        if (params.top_p < 1.0f) {
            llama_sampler_chain_add(backend_sampler, llama_sampler_init_top_p(params.top_p, 1));
        }
        if (params.temperature > 0.0f) {
            llama_sampler_chain_add(backend_sampler, llama_sampler_init_temp(params.temperature));
        }
        llama_sampler_chain_add(backend_sampler, llama_sampler_init_dist(params.seed));

        if (!llama_set_sampler(ctx, 0, backend_sampler)) {
            LOG_WRN("diffusion: backend sampler could not be attached, using CPU sampling\n");
            llama_sampler_free(backend_sampler);
            backend_sampler = nullptr;
            use_backend = false;
        } else {
            LOG_INF("diffusion: sampling on the backend\n");
        }
    }

    llama_batch batch = llama_batch_init(params.max_length, 0, 1);
    batch.n_tokens    = params.max_length;

    // EOT-tail shrink (threshold mode): once the canvas ends in a committed run of end
    // tokens, later steps stop feeding that tail through the model - the per-step
    // forward cost tracks the ANSWER length instead of the canvas length. A short EOT
    // anchor is kept visible so remaining positions still see "the end".
    int32_t cur_length = params.max_length;

    // Self-conditioning (DiffusionGemma): cache each step's canvas-row logits and feed them into the next
    // step (canvas = [n_input, max_length)); set_sc is a no-op for other models.
    llama_model *      sc_model = const_cast<llama_model *>(llama_get_model(ctx));
    const int32_t      sc_canvas = params.max_length - n_input;
    std::vector<float> sc_buffer;
    if (params.self_conditioning) {
        sc_buffer.assign((size_t) sc_canvas * n_vocab, 0.0f);
    }

    // Pre-allocate buffers for CFG if needed
    int32_t                  logits_size = n_vocab * params.max_length;
    std::vector<float>       cond_logits_buffer;
    std::vector<llama_token> un_x_buffer;
    if (params.cfg_scale > 0.0f) {
        cond_logits_buffer.resize(logits_size);
        un_x_buffer.resize(params.max_length);
    }

    // For block-based processing
    std::vector<int32_t> num_transfer_tokens;
    int32_t              num_blocks      = 1;
    int32_t              steps_per_block = params.steps;

    if (params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED) {
        GGML_ASSERT(params.max_length % params.block_length == 0);
        num_blocks = params.max_length / params.block_length;
        GGML_ASSERT(params.steps % num_blocks == 0);
        steps_per_block = params.steps / num_blocks;
    }

    std::vector<float> confidence(params.max_length);

    int64_t total_sampling_time = 0;
    int64_t total_time          = 0;
    int64_t time_start          = ggml_time_us();

    // with conf_threshold the schedule finishes early - count the steps actually run
    int32_t n_steps_done = 0;

    auto confidences_abs_for = [](size_t mask_idx, const std::vector<std::pair<float, int32_t>> & confs) -> float {
        for (const auto & c : confs) {
            if ((size_t) c.second == mask_idx) {
                return c.first;
            }
        }
        return -1.0f;
    };

    // Layer A cached decoding (docs/dllms/throughput-plans/01_layer_a.md): per kv-block, one
    // full WARM forward snapshots prefix (or full-canvas) K/V, then steps decode only the
    // suffix (prefix cache) or the block (dual cache) against the store.
    const int32_t kv_blk_sz = params.kv_block > 0 ? params.kv_block : params.kv_prefix;
    const bool    kv_dual   = params.kv_block > 0;
    const bool    kv_on     = kv_blk_sz > 0 && params.conf_threshold > 0.0f && !params.infill &&
                              params.schedule == DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED &&
                              params.cfg_scale <= 0.0f && !params.self_conditioning;
    if (kv_blk_sz > 0 && !kv_on) {
        LOG_WRN("%s: kv_prefix/kv_block requested but unsupported here (needs conf_threshold>0, "
                "timestep schedule, no infill/cfg/self-conditioning) - running uncached\n", __func__);
    }
    int32_t kv_s = n_input;        // current kv block start
    int32_t kv_e = 0;              // current kv block end
    bool    kv_warm_needed = true; // next forward must be a full WARM
    // cached K/V drift as the canvas fills (cosine similarity is only high between
    // ADJACENT steps - reusing a step-0 snapshot for many steps compounds the error and
    // measurably degrades output: the model EOT-floods). Two re-warm triggers: step
    // cadence (kv_rewarm) and commit mass (kv_rewarm_commits - drift tracks canvas
    // CHANGES, so counting commits is the more principled guard).
    const int32_t kv_rewarm = params.kv_rewarm > 0 ? params.kv_rewarm : 1000000000;
    int32_t kv_steps_since_warm   = 0;
    int32_t kv_commits_since_warm = 0;
    int32_t kv_span_hint          = 0;  // B3: next block size detected at the last warm
    int32_t remask_total          = 0;  // B5: per-run remask cap (progress guarantee)

    // Layer C4 in-run canvas growth: the canvas starts small and grows as the frontier
    // approaches it, so the allocation (max_length) no longer sets the per-step cost.
    // Square threshold path only - the kv modes re-derive block geometry from
    // cur_length and the non-threshold schedules pre-compute transfer counts from the
    // initial mask count.
    const bool grow_on = params.gen_initial > 0 && !kv_on && !params.infill &&
                         params.conf_threshold > 0.0f &&
                         params.schedule == DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED &&
                         params.cfg_scale <= 0.0f && !params.self_conditioning;
    if (params.gen_initial > 0 && !grow_on) {
        LOG_WRN("%s: gen_initial requested but unsupported here (needs conf_threshold>0, timestep "
                "schedule, no kv/infill/cfg/self-conditioning) - starting at max_length\n", __func__);
    }
    if (grow_on) {
        cur_length = std::min(params.max_length, n_input + params.gen_initial);
    }
    int32_t n_grows = 0;
    int32_t win_ext_max           = 0;  // C1a instrumentation: max window extension beyond
                                        // frontier+W forced by committed islands (C1b decision)
    int64_t win_rows_saved        = 0;  // C1a instrumentation: total rows pruned

    for (int block_num = 0; block_num < num_blocks; block_num++) {
        int32_t block_start = (params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED) ? n_input + block_num * params.block_length : 0;
        int32_t block_end   = (params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED) ?
                                  std::min(n_input + (block_num + 1) * params.block_length, params.max_length) :
                                  params.max_length;

        // Count masked tokens in current block for block-based processing
        if (params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED) {
            int32_t block_mask_count = 0;
            for (int i = block_start; i < block_end; i++) {
                if (output_tokens[i] == params.mask_token_id) {
                    block_mask_count++;
                }
            }
            num_transfer_tokens = get_num_transfer_tokens(block_mask_count, steps_per_block);
        }

        for (int32_t step = 0; step < steps_per_block; step++) {
            int32_t global_step = block_num * steps_per_block + step;

            if (params.step_callback) {
                if (!params.step_callback(
                        global_step, params.steps, output_tokens, params.max_length, params.step_callback_user_data)) {
                    break;
                }
            }

            // Layer C4: grow the canvas when the frontier nears its end. A committed
            // EOG means the answer is closing - never grow past it; the pre-filled
            // masks beyond cur_length (init above) make growth just a bound raise.
            if (grow_on && cur_length < params.max_length) {
                const llama_vocab * gvocab   = llama_model_get_vocab(model);
                int32_t             n_masked = 0;
                bool                has_eog  = false;
                for (int32_t i = n_input; i < cur_length && !has_eog; i++) {
                    const llama_token t = output_tokens[i];
                    if (t == params.mask_token_id) {
                        n_masked++;
                    } else if (llama_vocab_is_eog(gvocab, t) || t == llama_vocab_pad(gvocab)) {
                        has_eog = true;
                    }
                }
                if (!has_eog && n_masked < 8) {
                    cur_length = std::min(params.max_length, cur_length + 64);
                    n_grows++;
                }
            }

            // Layer A: pick phase + batch window for this step
            int32_t batch_first = 0;
            int32_t batch_last  = cur_length;

            // Layer C1a: contiguous suffix window on the square path - skip decoding
            // distant uncommitted masks. The window always covers the frontier + W AND
            // any committed non-EOG islands beyond it (they are real context; the
            // contiguous rule extends rather than drops - see 03_layer_c.md).
            if (!kv_on && params.window > 0 && params.conf_threshold > 0.0f) {
                int32_t first_mask = -1;
                int32_t last_committed_nontail = n_input - 1;
                const llama_vocab * wvocab = llama_model_get_vocab(model);
                for (int32_t i = n_input; i < cur_length; i++) {
                    const llama_token t = output_tokens[i];
                    if (t == params.mask_token_id) {
                        if (first_mask < 0) {
                            first_mask = i;
                        }
                    } else if (!llama_vocab_is_eog(wvocab, t) && t != llama_vocab_pad(wvocab)) {
                        last_committed_nontail = i;
                    }
                }
                if (first_mask >= 0) {
                    const int32_t frontier_end = std::min(cur_length, first_mask + params.window);
                    batch_last = std::min(cur_length,
                                          std::max(frontier_end, last_committed_nontail + 1));
                    win_ext_max    = std::max(win_ext_max, batch_last - frontier_end);
                    win_rows_saved += cur_length - batch_last;
                }
            }

            if (kv_on) {
                auto block_clean = [&](int32_t s, int32_t e) {
                    for (int32_t i = s; i < e; i++) {
                        if (output_tokens[i] == params.mask_token_id) {
                            return false;
                        }
                    }
                    return true;
                };
                const int32_t kv_blk_eff = (params.kv_span > 0.0f && kv_span_hint > 0)
                                               ? kv_span_hint : kv_blk_sz;
                kv_e = std::min(kv_s + kv_blk_eff, cur_length);
                while (kv_s < cur_length && block_clean(kv_s, kv_e)) {
                    kv_s           = kv_e;
                    kv_e           = std::min(kv_s + kv_blk_eff, cur_length);
                    kv_warm_needed = true;
                }
                if (kv_steps_since_warm >= kv_rewarm) {
                    kv_warm_needed = true;
                }
                if (params.kv_rewarm_commits > 0 && kv_commits_since_warm >= params.kv_rewarm_commits) {
                    kv_warm_needed = true;
                }
                if (kv_s >= cur_length) {
                    break;  // every position committed
                }

                // Dream shift_logits: include one extra leading row so position kv_s gets the
                // logits of kv_s-1 exactly (kv_s >= n_input >= 1, so kv_s-1 always exists)
                const int32_t b0 = params.shift_logits ? kv_s - 1 : kv_s;
                // suffix lookahead window: decode (and commit) only up to kv_e + W;
                // distant masks are dropped from the batch entirely (DPad-style - their
                // attention contribution is negligible and skipping them is pure savings).
                // In dual mode W > 0 turns the block into a sliding lookahead window.
                int32_t kv_batch_end =
                    params.kv_window > 0 ? std::min(cur_length, kv_e + params.kv_window)
                                         : (kv_dual ? kv_e : cur_length);
                // Layer B4 tail anchor, contiguity-safe variant: appending detached
                // anchor rows broke the example's row==batch-index assumption (positions
                // must stay monotonic or commits read the wrong rows - observed as
                // instant garbage + guard abort). Instead the window SNAPS to the end
                // when it gets close, so the final rows enter the batch contiguously.
                if (params.kv_window > 0 && params.kv_anchor > 0 &&
                        cur_length - kv_batch_end <= params.kv_anchor + 8) {
                    kv_batch_end = cur_length;
                }

                if (kv_warm_needed) {
                    llama_diffusion_set_phase(sc_model, /*WARM*/1, kv_dual ? cur_length : b0);
                } else if (!kv_dual) {
                    llama_diffusion_set_phase(sc_model, /*DECODE*/2, b0);
                    batch_first = b0;
                    batch_last  = kv_batch_end;
                } else {
                    llama_diffusion_set_block(sc_model, b0, cur_length);
                    llama_diffusion_set_phase(sc_model, /*BLOCK*/3, 0);
                    batch_first = b0;
                    batch_last  = kv_batch_end;
                }
            }

            // Setup batch
            batch.n_tokens = batch_last - batch_first;
            for (int32_t i = 0; i < batch.n_tokens; i++) {
                batch.token[i]     = output_tokens[batch_first + i];
                batch.pos[i]       = batch_first + i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = 1;
            }


            if (params.self_conditioning) {
                // step 0 has no previous prediction: keep the SC subgraph (stable graph shape) but gate it off
                llama_diffusion_set_sc(sc_model, sc_buffer.data(), global_step == 0 ? 0.0f : 1.0f, 1.0f, true);
            }

            float * logits = nullptr;

            if (params.cfg_scale > 0.0f) {
                int ret = llama_decode(ctx, batch);
                if (ret != 0) {
                    LOG_ERR("Failed to generate conditional");
                    break;
                }
                float * cond_logits_ptr = llama_get_logits(ctx);
                std::memcpy(cond_logits_buffer.data(), cond_logits_ptr, logits_size * sizeof(float));

                // Unconditional generation (mask input)
                std::copy(output_tokens, output_tokens + params.max_length, un_x_buffer.begin());
                for (int32_t i = 0; i < n_input; i++) {
                    un_x_buffer[i] = params.mask_token_id;
                }

                for (int32_t i = 0; i < params.max_length; i++) {
                    batch.token[i] = un_x_buffer[i];
                }
                ret = llama_decode(ctx, batch);
                if (ret != 0) {
                    LOG_ERR("Failed to generate unconditional");
                    break;
                }
                float * uncond_logits = llama_get_logits(ctx);

                // Apply CFG
                for (int32_t i = 0; i < logits_size; i++) {
                    cond_logits_buffer[i] =
                        uncond_logits[i] + (params.cfg_scale + 1.0f) * (cond_logits_buffer[i] - uncond_logits[i]);
                }
                logits = cond_logits_buffer.data();
            } else {
                int ret = llama_decode(ctx, batch);
                if (ret != 0) {
                    LOG_ERR("%s: failed to decode at step %d, ret = %d\n", __func__, global_step, ret);
                    break;
                }

                // verify on the first step that the whole chain actually runs on the
                // backend - a partially offloaded chain produces no sampled tokens
                if (use_backend && global_step == 0 &&
                        llama_get_sampled_token_ith(ctx, 0) == LLAMA_TOKEN_NULL) {
                    LOG_WRN("diffusion: backend sampling not supported by this device, using CPU sampling\n");
                    llama_set_sampler(ctx, 0, nullptr);
                    use_backend = false;

                    // raw logits were not extracted while the sampler was attached - redo
                    ret = llama_decode(ctx, batch);
                    if (ret != 0) {
                        LOG_ERR("%s: failed to decode at step %d, ret = %d\n", __func__, global_step, ret);
                        break;
                    }
                }

                if (!use_backend) {
                    logits = llama_get_logits(ctx);
                }
            }

            if (!use_backend && !logits) {
                LOG_ERR("%s: failed to get logits at step %d\n", __func__, global_step);
                break;
            }

            n_steps_done++;

            if (kv_on) {
                if (kv_warm_needed) {
                    kv_warm_needed        = false;  // the store is fresh
                    kv_steps_since_warm   = 0;
                    kv_commits_since_warm = 0;      // commits from here on stale the store
                } else {
                    kv_steps_since_warm++;
                }
            }

            if (params.self_conditioning) {
                std::memcpy(sc_buffer.data(), logits + (size_t) n_input * n_vocab,
                            (size_t) sc_canvas * n_vocab * sizeof(float));
            }

            // batch index whose output row predicts the token at pos (batch_first > 0 in
            // Layer A cached phases; the batch then starts at that absolute position)
            auto get_row_for_pos = [&](int32_t pos) -> int32_t {
                return (params.shift_logits ? std::max(pos - 1, 0) : pos) - batch_first;
            };

            auto get_logits_for_pos = [&](int32_t pos) -> const float * {
                return logits + (size_t) get_row_for_pos(pos) * n_vocab;
            };

            int64_t time_start_sampling = ggml_time_us();

            mask_positions.clear();
            // commits are possible wherever we HAVE logits: the whole batch window.
            // (restricting commits to the block inflated 15-step runs to the 128-step
            // cap; whole-window commits keep baseline step dynamics)
            const int32_t scan_lo = kv_on ? kv_s : 0;
            const int32_t scan_hi = std::min(batch_last, cur_length);
            for (int32_t i = scan_lo; i < scan_hi; i++) {
                if (output_tokens[i] == params.mask_token_id) {
                    // For block-based, only consider current block
                    if (params.schedule != DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED || (i >= block_start && i < block_end)) {
                        mask_positions.push_back(i);
                    }
                }
            }

            if (mask_positions.empty()) {
                // C1a: an empty WINDOW scan does not mean done - masks beyond batch_last
                // exist when the window just completed; the next step recomputes the
                // frontier and the window SLIDES forward
                if (!kv_on && params.window > 0 && batch_last < cur_length) {
                    bool more = false;
                    for (int32_t i = batch_last; i < cur_length; i++) {
                        if (output_tokens[i] == params.mask_token_id) {
                            more = true;
                            break;
                        }
                    }
                    if (more) {
                        continue;
                    }
                }
                break;
            }

            if (params.add_gumbel_noise && params.temperature > 0.0f) {
                add_gumbel_noise(logits, n_vocab, params.temperature, rng);
            }

            if (params.algorithm == DIFFUSION_ALGORITHM_ORIGIN) {
                int32_t transfer_count = calculate_transfer_count(
                    step, steps_per_block, mask_positions.size(), params.schedule, params.eps, num_transfer_tokens);
                float p_transfer = (float) transfer_count / mask_positions.size();

                for (int32_t pos : mask_positions) {
                    if (std::uniform_real_distribution<float>(0.0f, 1.0f)(rng) < p_transfer) {
                        if (use_backend) {
                            const llama_token tok = llama_get_sampled_token_ith(ctx, get_row_for_pos(pos));
                            GGML_ASSERT(tok != LLAMA_TOKEN_NULL);
                            output_tokens[pos] = tok;
                            continue;
                        }

                        const float * pos_logits = get_logits_for_pos(pos);
                        for (int32_t token_id = 0; token_id < n_vocab; token_id++) {
                            candidates[token_id].id    = token_id;
                            candidates[token_id].logit = pos_logits[token_id];
                            candidates[token_id].p     = 0.0f;
                        }
                        if (params.suppress_mask_token) {
                            candidates[params.mask_token_id].logit = -INFINITY;  // never reveal as mask
                        }

                        llama_token_data_array cur_p = {
                            candidates.data(),
                            (size_t) n_vocab,
                            -1,
                            false,
                        };

                        llama_sampler_apply(sampler, &cur_p);
                        output_tokens[pos] = cur_p.data[cur_p.selected].id;
                    }
                }
            } else {
                std::vector<std::pair<float, int32_t>> confidences;
                std::vector<llama_token>               sampled_tokens(mask_positions.size());

                if (use_backend) {
                    for (size_t i = 0; i < mask_positions.size(); i++) {
                        const int32_t pos = mask_positions[i];
                        const int32_t row = get_row_for_pos(pos);

                        const llama_token tok = llama_get_sampled_token_ith(ctx, row);
                        GGML_ASSERT(tok != LLAMA_TOKEN_NULL);

                        float conf = 0.0f;

                        if (params.algorithm == DIFFUSION_ALGORITHM_RANDOM) {
                            conf = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
                        } else {
                            const float *       probs   = llama_get_sampled_probs_ith(ctx, row);
                            const uint32_t      n_probs = llama_get_sampled_probs_count_ith(ctx, row);
                            const llama_token * cands   = llama_get_sampled_candidates_ith(ctx, row);
                            const uint32_t      n_cands = llama_get_sampled_candidates_count_ith(ctx, row);
                            GGML_ASSERT(probs != nullptr && cands != nullptr && n_probs == n_cands);

                            // candidates are not sorted - scan for the sampled token
                            uint32_t sel = 0;
                            for (uint32_t j = 0; j < n_cands; j++) {
                                if (cands[j] == tok) {
                                    sel = j;
                                    break;
                                }
                            }

                            // undo temperature sharpening only when comparing against an
                            // absolute threshold - for schedule ranking it changes nothing
                            const float temp_undo =
                                (params.conf_threshold > 0.0f && params.temperature > 0.0f) ? params.temperature : 1.0f;

                            conf = calculate_confidence_detempered(probs, n_probs, sel, params.algorithm, temp_undo, rng);
                        }

                        sampled_tokens[i] = tok;
                        confidences.emplace_back(conf, (int32_t) i);
                    }
                } else {
                    for (size_t i = 0; i < mask_positions.size(); i++) {
                        int32_t       pos        = mask_positions[i];
                        const float * pos_logits = get_logits_for_pos(pos);

                        for (int32_t token_id = 0; token_id < n_vocab; token_id++) {
                            candidates[token_id].logit = pos_logits[token_id];
                            candidates[token_id].p     = 0.0f;
                            candidates[token_id].id    = token_id;
                        }

                        if (params.suppress_mask_token) {
                            candidates[params.mask_token_id].logit = -INFINITY;  // never reveal as mask
                        }

                        llama_token_data_array cur_p = {
                            candidates.data(),
                            candidates.size(),
                            -1,
                            false,
                        };

                        llama_sampler_apply(sampler, &cur_p);
                        llama_token sampled_token = cur_p.data[cur_p.selected].id;

                        float conf;
                        if (params.conf_threshold > 0.0f && params.temperature > 0.0f &&
                                params.algorithm != DIFFUSION_ALGORITHM_RANDOM) {
                            // absolute threshold semantics - undo temperature sharpening
                            conf_probs.resize(cur_p.size);
                            for (size_t j = 0; j < cur_p.size; j++) {
                                conf_probs[j] = cur_p.data[j].p;
                            }
                            conf = calculate_confidence_detempered(conf_probs.data(), (uint32_t) cur_p.size,
                                                                   (uint32_t) cur_p.selected, params.algorithm,
                                                                   params.temperature, rng);
                        } else {
                            conf = calculate_confidence(cur_p, params.algorithm, rng);
                        }

                        sampled_tokens[i] = sampled_token;
                        confidences.emplace_back(conf, i);
                    }
                }

                if (params.conf_threshold > 0.0f) {
                    // parallel decoding: commit every position whose confidence clears the
                    // threshold; always commit at least the most confident one so that
                    // every step makes progress (ref: Fast-dLLM, arXiv:2505.22618)
                    size_t  best        = 0;
                    int32_t n_committed = 0;

                    // Layer B1 adaptive threshold, BLOCK-SCOPED (refuted lesson: global
                    // decay collapses quality - bench 13/45; the reference decays within a
                    // 32-token block that RESETS). Two zones: positions inside the active
                    // window [w_lo, w_hi) get the decayed tau; everything beyond keeps the
                    // base threshold. The window tracks the FIRST remaining mask.
                    float   tau_window = params.conf_threshold;
                    int32_t tau_w_hi   = INT32_MAX;
                    if (params.tau_alpha > 0.0f && !mask_positions.empty()) {
                        const int32_t w_lo = kv_on ? kv_s : mask_positions.front();
                        const int32_t w_hi = kv_on ? std::min(batch_last, cur_length)
                                                   : std::min(w_lo + 32, cur_length);
                        int32_t masks_in_w = 0;
                        for (int32_t mp : mask_positions) {
                            if (mp >= w_lo && mp < w_hi) {
                                masks_in_w++;
                            }
                        }
                        const float r_mask = w_hi > w_lo ? (float) masks_in_w / (float) (w_hi - w_lo) : 1.0f;
                        tau_window = params.conf_threshold * (1.0f - params.tau_alpha * (1.0f - r_mask));
                        if (params.tau_floor > 0.0f) {
                            tau_window = std::max(tau_window, params.tau_floor);
                        }
                        tau_w_hi = w_hi;
                    }

                    // Layer A EOG quarantine: stale cached K/V systematically INFLATE
                    // end-token confidence (low-information context -> EOT pathology) -
                    // observed as truncated outputs. Cached steps may commit text freely
                    // but never EOG; warm steps (exact full forwards, every kv_rewarm
                    // steps) confirm the real end. Text speed kept, endings exact.
                    const llama_vocab * cvocab = llama_model_get_vocab(model);
                    const bool kv_eog_allowed  = !kv_on || kv_steps_since_warm == 0;
                    auto eog_blocked = [&](llama_token t) {
                        return !kv_eog_allowed &&
                               (llama_vocab_is_eog(cvocab, t) || t == llama_vocab_pad(cvocab));
                    };

                    for (size_t i = 0; i < confidences.size(); i++) {
                        if (confidences[i].first > confidences[best].first) {
                            best = i;
                        }
                        const float tau_for_pos =
                            mask_positions[confidences[i].second] < tau_w_hi ? tau_window
                                                                             : params.conf_threshold;
                        if (confidences[i].first >= tau_for_pos) {
                            const int32_t mask_idx = confidences[i].second;
                            if (eog_blocked(sampled_tokens[mask_idx])) {
                                continue;
                            }
                            output_tokens[mask_positions[mask_idx]] = sampled_tokens[mask_idx];
                            if (params.output_confidences) {
                                params.output_confidences[mask_positions[mask_idx]] = confidences[i].first;
                            }
                            n_committed++;
                            kv_commits_since_warm++;
                        }
                    }

                    if (n_committed == 0) {
                        const int32_t mask_idx = confidences[best].second;
                        if (!eog_blocked(sampled_tokens[mask_idx])) {
                            output_tokens[mask_positions[mask_idx]] = sampled_tokens[mask_idx];
                            if (params.output_confidences) {
                                params.output_confidences[mask_positions[mask_idx]] = confidences[best].first;
                            }
                            kv_commits_since_warm++;
                        } else {
                            // a cached step wants to commit ONLY end tokens: the answer is
                            // likely complete - verify with an exact warm forward NOW
                            // instead of idling until the next scheduled re-warm
                            kv_warm_needed = true;
                        }
                    }

                    // Layer B3 (SlowFast span detection): at warm steps, measure the
                    // contiguous confident span from the block start - it becomes the
                    // NEXT block's size (dynamic block sizing; clamp keeps warms amortized)
                    if (kv_on && params.kv_span > 0.0f && kv_steps_since_warm == 0) {
                        std::vector<float> conf_of_mask(mask_positions.size(), -1.0f);
                        for (const auto & c : confidences) {
                            conf_of_mask[c.second] = c.first;
                        }
                        int32_t span  = 0;
                        size_t  m_idx = 0;
                        for (int32_t pos = kv_s; pos < cur_length; pos++) {
                            while (m_idx < mask_positions.size() && mask_positions[m_idx] < pos) {
                                m_idx++;
                            }
                            const bool is_masked = m_idx < mask_positions.size() && mask_positions[m_idx] == pos;
                            if (is_masked && conf_of_mask[m_idx] < params.kv_span) {
                                break;  // first unconfident masked position ends the span
                            }
                            span++;
                        }
                        kv_span_hint = std::max(8, std::min(64, span));
                    }

                    // Layer B2 (Prophet, arXiv:2508.19982): when every remaining masked
                    // position's top1-top2 prob gap clears the bar, the outcome is already
                    // decided - commit everything and finish. Exact steps only (cached
                    // steps route through a warm so drift never finalizes the canvas).
                    if (params.early_commit > 0.0f && use_backend &&
                            (!kv_on || kv_steps_since_warm == 0)) {
                        float min_gap = 1.0f;
                        for (int32_t pos : mask_positions) {
                            if (output_tokens[pos] != params.mask_token_id) {
                                continue;  // committed earlier in this very step
                            }
                            const int32_t row = get_row_for_pos(pos);
                            const float * pr  = llama_get_sampled_probs_ith(ctx, row);
                            const size_t  n   = llama_get_sampled_probs_count_ith(ctx, row);
                            const float gap = (pr && n >= 2) ? pr[0] - pr[1] : 0.0f;
                            min_gap = std::min(min_gap, gap);
                            if (min_gap < params.early_commit) {
                                break;
                            }
                        }
                        if (min_gap >= params.early_commit) {
                            for (size_t i = 0; i < mask_positions.size(); i++) {
                                const int32_t pos = mask_positions[i];
                                if (output_tokens[pos] == params.mask_token_id) {
                                    output_tokens[pos] = sampled_tokens[i];
                                    if (params.output_confidences) {
                                        params.output_confidences[pos] = confidences_abs_for(i, confidences);
                                    }
                                }
                            }
                            break;  // generation complete
                        }
                    }

                    // Layer B5 self-correction (ReMDM-inspired, budget-capped): a token
                    // committed in an EARLIER step whose row now prefers a different token
                    // by a clear margin was probably a bad early commit - re-mask it and
                    // let later steps redo it. Exact steps only; positions masked at THIS
                    // step's start are exempt (their commits reflect current logits);
                    // EOG/pad never remasked (the tail must stay terminal).
                    if (params.remask_margin > 0.0f && use_backend && remask_total < 16 &&
                            (!kv_on || kv_steps_since_warm == 0)) {
                        const llama_vocab * rvocab = llama_model_get_vocab(model);
                        const int32_t r_lo = kv_on ? kv_s : n_input;
                        const int32_t r_hi = kv_on ? std::min(batch_last, cur_length) : cur_length;
                        int32_t budget = params.remask_budget;
                        for (int32_t pos = r_lo; pos < r_hi && budget > 0 && remask_total < 16; pos++) {
                            const llama_token cur_tok = output_tokens[pos];
                            if (cur_tok == params.mask_token_id ||
                                    llama_vocab_is_eog(rvocab, cur_tok) || cur_tok == llama_vocab_pad(rvocab)) {
                                continue;
                            }
                            if (std::binary_search(mask_positions.begin(), mask_positions.end(), pos)) {
                                continue;  // committed this very step
                            }
                            const int32_t row = get_row_for_pos(pos);
                            if (row < 0 || row >= batch.n_tokens) {
                                continue;
                            }
                            const llama_token * cands = llama_get_sampled_candidates_ith(ctx, row);
                            const float *       pr    = llama_get_sampled_probs_ith(ctx, row);
                            const size_t        n     = llama_get_sampled_probs_count_ith(ctx, row);
                            if (!cands || !pr || n < 1 || cands[0] == cur_tok) {
                                continue;
                            }
                            float p_committed = 0.0f;
                            for (size_t k = 0; k < n; k++) {
                                if (cands[k] == cur_tok) {
                                    p_committed = pr[k];
                                    break;
                                }
                            }
                            if (pr[0] - p_committed >= params.remask_margin) {
                                output_tokens[pos] = params.mask_token_id;
                                if (params.output_confidences) {
                                    params.output_confidences[pos] = -1.0f;
                                }
                                budget--;
                                remask_total++;
                            }
                        }
                    }

                    // degeneracy guard: with a whole-sequence threshold the model's most confident early
                    // predictions can be the end-of-text padding tail; if end tokens flood the committed
                    // region while real positions are still masked, the draft is unrecoverable - abort
                    {
                        const llama_vocab * gvocab = llama_model_get_vocab(model);

                        // GENERAL criterion (third revision - the suffix-tail exclusions
                        // false-positived under kv blocks AND C1a windows, because every
                        // reduced-decode mode grows its EOT tail incrementally): an EOG
                        // token is "scattered" (degenerate signal) only when committed
                        // TEXT exists at a LATER position; an EOG run bordering masks or
                        // undecoded space is a tail in progress, in any decode mode.
                        const int32_t guard_end = kv_on ? kv_e : std::min(batch_last, cur_length);

                        int32_t n_committed_total = 0;
                        int32_t n_committed_eog   = 0;  // scattered only
                        int32_t n_masked          = 0;
                        bool    seen_text_after   = false;
                        for (int32_t i = guard_end - 1; i >= n_input; i--) {
                            const llama_token t = output_tokens[i];
                            if (t == params.mask_token_id) {
                                n_masked++;
                                continue;
                            }
                            n_committed_total++;
                            if (llama_vocab_is_eog(gvocab, t) || t == llama_vocab_pad(gvocab)) {
                                if (seen_text_after) {
                                    n_committed_eog++;
                                }
                            } else {
                                seen_text_after = true;
                            }
                        }
                        const int32_t n_gen = guard_end - n_input;
                        if (n_committed_total > 8 &&
                                n_committed_eog > 0.9f * n_committed_total &&
                                n_masked        > 0.2f * n_gen) {
                            LOG_WRN("%s: aborting degenerate draft at step %d (end tokens %d/%d committed, %d/%d still masked)\n",
                                    __func__, global_step, n_committed_eog, n_committed_total, n_masked, n_gen);
                            if (params.out_degenerate) {
                                *params.out_degenerate = true;
                            }
                            break;
                        }

                        // EOT-tail shrink (see cur_length above); positions >= tail are
                        // committed end tokens, so no mask can be beyond the new bound.
                        // Layer A composition: prefix mode shrinks freely (the cached prefix
                        // [0..P) is untouched by a suffix-end change; the graph just rebuilds
                        // once); dual mode shrinks only when the block is clean (the next step
                        // re-warms, refreshing the store at the new geometry)
                        bool kv_allow_shrink = true;
                        if (kv_on && kv_dual) {
                            for (int32_t i = kv_s; i < kv_e; i++) {
                                if (output_tokens[i] == params.mask_token_id) {
                                    kv_allow_shrink = false;
                                    break;
                                }
                            }
                        }
                        if (params.cfg_scale <= 0.0f && kv_allow_shrink) {
                            int32_t tail = cur_length;
                            while (tail > n_input) {
                                const llama_token t = output_tokens[tail - 1];
                                const bool committed_eot = t != params.mask_token_id &&
                                    (llama_vocab_is_eog(gvocab, t) || t == llama_vocab_pad(gvocab));
                                if (!committed_eot) {
                                    break;
                                }
                                tail--;
                            }
                            // clamp INTO [.., params.max_length]: for infill n_input ==
                            // max_length and an unclamped n_input+1 floor would grow the
                            // batch past its allocation (heap overflow, crashed the server)
                            cur_length = std::min(params.max_length,
                                                  std::max(n_input + 1, std::min(cur_length, tail + 4)));
                        }

                        // Layer A EOS early-exit (catalog C2, REQUIRED for block mode): a fully
                        // committed block ending in an EOG run means the answer is complete -
                        // filling the remaining blocks deterministically saves one warm forward
                        // PER REMAINING BLOCK (a short answer on a long canvas would otherwise
                        // EOT-fill block by block)
                        if (kv_on) {
                            bool block_done = true;
                            for (int32_t i = kv_s; i < kv_e; i++) {
                                if (output_tokens[i] == params.mask_token_id) {
                                    block_done = false;
                                    break;
                                }
                            }
                            if (block_done && kv_e < cur_length) {
                                int32_t     run = 0;
                                llama_token eog_tok = LLAMA_TOKEN_NULL;
                                for (int32_t i = kv_e - 1; i >= kv_s; i--) {
                                    const llama_token t = output_tokens[i];
                                    if (llama_vocab_is_eog(gvocab, t) || t == llama_vocab_pad(gvocab)) {
                                        run++;
                                        eog_tok = t;
                                    } else {
                                        break;
                                    }
                                }
                                if (run >= 2) {
                                    for (int32_t i = kv_e; i < params.max_length; i++) {
                                        output_tokens[i] = eog_tok;
                                    }
                                    break;  // generation complete
                                }
                            }
                        }
                    }
                } else {
                    int32_t transfer_count = calculate_transfer_count(
                        step, steps_per_block, mask_positions.size(), params.schedule, params.eps, num_transfer_tokens);

                    if (transfer_count > 0) {
                        if (params.alg_temp == 0.0f) {
                            std::partial_sort(confidences.begin(),
                                              confidences.begin() + std::min(transfer_count, (int32_t) confidences.size()),
                                              confidences.end(),
                                              [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
                                                  if (a.first != b.first) {
                                                      return a.first > b.first;
                                                  }
                                                  return a.second < b.second;
                                              });

                            for (int32_t i = 0; i < std::min(transfer_count, (int32_t) confidences.size()); i++) {
                                int32_t mask_idx   = confidences[i].second;
                                int32_t pos        = mask_positions[mask_idx];
                                output_tokens[pos] = sampled_tokens[mask_idx];
                                if (params.output_confidences) {
                                    params.output_confidences[pos] = confidences[i].first;
                                }
                            }
                        } else {
                            conf_candidates.clear();
                            for (size_t i = 0; i < confidences.size(); i++) {
                                float conf_logit = confidences[i].first / params.alg_temp;
                                conf_candidates.emplace_back(llama_token_data{ (int32_t) i, conf_logit, 0.0f });
                            }

                            llama_token_data_array conf_array = {
                                conf_candidates.data(),
                                conf_candidates.size(),
                                -1,
                                false,
                            };

                            for (int32_t i = 0; i < std::min(transfer_count, (int32_t) confidences.size()); i++) {
                                llama_sampler_apply(dist_sampler, &conf_array);
                                int32_t selected_idx = conf_array.selected;
                                int32_t mask_idx     = selected_idx;
                                int32_t pos          = mask_positions[mask_idx];
                                output_tokens[pos]   = sampled_tokens[mask_idx];
                                if (params.output_confidences) {
                                    params.output_confidences[pos] = confidences[mask_idx].first;
                                }

                                conf_candidates[selected_idx].p = 0.0f;
                                conf_array.selected             = -1;
                            }
                        }
                    }
                }
            }

            int64_t time_end_sampling = ggml_time_us();
            total_sampling_time += time_end_sampling - time_start_sampling;
        }
    }

    if (kv_on) {
        llama_diffusion_set_phase(sc_model, /*UNIFIED*/0, 0);  // restore for later requests
    }

    if (params.window > 0 && win_rows_saved > 0) {
        LOG_INF("diffusion window: %lld rows pruned total, max island extension %d (C1b decision metric)\n",
                (long long) win_rows_saved, win_ext_max);
    }

    if (grow_on) {
        LOG_INF("diffusion growth: started at %d, grew %d time(s), final active length %d (alloc %d)\n",
                std::min(params.max_length, n_input + params.gen_initial), n_grows, cur_length, params.max_length);
    }

    int64_t time_end = ggml_time_us();
    total_time += time_end - time_start;

    n_steps_done = std::max(n_steps_done, 1);

    LOG_INF("\ntotal time: %0.2fms, steps: %d, time per step: %0.2fms, sampling time per step: %0.2fms\n",
            total_time / 1000.0,
            n_steps_done,
            total_time / 1000.0 / n_steps_done,
            total_sampling_time / 1000.0 / n_steps_done);

    llama_batch_free(batch);
    llama_sampler_free(sampler);
    llama_sampler_free(dist_sampler);

    if (backend_sampler) {
        llama_set_sampler(ctx, 0, nullptr);
        llama_sampler_free(backend_sampler);
    }

    n_generated = params.max_length;
}

// Fast-dLLM v2 block-AR decode (see diffusion.h). Reference: modeling.py generate()
// of Efficient-Large-Model/Fast_dLLM_v2_1.5B - semantics replicated exactly, minus
// the KV caches (v1 re-forwards [0..block_end) each step; the fast-dllm graph builds
// the block-causal mask from positions, so a full square forward equals their
// cached block forward).
void diffusion_generate_block_ar(llama_context *          ctx,
                                 const llama_token *      input_tokens,
                                 llama_token *            output_tokens,
                                 int32_t                  n_input,
                                 const diffusion_params & params,
                                 int32_t &                n_generated) {
    n_generated = 0;
    if (!ctx || !input_tokens || !output_tokens || n_input <= 0 || params.max_length <= n_input) {
        return;
    }

    const int32_t bd  = params.block_length   > 0    ? params.block_length   : 32;
    const int32_t sb  = params.sub_block      > 0    ? params.sub_block      : 8;
    const float   thr = params.conf_threshold > 0.0f ? params.conf_threshold : 0.9f;
    const int32_t steps_cap = params.steps > 0 ? params.steps : 256;
    const int32_t L_max = params.max_length;

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    std::copy(input_tokens, input_tokens + n_input, output_tokens);

    llama_set_causal_attn(ctx, false);

    llama_batch batch = llama_batch_init(L_max, 0, 1);

    // E3 cached mode (docs/dllms/throughput-plans/05_layer_e.md): committed blocks'
    // K/V live in the model pkv store; each forward is only the active block instead
    // of the whole [committed | block] square. Committed content is final, so the
    // cached rows are exact - no rewarm machinery. The finalize WARM of each block
    // doubles as the AR step (its last row predicts the next block's first token),
    // which the uncached path pays a full square forward for.
    const bool    kv_on     = params.block_kv;
    llama_model * model_mut = const_cast<llama_model *>(model);
    int32_t       kv_P      = 0;        // rows committed to the store
    bool          ar_valid  = false;    // last WARM's rows hold the AR step; until the next decode
    int32_t       ar_base   = 0;        // position of the last WARM's row 0
    const int32_t store_cap = ((L_max + bd - 1) / bd) * bd;
    if (kv_on) {
        llama_diffusion_set_block(model_mut, 0, store_cap);  // allocate the store once
    }

    // E4 backend sampling (05_layer_e.md): a top_k -> dist chain delivers the top-K
    // candidates + PLAIN softmax probs per row (no temp/top_p samplers - the host
    // replicates the reference nucleus sampling over the K candidates, so the commit
    // confidence is the plain prob by construction). With the sampler attached the
    // full-vocab logits D2H is skipped entirely (needs_raw_logits false).
    bool use_backend = params.backend_sampling;
    if (use_backend && params.top_k <= 0) {
        LOG_WRN("block-ar: backend sampling requires --top-k > 0, using CPU sampling\n");
        use_backend = false;
    }
    struct llama_sampler * backend_sampler = nullptr;
    if (use_backend) {
        backend_sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(backend_sampler, llama_sampler_init_top_k(params.top_k));
        llama_sampler_chain_add(backend_sampler, llama_sampler_init_dist(params.seed));
        if (!llama_set_sampler(ctx, 0, backend_sampler)) {
            LOG_WRN("block-ar: backend sampler could not be attached, using CPU sampling\n");
            llama_sampler_free(backend_sampler);
            backend_sampler = nullptr;
            use_backend     = false;
        } else {
            LOG_INF("block-ar: sampling on the backend\n");
        }
    }

    const float * fwd_logits   = nullptr;  // CPU sampling path only
    bool          first_decode = true;

    // forward rows [from..to) at absolute positions (row i predicts pos from+i+1)
    auto forward_range = [&](int32_t from, int32_t to) -> bool {
        batch.n_tokens = to - from;
        for (int32_t i = from; i < to; i++) {
            batch.token[i - from]     = output_tokens[i];
            batch.pos[i - from]       = i;
            batch.n_seq_id[i - from]  = 1;
            batch.seq_id[i - from][0] = 0;
            batch.logits[i - from]    = 1;
        }
        if (llama_decode(ctx, batch) != 0) {
            return false;
        }
        // first decode verifies the chain actually runs on the backend (a partially
        // offloaded chain produces no sampled tokens) - masked-path precedent
        if (use_backend && first_decode && llama_get_sampled_token_ith(ctx, 0) == LLAMA_TOKEN_NULL) {
            LOG_WRN("block-ar: backend sampling not supported by this device, using CPU sampling\n");
            llama_set_sampler(ctx, 0, nullptr);
            use_backend = false;
            if (llama_decode(ctx, batch) != 0) {  // raw logits were skipped - redo
                return false;
            }
        }
        first_decode = false;
        if (!use_backend) {
            fwd_logits = llama_get_logits(ctx);
            return fwd_logits != nullptr;
        }
        return true;
    };

    // uncached square: rows [0..len)
    auto forward = [&](int32_t len) -> bool { return forward_range(0, len); };

    // reference sample_with_top_p: temperature 0 = argmax + plain softmax prob;
    // else nucleus sampling at temperature, confidence = renormalized prob of the
    // sampled token. Seeded - draft retries NEED diversity (repairs are impossible
    // for block-AR models, so redrafting is the only recovery path).
    std::mt19937 rng(params.seed);
    const float  temp  = params.temperature;
    const float  top_p = params.top_p > 0.0f && params.top_p < 1.0f ? params.top_p : 0.95f;

    std::vector<std::pair<float, int32_t>> cand;
    cand.reserve(4096);

    struct eb_cand {
        float       h;
        int32_t     pos;
        llama_token tok;
    };
    std::vector<eb_cand> eb_pend;

    // sample the token for pos from the LAST forward (rows based at `base` = position
    // of row 0: 0 for the uncached square, blk_start for cached block forwards),
    // returning the plain softmax prob of the sampled token as the commit confidence;
    // h_out (optional) = entropy of the plain distribution (E5c accept rule)
    auto predict = [&](int32_t pos, int32_t base, float & p_out, float * h_out = nullptr) -> llama_token {
        const int32_t r = pos - 1 - base;
        if (use_backend) {
            const float *       probs = llama_get_sampled_probs_ith(ctx, r);
            const uint32_t      n_pr  = llama_get_sampled_probs_count_ith(ctx, r);
            const llama_token * cands = llama_get_sampled_candidates_ith(ctx, r);
            const uint32_t      n_cd  = llama_get_sampled_candidates_count_ith(ctx, r);
            GGML_ASSERT(probs != nullptr && cands != nullptr && n_pr == n_cd && n_pr > 0);

            uint32_t am = 0;
            for (uint32_t j = 1; j < n_pr; j++) {
                if (probs[j] > probs[am]) {
                    am = j;
                }
            }
            if (h_out) {
                double h = 0.0;
                for (uint32_t j = 0; j < n_pr; j++) {
                    if (probs[j] > 0.0f) {
                        h -= (double) probs[j] * log((double) probs[j]);
                    }
                }
                *h_out = (float) h;
            }
            if (temp > 0.0f) {
                // tempered q_j ~ p_j^(1/temp) over the candidates; nucleus top_p over
                // q; confidence stays the PLAIN prob of the sampled candidate
                cand.clear();
                double Z_temp = 0.0;
                for (uint32_t j = 0; j < n_pr; j++) {
                    const double q = pow((double) probs[j], 1.0 / temp);
                    Z_temp += q;
                    cand.emplace_back((float) q, (int32_t) j);
                }
                if (Z_temp > 0.0) {
                    std::sort(cand.begin(), cand.end(), std::greater<>());
                    double cum    = 0.0;
                    size_t n_keep = 0;
                    while (n_keep < cand.size() && cum < top_p * Z_temp) {
                        cum += cand[n_keep++].first;
                    }
                    std::uniform_real_distribution<double> uni(0.0, cum);
                    double target = uni(rng), acc = 0.0;
                    size_t sel = 0;
                    for (size_t j = 0; j < n_keep; j++) {
                        acc += cand[j].first;
                        if (acc >= target) {
                            sel = j;
                            break;
                        }
                    }
                    const int32_t jc = cand[sel].second;
                    p_out            = probs[jc];
                    return cands[jc];
                }
                // tempered mass underflowed - argmax fallback
            }
            p_out = probs[am];
            return cands[am];
        }

        const float * row = fwd_logits + (size_t) r * n_vocab;
        int32_t am = 0;
        for (int32_t v = 1; v < n_vocab; v++) {
            if (row[v] > row[am]) {
                am = v;
            }
        }
        if (temp <= 0.0f) {
            double Z  = 0.0;
            double S1 = 0.0;  // sum e^x * x, for H = ln Z - S1/Z
            for (int32_t v = 0; v < n_vocab; v++) {
                const double x = (double) row[v] - row[am];
                const double e = exp(x);
                Z  += e;
                S1 += e * x;
            }
            if (h_out) {
                *h_out = (float) (log(Z) - S1 / Z);
            }
            p_out = (float) (1.0 / Z);
            return am;
        }
        // sample at temperature; the COMMIT confidence is the PLAIN softmax prob of
        // the sampled token (Layer B de-temper lesson: the threshold is an absolute
        // scale - temperature sharpening must not leak into it, or every sub-block
        // position clears 0.9 at once and simultaneous shifted commits emit
        // adjacent-duplicate garbage; observed before this fix)
        double Z_plain = 0.0;
        double S1      = 0.0;
        double Z_temp  = 0.0;
        for (int32_t v = 0; v < n_vocab; v++) {
            const double x = (double) row[v] - row[am];
            const double e = exp(x);
            Z_plain += e;
            S1      += e * x;
            Z_temp  += exp(x / temp);
        }
        if (h_out) {
            *h_out = (float) (log(Z_plain) - S1 / Z_plain);
        }
        cand.clear();
        for (int32_t v = 0; v < n_vocab; v++) {
            const float p = (float) (exp(((double) row[v] - row[am]) / temp) / Z_temp);
            if (p > 1e-8f) {
                cand.emplace_back(p, v);
            }
        }
        std::sort(cand.begin(), cand.end(), std::greater<>());
        double cum = 0.0;
        size_t n_keep = 0;
        while (n_keep < cand.size() && cum < top_p) {
            cum += cand[n_keep++].first;
        }
        std::uniform_real_distribution<double> uni(0.0, cum);
        double target = uni(rng), acc = 0.0;
        size_t sel = 0;
        for (size_t j = 0; j < n_keep; j++) {
            acc += cand[j].first;
            if (acc >= target) {
                sel = j;
                break;
            }
        }
        const int32_t v_sel = cand[sel].second;
        p_out = (float) (exp((double) row[v_sel] - row[am]) / Z_plain);
        return v_sel;
    };

    const int64_t t0      = ggml_time_us();
    int32_t       cur     = n_input;
    int32_t       n_steps = 0;
    bool          done    = false;

    LOG_INF("block-ar: n_input=%d bd=%d sub=%d kv=%d thr=%.2f L_max=%d\n",
            n_input, bd, sb, kv_on ? 1 : 0, thr, L_max);

    // cached mode: prefill the complete prompt blocks into the store, one WARM per
    // 32-aligned block (block-causality across the prompt is enforced by the feed
    // order; the all-allow mask only ever sees [earlier blocks | own block])
    if (kv_on) {
        while (kv_P + bd <= n_input && n_steps < steps_cap) {
            llama_diffusion_set_block(model_mut, kv_P, store_cap);   // s = write offset
            llama_diffusion_set_phase(model_mut, /*WARM*/1, kv_P);   // P = cached prefix
            ar_valid = forward_range(kv_P, kv_P + bd);
            n_steps++;
            if (!ar_valid) {
                done = true;
                break;
            }
            ar_base = kv_P;
            kv_P   += bd;
        }
    }

    while (cur < L_max && !done && n_steps < steps_cap) {
        // block-aligned boundary: one AR step produces the next block's first token
        // (reference: argmax of the last row after a block completes / aligned prefill).
        // Cached mode reads it from the last WARM's logits - no extra forward.
        if (cur % bd == 0) {
            int32_t base;
            if (kv_on) {
                if (!ar_valid) {
                    break;  // should not happen: every aligned boundary follows a WARM
                }
                base = ar_base;
            } else {
                const bool ok = forward(cur);
                n_steps++;
                if (!ok) {
                    break;
                }
                base = 0;
            }
            float p;
            const llama_token tok = predict(cur, base, p);
            ar_valid             = false;  // consumed; invalidated by the next decode
            output_tokens[cur++] = tok;
            if (llama_vocab_is_eog(vocab, tok)) {
                break;
            }
            if (cur >= L_max) {
                break;
            }
        }

        // open the block: pad to the boundary with masks
        const int32_t blk_end = std::min((cur / bd + 1) * bd, L_max);
        for (int32_t i = cur; i < blk_end; i++) {
            output_tokens[i] = params.mask_token_id;
        }
        const int32_t blk_start = (cur / bd) * bd;

        // sub-blocks strictly left to right
        for (int32_t s = blk_start; s < blk_end && !done; s += sb) {
            const int32_t e = std::min(s + sb, blk_end);
            for (;;) {
                int32_t n_masked = 0;
                for (int32_t i = s; i < e; i++) {
                    if (output_tokens[i] == params.mask_token_id) {
                        n_masked++;
                    }
                }
                if (n_masked == 0) {
                    break;
                }
                if (n_steps >= steps_cap) {
                    done = true;
                    break;
                }

                bool ok;
                if (kv_on) {
                    // forward only the active block vs the cached committed prefix
                    GGML_ASSERT(kv_P == blk_start);
                    llama_diffusion_set_phase(model_mut, /*DECODE*/2, kv_P);
                    ok = forward_range(blk_start, blk_end);
                } else {
                    ok = forward(blk_end);
                }
                n_steps++;
                if (!ok) {
                    done = true;
                    break;
                }
                const int32_t fwd_base = kv_on ? blk_start : 0;

                if (params.block_eb > 0.0f) {
                    // E5c: accept by ascending entropy while the cumulative entropy
                    // stays within the budget (DG-style); lowest-entropy position
                    // always commits (progress guarantee)
                    eb_pend.clear();
                    for (int32_t i = s; i < e; i++) {
                        if (output_tokens[i] != params.mask_token_id) {
                            continue;
                        }
                        float p, h;
                        const llama_token tok = predict(i, fwd_base, p, &h);
                        eb_pend.push_back({ h, i, tok });
                    }
                    std::sort(eb_pend.begin(), eb_pend.end(),
                              [](const auto & a, const auto & b) { return a.h < b.h; });
                    double cum = 0.0;
                    for (size_t j = 0; j < eb_pend.size(); j++) {
                        if (j > 0 && cum + eb_pend[j].h > params.block_eb) {
                            break;
                        }
                        output_tokens[eb_pend[j].pos] = eb_pend[j].tok;
                        cum += eb_pend[j].h;
                    }
                } else {
                // commit everything above thr in THIS sub-block; if nothing clears
                // the bar, the max-prob position commits anyway (progress guarantee)
                int32_t     n_committed = 0;
                float       best_p      = -1.0f;
                int32_t     best_pos    = -1;
                llama_token best_tok    = LLAMA_TOKEN_NULL;
                for (int32_t i = s; i < e; i++) {
                    if (output_tokens[i] != params.mask_token_id) {
                        continue;
                    }
                    float p;
                    const llama_token tok = predict(i, fwd_base, p);
                    if (p > thr) {
                        output_tokens[i] = tok;
                        n_committed++;
                    } else if (p > best_p) {
                        best_p   = p;
                        best_pos = i;
                        best_tok = tok;
                    }
                }
                if (n_committed == 0 && best_pos >= 0) {
                    output_tokens[best_pos] = best_tok;
                }
                }

                // reference stop rule: EOG in the generated region with no mask before it
                for (int32_t i = n_input; i < blk_end; i++) {
                    if (output_tokens[i] == params.mask_token_id) {
                        break;
                    }
                    if (llama_vocab_is_eog(vocab, output_tokens[i])) {
                        cur  = i + 1;
                        done = true;
                        break;
                    }
                }
            }
        }

        if (!done) {
            cur = blk_end;
            // EOG committed mid-block (with masks after it now filled): stop here too
            for (int32_t i = n_input; i < cur; i++) {
                if (llama_vocab_is_eog(vocab, output_tokens[i])) {
                    cur  = i + 1;
                    done = true;
                    break;
                }
            }
        }

        // cached mode: the block is final - one WARM pass writes its K/V into the
        // store and its last row IS the AR step for the next block
        if (kv_on && !done && blk_end < L_max && n_steps < steps_cap) {
            llama_diffusion_set_block(model_mut, kv_P, store_cap);
            llama_diffusion_set_phase(model_mut, /*WARM*/1, kv_P);
            ar_valid = forward_range(blk_start, blk_end);
            n_steps++;
            if (!ar_valid) {
                done = true;
            } else {
                ar_base = blk_start;
                kv_P    = blk_end;
            }
        }
    }

    // pad the tail so the caller's [n_input, max_length) slice detokenizes cleanly
    const llama_token pad = llama_vocab_eot(vocab) != LLAMA_TOKEN_NULL ? llama_vocab_eot(vocab)
                                                                       : llama_vocab_eos(vocab);
    for (int32_t i = cur; i < L_max; i++) {
        output_tokens[i] = pad;
    }

    if (kv_on) {
        llama_diffusion_set_phase(model_mut, /*UNIFIED*/0, 0);  // restore for later requests / kv-off calls
    }

    const int64_t t1 = ggml_time_us();
    LOG_INF("\nblock-ar%s: total time: %0.2fms, steps: %d, time per step: %0.2fms, generated %d tokens\n",
            kv_on ? " (block-kv)" : "", (t1 - t0) / 1000.0, n_steps, (t1 - t0) / 1000.0 / std::max(n_steps, 1),
            cur - n_input);

    if (backend_sampler) {
        llama_set_sampler(ctx, 0, nullptr);
        llama_sampler_free(backend_sampler);
    }

    llama_batch_free(batch);
    n_generated = L_max;
}

// Entropy-bound denoiser for DiffusionGemma-style canvas models (see diffusion.h). The canvas is
// random-initialized; each step samples a candidate per position, accepts the lowest-entropy positions
// within a mutual-information bound, and renoises the rest under a linear temperature schedule. The output
// is the stable argmax canvas. Mirrors the reference transformers EntropyBoundSampler; set_sc is a no-op
// for non-DiffusionGemma models.
void diffusion_generate_entropy_bound(llama_context *             ctx,
                                      const llama_token *         input_tokens,
                                      llama_token *               output_tokens,
                                      int32_t                     n_input,
                                      const diffusion_eb_params & params,
                                      int32_t &                   n_generated) {
    n_generated = 0;
    if (!ctx || !input_tokens || !output_tokens || n_input <= 0 || params.max_length <= n_input) {
        return;
    }

    llama_model * model   = const_cast<llama_model *>(llama_get_model(ctx));
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int32_t C       = params.max_length - n_input;            // canvas length
    const int32_t S       = std::max(1, params.max_denoising_steps);

    llama_set_causal_attn(ctx, false);
    std::copy(input_tokens, input_tokens + n_input, output_tokens);

    std::mt19937                           rng(params.seed);
    std::uniform_real_distribution<float>  uni01(0.0f, 1.0f);
    std::uniform_int_distribution<int32_t> vocab_dist(0, n_vocab - 1);

    std::vector<llama_token> current_canvas(C);                    // working (renoised) canvas, fed to the forward
    for (int32_t i = 0; i < C; i++) {
        current_canvas[i] = vocab_dist(rng);                      // random init (not mask)
    }

    std::vector<float>       sc_buffer((size_t) C * n_vocab, 0.0f); // previous step's raw logits, for self-cond
    std::vector<llama_token> argmax_canvas(C, 0);                  // model's best prediction = the output
    std::vector<llama_token> prev_argmax(C, -1);                  // stability history (-1 -> step 0 is unstable)
    std::vector<float>       entropy(C);
    std::vector<llama_token> denoiser(C);
    std::vector<int32_t>     order(C);
    std::vector<float>       u(C);                                // pre-drawn multinomial draws (determinism)
    std::vector<llama_token> renoise(C);                         // pre-drawn renoise tokens

    const unsigned hw  = std::thread::hardware_concurrency();
    const unsigned nth = std::max(1u, std::min(hw ? hw : 1u, 32u));

    llama_batch batch = llama_batch_init(params.max_length, 0, 1);

    // Cached path: PREFILL the prompt once (writing the prefix K/V store), then each step DECODE only the
    // canvas, reading the cached prefix - instead of re-decoding [prompt|canvas] every step. The packed
    // canvas logits then start at row 0 (cached) instead of row n_input (unified).
    const int32_t logit_off = params.kv_cache ? 0 : n_input;
    if (params.kv_cache) {
        llama_diffusion_set_phase(model, /*PKV_PREFILL=*/1, n_input);
        llama_diffusion_set_sc(model, nullptr, 0.0f, 1.0f, false);
        batch.n_tokens = n_input;
        for (int32_t i = 0; i < n_input; i++) {
            batch.token[i]     = input_tokens[i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = 1;  // encode() forces all rows to output anyway; set them so it stays quiet
        }
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: PREFILL decode failed\n", __func__);
            llama_diffusion_set_phase(model, /*PKV_UNIFIED=*/0, 0);
            llama_batch_free(batch);
            return;
        }
    }

    float   prev_temp_inv = 1.0f;
    int     held          = 0;
    bool    finished      = false;

    for (int32_t cur_step = S; cur_step >= 1 && !finished; --cur_step) {
        const int32_t step_idx = S - cur_step;                    // 0-based
        const float   t        = params.t_min + (params.t_max - params.t_min) * ((float) cur_step / (float) S);
        const float   temp_inv = 1.0f / t;

        if (params.kv_cache) {
            llama_diffusion_set_phase(model, /*PKV_DECODE=*/2, n_input);
            batch.n_tokens = C;
            for (int32_t i = 0; i < C; i++) {
                batch.token[i]     = current_canvas[i];
                batch.pos[i]       = n_input + i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = 1;
            }
        } else {
            batch.n_tokens = params.max_length;
            for (int32_t i = 0; i < params.max_length; i++) {
                batch.token[i]     = (i < n_input) ? input_tokens[i] : current_canvas[i - n_input];
                batch.pos[i]       = i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = 1;
            }
        }

        // self-conditioning = softmax(previous step's logits / previous t); gated off on the first step
        llama_diffusion_set_sc(model, sc_buffer.data(), step_idx == 0 ? 0.0f : 1.0f, prev_temp_inv, true);

        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: failed to decode at step %d\n", __func__, step_idx);
            break;
        }
        const float * logits = llama_get_logits(ctx);             // canvas rows packed: [C or max_length, n_vocab]

        // pre-draw the step's randomness single-threaded so the output is seed-reproducible
        for (int32_t pos = 0; pos < C; pos++) {
            u[pos]       = uni01(rng);
            renoise[pos] = vocab_dist(rng);
        }

        // per position: argmax, entropy of softmax(raw/t), and a multinomial sample; stash raw row for SC
        auto worker = [&](int32_t p0, int32_t p1) {
            for (int32_t pos = p0; pos < p1; pos++) {
                const float * row = logits + (size_t) (logit_off + pos) * n_vocab;
                float m = -INFINITY; int32_t amax = 0;
                for (int32_t v = 0; v < n_vocab; v++) {
                    const float z = row[v] * temp_inv;
                    if (z > m) { m = z; amax = v; }
                }
                float Z = 0.0f;
                for (int32_t v = 0; v < n_vocab; v++) {
                    Z += expf(row[v] * temp_inv - m);
                }
                const float target = u[pos] * Z;
                float   cum = 0.0f, H = 0.0f;
                int32_t sampled = n_vocab - 1; bool picked = false;
                for (int32_t v = 0; v < n_vocab; v++) {
                    const float e = expf(row[v] * temp_inv - m);
                    const float p = e / Z;
                    if (p > 0.0f) { H -= p * logf(p); }
                    cum += e;
                    if (!picked && cum >= target) { sampled = v; picked = true; }
                }
                entropy[pos]       = H;
                argmax_canvas[pos] = amax;
                denoiser[pos]      = sampled;
                std::memcpy(sc_buffer.data() + (size_t) pos * n_vocab, row, n_vocab * sizeof(float));
            }
        };
        {
            std::vector<std::thread> pool;
            const int32_t chunk = (C + (int32_t) nth - 1) / (int32_t) nth;
            for (unsigned ti = 0; ti < nth; ti++) {
                const int32_t p0 = (int32_t) ti * chunk;
                const int32_t p1 = std::min(p0 + chunk, C);
                if (p0 < p1) { pool.emplace_back(worker, p0, p1); }
            }
            for (auto & th : pool) { th.join(); }
        }

        // accept the lowest-entropy positions within the MI bound (sum of strictly-earlier entropies <= bound)
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return entropy[a] < entropy[b]; });
        std::vector<char> accepted(C, 0);
        double cumE = 0.0;
        for (int32_t k = 0; k < C; k++) {
            const int32_t pos = order[k];
            cumE += entropy[pos];
            if (cumE - entropy[pos] <= params.entropy_bound) { accepted[pos] = 1; }
        }

        // renoise: accepted -> sampled token, rest -> fresh random; the displayed/output canvas is the argmax
        float entropy_sum = 0.0f;
        for (int32_t pos = 0; pos < C; pos++) {
            current_canvas[pos]          = accepted[pos] ? denoiser[pos] : renoise[pos];
            output_tokens[n_input + pos] = argmax_canvas[pos];
            entropy_sum += entropy[pos];
        }

        // adaptive stop: argmax stable for stability_threshold steps AND confident (low mean entropy)
        held = (prev_argmax == argmax_canvas) ? held + 1 : 0;
        const bool confident = (entropy_sum / (float) C) < params.confidence_threshold;
        if (held >= params.stability_threshold && confident) { finished = true; }
        prev_argmax   = argmax_canvas;
        prev_temp_inv = temp_inv;

        if (params.step_callback &&
            !params.step_callback(step_idx, S, output_tokens, params.max_length, params.step_callback_user_data)) {
            break;
        }
    }

    if (params.output_confidences) {
        // final-step entropy per canvas position (lower = more confident); -1 over the prompt
        std::fill(params.output_confidences, params.output_confidences + params.max_length, -1.0f);
        for (int32_t pos = 0; pos < C; pos++) {
            params.output_confidences[n_input + pos] = entropy[pos];
        }
    }

    if (params.kv_cache) {
        llama_diffusion_set_phase(model, /*PKV_UNIFIED=*/0, 0);  // restore default for later turns / masked path
    }
    llama_batch_free(batch);
    n_generated = params.max_length;
}
