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

            // Layer A: pick phase + batch window for this step
            int32_t batch_first = 0;
            int32_t batch_last  = cur_length;
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
                const int32_t kv_batch_end =
                    params.kv_window > 0 ? std::min(cur_length, kv_e + params.kv_window)
                                         : (kv_dual ? kv_e : cur_length);

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
            const int32_t scan_hi = kv_on ? std::min(batch_last, cur_length) : cur_length;
            for (int32_t i = scan_lo; i < scan_hi; i++) {
                if (output_tokens[i] == params.mask_token_id) {
                    // For block-based, only consider current block
                    if (params.schedule != DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED || (i >= block_start && i < block_end)) {
                        mask_positions.push_back(i);
                    }
                }
            }

            if (mask_positions.empty()) {
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

                    // degeneracy guard: with a whole-sequence threshold the model's most confident early
                    // predictions can be the end-of-text padding tail; if end tokens flood the committed
                    // region while real positions are still masked, the draft is unrecoverable - abort
                    {
                        const llama_vocab * gvocab = llama_model_get_vocab(model);

                        // a contiguous committed-EOT SUFFIX is the normal end of a short
                        // answer - only end tokens scattered through the non-tail region
                        // signal degeneracy
                        // in Layer A block mode positions beyond kv_e are masked BY DESIGN -
                        // the guard must only judge the decoded region [n_input, kv_e)
                        const int32_t guard_end = kv_on ? kv_e : params.max_length;
                        int32_t tail_start = guard_end;
                        while (tail_start > n_input) {
                            const llama_token t = output_tokens[tail_start - 1];
                            if (t != params.mask_token_id &&
                                (llama_vocab_is_eog(gvocab, t) || t == llama_vocab_pad(gvocab))) {
                                tail_start--;
                            } else {
                                break;
                            }
                        }

                        int32_t n_committed_total = 0;
                        int32_t n_committed_eog   = 0;
                        int32_t n_masked          = 0;
                        for (int32_t i = n_input; i < tail_start; i++) {
                            if (output_tokens[i] == params.mask_token_id) {
                                n_masked++;
                            } else {
                                n_committed_total++;
                                if (llama_vocab_is_eog(gvocab, output_tokens[i]) ||
                                    output_tokens[i] == llama_vocab_pad(gvocab)) {
                                    n_committed_eog++;
                                }
                            }
                        }
                        const int32_t n_gen = tail_start - n_input;
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
