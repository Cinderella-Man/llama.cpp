// F2 STEP 2 v0 (07_layer_f.md): DFlash-style pairing prototype - FastDLLM-1.5B
// block-diffusion DRAFTS, an AR verifier (Qwen2.5-7B-Instruct, same tokenizer
// family) greedily VERIFIES. Output distribution == the verifier's greedy
// decode (lossless); the measurement is closed-loop acceptance + end-to-end
// tok/s vs the same-process AR baseline.
//
//   llama-dflash -m <verifier.gguf> --model-draft <fast-dllm.gguf> -ngl 99 \
//       --in cases.jsonl [--draft-len 32] [--max-tokens 256] [--ar]
//
// cases.jsonl lines: {"id": str, "user": str} (chat user content; the verifier's
// chat template is applied - identical Qwen2.5 template on both models).
// --ar skips drafting: pure greedy AR loop on the verifier (the baseline).
// v0 keeps the drafter STATELESS per round (its pkv store re-prefills committed
// blocks each call - ~8 ms per 32-block, measured negligible at probe lengths).

#include "diffusion.h"
#include "llama.h"
#include "common.h"
#include "arg.h"
#include "log.h"
#include "chat.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    std::string in_path;
    int32_t     draft_len  = 32;
    int32_t     max_tokens = 256;
    bool        ar_only    = false;
    std::vector<char *> passthru;
    for (int i = 0; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--in" && i + 1 < argc) {
            in_path = argv[++i];
        } else if (a == "--draft-len" && i + 1 < argc) {
            draft_len = atoi(argv[++i]);
        } else if (a == "--max-tokens" && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (a == "--ar") {
            ar_only = true;
        } else {
            passthru.push_back(argv[i]);
        }
    }
    if (in_path.empty()) {
        LOG_ERR("usage: %s -m verifier.gguf --model-draft drafter.gguf -ngl 99 --in cases.jsonl\n", argv[0]);
        return 1;
    }

    common_params params;
    if (!common_params_parse((int) passthru.size(), passthru.data(), params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }
    common_init();
    llama_backend_init();

    // verifier (AR, KV cache, causal)
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = params.n_gpu_layers;
    llama_model * model_v = llama_model_load_from_file(params.model.path.c_str(), mp);
    if (!model_v) {
        return 1;
    }
    const llama_vocab * vocab_v = llama_model_get_vocab(model_v);
    const int n_vocab_v = llama_vocab_n_tokens(vocab_v);

    llama_context_params cpv = llama_context_default_params();
    cpv.n_ctx   = 2048;
    cpv.n_batch = cpv.n_ubatch = 512;
    llama_context * ctx_v = llama_init_from_model(model_v, cpv);
    if (!ctx_v) {
        return 1;
    }

    // drafter (fast-dllm block-AR)
    llama_model *   model_d = nullptr;
    llama_context * ctx_d   = nullptr;
    llama_token     mask_id = LLAMA_TOKEN_NULL;
    if (!ar_only) {
        if (params.speculative.draft.mparams.path.empty()) {
            LOG_ERR("--model-draft required without --ar\n");
            return 1;
        }
        model_d = llama_model_load_from_file(params.speculative.draft.mparams.path.c_str(), mp);
        if (!model_d) {
            return 1;
        }
        llama_context_params cpd = llama_context_default_params();
        cpd.n_ctx   = 2048;
        cpd.n_batch = cpd.n_ubatch = 512;
        ctx_d = llama_init_from_model(model_d, cpd);
        if (!ctx_d) {
            return 1;
        }
        mask_id = llama_vocab_mask(llama_model_get_vocab(model_d));
    }

    auto templates = common_chat_templates_init(model_v, "");

    std::ifstream in(in_path);
    if (!in) {
        LOG_ERR("cannot open %s\n", in_path.c_str());
        return 1;
    }

    llama_batch batch = llama_batch_init(512, 0, 1);
    auto decode_v = [&](const llama_token * toks, int n, int pos0, bool all_logits) -> bool {
        batch.n_tokens = n;
        for (int i = 0; i < n; i++) {
            batch.token[i]     = toks[i];
            batch.pos[i]       = pos0 + i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = all_logits || i == n - 1;
        }
        return llama_decode(ctx_v, batch) == 0;
    };
    auto argmax_row = [&](int ith) -> llama_token {
        const float * row = llama_get_logits_ith(ctx_v, ith);
        int am = 0;
        for (int v = 1; v < n_vocab_v; v++) {
            if (row[v] > row[am]) {
                am = v;
            }
        }
        return am;
    };

    int64_t tot_tokens = 0, tot_us = 0, tot_rounds = 0, tot_accepted = 0, tot_drafted = 0;
    int64_t draft_us = 0, verify_us = 0;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const auto j = nlohmann::json::parse(line);
        const std::string id   = j.value("id", "?");
        const std::string user = j.at("user").get<std::string>();

        common_chat_templates_inputs inputs;
        common_chat_msg msg;
        msg.role    = "user";
        msg.content = user;
        inputs.messages.push_back(msg);
        inputs.add_generation_prompt = true;
        const std::string prompt = common_chat_templates_apply(templates.get(), inputs).prompt;

        std::vector<llama_token> committed = common_tokenize(vocab_v, prompt, true, true);
        const int n_p = (int) committed.size();
        committed.reserve(n_p + max_tokens + draft_len + 8);

        llama_memory_clear(llama_get_memory(ctx_v), true);

        const int64_t t0 = ggml_time_us();
        if (!decode_v(committed.data(), n_p, 0, false)) {
            LOG_ERR("%s: prefill failed\n", id.c_str());
            continue;
        }
        llama_token pred_next = argmax_row(n_p - 1);

        int rounds = 0, accepted_sum = 0;
        std::vector<llama_token> dbuf;
        while ((int) committed.size() - n_p < max_tokens) {
            const int n_c = (int) committed.size();

            if (ar_only) {
                committed.push_back(pred_next);
                if (llama_vocab_is_eog(vocab_v, pred_next)) {
                    break;
                }
                if (!decode_v(&committed[n_c], 1, n_c, false)) {
                    break;
                }
                pred_next = argmax_row(0);
                continue;
            }

            // draft: one block-AR call over [committed | draft_len masks]
            const int dl = std::min(draft_len, 512 - n_c);
            if (dl < 4) {
                break;  // drafter ubatch exhausted
            }
            dbuf.assign(n_c + dl, 0);
            diffusion_params dp;
            dp.mask_token_id    = mask_id;
            dp.max_length       = n_c + dl;
            dp.steps            = 256;
            dp.temperature      = 0.0f;
            dp.conf_threshold   = 0.9f;
            dp.block_length     = 32;
            dp.top_k            = 40;
            dp.block_kv         = true;
            dp.backend_sampling = true;
            int32_t n_gen = 0;
            const int64_t td = ggml_time_us();
            diffusion_generate_block_ar(ctx_d, committed.data(), dbuf.data(), n_c, dp, n_gen);
            draft_us += ggml_time_us() - td;
            const llama_token * draft = dbuf.data() + n_c;
            tot_drafted += dl;

            // verify: greedy walk; d[i] accepted iff it equals the verifier's
            // prediction conditioned on everything before it
            const int64_t tv = ggml_time_us();
            int A = 0;
            llama_token next;
            if (draft[0] != pred_next) {
                next = pred_next;  // round degenerates to one AR token
            } else {
                if (!decode_v(draft, dl, n_c, true)) {
                    break;
                }
                A = 1;
                while (A < dl && draft[A] == argmax_row(A - 1)) {
                    A++;
                }
                next = argmax_row(A - 1);  // verifier's token after the accepted run
                // roll the cache back to the accepted prefix
                llama_memory_seq_rm(llama_get_memory(ctx_v), 0, n_c + A, -1);
            }
            verify_us += ggml_time_us() - tv;

            for (int i = 0; i < A; i++) {
                committed.push_back(draft[i]);
            }
            committed.push_back(next);
            rounds++;
            accepted_sum += A;
            if (llama_vocab_is_eog(vocab_v, next) ||
                std::any_of(committed.end() - A - 1, committed.end() - 1,
                            [&](llama_token t) { return llama_vocab_is_eog(vocab_v, t); })) {
                break;
            }
            // feed the correction token; its row gives the next prediction
            const int n_now = (int) committed.size() - 1;
            if (!decode_v(&committed[n_now], 1, n_now, false)) {
                break;
            }
            pred_next = argmax_row(0);
        }
        const int64_t t1   = ggml_time_us();
        const int     gen  = (int) committed.size() - n_p;

        tot_tokens += gen;
        tot_us += t1 - t0;
        tot_rounds += rounds;
        tot_accepted += accepted_sum;

        std::string extra;
        if (!ar_only) {
            char buf[96];
            snprintf(buf, sizeof(buf), " | rounds %d, accepted/round %.2f", rounds,
                     rounds ? (double) accepted_sum / rounds : 0.0);
            extra = buf;
        }
        LOG_INF("%-14s %3d tok in %7.1f ms = %5.1f tok/s%s\n", id.c_str(), gen, (t1 - t0) / 1000.0,
                gen * 1000.0 / ((t1 - t0) / 1000.0), extra.c_str());
    }

    LOG_INF("\n%s: %lld tokens in %.1f ms = %.1f tok/s",
            ar_only ? "AR baseline" : "dflash", (long long) tot_tokens, tot_us / 1000.0,
            tot_tokens * 1000.0 / (tot_us / 1000.0));
    if (!ar_only && tot_rounds > 0) {
        LOG_INF(" | rounds %lld | mean accepted/round %.2f (+1 corrected) | draft share %.1f%%, verify %.1f%% | drafted-token efficiency %.1f%%",
                (long long) tot_rounds, (double) tot_accepted / tot_rounds,
                100.0 * draft_us / tot_us, 100.0 * verify_us / tot_us,
                100.0 * tot_accepted / std::max<int64_t>(tot_drafted, 1));
    }
    LOG_INF("\n");

    llama_batch_free(batch);
    llama_free(ctx_v);
    llama_model_free(model_v);
    if (ctx_d) {
        llama_free(ctx_d);
        llama_model_free(model_d);
    }
    llama_backend_free();
    return 0;
}
