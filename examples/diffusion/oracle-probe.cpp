// F2 STEP 1 oracle acceptance probe (07_layer_f.md): how many consecutive tokens
// of a FastDLLM-1.5B block draft does an AR verifier (Qwen2.5-7B-Instruct, same
// tokenizer family) accept under greedy decoding?
//
//   llama-oracle-probe -m <verifier.gguf> -ngl 99 -ub 1024 --in drafts.jsonl
//
// drafts.jsonl lines: {"id": str, "user": str (chat user content), "draft": str}
// The verifier applies ITS OWN chat template to `user` (identical Qwen2.5
// template as the drafter served), then one causal decode over [prompt | draft]
// with logits on every row; per 32-token block of the draft we report
// L = consecutive positions from the block start where argmax(row pos-1) equals
// the draft token. Drafts cross as TEXT (canonical re-tokenization, shared
// vocab) - fine for an oracle estimate.

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
    std::vector<char *> passthru;
    for (int i = 0; i < argc; i++) {
        if (std::string(argv[i]) == "--in" && i + 1 < argc) {
            in_path = argv[++i];
        } else {
            passthru.push_back(argv[i]);
        }
    }
    if (in_path.empty()) {
        LOG_ERR("usage: %s -m verifier.gguf -ngl 99 -ub 1024 --in drafts.jsonl\n", argv[0]);
        return 1;
    }

    common_params params;
    if (!common_params_parse((int) passthru.size(), passthru.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    common_init();
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = params.n_gpu_layers;
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mp);
    if (!model) {
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = params.n_ctx > 0 ? params.n_ctx : 2048;
    cp.n_batch = cp.n_ubatch = params.n_ubatch > 0 ? params.n_ubatch : 1024;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        return 1;
    }

    auto templates = common_chat_templates_init(model, "");

    std::ifstream in(in_path);
    if (!in) {
        LOG_ERR("cannot open %s\n", in_path.c_str());
        return 1;
    }

    const int BD = 32;

    std::vector<int> all_L;        // per-block acceptance lengths
    std::vector<int> first_L;      // first block of each draft (cleanest spec-round proxy)
    int64_t agree = 0, total = 0;
    int64_t verify_us = 0;
    int     n_verifies = 0;

    llama_batch batch = llama_batch_init(cp.n_ubatch, 0, 1);

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const auto j = nlohmann::json::parse(line);
        const std::string id    = j.value("id", "?");
        const std::string user  = j.at("user").get<std::string>();
        const std::string draft = j.at("draft").get<std::string>();

        common_chat_templates_inputs inputs;
        common_chat_msg msg;
        msg.role    = "user";
        msg.content = user;
        inputs.messages.push_back(msg);
        inputs.add_generation_prompt = true;
        const std::string prompt = common_chat_templates_apply(templates.get(), inputs).prompt;

        std::vector<llama_token> toks = common_tokenize(vocab, prompt, true, true);
        const int n_p = (int) toks.size();
        const std::vector<llama_token> dtoks = common_tokenize(vocab, draft, false, false);
        toks.insert(toks.end(), dtoks.begin(), dtoks.end());
        const int n_total = (int) toks.size();
        if (n_total > (int) cp.n_ubatch) {
            LOG_ERR("%s: %d tokens > ubatch, skipped\n", id.c_str(), n_total);
            continue;
        }

        llama_memory_clear(llama_get_memory(ctx), true);
        batch.n_tokens = n_total;
        for (int i = 0; i < n_total; i++) {
            batch.token[i]     = toks[i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = 1;
        }
        const int64_t t0 = ggml_time_us();
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: decode failed\n", id.c_str());
            continue;
        }
        llama_synchronize(ctx);
        verify_us += ggml_time_us() - t0;
        n_verifies++;

        std::vector<bool> match(n_total, false);
        for (int i = n_p; i < n_total; i++) {
            const float * row = llama_get_logits_ith(ctx, i - 1);
            int am = 0;
            for (int v = 1; v < n_vocab; v++) {
                if (row[v] > row[am]) {
                    am = v;
                }
            }
            match[i] = (am == toks[i]);
            agree += match[i] ? 1 : 0;
            total++;
        }

        std::string blocks_str;
        for (int b0 = n_p; b0 < n_total; b0 += BD) {
            const int be = std::min(b0 + BD, n_total);
            int L = 0;
            while (b0 + L < be && match[b0 + L]) {
                L++;
            }
            all_L.push_back(L);
            if (b0 == n_p) {
                first_L.push_back(L);
            }
            blocks_str += std::to_string(L) + " ";
        }
        LOG_INF("%-14s draft %3d tok, agree %3d/%3d, L per block: %s\n",
                id.c_str(), n_total - n_p,
                (int) std::count(match.begin() + n_p, match.end(), true), n_total - n_p,
                blocks_str.c_str());
    }

    if (!all_L.empty()) {
        auto mean = [](const std::vector<int> & v) {
            double s = 0;
            for (int x : v) {
                s += x;
            }
            return s / v.size();
        };
        std::vector<int> sorted = all_L;
        std::sort(sorted.begin(), sorted.end());
        LOG_INF("\nblocks: %zu | mean L %.2f | median L %d | mean first-block L %.2f | "
                "overall agreement %.1f%% | verifier forward %.1f ms avg\n",
                all_L.size(), mean(all_L), sorted[sorted.size() / 2], mean(first_L),
                100.0 * agree / std::max<int64_t>(total, 1), verify_us / 1000.0 / std::max(n_verifies, 1));
        int ge8 = 0, full = 0;
        for (int x : all_L) {
            ge8 += x >= 8;
            full += x >= BD;
        }
        LOG_INF("blocks with L>=8: %d/%zu | fully accepted (L=32): %d/%zu\n",
                ge8, all_L.size(), full, all_L.size());
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
