// The two bytes-per-token switches on a model file (a random qwen4exp from test-llama-archs, or a real model), on the
// CPU backend:
//
// - Q618, fewer routed experts on confident tokens (llama_set_moe_tail): at p = 1 the logits of a prompt equal the
//   switch-off logits bit for bit and every token keeps all its experts; below it the logits deviate and the
//   histogram counts fewer experts. Q660: the cut as GGML_OP_MOE_TAIL (the default; -1 ids for micro-batches of up to
//   8 tokens, which MUL_MAT_ID skips; with repeat the last kept expert instead; every slot's own expert in larger ones
//   or with skip = 0) gives the graph-level ops' logits and histogram bit for bit, in micro-batches of 64, 4 and 1
//   tokens.
// - Q619, the shortlist head with an exact bound (llama_head_bound_*): greedy and sampled generations (top-k, min-p,
//   temperature, penalties, a grammar) give the same tokens, with the same probabilities, with it on and off; the
//   sampled tokens that kept the shortlist's row and those that fell back to the full head are counted. With
//   --zero-tail the head's rows from k_low on are zero in both runs, so that the bound can exclude them and the
//   shortlist's rows stand (a random head's rows are all alike, and the bound excludes none of them). A random
//   router weighs its experts almost alike: --router-scale makes it favour a few, as a trained one does.
//
// Usage: test-bytes-switches -m <model.gguf> [-t threads] [-n prompts] [-g tokens] [-k k_low] [--zero-tail]
//                            [--router-scale s]

#include "../src/llama-model.h"
#include "../src/llama-ext.h"

#include "common.h"
#include "sampling.h"

#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int n_checks = 0;
static int n_fail   = 0;

#define CHECK(cond) do { n_checks++; if (!(cond)) { n_fail++; fprintf(stderr, "%s:%d: failed: %s\n", __FILE__, __LINE__, #cond); } } while (0)

struct args_t {
    std::string model;
    int  n_threads = 4;
    int  n_prompts = 24;
    int  n_gen     = 24;
    int  k_low     = 0;     // 0: a quarter of the vocabulary
    bool zero_tail = false;
    float router_scale = 1.0f; // the routers' F32 rows scaled by this (a sharper routing for the cut to act on)
};

static std::vector<llama_token> make_prompt(int i, int n_vocab) {
    std::mt19937 rng(1000 + i);
    std::uniform_int_distribution<int> len(4, 24);
    std::uniform_int_distribution<int> tok(0, n_vocab - 1);
    std::vector<llama_token> p(len(rng));
    for (auto & t : p) {
        t = tok(rng);
    }
    return p;
}

static llama_context * make_ctx(llama_model * model, int n_threads, int n_ubatch = 64) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = 256;
    cp.n_batch         = 256;
    cp.n_ubatch        = n_ubatch;
    cp.n_seq_max       = 1;
    cp.n_threads       = n_threads;
    cp.n_threads_batch = n_threads;
    return llama_init_from_model(model, cp);
}

// the logits of every position of a prompt
static std::vector<float> prompt_logits(llama_context * ctx, const std::vector<llama_token> & prompt, int n_vocab) {
    llama_memory_clear(llama_get_memory(ctx), true);
    common_batch batch(ctx);
    for (size_t i = 0; i < prompt.size(); ++i) {
        batch.add(prompt[i], (llama_pos) i, 0, true);
    }
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
        fprintf(stderr, "decode failed\n");
        exit(1);
    }
    std::vector<float> out;
    out.reserve(prompt.size()*n_vocab);
    for (size_t i = 0; i < prompt.size(); ++i) {
        const float * l = llama_get_logits_ith(ctx, (int32_t) i);
        out.insert(out.end(), l, l + n_vocab);
    }
    return out;
}

static int argmax(const float * v, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (v[i] > v[best]) {
            best = i;
        }
    }
    return best;
}

static void test_moe_tail(llama_model * model, const args_t & a) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int k       = (int) model->hparams.n_expert_used_max();
    if (model->hparams.n_expert == 0 || k < 2) {
        printf("moe tail: the model has %u routed experts, %d used: skipped\n", model->hparams.n_expert, k);
        return;
    }

    struct arm_t { float p; float w; bool renorm; };
    const arm_t arms[] = {
        { 1.00f, 0.00f, false }, // nothing dropped
        { 1.00f, 0.00f, true  },
        { 0.90f, 0.00f, false },
        { 0.90f, 0.00f, true  },
        { 0.80f, 0.00f, true  },
        { 0.60f, 0.00f, true  },
        { 1.00f, 0.75f/k, true }, // a minimum share
    };
    // how the cut runs: the graph-level ops (the reference), GGML_OP_MOE_TAIL with -1 ids up to 8 tokens, with the last
    // kept expert up to 8 tokens (repeat), and with every slot's own expert (skip 0)
    struct mode_t { const char * name; bool graph; int skip; bool repeat; };
    const mode_t modes[] = {
        { "graph",     true,  8, false },
        { "op",        false, 8, false },
        { "op-repeat", false, 8, true  },
        { "op-skip0",  false, 0, false },
    };

    // micro-batches of 64 tokens (the last kept expert), 4 and 1 (-1 ids)
    for (const int n_ubatch : { 64, 4, 1 }) {
        llama_context * c_off = make_ctx(model, a.n_threads, n_ubatch);

        std::vector<std::vector<float>> l_off(a.n_prompts);
        for (int i = 0; i < a.n_prompts; ++i) {
            l_off[i] = prompt_logits(c_off, make_prompt(i, n_vocab), n_vocab);
        }

        for (const auto & arm : arms) {
            std::vector<std::vector<float>> l_graph;
            std::vector<uint64_t> counts_graph;

            for (const auto & mode : modes) {
                llama_context * c_tail = make_ctx(model, a.n_threads, n_ubatch);
                llama_set_moe_tail_mode(c_tail, mode.graph, mode.skip, mode.repeat);
                llama_set_moe_tail(c_tail, true, arm.p, arm.w, arm.renorm, true);

                double   max_diff = 0.0;
                uint64_t n_pos = 0;
                uint64_t n_same_top = 0;
                bool     identical = true;  // to the switch off
                bool     same_graph = true; // to the graph-level ops
                std::vector<std::vector<float>> l_tail(a.n_prompts);
                for (int i = 0; i < a.n_prompts; ++i) {
                    const auto prompt = make_prompt(i, n_vocab);
                    l_tail[i] = prompt_logits(c_tail, prompt, n_vocab);
                    identical = identical && memcmp(l_off[i].data(), l_tail[i].data(), l_off[i].size()*sizeof(float)) == 0;
                    if (!mode.graph) {
                        same_graph = same_graph &&
                            memcmp(l_graph[i].data(), l_tail[i].data(), l_tail[i].size()*sizeof(float)) == 0;
                    }
                    for (size_t j = 0; j < prompt.size(); ++j) {
                        const float * x = l_off[i].data()  + j*n_vocab;
                        const float * y = l_tail[i].data() + j*n_vocab;
                        for (int t = 0; t < n_vocab; ++t) {
                            max_diff = std::max(max_diff, (double) std::fabs(x[t] - y[t]));
                        }
                        n_same_top += argmax(x, n_vocab) == argmax(y, n_vocab);
                        n_pos++;
                    }
                }

                std::vector<uint64_t> counts(k + 1, 0);
                const uint64_t n_pairs = llama_moe_tail_counts(c_tail, counts.data(), k + 1);
                double mean = 0.0;
                for (int c = 0; c <= k; ++c) {
                    mean += (double) c*counts[c];
                }
                mean /= std::max<uint64_t>(n_pairs, 1);

                printf("moe tail p = %.2f, w = %.3f, renorm %d, ubatch %2d, %-9s: %s, max |diff| %.3g, same top token "
                        "%.1f %% of %llu positions; %llu token-layers, mean %.2f of %d experts kept%s\n", arm.p, arm.w,
                        (int) arm.renorm, n_ubatch, mode.name, identical ? "identical logits" : "logits differ",
                        max_diff, 100.0*n_same_top/n_pos, (unsigned long long) n_pos, (unsigned long long) n_pairs,
                        mean, k, mode.graph ? "" : same_graph ? "; the graph's logits" : "; logits DIFFER from the graph's");

                CHECK(n_pairs > 0);
                if (arm.p == 1.0f && arm.w == 0.0f) {
                    // the switch on, nothing dropped: the same logits, every pair with all k experts
                    CHECK(identical);
                    CHECK(counts[k] == n_pairs);
                } else {
                    CHECK(mean < k);
                }
                if (mode.graph) {
                    l_graph      = std::move(l_tail);
                    counts_graph = counts;
                } else {
                    // the op drops the same slots with the same arithmetic, and a dropped expert adds 0 either way
                    CHECK(same_graph);
                    CHECK(counts == counts_graph);
                }
                llama_free(c_tail);
            }
        }
        llama_free(c_off);
    }
}

struct sampling_arm {
    const char * name;
    common_params_sampling p;
};

static std::vector<sampling_arm> sampling_arms() {
    std::vector<sampling_arm> arms;

    common_params_sampling greedy;
    greedy.temp = 0.0f;
    arms.push_back({ "greedy", greedy });

    common_params_sampling def; // top-k 40, top-p 0.95, min-p 0.05, temperature 0.8
    arms.push_back({ "default chain", def });

    common_params_sampling k5;
    k5.top_k = 5;
    k5.top_p = 1.0f;
    k5.min_p = 0.0f;
    k5.temp  = 1.0f;
    arms.push_back({ "top-k 5", k5 });

    common_params_sampling agent; // the agent check's (the makers' settings)
    agent.top_k = 20;
    agent.top_p = 0.95f;
    agent.min_p = 0.0f;
    agent.temp  = 1.0f;
    arms.push_back({ "agent check", agent });

    common_params_sampling minp;
    minp.top_k = 0;
    minp.top_p = 1.0f;
    minp.min_p = 0.2f;
    minp.temp  = 0.7f;
    arms.push_back({ "min-p 0.2", minp });

    common_params_sampling pen = def;
    pen.top_k          = 20;
    pen.penalty_last_n = 16;
    pen.penalty_repeat = 1.3f;
    pen.penalty_freq   = 0.2f;
    arms.push_back({ "top-k 20 + penalties", pen });

    common_params_sampling gram = def;
    gram.top_k   = 10;
    gram.grammar = common_grammar(COMMON_GRAMMAR_TYPE_USER, "root ::= [0-7]+"); // the test vocabulary's pieces are hex ids
    arms.push_back({ "top-k 10 + grammar", gram });

    common_params_sampling topp; // top-p alone reads the whole distribution: always the full head
    topp.top_k = 0;
    topp.top_p = 0.9f;
    topp.min_p = 0.0f;
    arms.push_back({ "top-p 0.9", topp });

    return arms;
}

struct gen_t {
    std::vector<llama_token> tokens;
    std::vector<float>       probs;
    common_sampler_stats     stats;
};

static gen_t generate(llama_context * ctx, llama_model * model, common_params_sampling p, uint32_t seed,
        const std::vector<llama_token> & prompt, int n_gen, bool grammar_ok) {
    gen_t g;
    p.seed = seed;
    if (!grammar_ok) {
        p.grammar = {};
    }
    common_sampler * smpl = common_sampler_init(model, p);
    if (!smpl) {
        fprintf(stderr, "sampler init failed\n");
        exit(1);
    }

    llama_memory_clear(llama_get_memory(ctx), true);
    common_batch batch(ctx);
    int32_t idx = -1;
    for (size_t i = 0; i < prompt.size(); ++i) {
        idx = batch.add(prompt[i], (llama_pos) i, 0, i + 1 == prompt.size());
    }
    llama_pos pos = (llama_pos) prompt.size();

    for (int s = 0; s < n_gen; ++s) {
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            fprintf(stderr, "decode failed\n");
            exit(1);
        }
        const llama_token id = common_sampler_sample(smpl, ctx, idx);
        const auto * cur = common_sampler_get_candidates(smpl, false);
        g.tokens.push_back(id);
        g.probs.push_back(cur->selected >= 0 ? cur->data[cur->selected].p : -1.0f);
        common_sampler_accept(smpl, id, true);

        batch.clear();
        idx = batch.add(id, pos++, 0, true);
    }

    g.stats = common_sampler_get_stats(smpl);
    common_sampler_free(smpl);
    return g;
}

static void test_head_bound(llama_model * model, const args_t & a) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t k_low = a.k_low > 0 ? (uint32_t) a.k_low : (uint32_t) n_vocab/4;

    llama_context * c_off = make_ctx(model, a.n_threads);
    llama_context * c_on  = make_ctx(model, a.n_threads);

    if (!llama_head_shortlist_init(c_on, k_low, 256) || !llama_head_bound_init(c_on)) {
        fprintf(stderr, "no shortlist head with a bound for this model\n");
        CHECK(false);
        llama_free(c_on);
        llama_free(c_off);
        return;
    }
    llama_set_head_bound(c_on, true);

    // the grammar needs the test vocabulary's hex pieces
    char piece[16] = {};
    const int n_piece = llama_token_to_piece(llama_model_get_vocab(model), 0x1a, piece, sizeof(piece), 0, false);
    const bool grammar_ok = n_piece == 2 && strncmp(piece, "1a", 2) == 0;

    uint64_t n_tok_all   = 0;
    uint64_t n_short_all = 0;
    for (const auto & arm : sampling_arms()) {
        uint64_t n_tok = 0;
        uint64_t n_same = 0;
        uint64_t n_short = 0;
        uint64_t n_full = 0;
        bool probs_same = true;
        for (int i = 0; i < a.n_prompts; ++i) {
            const auto prompt = make_prompt(i, n_vocab);
            const gen_t g_off = generate(c_off, model, arm.p, 42 + i, prompt, a.n_gen, grammar_ok);
            const gen_t g_on  = generate(c_on,  model, arm.p, 42 + i, prompt, a.n_gen, grammar_ok);
            for (int s = 0; s < a.n_gen; ++s) {
                n_same += g_off.tokens[s] == g_on.tokens[s];
                probs_same = probs_same && g_off.probs[s] == g_on.probs[s];
            }
            n_tok   += a.n_gen;
            n_short += g_on.stats.n_head_short;
            n_full  += g_on.stats.n_head_full;
            CHECK(g_off.stats.n_head_short == 0 && g_off.stats.n_head_full == 0);
        }
        printf("head bound, %-22s: %llu of %llu tokens the same, probabilities %s; %llu from the shortlist's rows, "
                "%llu from the full head's\n", arm.name, (unsigned long long) n_same, (unsigned long long) n_tok,
                probs_same ? "the same" : "DIFFER", (unsigned long long) n_short, (unsigned long long) n_full);
        CHECK(n_same == n_tok);
        CHECK(probs_same);
        // a grammar's resample checks the row a second time
        CHECK(arm.p.grammar.grammar.empty() ? n_short + n_full == n_tok : n_short + n_full >= n_tok);
        if (strcmp(arm.name, "top-p 0.9") == 0) {
            CHECK(n_short == 0);
        }
        n_tok_all   += n_tok;
        n_short_all += n_short;
    }

    uint64_t n_rows = 0;
    uint64_t n_full = 0;
    llama_head_bound_stats(c_on, &n_rows, &n_full);
    printf("head bound: %llu rows with the shortlist head, %llu full-head fallbacks; %llu of %llu tokens from the "
            "shortlist's rows; %u ids in the shortlist (k_low %u)\n", (unsigned long long) n_rows,
            (unsigned long long) n_full, (unsigned long long) n_short_all, (unsigned long long) n_tok_all,
            llama_head_shortlist_size(c_on), k_low);
    if (a.zero_tail) {
        // the bound can exclude zero rows: some rows must have stood
        CHECK(n_short_all > 0);
    }

    llama_free(c_on);
    llama_free(c_off);
}

int main(int argc, char ** argv) {
    args_t a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        if (s == "-m" && i + 1 < argc) {
            a.model = argv[++i];
        } else if (s == "-t" && i + 1 < argc) {
            a.n_threads = atoi(argv[++i]);
        } else if (s == "-n" && i + 1 < argc) {
            a.n_prompts = atoi(argv[++i]);
        } else if (s == "-g" && i + 1 < argc) {
            a.n_gen = atoi(argv[++i]);
        } else if (s == "-k" && i + 1 < argc) {
            a.k_low = atoi(argv[++i]);
        } else if (s == "--zero-tail") {
            a.zero_tail = true;
        } else if (s == "--router-scale" && i + 1 < argc) {
            a.router_scale = (float) atof(argv[++i]);
        } else {
            fprintf(stderr, "usage: %s -m <model.gguf> [-t threads] [-n prompts] [-g tokens] [-k k_low] [--zero-tail] [--router-scale s]\n", argv[0]);
            return 2;
        }
    }
    if (a.model.empty()) {
        fprintf(stderr, "usage: %s -m <model.gguf> [-t threads] [-n prompts] [-g tokens] [-k k_low] [--zero-tail] [--router-scale s]\n", argv[0]);
        return 2;
    }

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t devs[] = { cpu, nullptr };
    mp.devices   = devs;
    mp.load_mode = LLAMA_LOAD_MODE_NONE; // the tensors below are written in place: no read-only mapping
    llama_model * model = llama_model_load_from_file(a.model.c_str(), mp);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", a.model.c_str());
        return 1;
    }

    // a random router weighs the experts almost alike, so a cut would keep them all: scale its F32 rows up, so that
    // the tokens favour a few experts as a trained router does
    if (a.router_scale != 1.0f) {
        int n_scaled = 0;
        for (auto & layer : model->layers) {
            ggml_tensor * t = layer.ffn_gate_inp;
            if (t == nullptr || t->type != GGML_TYPE_F32) {
                continue;
            }
            std::vector<float> v(ggml_nelements(t));
            ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
            for (auto & x : v) {
                x *= a.router_scale;
            }
            ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
            n_scaled++;
        }
        printf("the routers of %d layers scaled by %g\n", n_scaled, a.router_scale);
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (a.zero_tail) {
        const uint32_t k_low = a.k_low > 0 ? (uint32_t) a.k_low : (uint32_t) n_vocab/4;
        ggml_tensor * out = model->output;
        const size_t row = out->nb[1];
        std::vector<uint8_t> zero((size_t) (n_vocab - k_low)*row, 0);
        ggml_backend_tensor_set(out, zero.data(), (size_t) k_low*row, zero.size());
        printf("the head's rows from %u on are zero (%s)\n", k_low, ggml_type_name(out->type));
    }

    test_moe_tail(model, a);
    test_head_bound(model, a);

    llama_model_free(model);
    llama_backend_free();

    printf("%d checks, %d failed\n", n_checks, n_fail);
    return n_fail == 0 ? 0 : 1;
}
