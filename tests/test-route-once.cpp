// Route once per verify block (Q860, llama_set_moe_route_once) on a model file (a random qwen4exp from
// test-llama-archs, or a real model), on the CPU backend:
//
// - off, k = 1 and a dry run (k = 5): the logits of every prompt equal the switch-off logits bit for bit;
// - k = 2, 3, 5 and n (the whole micro-batch): per MoE layer, from the graph's own tensors (the eval callback's
//   router probabilities, top-k and normalized weights), a block's first token reads its own top-k, every later token
//   reads exactly its block's first token's experts, in its own order and with its own normalized weights, so every
//   token's experts are a subset of its block's first token's; blocks restart at a sequence's start (two sequences in
//   one micro-batch); a layer cropped to the outputs takes its block's first output; the logits differ from off;
// - max: micro-batches above it route as usual (bit for bit the switch-off logits);
// - the histogram: every full block reads k_used distinct experts, no later token reads outside its block's set, and
//   the union under independent routing is larger.
//
// With -d <device> (e.g. CUDA0) the model runs on that device: the same checks, the exact arms within rounding (nmse
// <= 1e-6) instead of bit for bit, since the switch-off graph may take a fused router kernel there.
//
// Usage: test-route-once -m <model.gguf> [-d device] [-t threads] [-n prompts] [--router-scale s]

#include "../src/llama-model.h"
#include "../src/llama-ext.h"

#include "common.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

static int n_checks = 0;
static int n_fail   = 0;

#define CHECK(cond) do { n_checks++; if (!(cond)) { n_fail++; fprintf(stderr, "%s:%d: failed: %s\n", __FILE__, __LINE__, #cond); } } while (0)

struct args_t {
    std::string model;
    int   n_threads    = 4;
    int   n_prompts    = 16;
    float router_scale = 1.0f;
    std::string device;    // a backend device for the model (e.g. CUDA0); empty: the CPU
};

static const int N_UBATCH = 64;

// the router's tensors of one decode, by MoE layer: probabilities [n_expert, n_rows], the full argsort (its first
// n_used entries are the experts read), the normalized weights [n_used, n_rows]
struct layer_capture {
    int64_t n_expert = 0;
    int64_t n_rows   = 0;
    int64_t n_used   = 0;
    std::vector<float>   probs;
    std::vector<int32_t> order;
    std::vector<float>   weights;
};

struct capture_t {
    bool on = false;
    std::map<int, layer_capture> layers;
};

static int layer_of(const char * name, const char * prefix) {
    const size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) != 0) {
        return -1;
    }
    char * end = nullptr;
    const long il = strtol(name + n, &end, 10);
    return end != name + n && *end == '\0' ? (int) il : -1;
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * c = (capture_t *) user_data;
    const int il_p = layer_of(t->name, "ffn_moe_probs-");
    const int il_a = layer_of(t->name, "ffn_moe_argsort-");
    const int il_w = layer_of(t->name, "ffn_moe_weights_norm-");
    const bool want = c->on && (il_p >= 0 || il_a >= 0 || il_w >= 0);
    if (ask || !want) {
        return ask ? want : true;
    }
    if (!ggml_is_contiguous(t)) {
        fprintf(stderr, "%s is not contiguous\n", t->name);
        exit(1);
    }
    if (il_p >= 0) {
        auto & L = c->layers[il_p];
        L.n_expert = t->ne[0];
        L.n_rows   = t->ne[1];
        L.probs.resize(ggml_nelements(t));
        ggml_backend_tensor_get(t, L.probs.data(), 0, ggml_nbytes(t));
    } else if (il_a >= 0) {
        auto & L = c->layers[il_a];
        L.order.resize(ggml_nelements(t));
        ggml_backend_tensor_get(t, L.order.data(), 0, ggml_nbytes(t));
    } else {
        auto & L = c->layers[il_w];
        L.n_used = t->ne[0];
        L.weights.resize(ggml_nelements(t));
        ggml_backend_tensor_get(t, L.weights.data(), 0, ggml_nbytes(t));
    }
    return true;
}

static std::vector<llama_token> make_prompt(int i, int n_vocab, int len_min, int len_max) {
    std::mt19937 rng(2000 + i);
    std::uniform_int_distribution<int> len(len_min, len_max);
    std::uniform_int_distribution<int> tok(0, n_vocab - 1);
    std::vector<llama_token> p(len(rng));
    for (auto & t : p) {
        t = tok(rng);
    }
    return p;
}

static llama_context * make_ctx(llama_model * model, int n_threads, capture_t * cap) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx             = 512;
    cp.n_batch           = 256;
    cp.n_ubatch          = N_UBATCH;
    cp.n_seq_max         = 2;
    cp.n_threads         = n_threads;
    cp.n_threads_batch   = n_threads;
    cp.cb_eval           = cb_eval;
    cp.cb_eval_user_data = cap;
    return llama_init_from_model(model, cp);
}

// one decode of the given sequences (each from position 0), outputs where out(seq, i); the logits of every output
static std::vector<float> decode(llama_context * ctx, const std::vector<std::vector<llama_token>> & seqs, int n_vocab,
        bool (*out)(int, int)) {
    llama_memory_clear(llama_get_memory(ctx), true);
    common_batch batch(ctx);
    std::vector<int32_t> outs; // the batch indices with an output
    int32_t idx = 0;
    for (size_t s = 0; s < seqs.size(); ++s) {
        for (size_t i = 0; i < seqs[s].size(); ++i, ++idx) {
            const bool o = out((int) s, (int) i);
            batch.add(seqs[s][i], (llama_pos) i, (llama_seq_id) s, o);
            if (o) {
                outs.push_back(idx);
            }
        }
    }
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
        fprintf(stderr, "decode failed\n");
        exit(1);
    }
    std::vector<float> res;
    res.reserve(outs.size()*n_vocab);
    for (const int32_t j : outs) {
        const float * l = llama_get_logits_ith(ctx, j);
        if (l == nullptr) {
            fprintf(stderr, "no logits for batch index %d\n", j);
            exit(1);
        }
        res.insert(res.end(), l, l + n_vocab);
    }
    return res;
}

static bool out_all(int, int) { return true; }
static bool out_odd(int, int i) { return i % 2 == 1; }

// the routing of every captured layer against the leaders: lead[r] is row r's block's first row (r itself when off)
struct route_check {
    uint64_t n_rows      = 0;
    uint64_t n_set_bad   = 0; // rows whose experts are not their leader's own top-k
    uint64_t n_order_bad = 0; // rows whose experts are not in their own descending order
    uint64_t n_w_bad     = 0; // rows whose weights are not their own normalized probabilities
    uint64_t n_later     = 0; // later rows (r != lead[r])
    uint64_t n_changed   = 0; // later rows whose experts differ from their own top-k
};

static std::vector<int32_t> top_k_of(const float * p, int64_t n_expert, int64_t k) {
    std::vector<int32_t> ids(n_expert);
    for (int64_t e = 0; e < n_expert; ++e) {
        ids[e] = (int32_t) e;
    }
    std::partial_sort(ids.begin(), ids.begin() + k, ids.end(), [&](int32_t a, int32_t b) { return p[a] > p[b]; });
    ids.resize(k);
    std::sort(ids.begin(), ids.end());
    return ids;
}

static void check_layer(const layer_capture & L, const std::vector<int32_t> & lead, route_check & rc) {
    const int64_t n_exp  = L.n_expert;
    const int64_t n_used = L.n_used;
    if ((int64_t) lead.size() != L.n_rows || (int64_t) L.order.size() != n_exp*L.n_rows ||
            (int64_t) L.weights.size() != n_used*L.n_rows || n_used <= 0) {
        fprintf(stderr, "capture shape: rows %lld, leaders %zu, order %zu, weights %zu, used %lld\n",
                (long long) L.n_rows, lead.size(), L.order.size(), L.weights.size(), (long long) n_used);
        rc.n_set_bad++;
        return;
    }
    for (int64_t r = 0; r < L.n_rows; ++r) {
        const float   * p   = L.probs.data() + r*n_exp;
        const int32_t * ids = L.order.data() + r*n_exp;
        const float   * w   = L.weights.data() + r*n_used;

        std::vector<int32_t> got(ids, ids + n_used);
        std::sort(got.begin(), got.end());
        const auto want = top_k_of(L.probs.data() + (int64_t) lead[r]*n_exp, n_exp, n_used);
        rc.n_set_bad += got != want;

        bool ordered = true;
        double sum = 0.0;
        for (int64_t e = 0; e < n_used; ++e) {
            ordered = ordered && (e == 0 || p[ids[e - 1]] >= p[ids[e]]);
            sum += p[ids[e]];
        }
        rc.n_order_bad += !ordered;

        bool w_ok = true;
        for (int64_t e = 0; e < n_used; ++e) {
            const double ref = p[ids[e]]/sum;
            w_ok = w_ok && std::fabs(w[e] - ref) <= 1e-5*std::max(1.0, std::fabs(ref)) + 1e-7;
        }
        rc.n_w_bad += !w_ok;

        if (lead[r] != r) {
            rc.n_later++;
            rc.n_changed += got != top_k_of(p, n_exp, n_used);
        }
        rc.n_rows++;
    }
}

// the leaders of a micro-batch laid out as the sequences one after another (split_equal's order), blocks of k
// consecutive positions from each sequence's first token; a cropped layer has the outputs' rows only, each block led by
// its first output
static std::vector<int32_t> leaders(const std::vector<int> & lens, uint32_t k, bool (*out)(int, int), bool cropped) {
    std::vector<int32_t> lead;
    int32_t row = 0;
    for (size_t s = 0; s < lens.size(); ++s) {
        int32_t block_first_row = -1;
        int     block_start_pos = -1;
        for (int i = 0; i < lens[s]; ++i) {
            const int start = k <= 1 ? i : i - i % (int) k;
            if (start != block_start_pos) {
                block_start_pos = start;
                block_first_row = -1;
            }
            if (cropped && !out((int) s, i)) {
                continue;
            }
            if (block_first_row < 0) {
                block_first_row = row;
            }
            lead.push_back(k == 0 ? row : block_first_row);
            row++;
        }
    }
    return lead;
}

struct arm_t {
    const char * name;
    uint32_t k;
    uint32_t max;
    bool     hist;
    bool     dry;
};

static void run_arms(llama_model * model, const args_t & a) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int n_used  = (int) model->hparams.n_expert_used_max();
    if (model->hparams.n_expert == 0 || n_used < 1) {
        printf("route once: the model has no routed experts: skipped\n");
        return;
    }

    capture_t cap_off;
    capture_t cap_on;
    llama_context * c_off = make_ctx(model, a.n_threads, &cap_off);

    const arm_t arms[] = {
        { "k=1",             1,        0, false, false },
        { "k=1 hist",        1,        0, true,  false },
        { "k=5 dry hist",    5,        0, true,  true  },
        { "k=2 hist",        2,        0, true,  false },
        { "k=3",             3,        0, false, false },
        { "k=5 hist",        5,        0, true,  false },
        { "k=n hist",        N_UBATCH, 0, true,  false },
        { "k=5 max=4",       5,        4, false, false },
    };

    // the layouts: one sequence with every output, one with the odd positions as outputs (the last layer cropped),
    // two sequences of one length in one micro-batch
    struct layout_t {
        const char * name;
        int  n_seqs;
        bool (*out)(int, int);
    };
    const layout_t layouts[] = {
        { "one sequence",      1, out_all },
        { "odd outputs",       1, out_odd },
        { "two sequences",     2, out_all },
    };

    for (const auto & arm : arms) {
        llama_context * c_on = make_ctx(model, a.n_threads, &cap_on);
        llama_set_moe_route_once(c_on, arm.k, arm.max, arm.hist, arm.dry);

        // the switch's routing applies to micro-batches of at most max tokens; k = 1 and a dry run read every token's
        // own experts
        const bool exact = arm.k <= 1 || arm.dry;

        for (const auto & lay : layouts) {
            bool     identical  = true;
            double   err2       = 0.0; // the squared differences of the logits, and the squares of the off ones
            double   ref2       = 0.0;
            uint64_t n_pos      = 0;
            route_check rc_on;
            route_check rc_off;
            for (int i = 0; i < a.n_prompts; ++i) {
                std::vector<std::vector<llama_token>> seqs;
                std::vector<int> lens;
                const int len_max = lay.n_seqs == 2 ? N_UBATCH/2 : N_UBATCH;
                // the first prompt fills the micro-batch, so that k = n has full blocks
                const auto p0 = make_prompt(i, n_vocab, i == 0 ? len_max : 8, len_max);
                for (int s = 0; s < lay.n_seqs; ++s) {
                    auto p = p0;
                    if (s > 0) {
                        std::reverse(p.begin(), p.end());
                    }
                    seqs.push_back(p);
                    lens.push_back((int) p.size());
                }

                cap_off.on = true;
                cap_off.layers.clear();
                const auto l_off = decode(c_off, seqs, n_vocab, lay.out);
                cap_off.on = false;

                cap_on.on = true;
                cap_on.layers.clear();
                const auto l_on = decode(c_on, seqs, n_vocab, lay.out);
                cap_on.on = false;

                identical = identical && l_off.size() == l_on.size() &&
                    memcmp(l_off.data(), l_on.data(), l_off.size()*sizeof(float)) == 0;
                for (size_t j = 0; j < std::min(l_off.size(), l_on.size()); ++j) {
                    err2 += ((double) l_on[j] - l_off[j])*((double) l_on[j] - l_off[j]);
                    ref2 += (double) l_off[j]*l_off[j];
                }
                n_pos += l_off.size()/n_vocab;

                int n_tok = 0;
                for (int len : lens) {
                    n_tok += len;
                }
                const bool routed = !(arm.max > 0 && n_tok > (int) arm.max) && !arm.dry;
                const uint32_t k_eff = routed ? arm.k : 0;
                for (const auto & [il, L] : cap_on.layers) {
                    const bool cropped = L.n_rows != n_tok;
                    // a cropped layer of one row has nothing to share
                    const uint32_t k_l = cropped && L.n_rows <= 1 ? 0 : k_eff;
                    check_layer(L, leaders(lens, k_l, lay.out, cropped), rc_on);
                }
                for (const auto & [il, L] : cap_off.layers) {
                    check_layer(L, leaders(lens, 0, lay.out, L.n_rows != n_tok), rc_off);
                }
            }

            const double nmse = ref2 > 0.0 ? err2/ref2 : 0.0;
            printf("route once %-13s %-14s: %s (nmse %.3g) over %llu outputs; %llu router rows, %llu later; experts "
                    "not the leader's own top-k %llu, out of order %llu, weights off %llu; later rows with other "
                    "experts than their own top-k %llu (off: %llu rows, %llu bad)\n", arm.name, lay.name,
                    identical ? "identical logits" : "logits differ", nmse, (unsigned long long) n_pos,
                    (unsigned long long) rc_on.n_rows, (unsigned long long) rc_on.n_later,
                    (unsigned long long) rc_on.n_set_bad, (unsigned long long) rc_on.n_order_bad,
                    (unsigned long long) rc_on.n_w_bad, (unsigned long long) rc_on.n_changed,
                    (unsigned long long) rc_off.n_rows,
                    (unsigned long long) (rc_off.n_set_bad + rc_off.n_order_bad + rc_off.n_w_bad));

            CHECK(rc_off.n_rows > 0);
            CHECK(rc_off.n_set_bad == 0 && rc_off.n_order_bad == 0 && rc_off.n_w_bad == 0);
            CHECK(rc_on.n_rows == rc_off.n_rows);
            CHECK(rc_on.n_set_bad == 0);
            CHECK(rc_on.n_order_bad == 0);
            CHECK(rc_on.n_w_bad == 0);
            if (exact || arm.max > 0) {
                // bit for bit on the CPU; a GPU's off path may take a fused router kernel, so there within rounding
                CHECK(identical || (!a.device.empty() && nmse <= 1e-6));
                CHECK(rc_on.n_later == 0);
            } else {
                CHECK(!identical);
                CHECK(rc_on.n_later > 0);
                CHECK(rc_on.n_changed > 0);
            }
        }

        if (arm.hist) {
            const int n_counts = arm.k*n_used + 2;
            std::vector<uint64_t> read(n_counts, 0);
            std::vector<uint64_t> indep(n_counts, 0);
            uint64_t n_outside = 0;
            const uint64_t n_pairs = llama_moe_route_once_counts(c_on, read.data(), indep.data(), n_counts, &n_outside);
            double m_read  = 0.0;
            double m_indep = 0.0;
            for (int d = 0; d < n_counts; ++d) {
                m_read  += (double) d*read[d];
                m_indep += (double) d*indep[d];
            }
            m_read  /= std::max<uint64_t>(n_pairs, 1);
            m_indep /= std::max<uint64_t>(n_pairs, 1);
            printf("route once %-13s histogram: %llu full blocks x layers, distinct experts read mean %.2f, "
                    "independent %.2f, later token-layers reading outside their block's set %llu\n", arm.name,
                    (unsigned long long) n_pairs, m_read, m_indep, (unsigned long long) n_outside);

            CHECK(n_pairs > 0);
            if (arm.dry) {
                CHECK(read == indep);
                CHECK(m_indep > n_used || arm.k == 1);
            } else {
                CHECK(read[n_used] == n_pairs);
                CHECK(n_outside == 0);
                if (arm.k == 1) {
                    CHECK(indep[n_used] == n_pairs);
                } else {
                    CHECK(m_indep > n_used);
                }
            }
        }

        llama_free(c_on);
    }

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
        } else if (s == "--router-scale" && i + 1 < argc) {
            a.router_scale = (float) atof(argv[++i]);
        } else if (s == "-d" && i + 1 < argc) {
            a.device = argv[++i];
        } else {
            fprintf(stderr, "usage: %s -m <model.gguf> [-d device] [-t threads] [-n prompts] [--router-scale s]\n", argv[0]);
            return 2;
        }
    }
    if (a.model.empty()) {
        fprintf(stderr, "usage: %s -m <model.gguf> [-d device] [-t threads] [-n prompts] [--router-scale s]\n", argv[0]);
        return 2;
    }

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t dev = a.device.empty() ? ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)
                                              : ggml_backend_dev_by_name(a.device.c_str());
    if (dev == nullptr) {
        fprintf(stderr, "no device %s\n", a.device.c_str());
        return 1;
    }
    ggml_backend_dev_t devs[] = { dev, nullptr };
    mp.devices      = devs;
    mp.n_gpu_layers = 999;
    mp.load_mode = LLAMA_LOAD_MODE_NONE; // the routers below are written in place: no read-only mapping
    llama_model * model = llama_model_load_from_file(a.model.c_str(), mp);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", a.model.c_str());
        return 1;
    }

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

    run_arms(model, a);

    llama_model_free(model);
    llama_backend_free();

    printf("%d checks, %d failed\n", n_checks, n_fail);
    return n_fail == 0 ? 0 : 1;
}
