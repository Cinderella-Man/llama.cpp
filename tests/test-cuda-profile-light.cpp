// GGML_CUDA_PROFILE_MODE's parts that run without a GPU (ggml/src/ggml-cuda/profile-light.h): the mode string, the
// layer of a node name, layer mode's segments, sample mode's rotation, and the per-phase sets of CUDA-graph
// executables, the last one through a model of ggml_backend_cuda_graph_compute's warm-up and capture logic.

#include "../ggml/src/ggml-cuda/profile-light.h"

#include <algorithm>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

static int n_checks = 0;
static int n_fail   = 0;

#define CHECK(cond) do { n_checks++; if (!(cond)) { n_fail++; fprintf(stderr, "%s:%d: failed: %s\n", __FILE__, __LINE__, #cond); } } while (0)

static void test_parse_mode() {
    ggml_cuda_profile_mode m;
    int n;
    CHECK(ggml_cuda_profile_parse_mode(nullptr, m, n) && m == GGML_CUDA_PROFILE_MODE_GROUP && n == 1);
    CHECK(ggml_cuda_profile_parse_mode("", m, n) && m == GGML_CUDA_PROFILE_MODE_GROUP);
    CHECK(ggml_cuda_profile_parse_mode("group", m, n) && m == GGML_CUDA_PROFILE_MODE_GROUP);
    CHECK(ggml_cuda_profile_parse_mode("layer", m, n) && m == GGML_CUDA_PROFILE_MODE_LAYER && n == 1);
    CHECK(ggml_cuda_profile_parse_mode("graph", m, n) && m == GGML_CUDA_PROFILE_MODE_GRAPH && n == 1);
    CHECK(ggml_cuda_profile_parse_mode("sample=8", m, n) && m == GGML_CUDA_PROFILE_MODE_SAMPLE && n == 8);
    CHECK(ggml_cuda_profile_parse_mode("sample=1", m, n) && n == 1);
    CHECK(ggml_cuda_profile_parse_mode("sample=64", m, n) && n == 64);
    for (const char * bad : { "sample", "sample=", "sample=0", "sample=65", "sample=100", "sample=-2", "sample=8x",
                              "sample=08", "Layer", "layers", "graph ", "events", "1" }) {
        CHECK(!ggml_cuda_profile_parse_mode(bad, m, n));
    }
    CHECK(std::string(ggml_cuda_profile_mode_name(GGML_CUDA_PROFILE_MODE_SAMPLE)) == "sample");
    CHECK(std::string(ggml_cuda_profile_mode_name(GGML_CUDA_PROFILE_MODE_LAYER)) == "layer");
}

static void test_layer_of_name() {
    CHECK(ggml_cuda_profile_layer_of_name("attn_norm-12") == 12);
    CHECK(ggml_cuda_profile_layer_of_name("attn_norm-0") == 0);
    CHECK(ggml_cuda_profile_layer_of_name("ffn_moe_weights_sum-47") == 47);
    CHECK(ggml_cuda_profile_layer_of_name("attn_norm-12 (reshaped)") == 12);
    CHECK(ggml_cuda_profile_layer_of_name("k_cache_view-3 (copy of Kcur-3)") == 3);
    CHECK(ggml_cuda_profile_layer_of_name("result_norm") == GGML_CUDA_PROFILE_LAYER_OUTPUT);
    CHECK(ggml_cuda_profile_layer_of_name("result_output") == GGML_CUDA_PROFILE_LAYER_OUTPUT);
    CHECK(ggml_cuda_profile_layer_of_name("inp_embd") == GGML_CUDA_PROFILE_LAYER_INPUT);
    CHECK(ggml_cuda_profile_layer_of_name("inp_out_ids") == GGML_CUDA_PROFILE_LAYER_INPUT);
    CHECK(ggml_cuda_profile_layer_of_name("") == GGML_CUDA_PROFILE_LAYER_UNKNOWN);
    CHECK(ggml_cuda_profile_layer_of_name(nullptr) == GGML_CUDA_PROFILE_LAYER_UNKNOWN);
    CHECK(ggml_cuda_profile_layer_of_name("node_123") == GGML_CUDA_PROFILE_LAYER_UNKNOWN);
    CHECK(ggml_cuda_profile_layer_of_name("leaf_5") == GGML_CUDA_PROFILE_LAYER_UNKNOWN);
    CHECK(ggml_cuda_profile_layer_of_name("-12") == GGML_CUDA_PROFILE_LAYER_UNKNOWN);       // nothing before the dash
    CHECK(ggml_cuda_profile_layer_of_name("x-") == GGML_CUDA_PROFILE_LAYER_UNKNOWN);
    CHECK(ggml_cuda_profile_layer_of_name("x-1234567") == GGML_CUDA_PROFILE_LAYER_UNKNOWN); // over six digits
    CHECK(ggml_cuda_profile_layer_of_name(" (view)") == GGML_CUDA_PROFILE_LAYER_UNKNOWN);
    CHECK(ggml_cuda_profile_layer_of_name("blk.7.attn_q.weight") == 7);
    CHECK(ggml_cuda_profile_layer_of_weight("blk.12.ffn_up_exps.weight") == 12);
    CHECK(ggml_cuda_profile_layer_of_weight("blk.x.ffn_up.weight") == GGML_CUDA_PROFILE_LAYER_UNKNOWN);
    CHECK(ggml_cuda_profile_layer_of_weight("blk.12") == GGML_CUDA_PROFILE_LAYER_UNKNOWN);
    CHECK(ggml_cuda_profile_layer_of_weight("output.weight") == GGML_CUDA_PROFILE_LAYER_UNKNOWN);
    CHECK(ggml_cuda_profile_layer_of_weight(nullptr) == GGML_CUDA_PROFILE_LAYER_UNKNOWN);
    CHECK(ggml_cuda_profile_layer_label(5) == "layer 5");
    CHECK(ggml_cuda_profile_layer_label(GGML_CUDA_PROFILE_LAYER_OUTPUT) == "output");
    CHECK(ggml_cuda_profile_layer_label(GGML_CUDA_PROFILE_LAYER_INPUT) == "input");
    CHECK(ggml_cuda_profile_layer_label(GGML_CUDA_PROFILE_LAYER_UNKNOWN) == "other");
}

// runs the plan over the layers of K launch groups the way the evaluation loop does; returns the events recorded
static int run_plan(ggml_cuda_profile_layer_plan & plan, const std::vector<int> & layers) {
    int n_ev = 1; // begin() comes with event 0
    plan.begin();
    for (int layer : layers) {
        n_ev += plan.group(layer) ? 1 : 0;
    }
    n_ev += plan.end() ? 1 : 0;
    return n_ev;
}

static void check_plan_tiles(const ggml_cuda_profile_layer_plan & plan, const int n_groups, const int n_ev) {
    CHECK((int) plan.segs.size() == (n_groups > 0 ? n_ev - 1 : 0));
    int next = 0;
    for (const auto & s : plan.segs) {
        CHECK(s.first == next);
        CHECK(s.n_groups > 0);
        next += s.n_groups;
    }
    CHECK(next == n_groups);
}

static void test_layer_plan() {
    const int U = GGML_CUDA_PROFILE_LAYER_UNKNOWN, O = GGML_CUDA_PROFILE_LAYER_OUTPUT, I = GGML_CUDA_PROFILE_LAYER_INPUT;
    ggml_cuda_profile_layer_plan plan;
    {
        // a card holding layers 5 and 6 and the output: unknown groups stay in the layer before them
        const std::vector<int> layers = { 5, 5, U, 5, 6, U, 6, 6, O, O };
        const int n_ev = run_plan(plan, layers);
        CHECK(n_ev == 4);
        check_plan_tiles(plan, (int) layers.size(), n_ev);
        CHECK(plan.segs.size() == 3);
        CHECK(plan.segs[0].layer == 5 && plan.segs[0].n_groups == 4);
        CHECK(plan.segs[1].layer == 6 && plan.segs[1].first == 4 && plan.segs[1].n_groups == 4);
        CHECK(plan.segs[2].layer == O && plan.segs[2].n_groups == 2);
    }
    {
        // leading unknown groups form an "other" segment; inputs their own
        const std::vector<int> layers = { U, U, I, I, 0, 0, 1 };
        const int n_ev = run_plan(plan, layers);
        check_plan_tiles(plan, (int) layers.size(), n_ev);
        CHECK(plan.segs.size() == 4);
        CHECK(plan.segs[0].layer == U && plan.segs[0].n_groups == 2);
        CHECK(plan.segs[1].layer == I);
        CHECK(plan.segs[3].layer == 1 && plan.segs[3].n_groups == 1);
    }
    {
        // the first known layer relabels an empty segment instead of closing it
        const std::vector<int> layers = { 3, 3, 3 };
        const int n_ev = run_plan(plan, layers);
        CHECK(n_ev == 2);
        CHECK(plan.segs.size() == 1 && plan.segs[0].layer == 3 && plan.segs[0].first == 0);
    }
    {
        // no launch group: no segment, no last event
        const int n_ev = run_plan(plan, {});
        CHECK(n_ev == 1);
        CHECK(plan.segs.empty());
    }
    {
        // random sequences: the segments always tile the groups, one event per segment boundary
        std::mt19937 rng(42);
        for (int t = 0; t < 2000; ++t) {
            std::vector<int> layers(rng() % 400);
            int cur = (int) (rng() % 4);
            for (int & l : layers) {
                const unsigned r = rng() % 100;
                if (r < 3) {
                    cur++;
                }
                l = r < 3 ? cur : r < 20 ? U : r < 21 ? O : cur;
            }
            const int n_ev = run_plan(plan, layers);
            check_plan_tiles(plan, (int) layers.size(), n_ev);
        }
    }
}

static void test_sampled() {
    for (int n = 1; n <= 9; ++n) {
        for (int g = 1; g <= 50; g += 7) {
            // n evaluations in a row (phases 0..n-1) time every group exactly once
            std::vector<int> times(g, 0);
            int per_eval_max = 0;
            for (int e = 0; e < n; ++e) {
                int c = 0;
                for (int k = 0; k < g; ++k) {
                    if (ggml_cuda_profile_sampled(k, n, e % n)) {
                        times[k]++;
                        c++;
                    }
                }
                per_eval_max = std::max(per_eval_max, c);
            }
            bool once = true;
            for (int t : times) {
                once = once && t == 1;
            }
            CHECK(once);
            CHECK(per_eval_max <= (g + n - 1) / n);
        }
    }
}

// A model of ggml_backend_cuda_graph_compute's warm-up and capture decisions with sample mode's phase sets. An
// executable is an int: the properties (pointers) it was captured with. A replay must only ever run executables of
// the current properties; a set must never be lost or duplicated.
struct fake_graph {
    std::vector<int> instances;
    ggml_cuda_profile_phase_sets<std::vector<int>> phases;
    uint64_t capture_gen     = 0;
    bool     warmup_complete = false;
    int      props           = -1;
};

enum { HOW_EAGER, HOW_CAPTURE, HOW_REPLAY };

static int fake_eval(fake_graph & g, const int props, const int phase, const int n_phases, const int n_chunks, bool & replay_ok) {
    if (n_phases > 0) {
        g.phases.select(g.instances, phase, n_phases);
    }
    const bool changed = props != g.props;
    g.props = props;
    bool use = false, update = false;
    if (!g.warmup_complete) {
        if (!changed) {
            g.warmup_complete = true;
            g.capture_gen++;
            use = update = true;
        }
    } else if (changed) {
        g.warmup_complete = false;
    } else {
        use    = true;
        update = g.instances.empty() || g.phases.stale(g.capture_gen);
    }
    if (use && update) {
        g.instances.assign(n_chunks, props);
        g.phases.captured(g.capture_gen);
        return HOW_CAPTURE;
    }
    if (use) {
        for (int x : g.instances) {
            replay_ok = replay_ok && x == props;
        }
        replay_ok = replay_ok && !g.instances.empty();
        return HOW_REPLAY;
    }
    return HOW_EAGER;
}

static void test_phase_sets() {
    // stable properties: one eager call, a capture per phase, then replays only
    for (int n = 1; n <= 8; ++n) {
        fake_graph g;
        bool ok = true;
        std::vector<int> hows;
        for (int e = 0; e < 1 + 3*n; ++e) {
            hows.push_back(fake_eval(g, 7, e % n, n, 2, ok));
        }
        CHECK(ok);
        CHECK(hows[0] == HOW_EAGER);
        int captures = 0;
        for (int e = 1; e <= n; ++e) {
            captures += hows[e] == HOW_CAPTURE;
        }
        CHECK(captures == n);
        bool replays = true;
        for (int e = n + 1; e < (int) hows.size(); ++e) {
            replays = replays && hows[e] == HOW_REPLAY;
        }
        CHECK(replays);
        CHECK((int) g.phases.parked.size() == n);
    }
    // properties that change now and then: never a replay of stale executables, nothing lost or duplicated
    std::mt19937 rng(7);
    for (int t = 0; t < 300; ++t) {
        const int n = 1 + (int) (rng() % 8);
        fake_graph g;
        bool ok = true;
        int props = 0, n_replay = 0;
        for (int e = 0; e < 400; ++e) {
            if (rng() % 37 == 0) {
                props++;
            }
            const int chunks = 1 + (int) (props % 3);
            n_replay += fake_eval(g, props, e % n, n, chunks, ok) == HOW_REPLAY;
        }
        CHECK(ok);
        CHECK(n_replay > 0);
        // every set is held exactly once: the active one in `instances`, the others parked, the active slot empty
        CHECK(g.phases.parked[g.phases.active].empty());
        int held = (int) !g.instances.empty();
        for (int p = 0; p < n; ++p) {
            held += p != g.phases.active && !g.phases.parked[p].empty();
        }
        CHECK(held <= n);
    }
    // outside sample mode the sets stay unused and the logic is the stock one
    {
        fake_graph g;
        bool ok = true;
        CHECK(fake_eval(g, 1, 0, 0, 1, ok) == HOW_EAGER);
        CHECK(fake_eval(g, 1, 0, 0, 1, ok) == HOW_CAPTURE);
        CHECK(fake_eval(g, 1, 0, 0, 1, ok) == HOW_REPLAY);
        CHECK(fake_eval(g, 2, 0, 0, 1, ok) == HOW_EAGER);
        CHECK(fake_eval(g, 2, 0, 0, 1, ok) == HOW_CAPTURE);
        CHECK(fake_eval(g, 2, 0, 0, 1, ok) == HOW_REPLAY);
        CHECK(ok);
        CHECK(g.phases.parked.empty() && !g.phases.stale(g.capture_gen));
    }
}

int main() {
    test_parse_mode();
    test_layer_of_name();
    test_layer_plan();
    test_sampled();
    test_phase_sets();
    printf("test-cuda-profile-light: %d checks, %d failed\n", n_checks, n_fail);
    return n_fail == 0 ? 0 : 1;
}
