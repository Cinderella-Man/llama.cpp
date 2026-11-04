// LLAMA_SPEC_ADAPT's draft-length rule (common_spec_adapt_choose and its helpers in common/speculative.h): the cost
// table, the allowed lengths, hand-computed cases, the rule against an exhaustive search of its objective on random
// drafts and cost tables, the literal per-position rule, the running speed-up, and the environment parser. No model.

#include "speculative.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static int n_checks = 0;
static int n_fail   = 0;

#define CHECK(cond) do { n_checks++; if (!(cond)) { n_fail++; fprintf(stderr, "%s:%d: failed: %s\n", __FILE__, __LINE__, #cond); } } while (0)

static bool near(double a, double b, double tol = 1e-9) {
    return std::fabs(a - b) <= tol * std::max(1.0, std::fabs(b));
}

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

static const char * k_env[] = {
    "LLAMA_SPEC_ADAPT", "LLAMA_SPEC_ADAPT_COST", "LLAMA_SPEC_ADAPT_WIDTHS", "LLAMA_SPEC_ADAPT_DRAFT_MS",
    "LLAMA_SPEC_ADAPT_S", "LLAMA_SPEC_ADAPT_S0", "LLAMA_SPEC_ADAPT_EMA", "LLAMA_SPEC_ADAPT_SCALE", "LLAMA_SPEC_ADAPT_LOG",
};

static void clear_env() {
    for (const char * n : k_env) {
        set_env(n, nullptr);
    }
}

// path probabilities from conditional ones
static std::vector<double> paths(const std::vector<float> & p) {
    std::vector<double> P;
    double x = 1.0;
    for (const float q : p) {
        x *= q;
        P.push_back(x);
    }
    return P;
}

// the objective the rule climbs, at d drafted tokens kept: STEP E/C, otherwise E - lambda C (lambda = scale S / T1)
static double objective(const common_spec_adapt_params & params, const std::vector<float> & p, int32_t d, double s) {
    const auto P = paths(p);
    double e = 1.0;
    for (int32_t i = 0; i < d; ++i) {
        e += P[i];
    }
    const double c  = params.draft_ms + common_spec_adapt_cost(params, d + 1);
    const double t1 = common_spec_adapt_cost(params, 1);
    if (params.s_mode == common_spec_adapt_params::S_STEP) {
        return e / c;
    }
    return e - params.scale * s / t1 * c;
}

// the note's rule taken literally, one position at a time: keep position j while P_j > scale * r_j * S, r_j = (T(j + 1)
// - T(j)) / T(1), S the chain's own speed-up so far (STEP) or the given one
static int32_t literal_rule(const common_spec_adapt_params & params, const std::vector<float> & p, int32_t n_max, double s) {
    const auto   P  = paths(p);
    const double t1 = common_spec_adapt_cost(params, 1);
    double e = 1.0;
    double c = params.draft_ms + t1;
    int32_t d = 0;
    while (d < n_max && d < (int32_t) p.size()) {
        const double r  = (common_spec_adapt_cost(params, d + 2) - common_spec_adapt_cost(params, d + 1)) / t1;
        const double ss = params.s_mode == common_spec_adapt_params::S_STEP ? e * t1 / c : s;
        if (!(P[d] > params.scale * r * ss)) {
            break;
        }
        e += P[d];
        c += r * t1;
        ++d;
    }
    return d;
}

static void test_cost() {
    common_spec_adapt_params p;
    CHECK(near(common_spec_adapt_cost(p, 1), 136.0));
    CHECK(near(common_spec_adapt_cost(p, 3), 244.6));
    CHECK(near(common_spec_adapt_cost(p, 5), 359.9));
    CHECK(near(common_spec_adapt_cost(p, 6), 359.9 + 54.5));   // the last slope
    CHECK(near(common_spec_adapt_cost(p, 17), 359.9 + 12 * 54.5));

    p.cost_ms = { 100.0 };
    CHECK(near(common_spec_adapt_cost(p, 1), 100.0));
    CHECK(near(common_spec_adapt_cost(p, 4), 400.0));           // one value: every extra token a whole pass

    p.cost_ms = { 90.0, 98.0 };
    CHECK(near(common_spec_adapt_cost(p, 9), 98.0 + 7 * 8.0));
}

static void test_lengths() {
    common_spec_adapt_params p; // widths 1, 3, 5, 9, 17
    using v = std::vector<int32_t>;
    CHECK(common_spec_adapt_lengths(p, 0)  == v({ 0 }));
    CHECK(common_spec_adapt_lengths(p, 1)  == v({ 0, 1 }));
    CHECK(common_spec_adapt_lengths(p, 2)  == v({ 0, 2 }));
    CHECK(common_spec_adapt_lengths(p, 3)  == v({ 0, 2, 3 }));
    CHECK(common_spec_adapt_lengths(p, 4)  == v({ 0, 2, 4 }));
    CHECK(common_spec_adapt_lengths(p, 5)  == v({ 0, 2, 4, 5 }));
    CHECK(common_spec_adapt_lengths(p, 7)  == v({ 0, 2, 4, 7 }));
    CHECK(common_spec_adapt_lengths(p, 8)  == v({ 0, 2, 4, 8 }));
    CHECK(common_spec_adapt_lengths(p, 16) == v({ 0, 2, 4, 8, 16 }));
    CHECK(common_spec_adapt_lengths(p, 64) == v({ 0, 2, 4, 8, 16 }));

    p.widths = { 4, 2, 2, 1, 7 };
    CHECK(common_spec_adapt_lengths(p, 5) == v({ 0, 1, 3, 5 }));
}

static void test_hand() {
    common_spec_adapt_params p; // Q583's table, 20 ms draft, S_RUN
    const auto L7 = common_spec_adapt_lengths(p, 7); // 0, 2, 4, 7

    // a sure draft keeps everything, a hopeless one nothing
    p.s_mode = common_spec_adapt_params::S_STEP;
    CHECK(common_spec_adapt_choose(p, L7, std::vector<float>(7, 1.0f), 1.0).n_keep == 7);
    CHECK(common_spec_adapt_choose(p, L7, std::vector<float>(7, 0.0f), 1.0).n_keep == 0);

    // the note's DFlash2 chain (cumulative 0.65 / 0.42 / 0.30, then x0.75) on today's table keeps two, as Q561 chose:
    // 0 -> 2 gains 1.07 against 0.872 * 108.6 / 136 = 0.696; 2 -> 4 gains 0.525 against 1.064 * 115.3 / 136 = 0.902
    const std::vector<float> chain = { 0.65f, 0.42f / 0.65f, 0.30f / 0.42f, 0.75f, 0.75f, 0.75f, 0.75f };
    {
        const auto ch = common_spec_adapt_choose(p, L7, chain, 1.0);
        CHECK(ch.n_keep == 2);
        CHECK(near(ch.e_tau, 2.07, 1e-6));
        CHECK(near(ch.cost_ms, 20.0 + 244.6, 1e-9));
        CHECK(near(ch.s, 2.07 * 136.0 / 264.6, 1e-6));
    }

    // a fixed S: 2.0 stops at two, 1.0 goes to four (p = 0.9 each; thresholds 1.597 / 1.696 at S = 2)
    p.s_mode = common_spec_adapt_params::S_FIXED;
    const auto L4 = common_spec_adapt_lengths(p, 4);
    p.s_fixed = 2.0;
    CHECK(common_spec_adapt_choose(p, L4, std::vector<float>(4, 0.9f), 0.0).n_keep == 2);
    p.s_fixed = 1.0;
    CHECK(common_spec_adapt_choose(p, L4, std::vector<float>(4, 0.9f), 0.0).n_keep == 4);

    // S_RUN reads the running value; scale 2 at S 1 is S 2
    p.s_mode = common_spec_adapt_params::S_RUN;
    CHECK(common_spec_adapt_choose(p, L4, std::vector<float>(4, 0.9f), 2.0).n_keep == 2);
    CHECK(common_spec_adapt_choose(p, L4, std::vector<float>(4, 0.9f), 1.0).n_keep == 4);
    p.scale = 2.0;
    CHECK(common_spec_adapt_choose(p, L4, std::vector<float>(4, 0.9f), 1.0).n_keep == 2);
    p.scale = 1.0;

    // only lengths the draft reaches: a one-token draft under widths 1, 3, 5 keeps nothing
    CHECK(common_spec_adapt_choose(p, L4, { 1.0f }, 1.0).n_keep == 0);
    CHECK(common_spec_adapt_choose(p, L4, { 1.0f, 1.0f, 1.0f }, 1.0).n_keep == 2);
    CHECK(common_spec_adapt_choose(p, L4, {}, 1.0).n_keep == 0);

    // probabilities outside [0, 1] are clamped
    CHECK(common_spec_adapt_choose(p, L4, std::vector<float>(4, 7.0f), 1.0).n_keep == 4);
    CHECK(common_spec_adapt_choose(p, L4, std::vector<float>(4, -1.0f), 1.0).n_keep == 0);

    // after the kernels (the note's section 3.1, "local" column) the same chain keeps four or more
    common_spec_adapt_params q;
    q.cost_ms = { 90.0, 98.1, 106.0, 113.4, 120.7, 127.6, 134.45, 141.3 };
    q.s_mode  = common_spec_adapt_params::S_STEP;
    CHECK(common_spec_adapt_choose(q, common_spec_adapt_lengths(q, 7), chain, 1.0).n_keep >= 4);
}

// the rule climbs its objective to the top on random monotone drafts and convex cost tables
static void test_random() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> u(0.0, 1.0);

    int n_cases = 0;
    int n_literal = 0;
    for (int it = 0; it < 4000; ++it) {
        common_spec_adapt_params p;

        // convex: increments a + b k (past the table the last slope stays, which keeps it convex)
        const double t1 = 50.0 + 150.0 * u(rng);
        const double a  = 60.0 * u(rng);
        const double b  = 8.0 * u(rng) * u(rng);
        p.cost_ms.assign(1, t1);
        const int n_tab = 2 + (int) (u(rng) * 20);
        for (int k = 1; k < n_tab; ++k) {
            p.cost_ms.push_back(p.cost_ms.back() + a + b * k);
        }
        p.draft_ms = 50.0 * u(rng);

        const int mode = it % 3;
        p.s_mode  = mode == 0 ? common_spec_adapt_params::S_STEP : mode == 1 ? common_spec_adapt_params::S_FIXED :
                                                                                common_spec_adapt_params::S_RUN;
        p.s_fixed = 0.3 + 3.0 * u(rng);
        p.scale   = mode == 0 ? 1.0 : 0.5 + u(rng);
        const double s_run = 0.3 + 3.0 * u(rng);
        const double s     = mode == 1 ? p.s_fixed : s_run;

        const int32_t n_max = 1 + (int32_t) (u(rng) * 16);
        if (u(rng) < 0.3) {
            p.widths.clear();
            for (int32_t k = 1; k <= n_max + 1; ++k) {
                p.widths.push_back(k);
            }
        } else if (u(rng) < 0.5) {
            p.widths = { 1, 2, 4, 7, 12 };
        }
        const auto lengths = common_spec_adapt_lengths(p, n_max);

        std::vector<float> draft;
        const int32_t n_draft = (int32_t) (u(rng) * (n_max + 1));
        const double  level   = u(rng);
        for (int32_t i = 0; i < n_draft; ++i) {
            draft.push_back((float) std::min(1.0, std::max(0.0, level + 0.3 * (u(rng) - 0.5))));
        }

        const auto ch = common_spec_adapt_choose(p, lengths, draft, s_run);

        // exhaustive: the best allowed length the draft reaches (ties to the shorter)
        int32_t best   = 0;
        double  best_f = objective(p, draft, 0, s);
        for (const int32_t d : lengths) {
            if (d == 0 || d > n_draft) {
                continue;
            }
            const double f = objective(p, draft, d, s);
            if (f > best_f + 1e-12) {
                best   = d;
                best_f = f;
            }
        }
        CHECK(ch.n_keep == best);
        if (ch.n_keep != best) {
            fprintf(stderr, "  case %d: mode %d n_max %d n_draft %d: rule %d, best %d\n", it, mode, n_max, n_draft, ch.n_keep, best);
        }

        // every allowed width: the literal per-position rule
        if ((int32_t) lengths.size() == n_max + 1) {
            CHECK(ch.n_keep == literal_rule(p, draft, n_max, s));
            n_literal++;
        }

        // what the choice reports
        double e = 1.0;
        const auto P = paths(draft);
        for (int32_t i = 0; i < ch.n_keep; ++i) {
            e += P[i];
        }
        CHECK(near(ch.e_tau, e, 1e-9));
        CHECK(near(ch.cost_ms, p.draft_ms + common_spec_adapt_cost(p, ch.n_keep + 1), 1e-9));
        n_cases++;
    }
    printf("random: %d cases, %d with every width checked against the literal rule\n", n_cases, n_literal);
}

static void test_state() {
    common_spec_adapt_params p; // 20 ms draft, T(1) 136
    common_spec_adapt_state st;
    CHECK(near(st.s(p), p.s0));            // before any step
    st.init(p);
    CHECK(near(st.s(p), p.s0));

    // one step: tok 1 -> 31/32 + 3/32, cost 136 -> 31/32 * 136 + 264.6/32
    st.update(p, 2, 2);
    const double tok  = 31.0 / 32 * 1.0 + 3.0 / 32;
    const double cost = 31.0 / 32 * 136.0 + 264.6 / 32;
    CHECK(near(st.s(p), 136.0 * tok / cost, 1e-12));

    // a long run of the same step converges to its speed-up, 3 * 136 / 264.6
    for (int i = 0; i < 2000; ++i) {
        st.update(p, 2, 2);
    }
    CHECK(near(st.s(p), 3.0 * 136.0 / 264.6, 1e-9));

    // an update without init starts from s0
    common_spec_adapt_state st2;
    st2.update(p, 0, 0);
    CHECK(near(st2.s(p), 136.0 * 1.0 / (31.0 / 32 * 136.0 + 156.0 / 32), 1e-12));
}

static void test_env() {
    clear_env();
    {
        const auto p = common_spec_adapt_params_from_env();
        CHECK(!p.enabled);
        CHECK(p.log.empty());
        CHECK(p.cost_ms.size() == 5 && near(p.cost_ms[0], 136.0));
        CHECK((p.widths == std::vector<int32_t>{ 1, 3, 5, 9, 17 }));
        CHECK(p.s_mode == common_spec_adapt_params::S_RUN);
    }
    set_env("LLAMA_SPEC_ADAPT", "0");
    CHECK(!common_spec_adapt_params_from_env().enabled);
    set_env("LLAMA_SPEC_ADAPT", "1");
    set_env("LLAMA_SPEC_ADAPT_COST", "90, 98.1;106 # a comment");
    set_env("LLAMA_SPEC_ADAPT_WIDTHS", "1,2,3");
    set_env("LLAMA_SPEC_ADAPT_DRAFT_MS", "0");
    set_env("LLAMA_SPEC_ADAPT_S", "step");
    set_env("LLAMA_SPEC_ADAPT_SCALE", "1.5");
    set_env("LLAMA_SPEC_ADAPT_LOG", "/dev/null");
    {
        const auto p = common_spec_adapt_params_from_env();
        CHECK(p.enabled);
        CHECK((p.cost_ms == std::vector<double>{ 90.0, 98.1, 106.0 }));
        CHECK((p.widths == std::vector<int32_t>{ 1, 2, 3 }));
        CHECK(near(p.draft_ms, 0.0));
        CHECK(p.s_mode == common_spec_adapt_params::S_STEP);
        CHECK(near(p.scale, 1.5));
        CHECK(p.log == "/dev/null");
    }
    set_env("LLAMA_SPEC_ADAPT_S", "1.7");
    {
        const auto p = common_spec_adapt_params_from_env();
        CHECK(p.s_mode == common_spec_adapt_params::S_FIXED);
        CHECK(near(p.s_fixed, 1.7));
    }

    // a cost file
    {
        const std::string path = "test-spec-adapt-cost.txt";
        FILE * f = fopen(path.c_str(), "w");
        CHECK(f != nullptr);
        if (f) {
            fputs("# T(k) in ms\n100\n150   # k = 2\n\n210\n", f);
            fclose(f);
            set_env("LLAMA_SPEC_ADAPT_COST", path.c_str());
            const auto p = common_spec_adapt_params_from_env();
            CHECK((p.cost_ms == std::vector<double>{ 100.0, 150.0, 210.0 }));
            remove(path.c_str());
        }
    }

    // malformed values throw
    auto throws = [](const char * name, const char * value) {
        clear_env();
        set_env(name, value);
        bool thrown = false;
        try {
            common_spec_adapt_params_from_env();
        } catch (const std::invalid_argument &) {
            thrown = true;
        }
        return thrown;
    };
    CHECK(throws("LLAMA_SPEC_ADAPT_COST", "136,abc"));
    CHECK(throws("LLAMA_SPEC_ADAPT_COST", "136,0"));
    CHECK(throws("LLAMA_SPEC_ADAPT_COST", "# nothing"));
    CHECK(throws("LLAMA_SPEC_ADAPT_WIDTHS", "1,2.5"));
    CHECK(throws("LLAMA_SPEC_ADAPT_WIDTHS", "0,3"));
    CHECK(throws("LLAMA_SPEC_ADAPT_DRAFT_MS", "-1"));
    CHECK(throws("LLAMA_SPEC_ADAPT_S", "fast"));
    CHECK(throws("LLAMA_SPEC_ADAPT_EMA", "0"));
    CHECK(throws("LLAMA_SPEC_ADAPT_SCALE", "1,2"));
    clear_env();
}

int main() {
    test_cost();
    test_lengths();
    test_hand();
    test_random();
    test_state();
    test_env();

    printf("%d/%d checks passed\n", n_checks - n_fail, n_checks);
    return n_fail == 0 ? 0 : 1;
}
