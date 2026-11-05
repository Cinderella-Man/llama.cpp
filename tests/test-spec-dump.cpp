// LLAMA_SPEC_DUMP's reduction of a DFlash2 selector lattice (common_speculative_lattice_info): softmax rows, forward
// marginals, the candidates kept per position (the k_max highest marginals plus the walked one), the conditional rows
// between kept candidates, and the walk's indices; the selector's greedy walk and the full block's record
// (common_speculative_full_block_info). Synthetic lattices with known answers; no model.

#include "speculative.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static int n_checks = 0;
static int n_fail   = 0;

#define CHECK(cond) do { n_checks++; if (!(cond)) { n_fail++; fprintf(stderr, "%s:%d: failed: %s\n", __FILE__, __LINE__, #cond); } } while (0)

static bool near(double a, double b, double tol = 1e-5) {
    return std::fabs(a - b) <= tol;
}

// a lattice of n_pos rows of stride floats: top_k ids, then top_k * top_k scores (position 0: every row the same)
static std::vector<float> make_lattice(int32_t n_pos, int32_t top_k, size_t stride, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.0f, 2.0f);
    std::vector<float> lat((size_t) n_pos * stride, 0.0f);
    for (int32_t i = 0; i < n_pos; ++i) {
        float * row = lat.data() + (size_t) i * stride;
        for (int32_t k = 0; k < top_k; ++k) {
            row[k] = (float) (1000 * (i + 1) + 7 * k); // distinct ids per position
        }
        float * s = row + top_k;
        if (i == 0) {
            std::vector<float> r(top_k);
            for (auto & x : r) {
                x = g(rng);
            }
            for (int32_t a = 0; a < top_k; ++a) {
                for (int32_t b = 0; b < top_k; ++b) {
                    s[a * top_k + b] = r[b];
                }
            }
        } else {
            for (int32_t a = 0; a < top_k * top_k; ++a) {
                s[a] = g(rng);
            }
        }
    }
    return lat;
}

// the reference: softmax rows and forward marginals in double
static void reference(const std::vector<float> & lat, int32_t n_pos, int32_t top_k, size_t stride,
        std::vector<std::vector<double>> & prob, std::vector<std::vector<double>> & marg) {
    prob.assign(n_pos, std::vector<double>((size_t) top_k * top_k));
    marg.assign(n_pos, std::vector<double>(top_k));
    for (int32_t i = 0; i < n_pos; ++i) {
        const float * s = lat.data() + (size_t) i * stride + top_k;
        for (int32_t a = 0; a < top_k; ++a) {
            double mx = -1e30, sum = 0.0;
            for (int32_t b = 0; b < top_k; ++b) {
                mx = std::max(mx, (double) s[a * top_k + b]);
            }
            for (int32_t b = 0; b < top_k; ++b) {
                sum += std::exp(s[a * top_k + b] - mx);
            }
            for (int32_t b = 0; b < top_k; ++b) {
                prob[i][(size_t) a * top_k + b] = std::exp(s[a * top_k + b] - mx) / sum;
            }
        }
        for (int32_t b = 0; b < top_k; ++b) {
            if (i == 0) {
                marg[i][b] = prob[i][b];
            } else {
                double m = 0.0;
                for (int32_t a = 0; a < top_k; ++a) {
                    m += marg[i - 1][a] * prob[i][(size_t) a * top_k + b];
                }
                marg[i][b] = m;
            }
        }
    }
}

// the candidate index (into the full top_k) of a kept id
static int32_t cand_index(const std::vector<float> & lat, size_t stride, int32_t i, int32_t top_k, llama_token id) {
    const float * row = lat.data() + (size_t) i * stride;
    for (int32_t k = 0; k < top_k; ++k) {
        if ((llama_token) row[k] == id) {
            return k;
        }
    }
    return -1;
}

// keep everything: the rows are the full softmax, the marginals order the ids, kept mass 1, all paths sum to 1
static void test_keep_all() {
    const int32_t n_pos = 4, top_k = 5;
    const size_t  stride = 40;
    const auto lat = make_lattice(n_pos, top_k, stride, 1);
    std::vector<std::vector<double>> prob, marg;
    reference(lat, n_pos, top_k, stride, prob, marg);

    const std::vector<int32_t> walk = { 2, 0, 4, 1 };
    common_speculative_draft_info info;
    info.k_max = 8; // more than top_k: keep all five
    common_speculative_lattice_info(info, lat.data(), stride, n_pos, top_k, walk);

    CHECK(info.lattice);
    CHECK((int32_t) info.ids.size() == n_pos && (int32_t) info.p.size() == n_pos && (int32_t) info.pick.size() == n_pos);
    std::vector<std::vector<int32_t>> idx(n_pos);
    for (int32_t i = 0; i < n_pos; ++i) {
        CHECK((int32_t) info.ids[i].size() == top_k);
        CHECK(near(info.kept[i], 1.0, 1e-5));
        for (const auto id : info.ids[i]) {
            idx[i].push_back(cand_index(lat, stride, i, top_k, id));
        }
        for (size_t j = 1; j < idx[i].size(); ++j) {
            CHECK(marg[i][idx[i][j - 1]] >= marg[i][idx[i][j]]); // highest marginal first
        }
        CHECK(info.pick[i] >= 0 && idx[i][info.pick[i]] == walk[i]);
    }
    CHECK((int32_t) info.p[0].size() == top_k);
    for (int32_t b = 0; b < top_k; ++b) {
        CHECK(near(info.p[0][b], prob[0][idx[0][b]]));
    }
    for (int32_t i = 1; i < n_pos; ++i) {
        CHECK((int32_t) info.p[i].size() == top_k * top_k);
        for (int32_t a = 0; a < top_k; ++a) {
            double row_sum = 0.0;
            for (int32_t b = 0; b < top_k; ++b) {
                const double v = info.p[i][(size_t) a * top_k + b];
                row_sum += v;
                CHECK(near(v, prob[i][(size_t) idx[i - 1][a] * top_k + idx[i][b]]));
            }
            CHECK(near(row_sum, 1.0, 1e-5));
        }
    }
    // every path's probability, summed: 1
    std::vector<double> mass(info.p[0].begin(), info.p[0].end());
    for (int32_t i = 1; i < n_pos; ++i) {
        std::vector<double> next(top_k, 0.0);
        for (int32_t a = 0; a < top_k; ++a) {
            for (int32_t b = 0; b < top_k; ++b) {
                next[b] += mass[a] * info.p[i][(size_t) a * top_k + b];
            }
        }
        mass = next;
    }
    double total = 0.0;
    for (const double m : mass) {
        total += m;
    }
    CHECK(near(total, 1.0, 1e-5));
}

// keep two: the walked candidate joins when it is not among them, rows are sub-rows of the full softmax (sums < 1),
// the kept mass is the sum of the kept marginals, and a stopped walk leaves pick -1
static void test_cut() {
    const int32_t n_pos = 3, top_k = 6;
    const size_t  stride = 48;
    const auto lat = make_lattice(n_pos, top_k, stride, 7);
    std::vector<std::vector<double>> prob, marg;
    reference(lat, n_pos, top_k, stride, prob, marg);

    // walk the lowest marginal at position 0, so it is outside the top two
    int32_t lowest = 0;
    for (int32_t b = 1; b < top_k; ++b) {
        if (marg[0][b] < marg[0][lowest]) {
            lowest = b;
        }
    }
    const std::vector<int32_t> walk = { lowest, 3 }; // stopped before position 2
    common_speculative_draft_info info;
    info.k_max = 2;
    common_speculative_lattice_info(info, lat.data(), stride, n_pos, top_k, walk);

    CHECK(info.ids[0].size() == 3);
    CHECK(info.pick[0] == 2 && cand_index(lat, stride, 0, top_k, info.ids[0][2]) == lowest);
    CHECK(info.pick[2] == -1);
    CHECK(info.ids[2].size() == 2);
    for (int32_t i = 0; i < n_pos; ++i) {
        double kept = 0.0;
        for (const auto id : info.ids[i]) {
            kept += marg[i][cand_index(lat, stride, i, top_k, id)];
        }
        CHECK(near(info.kept[i], kept, 1e-5));
        CHECK(info.kept[i] < 1.0f);
    }
    CHECK(info.p[1].size() == info.ids[0].size() * info.ids[1].size());
    CHECK(info.p[2].size() == info.ids[1].size() * info.ids[2].size());
    for (size_t a = 0; a < info.ids[0].size(); ++a) {
        double row_sum = 0.0;
        for (size_t b = 0; b < info.ids[1].size(); ++b) {
            const double v = info.p[1][a * info.ids[1].size() + b];
            row_sum += v;
            const int32_t ia = cand_index(lat, stride, 0, top_k, info.ids[0][a]);
            const int32_t ib = cand_index(lat, stride, 1, top_k, info.ids[1][b]);
            CHECK(near(v, prob[1][(size_t) ia * top_k + ib]));
        }
        CHECK(row_sum < 1.0 + 1e-6);
    }
}

// the anchor row is the only one read at position 0: other rows' scores do not matter there
static void test_anchor_row() {
    const int32_t n_pos = 2, top_k = 3;
    const size_t  stride = 12;
    auto lat = make_lattice(n_pos, top_k, stride, 3);
    common_speculative_draft_info a, b;
    a.k_max = b.k_max = 3;
    common_speculative_lattice_info(a, lat.data(), stride, n_pos, top_k, { 0, 0 });
    float * s0 = lat.data() + top_k;
    for (int32_t r = 1; r < top_k; ++r) {
        for (int32_t k = 0; k < top_k; ++k) {
            s0[r * top_k + k] = 100.0f * (float) (r + k);
        }
    }
    common_speculative_lattice_info(b, lat.data(), stride, n_pos, top_k, { 0, 0 });
    CHECK(a.ids == b.ids && a.p == b.p && a.pick == b.pick);
}

// the greedy walk: each position takes the highest-scoring candidate given the previous one (the first scores from the
// anchor alone); the reference walks the scores by hand
static void test_walk() {
    for (const uint32_t seed : { 11u, 12u, 13u }) {
        const int32_t n_pos = 7, top_k = 6;
        const size_t  stride = 48;
        const auto lat = make_lattice(n_pos, top_k, stride, seed);
        const auto walk = common_speculative_lattice_walk(lat.data(), stride, n_pos, top_k);
        CHECK((int32_t) walk.size() == n_pos);
        int32_t prev = 0;
        for (int32_t i = 0; i < n_pos; ++i) {
            const float * s = lat.data() + (size_t) i * stride + top_k + (size_t) prev * top_k;
            int32_t best = 0;
            for (int32_t b = 1; b < top_k; ++b) {
                if (s[b] > s[best]) {
                    best = b;
                }
            }
            CHECK(walk[i] == best);
            prev = best;
        }
    }
    CHECK(common_speculative_lattice_walk(nullptr, 8, 0, 4).empty());
}

// the full block's record: it is the lattice reduction along the greedy walk, in fb_*, and the draft's own record is
// left as it was
static void test_full_block() {
    const int32_t n_pos = 7, top_k = 8;
    const size_t  stride = 80;
    const auto lat = make_lattice(n_pos, top_k, stride, 21);
    std::vector<std::vector<double>> prob, marg;
    reference(lat, n_pos, top_k, stride, prob, marg);

    // a draft's record of 2 positions from an unrelated block, which the full block must not touch
    const auto lat2 = make_lattice(2, top_k, stride, 99);
    common_speculative_draft_info info;
    info.k_max = 3;
    common_speculative_lattice_info(info, lat2.data(), stride, 2, top_k, { 1, 2 });
    const auto ids0 = info.ids; const auto p0 = info.p; const auto pick0 = info.pick; const auto kept0 = info.kept;

    common_speculative_full_block_info(info, lat.data(), stride, n_pos, top_k);
    CHECK(info.ids == ids0 && info.p == p0 && info.pick == pick0 && info.kept == kept0);
    CHECK(info.fb_width == n_pos + 1);
    CHECK((int32_t) info.fb_ids.size() == n_pos && (int32_t) info.fb_p.size() == n_pos &&
          (int32_t) info.fb_pick.size() == n_pos && (int32_t) info.fb_kept.size() == n_pos);

    common_speculative_draft_info ref;
    ref.k_max = 3;
    const auto walk = common_speculative_lattice_walk(lat.data(), stride, n_pos, top_k);
    common_speculative_lattice_info(ref, lat.data(), stride, n_pos, top_k, walk);
    CHECK(info.fb_ids == ref.ids && info.fb_p == ref.p && info.fb_pick == ref.pick && info.fb_kept == ref.kept);

    // every position keeps k_max candidates and the walked one, and the walk is among them
    for (int32_t i = 0; i < n_pos; ++i) {
        CHECK(info.fb_ids[i].size() == 3 || info.fb_ids[i].size() == 4);
        CHECK(info.fb_pick[i] >= 0 && cand_index(lat, stride, i, top_k, info.fb_ids[i][info.fb_pick[i]]) == walk[i]);
        CHECK(info.fb_kept[i] > 0.0f && info.fb_kept[i] <= 1.0f + 1e-6f);
    }
    // the first position is scored from the anchor: its probabilities are the softmax of the anchor's row
    for (size_t b = 0; b < info.fb_ids[0].size(); ++b) {
        CHECK(near(info.fb_p[0][b], prob[0][cand_index(lat, stride, 0, top_k, info.fb_ids[0][b])]));
    }

    // a block of one position after the anchor, and clear()
    common_speculative_draft_info one;
    one.k_max = 8;
    common_speculative_full_block_info(one, lat.data(), stride, 1, top_k);
    CHECK(one.fb_width == 2 && one.fb_ids.size() == 1 && one.fb_pick[0] >= 0);
    one.clear();
    CHECK(one.fb_width == 0 && one.fb_ids.empty() && one.fb_p.empty() && one.fb_pick.empty() && one.fb_kept.empty());

    // a shorter block of the same rows is a prefix of the longer block's walk: the greedy walk looks backward only
    const auto w3 = common_speculative_lattice_walk(lat.data(), stride, 3, top_k);
    CHECK(std::equal(w3.begin(), w3.end(), walk.begin()));
}

int main() {
    test_keep_all();
    test_cut();
    test_anchor_row();
    test_walk();
    test_full_block();
    printf("test-spec-dump: %d checks, %d failed\n", n_checks, n_fail);
    return n_fail == 0 ? 0 : 1;
}
