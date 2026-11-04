// LLAMA_SPEC_DUMP's reduction of a DFlash2 selector lattice (common_speculative_lattice_info): softmax rows, forward
// marginals, the candidates kept per position (the k_max highest marginals plus the walked one), the conditional rows
// between kept candidates, and the walk's indices. Synthetic lattices with known answers; no model.

#include "speculative.h"

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

int main() {
    test_keep_all();
    test_cut();
    test_anchor_row();
    printf("test-spec-dump: %d checks, %d failed\n", n_checks, n_fail);
    return n_fail == 0 ? 0 : 1;
}
