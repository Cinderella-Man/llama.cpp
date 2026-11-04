#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

struct llama_head_shortlist;

// The main model's output head over a shortlist, with an exact bound on every logit it leaves out (Q619). The
// shortlist (llama_head_shortlist) gives the exact logits of its ids, the other ids get -INFINITY; for such a row, the
// bound is at least every left-out logit the full head would have computed, from the head's input row h and
// Cauchy-Schwarz over the head's rows as the kernels read them:
//
//   z_t = <h', W_t> <= |h'| |W_t|                            (plain)
//   z_t = <h', mu> + <h', W_t - mu> <= <h', mu> + |h'| |W_t - mu|  (about mu, the mean row)
//
// where h' is h as the kernel quantizes it (|h' - h| <= eh, computed from h's blocks), and the maxima of |W_t| and
// |W_t - mu| over the ids left out are kept as the shortlist grows. A sampler that can't exclude every left-out
// token with the bound (the k-th highest logit of its candidates not above it) asks for the full head, computed again
// from h with the same kernel as the full product (full()).
struct llama_head_bound {
    uint32_t n_vocab = 0;
    uint32_t n_embd  = 0;

    // from the head's rows, dequantized as the kernels read them (init())
    std::vector<float>  norm;  // [n_vocab] |W_t|
    std::vector<float>  cnorm; // [n_vocab] |W_t - mu|
    std::vector<double> mu;    // [n_embd]  the mean row (of a sample of rows: any vector gives a valid bound)
    double              mu_norm = 0.0;

    std::vector<int32_t> by_norm;  // ids by descending norm
    std::vector<int32_t> by_cnorm; // ids by descending cnorm
    size_t               i_norm  = 0; // the first entries of by_norm / by_cnorm not in the shortlist
    size_t               i_cnorm = 0;

    // the last decode: its shortlist row (an output row index, -1 = none) and the head's input row for it
    int32_t            row = -1;
    std::vector<float> h; // [n_embd]
    bool               have_bound = false;
    float              bound      = INFINITY;

    // the full head on h, on the backend that holds the head (full())
    ggml_backend_t    backend = nullptr;
    ggml_tensor     * output  = nullptr;
    ggml_context_ptr  ctx;
    ggml_gallocr_t    galloc  = nullptr;
    ggml_cgraph     * gf      = nullptr;
    ggml_tensor     * h_in    = nullptr;
    ggml_tensor     * out     = nullptr;

    // counts: rows with the shortlist, fallbacks to the full head, rows whose top logit was above the bound (a greedy
    // sampler would keep the shortlist's logits), and the margins (bound - top logit) of the rows
    uint64_t n_rows   = 0;
    uint64_t n_full   = 0;
    uint64_t n_greedy = 0;
    double   sum_gap  = 0.0;
    double   min_gap  = INFINITY;
    double   sum_top  = 0.0;
    double   sum_bnd  = 0.0;
    double   sum_hn   = 0.0;

    // on the fallbacks, from the full head's row: the highest logit left out against the bound and the shortlist's top
    // (n_miss: rows whose top token was left out, which no bound could have kept)
    uint64_t n_miss       = 0;
    double   sum_excess   = 0.0; // bound - highest left-out logit
    double   sum_left_gap = 0.0; // shortlist's top - highest left-out logit

    ~llama_head_bound();

    // reads the head's rows (twice for a sample of them) and sets up the fallback graph on the backend
    bool init(ggml_tensor * output, ggml_backend_t backend);

    // the bound for the input row h, with the ids of sl left out
    float compute(const llama_head_shortlist & sl);

    // the full head's logits [n_vocab] for h into dst; counts how far the row's bound bnd was from the highest logit
    // left out by sl
    bool full(float * dst, const llama_head_shortlist & sl, float bnd);

    void log(const char * func) const;
};
