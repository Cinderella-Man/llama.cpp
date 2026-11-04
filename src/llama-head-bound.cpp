#include "llama-head-bound.h"

#include "llama-head-shortlist.h"
#include "llama-impl.h"

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <numeric>

llama_head_bound::~llama_head_bound() {
    if (galloc) {
        ggml_gallocr_free(galloc);
    }
}

bool llama_head_bound::init(ggml_tensor * output, ggml_backend_t backend) {
    const ggml_type type = output->type;
    const auto * traits = ggml_get_type_traits(type);
    if (type != GGML_TYPE_F32 && traits->to_float == nullptr) {
        LLAMA_LOG_ERROR("%s: no dequantization for the head's type %s\n", __func__, ggml_type_name(type));
        return false;
    }

    this->output  = output;
    this->backend = backend;

    n_embd  = (uint32_t) output->ne[0];
    n_vocab = (uint32_t) output->ne[1];

    const size_t row_size = output->nb[1];
    const size_t n_chunk  = std::max<size_t>(1, (8u << 20) / row_size); // rows per read

    std::vector<uint8_t> raw(n_chunk*row_size);
    std::vector<float>   row(n_embd);

    const auto read_rows = [&](uint32_t r0, uint32_t nr) {
        ggml_backend_tensor_get(output, raw.data(), (size_t) r0*row_size, (size_t) nr*row_size);
    };
    const auto dequant = [&](uint32_t r) {
        const uint8_t * src = raw.data() + (size_t) r*row_size;
        if (type == GGML_TYPE_F32) {
            memcpy(row.data(), src, n_embd*sizeof(float));
        } else {
            traits->to_float(src, row.data(), n_embd);
        }
    };

    // the mean row, of every 8th block of rows: any vector gives a valid bound, the mean a tight one
    mu.assign(n_embd, 0.0);
    {
        uint64_t n_mu = 0;
        for (uint32_t r0 = 0; r0 < n_vocab; r0 += 8*n_chunk) {
            const uint32_t nr = (uint32_t) std::min<size_t>(n_chunk, n_vocab - r0);
            read_rows(r0, nr);
            for (uint32_t r = 0; r < nr; ++r) {
                dequant(r);
                for (uint32_t j = 0; j < n_embd; ++j) {
                    mu[j] += row[j];
                }
            }
            n_mu += nr;
        }
        mu_norm = 0.0;
        for (uint32_t j = 0; j < n_embd; ++j) {
            mu[j] /= (double) n_mu;
            mu_norm += mu[j]*mu[j];
        }
        mu_norm = std::sqrt(mu_norm);
    }

    // the norms of every row, plain and about the mean, rounded up to F32
    norm.resize(n_vocab);
    cnorm.resize(n_vocab);
    for (uint32_t r0 = 0; r0 < n_vocab; r0 += n_chunk) {
        const uint32_t nr = (uint32_t) std::min<size_t>(n_chunk, n_vocab - r0);
        read_rows(r0, nr);
        for (uint32_t r = 0; r < nr; ++r) {
            dequant(r);
            double s = 0.0;
            double c = 0.0;
            for (uint32_t j = 0; j < n_embd; ++j) {
                const double v = row[j];
                const double d = v - mu[j];
                s += v*v;
                c += d*d;
            }
            norm [r0 + r] = std::nextafter((float) std::sqrt(s), INFINITY);
            cnorm[r0 + r] = std::nextafter((float) std::sqrt(c), INFINITY);
        }
    }

    by_norm.resize(n_vocab);
    std::iota(by_norm.begin(), by_norm.end(), 0);
    by_cnorm = by_norm;
    std::sort(by_norm.begin(),  by_norm.end(),  [&](int32_t a, int32_t b) { return norm[a]  > norm[b];  });
    std::sort(by_cnorm.begin(), by_cnorm.end(), [&](int32_t a, int32_t b) { return cnorm[a] > cnorm[b]; });
    i_norm  = 0;
    i_cnorm = 0;

    // the fallback: the same product as the graph's full head, on the head's backend
    {
        ggml_init_params params = {
            /*.mem_size   =*/ 4*ggml_tensor_overhead() + ggml_graph_overhead_custom(8, false),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx.reset(ggml_init(params));
        if (!ctx) {
            return false;
        }
        h_in = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_embd, 1);
        ggml_set_name(h_in, "head_bound_h");
        ggml_set_input(h_in);
        out = ggml_mul_mat(ctx.get(), output, h_in);
        ggml_set_name(out, "head_bound_full");
        ggml_set_output(out);
        gf = ggml_new_graph_custom(ctx.get(), 8, false);
        ggml_build_forward_expand(gf, out);

        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate the full head's graph on %s\n", __func__, ggml_backend_name(backend));
            return false;
        }
    }

    h.assign(n_embd, 0.0f);
    this->row = -1;

    return true;
}

float llama_head_bound::compute(const llama_head_shortlist & sl) {
    // the shortlist only grows, so the first ids outside it only move forward
    while (i_norm < by_norm.size() && sl.contains(by_norm[i_norm])) {
        i_norm++;
    }
    while (i_cnorm < by_cnorm.size() && sl.contains(by_cnorm[i_cnorm])) {
        i_cnorm++;
    }
    if (i_norm >= by_norm.size()) {
        return -INFINITY; // nothing left out
    }
    const double N = norm [by_norm [i_norm]];
    const double C = cnorm[by_cnorm[i_cnorm]];

    double hn2 = 0.0;
    double hm  = 0.0;
    for (uint32_t j = 0; j < n_embd; ++j) {
        hn2 += (double) h[j]*h[j];
        hm  += (double) h[j]*mu[j];
    }
    const double hn = std::sqrt(hn2);

    // |h' - h| for the ways the kernels quantize h: Q8_1 blocks of 32 (the CUDA matrix-vector kernels: d = amax/127,
    // stored as F16, so at most d/2 + 127 d 2^-11 per value), Q8_K blocks of 256 (the CPU k-quant dot products: d/2
    // per value) and F16 (2^-11 |h_j|, for F16 heads)
    double e32 = 0.0;
    double e256 = 0.0;
    for (uint32_t b = 0; b < n_embd; b += 32) {
        float amax = 0.0f;
        for (uint32_t j = b; j < std::min(n_embd, b + 32); ++j) {
            amax = std::max(amax, std::fabs(h[j]));
        }
        const double d = amax/127.0;
        e32 += 32.0*(d*(0.5 + 127.0/2048.0))*(d*(0.5 + 127.0/2048.0));
    }
    for (uint32_t b = 0; b < n_embd; b += 256) {
        float amax = 0.0f;
        for (uint32_t j = b; j < std::min(n_embd, b + 256); ++j) {
            amax = std::max(amax, std::fabs(h[j]));
        }
        const double d = amax/127.0;
        e256 += 256.0*(d*0.5)*(d*0.5);
    }
    const double eh  = std::max(std::sqrt(e32), std::sqrt(e256)) + hn/2048.0;
    const double hn1 = hn + eh;

    const double b_plain = hn1*N;
    const double b_mean  = hm + eh*mu_norm + hn1*C;

    // the kernels' own arithmetic: F32 accumulation and, in the Q8_1 path, the F16 block sums that the mins of
    // k-quants multiply, at most ~3e-3 |h| |W_t| together; 2e-2 of the largest |h'| |W_t| left out
    const double b = std::min(b_plain, b_mean) + 2e-2*hn1*N;

    sum_hn += hn;

    return std::nextafter((float) b, INFINITY);
}

bool llama_head_bound::full(float * dst, const llama_head_shortlist & sl, float bnd) {
    ggml_backend_tensor_set(h_in, h.data(), 0, (size_t) n_embd*sizeof(float));
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: the full head failed on %s\n", __func__, ggml_backend_name(backend));
        return false;
    }
    ggml_backend_tensor_get(out, dst, 0, (size_t) n_vocab*sizeof(float));

    // how far the bound was from the truth, and whether the shortlist held the top token
    float top_in  = -INFINITY;
    float top_out = -INFINITY;
    for (uint32_t t = 0; t < n_vocab; ++t) {
        if (sl.contains((int32_t) t)) {
            top_in = std::max(top_in, dst[t]);
        } else {
            top_out = std::max(top_out, dst[t]);
        }
    }
    if (std::isfinite(top_out)) {
        n_miss       += top_out >= top_in;
        sum_excess   += (double) bnd - top_out;
        sum_left_gap += (double) top_in - top_out;
    }

    return true;
}

void llama_head_bound::log(const char * func) const {
    if (n_rows == 0) {
        return;
    }
    const double n = (double) n_rows;
    const double f = (double) std::max<uint64_t>(n_full, 1);
    LLAMA_LOG_INFO("%s: head bound: %" PRIu64 " rows with the shortlist head, %" PRIu64 " full-head fallbacks (%.1f %%); "
            "top logit above the bound in %" PRIu64 " rows (%.1f %%); bound - top logit: mean %.2f, min %.2f; "
            "means: top logit %.2f, bound %.2f, |h| %.2f; on the fallbacks: the top token left out in %" PRIu64 ", "
            "bound - highest left-out logit %.2f, top - highest left-out logit %.2f (means)\n", func,
            n_rows, n_full, 100.0*n_full/n, n_greedy, 100.0*n_greedy/n, sum_gap/n, min_gap, sum_top/n, sum_bnd/n,
            sum_hn/n, n_miss, sum_excess/f, sum_left_gap/f);
}
