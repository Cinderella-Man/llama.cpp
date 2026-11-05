// llama-pipe-drafter-check: runs the staircase drafter (src/llama-pipe-drafter.cpp) on a test case written by
// bench/staircase_drafter/check_export.py and compares the newest window row's logits of every guess with the
// reference from drafter.Drafter.forward (float32 on the CPU), or times one guess.
//
//   llama-pipe-drafter-check drafter.gguf case.bin [--split 9,8,8,8,8,8] [--kind stair|pre] [--stair L9,...]
//                            [--dev N] [--cpu] [--host-table] [--tol T]
//   llama-pipe-drafter-check drafter.gguf --bench N [--split ...] [--window W] [--commits C] [--ctx P] [--dev N]
//                            [--host-table]
//
// --split: the target's layers per device the drafter derives its window from (Flint's 49 layers); --stair overrides
// that window. --cpu runs the drafter on the CPU instead of a GPU (dev is then ignored). --host-table keeps the
// embedding table in host memory, so the drafter looks the rows up on the host instead of gathering them on its
// device. Exit status 0 when every guess is within tolerance: the largest logit difference at most T (default 0.05)
// times the reference's standard deviation, and the same most likely id.
//
// --bench: the cost of one guess on one GPU, as the executor sees it (from the launch to the guess on the host), with a
// random embedding table and random hidden states: P committed positions first (default 2048), then N guesses, each
// with C new committed rows (default 1) and W tokens in flight (default: the split's window), every row's hidden
// states uploaded fresh (as in the pipeline, where each token in flight has gone one device deeper since the last
// guess). Prints the median, 90th percentile and mean in microseconds after 20 warm-up guesses.

#include "llama.h"
#include "llama-pipe-drafter.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static const int32_t N_TARGET = 49;   // Flint's layers: depth 49 is the final state

static bool read_all(FILE * f, void * dst, size_t n) {
    return fread(dst, 1, n, f) == n;
}

static bool g_cpu = false;

static ggml_backend_dev_t gpu(int idx) {
    if (g_cpu) {
        return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    for (size_t i = 0, k = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t di = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(di) == GGML_BACKEND_DEVICE_TYPE_GPU && (int) k++ == idx) {
            return di;
        }
    }
    return nullptr;
}

static std::vector<int32_t> parse_list(const std::string & s) {
    std::vector<int32_t> v;
    size_t p = 0;
    while (p < s.size()) {
        size_t q = s.find(',', p);
        if (q == std::string::npos) {
            q = s.size();
        }
        v.push_back(atoi(s.substr(p, q - p).c_str()));
        p = q + 1;
    }
    return v;
}

struct opts {
    std::vector<int32_t> split = { 9, 8, 8, 8, 8, 8 };
    std::string kind;
    std::string stair;
    int  dev_idx    = 0;
    bool host_table = false;
};

static std::unique_ptr<llama_pipe_drafter> load(const char * path, ggml_tensor * table, ggml_backend_dev_t dev, const opts & o,
                                                int32_t n_rows, int32_t n_ctx) {
    llama_pipe_draft_params params = llama_pipe_draft_default_params();
    params.path   = path;
    params.n_rows = n_rows;
    params.n_ctx  = n_ctx;
    params.stair  = o.stair.empty() ? nullptr : o.stair.c_str();
    params.kind   = o.kind.empty() ? nullptr : o.kind.c_str();
    return llama_pipe_drafter::load(params, table, dev, o.split, N_TARGET, true);
}

static int bench(const char * path_drafter, int n_iter, int n_win, int n_commit, int n_ctx_fill, const opts & o) {
    ggml_backend_dev_t dev = gpu(o.dev_idx);
    if (!dev) {
        fprintf(stderr, "no GPU %d\n", o.dev_idx);
        return 2;
    }
    const int64_t n_tab = 32000;
    const int64_t d     = 2048;
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<ggml_fp16_t> table(n_tab*d);
    for (auto & x : table) {
        x = ggml_fp32_to_fp16(0.02f*nd(rng));
    }
    ggml_init_params ip = { ggml_tensor_overhead()*2, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * t_table = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, d, n_tab);
    ggml_backend_buffer_type_t buft = o.host_table ? ggml_backend_cpu_buffer_type() : ggml_backend_dev_buffer_type(dev);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    ggml_backend_tensor_set(t_table, table.data(), 0, table.size()*sizeof(ggml_fp16_t));

    // the split's window decides how many tokens are in flight unless --window says
    int32_t n_rows = 16;
    if (n_win > 0) {
        n_rows = std::max(8, ((n_win + n_commit + 7)/8)*8);
    }
    auto dr = load(path_drafter, t_table, dev, o, n_rows, std::max(4096, n_ctx_fill + 64));
    if (!dr) {
        fprintf(stderr, "cannot load the drafter\n");
        return 2;
    }
    if (n_win <= 0) {
        n_win = (int) dr->window_depths().size();
        const int32_t need = std::max(8, ((n_win + n_commit + 7)/8)*8);
        if (need != n_rows) {
            dr = load(path_drafter, t_table, dev, o, need, std::max(4096, n_ctx_fill + 64));
        }
    }
    dr->reset(0, 0);
    std::uniform_int_distribution<int> tok(0, (int) n_tab - 1);
    {
        std::vector<llama_token> t(n_ctx_fill);
        std::vector<llama_pos> p(n_ctx_fill);
        std::vector<float> h((size_t) n_ctx_fill*d), h9((size_t) n_ctx_fill*d);
        for (int i = 0; i < n_ctx_fill; i++) {
            t[i] = tok(rng);
            p[i] = i;
        }
        for (auto & x : h) {
            x = nd(rng);
        }
        for (auto & x : h9) {
            x = 4*nd(rng);
        }
        dr->test_commit_bulk(t.data(), p.data(), n_ctx_fill, h.data(), h9.data());
        dr->synchronize();
    }
    // hidden states at every depth
    std::vector<int32_t> depths(N_TARGET);
    std::vector<float> taps((size_t) N_TARGET*d);
    for (int32_t i = 0; i < N_TARGET; i++) {
        depths[i] = i + 1;
    }
    for (auto & x : taps) {
        x = 3*nd(rng);
    }
    llama_pos pos = n_ctx_fill;
    std::vector<double> us, us_launch;   // launch to guess; the launch call alone (host work and enqueueing)
    const int n_warm = 20;
    for (int it = 0; it < n_warm + n_iter; it++) {
        int slot = 0;
        for (int c = 0; c < n_commit; c++) {
            dr->test_set_taps(slot, tok(rng), pos, taps.data(), depths.data(), N_TARGET);
            dr->commit(tok(rng), pos, slot);
            slot++;
            pos++;
        }
        std::vector<llama_pipe_draft_row> rows;
        for (int w = 0; w < n_win; w++) {
            const llama_token t = tok(rng);
            dr->test_set_taps(slot, t, pos + w, taps.data(), depths.data(), N_TARGET);
            rows.push_back({ t, pos + w, slot });
            slot++;
        }
        const int64_t t0 = ggml_time_us();
        dr->launch(rows.data(), (int32_t) rows.size());
        const int64_t tl = ggml_time_us();
        dr->get(0.5f, nullptr);
        const int64_t t1 = ggml_time_us();
        if (it >= n_warm) {
            us.push_back((double) (t1 - t0));
            us_launch.push_back((double) (tl - t0));
        }
    }
    std::sort(us.begin(), us.end());
    std::sort(us_launch.begin(), us_launch.end());
    double mean = 0.0;
    for (double x : us) {
        mean += x;
    }
    mean /= us.size();
    std::string sp;
    for (auto s : o.split) {
        sp += (sp.empty() ? "" : "/") + std::to_string(s);
    }
    printf("DRAFTER_BENCH dev=%s split=%s window=%d commits=%d ctx=%d n=%d median_us=%.0f p90_us=%.0f mean_us=%.0f "
            "launch_us=%.0f table=%s\n", ggml_backend_dev_name(dev), sp.c_str(), n_win, n_commit, n_ctx_fill, n_iter,
            us[us.size()/2], us[(size_t) (us.size()*0.9)], mean, us_launch[us_launch.size()/2], o.host_table ? "host" : "device");
    dr.reset();
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s drafter.gguf case.bin [--split a,b,..] [--kind stair|pre] [--stair L9,..] [--dev N] [--cpu] "
                "[--host-table] [--tol T]\n       %s drafter.gguf --bench N [--split ..] [--window W] [--commits C] [--ctx P]\n",
                argv[0], argv[0]);
        return 2;
    }
    const char * path_drafter = argv[1];
    const char * path_case    = argv[2];
    opts  o;
    float tol = 0.05f;
    int   n_bench = 0, n_win = 0, n_commit = 1, n_ctx_fill = 2048;
    int   i0 = 3;
    if (std::string(argv[2]) == "--bench") {
        n_bench = atoi(argv[3]);
        i0 = 4;
    }
    for (int i = i0; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--window" && i + 1 < argc) {
            n_win = atoi(argv[++i]);
        } else if (a == "--commits" && i + 1 < argc) {
            n_commit = atoi(argv[++i]);
        } else if (a == "--ctx" && i + 1 < argc) {
            n_ctx_fill = atoi(argv[++i]);
        } else if (a == "--split" && i + 1 < argc) {
            o.split = parse_list(argv[++i]);
        } else if (a == "--kind" && i + 1 < argc) {
            o.kind = argv[++i];
        } else if (a == "--stair" && i + 1 < argc) {
            o.stair = argv[++i];
        } else if (a == "--dev" && i + 1 < argc) {
            o.dev_idx = atoi(argv[++i]);
        } else if (a == "--host-table") {
            o.host_table = true;
        } else if (a == "--cpu") {
            g_cpu = true;
        } else if (a == "--tol" && i + 1 < argc) {
            tol = (float) atof(argv[++i]);
        } else {
            fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }

    ggml_backend_load_all();
    llama_backend_init();

    if (n_bench > 0) {
        const int rc = bench(path_drafter, n_bench, n_win, n_commit, n_ctx_fill, o);
        llama_backend_free();
        return rc;
    }

    FILE * f = fopen(path_case, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path_case);
        return 2;
    }
    int32_t hdr[8];
    if (!read_all(f, hdr, sizeof(hdr)) || hdr[0] != 0x53434831) {
        fprintf(stderr, "%s is not a test case\n", path_case);
        return 2;
    }
    const int32_t n_tab = hdr[1], d = hdr[2], n_prompt = hdr[3], n_steps = hdr[4], n_v1 = hdr[5], n_tap = hdr[6];
    const bool fuse = hdr[7] != 0;
    // the depths of a row's hidden states: the six-card anchors and the final state, or every depth 1..49
    std::vector<int32_t> tap_depths;
    if (n_tap == 6) {
        tap_depths = { 9, 17, 25, 33, 41, N_TARGET };
    } else {
        for (int32_t i = 1; i <= n_tap; i++) {
            tap_depths.push_back(i);
        }
    }
    std::vector<float> table((size_t) n_tab*d);
    std::vector<int32_t> ptok(n_prompt);
    std::vector<float> pfull((size_t) n_prompt*d), pl9((size_t) n_prompt*d);
    if (!read_all(f, table.data(), table.size()*4) || !read_all(f, ptok.data(), ptok.size()*4) ||
            !read_all(f, pfull.data(), pfull.size()*4) || !read_all(f, pl9.data(), pl9.size()*4)) {
        fprintf(stderr, "short case file\n");
        return 2;
    }

    // the device and the embedding table on it (or in host memory)
    ggml_backend_dev_t dev = gpu(o.dev_idx);
    if (!dev) {
        fprintf(stderr, "no GPU %d\n", o.dev_idx);
        return 2;
    }
    ggml_init_params ip = { ggml_tensor_overhead()*2, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * t_table = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, n_tab);
    ggml_backend_buffer_type_t buft = o.host_table ? ggml_backend_cpu_buffer_type() : ggml_backend_dev_buffer_type(dev);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    ggml_backend_tensor_set(t_table, table.data(), 0, table.size()*4);

    auto dr = load(path_drafter, t_table, dev, o, 16, 1024);
    if (!dr) {
        fprintf(stderr, "cannot load the drafter\n");
        return 2;
    }
    dr->set_keep_logits(true);
    dr->reset(0, 0);

    std::vector<llama_pos> ppos(n_prompt);
    for (int32_t i = 0; i < n_prompt; i++) {
        ppos[i] = i;
    }
    dr->test_commit_bulk(ptok.data(), ppos.data(), n_prompt, pfull.data(), fuse ? pl9.data() : nullptr);

    int n_bad = 0;
    double worst = 0.0;
    std::vector<float> taps((size_t) n_tap*d), expect(n_v1);
    for (int32_t s = 0; s < n_steps; s++) {
        int32_t nc[2];
        if (!read_all(f, nc, sizeof(nc))) {
            fprintf(stderr, "short case file at step %d\n", s);
            return 2;
        }
        std::vector<llama_pipe_draft_row> rows;
        for (int32_t r = 0; r < nc[0] + nc[1]; r++) {
            int32_t tp[2];
            if (!read_all(f, tp, sizeof(tp)) || !read_all(f, taps.data(), taps.size()*4)) {
                fprintf(stderr, "short case file at step %d\n", s);
                return 2;
            }
            // every row is the latest evaluation of its own slot
            const int32_t slot = r;
            dr->test_set_taps(slot, tp[0], tp[1], taps.data(), tap_depths.data(), n_tap);
            if (r < nc[0]) {
                dr->commit(tp[0], tp[1], slot);
            } else {
                rows.push_back({ tp[0], tp[1], slot });
            }
        }
        if (!read_all(f, expect.data(), expect.size()*4)) {
            fprintf(stderr, "short case file at step %d\n", s);
            return 2;
        }
        dr->launch(rows.data(), (int32_t) rows.size());
        float p_top = 0.0f;
        dr->get(-1.0f, &p_top);
        const std::vector<float> & got = dr->last_logits();
        if ((int32_t) got.size() != n_v1) {
            fprintf(stderr, "step %d: %zu logits, expected %d\n", s, got.size(), n_v1);
            return 1;
        }
        double mean = 0.0, var = 0.0, max_diff = 0.0, sq = 0.0;
        int arg_e = 0, arg_g = 0;
        for (int32_t i = 0; i < n_v1; i++) {
            mean += expect[i];
            if (expect[i] > expect[arg_e]) {
                arg_e = i;
            }
            if (got[i] > got[arg_g]) {
                arg_g = i;
            }
        }
        mean /= n_v1;
        for (int32_t i = 0; i < n_v1; i++) {
            var += (expect[i] - mean)*(expect[i] - mean);
            const double df = fabs((double) got[i] - expect[i]);
            max_diff = std::max(max_diff, df);
            sq += df*df;
        }
        const double sd = sqrt(var/n_v1);
        const double rel = max_diff/sd;
        worst = std::max(worst, rel);
        const bool ok = rel <= tol && arg_e == arg_g && std::isfinite(max_diff);
        n_bad += !ok;
        printf("step %d: %d commits, %d in flight: max |diff| %.5f (%.4f sd), rms %.5f, top id %d / %d, p_top %.4f %s\n",
                s, nc[0], nc[1], max_diff, rel, sqrt(sq/n_v1), arg_e, arg_g, p_top, ok ? "ok" : "FAIL");
    }
    fclose(f);
    std::string st;
    for (auto x : dr->window_depths()) {
        st += (st.empty() ? "" : ",") + std::to_string(x);
    }
    printf("DRAFTER_CHECK steps=%d bad=%d worst_rel=%.5f table=%s fuse=%d window=%s older=%d\n", n_steps, n_bad, worst,
            o.host_table ? "host" : "device", (int) fuse, st.c_str(), dr->overflow_depth());

    dr.reset();
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    llama_backend_free();
    return n_bad == 0 ? 0 : 1;
}
