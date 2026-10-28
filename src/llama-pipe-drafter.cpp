#include "llama-pipe-drafter.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-graph.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

// positions of cells nothing may see, and of the scratch rows that pad a graph (they see every real cell, so their
// attention stays finite, and nothing sees them)
static constexpr float POS_NONE = 2e9f;
static constexpr float POS_PAD  = 1e9f;

static constexpr int GRAPH_NODES = 1024;

std::vector<int32_t> llama_pipe_split_depths(const std::vector<int32_t> & split, const std::string & kind, const double timing[5],
                                             int32_t max_w);

llama_pipe_drafter::graph_inst::~graph_inst() {
    if (galloc) {
        ggml_gallocr_free(galloc);
    }
    if (buf_in) {
        ggml_backend_buffer_free(buf_in);
    }
    if (ctx_in) {
        ggml_free(ctx_in);
    }
    if (ctx) {
        ggml_free(ctx);
    }
}

llama_pipe_drafter::~llama_pipe_drafter() {
    if (backend) {
        ggml_backend_synchronize(backend);
    }
    g_step.reset();
    g_bulk_small.reset();
    g_bulk_large.reset();
    for (auto * ev : ev_tap) {
        if (ev) {
            ggml_backend_event_free(ev);
        }
    }
    for (auto * ev : ev_launch) {
        if (ev) {
            ggml_backend_event_free(ev);
        }
    }
    for (auto & c : caps) {
        if (c.ev_full) {
            ggml_backend_event_free(c.ev_full);
        }
        if (c.ev_fz) {
            ggml_backend_event_free(c.ev_fz);
        }
        if (c.ev_done) {
            ggml_backend_event_free(c.ev_done);
        }
        if (c.buf) {
            ggml_backend_buffer_free(c.buf);
        }
    }
    if (buf_stage) {
        ggml_backend_buffer_free(buf_stage);
    }
    if (buf_tap) {
        ggml_backend_buffer_free(buf_tap);
    }
    if (buf_s) {
        ggml_backend_buffer_free(buf_s);
    }
    if (ctx_s) {
        ggml_free(ctx_s);
    }
    if (buf_w) {
        ggml_backend_buffer_free(buf_w);
    }
    if (ctx_w) {
        ggml_free(ctx_w);
    }
    if (backend) {
        ggml_backend_free(backend);
    }
}

std::unique_ptr<llama_pipe_drafter> llama_pipe_drafter::load(const llama_pipe_draft_params & params, ggml_tensor * tok_embd,
                                                             ggml_backend_dev_t dev, const std::vector<int32_t> & split,
                                                             int32_t n_layer_target, bool dev_has_layers) {
    std::unique_ptr<llama_pipe_drafter> d(new llama_pipe_drafter());
    try {
        if (!d->init(params, tok_embd, dev, split, n_layer_target, dev_has_layers)) {
            return nullptr;
        }
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        return nullptr;
    }
    return d;
}

static int64_t gguf_key(const gguf_context * g, const char * name, bool required = true) {
    const int64_t k = gguf_find_key(g, name);
    if (k < 0 && required) {
        throw std::runtime_error(std::string("missing key ") + name);
    }
    return k;
}

static std::vector<int32_t> gguf_i32_array(const gguf_context * g, const char * name) {
    const int64_t k = gguf_find_key(g, name);
    if (k < 0) {
        return {};
    }
    if (gguf_get_arr_type(g, k) != GGUF_TYPE_INT32) {
        throw std::runtime_error(std::string(name) + " must be an int32 array");
    }
    const int32_t * a = (const int32_t *) gguf_get_arr_data(g, k);
    return std::vector<int32_t>(a, a + gguf_get_arr_n(g, k));
}

// "emb", "full", "L<n>" or "<n>" -> depth (0, n_target, n); -1 if unknown
static int32_t parse_depth(const std::string & s, int32_t n_target) {
    if (s == "emb") {
        return 0;
    }
    if (s == "full") {
        return n_target;
    }
    const char * p = s.c_str() + (!s.empty() && s[0] == 'L' ? 1 : 0);
    char * end = nullptr;
    const long v = strtol(p, &end, 10);
    if (end == p || *end != '\0' || v < 0 || v > n_target) {
        return -1;
    }
    return (int32_t) v;
}

static std::string depth_str(int32_t d, int32_t n_target) {
    return d == 0 ? "emb" : d >= n_target ? "full" : "L" + std::to_string(d);
}

std::vector<int32_t> llama_pipe_split_depths(const std::vector<int32_t> & split, const std::string & kind, const double timing[5],
                                             int32_t max_w) {
    const int n = (int) split.size();
    std::vector<int32_t> out;
    if (n == 0) {
        return out;
    }
    const double layer_ms = std::max(timing[2], timing[0] + (n - 6)*timing[1]);
    std::vector<double> stages(n), ends;
    for (int i = 0; i < n; i++) {
        stages[i] = layer_ms*split[i];
    }
    stages[n - 1] += timing[3];
    const double cycle = *std::max_element(stages.begin(), stages.end()) + timing[4];
    double t = 0.0;
    for (int i = 0; i < n; i++) {
        t += stages[i];
        ends.push_back(t);
        t += timing[4];
    }
    std::vector<int32_t> done(n);
    for (int k = 0, acc = 0; k < n; k++) {
        acc += split[k];
        done[k] = acc;
    }
    if (kind == "miss") {
        return { done[0] };
    }
    const double t0 = kind == "stair" ? stages[0] : 0.0;
    for (int m = 0; m < max_w; m++) {
        int last = -1;
        for (int j = 0; j < n; j++) {
            if (ends[j] <= t0 + m*cycle + 1e-9) {
                last = j;
            }
        }
        if (last == n - 1) {
            break;
        }
        out.push_back(last >= 0 ? done[last] : 0);
    }
    return out;
}

std::vector<int32_t> llama_pipe_split_depths(const std::vector<int32_t> & split, const std::string & kind, const float timing[5],
                                             int32_t max_w) {
    // the head file stores the timing in float32: back to the decimals drafter.py uses
    double t[5];
    for (int i = 0; i < 5; i++) {
        t[i] = std::round((double) timing[i]*1e6)/1e6;
    }
    return llama_pipe_split_depths(split, kind, t, max_w);
}

int32_t llama_pipe_drafter::depth_name(int32_t depth) const {
    if (depth >= n_target) {
        return n_target;
    }
    int32_t best = 0;
    for (int32_t r : recorded) {
        if (r <= depth && r > best) {
            best = r;
        }
    }
    return best;
}

bool llama_pipe_drafter::init(const llama_pipe_draft_params & params, ggml_tensor * tok_embd_, ggml_backend_dev_t dev_,
                              const std::vector<int32_t> & split_, int32_t n_layer_target, bool dev_has_layers) {
    dev = dev_;
    tok_embd = tok_embd_;
    split = split_;

    // file: metadata and tensor shapes, then the data onto dev
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &ctx_w };
    gguf_context * g = gguf_init_from_file(params.path, gp);
    if (!g) {
        LLAMA_LOG_ERROR("%s: cannot read %s\n", __func__, params.path);
        return false;
    }
    struct gguf_guard { gguf_context * g; ~gguf_guard() { gguf_free(g); } } guard { g };

    {
        const int64_t k = gguf_key(g, "general.architecture");
        if (std::string(gguf_get_val_str(g, k)) != "staircase-drafter") {
            LLAMA_LOG_ERROR("%s: %s is not a staircase drafter\n", __func__, params.path);
            return false;
        }
    }
    n_embd        = (int32_t) gguf_get_val_u32(g, gguf_key(g, "staircase.n_embd"));
    n_layer       = (int32_t) gguf_get_val_u32(g, gguf_key(g, "staircase.n_layer"));
    n_head        = (int32_t) gguf_get_val_u32(g, gguf_key(g, "staircase.n_head"));
    n_head_kv     = (int32_t) gguf_get_val_u32(g, gguf_key(g, "staircase.n_head_kv"));
    head_dim      = (int32_t) gguf_get_val_u32(g, gguf_key(g, "staircase.head_dim"));
    n_ff          = (int32_t) gguf_get_val_u32(g, gguf_key(g, "staircase.n_ff"));
    n_vocab_draft = (int32_t) gguf_get_val_u32(g, gguf_key(g, "staircase.n_vocab"));
    rms_eps       = gguf_get_val_f32(g, gguf_key(g, "staircase.rms_eps"));
    rope_theta    = gguf_get_val_f32(g, gguf_key(g, "staircase.rope_theta"));
    fuse          = gguf_get_val_bool(g, gguf_key(g, "staircase.fuse"));
    alpha         = gguf_get_val_f32(g, gguf_key(g, "staircase.alpha"));
    {
        const int64_t k = gguf_key(g, "staircase.general", false);
        general = k >= 0 && gguf_get_val_bool(g, k);
    }
    {
        const int64_t k = gguf_key(g, "staircase.profile", false);
        profile = k >= 0 ? gguf_get_val_str(g, k) : "";
    }
    {
        const int64_t k = gguf_key(g, "staircase.target_layers", false);
        n_target = k >= 0 ? (int32_t) gguf_get_val_u32(g, k) : n_layer_target;
        if (n_target != n_layer_target) {
            LLAMA_LOG_ERROR("%s: the drafter was trained on a target with %d layers, this model has %d\n", __func__, n_target,
                    n_layer_target);
            return false;
        }
    }
    // depths: projection and depth-embedding row per depth, and the depths the head was trained on
    depth_proj = gguf_i32_array(g, "staircase.depth_proj");
    if (!depth_proj.empty()) {
        depth_emb_row = gguf_i32_array(g, "staircase.depth_emb_row");
        recorded      = gguf_i32_array(g, "staircase.recorded");
        fuse_depths   = gguf_i32_array(g, "staircase.fuse_depths");
        if ((int32_t) depth_proj.size() != n_target + 1 || (int32_t) depth_emb_row.size() != n_target + 1 || recorded.empty()) {
            LLAMA_LOG_ERROR("%s: staircase.depth_proj / depth_emb_row / recorded do not match %d target layers\n", __func__,
                    n_target);
            return false;
        }
    } else {
        // a six-anchor file: one projection and one anchor embedding per anchor layer, then the final state
        recorded = gguf_i32_array(g, "staircase.anchor_layers");
        if (recorded.empty()) {
            LLAMA_LOG_ERROR("%s: no staircase.anchor_layers\n", __func__);
            return false;
        }
        const int32_t na = (int32_t) recorded.size();
        depth_proj.assign(n_target + 1, -1);
        depth_emb_row.assign(n_target + 1, -1);
        depth_emb_row[0] = 0;
        for (int32_t k = 0; k < na; k++) {
            depth_proj[recorded[k]]    = k;
            depth_emb_row[recorded[k]] = k + 1;
        }
        depth_proj[n_target]    = na;
        depth_emb_row[n_target] = na + 1;
    }
    if (fuse && fuse_depths.empty()) {
        fuse_depths = recorded;
    }
    std::sort(recorded.begin(), recorded.end());
    {
        const int64_t k = gguf_find_key(g, "staircase.timing");
        if (k >= 0 && gguf_get_arr_type(g, k) == GGUF_TYPE_FLOAT32 && gguf_get_arr_n(g, k) == 5) {
            memcpy(timing, gguf_get_arr_data(g, k), sizeof(timing));
        }
    }
    {
        const int64_t k = gguf_key(g, "staircase.committed", false);
        committed = parse_depth(k >= 0 ? gguf_get_val_str(g, k) : "full", n_target);
        if (committed < 0) {
            LLAMA_LOG_ERROR("%s: bad committed anchor\n", __func__);
            return false;
        }
    }
    if (params.alpha > 0.0f) {
        alpha = params.alpha;
    }
    top_k       = std::max(1, params.top_k);
    n_rows_step = std::max(2, params.n_rows);

    // the run's window: from the split (or given), each depth the deepest trained one at or below it
    kind = params.kind && params.kind[0] ? params.kind : (dev_has_layers ? "stair" : "pre");
    if (kind != "stair" && kind != "pre") {
        LLAMA_LOG_ERROR("%s: kind must be stair or pre, not %s\n", __func__, kind.c_str());
        return false;
    }
    std::vector<int32_t> done;
    for (int32_t s : split) {
        done.push_back((done.empty() ? 0 : done.back()) + s);
    }
    if (params.stair && params.stair[0]) {
        std::string s = params.stair;
        size_t p = 0;
        while (p <= s.size()) {
            size_t q = s.find(',', p);
            if (q == std::string::npos) {
                q = s.size();
            }
            const std::string tok = s.substr(p, q - p);
            if (!tok.empty()) {
                const int32_t d = parse_depth(tok, n_target);
                if (d < 0 || d >= n_target) {
                    LLAMA_LOG_ERROR("%s: bad staircase entry '%s' (emb, L<n> or <n> below the final state)\n", __func__, tok.c_str());
                    return false;
                }
                stair.push_back(d);
            }
            p = q + 1;
        }
    } else {
        for (int32_t d : llama_pipe_split_depths(split, kind, timing, 12)) {
            stair.push_back(depth_name(d));
        }
    }
    if (stair.empty()) {
        stair.push_back(kind == "stair" && !done.empty() ? depth_name(done[0]) : 0);
    }
    // the window ends at the first token that has passed the last device: older tokens in flight are complete and read
    // like committed rows
    depth_deep = committed;

    // taps: every depth a row may read (the window, the fused depths, the committed depth), then the final state
    {
        std::vector<int32_t> need(stair);
        if (fuse) {
            need.insert(need.end(), fuse_depths.begin(), fuse_depths.end());
        }
        if (committed > 0 && committed < n_target) {
            need.push_back(committed);
        }
        std::sort(need.begin(), need.end());
        need.erase(std::unique(need.begin(), need.end()), need.end());
        for (int32_t d : need) {
            if (d > 0 && d < n_target) {
                tap_depth.push_back(d);
            }
        }
        tap_depth.push_back(n_target);
        tap_of_depth.assign(n_target + 1, -1);
        for (size_t t = 0; t < tap_depth.size(); t++) {
            tap_of_depth[tap_depth[t]] = (int32_t) t;
        }
    }
    // every depth a row may have needs its projection and embedding
    {
        std::vector<int32_t> used(stair);
        used.push_back(depth_deep);
        used.push_back(committed);
        for (int32_t d : used) {
            if (d < 0 || d > n_target || depth_emb_row[d] < 0 || (d > 0 && depth_proj[d] < 0)) {
                LLAMA_LOG_ERROR("%s: the head has no projection for depth %d (trained on", __func__, d);
                for (int32_t r : recorded) {
                    LLAMA_LOG_CONT(" %d", r);
                }
                LLAMA_LOG_CONT(")\n");
                return false;
            }
        }
    }

    {
        const int64_t it = gguf_find_tensor(g, "sd.vocab");
        if (it < 0) {
            LLAMA_LOG_ERROR("%s: missing tensor sd.vocab\n", __func__);
            return false;
        }
    }

    if (dev == nullptr) {
        LLAMA_LOG_ERROR("%s: no device\n", __func__);
        return false;
    }
    backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) {
        LLAMA_LOG_ERROR("%s: cannot start a backend on %s\n", __func__, ggml_backend_dev_name(dev));
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    buft_host = ggml_backend_dev_host_buffer_type(dev);
    if (!buft_host) {
        buft_host = ggml_backend_cpu_buffer_type();
    }

    // cost options from the environment: the output head's draft ids (LLAMA_DRAFT_VOCAB=K keeps the K most frequent,
    // the vocabulary's first K, and the other class) and type (LLAMA_DRAFT_HEAD=q8_0 or f16, converted at load), the
    // last block for the output row only (LLAMA_DRAFT_PRUNE, on unless 0), one product for all depth projections
    // (LLAMA_DRAFT_ANC=dense), the step graph's products in groups of rows (LLAMA_DRAFT_SPLIT=4: the Pascal
    // multi-token kernels take up to 4)
    auto env_str = [](const char * name) {
        const char * v = getenv(name);
        return std::string(v ? v : "");
    };
    {
        const std::string v = env_str("LLAMA_DRAFT_VOCAB");
        const int32_t k = v.empty() ? 0 : atoi(v.c_str());
        n_short = k > 0 && k < n_vocab_draft ? k : n_vocab_draft;
    }
    prune_last = env_str("LLAMA_DRAFT_PRUNE") != "0";
    anc_dense  = env_str("LLAMA_DRAFT_ANC") == "dense";
    split_cols = std::max(0, atoi(env_str("LLAMA_DRAFT_SPLIT").c_str()));
    ggml_tensor * th = ggml_get_tensor(ctx_w, "sd.head.weight");
    ggml_tensor * tb = ggml_get_tensor(ctx_w, "sd.head.bias");
    if (th == nullptr || tb == nullptr || th->ne[1] != n_vocab_draft + 1 || tb->ne[0] != n_vocab_draft + 1) {
        LLAMA_LOG_ERROR("%s: the output head does not match %d draft ids\n", __func__, n_vocab_draft);
        return false;
    }
    const ggml_type head_file_type = th->type;
    ggml_type head_type = head_file_type;
    {
        const std::string v = env_str("LLAMA_DRAFT_HEAD");
        if (v == "q8_0") {
            head_type = GGML_TYPE_Q8_0;
        } else if (v == "f16") {
            head_type = GGML_TYPE_F16;
        } else if (!v.empty()) {
            LLAMA_LOG_ERROR("%s: LLAMA_DRAFT_HEAD must be q8_0 or f16, not %s\n", __func__, v.c_str());
            return false;
        }
    }
    const bool head_converted = n_short < n_vocab_draft || head_type != head_file_type;
    if (head_converted) {
        // the tensors take the run's shape and type before the allocation; the file's rows are converted on reading
        th->type  = head_type;
        th->ne[1] = n_short + 1;
        th->nb[0] = ggml_type_size(head_type);
        th->nb[1] = ggml_row_size(head_type, th->ne[0]);
        th->nb[2] = th->nb[1]*th->ne[1];
        th->nb[3] = th->nb[2]*th->ne[2];
        tb->ne[0] = n_short + 1;
        tb->nb[1] = tb->nb[0]*tb->ne[0];
        tb->nb[2] = tb->nb[1]*tb->ne[1];
        tb->nb[3] = tb->nb[2]*tb->ne[2];
    }

    // weights: the vocabulary map stays on the host too
    buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, buft);
    if (!buf_w) {
        LLAMA_LOG_ERROR("%s: cannot allocate the weights on %s\n", __func__, ggml_backend_dev_name(dev));
        return false;
    }
    {
        FILE * f = fopen(params.path, "rb");
        if (!f) {
            LLAMA_LOG_ERROR("%s: cannot open %s\n", __func__, params.path);
            return false;
        }
        // the file is read and uploaded in pieces of at most CHUNK bytes, so that loading takes a few MB of host memory
        // (the rig's host has 4 GB, most of it taken by the target's load)
        constexpr size_t CHUNK = 8u << 20;
        const size_t data_off = gguf_get_data_offset(g);
        std::vector<uint8_t> tmp, out;
        std::vector<float> rows;
        auto read_at = [&](size_t off, size_t n) {
            tmp.resize(n);
            return fseek(f, (long) off, SEEK_SET) == 0 && fread(tmp.data(), 1, n, f) == n;
        };
        bool ok = true;
        for (int64_t i = 0; ok && i < gguf_get_n_tensors(g); i++) {
            const char * name = gguf_get_tensor_name(g, i);
            ggml_tensor * t = ggml_get_tensor(ctx_w, name);
            const size_t nb = gguf_get_tensor_size(g, i);
            const size_t off = data_off + gguf_get_tensor_offset(g, i);
            if (head_converted && t == th) {
                // rows 0..n_short-1, then the other class (the file's last row), converted a piece of rows at a time
                const size_t rs_in  = ggml_row_size(head_file_type, n_embd);
                const size_t rs_out = ggml_row_size(head_type, n_embd);
                const int64_t per = std::max<int64_t>(1, (int64_t) (CHUNK/rs_in));
                auto convert = [&](int64_t src, int64_t n, int64_t dst) {
                    if (!read_at(off + (size_t) src*rs_in, (size_t) n*rs_in)) {
                        return false;
                    }
                    rows.resize((size_t) n*n_embd);
                    out.resize((size_t) n*rs_out);
                    for (int64_t r = 0; r < n; r++) {
                        ggml_get_type_traits(head_file_type)->to_float(tmp.data() + (size_t) r*rs_in, rows.data() + (size_t) r*n_embd, n_embd);
                    }
                    if (head_type == GGML_TYPE_F16) {
                        ggml_fp32_to_fp16_row(rows.data(), (ggml_fp16_t *) out.data(), (int64_t) rows.size());
                    } else {
                        ggml_quantize_chunk(head_type, rows.data(), out.data(), 0, n, n_embd, nullptr);
                    }
                    ggml_backend_tensor_set(t, out.data(), (size_t) dst*rs_out, out.size());
                    return true;
                };
                for (int64_t r0 = 0; ok && r0 < n_short; r0 += per) {
                    ok = convert(r0, std::min<int64_t>(per, n_short - r0), r0);
                }
                ok = ok && convert(n_vocab_draft, 1, n_short);
                continue;
            }
            if (head_converted && t == tb) {
                ok = read_at(off, nb);
                if (ok) {
                    std::vector<float> b(n_short + 1);
                    for (int32_t r = 0; r <= n_short; r++) {
                        b[r] = ((const float *) tmp.data())[r < n_short ? r : n_vocab_draft];
                    }
                    ggml_backend_tensor_set(t, b.data(), 0, b.size()*sizeof(float));
                }
                continue;
            }
            for (size_t o = 0; ok && o < nb; o += CHUNK) {
                const size_t n = std::min(CHUNK, nb - o);
                ok = read_at(off + o, n);
                if (ok && strcmp(name, "sd.vocab") == 0) {
                    vocab.insert(vocab.end(), (const int32_t *) tmp.data(), (const int32_t *) (tmp.data() + n));
                }
                if (ok) {
                    ggml_backend_tensor_set(t, tmp.data(), o, n);
                }
            }
            if (!ok) {
                LLAMA_LOG_ERROR("%s: short read of %s\n", __func__, name);
            }
        }
        fclose(f);
        if (!ok) {
            LLAMA_LOG_ERROR("%s: cannot read the weights of %s\n", __func__, params.path);
            return false;
        }
    }
    auto get = [&](const std::string & name, bool required = true) {
        ggml_tensor * t = ggml_get_tensor(ctx_w, name.c_str());
        if (!t && required) {
            throw std::runtime_error("missing tensor " + name);
        }
        return t;
    };
    w_emb      = get("sd.emb.weight");
    w_anc      = get("sd.anc.weight");
    anchor_emb = get("sd.anchor_emb.weight");
    w_fuse     = fuse ? get("sd.fuse.weight") : nullptr;
    w_head     = get("sd.head.weight");
    b_head     = get("sd.head.bias");
    blocks.resize(n_layer);
    for (int32_t l = 0; l < n_layer; l++) {
        const std::string p = "sd.blk." + std::to_string(l) + ".";
        blocks[l] = { get(p + "wq"), get(p + "wk"), get(p + "wv"), get(p + "wo"), get(p + "w1"), get(p + "w3"), get(p + "w2") };
    }
    const int32_t max_proj = *std::max_element(depth_proj.begin(), depth_proj.end());
    const int32_t max_row  = *std::max_element(depth_emb_row.begin(), depth_emb_row.end());
    if ((int32_t) vocab.size() != n_vocab_draft || w_head->ne[1] != n_short + 1 || w_anc->ne[2] <= max_proj ||
            anchor_emb->ne[1] <= max_row || w_emb->ne[0] != n_embd || (fuse && w_fuse->ne[2] != (int64_t) fuse_depths.size())) {
        LLAMA_LOG_ERROR("%s: tensor shapes do not match the metadata\n", __func__);
        return false;
    }

    // where the input embeddings come from
    if (tok_embd == nullptr || tok_embd->ne[0] != n_embd) {
        LLAMA_LOG_ERROR("%s: the target's embedding table does not match the drafter's width %d\n", __func__, n_embd);
        return false;
    }
    emb_in_graph = false;
    if (tok_embd->buffer && !ggml_backend_buffer_is_host(tok_embd->buffer) &&
            ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tok_embd->buffer)) == dev) {
        // only if the device can gather rows of that type
        ggml_init_params ip = { ggml_tensor_overhead()*4, nullptr, true };
        ggml_context * c = ggml_init(ip);
        ggml_tensor * ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4);
        ggml_tensor * op  = ggml_get_rows(c, tok_embd, ids);
        emb_in_graph = ggml_backend_supports_op(backend, op);
        ggml_free(c);
    }
    if (!emb_in_graph && !(tok_embd->buffer && ggml_backend_buffer_is_host(tok_embd->buffer))) {
        LLAMA_LOG_ERROR("%s: the embedding table is on another device than the drafter (%s); put the drafter there or the "
                "table in host memory (LLAMA_INPUT_ON_GPU=0)\n", __func__, ggml_backend_dev_name(dev));
        return false;
    }
    n_tap = (int32_t) tap_depth.size() + (emb_in_graph ? 0 : 1);
    if (n_tap > 64) {
        LLAMA_LOG_ERROR("%s: %d taps per evaluation, at most 64\n", __func__, n_tap);
        return false;
    }

    // state: K/V cells, cell positions, the store of uploaded taps
    {
        n_wcell   = std::max(32, GGML_PAD(n_rows_step, 8));
        n_scratch = std::max(16, GGML_PAD(n_rows_step, 8));
        const int32_t n_req = std::max(256, params.n_ctx);
        n_cells = GGML_PAD(n_req + n_wcell + n_scratch, 256);
        n_ring  = n_cells - n_wcell - n_scratch;

        ggml_init_params ip = { ggml_tensor_overhead()*(2*n_layer + 4), nullptr, true };
        ctx_s = ggml_init(ip);
        for (int32_t l = 0; l < n_layer; l++) {
            k_cache.push_back(ggml_new_tensor_2d(ctx_s, GGML_TYPE_F16, head_dim*n_head_kv, n_cells));
            v_cache.push_back(ggml_new_tensor_2d(ctx_s, GGML_TYPE_F16, head_dim*n_head_kv, n_cells));
        }
        cell_pos = ggml_new_tensor_2d(ctx_s, GGML_TYPE_F32, 1, n_cells);
        store    = ggml_new_tensor_2d(ctx_s, GGML_TYPE_F32, n_embd, 1 + n_eval_ring*n_tap);
        buf_s = ggml_backend_alloc_ctx_tensors_from_buft(ctx_s, buft);
        if (!buf_s) {
            LLAMA_LOG_ERROR("%s: cannot allocate the drafter's state\n", __func__);
            return false;
        }
        // every cell finite (a masked cell is still multiplied by zero), every cell invisible
        ggml_backend_buffer_clear(buf_s, 0);
        cell_pos_host.assign(n_cells, POS_NONE);
        cell_pos_dirty = true;
    }

    // taps: pinned rows and their events
    buf_tap = ggml_backend_buft_alloc_buffer(buft_host, (size_t) n_eval_ring*n_tap*n_embd*sizeof(float));
    if (!buf_tap) {
        LLAMA_LOG_ERROR("%s: cannot allocate the tap buffer\n", __func__);
        return false;
    }
    tap_host = (float *) ggml_backend_buffer_get_base(buf_tap);
    memset(tap_host, 0, ggml_backend_buffer_get_size(buf_tap));
    ev_tap.assign((size_t) n_eval_ring*n_tap, nullptr);
    ev_tap_dev.assign((size_t) n_eval_ring*n_tap, nullptr);
    uploaded.assign(n_eval_ring, 0);

    ev_launch.resize(n_launch_ring);
    for (auto & ev : ev_launch) {
        ev = ggml_backend_event_new(dev);
    }
    launch_pending.assign(n_launch_ring, false);
    for (auto & c : caps) {
        c.ev_done = ggml_backend_event_new(dev);
    }

    ensure_graphs();
    if (!graphs_valid) {
        return false;
    }

    std::string sp, st, tp;
    for (auto s : split) {
        sp += (sp.empty() ? "" : "/") + std::to_string(s);
    }
    for (auto d : stair) {
        st += (st.empty() ? "" : ",") + depth_str(d, n_target);
    }
    for (auto d : tap_depth) {
        tp += (tp.empty() ? "" : ",") + depth_str(d, n_target);
    }
    LLAMA_LOG_INFO("%s: staircase drafter %s (profile %s%s): %d layers, %d draft ids%s; split %s, %s window %s, older tokens "
            "%s, taps %s; alpha %.2f, top-%d; on %s, %d cells, %d rows per guess, embeddings %s; output head %d ids + other "
            "(%s), last block %s, projections %s, products in groups of %d rows\n", __func__, params.path,
            profile.c_str(), general ? ", depth-general" : "", n_layer, n_vocab_draft, fuse ? ", fused" : "", sp.c_str(),
            kind.c_str(), st.c_str(), depth_str(depth_deep, n_target).c_str(), tp.c_str(), alpha, top_k,
            ggml_backend_dev_name(dev), n_ring, n_rows_step, emb_in_graph ? "gathered on the device" : "looked up on the host",
            n_short, ggml_type_name(w_head->type), prune_last ? "on the output row" : "on every row",
            anc_dense ? "dense" : "per row", split_cols > 0 ? split_cols : n_rows_step);
    return true;
}

std::vector<int32_t> llama_pipe_drafter::tap_layers() const {
    std::vector<int32_t> res;
    for (int32_t d : tap_depth) {
        if (d < n_target) {
            res.push_back(d);
        }
    }
    return res;
}

void llama_pipe_drafter::lookup_emb(llama_token tok, float * dst) const {
    const auto * tr = ggml_get_type_traits(tok_embd->type);
    const char * row = (const char *) tok_embd->data + (size_t) tok*tok_embd->nb[1];
    if (tok_embd->type == GGML_TYPE_F32) {
        memcpy(dst, row, n_embd*sizeof(float));
    } else {
        tr->to_float(row, dst, n_embd);
    }
}

// one graph: n_rows rows (bulk: committed rows whose hidden states come from h_full / h_fuse; else rows of any depth
// gathered from the store, and the head on out_row)
bool llama_pipe_drafter::build_graph(graph_inst & g, int32_t n_rows, bool bulk) {
    g.n_rows = n_rows;
    g.bulk   = bulk;
    const int32_t N  = n_rows;
    const int32_t NW = n_wcell + n_scratch;
    const int32_t n_fuse = (int32_t) fuse_depths.size();

    // inputs: small ones packed in one blob (one upload per launch), large ones after it
    {
        ggml_init_params ip = { ggml_tensor_overhead()*32, nullptr, true };
        g.ctx_in = ggml_init(ip);
    }
    struct place { ggml_tensor * t; size_t off; };
    std::vector<place> small, large;
    size_t off = 0;
    auto add_small = [&](ggml_type type, int64_t ne0, int64_t ne1 = 1) {
        ggml_tensor * t = ggml_new_tensor_2d(g.ctx_in, type, ne0, ne1);
        ggml_set_input(t);
        small.push_back({ t, off });
        off = GGML_PAD(off + ggml_nbytes(t), 256);
        return t;
    };
    g.tokens     = add_small(GGML_TYPE_I32, N);
    g.rope_pos   = add_small(GGML_TYPE_I32, N);
    g.row_pos    = add_small(GGML_TYPE_F32, 1, N);
    g.cellval    = add_small(GGML_TYPE_F32, 1, N);
    g.kv_idx     = add_small(GGML_TYPE_I64, N);
    g.anc_emb_id = add_small(GGML_TYPE_I32, N);
    g.wcell_idx  = add_small(GGML_TYPE_I64, NW);
    g.wcell_pos  = add_small(GGML_TYPE_F32, 1, NW);
    if (!bulk) {
        g.anc_id   = add_small(GGML_TYPE_I32, 1, N);
        g.anc_idx  = add_small(GGML_TYPE_I32, N);
        g.fuse_idx = fuse ? add_small(GGML_TYPE_I32, n_fuse*N) : nullptr;
        g.out_row  = add_small(GGML_TYPE_I32, 1);
        g.emb_idx  = emb_in_graph ? nullptr : add_small(GGML_TYPE_I32, N);   // store rows of the embeddings
        g.out_pos  = add_small(GGML_TYPE_I32, 1);
        g.anc_sel  = anc_dense ? add_small(GGML_TYPE_I32, N) : nullptr;
    }
    g.blob_size = off;
    size_t off_large = off;
    auto add_large = [&](int64_t ne0, int64_t ne1) {
        ggml_tensor * t = ggml_new_tensor_2d(g.ctx_in, GGML_TYPE_F32, ne0, ne1);
        ggml_set_input(t);
        large.push_back({ t, off_large });
        off_large = GGML_PAD(off_large + ggml_nbytes(t), 256);
        return t;
    };
    const bool fuse_committed = fuse && committed == n_target;
    if (bulk) {
        g.h_full   = add_large(n_embd, N);
        g.h_fuse   = fuse_committed ? add_large(n_embd, N) : nullptr;
        g.emb_rows = emb_in_graph ? nullptr : add_large(n_embd, N);
    }
    g.buf_in = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev), off_large);
    if (!g.buf_in) {
        return false;
    }
    ggml_backend_buffer_clear(g.buf_in, 0);
    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(g.buf_in);
    g.blob = ggml_new_tensor_1d(g.ctx_in, GGML_TYPE_I8, g.blob_size);
    ggml_backend_tensor_alloc(g.buf_in, g.blob, base);
    for (auto & p : small) {
        ggml_backend_tensor_alloc(g.buf_in, p.t, base + p.off);
    }
    for (auto & p : large) {
        ggml_backend_tensor_alloc(g.buf_in, p.t, base + p.off);
    }

    ggml_init_params ip = { ggml_tensor_overhead()*GRAPH_NODES + ggml_graph_overhead_custom(GRAPH_NODES, false), nullptr, true };
    g.ctx = ggml_init(ip);
    ggml_context * ctx = g.ctx;
    g.gf = ggml_new_graph_custom(ctx, GRAPH_NODES, false);
    ggml_cgraph * gf = g.gf;

    // cell positions: every row writes its own, the window and scratch cells are rewritten as a block
    ggml_tensor * cp = ggml_set_rows(ctx, cell_pos, g.cellval, g.kv_idx);
    cp = ggml_set_rows(ctx, cp, g.wcell_pos, g.wcell_idx);
    // mask[c, r] = 0 where cell c's position <= row r's position, else -inf
    ggml_tensor * mask = nullptr;
    ggml_tensor * mask_out = nullptr;   // the output row's (pruned last block)
    const bool prune = !bulk && prune_last;
    {
        ggml_tensor * cpc  = ggml_reshape_2d(ctx, cp, n_cells, 1);
        ggml_tensor * rp   = ggml_repeat_4d(ctx, g.row_pos, n_cells, N, 1, 1);
        ggml_tensor * diff = ggml_sub(ctx, rp, cpc);
        ggml_tensor * vis  = ggml_step(ctx, ggml_scale_bias(ctx, diff, 1.0f, 0.5f));
        ggml_tensor * m32  = ggml_scale_bias(ctx, vis, 1e30f, -1e30f);
        mask = ggml_cast(ctx, m32, GGML_TYPE_F16);
        if (prune) {
            mask_out = ggml_cast(ctx, ggml_get_rows(ctx, m32, g.out_row), GGML_TYPE_F16);
        }
    }

    // rows: embedding, projected hidden state, fused shallower depths, depth embedding
    ggml_tensor * e;
    if (emb_in_graph) {
        e = ggml_get_rows(ctx, tok_embd, g.tokens);
    } else if (bulk) {
        e = g.emb_rows;
    } else {
        e = ggml_get_rows(ctx, store, g.emb_idx);
    }
    // a matrix product of the step graph, in groups of split_cols rows if set
    auto mm = [&](ggml_tensor * w, ggml_tensor * a) {
        const int64_t nc = a->ne[1];
        if (bulk || split_cols <= 0 || nc <= split_cols) {
            return ggml_mul_mat(ctx, w, a);
        }
        ggml_tensor * out = nullptr;
        for (int64_t c0 = 0; c0 < nc; c0 += split_cols) {
            const int64_t m = std::min<int64_t>(split_cols, nc - c0);
            ggml_tensor * y = ggml_mul_mat(ctx, w, ggml_view_2d(ctx, a, a->ne[0], m, a->nb[1], (size_t) c0*a->nb[1]));
            out = out ? ggml_concat(ctx, out, y, 1) : y;
        }
        return out;
    };
    ggml_tensor * x = mm(w_emb, ggml_rms_norm(ctx, e, rms_eps));
    auto mat = [&](ggml_tensor * w, int32_t k) {   // the k-th matrix of a stacked [d, d, n] tensor
        return ggml_view_2d(ctx, w, w->ne[0], w->ne[1], w->nb[1], (size_t) k*w->nb[2]);
    };
    if (bulk) {
        // h_full holds the committed depth's states (normally the final state)
        if (committed > 0) {
            x = ggml_add(ctx, x, ggml_mul_mat(ctx, mat(w_anc, depth_proj[committed]), ggml_rms_norm(ctx, g.h_full, rms_eps)));
        }
        if (fuse_committed) {
            x = ggml_add(ctx, x, ggml_mul_mat(ctx, mat(w_fuse, 0), ggml_rms_norm(ctx, g.h_fuse, rms_eps)));
        }
    } else {
        ggml_tensor * h  = ggml_rms_norm(ctx, ggml_get_rows(ctx, store, g.anc_idx), rms_eps);
        if (anc_dense) {
            // every projection of every row in one product, then each row's own (row i, projection k: i*n_proj + k)
            const int64_t n_proj = w_anc->ne[2];
            ggml_tensor * hall = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, w_anc, n_embd, n_embd*n_proj), h);
            x = ggml_add(ctx, x, ggml_get_rows(ctx, ggml_reshape_2d(ctx, hall, n_embd, n_proj*N), g.anc_sel));
        } else {
            ggml_tensor * ha = ggml_mul_mat_id(ctx, w_anc, ggml_reshape_3d(ctx, h, n_embd, 1, N), g.anc_id);
            x = ggml_add(ctx, x, ggml_reshape_2d(ctx, ha, n_embd, N));
        }
        if (fuse) {
            ggml_tensor * hf = ggml_rms_norm(ctx, ggml_get_rows(ctx, store, g.fuse_idx), rms_eps);
            for (int32_t k = 0; k < n_fuse; k++) {
                ggml_tensor * hk = ggml_view_2d(ctx, hf, n_embd, N, hf->nb[1], (size_t) k*N*hf->nb[1]);
                x = ggml_add(ctx, x, ggml_mul_mat(ctx, mat(w_fuse, k), hk));
            }
        }
    }
    x = ggml_add(ctx, x, ggml_get_rows(ctx, anchor_emb, g.anc_emb_id));

    const float kq_scale = 1.0f/sqrtf((float) head_dim);
    for (int32_t l = 0; l < n_layer; l++) {
        const auto & b = blocks[l];
        const bool last = l == n_layer - 1;

        ggml_tensor * hn = ggml_rms_norm(ctx, x, rms_eps);
        ggml_tensor * k = ggml_reshape_3d(ctx, mm(b.wk, hn), head_dim, n_head_kv, N);
        k = ggml_rope_ext(ctx, k, g.rope_pos, nullptr, head_dim, 0, 0, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * v = mm(b.wv, hn);
        ggml_tensor * kc = ggml_set_rows(ctx, k_cache[l], ggml_reshape_2d(ctx, k, head_dim*n_head_kv, N), g.kv_idx);
        ggml_tensor * vc = ggml_set_rows(ctx, v_cache[l], v, g.kv_idx);
        if (bulk && last) {
            // committed rows of the last block only leave their keys and values
            ggml_build_forward_expand(gf, kc);
            ggml_build_forward_expand(gf, vc);
            break;
        }
        // the last block of a guess: every row leaves its keys and values, only the output row goes on
        const bool one = prune && last;
        const int32_t nq = one ? 1 : N;
        if (one) {
            x  = ggml_get_rows(ctx, x, g.out_row);
            hn = ggml_get_rows(ctx, hn, g.out_row);
        }
        ggml_tensor * q = ggml_reshape_3d(ctx, mm(b.wq, hn), head_dim, n_head, nq);
        q = ggml_rope_ext(ctx, q, one ? g.out_pos : g.rope_pos, nullptr, head_dim, 0, 0, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        const size_t es  = ggml_element_size(kc);
        const size_t row = (size_t) head_dim*n_head_kv*es;
        ggml_tensor * kv = ggml_permute(ctx, ggml_view_3d(ctx, kc, head_dim, n_head_kv, n_cells, head_dim*es, row, 0), 0, 2, 1, 3);
        ggml_tensor * vv = ggml_permute(ctx, ggml_view_3d(ctx, vc, head_dim, n_head_kv, n_cells, head_dim*es, row, 0), 0, 2, 1, 3);
        ggml_tensor * qp = ggml_permute(ctx, q, 0, 2, 1, 3);

        ggml_tensor * att = ggml_flash_attn_ext(ctx, qp, kv, vv, one ? mask_out : mask, kq_scale, 0.0f, 0.0f);
        ggml_prec_set_acc(att, GGML_PREC_F32);
        att = ggml_reshape_2d(ctx, att, head_dim*n_head, nq);
        x = ggml_add(ctx, x, mm(b.wo, att));

        ggml_tensor * hm = ggml_rms_norm(ctx, x, rms_eps);
        ggml_tensor * gu = ggml_swiglu_split(ctx, mm(b.w1, hm), mm(b.w3, hm));
        x = ggml_add(ctx, x, mm(b.w2, gu));
    }

    if (!bulk) {
        ggml_tensor * f = ggml_rms_norm(ctx, prune && n_layer > 0 ? x : ggml_get_rows(ctx, x, g.out_row), rms_eps);
        ggml_tensor * logits = ggml_add(ctx, ggml_mul_mat(ctx, w_head, f), b_head);
        ggml_tensor * probs  = ggml_soft_max_ext(ctx, logits, nullptr, alpha, 0.0f);
        const int32_t K = std::min(top_k, n_short + 1);
        g.top_ids = ggml_top_k(ctx, probs, K);
        g.top_p   = ggml_get_rows(ctx, ggml_reshape_2d(ctx, probs, 1, n_short + 1), g.top_ids);
        ggml_set_output(g.top_ids);
        ggml_set_output(g.top_p);
        ggml_build_forward_expand(gf, g.top_ids);
        ggml_build_forward_expand(gf, g.top_p);
        if (keep_logits) {
            g.logits = logits;
            ggml_set_output(logits);
            ggml_build_forward_expand(gf, logits);
        }
    }

    g.galloc = ggml_gallocr_new(ggml_backend_dev_buffer_type(dev));
    if (!ggml_gallocr_alloc_graph(g.galloc, gf)) {
        LLAMA_LOG_ERROR("%s: cannot allocate the drafter's compute buffer\n", __func__);
        return false;
    }
    // every op must run on the drafter's device
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (!ggml_backend_supports_op(backend, node)) {
            LLAMA_LOG_ERROR("%s: %s cannot run %s (%s)\n", __func__, ggml_backend_dev_name(dev), ggml_op_desc(node), node->name);
            return false;
        }
    }
    return true;
}

void llama_pipe_drafter::ensure_graphs() {
    if (graphs_valid) {
        return;
    }
    synchronize();
    g_step.reset(new graph_inst());
    g_bulk_small.reset(new graph_inst());
    g_bulk_large.reset(new graph_inst());
    if (!build_graph(*g_step, n_rows_step, false) || !build_graph(*g_bulk_small, 64, true) ||
            !build_graph(*g_bulk_large, 256, true)) {
        LLAMA_LOG_ERROR("%s: cannot build the drafter's graphs\n", __func__);
        graphs_valid = false;
        return;
    }
    // per launch: the blob, a copy of the cell positions, the top-k and optionally the logits
    const size_t blob_max = std::max({ g_step->blob_size, g_bulk_small->blob_size, g_bulk_large->blob_size });
    off_cp  = GGML_PAD(blob_max, 256);
    off_ids = off_cp  + GGML_PAD((size_t) n_cells*sizeof(float), 256);
    off_p   = off_ids + GGML_PAD((size_t) top_k*4, 256);
    off_lg  = off_p   + GGML_PAD((size_t) top_k*4, 256);
    stage_stride = off_lg + (keep_logits ? GGML_PAD((size_t) (n_short + 1)*sizeof(float), 256) : 0);
    if (buf_stage) {
        ggml_backend_buffer_free(buf_stage);
    }
    buf_stage  = ggml_backend_buft_alloc_buffer(buft_host, stage_stride*n_launch_ring);
    stage_host = (uint8_t *) ggml_backend_buffer_get_base(buf_stage);
    launch_pending.assign(n_launch_ring, false);
    launch_last = -1;
    graphs_valid = true;
}

void llama_pipe_drafter::synchronize() {
    if (backend) {
        ggml_backend_synchronize(backend);
    }
    std::fill(launch_pending.begin(), launch_pending.end(), false);
    for (auto & c : caps) {
        c.pending = false;
    }
}

void llama_pipe_drafter::stage_wait(int32_t i) {
    if (launch_pending[i]) {
        if (ev_launch[i]) {
            ggml_backend_event_synchronize(ev_launch[i]);
        } else {
            ggml_backend_synchronize(backend);
        }
        launch_pending[i] = false;
    }
}

// marks the end of launch i on the drafter's stream (a device without events has finished it already)
void llama_pipe_drafter::launch_done(int32_t i) {
    if (ev_launch[i]) {
        ggml_backend_event_record(ev_launch[i], backend);
        launch_pending[i] = true;
    } else {
        launch_pending[i] = false;
    }
}

void llama_pipe_drafter::on_submit(int32_t i_slot, llama_token tok, llama_pos pos, llama_seq_id seq_id, const llm_graph_result * res,
                                   ggml_backend_sched_t sched) {
    GGML_UNUSED(pos);
    if ((int32_t) eval_of_slot.size() <= i_slot) {
        eval_of_slot.resize(i_slot + 1, -1);
    }
    if (seq_id != seq || seq < 0) {
        eval_of_slot[i_slot] = -1;
        return;
    }
    const int64_t e = n_eval++;
    const int32_t r = (int32_t) (e % n_eval_ring);
    eval_of_slot[i_slot] = e;
    uploaded[r] = 0;

    for (int32_t t = 0; t < (int32_t) tap_depth.size(); t++) {
        const int32_t d = tap_depth[t];
        ggml_tensor * src = d < n_target ? res->get_layer_inp(d) : res->get_embd();
        if (src == nullptr || ggml_nelements(src) < n_embd) {
            continue;
        }
        ggml_backend_t be = ggml_backend_sched_get_tensor_backend(sched, src);
        if (be == nullptr || src->data == nullptr) {
            continue;
        }
        float * dst = tap_host + ((size_t) r*n_tap + t)*n_embd;
        // the last row: the token's own (a one-token evaluation has one)
        const size_t off = (size_t) (src->ne[1] - 1)*src->nb[1];
        ggml_backend_tensor_get_async(be, src, dst, off, (size_t) n_embd*sizeof(float));
        const size_t ie = (size_t) r*n_tap + t;
        ggml_backend_dev_t dv = ggml_backend_get_device(be);
        if (ev_tap[ie] == nullptr || ev_tap_dev[ie] != dv) {
            if (ev_tap[ie]) {
                ggml_backend_event_free(ev_tap[ie]);
            }
            ev_tap[ie]     = ggml_backend_event_new(dv);
            ev_tap_dev[ie] = dv;
        }
        if (ev_tap[ie]) {
            ggml_backend_event_record(ev_tap[ie], be);
        } else {
            ggml_backend_synchronize(be);
        }
    }
    if (!emb_in_graph) {
        lookup_emb(tok, tap_host + ((size_t) r*n_tap + tap_depth.size())*n_embd);
    }
}

void llama_pipe_drafter::test_set_taps(int32_t i_slot, llama_token tok, llama_pos pos, const float * h, const int32_t * depths,
                                       int32_t n) {
    GGML_UNUSED(pos);
    if ((int32_t) eval_of_slot.size() <= i_slot) {
        eval_of_slot.resize(i_slot + 1, -1);
    }
    const int64_t e = n_eval++;
    const int32_t r = (int32_t) (e % n_eval_ring);
    eval_of_slot[i_slot] = e;
    uploaded[r] = 0;
    for (int32_t t = 0; t < (int32_t) tap_depth.size(); t++) {
        const size_t ie = (size_t) r*n_tap + t;
        float * dst = tap_host + ie*n_embd;
        memset(dst, 0, n_embd*sizeof(float));
        for (int32_t i = 0; i < n; i++) {
            if (depths[i] == tap_depth[t]) {
                memcpy(dst, h + (size_t) i*n_embd, n_embd*sizeof(float));
            }
        }
        if (ev_tap[ie]) {
            ggml_backend_event_free(ev_tap[ie]);
            ev_tap[ie] = nullptr;
        }
    }
    if (!emb_in_graph) {
        lookup_emb(tok, tap_host + ((size_t) r*n_tap + tap_depth.size())*n_embd);
    }
}

void llama_pipe_drafter::upload_tap(int64_t e, int32_t t) {
    const int32_t r = (int32_t) (e % n_eval_ring);
    if (uploaded[r] & (1ull << t)) {
        return;
    }
    const size_t ie = (size_t) r*n_tap + t;
    if (ev_tap[ie]) {
        ggml_backend_event_wait(backend, ev_tap[ie]);
    }
    const size_t row = 1 + ie;
    ggml_backend_tensor_set_async(backend, store, tap_host + ie*n_embd, row*n_embd*sizeof(float), (size_t) n_embd*sizeof(float));
    uploaded[r] |= 1ull << t;
}

// the store row of an evaluation's hidden state at a depth (uploaded on first use); 0 (zeros) for depth 0 or a depth
// that is not tapped
int32_t llama_pipe_drafter::store_row(int64_t e, int32_t depth) {
    const int32_t t = depth > 0 && depth <= n_target ? tap_of_depth[depth] : -1;
    if (t < 0) {
        return 0;
    }
    upload_tap(e, t);
    return (int32_t) (1 + (e % n_eval_ring)*n_tap + t);
}

void llama_pipe_drafter::reset(llama_seq_id seq_id, llama_pos n_keep) {
    if (seq_id != seq) {
        n_keep = 0;
    }
    seq = seq_id;
    commits.clear();
    launch_last = -1;
    for (int32_t c = 0; c < n_ring; c++) {
        if (cell_pos_host[c] != POS_NONE && cell_pos_host[c] >= (float) n_keep) {
            cell_pos_host[c] = POS_NONE;
            cell_pos_dirty = true;
        }
    }
}

void llama_pipe_drafter::commit(llama_token tok, llama_pos pos, int32_t i_slot) {
    if (seq < 0 || i_slot < 0 || i_slot >= (int32_t) eval_of_slot.size() || eval_of_slot[i_slot] < 0) {
        return;
    }
    commits.push_back({ tok, pos, eval_of_slot[i_slot] });
}

void llama_pipe_drafter::write_blob(graph_inst & g, uint8_t * dst, const std::vector<int32_t> & tok,
                                    const std::vector<int32_t> & rope, const std::vector<float> & rpos,
                                    const std::vector<float> & cval, const std::vector<int32_t> & aid,
                                    const std::vector<int32_t> & aeid, const std::vector<int32_t> & aidx,
                                    const std::vector<int32_t> & fidx, const std::vector<int64_t> & kvi,
                                    const std::vector<float> & wpos, int32_t out_row) {
    auto put = [&](ggml_tensor * t, const void * src, size_t n) {
        if (t) {
            GGML_ASSERT(n <= ggml_nbytes(t));
            memcpy(dst + ((const uint8_t *) t->data - (const uint8_t *) g.blob->data), src, n);
        }
    };
    const int32_t NW = n_wcell + n_scratch;
    std::vector<int64_t> widx(NW);
    for (int32_t j = 0; j < NW; j++) {
        widx[j] = n_ring + j;
    }
    put(g.tokens,     tok.data(),  tok.size()*4);
    put(g.rope_pos,   rope.data(), rope.size()*4);
    put(g.row_pos,    rpos.data(), rpos.size()*4);
    put(g.cellval,    cval.data(), cval.size()*4);
    put(g.kv_idx,     kvi.data(),  kvi.size()*8);
    put(g.anc_emb_id, aeid.data(), aeid.size()*4);
    put(g.wcell_idx,  widx.data(), widx.size()*8);
    put(g.wcell_pos,  wpos.data(), wpos.size()*4);
    if (!g.bulk) {
        put(g.anc_id,  aid.data(),  aid.size()*4);
        put(g.anc_idx, aidx.data(), aidx.size()*4);
        if (g.fuse_idx) {
            put(g.fuse_idx, fidx.data(), fidx.size()*4);
        }
        put(g.out_row, &out_row, 4);
        put(g.out_pos, &rope[out_row], 4);
        if (g.anc_sel) {
            const int32_t n_proj = (int32_t) w_anc->ne[2];
            std::vector<int32_t> sel(aid.size());
            for (size_t i = 0; i < aid.size(); i++) {
                sel[i] = (int32_t) i*n_proj + aid[i];
            }
            put(g.anc_sel, sel.data(), sel.size()*4);
        }
    }
}

// one step launch: rows (commits first, then the window), the head on row i_out
void llama_pipe_drafter::run_rows(const std::vector<int64_t> & rows_eval, const std::vector<int32_t> & rows_depth,
                                  const std::vector<llama_token> & rows_tok, const std::vector<llama_pos> & rows_pos,
                                  const std::vector<bool> & rows_window, int32_t i_out) {
    graph_inst & g = *g_step;
    const int32_t N  = g.n_rows;
    const int32_t NW = n_wcell + n_scratch;
    const int32_t n_fuse = (int32_t) fuse_depths.size();
    const int32_t n_real = (int32_t) rows_tok.size();
    GGML_ASSERT(n_real <= N);

    std::vector<int32_t> tok(N, 0), rope(N, 0), aid(N, 0), aeid(N, depth_emb_row[0]), aidx(N, 0),
                         fidx(fuse ? (size_t) n_fuse*N : 0, 0);
    std::vector<int32_t> eidx(emb_in_graph ? 0 : N, 0);
    std::vector<float>   rpos(N, POS_PAD), cval(N, POS_NONE), wpos(NW, POS_NONE);
    std::vector<int64_t> kvi(N);
    for (int32_t i = 0; i < N; i++) {
        kvi[i] = n_ring + n_wcell + (i % n_scratch);
    }
    for (int32_t i = 0; i < n_real; i++) {
        const int64_t e = rows_eval[i];
        const int32_t d = rows_depth[i];
        tok[i]  = rows_tok[i];
        rope[i] = rows_pos[i];
        rpos[i] = (float) rows_pos[i];
        cval[i] = (float) rows_pos[i];
        aeid[i] = depth_emb_row[d];
        aid[i]  = d > 0 ? depth_proj[d] : 0;
        aidx[i] = store_row(e, d);
        if (rows_window[i]) {
            const int32_t c = rows_pos[i] % n_wcell;
            kvi[i] = n_ring + c;
            wpos[c] = (float) rows_pos[i];
        } else {
            kvi[i] = rows_pos[i] % n_ring;
            cell_pos_host[kvi[i]] = (float) rows_pos[i];
        }
        if (fuse) {
            for (int32_t k = 0; k < n_fuse; k++) {
                // window rows: every shallower fused depth; committed rows and complete tokens in flight: the first fused
                // depth next to the final state
                const bool use = rows_window[i] && d < n_target ? fuse_depths[k] < d : (k == 0 && d == n_target);
                fidx[(size_t) k*N + i] = use ? store_row(e, fuse_depths[k]) : 0;
            }
        }
        if (!emb_in_graph) {
            const int32_t t = (int32_t) tap_depth.size();
            upload_tap(e, t);
            eidx[i] = (int32_t) (1 + (e % n_eval_ring)*n_tap + t);
        }
    }

    const int32_t li = launch_next;
    launch_next = (launch_next + 1) % n_launch_ring;
    stage_wait(li);
    uint8_t * st = stage_host + (size_t) li*stage_stride;
    write_blob(g, st, tok, rope, rpos, cval, aid, aeid, aidx, fidx, kvi, wpos, i_out);
    if (!emb_in_graph) {
        memcpy(st + ((const uint8_t *) g.emb_idx->data - (const uint8_t *) g.blob->data), eidx.data(), eidx.size()*4);
    }
    uint8_t * st_cp  = st + off_cp;
    uint8_t * st_ids = st + off_ids;
    uint8_t * st_p   = st + off_p;
    uint8_t * st_lg  = st + off_lg;

    if (cell_pos_dirty) {
        memcpy(st_cp, cell_pos_host.data(), (size_t) n_cells*sizeof(float));
        ggml_backend_tensor_set_async(backend, cell_pos, st_cp, 0, (size_t) n_cells*sizeof(float));
        cell_pos_dirty = false;
    }
    ggml_backend_tensor_set_async(backend, g.blob, st, 0, g.blob_size);
    ggml_backend_graph_compute_async(backend, g.gf);
    ggml_backend_tensor_get_async(backend, g.top_ids, st_ids, 0, ggml_nbytes(g.top_ids));
    ggml_backend_tensor_get_async(backend, g.top_p,   st_p,   0, ggml_nbytes(g.top_p));
    if (keep_logits && g.logits) {
        ggml_backend_tensor_get_async(backend, g.logits, st_lg, 0, ggml_nbytes(g.logits));
    }
    launch_done(li);
    launch_last = li;
    n_launch++;
    n_rows_launched += n_real;
}

int32_t llama_pipe_drafter::launch(const llama_pipe_draft_row * rows, int32_t n_rows) {
    if (seq < 0 || !graphs_valid) {
        return -1;
    }
    const int32_t N = g_step->n_rows;

    // the window: the newest N - 1 tokens in flight at most (one row is kept for a commit)
    const int32_t w0 = std::max(0, n_rows - (N - 1));
    const int32_t n_win = n_rows - w0;

    std::sort(commits.begin(), commits.end(), [](const pending & a, const pending & b) { return a.pos < b.pos; });

    std::vector<int64_t> re;
    std::vector<int32_t> rd;
    std::vector<llama_token> rt;
    std::vector<llama_pos> rp;
    std::vector<bool> rw;
    auto flush_commits = [&](size_t n_take, bool with_window) {
        re.clear(); rd.clear(); rt.clear(); rp.clear(); rw.clear();
        for (size_t i = 0; i < n_take; i++) {
            re.push_back(commits[i].eval);
            rd.push_back(committed);
            rt.push_back(commits[i].tok);
            rp.push_back(commits[i].pos);
            rw.push_back(false);
        }
        commits.erase(commits.begin(), commits.begin() + n_take);
        int32_t i_out = 0;
        if (with_window) {
            for (int32_t j = w0; j < n_rows; j++) {
                const int32_t off = n_rows - 1 - j;   // 0 = the newest
                const int32_t d = off < (int32_t) stair.size() ? stair[off] : depth_deep;
                const int32_t s = rows[j].slot;
                if (s < 0 || s >= (int32_t) eval_of_slot.size() || eval_of_slot[s] < 0) {
                    continue;
                }
                re.push_back(eval_of_slot[s]);
                rd.push_back(d);
                rt.push_back(rows[j].token);
                rp.push_back(rows[j].pos);
                rw.push_back(true);
            }
            i_out = (int32_t) rt.size() - 1;
        }
        run_rows(re, rd, rt, rp, rw, std::max(0, i_out));
    };
    // a commit whose taps were overwritten by later evaluations is dropped (its row stays missing)
    commits.erase(std::remove_if(commits.begin(), commits.end(),
            [&](const pending & c) { return c.eval < n_eval - n_eval_ring + 2; }), commits.end());
    while ((int32_t) commits.size() > N - n_win) {
        // commit-only graphs of up to N rows until the rest fits next to the window
        flush_commits(std::min(commits.size(), (size_t) N), false);
    }
    launch_has_out = n_win > 0;
    flush_commits(commits.size(), true);
    return 0;
}

llama_token llama_pipe_drafter::get(float u, float * p_top) {
    if (launch_last < 0 || !launch_has_out) {
        return -1;
    }
    const int32_t li = launch_last;
    stage_wait(li);
    const uint8_t * st  = stage_host + (size_t) li*stage_stride;
    const int32_t * ids = (const int32_t *) (st + off_ids);
    const float *   ps  = (const float *) (st + off_p);
    const int32_t K = (int32_t) ggml_nelements(g_step->top_ids);

    // the head's rows: the first n_short draft ids, then the other class
    std::vector<std::pair<float, int32_t>> c;
    for (int32_t i = 0; i < K; i++) {
        c.push_back({ ps[i], ids[i] < n_short ? ids[i] : n_vocab_draft });
    }
    std::sort(c.begin(), c.end(), [](const auto & a, const auto & b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
    if (keep_logits) {
        const float * lg = (const float *) (st + off_lg);
        logits_host.assign(lg, lg + n_short + 1);
    }
    if (p_top) {
        *p_top = c.empty() ? 0.0f : c[0].first;
    }
    // the best token of the draft vocabulary (the "other" class names none)
    int32_t best = -1;
    for (const auto & x : c) {
        if (x.second < n_vocab_draft) {
            best = x.second;
            break;
        }
    }
    int32_t pick = best;
    if (u >= 0.0f) {
        // the sampler draws by the same uniform against its distribution in descending order
        float cum = 0.0f;
        for (const auto & x : c) {
            cum += x.first;
            if (u < cum) {
                pick = x.second < n_vocab_draft ? x.second : best;
                break;
            }
        }
    }
    return pick >= 0 ? vocab[pick] : -1;
}

// committed rows from host rows (the committed depth's states and the first fused depth's) in capture i_cap, from row0
void llama_pipe_drafter::commit_rows_bulk(const llama_token * toks, const llama_pos * pos, int32_t n, int32_t i_cap, size_t row0) {
    capture & cap = caps[i_cap];
    const int32_t NW = n_wcell + n_scratch;
    const bool fuse_committed = fuse && committed == n_target;
    int32_t done = 0;
    while (done < n) {
        const int32_t m = std::min(n - done, n - done > 64 ? 256 : 64);
        graph_inst & g = m > 64 ? *g_bulk_large : *g_bulk_small;
        const int32_t N = g.n_rows;
        std::vector<int32_t> tok(N, 0), rope(N, 0), aeid(N, depth_emb_row[committed]), empty;
        std::vector<float>   rpos(N, POS_PAD), cval(N, POS_NONE), wpos(NW, POS_NONE);
        std::vector<int64_t> kvi(N);
        for (int32_t i = 0; i < N; i++) {
            if (i < m) {
                tok[i]  = toks[done + i];
                rope[i] = pos[done + i];
                rpos[i] = (float) pos[done + i];
                cval[i] = (float) pos[done + i];
                kvi[i]  = pos[done + i] % n_ring;
                cell_pos_host[kvi[i]] = (float) pos[done + i];
            } else {
                kvi[i] = n_ring + n_wcell + (i % n_scratch);
            }
        }
        if (cap.ev_full && committed > 0) {
            ggml_backend_event_wait(backend, cap.ev_full);
        }
        if (cap.ev_fz && fuse_committed) {
            ggml_backend_event_wait(backend, cap.ev_fz);
        }
        const size_t rb = (size_t) n_embd*sizeof(float);
        if (committed > 0) {
            ggml_backend_tensor_set_async(backend, g.h_full, cap.full + (row0 + done)*n_embd, 0, m*rb);
        }
        if (fuse_committed && g.h_fuse) {
            ggml_backend_tensor_set_async(backend, g.h_fuse, cap.fz + (row0 + done)*n_embd, 0, m*rb);
        }
        if (!emb_in_graph) {
            // the embeddings were looked up after the committed rows: see cap_take
            ggml_backend_tensor_set_async(backend, g.emb_rows, cap.full + (cap.n_rows_max + row0 + done)*n_embd, 0, m*rb);
        }
        const int32_t li = launch_next;
        launch_next = (launch_next + 1) % n_launch_ring;
        stage_wait(li);
        uint8_t * st = stage_host + (size_t) li*stage_stride;
        write_blob(g, st, tok, rope, rpos, cval, empty, aeid, empty, empty, kvi, wpos, 0);
        if (cell_pos_dirty) {
            // the invalidated cells go up before these rows' own positions (which the graph writes)
            uint8_t * st_cp = st + off_cp;
            memcpy(st_cp, cell_pos_host.data(), (size_t) n_cells*sizeof(float));
            ggml_backend_tensor_set_async(backend, cell_pos, st_cp, 0, (size_t) n_cells*sizeof(float));
            cell_pos_dirty = false;
        }
        ggml_backend_tensor_set_async(backend, g.blob, st, 0, g.blob_size);
        ggml_backend_graph_compute_async(backend, g.gf);
        launch_done(li);
        done += m;
        n_bulk_rows += m;
    }
    if (cap.ev_done) {
        ggml_backend_event_record(cap.ev_done, backend);
        cap.pending = true;
    }
}

// the next capture buffer, large enough for n_rows rows: committed-depth rows, then embedding rows, then fused rows
llama_pipe_drafter::capture & llama_pipe_drafter::cap_take(int32_t & ic, size_t n_rows) {
    ic = cap_next;
    cap_next = (cap_next + 1) % n_cap_ring;
    capture & cap = caps[ic];
    if (cap.pending) {
        ggml_backend_event_synchronize(cap.ev_done);
        cap.pending = false;
    }
    if (cap.n_rows_max < n_rows) {
        if (cap.buf) {
            ggml_backend_buffer_free(cap.buf);
        }
        cap.n_rows_max = std::max<size_t>(n_rows, 512);
        cap.buf  = ggml_backend_buft_alloc_buffer(buft_host, cap.n_rows_max*3*n_embd*sizeof(float));
        cap.full = (float *) ggml_backend_buffer_get_base(cap.buf);
        cap.fz   = cap.full + 2*cap.n_rows_max*n_embd;
    }
    return cap;
}

void llama_pipe_drafter::on_ubatch(const llama_ubatch & ub, const llm_graph_result * res, ggml_backend_sched_t sched) {
    if (seq < 0 || ub.token == nullptr || ub.n_tokens == 0 || !graphs_valid) {
        return;
    }
    const int32_t n = (int32_t) ub.n_tokens;
    // the rows of the drafter's sequence (normally all of them)
    std::vector<int32_t> idx;
    for (int32_t i = 0; i < n; i++) {
        if (ub.seq_id[i][0] == seq) {
            idx.push_back(i);
        }
    }
    if (idx.empty()) {
        return;
    }
    // the committed depth of every position: the final state (from the all-positions branch, or the outputs when every
    // position is one), or a layer input; the first fused depth too for a fused head
    ggml_tensor * t_full = nullptr;
    if (committed == n_target) {
        t_full = res->t_embd_all;
        if (t_full == nullptr && res->get_embd() && res->get_embd()->ne[1] == n) {
            t_full = res->get_embd();
        }
    } else if (committed > 0) {
        t_full = res->get_layer_inp(committed);
    }
    const bool fuse_committed = fuse && committed == n_target;
    ggml_tensor * t_fz = fuse_committed ? res->get_layer_inp(fuse_depths[0]) : nullptr;
    if ((committed > 0 && (t_full == nullptr || t_full->ne[1] != n)) || (fuse_committed && (t_fz == nullptr || t_fz->ne[1] != n))) {
        return;
    }
    // computed in this graph (a tensor outside it has no backend and no data)
    if ((t_full && (!ggml_backend_sched_get_tensor_backend(sched, t_full) || !t_full->data)) ||
            (t_fz && (!ggml_backend_sched_get_tensor_backend(sched, t_fz) || !t_fz->data))) {
        LLAMA_LOG_WARN("%s: the micro-batch's hidden states are not in its graph: its positions stay uncommitted\n", __func__);
        return;
    }

    int32_t ic = 0;
    capture & cap = cap_take(ic, n);
    auto grab = [&](ggml_tensor * t, float * dst, ggml_backend_event_t & ev, ggml_backend_dev_t & ev_dev) {
        ggml_backend_t be = ggml_backend_sched_get_tensor_backend(sched, t);
        ggml_backend_tensor_get_async(be, t, dst, 0, (size_t) n*n_embd*sizeof(float));
        ggml_backend_dev_t dv = ggml_backend_get_device(be);
        if (ev == nullptr || ev_dev != dv) {
            if (ev) {
                ggml_backend_event_free(ev);
            }
            ev = ggml_backend_event_new(dv);
            ev_dev = dv;
        }
        if (ev) {
            ggml_backend_event_record(ev, be);
        } else {
            ggml_backend_synchronize(be);
        }
    };
    if (committed > 0) {
        grab(t_full, cap.full, cap.ev_full, cap.dev_full);
    }
    if (fuse_committed) {
        grab(t_fz, cap.fz, cap.ev_fz, cap.dev_fz);
    }
    if (!emb_in_graph) {
        for (int32_t i = 0; i < n; i++) {
            lookup_emb(ub.token[i], cap.full + (cap.n_rows_max + i)*n_embd);
        }
    }

    // runs of consecutive rows of the sequence
    std::vector<llama_token> toks;
    std::vector<llama_pos>   ps;
    size_t k = 0;
    while (k < idx.size()) {
        size_t k1 = k + 1;
        while (k1 < idx.size() && idx[k1] == idx[k1 - 1] + 1) {
            k1++;
        }
        toks.clear();
        ps.clear();
        for (size_t j = k; j < k1; j++) {
            toks.push_back(ub.token[idx[j]]);
            ps.push_back(ub.pos[idx[j]]);
        }
        commit_rows_bulk(toks.data(), ps.data(), (int32_t) toks.size(), ic, (size_t) idx[k]);
        k = k1;
    }
}

void llama_pipe_drafter::test_commit_bulk(const llama_token * toks, const llama_pos * pos, int32_t n, const float * h_full,
                                          const float * h_fuse) {
    int32_t ic = 0;
    capture & cap = cap_take(ic, n);
    memcpy(cap.full, h_full, (size_t) n*n_embd*sizeof(float));
    if (fuse && h_fuse) {
        memcpy(cap.fz, h_fuse, (size_t) n*n_embd*sizeof(float));
    }
    if (!emb_in_graph) {
        for (int32_t i = 0; i < n; i++) {
            lookup_emb(toks[i], cap.full + (cap.n_rows_max + i)*n_embd);
        }
    }
    if (cap.ev_full) {
        ggml_backend_event_free(cap.ev_full);
        cap.ev_full = nullptr;
    }
    if (cap.ev_fz) {
        ggml_backend_event_free(cap.ev_fz);
        cap.ev_fz = nullptr;
    }
    commit_rows_bulk(toks, pos, n, ic, 0);
}
