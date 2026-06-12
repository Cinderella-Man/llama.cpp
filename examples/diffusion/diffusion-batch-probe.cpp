// Layer D0 probe (04_layer_d.md step 1): multi-sequence batching on the SQUARE
// (no-cache) diffusion path. Answers, on real hardware:
//   1. IDENTITY  - K copies of one canvas batched under distinct seq_ids: do the
//                  per-row logits match a single-canvas forward (and does argmax)?
//   2. MIXED     - two different-length canvases batched vs run separately.
//   3. TIMING    - ms per forward for K in {1,2,3,4,8} copies vs K sequential
//                  forwards (the fixed-cost-floor amortization GO/NO-GO).
//   4. COMMITS   - commits-per-step histogram of a threshold decode (the D1/D2
//                  speculation headroom: how many steps commit only 1-3 tokens).
//
//   llama-diffusion-batch-probe -m <model.gguf> -ngl 99 [-ub 1024]

#include "diffusion.h"
#include "llama.h"
#include "common.h"
#include "arg.h"
#include "log.h"
#include "chat.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

static std::vector<llama_token> make_canvas(const llama_vocab * vocab, const char * prompt, int n_masks) {
    std::vector<llama_token> toks(512);
    int n = llama_tokenize(vocab, prompt, (int32_t) strlen(prompt), toks.data(), (int32_t) toks.size(), true, true);
    toks.resize(std::max(n, 0));
    const llama_token mask = llama_vocab_mask(vocab);
    for (int i = 0; i < n_masks; i++) {
        toks.push_back(mask);
    }
    return toks;
}

// one forward of `seqs` canvases in a single batch; returns per-row logits copy
static std::vector<float> forward_batched(llama_context * ctx, const std::vector<std::vector<llama_token>> & seqs,
                                          int n_vocab, double * ms_out) {
    int n_total = 0;
    for (const auto & s : seqs) {
        n_total += (int) s.size();
    }
    llama_batch batch = llama_batch_init(n_total, 0, 1);
    batch.n_tokens    = n_total;
    int r = 0;
    for (size_t si = 0; si < seqs.size(); si++) {
        for (size_t i = 0; i < seqs[si].size(); i++, r++) {
            batch.token[r]     = seqs[si][i];
            batch.pos[r]       = (llama_pos) i;
            batch.n_seq_id[r]  = 1;
            batch.seq_id[r][0] = (llama_seq_id) si;
            batch.logits[r]    = 1;
        }
    }
    const int64_t t0 = ggml_time_us();
    const int ret = llama_decode(ctx, batch);
    llama_synchronize(ctx);  // llama_decode is async; sync before stopping the clock
    const int64_t t1 = ggml_time_us();
    if (ms_out) {
        *ms_out = (t1 - t0) / 1000.0;
    }
    std::vector<float> out;
    if (ret == 0) {
        // _ith access respects the output-order mapping (multi-seq ubatch splitting
        // may permute rows relative to submission order; the raw buffer is unmapped)
        out.resize((size_t) n_total * n_vocab);
        for (int i = 0; i < n_total; i++) {
            const float * lg = llama_get_logits_ith(ctx, i);
            std::memcpy(out.data() + (size_t) i * n_vocab, lg, (size_t) n_vocab * sizeof(float));
        }
    } else {
        LOG_ERR("decode failed: %d\n", ret);
    }
    llama_batch_free(batch);
    return out;
}

struct commit_stats {
    int prev_masks = -1;
    std::map<int, int> hist; // commits-per-step -> count
};

static bool count_commits_cb(int32_t step, int32_t total, const llama_token * toks, int32_t n, void * ud) {
    (void) step; (void) total;
    auto * st = (commit_stats *) ud;
    // mask id passed via hist[-1] hack-free: recompute from sentinel stored in prev_masks at init
    extern llama_token g_probe_mask_id;
    int masks = 0;
    for (int32_t i = 0; i < n; i++) {
        if (toks[i] == g_probe_mask_id) {
            masks++;
        }
    }
    if (st->prev_masks >= 0 && st->prev_masks > masks) {
        st->hist[st->prev_masks - masks]++;
    }
    st->prev_masks = masks;
    return true;
}

llama_token g_probe_mask_id = 0;

int main(int argc, char ** argv) {
    common_params params;
    params.sampling.backend_sampling = true;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_DIFFUSION)) {
        return 1;
    }
    common_init();
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = params.n_gpu_layers;
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mp);
    if (!model) { return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    g_probe_mask_id = llama_vocab_mask(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = params.n_ctx > 0 ? params.n_ctx : 4096;
    cp.n_batch = cp.n_ubatch = params.n_ubatch > 0 ? params.n_ubatch : 1024;
    cp.n_seq_max = 8;
    cp.flash_attn_type = params.flash_attn_type;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { LOG_ERR("context creation failed (n_seq_max=8)\n"); return 1; }
    llama_set_causal_attn(ctx, false);

    auto canvas_a = make_canvas(vocab, "Write a short poem about rivers.", 96);
    auto canvas_b = make_canvas(vocab, "Explain what a stack data structure is.", 64);
    const int La = (int) canvas_a.size(), Lb = (int) canvas_b.size();
    LOG_INF("canvas A: %d tok, canvas B: %d tok\n", La, Lb);

    // ---- 1. IDENTITY: A alone vs A+A batched ------------------------------------
    double ms;
    auto ref  = forward_batched(ctx, { canvas_a }, n_vocab, &ms);
    auto dup  = forward_batched(ctx, { canvas_a, canvas_a }, n_vocab, &ms);
    if (ref.empty() || dup.empty()) { LOG_ERR("PROBE 1 decode failed\n"); return 1; }
    double md0 = 0, md1 = 0;
    int agree0 = 0, agree1 = 0;
    for (int i = 0; i < La; i++) {
        int am_r = 0, am_0 = 0, am_1 = 0;
        for (int v = 0; v < n_vocab; v++) {
            const float r0 = ref[(size_t) i * n_vocab + v];
            const float c0 = dup[(size_t) i * n_vocab + v];
            const float c1 = dup[(size_t) (La + i) * n_vocab + v];
            md0 = std::max(md0, (double) std::fabs(c0 - r0));
            md1 = std::max(md1, (double) std::fabs(c1 - r0));
            if (c0 > dup[(size_t) i * n_vocab + am_0]) am_0 = v;
            if (c1 > dup[(size_t) (La + i) * n_vocab + am_1]) am_1 = v;
            if (r0 > ref[(size_t) i * n_vocab + am_r]) am_r = v;
        }
        agree0 += am_0 == am_r;
        agree1 += am_1 == am_r;
    }
    double md01 = 0;
    int agree01 = 0;
    for (int i = 0; i < La; i++) {
        int a0 = 0, a1 = 0;
        for (int v = 0; v < n_vocab; v++) {
            const float c0 = dup[(size_t) i * n_vocab + v];
            const float c1 = dup[(size_t) (La + i) * n_vocab + v];
            md01 = std::max(md01, (double) std::fabs(c1 - c0));
            if (c0 > dup[(size_t) i * n_vocab + a0]) a0 = v;
            if (c1 > dup[(size_t) (La + i) * n_vocab + a1]) a1 = v;
        }
        agree01 += a0 == a1;
    }
    LOG_INF("PROBE1 identity A|A vs A: max|dlogit| seq0=%.3e seq1=%.3e seq0-vs-seq1=%.3e, "
            "argmax agree vs ref %d/%d, %d/%d; seq0 vs seq1 %d/%d\n",
            md0, md1, md01, agree0, La, agree1, La, agree01, La);

    // per-row divergence profile (seq0 vs ref): where does it start?
    for (int i = 0; i < La; i += 8) {
        double d = 0;
        for (int v = 0; v < n_vocab; v++) {
            d = std::max(d, (double) std::fabs(dup[(size_t) i * n_vocab + v] - ref[(size_t) i * n_vocab + v]));
        }
        LOG_INF("PROBE1b row %3d: max|dlogit| %.3e\n", i, d);
    }

    // control: both copies under the SAME seq id (deliberate all-to-all contamination)
    {
        llama_batch b2 = llama_batch_init(2 * La, 0, 1);
        b2.n_tokens = 2 * La;
        for (int r2 = 0; r2 < 2 * La; r2++) {
            b2.token[r2] = canvas_a[r2 % La];
            b2.pos[r2] = r2 % La;
            b2.n_seq_id[r2] = 1;
            b2.seq_id[r2][0] = 0;  // SAME seq for both copies
            b2.logits[r2] = 1;
        }
        if (llama_decode(ctx, b2) == 0) {
            double mds = 0;
            for (int i = 0; i < La; i++) {
                const float * l0 = llama_get_logits_ith(ctx, i);
                for (int v = 0; v < n_vocab; v++) {
                    mds = std::max(mds, (double) std::fabs(l0[v] - ref[(size_t) i * n_vocab + v]));
                }
            }
            LOG_INF("PROBE1c control same-seq A|A vs A: seq0 max|dlogit| %.3e (expect LARGE if mask works)\n", mds);
        }
        llama_batch_free(b2);
    }

    // ---- 1d. CONTAMINATION DIRECT TEST: seq0 fixed, seq1 content swapped ----------
    {
        std::vector<llama_token> noise = canvas_a;
        for (size_t i = 0; i < noise.size(); i++) {
            noise[i] = (llama_token) (1000 + (i * 37) % 5000);  // arbitrary non-mask junk
        }
        auto run1 = forward_batched(ctx, { canvas_a, canvas_a }, n_vocab, &ms);
        auto run2 = forward_batched(ctx, { canvas_a, noise   }, n_vocab, &ms);
        double d = 0;
        int flips = 0;
        for (int i = 0; i < La; i++) {
            int a1 = 0, a2 = 0;
            for (int v = 0; v < n_vocab; v++) {
                const float x1 = run1[(size_t) i * n_vocab + v];
                const float x2 = run2[(size_t) i * n_vocab + v];
                d = std::max(d, (double) std::fabs(x1 - x2));
                if (x1 > run1[(size_t) i * n_vocab + a1]) a1 = v;
                if (x2 > run2[(size_t) i * n_vocab + a2]) a2 = v;
            }
            flips += a1 != a2;
        }
        LOG_INF("PROBE1d seq0 logits, neighbor A vs noise: max|dlogit| %.3e, argmax flips %d/%d "
                "(>0 => cross-seq contamination PROVEN)\n", d, flips, La);
    }

    // ---- 1e. WHERE do the batch-shape argmax flips sit? (Finding 2 follow-up) ----
    {
        int flips_hi = 0, flips_lo = 0;
        for (int i = 0; i < La; i++) {
            int am_r = 0, am_0 = 0;
            for (int v = 0; v < n_vocab; v++) {
                if (ref[(size_t) i * n_vocab + v] > ref[(size_t) i * n_vocab + am_r]) am_r = v;
                if (dup[(size_t) i * n_vocab + v] > dup[(size_t) i * n_vocab + am_0]) am_0 = v;
            }
            if (am_0 != am_r) {
                // softmax prob of the ref argmax (confidence of the flipped decision)
                double mx = ref[(size_t) i * n_vocab + am_r], Z = 0;
                for (int v = 0; v < n_vocab; v++) Z += std::exp((double) ref[(size_t) i * n_vocab + v] - mx);
                const double p = 1.0 / Z;
                LOG_INF("PROBE1e flip at row %d: ref-argmax prob %.3f\n", i, p);
                (p >= 0.5 ? flips_hi : flips_lo)++;
            }
        }
        LOG_INF("PROBE1e flips: %d at conf>=0.5 (commit-relevant), %d below\n", flips_hi, flips_lo);
    }

    // ---- 2. MIXED lengths: A+B batched vs each alone -----------------------------
    auto refb = forward_batched(ctx, { canvas_b }, n_vocab, &ms);
    auto mix  = forward_batched(ctx, { canvas_a, canvas_b }, n_vocab, &ms);
    double mda = 0, mdb = 0;
    for (int i = 0; i < La; i++)
        for (int v = 0; v < n_vocab; v++)
            mda = std::max(mda, (double) std::fabs(mix[(size_t) i * n_vocab + v] - ref[(size_t) i * n_vocab + v]));
    for (int i = 0; i < Lb; i++)
        for (int v = 0; v < n_vocab; v++)
            mdb = std::max(mdb, (double) std::fabs(mix[(size_t) (La + i) * n_vocab + v] - refb[(size_t) i * n_vocab + v]));
    LOG_INF("PROBE2 mixed A+B vs separate: max|dlogit| A=%.3e B=%.3e\n", mda, mdb);

    // ---- 3. TIMING: K copies batched vs K sequential ------------------------------
    LOG_INF("PROBE3 timing (canvas %d tok), 3 warmup + 16 timed:\n", La);
    for (int K : { 1, 2, 3, 4, 8 }) {
        if (K * La > (int) cp.n_ubatch) { LOG_INF("  K=%d skipped (exceeds ub %d)\n", K, cp.n_ubatch); continue; }
        std::vector<std::vector<llama_token>> copies(K, canvas_a);
        double acc = 0;
        for (int it = 0; it < 19; it++) {
            double m = 0;
            forward_batched(ctx, copies, n_vocab, &m);
            if (it >= 3) acc += m;
        }
        const double batched = acc / 16.0;
        // sequential: K single forwards
        acc = 0;
        for (int it = 0; it < 19; it++) {
            double tot = 0;
            for (int k = 0; k < K; k++) { double m = 0; forward_batched(ctx, { canvas_a }, n_vocab, &m); tot += m; }
            if (it >= 3) acc += tot;
        }
        const double sequential = acc / 16.0;
        LOG_INF("  K=%d: batched %7.2f ms vs sequential %7.2f ms -> %.2fx\n",
                K, batched, sequential, sequential / batched);
    }

    // 32-row regime (D1/D2 block shape proxy): tiny canvas
    auto tiny = make_canvas(vocab, "x", 24); // ~32 rows total
    LOG_INF("PROBE3b timing (tiny canvas %d tok):\n", (int) tiny.size());
    for (int K : { 1, 2, 4, 8 }) {
        std::vector<std::vector<llama_token>> copies(K, tiny);
        double acc = 0;
        for (int it = 0; it < 19; it++) { double m = 0; forward_batched(ctx, copies, n_vocab, &m); if (it >= 3) acc += m; }
        const double batched = acc / 16.0;
        acc = 0;
        for (int it = 0; it < 19; it++) {
            double tot = 0;
            for (int k = 0; k < K; k++) { double m = 0; forward_batched(ctx, { tiny }, n_vocab, &m); tot += m; }
            if (it >= 3) acc += tot;
        }
        const double sequential = acc / 16.0;
        LOG_INF("  K=%d: batched %7.2f ms vs sequential %7.2f ms -> %.2fx\n", K, batched, sequential, sequential / batched);
    }

    // ---- 4. commits-per-step histogram of a real threshold decode -----------------
    LOG_INF("PROBE4 commits-per-step (threshold 0.6, 3 prompts x seed 3):\n");
    const char * prompts[] = {
        "Write an Elixir GenServer module implementing a counter with increment, decrement and get operations. Reply with only the code.",
        "Write Elixir code: a Stack module with push/2, pop/1, peek/1, size/1, to_list/1, reverse/1 and map/2. Reply with only the code.",
        "Write a haiku about rivers.",
    };
    common_chat_templates_ptr tmpls = common_chat_templates_init(model, "");
    commit_stats agg;
    for (const char * p : prompts) {
        // chat-template like the server: raw prompts EOT-bail on Dream (Layer C finding)
        common_chat_templates_inputs ci;
        common_chat_msg msg;
        msg.role    = "user";
        msg.content = p;
        ci.messages.push_back(msg);
        ci.add_generation_prompt = true;
        const std::string formatted = common_chat_templates_apply(tmpls.get(), ci).prompt;
        std::vector<llama_token> in(512);
        int n_in = llama_tokenize(vocab, formatted.c_str(), (int32_t) formatted.size(),
                                  in.data(), (int32_t) in.size(), true, true);
        in.resize(std::max(n_in, 0));
        std::vector<llama_token> out(512);
        diffusion_params dp;
        dp.mask_token_id  = g_probe_mask_id;
        dp.shift_logits   = true;
        dp.seed           = 3;
        dp.temperature    = 0.2f;
        dp.steps          = 128;
        dp.top_k          = 40;
        dp.top_p          = 0.95f;  // diffusion.h default 0 would add a top-p(0) sampler:
                                    // only top-1 survives -> conf 1.0 -> one-step flood
        dp.eps            = 1e-3f;
        dp.conf_threshold = 0.6f;
        dp.backend_sampling = true;
        dp.max_length     = std::min<int32_t>((int32_t) in.size() + 192, 512);
        commit_stats st;
        dp.step_callback = count_commits_cb;
        dp.step_callback_user_data = &st;
        int32_t n_gen = 0;
        diffusion_generate(ctx, in.data(), out.data(), (int32_t) in.size(), dp, n_gen);
        // show what was decoded (degenerate-flood diagnosis)
        std::string txt;
        char piece[64];
        for (int32_t i = (int32_t) in.size(); i < dp.max_length && (int) txt.size() < 80; i++) {
            const int np = llama_token_to_piece(vocab, out[i], piece, sizeof(piece) - 1, 0, false);
            if (np > 0) txt.append(piece, np);
        }
        LOG_INF("  decoded [%.*s...]\n", 70, txt.c_str());
        for (auto & kv : st.hist) agg.hist[kv.first] += kv.second;
    }
    int steps_total = 0, commits_total = 0, stall_steps = 0;
    for (auto & kv : agg.hist) {
        steps_total += kv.second;
        commits_total += kv.first * kv.second;
        if (kv.first <= 3) stall_steps += kv.second;
        LOG_INF("  %3d commits: %d steps\n", kv.first, kv.second);
    }
    LOG_INF("PROBE4 total: %d steps, %d commits, avg %.2f/step, stall(<=3) steps %d (%.0f%%)\n",
            steps_total, commits_total, steps_total ? (double) commits_total / steps_total : 0,
            stall_steps, steps_total ? 100.0 * stall_steps / steps_total : 0);

    // ---- 5. KV-BLOCK row-scaling: W-row decode vs the shared frozen store ---------
    // The speculative verify batch (K copies of a 32-row block) has the same row
    // count and store geometry as a single (K*32)-row block decode - mask content
    // does not change the timing. GO/NO-GO: K=4 (128 rows) <= 2x the 32-row step?
    {
        const int Lc = 256;
        auto canvas = make_canvas(vocab, "Write a long detailed essay about rivers and their role in history.", Lc);
        canvas.resize(Lc);
        llama_model * m = const_cast<llama_model *>(llama_get_model(ctx));
        // WARM at full canvas (dual mode geometry): writes the store
        llama_diffusion_set_phase(m, /*WARM*/1, Lc);
        double ms_warm = 0;
        forward_batched(ctx, { canvas }, n_vocab, &ms_warm);
        LOG_INF("PROBE5 kv row-scaling (store L=%d, warm %.2f ms):\n", Lc, ms_warm);
        const int s = 64;
        double base32 = 0;
        for (int W : { 32, 64, 96, 128 }) {
            std::vector<llama_token> blk(canvas.begin() + s, canvas.begin() + s + W);
            double acc = 0;
            for (int it = 0; it < 19; it++) {
                llama_diffusion_set_block(m, s, Lc);
                llama_diffusion_set_phase(m, /*BLOCK*/3, 0);
                llama_batch b = llama_batch_init(W, 0, 1);
                b.n_tokens = W;
                for (int i = 0; i < W; i++) {
                    b.token[i] = blk[i]; b.pos[i] = s + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
                }
                const int64_t t0 = ggml_time_us();
                llama_decode(ctx, b);
                llama_synchronize(ctx);
                const int64_t t1 = ggml_time_us();
                llama_batch_free(b);
                if (it >= 3) acc += (t1 - t0) / 1000.0;
            }
            const double t = acc / 16.0;
            if (W == 32) base32 = t;
            LOG_INF("  W=%3d rows vs store: %7.2f ms/step (%.2fx of W=32)\n", W, t,
                    base32 > 0 ? t / base32 : 0.0);
        }
        llama_diffusion_set_phase(m, /*UNIFIED*/0, 0);
    }

    // ---- 6. ORACLE chain-acceptance: would step t's draft survive step t+1? -------
    // Mini greedy threshold decode (argmax + raw-prob conf; approximates the engine
    // at temp 0.2 after de-temper). At each step record argmax for EVERY masked
    // position; when a position commits at step t+Delta, check agreement with the
    // argmax recorded at earlier steps (Delta = 1, 2, 3): the SSD chain hit-rate.
    {
        common_chat_templates_inputs ci;
        common_chat_msg msg;
        msg.role = "user";
        msg.content = "Write Elixir code for the following task. Reply with ONLY a single "
                      "```elixir code block, no explanation.\n\nTask: a GenServer module "
                      "implementing a counter with increment, decrement and get operations.";
        ci.messages.push_back(msg);
        ci.add_generation_prompt = true;
        common_chat_templates_ptr tm2 = common_chat_templates_init(model, "");
        const std::string fp = common_chat_templates_apply(tm2.get(), ci).prompt;
        std::vector<llama_token> canvas(512);
        int n_in = llama_tokenize(vocab, fp.c_str(), (int32_t) fp.size(), canvas.data(), (int32_t) canvas.size(), true, true);
        canvas.resize(n_in);
        const int n_gen = 160;
        for (int i = 0; i < n_gen; i++) canvas.push_back(g_probe_mask_id);
        const int L = (int) canvas.size();

        std::vector<std::vector<llama_token>> pred_hist;  // per step: argmax per position (or -1)
        int hits1 = 0, tot1 = 0, hits2 = 0, tot2 = 0, hits3 = 0, tot3 = 0;
        for (int step = 0; step < 64; step++) {
            auto lg = forward_batched(ctx, { canvas }, n_vocab, nullptr);
            if (lg.empty()) break;
            std::vector<llama_token> pred(L, -1);
            std::vector<std::pair<double, int>> conf;  // (prob, pos)
            for (int i = n_in; i < L; i++) {
                if (canvas[i] != g_probe_mask_id) continue;
                // shift_logits: row i-1 predicts pos i (Dream)
                const float * row = lg.data() + (size_t) (i - 1) * n_vocab;
                int am = 0;
                for (int v = 1; v < n_vocab; v++) if (row[v] > row[am]) am = v;
                double mx = row[am], Z = 0;
                for (int v = 0; v < n_vocab; v++) Z += std::exp((double) row[v] - mx);
                pred[i] = am;
                conf.emplace_back(1.0 / Z, i);
            }
            if (conf.empty()) break;
            // commit rule: all above 0.6, else best one
            int committed = 0;
            for (auto & c : conf) {
                if (c.first >= 0.6) {
                    const int pos = c.second;
                    // score the chain draft: did earlier-step predictions foresee this?
                    for (int d = 1; d <= 3; d++) {
                        if ((int) pred_hist.size() >= d) {
                            const llama_token old = pred_hist[pred_hist.size() - d][pos];
                            if (old >= 0) {
                                (d == 1 ? tot1 : d == 2 ? tot2 : tot3)++;
                                if (old == pred[pos]) (d == 1 ? hits1 : d == 2 ? hits2 : hits3)++;
                            }
                        }
                    }
                    canvas[pos] = pred[pos];
                    committed++;
                }
            }
            if (committed == 0) {
                auto best = std::max_element(conf.begin(), conf.end());
                canvas[best->second] = pred[best->second];
            }
            pred_hist.push_back(std::move(pred));
        }
        LOG_INF("PROBE6 oracle chain acceptance (counter task, greedy/0.6): "
                "Delta1 %d/%d (%.0f%%), Delta2 %d/%d (%.0f%%), Delta3 %d/%d (%.0f%%)\n",
                hits1, tot1, tot1 ? 100.0 * hits1 / tot1 : 0,
                hits2, tot2, tot2 ? 100.0 * hits2 / tot2 : 0,
                hits3, tot3, tot3 ? 100.0 * hits3 / tot3 : 0);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
