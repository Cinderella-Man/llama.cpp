// Pipelined speculation for one sequence on a layer split.
//
// With the layers split over several devices, plain decoding keeps one device busy at a time. Here a guessed token
// enters the first device while the later devices still work on the current token: up to D guesses are in flight,
// each an ordinary one-token evaluation (llama_pipe_*). When the oldest token's logits arrive, the next token is
// sampled once, in order, exactly as plain decoding samples it, and compared with the guess that is already in flight
// behind it (sample-and-match):
//   - same token: the guess's evaluation is the right one and keeps going;
//   - different token: every guess behind is dropped, its positions are removed from the KV cache, and the sampled
//     token is submitted instead.
// The output is therefore the output of plain decoding with the same sampler and seed.
//
// One run ("job") is set by the --pipe-* flags; --pipe-jobs FILE runs a list of jobs with one model load, one per line
// as key=value pairs that override the flags (keys: name mode guess depth copy_min miss oracle prompt n spec probe chat
// out state_out trace; oracle=@<name> replays the tokens of an earlier job in the list). Each job prints one PIPESPEC line.
//
//   --pipe-mode plain|pipe  plain decoding with llama_decode, one token per call (the reference), or pipelined decoding
//   --pipe-depth D          guesses in flight (0 = the pipe API without guesses)
//   --pipe-guess G          none
//                           copy: the token after the longest earlier match of the history's suffix, if that match is
//                             at least --pipe-copy-min tokens long (history includes the guesses in flight)
//                           ngram-simple: llama.cpp's ngram-simple lookup on the same history (--spec-ngram-simple-size-n/-m)
//                           spec: any llama.cpp drafter, --pipe-spec TYPE (e.g. ngram-map-k), drafted from committed
//                             tokens only and consumed one guess at a time
//                           oracle: the tokens of a reference run (--pipe-oracle FILE from --pipe-out), with a share
//                             --pipe-miss of the guesses made wrong on purpose (deterministic per position)
//   --pipe-prompt FILE      prompt file (default: -p / -f)
//   --pipe-chat             wrap the prompt as one user message with the model's chat template
//   --pipe-out FILE         one line per generated token: index, token id, FNV-1a hash of the logits it was sampled from
//   --pipe-probe N          after generation, decode N more tokens with llama_decode; their logits hash checks every KV
//                           cell that is still visible
//   --pipe-state-out FILE   the sequence's KV state (llama_state_seq_get_data) after generation
//   --pipe-trace FILE       one line per sampled token: microseconds since the first submit, guesses in flight, outcome
//   --pipe-backend-greedy   greedy sampling on the GPU (llama_set_sampler) in both modes: only the token id comes back
//                           instead of the logits; the logits hashes then hash the token ids

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ngram-map.h"
#include "speculative.h"

#include <algorithm>
#include <clocale>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// greedy sampling on the device that computes the logits (--pipe-backend-greedy)
static bool g_backend_greedy = false;

struct pipe_job {
    std::string name     = "run";
    std::string mode     = "pipe";
    std::string guess    = "none";
    int         depth    = 0;
    int         copy_min = 4;
    double      miss     = 0.0;
    std::string oracle;            // file written by --pipe-out, or @<job name>
    std::string prompt;            // file; empty = -p / -f
    int         n_predict = 256;
    std::string spec     = "ngram-map-k";
    int         probe    = 0;
    bool        chat     = false;
    std::string out;
    std::string state_out;
    std::string trace;
};

static bool job_set(pipe_job & j, const std::string & k, const std::string & v) {
    if      (k == "name")      { j.name      = v; }
    else if (k == "mode")      { j.mode      = v; }
    else if (k == "guess")     { j.guess     = v; }
    else if (k == "depth")     { j.depth     = atoi(v.c_str()); }
    else if (k == "copy_min")  { j.copy_min  = atoi(v.c_str()); }
    else if (k == "miss")      { j.miss      = atof(v.c_str()); }
    else if (k == "oracle")    { j.oracle    = v; }
    else if (k == "prompt")    { j.prompt    = v; }
    else if (k == "n")         { j.n_predict = atoi(v.c_str()); }
    else if (k == "spec")      { j.spec      = v; }
    else if (k == "probe")     { j.probe     = atoi(v.c_str()); }
    else if (k == "chat")      { j.chat      = v != "0"; }
    else if (k == "out")       { j.out       = v; }
    else if (k == "state_out") { j.state_out = v; }
    else if (k == "trace")     { j.trace     = v; }
    else {
        return false;
    }
    return true;
}

// removes the --pipe-* arguments so that the common parser sees only its own
static bool parse_pipe_args(int & argc, char ** argv, pipe_job & def, std::string & jobs_file, int & n_slots) {
    static const std::map<std::string, std::string> keys = {
        {"--pipe-name", "name"}, {"--pipe-mode", "mode"}, {"--pipe-guess", "guess"}, {"--pipe-depth", "depth"},
        {"--pipe-copy-min", "copy_min"}, {"--pipe-miss", "miss"}, {"--pipe-oracle", "oracle"},
        {"--pipe-prompt", "prompt"}, {"--pipe-spec", "spec"}, {"--pipe-probe", "probe"}, {"--pipe-out", "out"},
        {"--pipe-state-out", "state_out"}, {"--pipe-trace", "trace"},
    };
    int j = 1;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a.rfind("--pipe-", 0) != 0) {
            argv[j++] = argv[i];
            continue;
        }
        if (a == "--pipe-chat") {
            def.chat = true;
            continue;
        }
        if (a == "--pipe-backend-greedy") {
            g_backend_greedy = true;
            continue;
        }
        if (i + 1 >= argc) {
            fprintf(stderr, "missing value for %s\n", a.c_str());
            return false;
        }
        const std::string v = argv[++i];
        if (a == "--pipe-jobs") {
            jobs_file = v;
        } else if (a == "--pipe-slots") {
            n_slots = atoi(v.c_str());
        } else if (keys.count(a)) {
            job_set(def, keys.at(a), v);
        } else {
            fprintf(stderr, "unknown argument %s\n", a.c_str());
            return false;
        }
    }
    argc = j;
    return true;
}

static uint64_t fnv1a(const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *) data;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// deterministic per-position coin for the oracle's injected misses
static double unit_hash(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x =  x ^ (x >> 31);
    return (x >> 11) * (1.0 / 9007199254740992.0);
}

// the sampler chain used by both modes, applied to a raw logits row
struct row_sampler {
    llama_sampler * chain = nullptr;
    std::vector<llama_token_data> cur;

    row_sampler(const common_params_sampling & sp, int n_vocab) : cur(n_vocab) {
        chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
        if (sp.temp <= 0.0f) {
            llama_sampler_chain_add(chain, llama_sampler_init_greedy());
        } else {
            llama_sampler_chain_add(chain, llama_sampler_init_top_k(sp.top_k));
            llama_sampler_chain_add(chain, llama_sampler_init_top_p(sp.top_p, 1));
            llama_sampler_chain_add(chain, llama_sampler_init_min_p(sp.min_p, 1));
            llama_sampler_chain_add(chain, llama_sampler_init_temp(sp.temp));
            llama_sampler_chain_add(chain, llama_sampler_init_dist(sp.seed));
        }
    }

    ~row_sampler() { llama_sampler_free(chain); }

    llama_token sample(const float * logits) {
        for (size_t i = 0; i < cur.size(); i++) {
            cur[i] = { (llama_token) i, logits[i], 0.0f };
        }
        llama_token_data_array arr = { cur.data(), cur.size(), -1, false };
        llama_sampler_apply(chain, &arr);
        GGML_ASSERT(arr.selected >= 0 && arr.selected < (int64_t) arr.size);
        const llama_token id = arr.data[arr.selected].id;
        llama_sampler_accept(chain, id);
        return id;
    }
};

// the token after the longest earlier occurrence of the history's suffix (the most recent one on ties), or -1 if
// that occurrence is shorter than min_match
static llama_token copy_guess(const std::vector<llama_token> & h, int min_match) {
    const int n = (int) h.size();
    if (n < 2) {
        return -1;
    }
    const int max_len = 64;
    int best_len = 0;
    int best_end = -1;
    for (int i = n - 2; i >= 0; --i) {
        if (h[i] != h[n - 1]) {
            continue;
        }
        int len = 1;
        while (len < max_len && i - len >= 0 && h[i - len] == h[n - 1 - len]) {
            len++;
        }
        if (len > best_len) {
            best_len = len;
            best_end = i;
            if (len == max_len) {
                break;
            }
        }
    }
    if (best_len < min_match) {
        return -1;
    }
    return h[best_end + 1];
}

static std::vector<llama_token> read_tokens(const std::string & path) {
    // the --pipe-out format: "<index> <token> <hash>" per line
    std::vector<llama_token> res;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        long idx; long tok;
        if (ss >> idx >> tok) {
            res.push_back((llama_token) tok);
        }
    }
    return res;
}

static std::string read_file(const std::string & path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

struct job_result {
    std::vector<llama_token> gen;    // gen[j] sits at position n_prompt + j
    std::vector<uint64_t>    hashes; // logits hash of each sampled token
};

static bool run_job(const pipe_job & job, common_params & params, llama_model * model, llama_context * ctx,
                    const std::map<std::string, job_result> & done_jobs, job_result & res) {
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);
    llama_memory_t      mem     = llama_get_memory(ctx);
    const llama_seq_id  seq     = 0;

    llama_memory_clear(mem, false);

    std::string prompt = job.prompt.empty() ? params.prompt : read_file(job.prompt);
    if (job.chat) {
        auto tmpls = common_chat_templates_init(model, params.chat_template);
        common_chat_templates_inputs inputs;
        common_chat_msg msg;
        msg.role    = "user";
        msg.content = prompt;
        inputs.messages = { msg };
        inputs.add_generation_prompt = true;
        inputs.use_jinja = true;
        prompt = common_chat_templates_apply(tmpls.get(), inputs).prompt.str();
    }

    const std::vector<llama_token> inp = common_tokenize(ctx, prompt, true, true);
    const int n_prompt  = (int) inp.size();
    const int n_predict = job.n_predict;

    if (n_prompt + n_predict + job.depth + 8 + job.probe > (int) llama_n_ctx(ctx)) {
        LOG_ERR("%s: %s: prompt (%d) + n_predict (%d) do not fit the context (%d)\n", __func__, job.name.c_str(),
                n_prompt, n_predict, llama_n_ctx(ctx));
        return false;
    }

    // prompt: identical in both modes
    {
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        for (int i = 0; i < n_prompt; i++) {
            batch.token[i]     = inp[i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = seq;
            batch.logits[i]    = i == n_prompt - 1;
        }
        batch.n_tokens = n_prompt;
        for (int i0 = 0; i0 < n_prompt; i0 += llama_n_batch(ctx)) {
            const int n = std::min((int) llama_n_batch(ctx), n_prompt - i0);
            llama_batch view = { n, batch.token + i0, nullptr, batch.pos + i0, batch.n_seq_id + i0, batch.seq_id + i0, batch.logits + i0 };
            if (llama_decode(ctx, view) != 0) {
                LOG_ERR("%s: prompt decode failed\n", __func__);
                llama_batch_free(batch);
                return false;
            }
        }
        llama_batch_free(batch);
    }

    row_sampler smpl(params.sampling, n_vocab);

    auto & gen    = res.gen;
    auto & hashes = res.hashes;
    gen.clear();
    hashes.clear();

    auto take = [&](const float * logits) {
        const llama_token id = smpl.sample(logits);
        gen.push_back(id);
        hashes.push_back(fnv1a(logits, sizeof(float)*n_vocab));
        return id;
    };
    auto take_id = [&](llama_token id) {
        gen.push_back(id);
        hashes.push_back(fnv1a(&id, sizeof(id)));
        return id;
    };

    if (g_backend_greedy) {
        take_id(llama_get_sampled_token_ith(ctx, -1));
    } else {
        take(llama_get_logits_ith(ctx, -1));
    }

    // stats
    int n_submit = 0, n_guess = 0, n_hit = 0, n_miss = 0, n_wasted = 0;
    double depth_sum = 0.0;
    int    depth_n   = 0;

    struct trace_row { int64_t t_us; int inflight; char outcome; };
    std::vector<trace_row> trace;

    const int64_t t_gen_start = ggml_time_us();

    auto done = [&]() {
        return (int) gen.size() >= n_predict || llama_vocab_is_eog(vocab, gen.back());
    };

    if (job.mode == "plain") {
        while (!done()) {
            llama_token id  = gen.back();
            llama_pos   pos = n_prompt + (int) gen.size() - 1;
            llama_batch b = llama_batch_get_one(&id, 1);
            b.pos = &pos;
            if (llama_decode(ctx, b) != 0) {
                LOG_ERR("%s: decode failed at pos %d\n", __func__, pos);
                return false;
            }
            n_submit++;
            if (g_backend_greedy) {
                take_id(llama_get_sampled_token_ith(ctx, -1));
            } else {
                take(llama_get_logits_ith(ctx, -1));
            }
            if (!job.trace.empty()) {
                trace.push_back({ ggml_time_us() - t_gen_start, 0, 'p' });
            }
        }
    } else if (job.mode == "pipe") {
        const int D       = std::max(0, job.depth);
        const int n_slots = llama_pipe_n_slots(ctx);
        if (n_slots < D + 2) {
            LOG_ERR("%s: depth %d needs at least %d slots, the context has %d\n", __func__, D, D + 2, n_slots);
            return false;
        }

        std::vector<llama_token> oracle;
        if (job.guess == "oracle") {
            if (!job.oracle.empty() && job.oracle[0] == '@') {
                auto it = done_jobs.find(job.oracle.substr(1));
                if (it != done_jobs.end()) {
                    oracle = it->second.gen;
                }
            } else {
                oracle = read_tokens(job.oracle);
            }
            if (oracle.empty()) {
                LOG_ERR("%s: no oracle tokens in '%s'\n", __func__, job.oracle.c_str());
                return false;
            }
        }

        common_speculative * spec = nullptr;
        if (job.guess == "spec") {
            params.speculative.types = common_speculative_types_from_names({ job.spec });
            spec = common_speculative_init(params.speculative, 1);
            if (!spec) {
                LOG_ERR("%s: failed to initialize the drafter '%s'\n", __func__, job.spec.c_str());
                return false;
            }
        }
        const common_ngram_simple_config simple_cfg = {
            params.speculative.ngram_simple.size_n, params.speculative.ngram_simple.size_m };

        struct entry { int slot; llama_token tok; llama_pos pos; };
        std::deque<entry> win; // win[0]: the newest committed token, in flight; win[1..]: guesses
        int next_slot = 0;
        bool failed = false;

        auto submit = [&](llama_token tok, llama_pos pos) {
            // round robin: the least recently used slot that is not in the window
            for (;;) {
                const int s = next_slot;
                next_slot = (next_slot + 1) % n_slots;
                bool used = false;
                for (const auto & e : win) {
                    used = used || e.slot == s;
                }
                if (used) {
                    continue;
                }
                const int ret = llama_pipe_submit(ctx, s, tok, pos, seq);
                if (ret != 0) {
                    LOG_ERR("%s: llama_pipe_submit failed (%d) at pos %d\n", __func__, ret, pos);
                    failed = true;
                    return;
                }
                win.push_back({ s, tok, pos });
                n_submit++;
                return;
            }
        };

        // committed history for the guessers: prompt + generated tokens
        std::vector<llama_token> hist(inp);
        hist.insert(hist.end(), gen.begin(), gen.end());

        std::deque<llama_token> chain;  // drafted but not yet submitted (guess "spec")
        int  chain_hits = 0;
        bool chain_live = false;

        if (spec) {
            std::vector<llama_token> prompt_tgt(hist.begin(), hist.end() - 1);
            common_speculative_begin(spec, seq, prompt_tgt);
        }

        // history including the guesses in flight
        auto hist_inflight = [&]() {
            std::vector<llama_token> h(hist);
            for (size_t i = 1; i < win.size(); i++) {
                h.push_back(win[i].tok);
            }
            return h;
        };

        // the next guess, for the position after win.back(), or -1
        auto next_guess = [&]() -> llama_token {
            const llama_pos pos = win.back().pos + 1;
            if (job.guess == "oracle") {
                const int j = pos - n_prompt;
                if (j < 0 || j >= (int) oracle.size()) {
                    return -1;
                }
                llama_token g = oracle[j];
                if (job.miss > 0.0 && unit_hash((uint64_t) j) < job.miss) {
                    g = (g + 1) % n_vocab;
                }
                return g;
            }
            if (job.guess == "copy") {
                return copy_guess(hist_inflight(), job.copy_min);
            }
            if (job.guess == "ngram-simple") {
                std::vector<llama_token> h = hist_inflight();
                const llama_token last = h.back();
                h.pop_back();
                const auto d = common_ngram_simple_draft(simple_cfg, h, last);
                return d.empty() ? -1 : d[0];
            }
            if (job.guess == "spec") {
                if (chain.empty() && win.size() == 1) {
                    if (chain_live) {
                        common_speculative_accept(spec, seq, chain_hits);
                    }
                    std::vector<llama_token> prompt_tgt(hist.begin(), hist.end() - 1);
                    std::vector<llama_token> draft;
                    common_speculative_get_draft_params(spec, seq) = {
                        /* .drafting = */ true,
                        /* .n_max    = */ std::max(1, n_predict - (int) gen.size()),
                        /* .pos0     = */ (llama_pos) (hist.size() - 1),
                        /* .id_last  = */ hist.back(),
                        /* .prompt   = */ &prompt_tgt,
                        /* .result   = */ &draft,
                    };
                    common_speculative_draft(spec);
                    chain.assign(draft.begin(), draft.end());
                    chain_hits = 0;
                    chain_live = !draft.empty();
                }
                if (chain.empty()) {
                    return -1;
                }
                const llama_token g = chain.front();
                chain.pop_front();
                return g;
            }
            return -1;
        };

        submit(gen.back(), n_prompt + (int) gen.size() - 1);

        while (!failed && !done()) {
            // keep up to D guesses in flight behind the newest committed token
            while (!failed && (int) win.size() < D + 1 && win.back().pos + 1 < n_prompt + n_predict) {
                const llama_token g = next_guess();
                if (g < 0) {
                    break;
                }
                submit(g, win.back().pos + 1);
                n_guess++;
            }
            if (failed) {
                break;
            }

            const int inflight = (int) win.size() - 1;
            depth_sum += inflight;
            depth_n++;

            const entry cur = win.front();
            win.pop_front();

            const llama_token id = g_backend_greedy ? take_id(llama_pipe_get_sampled(ctx, cur.slot))
                                                    : take(llama_pipe_get_logits(ctx, cur.slot));
            hist.push_back(id);

            char outcome = 'n'; // no guess in flight
            if (!win.empty()) {
                outcome = win.front().tok == id ? 'h' : 'm';
            }
            if (!job.trace.empty()) {
                trace.push_back({ ggml_time_us() - t_gen_start, inflight, outcome });
            }

            if (done()) {
                break;
            }

            if (outcome == 'h') {
                n_hit++;
                chain_hits++;
                continue;
            }

            if (outcome == 'm') {
                // wrong guess: drop it and everything behind it
                n_miss++;
                n_wasted += (int) win.size();
                if (!llama_memory_seq_rm(mem, seq, win.front().pos, -1)) {
                    LOG_ERR("%s: llama_memory_seq_rm failed\n", __func__);
                    failed = true;
                    break;
                }
                win.clear();
                chain.clear();
            }

            submit(id, cur.pos + 1);
        }

        // drop whatever is still in flight beyond the last fed token: plain decoding never fed it
        llama_pipe_drain(ctx);
        llama_memory_seq_rm(mem, seq, n_prompt + (llama_pos) gen.size() - 1, -1);

        if (spec) {
            common_speculative_free(spec);
        }
        if (failed) {
            return false;
        }
    } else {
        LOG_ERR("%s: unknown mode '%s'\n", __func__, job.mode.c_str());
        return false;
    }

    const double t_gen = (ggml_time_us() - t_gen_start)/1e6;

    {
        std::string text;
        for (auto id : gen) {
            text += common_token_to_piece(ctx, id);
        }
        LOG("\n[%s]\n%s\n\n", job.name.c_str(), text.c_str());
    }

    if (!job.out.empty()) {
        FILE * f = fopen(job.out.c_str(), "w");
        for (size_t i = 0; i < gen.size(); i++) {
            fprintf(f, "%zu %d %016" PRIx64 "\n", i, gen[i], hashes[i]);
        }
        fclose(f);
    }

    if (!job.trace.empty()) {
        FILE * f = fopen(job.trace.c_str(), "w");
        fprintf(f, "t_us\tinflight\toutcome\n");
        for (const auto & r : trace) {
            fprintf(f, "%" PRId64 "\t%d\t%c\n", r.t_us, r.inflight, r.outcome);
        }
        fclose(f);
    }

    uint64_t h_state = 0;
    {
        const size_t n = llama_state_seq_get_size(ctx, seq);
        std::vector<uint8_t> buf(n);
        const size_t n_written = llama_state_seq_get_data(ctx, buf.data(), buf.size(), seq);
        h_state = fnv1a(buf.data(), n_written);
        if (!job.state_out.empty()) {
            FILE * f = fopen(job.state_out.c_str(), "wb");
            fwrite(buf.data(), 1, n_written, f);
            fclose(f);
        }
    }

    uint64_t h_probe = 0;
    if (job.probe > 0) {
        // continue with the prompt's first tokens: their logits read every KV cell that is still visible
        std::vector<uint64_t> ph;
        for (int i = 0; i < job.probe; i++) {
            llama_token id  = inp[i % n_prompt];
            llama_pos   pos = n_prompt + (llama_pos) gen.size() - 1 + i;
            llama_batch b = llama_batch_get_one(&id, 1);
            b.pos = &pos;
            if (llama_decode(ctx, b) != 0) {
                LOG_ERR("%s: probe decode failed\n", __func__);
                return false;
            }
            if (g_backend_greedy) {
                const llama_token pid = llama_get_sampled_token_ith(ctx, -1);
                ph.push_back(fnv1a(&pid, sizeof(pid)));
            } else {
                ph.push_back(fnv1a(llama_get_logits_ith(ctx, -1), sizeof(float)*n_vocab));
            }
        }
        h_probe = fnv1a(ph.data(), ph.size()*sizeof(uint64_t));
    }

    const int n_gen = (int) gen.size();
    LOG_INF("PIPESPEC name=%s mode=%s guess=%s depth=%d n_prompt=%d n_gen=%d t_gen=%.3f tok_s=%.3f submits=%d guesses=%d "
            "hits=%d misses=%d wasted=%d avg_inflight_guesses=%.3f tokens=%016" PRIx64 " logits=%016" PRIx64 " state=%016" PRIx64
            " probe=%016" PRIx64 "\n",
            job.name.c_str(), job.mode.c_str(), job.mode == "plain" ? "-" : job.guess.c_str(), job.mode == "plain" ? 0 : job.depth,
            n_prompt, n_gen, t_gen, (n_gen - 1)/t_gen, n_submit, n_guess, n_hit, n_miss, n_wasted,
            depth_n ? depth_sum/depth_n : 0.0,
            fnv1a(gen.data(), gen.size()*sizeof(llama_token)), fnv1a(hashes.data(), hashes.size()*sizeof(uint64_t)),
            h_state, h_probe);

    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    pipe_job    def;
    std::string jobs_file;
    int         n_slots = 0;
    if (!parse_pipe_args(argc, argv, def, jobs_file, n_slots)) {
        return 1;
    }

    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }
    if (params.n_predict > 0) {
        def.n_predict = params.n_predict;
    }

    std::vector<pipe_job> jobs;
    if (jobs_file.empty()) {
        jobs.push_back(def);
    } else {
        std::ifstream f(jobs_file);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            pipe_job j = def;
            std::istringstream ss(line);
            std::string kv;
            while (ss >> kv) {
                const auto eq = kv.find('=');
                if (eq == std::string::npos || !job_set(j, kv.substr(0, eq), kv.substr(eq + 1))) {
                    fprintf(stderr, "bad job field '%s' in: %s\n", kv.c_str(), line.c_str());
                    return 1;
                }
            }
            jobs.push_back(j);
        }
    }

    int depth_max = -1;
    for (const auto & j : jobs) {
        if (j.mode == "pipe") {
            depth_max = std::max(depth_max, j.depth);
        }
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (!model || !ctx) {
        LOG_ERR("%s: failed to load the model\n", __func__);
        return 1;
    }

    llama_sampler * smpl_backend = nullptr;
    if (g_backend_greedy) {
        if (params.sampling.temp > 0.0f) {
            LOG_ERR("%s: --pipe-backend-greedy needs --temp 0\n", __func__);
            return 1;
        }
        smpl_backend = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl_backend, llama_sampler_init_greedy());
        if (!llama_set_sampler(ctx, 0, smpl_backend)) {
            LOG_ERR("%s: the greedy sampler cannot run on the backend\n", __func__);
            return 1;
        }
    }

    if (depth_max >= 0) {
        const int n = n_slots > 0 ? n_slots : 2*(depth_max + 1);
        if (llama_pipe_init(ctx, n) != 0) {
            LOG_ERR("%s: llama_pipe_init failed\n", __func__);
            return 1;
        }
    }

    std::map<std::string, job_result> done_jobs;
    int n_failed = 0;
    for (const auto & j : jobs) {
        job_result res;
        if (run_job(j, params, model, ctx, done_jobs, res)) {
            done_jobs[j.name] = std::move(res);
        } else {
            LOG_ERR("%s: job %s failed\n", __func__, j.name.c_str());
            n_failed++;
        }
    }

    llama_backend_free();

    if (smpl_backend) {
        llama_sampler_free(smpl_backend);
    }

    return n_failed == 0 ? 0 : 1;
}
