#pragma once

#include "llama.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct llama_model;
struct llama_ubatch;
class llm_graph_result;

// The staircase drafter (Q20, docs/drafter.md): a small transformer on one device that guesses the next token for
// pipelined speculation from the hidden states the pipe slots compute anyway. Every token contributes one row: its
// input embedding plus one projected hidden state of the target, the deepest one that exists when the guess is made,
// identified by its depth (n = the input of layer n, the target's layer count = the final hidden state after the output
// norm, 0 = no hidden state). Committed tokens use the final state; the tokens still in flight (the "window") use the
// depths their evaluations have reached, the newest the shallowest, up to the first token that has passed the last
// device: that one and every older token in flight use the final state too.
//
// The window comes from the run's layer split (layers per device): with every device busy a token enters the first
// device once per cycle, so the token m cycles older has finished every device that ends within m cycles (the stage
// times of the head file's timing model). Each depth then becomes the deepest depth the head was trained on at or below
// it. A head that runs on a device holding layers starts right after that device's stage ("stair"); a head on a spare
// device starts with the cycle ("pre").
//
// Data flow:
//   - every one-token pipe evaluation copies the inputs of the window's layers and its final hidden state into pinned
//     host memory on the device that computes them, right after that device's work for the token, and records an event
//     (on_submit, called by llama_context::pipe_submit_impl);
//   - a guess (launch) waits on its own stream for the events of the rows it reads, uploads each once into a device
//     store, runs the head (commit rows, then window rows) and copies back the top-k of its distribution;
//   - prompt micro-batches (llama_decode) hand over the final hidden state of every position (on_ubatch), which are
//     committed in bulk on the drafter's stream.
// The drafter only decides which token is guessed; the pipe executor still samples every token and compares, so the
// output never depends on it.
struct llama_pipe_drafter {
    ~llama_pipe_drafter();

    // loads the drafter file onto dev; tok_embd: the target model's input embedding table (on dev, or in host memory);
    // split: the target's layers per device in order (the output head on the last); n_layer_target: its layer count;
    // dev_has_layers: dev also computes target layers (the head then starts after that device's stage)
    static std::unique_ptr<llama_pipe_drafter> load(const llama_pipe_draft_params & params, ggml_tensor * tok_embd,
                                                    ggml_backend_dev_t dev, const std::vector<int32_t> & split,
                                                    int32_t n_layer_target, bool dev_has_layers);

    // layers whose inputs the pipe evaluations copy out (depths of the window other than the final state)
    std::vector<int32_t> tap_layers() const;
    ggml_backend_dev_t device() const { return dev; }

    // a one-token pipe evaluation of slot i_slot was just enqueued (all splits); copies out its taps
    void on_submit(int32_t i_slot, llama_token tok, llama_pos pos, llama_seq_id seq, const llm_graph_result * res,
                   ggml_backend_sched_t sched);

    // a llama_decode micro-batch was just enqueued: its positions of the drafter's sequence are committed
    void on_ubatch(const llama_ubatch & ubatch, const llm_graph_result * res, ggml_backend_sched_t sched);

    // starts over on sequence seq, keeping the committed rows of positions below n_keep
    void reset(llama_seq_id seq, llama_pos n_keep);

    // the token at pos, evaluated by pipe slot i_slot, is part of the output: its row becomes a committed row
    void commit(llama_token tok, llama_pos pos, int32_t i_slot);

    // enqueues a guess for the position after rows[n_rows - 1] (rows: the tokens in flight, oldest first, each with its
    // pipe slot); the pending commits go first
    int32_t launch(const llama_pipe_draft_row * rows, int32_t n_rows);

    // waits for the last launch: the guess drawn with the uniform u in [0, 1) against the drafter's distribution
    // (sharpened by alpha, its top-k candidates in descending order), the most likely token for u < 0; -1 if the
    // drafter has no token (p_top: the most likely token's probability)
    llama_token get(float u, float * p_top);

    // test hook: the logits of the newest window row of the last launch (n_vocab_draft + 1 values) after get()
    const std::vector<float> & last_logits() const { return logits_host; }
    void set_keep_logits(bool keep) { keep_logits = keep; graphs_valid = false; ensure_graphs(); }

    // test hook: the taps of a virtual evaluation from host rows: h holds n rows of n_embd floats at the given depths
    // (depths the drafter taps but h lacks become zeros)
    void test_set_taps(int32_t i_slot, llama_token tok, llama_pos pos, const float * h, const int32_t * depths, int32_t n);
    // test hook: commit rows from host memory, as on_ubatch does (h_full, h_fuse: n rows of n_embd floats)
    void test_commit_bulk(const llama_token * toks, const llama_pos * pos, int32_t n, const float * h_full, const float * h_fuse);
    void synchronize();

    // the window depths in use (newest first) and the depth of older tokens in flight
    const std::vector<int32_t> & window_depths() const { return stair; }
    int32_t overflow_depth() const { return depth_deep; }

    // stats
    int64_t n_launch = 0;
    int64_t n_rows_launched = 0;
    int64_t n_bulk_rows = 0;

private:
    llama_pipe_drafter() = default;

    struct graph_inst;

    bool init(const llama_pipe_draft_params & params, ggml_tensor * tok_embd, ggml_backend_dev_t dev,
              const std::vector<int32_t> & split, int32_t n_layer_target, bool dev_has_layers);
    bool build_graph(graph_inst & g, int32_t n_rows, bool bulk);
    void ensure_graphs();
    int32_t depth_name(int32_t depth) const;   // the deepest trained depth at or below depth
    void upload_tap(int64_t eval, int32_t tap);
    int32_t store_row(int64_t eval, int32_t depth);
    void run_rows(const std::vector<int64_t> & rows_eval, const std::vector<int32_t> & rows_depth,
                  const std::vector<llama_token> & rows_tok, const std::vector<llama_pos> & rows_pos,
                  const std::vector<bool> & rows_window, int32_t i_out);
    void lookup_emb(llama_token tok, float * dst) const;

    // hyperparameters (from the file)
    int32_t n_embd = 0, n_layer = 0, n_head = 0, n_head_kv = 0, head_dim = 0, n_ff = 0;
    int32_t n_vocab_draft = 0;
    // the output head as run: its first n_short draft ids (the most frequent) and the other class, n_short + 1 rows
    int32_t n_short = 0;
    // cost options (LLAMA_DRAFT_* environment): the last block for the output row only, and all depth projections as
    // one matrix product instead of one per row; both exact
    bool    prune_last = true;
    bool    anc_dense  = false;
    // the step graph's matrix products in groups of split_cols rows (0: all rows at once), for kernels that take few
    int32_t split_cols = 0;
    float   rms_eps = 1e-6f, rope_theta = 1e5f;
    bool    fuse = false;
    bool    general = false;
    int32_t n_target = 49;                // depth of the final state
    std::vector<int32_t> depth_proj;      // [n_target + 1] projection matrix per depth, -1 none
    std::vector<int32_t> depth_emb_row;   // [n_target + 1] row of the depth embedding per depth, -1 none
    std::vector<int32_t> recorded;        // layer inputs the head was trained on
    std::vector<int32_t> fuse_depths;     // depths of the fused projections (fused heads)
    float   timing[5] = { 0.435f, 0.024f, 0.3f, 2.5f, 0.3f };   // ms per layer at 6 cards, per extra card, minimum; head; hop
    std::vector<int32_t> vocab;           // draft id -> target token id
    float   alpha = 1.0f;
    int32_t top_k = 16;
    std::string profile;

    // the run: layer split, window depths (newest first), the depth of older tokens in flight (= committed), committed depth
    std::vector<int32_t> split;
    std::string kind = "stair";
    std::vector<int32_t> stair;
    int32_t depth_deep = 0;
    int32_t committed = 0;

    // taps: the depths copied out of every evaluation (the final state last), and the tap of each depth (-1 none)
    std::vector<int32_t> tap_depth;
    std::vector<int32_t> tap_of_depth;

    // devices and streams
    ggml_backend_dev_t dev = nullptr;
    ggml_backend_t     backend = nullptr;       // the drafter's own stream on dev
    ggml_backend_buffer_type_t buft_host = nullptr;

    // weights
    ggml_context *        ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    ggml_tensor * w_emb = nullptr, * w_anc = nullptr, * anchor_emb = nullptr, * w_fuse = nullptr;
    ggml_tensor * w_head = nullptr, * b_head = nullptr;
    struct block { ggml_tensor * wq, * wk, * wv, * wo, * w1, * w3, * w2; };
    std::vector<block> blocks;

    // the target's embedding table: read in the graph when it is on dev, else looked up on the host
    ggml_tensor * tok_embd = nullptr;
    bool emb_in_graph = false;

    // state on dev: K/V cells (ring of committed positions, window cells, scratch cells), each cell's position, the
    // store of uploaded taps (row 0 zero, then n_eval_ring evaluations x n_tap rows)
    int32_t n_ring = 0, n_wcell = 32, n_scratch = 16, n_cells = 0;
    // stage layout per launch: blob, cell positions, top-k ids, top-k probabilities, logits
    size_t off_cp = 0, off_ids = 0, off_p = 0, off_lg = 0;
    static constexpr int32_t n_eval_ring = 64;
    int32_t n_tap = 0;                       // tapped depths (+ an embedding row with a host lookup)
    ggml_context *        ctx_s = nullptr;
    ggml_backend_buffer_t buf_s = nullptr;
    std::vector<ggml_tensor *> k_cache, v_cache;
    ggml_tensor * cell_pos = nullptr;        // F32 [1, n_cells]
    ggml_tensor * store = nullptr;           // F32 [n_embd, 1 + n_eval_ring*n_tap]
    std::vector<float> cell_pos_host;        // mirror of the committed part
    bool cell_pos_dirty = true;

    // taps: pinned host rows and events per evaluation (ring), which rows are in the store
    ggml_backend_buffer_t buf_tap = nullptr;
    float * tap_host = nullptr;              // [n_eval_ring][n_tap][n_embd]
    std::vector<ggml_backend_event_t> ev_tap;      // [n_eval_ring][n_tap]
    std::vector<ggml_backend_dev_t>   ev_tap_dev;
    std::vector<uint64_t> uploaded;          // [n_eval_ring] bit per tap
    std::vector<int64_t>  eval_of_slot;      // pipe slot -> its last evaluation id
    int64_t n_eval = 0;

    // pending commits (pos order)
    struct pending { llama_token tok; llama_pos pos; int64_t eval; };
    std::vector<pending> commits;
    llama_seq_id seq = 0;

    // graphs: step (window + commits + head) and bulk commits
    struct graph_inst {
        int32_t n_rows = 0;
        bool    bulk   = false;
        ggml_context *        ctx = nullptr;     // graph nodes
        ggml_context *        ctx_in = nullptr;  // inputs
        ggml_backend_buffer_t buf_in = nullptr;  // inputs on dev (one blob)
        ggml_tensor * blob = nullptr;
        size_t        blob_size = 0;
        ggml_cgraph * gf = nullptr;
        ggml_gallocr_t galloc = nullptr;
        // inputs (views into blob)
        ggml_tensor * tokens = nullptr, * rope_pos = nullptr, * row_pos = nullptr, * cellval = nullptr;
        ggml_tensor * anc_id = nullptr, * anc_emb_id = nullptr, * anc_idx = nullptr, * fuse_idx = nullptr;
        ggml_tensor * kv_idx = nullptr, * wcell_idx = nullptr, * wcell_pos = nullptr, * out_row = nullptr;
        ggml_tensor * h_full = nullptr, * h_fuse = nullptr, * emb_rows = nullptr;   // bulk (emb_rows: host lookup)
        ggml_tensor * emb_idx = nullptr;                                          // step, host lookup: store rows
        ggml_tensor * out_pos = nullptr;                                          // step: the output row's position
        ggml_tensor * anc_sel = nullptr;                                          // step, dense: each row's projection
        // outputs
        ggml_tensor * top_ids = nullptr, * top_p = nullptr, * logits = nullptr;
        ~graph_inst();
    };
    std::unique_ptr<graph_inst> g_step;
    std::unique_ptr<graph_inst> g_bulk_small, g_bulk_large;
    bool graphs_valid = false;
    int32_t n_rows_step = 8;

    // launches: pinned staging and results per ring entry
    static constexpr int32_t n_launch_ring = 4;
    ggml_backend_buffer_t buf_stage = nullptr;
    uint8_t * stage_host = nullptr;
    size_t    stage_stride = 0;
    std::vector<ggml_backend_event_t> ev_launch;
    std::vector<bool> launch_pending;
    int32_t launch_next = 0;
    int32_t launch_last = -1;          // the entry get() reads
    bool    launch_has_out = false;
    bool    keep_logits = false;
    std::vector<float> logits_host;

    // prompt captures (pinned, per micro-batch ring entry): committed-depth rows, embedding rows (host lookup), the
    // first fused depth's rows (fused heads)
    static constexpr int32_t n_cap_ring = 3;
    struct capture {
        ggml_backend_buffer_t buf = nullptr;
        float * full = nullptr;
        float * fz = nullptr;
        size_t  n_rows_max = 0;
        ggml_backend_event_t ev_full = nullptr, ev_fz = nullptr;
        ggml_backend_dev_t   dev_full = nullptr, dev_fz = nullptr;
        ggml_backend_event_t ev_done = nullptr;
        bool pending = false;
    };
    capture caps[n_cap_ring];
    int32_t cap_next = 0;

    void stage_wait(int32_t i);
    void launch_done(int32_t i);
    capture & cap_take(int32_t & ic, size_t n_rows);
    void write_blob(graph_inst & g, uint8_t * dst, const std::vector<int32_t> & tok, const std::vector<int32_t> & rope,
                    const std::vector<float> & rpos, const std::vector<float> & cval, const std::vector<int32_t> & aid,
                    const std::vector<int32_t> & aeid, const std::vector<int32_t> & aidx, const std::vector<int32_t> & fidx,
                    const std::vector<int64_t> & kvi, const std::vector<float> & wpos, int32_t out_row);
    void commit_rows_bulk(const llama_token * toks, const llama_pos * pos, int32_t n, int32_t i_cap, size_t row0);
};

// The window of a guesser for a layer split (layers per device, the output head on the last; timing as in the head file),
// newest token first: the depth (layers done) of each token in flight, up to the first complete one. kind "stair": a
// head that starts right after the first device's stage; "pre": at the start of a cycle; "miss": the first guess after
// a miss. Mirrors split_depths in bench/staircase_drafter/drafter.py.
std::vector<int32_t> llama_pipe_split_depths(const std::vector<int32_t> & split, const std::string & kind, const float timing[5],
                                             int32_t max_w = 12);
