#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;

// comma separated list the provided types
std::string common_speculative_type_name_str(const std::vector<enum common_speculative_type> & types);

// comma separated list of all types
const char * common_speculative_all_types_str();

// parse user provided types
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

// infer the spec types from the GGUF metadata of a draft model; empty if unknown
std::vector<enum common_speculative_type> common_speculative_types_from_gguf(const std::string & path);

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// return the max number of draft tokens based on the speculative parameters
int32_t common_speculative_n_max(const common_params_speculative * spec);

// return the max number of draft tokens from the initialized implementations
int32_t common_speculative_n_max(const common_speculative * spec);

// validate and resolve the unconditional synthetic acceptance rates
std::vector<double> common_speculative_synth_rates_resolve(const common_params_speculative * spec, int32_t n_max);

// return the conditional synthetic acceptance probabilities
const std::vector<double> & common_speculative_get_synth_probs(const common_speculative * spec);

common_params common_base_params_to_speculative(const common_params & params);

struct common_speculative_output_limits {
    int32_t total;
    int32_t per_seq;
};

// return the output limits needed for speculative decoding
common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft);

// return true if the target and draft models have compatible vocabs
bool common_speculative_are_compatible(const llama_model * model_tgt, const llama_model * model_dft);

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);

void common_speculative_free(common_speculative * spec);

// What a drafter proposed at each position of a draft, for offline analysis of token trees and draft lengths
// (llama-server's LLAMA_SPEC_DUMP). An implementation fills it only when the caller sets
// common_speculative_draft_params::info; nothing is computed otherwise.
struct common_speculative_draft_info {
    // in
    int32_t k_max      = 8;     // the most candidates kept per position
    // DFlash2: also decode a second block, for this record only, and keep its lattice in fb_* (0: no; 1: the trained
    // block; n >= 2: a block of n tokens, the anchor and n - 1 masks, capped at the trained block). The draft is
    // decoded first, from the block the line would decode without the record, so it is the same either way.
    int32_t full_block = 0;

    // out, one entry per position after the anchor of the draft's own block (DFlash2: n_max positions)
    std::string type;            // the implementation that filled it
    bool        lattice = false; // true: DFlash2's selector, p[i] holds the conditional rows; false: marginals
    std::vector<std::vector<llama_token>> ids;  // candidates, highest marginal probability first
    // lattice: p[0] has ids[0].size() values (given the anchor), p[i > 0] ids[i - 1].size() * ids[i].size() values,
    //          row-major, row = the candidate taken at position i - 1: the selector's probabilities over its whole
    //          top-k, so a row sums to less than 1 when candidates were cut
    // else:    p[i] has ids[i].size() values, the drafter's probabilities given the drafted prefix
    std::vector<std::vector<float>> p;
    std::vector<int32_t> pick; // the drafter's own walk: index into ids[i] (-1: the walk stopped before it)
    std::vector<float>   kept; // lattice: the selector's marginal mass of the kept candidates per position

    // out, full_block: the second block's lattice, in the same form as ids, p, pick and kept (always a lattice), one
    // entry per position after the anchor; fb_pick is the selector's greedy walk over the whole block
    int32_t fb_width = 0; // the block's tokens, the anchor included (0: not decoded)
    std::vector<std::vector<llama_token>> fb_ids;
    std::vector<std::vector<float>>       fb_p;
    std::vector<int32_t>                  fb_pick;
    std::vector<float>                    fb_kept;

    void clear() {
        type.clear();
        lattice = false;
        ids.clear();
        p.clear();
        pick.clear();
        kept.clear();
        fb_width = 0;
        fb_ids.clear();
        fb_p.clear();
        fb_pick.clear();
        fb_kept.clear();
    }
};

// Reduce a DFlash2 selector lattice to info (exposed for tests). Each of the n_pos rows, row_stride floats apart,
// holds top_k candidate ids (as floats) and top_k * top_k scores, row-major with row = the predecessor's candidate
// index (the first position's rows all score from the anchor). walk[i] is the candidate index the drafter took at
// position i, -1 after it stopped. Probabilities are softmax(scores) at temperature 1, the forward marginals rank
// the candidates, and each position keeps the info.k_max highest plus the walked one.
void common_speculative_lattice_info(common_speculative_draft_info & info, const float * lattice, size_t row_stride,
        int32_t n_pos, int32_t top_k, const std::vector<int32_t> & walk);

// The selector's greedy walk over a lattice laid out as above: the candidate index taken at each of n_pos positions,
// each the highest-scoring candidate given the previous one (the first position scores from the anchor alone).
std::vector<int32_t> common_speculative_lattice_walk(const float * lattice, size_t row_stride, int32_t n_pos, int32_t top_k);

// Reduce a whole decoded block's lattice (n_pos positions after the anchor) along its greedy walk into info.fb_*,
// leaving the draft's own record (ids, p, pick, kept) alone. The candidates kept are info.k_max's.
void common_speculative_full_block_info(common_speculative_draft_info & info, const float * lattice, size_t row_stride,
        int32_t n_pos, int32_t top_k);

struct common_speculative_draft_params {
    // this flag is used to chain the drafts through all the available implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    // at the end of the draft() call, all drafting flags will be reset to false
    bool drafting = false;

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    int32_t n_max = -1;

    llama_pos   pos0;
    llama_token id_last;

    // TODO: remove in the future by keeping track of the prompt from the _begin() call and the consecutive accept calls
    const llama_tokens * prompt;

    // the generated draft from the last _draft() call
    llama_tokens * result;

    // candidate distribution per drafted token; set it to make draft-simple, draft-mtp and draft-dflash (DFlash2 only) sample
    std::vector<std::vector<llama_token_data>> * result_q = nullptr;

    // the target's temp and seed, read only when the drafter samples probabilistically
    float    temp = 1.0f;
    uint32_t seed = LLAMA_DEFAULT_SEED;

    // when set, the implementation that drafts records its candidates here (see common_speculative_draft_info)
    common_speculative_draft_info * info = nullptr;
};

common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// process the batch and update the internal state of the speculative context
bool common_speculative_process(common_speculative * spec, const common_batch & batch);

// generate drafts for the sequences specified with `common_speculative_get_draft_params`
void common_speculative_draft(common_speculative * spec);

// informs the speculative context that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, llama_seq_id, uint16_t n_accepted);

// (optional) get/set internal state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data);
void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;

struct common_speculative_init_result {
    common_speculative_init_result(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
    ~common_speculative_init_result();

    llama_model   * model();
    llama_context * context();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using common_speculative_init_result_ptr = std::unique_ptr<common_speculative_init_result>;

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
