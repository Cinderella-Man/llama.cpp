#pragma once

#include "llama.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

// Guesses the next token of a sequence for pipelined speculation (llama_pipe_*): the guess only decides how much work
// runs ahead, every token is still sampled and compared, so a wrong guess costs time and never changes the output.
//
// copy: the token after the longest earlier occurrence (up to 12 tokens, the most recent one on ties) of the history's
//       suffix, if that occurrence is at least min_match tokens long; the newest 256 occurrences of the last token are
//       searched
// cont: the copy lookup if it finds a match, else the most frequent continuation of the longest 1..8-token context at
//       the end of the history among the request's own tokens (prompt and committed output). With a store
//       (common_pipe_store, Q161): the copy lookup, else the request's continuations of contexts of 2 tokens or more,
//       else the store's most frequent continuation of the longest context (2 tokens or more) it holds, if that token
//       has at least 30 % of the context's occurrences, else the request's continuations of any context. With cont_any
//       off (the staircase drafter guesses next): the copy lookup, the request's continuations of contexts of 2 tokens
//       or more, the store if there is one, and nothing from a 1-token context
struct common_pipe_store;

struct common_pipe_guesser {
    enum type { NONE, COPY, CONT };

    type kind      = CONT;
    int  min_match = 4;

    // continuations of other answers for cont (not owned; the slots share it), or nullptr
    common_pipe_store * store = nullptr;

    // guesses that came from the store (counted up; the owner clears it)
    int32_t n_store = 0;

    // cont: the request's continuations of any context as the last guess
    bool cont_any = true;

    // starts a request: the prompt's tokens (the counts of the continuation lookup cover its last max_count tokens)
    void reset(const llama_token * tokens, size_t n);

    // appends a committed token
    void push(llama_token tok);

    // the guess for the token after the history plus the tokens in flight behind it, or -1
    llama_token guess(const llama_token * inflight, size_t n_inflight);

    size_t size() const { return hist.size(); }

private:
    static constexpr int    max_ctx   = 8;
    static constexpr int    max_copy  = 12;
    static constexpr int    max_cand  = 256;
    static constexpr size_t max_count = 32768;

    std::vector<llama_token> hist;

    // positions of each token value in hist (ascending), for the copy lookup
    std::unordered_map<llama_token, std::vector<int32_t>> pos_of;

    // context hash -> (next token, count) for the continuation lookup, over hist[count_from, n_counted)
    std::unordered_map<uint64_t, std::vector<std::pair<llama_token, int32_t>>> counts;
    size_t count_from = 0;
    size_t n_counted  = 0;

    static uint64_t key(const llama_token * p, int n);

    void index(int32_t i);
    void count_to(size_t n);

    llama_token copy_guess() const;
    llama_token cont_guess(int min_len = 1) const;   // from a context of at least min_len tokens
};
