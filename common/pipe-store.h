#pragma once

#include "llama.h"

#include <cstdint>
#include <string>

// A continuation store for pipelined speculation (registry row Q161): the generated tokens of many earlier answers as a
// suffix array over the reversed token stream, memory-mapped from a file. A query reads the context backwards (the last
// token first) and narrows the suffix array one token at a time; the guess is the most frequent next token after the
// longest context that occurs in the store (SuffixDecoding's frequency scoring, one token at a time).
//
// File (built by the public repository's bench/ngram_replay/datastore.cpp): the 8 bytes "PIPESTO1", the token count n
// as int64, then n tokens (the stream reversed) and n suffix-array entries of that reversed stream, int32 each, all
// little-endian. Answers are separated by the token -1, so no context matches across two answers. 8 bytes per token.
struct common_pipe_store;

// nullptr if the file cannot be read or is not a store
common_pipe_store * common_pipe_store_load(const std::string & path);
void                common_pipe_store_free(common_pipe_store * store);

// stored tokens, separators included
int64_t common_pipe_store_size(const common_pipe_store * store);

// The longest suffix of ctx[0..n) (at most max_len tokens) that occurs in the store with a token after it: returns its
// length (0: none) and fills its most frequent next token (the smallest id on ties), that token's count, and the count
// of all next tokens. Thread-safe.
int common_pipe_store_longest(common_pipe_store * store, const llama_token * ctx, int n, int max_len,
                              llama_token * tok, int32_t * cnt, int32_t * occ);

// The store's guess for pipelined speculation: the most frequent next token of the longest context of 2..32 tokens at
// the end of ctx[0..n) that the store holds, if that token has at least 30 % of the context's occurrences; else -1
// (the rule Q161 found best on held-out transcripts)
llama_token common_pipe_store_guess(common_pipe_store * store, const llama_token * ctx, int n);
