#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// A draft's token embedding rows, cached on the card that runs its layer (Q961). With the table on another card (S15's
// -otd token_embd=CUDA4) every draft pass evaluates that card for one row and crosses to the layer's card through the
// host, since the cards have no peer-to-peer. The cache holds n_rows rows of the table in the table's own type and
// bytes, in a buffer of the layer's card, and a batch of at most max_tokens tokens reads its rows there: get_rows over
// the same bytes, so the draft computes the same numbers. A row comes from the table when a token first needs it and
// takes the slot read longest ago; larger batches (prompts) read the table as before.
struct llama_embd_cache {
    const ggml_tensor * src = nullptr; // the table [n_embd, n_vocab], contiguous

    uint32_t n_vocab    = 0;
    uint32_t n_rows     = 0;
    uint32_t max_tokens = 0;
    size_t   row_size   = 0; // bytes of one row in the table's type

    std::vector<int32_t>  slot_of;  // [n_vocab]: the token's slot, -1 if it is not cached
    std::vector<int32_t>  id_of;    // [n_rows]: the slot's token, -1 while the slot is free
    std::vector<uint64_t> last_use; // [n_rows]: the batch that last read the slot
    uint32_t              n_used  = 0; // slots handed out so far (the free ones are n_used ... n_rows - 1)
    uint64_t              n_batch = 0;

    std::vector<int32_t> slots; // the slots of the last prepared batch, one per token
    std::vector<uint8_t> row;   // one row, on its way from the table to the cache

    // device copy: rows [n_embd, n_rows] in one buffer of the layer's buffer type (alloc())
    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf;
    ggml_tensor * rows = nullptr;

    // tokens read from the cache, of them uploaded first, and batches that read the table
    uint64_t n_hit   = 0;
    uint64_t n_miss  = 0;
    uint64_t n_plain = 0;

    // the host side; false for sizes it can't serve (max_tokens must fit in n_rows)
    bool init(const ggml_tensor * src, uint32_t n_rows, uint32_t max_tokens);

    // the device rows, in a buffer of type buft (the layer's)
    bool alloc(ggml_backend_buffer_type_t buft);

    // maps the n tokens of a batch to slots (slots), uploading the rows the cache does not hold yet; backend runs the
    // graphs that read the rows: it finishes its queued work before the first row of a batch is overwritten. False
    // when the batch reads the table instead: more than max_tokens tokens, or an id outside the vocabulary
    bool prepare(const int32_t * ids, uint32_t n, ggml_backend_t backend);
};
