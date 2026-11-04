#include "llama-embd-cache.h"

#include "ggml-alloc.h"

#include <algorithm>

bool llama_embd_cache::init(const ggml_tensor * src, uint32_t n_rows, uint32_t max_tokens) {
    if (src == nullptr || ggml_n_dims(src) != 2 || !ggml_is_contiguous(src) || n_rows == 0 || max_tokens == 0 ||
            max_tokens > n_rows || (int64_t) n_rows > src->ne[1]) {
        return false;
    }

    this->src        = src;
    this->n_vocab    = (uint32_t) src->ne[1];
    this->n_rows     = n_rows;
    this->max_tokens = max_tokens;
    this->row_size   = src->nb[1];

    slot_of.assign(n_vocab, -1);
    id_of.assign(n_rows, -1);
    last_use.assign(n_rows, 0);
    n_used  = 0;
    n_batch = 0;

    slots.clear();
    row.resize(row_size);

    n_hit = n_miss = n_plain = 0;

    return true;
}

bool llama_embd_cache::alloc(ggml_backend_buffer_type_t buft) {
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx.reset(ggml_init(params));
    if (!ctx) {
        return false;
    }

    rows = ggml_new_tensor_2d(ctx.get(), src->type, src->ne[0], n_rows);
    ggml_set_name(rows, "embd_cache_rows");

    buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
    if (!buf) {
        rows = nullptr;
        return false;
    }
    // placed like weights: the get_rows that reads it runs where it is
    ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    return true;
}

bool llama_embd_cache::prepare(const int32_t * ids, uint32_t n, ggml_backend_t backend) {
    if (rows == nullptr || n == 0 || n > max_tokens) {
        n_plain++;
        return false;
    }
    for (uint32_t i = 0; i < n; ++i) {
        if (ids[i] < 0 || (uint32_t) ids[i] >= n_vocab) {
            n_plain++;
            return false;
        }
    }

    ++n_batch;
    slots.resize(n);

    // the rows held already first, so that none of them gives way to another token of the same batch
    for (uint32_t i = 0; i < n; ++i) {
        const int32_t s = slot_of[ids[i]];
        slots[i] = s;
        if (s >= 0) {
            last_use[s] = n_batch;
        }
    }

    bool synced = false;
    for (uint32_t i = 0; i < n; ++i) {
        if (slots[i] >= 0) {
            n_hit++;
            continue;
        }

        const int32_t id = ids[i];
        if (slot_of[id] >= 0) {
            // the same token earlier in this batch
            slots[i] = slot_of[id];
            n_hit++;
            continue;
        }

        // a free slot, else the one read longest ago; max_tokens <= n_rows leaves one this batch does not read
        int32_t v = -1;
        if (n_used < n_rows) {
            v = (int32_t) n_used++;
        } else {
            uint64_t oldest = n_batch;
            for (uint32_t s = 0; s < n_rows; ++s) {
                if (last_use[s] < oldest) {
                    oldest = last_use[s];
                    v      = (int32_t) s;
                }
            }
        }
        GGML_ASSERT(v >= 0);

        // a graph queued on the layer's card may still read the slot
        if (!synced && backend != nullptr) {
            ggml_backend_synchronize(backend);
            synced = true;
        }

        if (id_of[v] >= 0) {
            slot_of[id_of[v]] = -1;
        }
        ggml_backend_tensor_get(src, row.data(), (size_t) id*src->nb[1], row_size);
        ggml_backend_tensor_set(rows, row.data(), (size_t) v*rows->nb[1], row_size);

        id_of[v]    = id;
        slot_of[id] = v;
        last_use[v] = n_batch;
        slots[i]    = v;
        n_miss++;
    }

    return true;
}
