#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"
#include "llama-arch.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

//
// llama architecture queries
//

// all the LLM architectures known to the library
LLAMA_API std::vector<llm_arch> llama_arch_all();

// the canonical name of an architecture (e.g. "llama"), or "unknown"
LLAMA_API const char * llama_arch_name(llm_arch arch);

// the architecture with the given canonical name, or LLM_ARCH_UNKNOWN
LLAMA_API llm_arch llama_arch_from_string(const std::string & name);

// the arch-resolved GGUF key of a KV id, e.g. LLM_KV_CONTEXT_LENGTH -> "llama.context_length"
LLAMA_API std::string llama_arch_kv_name(llm_arch arch, llm_kv key);

LLAMA_API bool llama_arch_supports_sm_tensor(llm_arch arch);

LLAMA_API bool llama_arch_supports_mixed_batch(llm_arch arch);

// FIXME temporary function for better error messages
LLAMA_API bool llama_arch_supports_save(llm_arch arch);

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// the backend buffers that a state write of this memory would touch - empty caches report none
LLAMA_API std::vector<ggml_backend_buffer_t> llama_memory_get_buffers(const llama_memory_t mem);

// Turn the KV cache attention rotation on or off for the contexts created from this model.
// Useful for testing purposes.
LLAMA_API void llama_model_set_attn_rot(struct llama_model * model, bool value);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// Marks the entries that a decision head (clef, vllm-decision2) reads, the default is 0
// See https://github.com/ggml-org/llama.cpp/pull/29831 for details
// A run of entries with the same value is one span, spans must be separated by entries with value 0
// An option belongs to the last question before it
enum llama_decision_order {
    LLAMA_DECISION_ORDER_NONE            = 0, // not read by the head
    LLAMA_DECISION_ORDER_QUESTION_NOUL   = 1, // text of a question
    LLAMA_DECISION_ORDER_QUESTION_CHOICE = 2,
    LLAMA_DECISION_ORDER_QUESTION_SCORE  = 3,
    LLAMA_DECISION_ORDER_OPTION          = 4, // text of an option
};
// The embeddings output has one value per entry: row i is the score of option i
LLAMA_API bool llama_batch_ext_set_decision_order(struct llama_batch_ext * batch, int32_t idx, enum llama_decision_order order);

// Label of option i of a decision model ("<arch>.decision.labels"), nullptr past the last one
LLAMA_API const char * llama_model_decision_label(const struct llama_model * model, uint32_t i);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

// Write the enabled layer inputs to a device buffer [n_layer_inp*n_embd, n_batch] instead of host memory.
// Row r holds the token at batch index r, layers in ascending layer id order.
LLAMA_API void llama_set_embeddings_layer_inp_dev(struct llama_context * ctx, bool value);

// Whether decode accepts entries added with llama_batch_ext_add_embd_row()
// (needs a DFlash/DSpark decoder context whose ctx_other has a layer input device buffer).
LLAMA_API bool llama_supports_embd_row(const struct llama_context * ctx);

// Add an entry whose embedding is row `row` of the layer input device buffer of ctx_other.
// All entries of the batch must be of this kind. Positions are set with llama_batch_ext_set_pos(), n_pos_per_embd of them.
// Returns the batch index (>= 0), or the llama_batch_ext_add() error codes.
LLAMA_API int32_t llama_batch_ext_add_embd_row(struct llama_batch_ext * batch, llama_seq_id seq_id, int32_t row);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

// A shortlist for a draft context's output head (Q592): while it is on, a decode with one output computes the logits
// of the token ids below k_low and of up to m_cap further ids only, and gives every other token -INFINITY. The
// further ids are added with llama_head_shortlist_add (ids already in the list and ids past the room are skipped).
// Supported by the qwen4exp MTP draft. Returns false when the head can't take one (scaled, repacked or split).
LLAMA_API bool     llama_head_shortlist_init    (struct llama_context * ctx, uint32_t k_low, uint32_t m_cap);
LLAMA_API uint32_t llama_head_shortlist_add     (struct llama_context * ctx, const llama_token * ids, size_t n_ids);
LLAMA_API bool     llama_head_shortlist_contains(const struct llama_context * ctx, llama_token id);
LLAMA_API uint32_t llama_head_shortlist_size    (const struct llama_context * ctx); // ids in the list, 0 without one
LLAMA_API void     llama_set_head_shortlist     (struct llama_context * ctx, bool value);

// The main model's head over the context's shortlist (llama_head_shortlist_init) with an exact bound on every other
// logit (Q619): while it is on, a decode with one output computes the shortlist's logits and gives every other token
// -INFINITY; llama_head_bound_ith(i) is then at least every left-out logit the full head would have computed (Cauchy-
// Schwarz on the head's input row and the head's row norms), and +INFINITY for a row with the full head's logits.
// A sampler that can't exclude every left-out token with it calls llama_head_bound_full(i), which computes the full
// head for that row with the same kernel and writes it into the logits. The context's tokens join the shortlist as
// they are decoded. Supported by the qwen4exp main graph; init reads the head's rows once (norms) and returns false
// for heads it can't bound (scaled, with a precision policy, or without a shortlist).
LLAMA_API bool  llama_head_bound_init (struct llama_context * ctx);
LLAMA_API void  llama_set_head_bound  (struct llama_context * ctx, bool value);
LLAMA_API float llama_head_bound_ith  (struct llama_context * ctx, int32_t i);
LLAMA_API bool  llama_head_bound_full (struct llama_context * ctx, int32_t i);
LLAMA_API void  llama_head_bound_stats(const struct llama_context * ctx, uint64_t * n_rows, uint64_t * n_full);

// Fewer routed experts on confident tokens (Q618): after the router's top-k, a slot's weight becomes 0 when the slots
// ranked above it already hold a share p of the token's routed weight, or its own share is below w (w < 1/k); renorm
// rescales the kept weights to the token's sum of its k weights. The experts are still computed (graph level: their
// outputs are weighted 0). hist counts the experts kept per token and MoE layer (a synchronous read per decode),
// logged every 65,536 tokens and when the context is freed. LLAMA_MOE_TAIL=p=<p>,w=<w>,renorm=<0|1>,hist=<0|1> turns
// it on for every context. llama_moe_tail_counts fills counts[n] with the (token, layer) pairs that kept n experts,
// n < n_counts, and returns the pairs counted.
LLAMA_API void     llama_set_moe_tail   (struct llama_context * ctx, bool on, float p, float w, bool renorm, bool hist);
LLAMA_API uint64_t llama_moe_tail_counts(const struct llama_context * ctx, uint64_t * counts, int32_t n_counts);

// Route once per verify block (Q860): the tokens of a micro-batch form blocks of up to k consecutive positions of one
// sequence; a block's first token picks its routed experts as usual and every later token picks its own top-k only
// among those (its own router scores masked to that set, then the usual weight normalization), so a pass over the
// block reads one token's experts per layer. Lossy. k = 0: off; k = 1: the masked path with every token its own
// block's first (the same experts). max > 0: only micro-batches of at most max tokens (verify batches, not prompts).
// hist counts the distinct experts per full block and MoE layer, as read and under independent routing (a synchronous
// read per decode), logged every 65,536 tokens and when the context is freed; dry keeps the routing as it is and only
// counts. LLAMA_MOE_ROUTE_ONCE=<k>[,max=<n>][,hist=<0|1>][,dry=<0|1>] turns it on for every context.
// llama_moe_route_once_counts fills read[d] / indep[d] (d < n_counts) with the full blocks' (block, layer) pairs that
// read d distinct experts / would read them under independent routing, sets *n_outside to the later tokens'
// (token, layer) pairs that read an expert outside their block's first token's set, and returns the pairs counted.
LLAMA_API void     llama_set_moe_route_once   (struct llama_context * ctx, uint32_t k, uint32_t max, bool hist,
        bool dry);
LLAMA_API uint64_t llama_moe_route_once_counts(const struct llama_context * ctx, uint64_t * read, uint64_t * indep,
        int32_t n_counts, uint64_t * n_outside);

//
// model/context data extraction
//

LLAMA_API int32_t llama_model_dflash_selector_top_k(const struct llama_model * model);

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);

// same as above, but only the rows of the given tokens, in order (n_tokens * n_embd elements)
// returns the number of elements written or 0 on error
LLAMA_API uint32_t llama_model_get_tok_embd_rows(const struct llama_model * model, const llama_token * tokens, uint32_t n_tokens, float * out);

// realtime models: number of tokens the output lags behind the audio input, used by mtmd; 0 if not applicable
LLAMA_API uint32_t llama_model_n_tokens_delay(const struct llama_model * model);
