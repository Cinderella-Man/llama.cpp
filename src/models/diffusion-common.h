#pragma once

// Shared machinery for cached diffusion decoding (docs/dllms/throughput-plans/01_layer_a.md).
// Used by dream.cpp, llada.cpp and diffusion-gemma.cpp. The pkv state struct itself lives in
// models.h (llama_diffusion_pkv) so the model structs can embed it without include cycles.

#include "models.h"

// Decode-phase mask: queries of the current batch (C rows) over [cached prefix (first n_kv - C)
// | fresh batch K,V (last C)], rectangular [n_kv, C]. For fully bidirectional models
// (Dream/LLaDA: swa_type NONE) every key is visible; for DiffusionGemma the SWA variant clips
// canvas->prompt reach to the last (n_swa-1) prompt positions.
//
// Implementer notes (see guide sec 3e0): the caller MUST set self_kq_mask_cnv = self_kq_mask
// (build_attn consumes the _cnv alias), and the mask's ne[0] must equal the attended K's row
// count exactly - both derive from the same n_kv.
class llm_graph_input_attn_diffusion_decode : public llm_graph_input_attn_no_cache {
public:
    llm_graph_input_attn_diffusion_decode(const llama_hparams & hparams, const llama_cparams & cparams,
                                          int64_t n_prompt, int64_t n_canvas) :
        llm_graph_input_attn_no_cache(hparams, cparams), n_prompt(n_prompt), n_canvas(n_canvas) {}
    ~llm_graph_input_attn_diffusion_decode() = default;

    void set_input(const llama_ubatch * /*ubatch*/) override {
        const int64_t P    = n_prompt;
        const int64_t C    = n_canvas;
        const int64_t n_kv = P + C;
        const int64_t canvas_prompt_lo = P - (int64_t) hparams.n_swa + 1;

        const auto fill = [&](auto * data, bool swa) {
            using T = std::remove_reference_t<decltype(*data)>;
            std::fill(data, data + n_kv * C, llama_cast<T>(-INFINITY));
            for (int64_t q = 0; q < C; ++q) {            // batch query (position P+q)
                const uint64_t row = q * n_kv;
                for (int64_t k = 0; k < n_kv; ++k) {     // key: k<P cached prefix, else batch
                    bool allow;
                    if (k < P) {
                        allow = swa ? (k >= canvas_prompt_lo) : true;
                    } else {
                        allow = true;                     // bidirectional over the batch
                    }
                    if (allow) {
                        data[row + k] = llama_cast<T>(0.0f);
                    }
                }
            }
        };

        GGML_ASSERT(self_kq_mask && ggml_backend_buffer_is_host(self_kq_mask->buffer));
        if (self_kq_mask->type == GGML_TYPE_F16) {
            fill((ggml_fp16_t *) self_kq_mask->data, false);
        } else {
            fill((float *) self_kq_mask->data, false);
        }
        if (self_kq_mask_swa) {
            GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask_swa->buffer));
            if (self_kq_mask_swa->type == GGML_TYPE_F16) {
                fill((ggml_fp16_t *) self_kq_mask_swa->data, true);
            } else {
                fill((float *) self_kq_mask_swa->data, true);
            }
        }
    }

    // reusable when the batch width is unchanged; pkv-state changes are guarded by
    // llm_graph_input_diffusion_phase (Dream/LLaDA). DiffusionGemma constructs this class
    // without that marker and relies on shape changes - keep its conservative behavior by
    // only allowing reuse when explicitly enabled.
    bool can_reuse(const llm_graph_params & params) override {
        return allow_reuse && params.ubatch.n_tokens == n_canvas;
    }

    int64_t n_prompt;
    int64_t n_canvas;
    bool    allow_reuse = false;
};

// Phase-marker graph input: carries the pkv state a graph was built for and blocks graph
// reuse whenever the CURRENT model pkv state differs. Without it, two phases with identical
// batch shapes (e.g. consecutive WARMs of different blocks, or UNIFIED after a cached run)
// would silently reuse each other's graphs - whose store-write views bake in the OLD P/s.
// Adding it in EVERY phase (UNIFIED included) also re-enables llama.cpp's graph reuse
// WITHIN a block (same phase, same shapes), which a blanket can_reuse=false would forfeit -
// measured at ~3x per-step wall on small canvases.
class llm_graph_input_diffusion_phase : public llm_graph_input_i {
public:
    llm_graph_input_diffusion_phase(const llama_diffusion_pkv * pkv) :
        pkv(pkv), phase(pkv->phase), P(pkv->P), L(pkv->L), s(pkv->s) {}

    void set_input(const llama_ubatch * /*ubatch*/) override {}

    bool can_reuse(const llm_graph_params & /*params*/) override {
        return pkv->phase == phase && pkv->P == P && pkv->L == L && pkv->s == s;
    }

    const llama_diffusion_pkv * pkv;
    int     phase;
    int64_t P, L, s;
};

// Lazily (re)allocate a device-resident F32 K/V store (per-layer, grow-only) with capacity
// `cap` rows, on layer-0's buffer type (single-GPU per model; do not use with -sm row/layer
// splits). Mirrors dg_ensure_pkv_store (diffusion-gemma.cpp) for the shared pkv struct.
static inline void llama_diffusion_pkv_ensure_store(const llama_model_base & m, llama_diffusion_pkv & pkv, int64_t cap) {
    if (pkv.buf != nullptr && pkv.cap >= cap) {
        return;
    }
    if (pkv.buf) { ggml_backend_buffer_free(pkv.buf); pkv.buf = nullptr; }
    if (pkv.ctx) { ggml_free(pkv.ctx); pkv.ctx = nullptr; }
    pkv.k.clear();
    pkv.v.clear();

    const int n_layer = (int) m.hparams.n_layer();

    ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (size_t) (2 * n_layer + 4),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    pkv.ctx = ggml_init(ip);
    GGML_ASSERT(pkv.ctx != nullptr);
    pkv.k.resize(n_layer);
    pkv.v.resize(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        const int64_t hd  = m.hparams.n_embd_head_k(il);
        const int64_t nkv = m.hparams.n_head_kv(il);
        pkv.k[il] = ggml_new_tensor_3d(pkv.ctx, GGML_TYPE_F32, hd, nkv, cap);
        pkv.v[il] = ggml_new_tensor_3d(pkv.ctx, GGML_TYPE_F32, hd, nkv, cap);
        ggml_format_name(pkv.k[il], "diff_pkv_k_l%d", il);
        ggml_format_name(pkv.v[il], "diff_pkv_v_l%d", il);
    }

    ggml_backend_dev_t dev = m.dev_layer(0);
    ggml_backend_buffer_type_t buft = dev ? ggml_backend_dev_buffer_type(dev)
                                          : ggml_backend_cpu_buffer_type();
    pkv.buf = ggml_backend_alloc_ctx_tensors_from_buft(pkv.ctx, buft);
    GGML_ASSERT(pkv.buf != nullptr);
    pkv.cap = cap;
}
