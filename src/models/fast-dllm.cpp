#include "models.h"
#include "diffusion-common.h"

// Fast-dLLM v2 (arXiv:2509.26328): Qwen2.5 converted to a BLOCK-diffusion model.
// Attention at inference is block-causal over the whole sequence: bidirectional
// within each bd_size block, causal across blocks (modeling.py eval_block_diff_mask:
// allow iff q_block >= kv_block). Token-shifted like Dream (row i predicts pos i+1).
// v1 runs the UNCACHED square path only; committed blocks never change, so a cached
// path can later reuse the standard prefix machinery.

// square no-cache mask, block-causal by POSITION (not row index), per sequence
class llm_graph_input_attn_block_causal : public llm_graph_input_attn_no_cache {
public:
    llm_graph_input_attn_block_causal(const llama_hparams & hparams, const llama_cparams & cparams,
                                      uint32_t block_size) :
        llm_graph_input_attn_no_cache(hparams, cparams), block_size(block_size) {}
    ~llm_graph_input_attn_block_causal() = default;

    void set_input(const llama_ubatch * ubatch) override {
        const int64_t n = ubatch->n_tokens;  // square: n_kv == n_tokens

        const auto fill = [&](auto * data) {
            using T = std::remove_reference_t<decltype(*data)>;
            std::fill(data, data + n * n, llama_cast<T>(-INFINITY));
            for (int64_t i1 = 0; i1 < n; ++i1) {
                const llama_seq_id s1 = ubatch->seq_id[i1][0];
                const llama_pos    p1 = ubatch->pos[i1];
                for (int64_t i0 = 0; i0 < n; ++i0) {
                    if (ubatch->seq_id[i0][0] != s1) {
                        continue;
                    }
                    const llama_pos p0 = ubatch->pos[i0];
                    if (p1 / (llama_pos) block_size >= p0 / (llama_pos) block_size) {
                        data[i1 * n + i0] = llama_cast<T>(0.0f);
                    }
                }
            }
        };

        GGML_ASSERT(self_kq_mask);
        if (self_kq_mask->type == GGML_TYPE_F16) {
            fill((ggml_fp16_t *) self_kq_mask->data);
        } else {
            fill((float *) self_kq_mask->data);
        }
    }

    const uint32_t block_size;
};

void llama_model_fast_dllm::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key("fast-dllm.block_size", bd_size, false);

    switch (hparams.n_layer()) {
        case 28:
            type = LLM_TYPE_1_5B;
            break;
        default:
            type = LLM_TYPE_UNKNOWN;
    }
    // bidirectional within blocks - the graph builds its own block-causal mask
    hparams.causal_attn = false;
}

void llama_model_fast_dllm::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    // tied embeddings (Qwen2.5-1.5B)
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_fast_dllm::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_fast_dllm::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    // qwen2 forward with a block-causal square mask
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    const auto & dmodel = (const llama_model_fast_dllm &) model;

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();

    // block-causal mask input (replaces build_attn_inp_no_cache)
    llm_graph_input_attn_no_cache * inp_attn = nullptr;
    {
        auto uptr = std::make_unique<llm_graph_input_attn_block_causal>(hparams, cparams, dmodel.bd_size);
        const auto type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;
        uptr->self_kq_mask = ggml_new_tensor_4d(ctx0, type_mask, n_tokens, n_tokens, 1, 1);
        ggml_set_input(uptr->self_kq_mask);
        uptr->self_kq_mask_cnv = uptr->self_kq_mask;
        inp_attn = (llm_graph_input_attn_no_cache *) res->add_input(std::move(uptr));
    }

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);

            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f / sqrtf(float(n_embd_head)), il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
            model.layers[il].ffn_up, NULL, NULL,
            model.layers[il].ffn_gate, NULL, NULL,
            model.layers[il].ffn_down, NULL, NULL,
            NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
