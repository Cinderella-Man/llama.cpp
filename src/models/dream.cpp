#include "models.h"
#include "diffusion-common.h"

void llama_model_dream::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    // Dream models are primarily 7B with 28 layers
    switch (hparams.n_layer()) {
        case 28:
            type = LLM_TYPE_7B;
            break;
        default:
            type = LLM_TYPE_UNKNOWN;
    }
    // Set non-causal attention for diffusion models
    hparams.causal_attn = false;
}

void llama_model_dream::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    output_b    = create_tensor(tn(LLM_TENSOR_OUTPUT,      "bias"),   {n_vocab}, TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed
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

std::unique_ptr<llm_graph_context> llama_model_dream::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_dream::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    //copied from qwen2
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    // cached diffusion decoding (Layer A): phase selected via llama_diffusion_set_phase /
    // llama_diffusion_set_block before llama_decode; UNIFIED is byte-identical to the
    // original graph
    const auto &  dmodel = (const llama_model_dream &) model;
    const int     phase  = dmodel.pkv.phase;
    const int64_t pkv_P  = dmodel.pkv.P;
    const int64_t pkv_L  = dmodel.pkv.L;
    const int64_t pkv_s  = dmodel.pkv.s;

    // IMPORTANT (discovered the hard way): scheduler reserve graphs are built with
    // arbitrary n_tokens (1 and n_ubatch) under WHATEVER phase is currently set - every
    // phase below must therefore produce a buildable, shape-consistent graph for ANY
    // n_tokens. WARM clamps its store write; BLOCK falls back to a DECODE-shaped graph
    // when the batch cannot be a true block (s + n_tokens > L).
    const bool is_warm     = phase == llama_diffusion_pkv::WARM   && pkv_P > 0;
    const bool want_block  = phase == llama_diffusion_pkv::BLOCK  && pkv_L > 0;
    const bool is_block    = want_block && pkv_s + (int64_t) n_tokens <= pkv_L;
    const bool is_decode   = (phase == llama_diffusion_pkv::DECODE && pkv_P > 0) ||
                             (want_block && !is_block && pkv_s > 0);
    const int64_t P_decode = (phase == llama_diffusion_pkv::DECODE) ? pkv_P : pkv_s;

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    // phase-marker input: keys graph reuse on the pkv state (see diffusion-common.h) -
    // REQUIRED for correctness (same-shape graphs across phase changes must not be
    // reused) and for performance (same-phase same-shape steps within a block MUST be)
    res->add_input(std::make_unique<llm_graph_input_diffusion_phase>(&dmodel.pkv));

    llm_graph_input_attn_no_cache * inp_attn = nullptr;
    if (is_decode || is_block) {
        // rectangular mask [n_kv, n_tokens]; Dream is fully bidirectional (all-allow)
        const int64_t n_kv = is_block ? pkv_L : P_decode + n_tokens;
        auto uptr = std::make_unique<llm_graph_input_attn_diffusion_decode>(
                        hparams, cparams, n_kv - n_tokens, n_tokens);
        uptr->allow_reuse = true;  // pkv-state changes are guarded by the phase marker
        const auto type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;
        uptr->self_kq_mask = ggml_new_tensor_4d(ctx0, type_mask, n_kv, n_tokens, 1, 1);
        ggml_set_input(uptr->self_kq_mask);
        uptr->self_kq_mask_cnv = uptr->self_kq_mask;
        inp_attn = (llm_graph_input_attn_no_cache *) res->add_input(std::move(uptr));
    } else {
        inp_attn = build_attn_inp_no_cache();
    }

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
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

            ggml_tensor * Katt = Kcur;
            ggml_tensor * Vatt = Vcur;

            if (is_warm) {
                // store the first P rows of the roped K/V for the coming DECODE/BLOCK steps
                // (sub-view -> cpy precedent: diffusion-gemma.cpp DG_DUMP_KV_LAYER path);
                // clamped so scheduler reserve graphs (arbitrary n_tokens) still build
                const int64_t Pw = std::min<int64_t>(pkv_P, n_tokens);
                if (Pw > 0) {
                    ggml_tensor * kP = ggml_view_3d(ctx0, Kcur, n_embd_head, n_head_kv, Pw,
                                                    Kcur->nb[1], Kcur->nb[2], 0);
                    ggml_tensor * vP = ggml_view_3d(ctx0, Vcur, n_embd_head, n_head_kv, Pw,
                                                    Vcur->nb[1], Vcur->nb[2], 0);
                    ggml_tensor * sk = ggml_view_3d(ctx0, dmodel.pkv.k[il], n_embd_head, n_head_kv, Pw,
                                                    dmodel.pkv.k[il]->nb[1], dmodel.pkv.k[il]->nb[2], 0);
                    ggml_tensor * sv = ggml_view_3d(ctx0, dmodel.pkv.v[il], n_embd_head, n_head_kv, Pw,
                                                    dmodel.pkv.v[il]->nb[1], dmodel.pkv.v[il]->nb[2], 0);
                    ggml_build_forward_expand(gf, ggml_cpy(ctx0, kP, sk));
                    ggml_build_forward_expand(gf, ggml_cpy(ctx0, vP, sv));
                }
            } else if (is_decode) {
                // prefix cache: attend [store(0..P) | fresh suffix]
                ggml_tensor * pk = ggml_view_3d(ctx0, dmodel.pkv.k[il], n_embd_head, n_head_kv, P_decode,
                                                dmodel.pkv.k[il]->nb[1], dmodel.pkv.k[il]->nb[2], 0);
                ggml_tensor * pv = ggml_view_3d(ctx0, dmodel.pkv.v[il], n_embd_head, n_head_kv, P_decode,
                                                dmodel.pkv.v[il]->nb[1], dmodel.pkv.v[il]->nb[2], 0);
                ggml_tensor * Kc = Kcur;
                ggml_tensor * Vc = Vcur;
                if (dmodel.pkv.k[il]->type != Kcur->type) {  // F16 store: match types for concat
                    Kc = ggml_cast(ctx0, Kcur, dmodel.pkv.k[il]->type);
                    Vc = ggml_cast(ctx0, Vcur, dmodel.pkv.v[il]->type);
                }
                Katt = ggml_concat(ctx0, pk, Kc, 2);
                Vatt = ggml_concat(ctx0, pv, Vc, 2);
            } else if (is_block) {
                // dual cache: write the block rows into the store for the NEXT step, and
                // attend the 3-way concat [store(0..s) | fresh block | store(s+C..L)] -
                // the read never aliases the written region (guide sec 9c)
                const size_t nb2k = dmodel.pkv.k[il]->nb[2];
                const size_t nb2v = dmodel.pkv.v[il]->nb[2];
                ggml_tensor * dk = ggml_view_3d(ctx0, dmodel.pkv.k[il], n_embd_head, n_head_kv, n_tokens,
                                                dmodel.pkv.k[il]->nb[1], nb2k, (size_t) pkv_s * nb2k);
                ggml_tensor * dv = ggml_view_3d(ctx0, dmodel.pkv.v[il], n_embd_head, n_head_kv, n_tokens,
                                                dmodel.pkv.v[il]->nb[1], nb2v, (size_t) pkv_s * nb2v);
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, dk));
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, dv));

                Katt = Kcur;
                Vatt = Vcur;
                if (dmodel.pkv.k[il]->type != Kcur->type) {  // F16 store
                    Katt = ggml_cast(ctx0, Kcur, dmodel.pkv.k[il]->type);
                    Vatt = ggml_cast(ctx0, Vcur, dmodel.pkv.v[il]->type);
                }
                if (pkv_s > 0) {
                    ggml_tensor * prek = ggml_view_3d(ctx0, dmodel.pkv.k[il], n_embd_head, n_head_kv, pkv_s,
                                                      dmodel.pkv.k[il]->nb[1], nb2k, 0);
                    ggml_tensor * prev = ggml_view_3d(ctx0, dmodel.pkv.v[il], n_embd_head, n_head_kv, pkv_s,
                                                      dmodel.pkv.v[il]->nb[1], nb2v, 0);
                    Katt = ggml_concat(ctx0, prek, Katt, 2);
                    Vatt = ggml_concat(ctx0, prev, Vatt, 2);
                }
                const int64_t post = pkv_L - (pkv_s + n_tokens);
                if (post > 0) {
                    ggml_tensor * postk = ggml_view_3d(ctx0, dmodel.pkv.k[il], n_embd_head, n_head_kv, post,
                                                       dmodel.pkv.k[il]->nb[1], nb2k, (size_t) (pkv_s + n_tokens) * nb2k);
                    ggml_tensor * postv = ggml_view_3d(ctx0, dmodel.pkv.v[il], n_embd_head, n_head_kv, post,
                                                       dmodel.pkv.v[il]->nb[1], nb2v, (size_t) (pkv_s + n_tokens) * nb2v);
                    Katt = ggml_concat(ctx0, Katt, postk, 2);
                    Vatt = ggml_concat(ctx0, Vatt, postv, 2);
                }
            }

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Katt, Vatt, nullptr, nullptr, nullptr, 1.0f / sqrtf(float(n_embd_head)), il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
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

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
