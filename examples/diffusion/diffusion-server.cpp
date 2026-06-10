// llama-diffusion-server: a general HTTP server for diffusion models (masked diffusion like
// Dream/LLaDA/DiffuCoder and canvas block-diffusion like DiffusionGemma) wrapping the same
// generation code as llama-diffusion-cli. The model is loaded once; requests are serialized
// (the GPU is a serial resource and canvas models keep per-request state on the model).
//
//   GET  /health     -> model + capability report (family, mask piece, canvas/eb defaults)
//   POST /tokenize   {"content": str, "add_special"?: bool, "parse_special"?: bool} -> {"tokens": [...]}
//   POST /detokenize {"tokens": [...]} -> {"content": str}
//   POST /generate   see handle_generate() below
//
// CLI flags (all llama-diffusion-cli flags work) set the per-request DEFAULTS; request body
// fields override them per call.

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "diffusion.h"
#include "llama.h"
#include "log.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

struct server_state {
    common_params       params;
    llama_model *       model = nullptr;
    llama_context *     ctx   = nullptr;
    const llama_vocab * vocab = nullptr;

    llama_token mask_token_id = LLAMA_TOKEN_NULL;
    std::string mask_piece;

    int64_t canvas_length = 0;     // > 0 -> canvas/block-diffusion model (entropy-bound path)
    bool    shift_logits  = false;

    diffusion_eb_params eb_defaults;  // resolved from GGUF metadata + CLI overrides (canvas models)

    common_chat_templates_ptr chat_templates;

    std::mutex mutex;  // one generation at a time
};

static std::string meta_str(const llama_model * model, const char * key) {
    char buf[128];
    if (llama_model_meta_val_str(model, key, buf, sizeof(buf)) >= 0) {
        return buf;
    }
    return "";
}

static float meta_f(const llama_model * model, const char * key, float def) {
    const std::string s = meta_str(model, key);
    return s.empty() ? def : strtof(s.c_str(), nullptr);
}

static int32_t meta_i(const llama_model * model, const char * key, int32_t def) {
    const std::string s = meta_str(model, key);
    return s.empty() ? def : (int32_t) strtol(s.c_str(), nullptr, 10);
}

// build the per-request masked-diffusion params from server defaults + request overrides
static diffusion_params make_masked_params(const server_state & st, const json & req) {
    const common_params & p = st.params;

    diffusion_params dp;
    dp.mask_token_id    = st.mask_token_id;
    dp.shift_logits     = st.shift_logits;
    dp.seed             = req.value("seed",           (int32_t) p.sampling.seed);
    dp.temperature      = req.value("temp",           p.sampling.temp);
    dp.steps            = req.value("steps",          p.diffusion.steps);
    dp.algorithm        = (diffusion_algorithm) req.value("algorithm", p.diffusion.algorithm);
    dp.top_p            = req.value("top_p",          p.sampling.top_p);
    dp.top_k            = req.value("top_k",          p.sampling.top_k);
    dp.add_gumbel_noise = req.value("add_gumbel_noise", p.diffusion.add_gumbel_noise);
    dp.cfg_scale        = req.value("cfg_scale",      p.diffusion.cfg_scale);
    dp.alg_temp         = req.value("alg_temp",       p.diffusion.alg_temp);
    dp.conf_threshold   = req.value("conf_threshold", p.diffusion.conf_threshold);
    dp.backend_sampling = req.value("backend_sampling", p.sampling.backend_sampling);

    const float   eps          = req.value("eps",          p.diffusion.eps);
    const int32_t block_length = req.value("block_length", p.diffusion.block_length);
    if (block_length > 0) {
        dp.schedule     = DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED;
        dp.block_length = block_length;
    } else {
        dp.schedule = DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED;
        dp.eps      = eps > 0 ? eps : 1e-3f;
    }
    return dp;
}

// POST /generate
//
// request:  {"prompt": str (required),
//            "raw"?: bool (skip the chat template; default false),
//            "infill"?: bool (prompt is a canvas with mask-piece markers; masked models only),
//            "n_predict"?: int (canvas models: drives the block count),
//            "return_confidences"?: bool,
//            "seed"?, "steps"?, "conf_threshold"?, "algorithm"?, "temp"?, "top_k"?, "top_p"?,
//            "eps"?, "block_length"?, "alg_temp"?, "cfg_scale"?, "add_gumbel_noise"?,
//            "backend_sampling"?,
//            "eb"?: {"max_steps"?, "t_min"?, "t_max"?, "entropy_bound"?, "stability"?,
//                    "confidence"?, "kv_cache"?}}
// response: {"text": str, "family": "masked"|"canvas", "ms_total": float, "degenerate": bool,
//            "n_prompt_tokens": int, "confidences"?: [float per generated position]}
static json handle_generate(server_state & st, const json & req) {
    const std::string prompt = req.at("prompt").get<std::string>();
    const bool        raw    = req.value("raw", false);
    const bool        infill = req.value("infill", false);
    const bool        want_conf = req.value("return_confidences", false);

    if (infill && st.canvas_length > 0) {
        throw std::runtime_error("infill is not supported for canvas models yet");
    }

    // chat-format unless raw/infill
    std::string formatted = prompt;
    if (!raw && !infill && st.chat_templates) {
        common_chat_templates_inputs inputs;
        common_chat_msg msg;
        const std::string system = req.value("system", st.params.system_prompt);
        if (!system.empty()) {
            common_chat_msg s;
            s.role    = "system";
            s.content = system;
            inputs.messages.push_back(s);
        }
        msg.role    = "user";
        msg.content = prompt;
        inputs.messages.push_back(msg);
        inputs.add_generation_prompt = true;
        formatted = common_chat_templates_apply(st.chat_templates.get(), inputs).prompt;
    }

    std::vector<llama_token> prefix = common_tokenize(st.vocab, formatted,
                                                      /*add special*/ true, /*parse special*/ true);
    const int32_t n_input = (int32_t) prefix.size();
    const int32_t n_ub    = (int32_t) st.params.n_ubatch;

    if (n_input >= (int32_t) llama_n_ctx(st.ctx) || (infill && n_input > n_ub)) {
        throw std::runtime_error("prompt too long for the configured context/ubatch");
    }

    std::vector<llama_token> output_tokens(n_ub);
    std::vector<float>       confidences;
    bool                     degenerate = false;

    json res;
    const int64_t t0 = ggml_time_us();

    if (st.canvas_length <= 0) {
        // masked diffusion (Dream/LLaDA/DiffuCoder)
        diffusion_params dp = make_masked_params(st, req);
        dp.infill     = infill;
        dp.max_length = infill ? n_input : n_ub;
        if (want_conf) {
            confidences.assign(dp.max_length, -1.0f);
            dp.output_confidences = confidences.data();
        }
        dp.out_degenerate = &degenerate;

        int32_t n_generated = 0;
        diffusion_generate(st.ctx, prefix.data(), output_tokens.data(), n_input, dp, n_generated);
        if (n_generated <= (infill ? 0 : n_input)) {
            throw std::runtime_error("generation failed");
        }

        if (infill) {
            output_tokens.resize(n_generated);
            res["text"] = common_detokenize(st.vocab, output_tokens, false);
        } else {
            res["text"] = common_detokenize(st.vocab,
                std::vector<llama_token>(output_tokens.begin() + n_input, output_tokens.begin() + n_generated), false);
        }
        if (want_conf) {
            // report the generated region only (whole canvas for infill)
            confidences.erase(confidences.begin(), confidences.begin() + (infill ? 0 : n_input));
            res["confidences"] = confidences;
        }
        res["family"] = "masked";
    } else {
        // canvas block-diffusion (DiffusionGemma): entropy-bound denoise, block-autoregressive
        diffusion_eb_params eb = st.eb_defaults;
        if (req.contains("eb")) {
            const json & e = req["eb"];
            eb.max_denoising_steps  = e.value("max_steps",     eb.max_denoising_steps);
            eb.t_min                = e.value("t_min",         eb.t_min);
            eb.t_max                = e.value("t_max",         eb.t_max);
            eb.entropy_bound        = e.value("entropy_bound", eb.entropy_bound);
            eb.stability_threshold  = e.value("stability",     eb.stability_threshold);
            eb.confidence_threshold = e.value("confidence",    eb.confidence_threshold);
            eb.kv_cache             = e.value("kv_cache",      eb.kv_cache);
        }
        eb.seed = req.value("seed", (int32_t) st.params.sampling.seed);

        const int32_t cl        = (int32_t) st.canvas_length;
        const int32_t n_predict = req.value("n_predict", cl);
        const int     n_blocks  = std::max(1, (n_predict + cl - 1) / cl);

        // trim a denoised canvas at the first end token or a repetition loop (mirrors the CLI)
        auto trim_canvas = [&](const llama_token * canvas, size_t n) -> size_t {
            size_t cut = n;
            for (size_t i = 0; i < n; i++) {
                if (llama_vocab_is_eog(st.vocab, canvas[i])) { cut = i; break; }
            }
            for (size_t i = 0; i + 1 < cut; i++) {
                bool loop = false;
                for (size_t stride = 1; stride <= 2 && !loop; stride++) {
                    size_t reps = 0;
                    for (size_t j = i; j + stride < n && canvas[j] == canvas[j + stride]; j += stride) { reps++; }
                    loop = reps >= 6;
                }
                if (loop) { cut = i; break; }
            }
            return cut;
        };

        std::vector<llama_token> response_tokens;
        for (int b = 0; b < n_blocks; b++) {
            const int32_t prefix_len = (int32_t) prefix.size();
            const int32_t max_length = prefix_len + cl;
            if (max_length > n_ub) {
                if (b == 0) {
                    throw std::runtime_error("prompt + canvas exceeds n_ubatch; raise -ub/-c");
                }
                break;
            }
            eb.max_length = max_length;
            if (want_conf) {
                confidences.assign(max_length, -1.0f);
                eb.output_confidences = confidences.data();
            }

            int32_t n_generated = 0;
            diffusion_generate_entropy_bound(st.ctx, prefix.data(), output_tokens.data(), prefix_len, eb, n_generated);
            if (n_generated <= prefix_len) {
                if (b == 0) {
                    throw std::runtime_error("generation failed");
                }
                break;
            }

            const llama_token * canvas = output_tokens.data() + prefix_len;
            const size_t        cut    = trim_canvas(canvas, (size_t) cl);
            response_tokens.insert(response_tokens.end(), canvas, canvas + cut);
            if (cut < (size_t) cl) {
                break;
            }
            prefix.insert(prefix.end(), canvas, canvas + cut);
        }

        res["text"]   = common_detokenize(st.vocab, response_tokens, false);
        res["family"] = "canvas";
        if (want_conf && !confidences.empty()) {
            // entropies of the FINAL block's canvas (lower = more confident)
            confidences.erase(confidences.begin(), confidences.begin() + ((int32_t) prefix.size() - (int32_t) st.canvas_length < 0 ? 0 : (int32_t) confidences.size() - (int32_t) st.canvas_length));
            res["confidences"]     = confidences;
            res["confidence_kind"] = "entropy";
        }
    }

    res["ms_total"]        = (ggml_time_us() - t0) / 1000.0;
    res["degenerate"]      = degenerate;
    res["n_prompt_tokens"] = n_input;
    return res;
}

int main(int argc, char ** argv) {
    ggml_time_init();

    server_state st;

    common_init();

    st.params.sampling.backend_sampling = true;  // same default as llama-diffusion-cli

    if (!common_params_parse(argc, argv, st.params, LLAMA_EXAMPLE_DIFFUSION)) {
        return 1;
    }

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers       = st.params.n_gpu_layers;
    model_params.devices            = st.params.devices.data();
    model_params.use_mmap           = st.params.use_mmap;
    model_params.use_direct_io      = st.params.use_direct_io;
    model_params.use_mlock          = st.params.use_mlock;
    model_params.check_tensors      = st.params.check_tensors;
    if (!st.params.tensor_buft_overrides.empty()) {
        GGML_ASSERT(st.params.tensor_buft_overrides.back().pattern == nullptr);
        model_params.tensor_buft_overrides = st.params.tensor_buft_overrides.data();
    }

    st.model = llama_model_load_from_file(st.params.model.path.c_str(), model_params);
    if (!st.model) {
        LOG_ERR("error: failed to load model '%s'\n", st.params.model.path.c_str());
        return 1;
    }
    if (!llama_model_is_diffusion(st.model)) {
        LOG_ERR("error: not a diffusion model\n");
        return 1;
    }

    st.vocab         = llama_model_get_vocab(st.model);
    st.canvas_length = meta_i(st.model, "diffusion.canvas_length", 0);
    st.mask_token_id = llama_vocab_mask(st.vocab);

    if (st.mask_token_id != LLAMA_TOKEN_NULL) {
        char piece[64];
        const int n = llama_token_to_piece(st.vocab, st.mask_token_id, piece, sizeof(piece) - 1, 0, true);
        if (n > 0) {
            piece[n] = '\0';
            st.mask_piece = piece;
        }
    } else if (st.canvas_length <= 0) {
        LOG_ERR("error: masked-diffusion model without a mask token\n");
        return 1;
    }

    // canvas models self-condition: enable before context creation so the reserve covers it
    if (st.canvas_length > 0) {
        llama_diffusion_set_sc(st.model, nullptr, 0.0f, 1.0f, true);
    }

    // size the context to fit [prompt | canvas] for canvas models (mirrors the CLI's -n logic)
    if (st.canvas_length > 0) {
        const int32_t needed = (int32_t) st.canvas_length + 2048;
        st.params.n_ubatch = std::max((int32_t) st.params.n_ubatch, needed);
        st.params.n_batch  = std::max((int32_t) st.params.n_batch,  st.params.n_ubatch);
        st.params.n_ctx    = std::max((int32_t) st.params.n_ctx,    st.params.n_batch);
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = st.params.n_ctx;
    ctx_params.n_batch         = st.params.n_batch;
    ctx_params.n_ubatch        = st.params.n_ubatch;
    ctx_params.flash_attn_type = st.params.flash_attn_type;
    ctx_params.no_perf         = st.params.no_perf;
    ctx_params.type_k          = st.params.cache_type_k;
    ctx_params.type_v          = st.params.cache_type_v;

    st.ctx = llama_init_from_model(st.model, ctx_params);
    if (!st.ctx) {
        LOG_ERR("error: failed to create context\n");
        return 1;
    }
    llama_set_n_threads(st.ctx, st.params.cpuparams.n_threads, st.params.cpuparams_batch.n_threads);

    st.shift_logits = meta_str(st.model, "diffusion.shift_logits") == "true" ||
                      (meta_str(st.model, "diffusion.shift_logits").empty() && st.canvas_length == 0);

    if (st.canvas_length > 0) {
        st.eb_defaults.max_denoising_steps  = meta_i(st.model, "diffusion.eb_max_steps", 48);
        st.eb_defaults.t_min                = meta_f(st.model, "diffusion.eb_t_min", 0.4f);
        st.eb_defaults.t_max                = meta_f(st.model, "diffusion.eb_t_max", 0.8f);
        st.eb_defaults.entropy_bound        = meta_f(st.model, "diffusion.eb_entropy_bound", 0.1f);
        st.eb_defaults.stability_threshold  = meta_i(st.model, "diffusion.eb_stability_threshold", 1);
        st.eb_defaults.confidence_threshold = meta_f(st.model, "diffusion.eb_confidence_threshold", 0.005f);
        st.eb_defaults.kv_cache             = st.params.diffusion.eb_kv_cache != 2;
    }

    st.chat_templates = common_chat_templates_init(st.model, "");

    httplib::Server svr;

    svr.set_exception_handler([](const httplib::Request &, httplib::Response & res, const std::exception_ptr & ep) {
        std::string msg = "unknown error";
        try {
            if (ep) { std::rethrow_exception(ep); }
        } catch (const std::exception & e) {
            msg = e.what();
        }
        res.status = 500;
        res.set_content(json{{"error", msg}}.dump(), "application/json");
    });

    svr.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        json j = {
            {"status",        "ok"},
            {"model",         st.params.model.path},
            {"family",        st.canvas_length > 0 ? "canvas" : "masked"},
            {"mask_token_id", st.mask_token_id},
            {"mask_piece",    st.mask_piece},
            {"canvas_length", st.canvas_length},
            {"n_ctx",         (int) llama_n_ctx(st.ctx)},
            {"n_ubatch",      (int) st.params.n_ubatch},
        };
        if (st.canvas_length > 0) {
            j["eb_defaults"] = {
                {"max_steps",     st.eb_defaults.max_denoising_steps},
                {"t_min",         st.eb_defaults.t_min},
                {"t_max",         st.eb_defaults.t_max},
                {"entropy_bound", st.eb_defaults.entropy_bound},
                {"stability",     st.eb_defaults.stability_threshold},
                {"confidence",    st.eb_defaults.confidence_threshold},
                {"kv_cache",      st.eb_defaults.kv_cache},
            };
        }
        res.set_content(j.dump(), "application/json");
    });

    svr.Post("/tokenize", [&](const httplib::Request & req, httplib::Response & res) {
        const json body = json::parse(req.body);
        const std::vector<llama_token> tokens = common_tokenize(st.vocab,
            body.at("content").get<std::string>(),
            body.value("add_special", false),
            body.value("parse_special", true));
        res.set_content(json{{"tokens", tokens}, {"n", tokens.size()}}.dump(), "application/json");
    });

    svr.Post("/detokenize", [&](const httplib::Request & req, httplib::Response & res) {
        const json body = json::parse(req.body);
        const std::vector<llama_token> tokens = body.at("tokens").get<std::vector<llama_token>>();
        res.set_content(json{{"content", common_detokenize(st.vocab, tokens, false)}}.dump(), "application/json");
    });

    svr.Post("/generate", [&](const httplib::Request & req, httplib::Response & res) {
        std::lock_guard<std::mutex> lock(st.mutex);
        const json body = json::parse(req.body);
        res.set_content(handle_generate(st, body).dump(), "application/json");
    });

    const std::string host = st.params.hostname.empty() ? "127.0.0.1" : st.params.hostname;
    const int         port = st.params.port > 0 ? st.params.port : 8089;

    LOG_INF("diffusion-server: %s model, listening on http://%s:%d\n",
            st.canvas_length > 0 ? "canvas" : "masked", host.c_str(), port);

    svr.listen(host, port);

    llama_free(st.ctx);
    llama_model_free(st.model);
    llama_backend_free();
    return 0;
}
