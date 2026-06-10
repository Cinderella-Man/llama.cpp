# DiffusionGemma (2026-06-10) - research + llama.cpp support plan

Sources: Google announcement + developer guide (fetched), HF model card + config.json +
tokenizer + safetensors index for `google/diffusiongemma-26B-A4B-it`, llama.cpp PR #24423
raw diff, Unsloth GGUF card, local fork code. Facts below are quoted from those; inferences
are marked.

## 1. What it is

- 26B MoE (25.8B total, 3.8B active per token), Gemma 4 backbone
  (`DiffusionGemmaForBlockDiffusion`, model_type `diffusion_gemma`). Apache 2.0, weights
  public: `google/diffusiongemma-26B-A4B-it` (51.6 GB BF16, 11 shards).
- Config highlights: 30 layers, hidden 2816, 16 heads / 8 KV (2 global KV), head_dim 256
  (global 512), experts 128 top-8 (moe_intermediate 704) PLUS a shared dense mlp per layer,
  vocab 262144, context 262144, sliding window 1024 with 5:1 sliding/full layer pattern,
  dual rope (full: theta 1e6 "proportional" partial_rotary 0.25; sliding: theta 1e4),
  final_logit_softcapping 30, canvas_length 256. Multimodal (vision tower in the safetensors;
  text-only is the inference target for now).
- Claims: "up to 4x faster generation", "1000+ tok/s on H100, 700+ on RTX 5090", quality
  "lower than standard Gemma 4". Day-one support: transformers, vLLM (via transformers
  backend), MLX, Unsloth; "llama.cpp (coming soon)".
- Unsloth already published GGUFs: `unsloth/diffusiongemma-26B-A4B-it-GGUF` - Q4_K_M 16 GB,
  Q5_K_M 18 GB, Q6_K 21 GB, Q8_0 25 GB, BF16 47 GB.

## 2. How it works (and how it differs from Dream/LLaDA)

UNIFORM-STATE DIFFUSION (USD), not masked diffusion:
- Corruption replaces tokens with RANDOM vocabulary tokens, not a [MASK] token. The canvas
  starts as 256 random tokens (`std::uniform_int_distribution(0, n_vocab-1)` in PR #24423)
  and is iteratively refined. No mask token participates in the diffusion process (the
  Gemma vocab does contain a `<mask>` token, but USD does not use it).
- Research basis: The Diffusion Duality (arXiv 2506.10892 - USD emerges from Gaussian
  diffusion via argmax; enables few-step generation), Block Diffusion (arXiv 2503.09573).
- Self-correction is native: ANY token can be re-noised (replaced with a fresh random
  token) if confidence drops - this is RemeDi-style remasking, trained in.

BLOCK-AUTOREGRESSIVE with a real KV cache:
- Prefill: prompt is ingested with CAUSAL attention, writing a KV cache (one pass).
- Denoise: the 256-token canvas attends bidirectionally to itself AND to the cached
  prompt; per PR #24423 the mask is "prompt queries causal over prompt; canvas queries
  bidirectional", with SWA clipping canvas->prompt reach to the last n_swa-1 tokens in
  sliding layers.
- Commit: the accepted canvas becomes prefix (written to cache), next 256-token block
  starts. Variable-length generation beyond 256 = repeat.
- This is EXACTLY the prefix-cache pattern from docs/dllm-engine-improvements.md item 5
  (PR-17454 style), but TRAINED for it - no approximation error.

ENTROPY-BOUND SAMPLER (per PR #24423 + dev guide):
- Per step: softmax(logits/t) with a temperature schedule (t_max -> t_min), multinomial
  sample per position; accept the lowest-entropy positions while cumulative entropy <=
  entropy_bound (default 0.1); RE-NOISE rejected positions with fresh random tokens;
  adaptive stop when argmax is stable for stability_threshold steps AND mean entropy <
  confidence_threshold. Max ~48 steps per 256-token canvas.
- Self-conditioning: each step's canvas logits feed back into the next step
  (`llama_diffusion_set_sc`, temp_inv = 1/t).
- This is our `--diffusion-conf-threshold` made rigorous and TRAINED-IN (the quality cliff
  we measured on Dream should not exist here).

## 3. Ecosystem state (verified 2026-06-10)

- llama.cpp support EXISTS AS A SAME-DAY DRAFT: PR #24423 by danielhanchen (Unsloth),
  disclosed "Heavy usage of AI, but verified logits matching with transformers". It adds:
  `LLM_ARCH_DIFFUSION_GEMMA` ("diffusion-gemma"), src/models/diffusion-gemma.cpp, a new
  region-aware mask input class `llm_graph_input_attn_diffusion`, phase API
  `llama_diffusion_set_phase(model, UNIFIED|PREFILL|DECODE, P)`, conversion
  (conversion/diffusion_gemma.py), GGUF keys `diffusion.canvas_length`,
  `diffusion.eb_max_steps`, `diffusion.eb_t_min/t_max`, `diffusion.eb_entropy_bound`,
  `diffusion.eb_stability_threshold`, `diffusion.eb_confidence_threshold`, a
  `diffusion_generate_entropy_bound()` in examples/diffusion, plus TWO model-specific new
  examples (diffusion-gemma-eval, diffusion-gemma-server).
- Reviewer friction already: pwilkin - "a ton of debugging stuff left in there" and
  questioning "the idea to make a server just for one model", asking for "a general
  diffusion-server" instead. Expect revision rounds before merge.
- Default path today: unified no-cache forward over [prompt | canvas] every step; the
  prefix-KV-cache path is opt-in.
- Our fork's base already contains the full Gemma 4 MoE backbone:
  src/models/gemma4.cpp loads shared mlp + 128-expert MoE (ffn_gate_inp router,
  gate_up_exps, per-expert scale; lines 93-122), SWA pattern, dual rope, softcapping -
  the architecture work is mostly the diffusion mask + loop, not the backbone.

## 4. What this changes for us

1. The kintsugi engine target upgrade path: DiffusionGemma vs Dream-7B = 256K context (vs
   2048!), trained parallel decoding (no threshold quality cliff), native KV-cached blocks
   (the seq-length cost problem solved by training, not approximation), Apache 2.0, and a
   much stronger backbone - at the price of 16 GB Q4_K_M.
2. Hardware: does NOT fit the 8 GB laptop VRAM. Options: `--n-cpu-moe` (experts in RAM,
   attention+shared on GPU - 3.8B active params make this plausible; 65 GB RAM is plenty),
   or the 2x3090 box (fits Q8_0 fully). Benchmark before committing the harness to it.
3. Infill semantics change (harness Masker impact): no mask token in the diffusion
   process. A hole = positions initialized to RANDOM tokens and left uncommitted, fixed
   text = committed positions. The `<|mask|>`-marker UX can be PRESERVED by having the CLI
   translate marker positions into random-init/uncommitted positions (markers never reach
   the model). Re-noising gives a natively-trained "repair these spans" operation.
4. Our improvements doc gets validated: item 5 (prefix KV cache) is how this model WANTS
   to run; items 2/3 (general diffusion server + confidence export) are what reviewers are
   asking the draft PR to become.
5. MoE caveat: MUL_MAT_ID at canvas batch sizes likely disables CUDA graphs
   (ggml-cuda.cu:3316-3327 gating) - the battery-mode graph win may not apply; measure.

## 5. Support plan

### Path A - test the draft PR on the fork (days, do first)
1. Fetch PR branch: `git fetch origin pull/24423/head:diffusiongemma-pr` and merge onto the
   fork. EXPECT CONFLICTS in examples/diffusion/* (the PR rewrites diffusion-cli for
   multi-turn + entropy-bound; our fork rewrote the same files for backend sampling +
   threshold + infill). Resolution strategy: keep our diffusion.cpp loop, graft their
   `diffusion_generate_entropy_bound()` alongside as a new schedule/algorithm; keep their
   src/ + conversion changes verbatim.
2. Download `unsloth/diffusiongemma-26B-A4B-it-GGUF` Q4_K_M (16 GB); run with
   `--n-cpu-moe` variants on the laptop; measure tok/s + quality on Elixir prompts vs
   Dream-7B; same on the 3090 box later.
3. Verify our backend (GPU) sampling against it: canvas decode produces 256 output rows -
   our multi-row sampling applies directly IF the entropy-bound sampler is expressed via
   the sampler-chain confidence machinery; the PR's sampler is CPU-side (full-logits D2H
   again) - measure the difference.

### Path B - the contribution opening (maintainer-aligned, human-authored per AGENTS.md)
The draft PR's weaknesses are exactly our fork's strengths:
- General diffusion-server (pwilkin's explicit ask) = docs/dllm-engine-improvements.md
  item 2 design, generalized to serve dream/llada AND diffusion-gemma.
- GPU sampling for the entropy-bound loop: per-position entropy IS our confidence
  machinery (entropy algorithm + de-tempered thresholding already in the fork); moving
  the PR's CPU sampler onto the backend-sampling path removes the 256-row x 262144-vocab
  logits D2H per step (262144 vocab x 256 rows x 4 B = 256 MB/step!).
- The eb_* GGUF keys map cleanly onto generalized `--diffusion-*` params (the LLaDA-2.0
  review precedent: general params, not model-specific code).
If pursued: coordinate on the PR thread first; contributions must be human-authored.

### Path C - independent fork support (if the PR stalls)
Implement in our fork in this order (most pieces verified feasible this session):
1. Conversion + arch: take the PR's conversion/diffusion_gemma.py + arch tables
   (mechanical); model graph = gemma4.cpp + region-aware mask.
2. Mask: the PR's `llm_graph_input_attn_diffusion` approach is right; alternatively the
   UNIFIED phase works today via build_attn_inp_no_cache + causal toggling per region
   (our verified kv-mask causal=false supports the cached path - llama-kv-cache.cpp:
   1676-1713).
3. Loop: add a `uniform_state` mode to examples/diffusion: committed[] bitmap instead of
   mask-token sentinel, random canvas init, entropy-bound accept + re-noise + adaptive
   stop (all maps onto our existing confidence/threshold code), block commit via
   llama_memory_seq_rm/prefill (engine-improvements item 5 mechanics).
4. GPU sampling: entropy per position via our backend probs path (top-k first to bound
   the 262144-vocab readback; verify entropy over top-k approximates the full-vocab
   entropy well enough for the bound - validate against transformers).
5. Infill: marker translation to uncommitted positions (keeps the kintsugi Masker
   interface unchanged).

## 6. Open questions / verify-at-build-time

- Per-expert scale / router details: gemma4.cpp loads per-expert scale; confirm the PR
  conversion maps `router.per_expert_scale` + `layer_scalar` + the doubled pre/post-ffn
  norms (the HF index shows pre_feedforward_layernorm_2, post_feedforward_layernorm_1/2 -
  more norms than dense Gemma 4; gemma4.cpp must already handle the A4B variant since
  upstream supports gemma4 MoE - verify tensor-by-tensor on first conversion).
- Whether entropy-over-top-k suffices for the entropy_bound (vs full-vocab entropy).
- Self-conditioning (`llama_diffusion_set_sc`) - whether the fork adopts it (extra input
  tensor per step; the PR feeds previous logits back - graph implications for reuse).
- CUDA-graph eligibility with MoE at 256-token canvas (measure GGML_CUDA_DISABLE_GRAPHS
  A/B once running).
- Quality on Elixir vs Dream-7B/DiffuCoder - the kintsugi benchmark decides the default
  engine model.
