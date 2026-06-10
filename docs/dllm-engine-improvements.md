# Engine improvements for the kintsugi vision - deep research + implementation plans


Every claim below was verified by reading code (file:line), running experiments on this
machine, or fetching primary sources. Corrections to the original ranking are called out.

## Headline corrections from this research (updated after AC-power re-measurement)

1. BATCHED MULTI-CANDIDATE GENERATION IS NOT WORTH BUILDING ON THIS GPU. The clean AC curve
   (llama-bench, r=3): pp512 = 2229 t/s (230 ms), pp1024 = 2128 t/s (481 ms), pp2048 =
   2123 t/s (965 ms) - throughput is FLAT from 512 to 2048, i.e. the RTX 5070 is already
   saturated at 512 tokens. 4x512-in-one-batch costs 965 ms vs 4x230 = 920 ms serial:
   batching is ~5% WORSE than serial (the dense 2048x2048 mask spends cross-sequence
   attention FLOPs for nothing). The original "4 drafts for 1.5x" premise (GPU
   underutilized at 512) is FALSE on this hardware. Recommendation changed: run candidates
   SERIALLY through the daemon queue; re-evaluate batching on the 2x3090 box only
   (measure its pp curve first - flat curve = don't build it there either).
2. CUDA graphs (item 6a) are already active, and their benefit is POWER-STATE DEPENDENT:
   battery 336 vs 506 ms/step (1.5x - the 1.5 GHz battery-clocked CPU is kernel-launch-bound
   and graphs collapse launches into one), AC 245.35 vs 246.19 ms/step (zero - the GPU is
   the bottleneck and launch latency hides behind the async queue). Nothing to implement;
   every future change must PRESERVE graph-replay compatibility, and the payoff case is
   exactly the unplugged-laptop regime the harness lives in.
3. llama-bench crashes on diffusion models when -p exceeds -ub (our encoder assert,
   llama-context.cpp:1543) - run with `-ub <p> -b <p>`.
4. gh CLI is not installed on this machine - GitHub research goes through web fetch.
5. The daemon needs NO engine changes at all: the sampler-attach re-reserve was MEASURED at
   ~11 ms via --verbose timestamps (the earlier ~1 s estimate was CUDA-graph warmup
   misattributed), and fresh per-request chains sidestep the one-way init assert. The
   daemon is now purely additive (~200 LOC HTTP tool).
6. The prefix-KV-cache route (item 5) is fully de-risked at the mask level: the stock
   KV-cache mask supports causal_attn=false natively (llama-kv-cache.cpp:1676-1713; only
   empty-cell and seq-membership masking applies) and llama_memory_seq_rm (llama.h:717)
   provides the per-step active-block clear.

---

## 1. Batched multi-candidate generation

STATUS AFTER AC MEASUREMENT: DROPPED for the RTX 5070 (see headline correction 1 - serial
drafts through the daemon are faster). The verified facts and plan below are kept because
they remain valid groundwork IF the 2x3090 pp curve shows headroom (throughput rising
from 512 to 2048), and because the multi-seq mechanics double as documentation of what the
multi-row sampling work already supports.

### Verified facts
- Cross-seq isolation: `llm_graph_input_attn_no_cache::set_input` (src/llama-graph.cpp:
  424-483) masks token i0 from i1 unless `seq_id[i0][0] == seq_id[i1][0]` (-INF otherwise);
  works for both F32 and the FA F16 mask path (type chosen at :2179).
- Order/contiguity: `split_simple` (src/llama-batch.cpp:474-506) preserves caller order, so
  seq0's 512 tokens then seq1's 512 give contiguous output-row ranges per seq - exactly what
  the multi-row sampling contiguity rule requires (build_seq_to_output_rows,
  llama-context.cpp:1349-1375; build_sampling, llama-graph.cpp:3043+).
- Positions are per-token (each seq can be 0..511 independently); `n_seq_max` default 1, max
  LLAMA_MAX_SEQ=256 (llama-cparams.h:7, llama-context.cpp:49-52) - set 4 at context init.
- Per-seq samplers: `llama_set_sampler(ctx, s, chain_s)` for s=0..3, each chain with its own
  dist seed -> each gets its own logit-row view + own RNG input (build_sampling loops the
  samplers map; input_sampling::set_input calls each chain's set_input).
- Readback: `output_ids[i] = i` in encode; `llama_get_sampled_token_ith(ctx, global_idx)`
  resolves any row; global_idx = s*max_length + pos.
- Memory at n_tokens=2048: kq mask 8 MB (F16); HOST pinned output buffers are the real cost:
  logits 1.2 GB + sampling.logits 1.2 GB + probs 1.2 GB + candidates 1.2 GB ~= 5 GB pinned
  (output_reserve, llama-context.cpp:2048-2155). Fine at 65 GB RAM; flag in docs. A future
  `n_outputs_max`-aware trim or skipping sampling.logits (unused by diffusion) could halve it.

### Implementation plan (example-level; no core changes needed)
1. diffusion.h: add `int32_t n_candidates = 1` (and per-candidate seeds derived seed+s).
2. diffusion.cpp `diffusion_generate`:
   - output_tokens becomes n_candidates x max_length (caller-provided flat array).
   - batch: n_tokens = n_candidates*max_length; token/pos/seq loops as researched
     (seq s tokens at offset s*max_length, pos 0..max_length-1, seq_id s).
   - per-seq backend chains (dist seed = params.seed + s); warmup probe checks row 0 of each
     seq (s*max_length); CPU fallback stays single-candidate (or loops seqs reading
     llama_get_logits_ith with global rows - cheap to support).
   - mask_positions/mask scan/commit/schedule state per seq (small structs); threshold and
     schedule logic unchanged per seq; a seq that finishes early simply has no masked
     positions (its rows are still computed - acceptable v1; see improvement 4).
   - get_row_for_pos(s, pos) = s*max_length + (shift_logits ? max(pos-1,0) : pos).
3. diffusion-cli: `--diffusion-candidates N`; context n_seq_max = N; n_ubatch >= N*max_length
   (validate; the encoder assert at llama-context.cpp:1543 is the failure mode).
4. CFG: excluded in batched mode (falls to single-candidate path), like the other fallbacks.
5. graph_reserve: our !memory reserve marks min(n_tokens, n_outputs) outputs with
   seq i/n_seq_tokens - already multi-seq correct (n_seqs = n_seq_max).

Effort: ~150-250 example lines. Risks: pinned-host growth (5 GB); per-step readback is
n_candidates x masked-count `_ith` calls (cheap, first call syncs); VRAM compute buffer at
2048 (measure; FA tiles 2048^2). CUDA-graph note: batch shape is constant across steps ->
graph replay preserved.
Measured expectation: 4 candidates per ~3.5x single-draft wall-clock (battery measurement;
re-measure on AC). Harness value: best-of-4 drafts -> pick first verified.

## 2. Daemon (resident model server)

### Verified facts (CORRECTED this round - measured, not inferred)
- THE RE-RESERVE COST CLAIM WAS WRONG: measured with --verbose timestamps, the
  sampler-attach-triggered `sched_reserve` (the second reserve, after context creation)
  runs 4.073.412 -> ~4.084 = **~11 ms**, not ~1 s. The earlier "1 s" was a misattribution
  of CUDA-graph warmup + first-step graph build seen in low-step-count runs. Same-shape
  re-reserves skip buffer reallocation (alloc only grows; llama-context.cpp output_reserve
  and ggml-alloc behavior) - hence the millisecond scale.
- Consequence: per-request chain RECREATION is perfectly viable. A FRESH chain per request
  sidesteps the one-way backend-init asserts entirely (llama-sampler.cpp:543-548 base init
  just sets {is_init, support}; chain init :714-743 probes each sampler once - both asserts
  only fire on REUSING an initialized chain, which a fresh chain never does). Cost per
  request: chain construction (~us) + set_sampler + 11 ms re-reserve.
- `llama_sampler_reset` on dist re-seeds from the ORIGINAL ctor seed (dist_reset,
  llama-sampler.cpp:1128-1132); seed field is `const` (:1043); no seed-change API exists -
  but with per-request fresh chains none is needed. `llama_sampler_dist_set_seed` remains a
  ~10 LOC optional micro-optimization (saves the 11 ms re-reserve per request).
- Per-call state in diffusion_generate today: CPU chains + backend chain + batch + scratch
  vectors are all allocated/attached/freed per call (diffusion.cpp:188-261) - with the 11 ms
  measurement, this existing structure is daemon-compatible AS IS; only the model/context
  load (2-4 s) needs hoisting.
- cpp-httplib vendored at vendor/cpp-httplib/httplib.h (0.46.1); tools/server links target
  `cpp-httplib`; usage pattern in tools/server/server-http.cpp:18-275 (`server_http_context::
  Impl` owns `std::unique_ptr<httplib::Server>`, set_exception_handler/set_error_handler/
  set_pre_routing_handler middleware, SO_REUSEADDR socket opts) - no minimal example exists,
  but the pattern is copyable.

### Implementation plan (SIMPLIFIED by the 11 ms finding)
1. NO core changes required. (Optional later: `llama_sampler_dist_set_seed` to shave the
   11 ms re-reserve and keep one long-lived chain.)
2. examples/diffusion: keep diffusion_generate as is (it already builds/attaches/frees per
   call, which is now known to be cheap); the daemon only hoists model + context creation.
3. New tool tools/diffusion-server/ (or examples/): cpp-httplib server, single worker mutex
   (GPU serial), endpoints:
   - GET /health -> {status, model, mask_piece (token_to_piece special=true), n_ubatch,
     n_candidates_max}
   - POST /generate -> body/response per the harness doc (docs/dllm-elixir-harness.md sec 4),
     plus `n_candidates`, `confidences` (improvement 3), `seed`.
   - POST /tokenize, /detokenize (improvement 3).
4. Graph-shape note: requests with the same max_length reuse the llama graph AND the CUDA
   graph; a different max_length rebuilds both (first 2 steps uncaptured) - document that the
   harness should quantize canvas sizes (e.g. 64/128/256/512) to maximize reuse.

Effort: ~200-250 LOC total (server ~200; no example split, no core API - both eliminated by
the 11 ms measurement). Risks: none structural; per-request determinism comes free with
fresh per-request chains (each constructed with the request's seed).

## 3. /tokenize + per-position confidences

### Verified facts
- `llama_tokenize` (include/llama.h:1134-1141) and `common_detokenize` (common/common.h:
  982-990) - trivial endpoint material.
- Confidences are computed per masked position per step in both sampling paths
  (diffusion.cpp backend ~:466-504, CPU ~:505-543) and consumed at three commit sites:
  threshold (:546-567), partial_sort (:572-588), alg_temp (:589-613). ORIGIN commits without
  confidences.
- Existing output plumbing: output_tokens in-place + `n_generated` out-ref; step callback
  receives tokens only (diffusion.h:21-25).

### Implementation plan
1. diffusion.h: `float * output_confidences = nullptr` field in diffusion_params (size
   max_length, caller-allocated; chosen over a signature change to keep the CLI untouched).
2. diffusion.cpp: init to -1.0f for all positions; at each of the 3 commit sites record
   `output_confidences[pos] = confidences[mask_idx].first` (the de-tempered commit-time
   value); ORIGIN records the sampled prob if available else leaves -1.
3. Server: `"confidences": [floats]` for generated positions in /generate response;
   /tokenize: {text, add_special, parse_special} -> {tokens: [ids], n}; /detokenize inverse.
4. Harness use: exact hole sizing via /tokenize (kills the bytes/3 estimate + retry ladder);
   proactive low-confidence remasking before compiling (RemeDi-style, threshold configurable).

Effort: ~60-100 LOC. Risks: none; confidences are already computed - this only records them.

## 4. EOT-tail truncation (variable-length steps)

### Verified facts
- Changing batch.n_tokens between steps breaks llama-graph reuse (params/input shape
  comparison) -> full graph rebuild + sched alloc (llama-context.cpp:1276-1317); reserve
  covers the worst case so no realloc; ADDITIONALLY each shape change resets the CUDA-graph
  warmup (ggml-cuda.cu:4529-4548 - needs 2 stable calls to re-capture). So every shrink costs
  one rebuild + 2 uncaptured steps (~1.5x step cost for 2 steps, measured graph value 1.5x).
- EOT detection: `llama_vocab_eot/eos` exist (llama-vocab.cpp:4121-4127); no stop logic in
  the diffusion example today. Dream-7B special tokens VERIFIED from the vocab load log
  (--verbose): BOS = EOS = PAD = 151643 `<|endoftext|>`, EOT = 151645 `<|im_end|>`,
  MASK = 151666. The padded tail is a run of 151643; the answer ends `... <|im_end|>
  <|endoftext|>...` - so the shrink rule targets the trailing run of EOS/PAD (151643),
  preserving the `<|im_end|>` and a small guard.
- shift_logits row mapping adapts automatically (rows are batch-relative).

### Implementation plan
1. diffusion.cpp, end of each step (timestep schedule only; block schedule already bounds
   compute differently): find last index >= n_input that is neither mask nor EOT/EOS; let
   new_len = that+1 + guard(8). If cur_len - new_len >= SHRINK_CHUNK (64, config) AND the
   remaining masked positions all lie < new_len: shrink max_length for subsequent steps
   (batch.n_tokens, mask scans, output length). Chunking amortizes the rebuild + CUDA-graph
   re-warmup; only ever shrink, never grow.
2. n_generated returns the final (shrunk) length; CLI prints accordingly (already resizes).
3. Quality caveat (must benchmark): the model was trained on full-length sequences;
   truncating committed-EOT context shifts the distribution slightly. Gate behind
   `--diffusion-shrink` until A/B'd on the harness benchmark set.
4. Batched-candidates interaction: per-seq lengths diverge -> shrink to the max needed
   across seqs only (simple), or repack seqs (complex - defer).

Effort: ~60-100 example LOC. Expected gain: for a 150-token answer in a 512 canvas, late
steps drop toward ~1/3 cost; with threshold decoding (few steps) the gain is smaller -
measure before keeping. Priority below 1-3.

## 5. Fast-dLLM approximate KV caching

### Verified facts
- Paper (arXiv 2505.22618): training-free; (a) block-wise KV cache - cache K/V of tokens
  OUTSIDE the active block (DualCache = both prefix and suffix), refresh at block
  boundaries; accepted approximation = cached K/V go slightly stale as the block fills, with
  "negligible" reported quality loss; (b) confidence-threshold parallel decoding (we already
  shipped this); combined up to 27.6x reported throughput on long sequences.
- PR #17454 (LLaDA 2.0): VERIFIED via web fetch this session. Status: draft since Nov 2025,
  last activity Feb 2026. Touches src/models/llada2.cpp (new), convert script, arch tables,
  diffusion-cli. Its "hybrid diffusion" mode (`--diffusion-hybrid`) is EXACTLY the
  prefix-cache blueprint: "use kv cache outside of the diffusion block, speeding the model
  up when context builds up" - on entering a new block it "commits the previous block to KV
  cache" and decodes only the active block; the author notes block TRUNCATION is required
  ("without truncate, the output becomes gibberish"). Reviewer guidance from am17an:
  "Preferably there should be no LLaDA 2.0 specific stuff in diffusion-cli" and "keep any
  kv-cache related stuff for a subsequent PR ... get that in first, before tackling
  optimisations." Net: the upstream-blessed shape is a REAL llama_kv_cache (model creates
  memory) + diffusion-cli committing finalized blocks - much simpler than the T5-style
  frozen-K/V-inputs sketch below, at least for the prefix-only (no suffix cache) variant.
  LLaDA 2.0 was TRAINED for this hybrid regime; applying it to Dream/LLaDA-1 is the
  Fast-dLLM approximation and needs the quality gate.
- Architecture: diffusion archs return nullptr from create_memory (llama-model.cpp
  ~2011-2017) -> encode() path, build_attn_inp_no_cache (n_tokens x n_tokens mask).
- VERIFIED this round: the standard KV-cache mask path natively supports NON-CAUSAL
  attention - `set_input_kq_mask` is templated on `causal` with a false instantiation
  (llama-kv-cache.cpp:1676-1713), and with causal=false the only skip rules are empty cell
  and sequence membership (:1617-1624): a batch token attends bidirectionally to EVERY
  cached cell of its sequence. `llama_memory_seq_rm` exists (llama.h:717) for clearing the
  active-block range between steps. So the PR-17454-style route needs NO core mask surgery:
  give dream/llada a llama_kv_cache, keep cparams.causal_attn=false, decode the active
  block per step (its tokens see each other AND the cached prefix), seq_rm + re-decode the
  block each step, commit at block boundaries. The T5 cross-attention "frozen K/V as graph
  inputs" pattern (cross.v_embd, llama-context.cpp:1494-1517) remains the fallback only if
  suffix caching (true DualCache) proves necessary.

### Implementation sketch (the only multi-week item; do AFTER 1-3)
Phase 0 (design spike): read PR #17454 diff fully + Fast-dLLM reference impl (github
NVlabs/Fast-dLLM); decide between (a) PR-17454-style real KV cache with block commit
(prefix-only cache - now the RECOMMENDED route given the verified PR mechanics above:
give dream/llada a standard llama_kv_cache, run block-wise, commit finished blocks,
decode only the active block with causal_attn off within the block) vs (b) example-level
"frozen-KV graph inputs" T5-style (needed only if suffix caching - true DualCache - proves
necessary for quality). The earlier T5-style sketch is kept for reference:
- new graph variant for dream/llada: inputs per layer `k_frozen[l], v_frozen[l]`
  (n_frozen x d) + active-block tokens; attention = softmax([Q_act x K_frozen ; Q_act x
  K_act]) - i.e., concat cached and fresh K/V (ggml_concat) under a combined mask;
- the example runs: full forward at block start (captures K/V of all out-of-block tokens via
  llama_set_callback or extra graph outputs - needs a build flag in the dream/llada builders
  to expose per-layer K/V as outputs, ~the largest core touch);
- subsequent steps within the block: forward ONLY the block tokens (n_tokens = block_len)
  with frozen K/V inputs; refresh at block boundaries.
- compute per step drops from O(L * L) attention + O(L) FFN to O(B * L) attention + O(B)
  FFN, B = block (32) vs L (512): FFN cost /16.
Phase 1: dream-only prototype behind `--diffusion-kv-cache`; A/B quality on the kintsugi
benchmark (compile/test pass rate) before any default-on.
Effort: realistically 800-1500 LOC across llama-graph/models/example + weeks of validation;
agent's 400-600 LOC estimate covers the happy path only. The 27.6x paper number is at 1k+
sequence lengths; at our 512/short-canvas regime expect single-digit x at best - and
improvements 1-4 + threshold decoding already harvested much of the same headroom.

## 6. Micro/robustness tier

### 6a. CUDA Graphs - DONE, became a constraint
Measured on BATTERY: 336 ms/step (graphs on, default) vs 506 ms/step
(GGML_CUDA_DISABLE_GRAPHS=1) - 1.5x. Measured on AC: 245.35 vs 246.19 ms/step - zero
difference. Interpretation: graphs eliminate CPU-side kernel-launch overhead; with the CPU
power-limited (battery, 1.5 GHz) launch submission is the bottleneck and graphs save 34%;
with the CPU at full clock the GPU is the bottleneck and launches hide behind the async
queue. Already active for diffusion (gating at ggml-cuda.cu:3296-3336 + 4493-4548: Ampere+,
no split buffers, no big-batch MUL_MAT_ID, 2-call stable warmup; our static step qualifies).
Action: none, EXCEPT every future change must keep step-graph properties stable (pointers +
shapes); shape-changing features (EOT shrink, varying canvas sizes) pay a 2-step re-warmup -
already accounted for above. The protection matters most for the unplugged-laptop regime.

### 6b. Batched/segmented top-k
top-k.cu:66-72 loops DeviceTopK per row (512 sequential launches at our shape); CCCL
DeviceSegmentedTopK (nvidia/cccl#6391) not yet released. The argsort fallback IS batched
(single segmented radix sort) at ~311 MB transient pool cost - BUT note (verified in the
ifdef structure, top-k.cu:66-96): the CUB DeviceTopK loop and the argsort fallback are
COMPILE-TIME exclusive (`#ifdef CUB_TOP_K_AVAILABLE` / `#elif GGML_CUDA_USE_CUB` / `#else`),
so a runtime heuristic first requires moving the argsort path out of the #elif so both are
compiled (~30-40 LOC, not 15). Plan: restructure + `if (nrows > 64) use argsort path` +
benchmark both at [151936 x 512]. Expected: a few ms/step at most (sampling is already
2 ms) - LOW priority, do opportunistically; revisit when CCCL ships segmented top-k.

### 6c. Degeneracy guard (EOT-flood abort)
From our measured failure mode (whole-sequence threshold 0.8 -> empty output). Plan
(example-level, ~30 LOC): after each step in threshold mode, if committed-EOT count in the
generation region exceeds X% (config, default 90%) while masked positions remain > Y%
(default 20%), abort and return a `degenerate` flag (server: "degenerate": true; CLI: warn).
The harness treats it like a failed draft (new seed). Zero risk; pure detection.

### 6d. LoRA with dream arch - works in principle, untested in practice
Verified FIRST-HAND this session: adapter loading matches by tensor name with an arch
string EQUALITY check (llama-adapter.cpp:209 - adapter GGUF arch must be "dream"); dream.cpp
routes every matmul through the LoRA-aware helpers - build_qkv (dream.cpp:81) ->
build_lora_mm for wq/wk/wv (llama-graph.cpp:1201/1211/1221, fused wqkv :1181), build_ffn
(dream.cpp:109) -> build_lora_mm (llama-graph.cpp:1259/1276+), build_attn handles wo, and
the lm_head is a direct build_lora_mm (dream.cpp:132); llada.cpp identical;
convert_lora_to_gguf.py supports the archs (DreamModel/LLaDAModelLM registered). Untested:
an actual PEFT adapter trained on Dream HF -> convert -> load -> A/B.
Plan: no engine code; a validation task for the 2x3090 milestone - train a 100-step smoke
LoRA on Dream, convert, run `llama-diffusion-cli --lora`, compare outputs. If conversion
trips on PEFT module naming, fixes live in convert_lora_to_gguf.py, not C++.

---

## Interface decisions after the PR #24423 merge (2026-06-10)

Context: PR #24423 (DiffusionGemma, same-day draft by danielhanchen/Unsloth) was merged into
the fork on branch `diffusiongemma-test` (merge 4c746fe14 + fix 93dd5e280 wiring --cpu-moe
into diffusion-cli). All masked-diffusion regressions pass post-merge (14/14 sampler tests
CPU+GPU; Dream standard/threshold/infill; DiffuCoder threshold). DiffusionGemma Q4_K_M runs
on the 8 GB laptop at 3.9 GB VRAM via `--cpu-moe` (experts stream from RAM): 17 entropy-bound
steps per 256-token canvas at ~2.7 s/step.

DECISIONS (recorded so future work does not drift):

1. The kintsugi harness MUST NOT couple to the merged CLI's output format. It is a draft
   PR's incidental logging, already under reviewer pressure (pwilkin: debug cruft, rework
   the server approach) - every PR revision can change it.
2. The harness's stable contract is OUR HTTP daemon (improvement 2): a /generate endpoint
   returning JSON we define and version. Build it before deepening any CLI integration.
3. Until the daemon exists, CLI output parsing is allowed ONLY inside the harness's
   ModelClient module (M0 stopgap per docs/dllm-elixir-harness.md) and is treated as
   disposable.
4. Do NOT adopt the PR's examples/diffusion-gemma-server: it is a model-specific
   persistent-logits server for a Python eval driver, and it is the part of the PR that
   reviewers explicitly want replaced by a general diffusion server. Our daemon should BE
   that general server (serving dream/llada masked diffusion AND diffusion-gemma canvas
   diffusion through one API).
5. DO adopt the PR's GGUF metadata convention as the model-capability interface:
   `diffusion.canvas_length` (canvas autodetect) and `diffusion.eb_*` (entropy-bound
   defaults). It is model-side, likely to survive review, and both the CLI and our daemon
   should read and expose it identically (the daemon's /health already plans to surface
   model capabilities - add canvas_length and the eb defaults there).
6. On this laptop the engine-model split is: Dream-7B/DiffuCoder = fast iteration
   (245 ms/step fully on GPU); DiffusionGemma = quality option (~45 s per 256-token block,
   expert-streaming-bound). The 3090 box likely flips this; the kintsugi benchmark
   (compile/test pass-rate per wall-clock) makes the final call, not tok/s.

## Post-merge re-baseline (verified in the merged tree, 2026-06-10)

The #24423 merge changed the ground truth under several items. Verified first-hand:

- ITEM 5 (prefix KV cache) - PARTIALLY DELIVERED for canvas models, via a THIRD pattern not
  on our list: a MODEL-OWNED store. `dg_ensure_pkv_store` (src/models/diffusion-gemma.cpp:
  629-656) lazily allocates per-layer F32 K/V tensors [head_dim, n_kv, P] in a model-held
  ggml context (grow-only, single-device - hence the multi-GPU auto-off); the PREFILL-phase
  graph persists prompt K/V into it in-graph (:440), DECODE reads it; phase selection via
  `llama_diffusion_set_phase(model, ...)` (include/llama.h:574) which dynamic_casts the
  model and mutates it. Measured working (kv_cache=on in all runs). CAVEAT for upstreaming
  and for generalizing to Dream/LLaDA: mutable per-request state on llama_model (not
  llama_context) via dynamic_cast is the part most likely to be reworked in review - treat
  the MECHANICS as proven, the API SHAPE as temporary. Item 5 for masked models remains
  open; we now have three candidate patterns (17454 real-kv_cache, 24423 model store,
  T5-style inputs) with an in-tree working example of the second.
- NEW CONTRIBUTION SURFACE CONFIRMED: the entropy-bound sampler is a full-vocab CPU loop -
  `llama_get_logits` then per-position passes over all 262144 vocab entries for softmax/
  entropy/multinomial, plus a C x n_vocab memcpy for self-conditioning
  (examples/diffusion/diffusion.cpp:784-818). On the laptop it is hidden behind the 1.1-2.7
  s/step expert-streaming forward; on the 3090 box (forward likely ~100-200 ms) it will
  dominate - the exact Dream story repeating at 262k vocab (256 MB logits D2H + CPU loops
  per step). Grafting our backend sampling onto the EB path is the high-value perf work for
  DiffusionGemma; open question stands: is entropy over top-k a sufficient proxy for the
  full-vocab entropy bound (validate against transformers).
- Self-conditioning cost RESOLVED (read at diffusion-gemma.cpp:550-600): `sc_embT` is an
  F16 [262144, 2816] soft-embedding transpose = 1.48 GB allocated on dev_layer(0)'s buffer
  type - i.e. ON THE GPU with -ngl 99 - as a separate WEIGHTS-usage buffer, built by
  host-dequantizing tok_embd. Corrected VRAM ledger for the laptop run: 1.58 GB non-expert
  model + 1.48 GB sc_embT + 2.33 GB compute = ~5.4 GB (which is why --n-cpu-moe 26 OOM'd,
  not fragmentation). Obvious optimizations: quantize sc_embT (Q8: -0.7 GB) or fold the
  soft-embedding through the existing tok_embd instead of a materialized transpose.
- The 2.33 GB compute buffer DECODED: it is ~95% the worst-case logits tensor
  (2304 rows x 262144 x 4 B = 2304 MiB + ~25 MiB of other nodes). NOT an artifact of our
  reserve change - encode() passes output_all=true and IGNORES batch logits flags (the EB
  code even comments "encode() forces all rows to output anyway", diffusion.cpp:737), so
  any unified-mode step and every PREFILL computes the lm_head over ALL rows including the
  prompt, and copies n_tokens x 262144 floats to the host. Engine-level fix worth
  pursuing: make encode() honor per-token logits flags (prefill needs NO logits; unified
  needs canvas rows only) - saves up to ~2 GB D2H per prefill + shrinks the worst-case
  compute reserve by ~2.1 GB. Benefits every encoder-path model, upstream-relevant.
- DRAFT LIMITATION FOUND: PREFILL is a single un-chunked encode() bounded by the
  `n_ubatch >= n_tokens` assert, so the maximum prompt is ~n_ubatch (2-4K on consumer
  VRAM). The model's advertised 262144 context is UNREACHABLE in this implementation -
  chunked prefill into the pkv store is a missing piece (and another reason the draft will
  evolve; do not build on its internals).
- Safety checks for our features confirmed in the merged tree:
  LLM_ARCH_DIFFUSION_GEMMA returns nullptr from create_memory (llama-model.cpp:2017) ->
  encode() path, same as dream/llada - our multi-row backend sampling machinery applies
  mechanically to canvas rows; and diffusion_generate_entropy_bound never calls
  llama_set_sampler (attachment exists only in diffusion_generate, which detaches on exit,
  diffusion.cpp:256/394/670) -> needs_raw_logits stays true for EB runs and the CPU loop
  always gets its logits. No cross-path contamination.
- CUDA graphs on the MoE (6a addendum): measured ZERO delta on the laptop AC config
  (1134.04 vs 1137.69 ms/step with GGML_CUDA_DISABLE_GRAPHS=1) - the run is expert-
  streaming-bound, so graph eligibility under MUL_MAT_ID is currently moot here;
  re-measure on the 3090 box where compute moves on-GPU.
- ITEM 4 (EOT shrink) SCOPE NARROWED: canvas models already have variable-length handling
  (trim_canvas EOG/repetition cut + adaptive stop + block budget, diffusion-cli.cpp);
  EOT-tail shrink remains relevant ONLY for the Dream/LLaDA masked path.
- ITEM 2 (daemon) gains canvas duties: serve BOTH model families through one API; manage
  the canvas per-request state (`llama_diffusion_set_sc`/`set_phase` are MODEL-level
  mutations - safe only under the daemon's single-request GPU mutex); surface
  canvas_length + eb_* defaults in /health (decision 5).
- diffusion-gemma-server characterization CONFIRMED from source: "output: C * n_vocab
  float32 canvas-row logits" over a file/stdio protocol for a Python driver
  (examples/diffusion-gemma-server/diffusion-gemma-server.cpp:10) - an eval logits dump,
  not a generation API; decision 4 stands.

## Revised priority order (post-research, post-AC-measurement)

1. Daemon (2) - unblocks the harness M1; needs NO core changes for masked models (11 ms
   per-request attach); now also serves canvas models (single-request mutex covers the
   model-level set_sc/set_phase state).
2. /tokenize + confidences (3) - tiny; multiplies harness masking precision. Extend to the
   EB path: canvas per-position entropies are the natural confidence export.
3. Degeneracy guard (6c) - 30 LOC insurance for threshold mode (masked path; the EB path
   already has trim_canvas repetition cuts).
4. EB backend sampling graft (NEW, post-merge) - move the entropy-bound sampler onto the
   backend-sampling path; becomes the dominant win the moment DiffusionGemma runs on a GPU
   that fits the experts (3090 box). Validate entropy-over-top-k first.
4b. encode() logits-flags fix (NEW) - make encode() honor per-token logits flags instead of
   output_all=true: skips the lm_head over prompt rows in PREFILL, cuts up to ~2 GB D2H per
   prefill and ~2.1 GB off the worst-case compute reserve at 262k vocab. Pairs naturally
   with 4; benefits all encoder-path models.
5. EOT shrink (4) - masked models only now; measure-gated.
6. Prefix KV cache for MASKED models (5) - partially delivered for canvas models by the
   merge (model-store pattern, mechanics proven in-tree); for Dream/LLaDA pick between the
   three patterns after the kintsugi benchmark exists.
7. Segmented top-k (6b) - opportunistic.
8. LoRA smoke test (6d) - when the 3090 box enters the picture.
9. Batched candidates (1) - DROPPED on the 5070 (measured ~5% worse than serial);
   re-evaluate only if the 3090 pp curve shows rising throughput from 512 to 2048.

## Implementation log (items 1, 3, 6c - DONE, verified end-to-end 2026-06-10)

### Item 3 - per-position confidence export
- examples/diffusion/diffusion.h: `diffusion_params.output_confidences` (float*, optional,
  caller-allocated size max_length; -1 = prompt/uncommitted/ORIGIN) and the same field on
  `diffusion_eb_params` (final-step ENTROPY per canvas position, lower = more confident).
- examples/diffusion/diffusion.cpp: array initialized to -1 at generate start; recorded at
  ALL FOUR masked commit sites (threshold commits + its single-best fallback, partial_sort
  commits, alg_temp dist commits); the EB path fills entropies after the denoise loop from
  the surviving `entropy` vector (the natural per-position uncertainty; multi-block runs
  report the FINAL block only - documented server-side).
- ORIGIN records nothing by design (it computes no confidences).

### Item 6c - degeneracy guard
- diffusion.h: `diffusion_params.out_degenerate` (bool*, optional).
- diffusion.cpp, threshold branch only: after each commit pass, scan the generation region;
  abort when committed > 8 AND committed end-tokens (llama_vocab_is_eog || pad) > 90% of
  committed AND masked > 20% of the region. Thresholds are named constants in code.
- VERIFIED on the known failure config (Dream, threshold 0.8, temp 0.2, timestep): aborts at
  step 0 with "end tokens 12/12 committed, 464/476 still masked" in 0.8 s - previously 15.4 s
  of wasted steps producing empty output.

### Item 1 - llama-diffusion-server (the general diffusion daemon)
- examples/diffusion/diffusion-server.cpp (~430 lines) + CMake target
  `llama-diffusion-server` (links llama-diffusion static lib + cpp-httplib; JSON via
  `<nlohmann/json.hpp>` and httplib via `<cpp-httplib/httplib.h>` - both resolve through
  llama-common's PUBLIC ../vendor include).
- One process = one model loaded once; ALL llama-diffusion-cli flags work and become the
  per-request DEFAULTS; request body fields override per call. Generation serialized by a
  mutex (GPU serial; canvas models keep per-request state on the model - the mutex makes
  set_sc/set_phase safe).
- Endpoints:
  GET /health -> {status, model, family: masked|canvas, mask_token_id, mask_piece,
    canvas_length, n_ctx, n_ubatch, eb_defaults?{...}} (decision 5: capabilities surfaced).
  POST /tokenize {content, add_special?=false, parse_special?=true} -> {tokens, n}
  POST /detokenize {tokens} -> {content}
  POST /generate {prompt, raw?, infill?, n_predict?, return_confidences?, seed?, steps?,
    conf_threshold?, algorithm?, temp?, top_k?, top_p?, eps?, block_length?, alg_temp?,
    cfg_scale?, add_gumbel_noise?, backend_sampling?, eb?{max_steps,t_min,t_max,
    entropy_bound,stability,confidence,kv_cache}}
    -> {text, family, ms_total, degenerate, n_prompt_tokens, confidences?,
        confidence_kind?: "entropy"(canvas)}
  Masked family runs diffusion_generate (chat template unless raw/infill; infill = canvas
  verbatim, whole canvas returned); canvas family runs the EB block-autoregressive loop
  (same trim_canvas EOG/repetition logic as the CLI).
- Implementation gotchas found (reimplementation notes):
  - `--port`/`--host` args are gated to server examples - the daemon uses common_params
    defaults (port 8080); adding the args to LLAMA_EXAMPLE_DIFFUSION is a 2-line follow-up.
  - `diffusion_params.infill` MUST be set alongside max_length = n_input - the engine guard
    rejects max_length == n_input unless infill is set (first server bug, fixed).
  - Canvas models need set_sc enabled BEFORE context creation (reserve sizing) and the
    ubatch grown to canvas_length + headroom - both mirrored from the CLI init.
- VERIFIED end-to-end on both families:
  - Dream-7B: /health (mask_piece "<|mask|>"), /tokenize, /generate with threshold 0.6 +
    confidences (481 committed positions, min 0.603 = consistent with the threshold),
    /generate infill -> "def add(a, b), do:  a + b" with -1 confidences over fixed text and
    real values over the hole.
  - DiffusionGemma Q4_K_M (--cpu-moe): /health reports family=canvas, canvas_length 256,
    eb defaults from GGUF metadata; /generate returned a 256-entropy vector with mean
    0.0036 (consistent with the 0.005 adaptive-stop threshold) in 14.6 s.

## Verification

- Each item lands with an A/B on the standard command (Dream-7B, 128-step budget, canonical
  AC matrix protocol from docs/diffusion-gpu-sampling-plan.md sec "Task 6") + the 14-test
  backend-sampler suite + (once kintsugi M2 exists) compile/test pass-rate on the benchmark
  set as the quality gate.
- Batched candidates additionally: per-seq outputs differ across seeds; seq 0 output at
  n_candidates=4 equals... (note: NOT bitwise-equal to n_candidates=1 - RNG streams and
  kernel reduction orders differ; compare distributionally).
- The pp512/1024/2048 curve WAS re-run on AC (r=3): 2228.82 +- 312 / 2127.74 +- 58 /
  2122.65 +- 5 t/s - flat; this is the measurement that dropped batched candidates.
  (The earlier battery curve - 1554/1437/1777 t/s, +-25% - is kept only as a record of why
  battery measurements cannot be trusted for design decisions.)
