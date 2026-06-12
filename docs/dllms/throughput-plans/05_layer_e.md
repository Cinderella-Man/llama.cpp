# Layer E: model-side - PLAN / IMPLEMENTATION GUIDE (E2 first)

## Context

Layer E of docs/dllms/dllm-throughput-catalog.md: the path to TINY models. Catalog
investigation order item 2, and the named RE-ENTRY TRIGGER for parked Layer D
(04_layer_d.md question 7: at 1.5B the per-row cost shrinks ~5x while the ~18 ms
fixed floor does not - batching/speculation may revive). E1 (SDTT distillation, a
training run) is scoped only after E2 establishes what a tiny dLLM is worth here.

E2 mission: pull Efficient-Large-Model/Fast_dLLM_v2_1.5B, convert to GGUF, run it
on OUR engine, measure against Dream-7B on the bench. "Cheapest path to a tiny
model TODAY" (catalog). A 1.5B block-dLLM fits a 3 GB P106-090 - the rig story.

## E2 source facts (verified 2026-06-12)

- HF: Efficient-Large-Model/Fast_dLLM_v2_1.5B, Apache 2.0. arXiv:2509.26328
  (Fast-dLLM v2): AR-to-block-diffusion conversion of Qwen2.5-1.5B-Instruct with
  ~1B fine-tuning tokens (500x less than Dream's 580B); paper claims 2.5x faster
  than the AR original at batch 1, quality parity on standard benchmarks.
- 1.54B params (1.31B non-embedding), 28 layers, BF16.
- Inference contract (model card quickstart): custom generate(input_ids,
  max_new_tokens, small_block_size=8, threshold=0.9) - blockwise bidirectional
  attention via complementary masks, TOKEN SHIFT within blocks (AR-style
  next-token parameterization, like Dream's shift_logits), hierarchical
  block/sub-block caching.
- Decode semantics vs our engine: block-CAUSAL across blocks + bidirectional
  within an 8-token block + confidence-threshold parallel commits. Our nearest
  machinery: Dream-style masked decode (threshold path) for within-block,
  DiffusionGemma block-AR precedent for blockwise progression, kv_block dual
  cache for the cached path. Expect engine-gap work; v1 target is CORRECT, not
  fast: the square path doing one 8-block at a time with bidirectional masks
  inside the block and committed context behind it.

## E2 work log (everything, including dead ends)

### 2026-06-12: checkpoint pulled and dissected
- Downloaded to ~/models/fast-dllm-v2-1.5b (3.1 GB safetensors, BF16).
- config.json: model_type Fast_dLLM_Qwen = EXACTLY Qwen2.5-1.5B-Instruct hparams
  (1536 hidden, 28 layers, 12 heads / 2 KV heads GQA, qkv-bias, tied embeddings,
  rope_theta 1e6, vocab 151936) + ONE extra field: bd_size 32. Tensor names are
  byte-for-byte Qwen2 (verified via safetensors header; 338 tensors, no lm_head -
  tied). Chat template = standard Qwen2.5 im_start/im_end.
- Mask token: "|<MASK>|" id 151665 (added_tokens.json; NOT in config - the
  converter must set tokenizer.ggml.mask_token_id itself). EOS 151645 (im_end);
  generation_config also lists 151643 (endoftext).
- modeling.py = the inference ground truth, extracted:
  * Attention at inference: eval_block_diff_mask = block_q >= block_kv at
    block_size 32 over the WHOLE sequence (prompt included) - pure BLOCK-CAUSAL:
    bidirectional within each 32-block, causal across blocks. (The fancy 3-part
    complementary mask is TRAINING-only.)
  * TOKEN SHIFT identical to Dream: logits = cat([logits[:,:1], logits[:,:-1]]) -
    row i-1 predicts position i, first row duplicated; our shift_logits machinery
    (get_row_for_pos = max(pos-1,0)) is exactly this.
  * Decode loop: per 32-block: canvas += 32 masks; 4 SUB-BLOCKS of 8, strictly
    left to right; per step forward the last 32 rows (vs cached committed blocks),
    commit within the CURRENT sub-block only: all probs > threshold (card: 0.9)
    PLUS always the argmax-prob one; when the block is mask-free, one more
    forward writes it to the cache AND the last row's argmax becomes the first
    token of the NEXT block (AR step across blocks). EOS stop once no masks
    remain before it; trailing tokens truncated.
  * Committed blocks never change -> a STANDARD causal KV cache works for them
    at block granularity (the model card's "hierarchical cache"; their
    use_block_cache sub-block variant is an optimization, not v1).
  * Defaults: top_p 0.95, temperature 0 (argmax), threshold from caller.
- Semi-AR consequence: canvas grows 32/block - the model NEVER sees masks beyond
  the current block, so Layer C4's stub-adaptation failure mode does not apply.
  No degeneracy-guard analog needed for v1 (EOS handling is explicit).

### Engine integration plan (v1 = correct, uncached square)
1. Converter (conversion/qwen.py): register Fast_dLLM_QwenForCausalLM, subclass
   Qwen2Model, model_arch = new MODEL_ARCH.FAST_DLLM; add_causal_attention(false)
   (RND1 precedent), add_mask_token_id(151665) read from tokenizer added vocab,
   write bd_size as fast-dllm.block_size kv.
2. Fork arch: LLM_ARCH_FAST_DLLM "fast-dllm": qwen2 tensor table + hparams load;
   create_memory -> nullptr (memoryless, Dream/RND1 precedent);
   llm_arch_is_diffusion includes it (server gate).
3. Graph src/models/fast-dllm.cpp: qwen2 forward with a custom no-cache attn
   input whose set_input fills BLOCK-CAUSAL (same-seq && qblock >= kvblock,
   block 32 from hparams) - subclass pattern from dream.cpp's custom mask input.
   v1 has NO pkv phases (square only; kv via the standard committed-block
   property comes later).
4. Example: new diffusion_generate_block_ar() in diffusion.cpp implementing the
   reference loop verbatim (uncached: re-forward [committed | block] each step);
   routed when the model arch is fast-dllm; CLI/server pass block/sub-block/
   threshold via existing flags (--diffusion-block-length, new sub-block flag).
5. Measure: tok/s + quality sanity vs Dream-7B; bench v2; THEN the Layer D
   re-entry checkpoint (re-run llama-diffusion-batch-probe on the 1.5B).

### 2026-06-12 (same session): IMPLEMENTED END TO END - converted, running, measured

Conversion (first try, no dead ends):
- conversion/qwen.py: FastDLLMQwenModel registered (Qwen2Model subclass,
  MODEL_ARCH.FAST_DLLM, add_causal_attention(false), mask id from added vocab,
  bd_size -> "fast-dllm.block_size" kv) + registry entry in conversion/__init__.py
  + gguf-py constants (enum, name, Dream-cloned tensor table).
- F16 GGUF 3.1 GB; Q4_K_M = 986 MB - FITS A 3 GB P106-090. (Vocab quirk logged:
  token 128247 '</s>' flagged not-control by loader; harmless.)

Fork arch (one pass, compiles clean):
- LLM_ARCH_FAST_DLLM in llama-arch.{h,cpp} (name, is_diffusion), llama-model.cpp
  (factory, create_memory nullptr, NEOX rope), models.h class with bd_size member
  (read via ml.get_key string overload, default 32).
- src/models/fast-dllm.cpp: qwen2 graph + llm_graph_input_attn_block_causal -
  square no-cache mask filled BLOCK-CAUSAL BY POSITION (same-seq && p_q/bd >=
  p_kv/bd). ~190 lines, all from dream.cpp/qwen2 precedent.

Decode loop (examples/diffusion/diffusion.cpp diffusion_generate_block_ar):
reference-faithful, uncached v1; routed by arch in CLI + server ("block_ar" in
/health; infill rejected - see LIMITATION below). Three bugs found and fixed:
1. CLI legacy assert needs --diffusion-block-length 32 alongside (eps^block xor).
2. ARGMAX-ONLY v1 was seed-deaf: all bench redrafts byte-identical AND repairs
   impossible -> p-tier 0/18. Fixed with reference sample_with_top_p (seeded
   nucleus sampling at temperature).
3. THE DE-TEMPER LESSON STRIKES AGAIN (Layer B): gating commits on the
   temp-sharpened (or nucleus-renormalized) prob floods every sub-block position
   past any threshold -> simultaneous shifted commits emit adjacent-duplicate
   garbage ("moduleirmodulemodule"). Commit confidence MUST be the plain softmax
   prob of the sampled token. Same scale law as our Dream threshold machinery.
- Threshold is the MODEL'S scale: 0.6 (Dream-tuned) corrupts ("```moduleir");
  0.9 (model card) is correct. Bench profile "fastdllm" carries it.

LIMITATION (architectural, important): block-causal attention means a masked
span mid-sequence cannot see the code AFTER it - INFILL IS STRUCTURALLY
UNSUPPORTED. kintsugi's repair ladder (the project's core mechanism) does not
apply to block-AR models; recovery is whole-draft retry only. Server rejects
infill requests for block_ar models explicitly.

MEASURED (5070 laptop, Q4_K_M, uncached square v1):
- Smoke: coherent 171-token poem, 23.2 ms/step; code drafts ~90 tok/s engine-side
  (61 tok in 0.68 s); templated Doubler draft: clean and correct in 315-500 ms vs
  Dream-7B ~1100 ms = ~3x faster per draft, on an UNCACHED path.
- Commits/step ~1.85 on code at thr 0.9 (parallelism engages on structured text;
  ~1.0 on prose - the content split again).
- BENCH (profile fastdllm, thr 0.9): 21/48 - p 12/18, h 9/9 (Credence-only),
  m/c/a 0, i 0/6 (architectural). Dream-7B full system: 35/48.
- THE FAIR COMPARISON (draft + Credence only, i-tier excluded): FastDLLM-1.5B
  21/42 BEATS Dream-7B 15/42. The 1.5B OUT-DRAFTS the 7B (Qwen2.5-Instruct
  lineage + 2026 conversion recipe vs 2025 Dream); Dream's system lead is
  entirely its repair ladder. p-tier wall: FastDLLM 87.7 s vs Dream 45.9 s -
  failed cases burn 3 full redrafts (no cheap repairs to lean on).

LAYER D RE-ENTRY CHECKPOINT (probe re-run on the 1.5B): still PARKED. Per-row
cost shrank ~3x but the fixed floor shrank WITH it (tiny forward 7.2 ms vs 27 ms
on the 7B): 103-tok canvas K=2-4 batching = 1.20-1.23x (vs 1.00x), tiny-canvas
K=8 = 2.35x, W-scaling 128 rows = 2.78x of 32. The "floor share grows at 1.5B"
hypothesis is REFUTED. (Caveat: PROBE5/6 drive Dream-style machinery this arch
does not use - treat as row-scaling proxies, not decode-path numbers.)

### Verdict and next moves
- E2 EXISTENCE PROOF: DELIVERED. A 986 MB block-dLLM runs on our engine TODAY,
  out-drafts Dream-7B head-to-head, ~3x faster per draft uncached, and fits the
  P106 fleet. The catalog's "cheapest path to a tiny model" claim is CONFIRMED.
- The strategic shape it exposes: FastDLLM-1.5B DRAFTS + Dream-7B REPAIRS is
  exactly catalog D4 (two-model draft/verify) - arrived at from measurements,
  not theory. kintsugi already serializes engines; a two-server split is
  harness-only work. THE natural next experiment.
- v1 leaves on the table: KV caching for committed blocks (standard causal cache
  fits the semantics - big per-step win at longer contexts), sub-block flag
  plumbed but untuned, P106 validation run, temperature/top_p semantics beyond
  the reference defaults.
- E1 (SDTT distillation) scoping now has its baseline: any distilled student
  must beat 21/42 draft-only at <= 1 GB to earn a training run.

## Unresolved questions
1. D4 hybrid (FastDLLM drafts + Dream repairs): build the two-server harness
   path next session (recommended - it is pure kintsugi work, no engine risk)?
2. KV cache for committed blocks: standard llama cache or the fork's pkv
   machinery (recommended: standard - block-causal IS causal at block
   granularity; the fork machinery is for bidirectional-over-store, not needed)?
3. Bench accounting for block-AR models: keep i-tier as structural zeros in the
   48-case total, or report /42 alongside (recommended: report both, the doc
   already does)?

## E3: KV cache for committed blocks (started 2026-06-12 evening)

PROGRESS LOG (updated as work lands; resume from here after any crash):
- [x] Design written (this section)
- [x] Engine: pkv plumbing for fast-dllm (models.h, set_phase/set_block dispatch)
- [x] Engine: fast-dllm.cpp DECODE/WARM graph phases
- [x] Loop: diffusion_generate_block_ar cached mode + CLI/server flag
- [x] Build clean
- [x] Equivalence gate: temp-0 argmax outputs cached vs uncached (expect identical
      or near; committed blocks are FINAL when warmed - no Dream-style drift/rewarm)
- [x] Speed: CLI ms/step at n_gen 128/256/384
- [x] Bench: ONE kintsugi run (profile fastdllm + kv), compare 21/48 + walls
- [x] Docs/memory updated, committed

### Design (decided after reading the code, 0 new machinery needed)

Reuses the Layer A shared pkv store VERBATIM (src/models/diffusion-common.h:
llama_diffusion_pkv in models.h:463, llm_graph_input_attn_diffusion_decode,
llm_graph_input_diffusion_phase, llama_diffusion_pkv_ensure_store). Standard llama
KV cache route REJECTED: create_memory is arch-level (nullptr for FAST_DLLM,
llama-model.cpp:2019), would need per-step llama_memory_seq_rm churn, and the pkv
machinery is already proven on Dream/LLaDA/DiffusionGemma.

Key insight: the current 32-block attends ALL committed prefix + itself
bidirectionally = the rectangular ALL-ALLOW mask llm_graph_input_attn_diffusion_decode
already fills. No custom cached mask needed. Block-causality within a multi-block
prompt is enforced by feeding ONE 32-aligned block per llama_decode during prefill.

Engine changes:
1. models.h: add `mutable llama_diffusion_pkv pkv;` to llama_model_fast_dllm.
2. diffusion-gemma.cpp llama_diffusion_set_phase + set_block (lines 616/646): add
   llama_model_fast_dllm to the dynamic_cast chain (shared-pkv branch).
3. fast-dllm.cpp graph, keyed off pkv.phase (copy dream.cpp:90-209 pattern):
   - UNIFIED (0): existing square block-causal path, unchanged (v1 default).
   - DECODE (2): batch = current block rows; mask = diffusion_decode rect
     [P + n_tokens, n_tokens] all-allow, allow_reuse=true + phase marker; per layer
     concat [store(0..P) | fresh roped K/V] (dream.cpp is_decode branch, 156-168).
   - WARM (1) = COMMIT: same attention as DECODE *plus* ggml_cpy of ALL batch K/V
     rows into the store at offset pkv.s (write-at-offset precedent: dream.cpp
     is_block, 173-180). Used for prompt prefill (block-by-block) and the one
     finalize pass per completed block.

Loop changes (diffusion_generate_block_ar, examples/diffusion/diffusion.cpp:1191):
- New param block_kv (default OFF for v1 equivalence), CLI --diffusion-block-kv,
  server passes through; bench profile carries it.
- Cached flow: P=0; prefill complete prompt blocks one WARM(P, s=P) decode each,
  P+=bd; per generation block: denoise steps are DECODE(P) over rows
  [blk_start, blk_end) (logit row index = pos-1-blk_start, NOT pos-1); on block
  completion one WARM(P, s=blk_start) finalize pass whose last-row logits ARE the
  AR step for the next block's first token (v1 pays a full square forward for
  this; cached mode gets it free). Prompt-tail rows inside the first generation
  block (n_input % bd != 0) ride along in the batch and get stored at finalize -
  matches reference semantics (block K/V computed from final block content).
- Store capacity: ensure via set_block(s, L_max) up front; F32, 1.5B GQA =
  28 layers x 2 x 256 x 4 B = 57 KB/token -> 1024-token ctx = 59 MB. Fits.

Why no rewarm machinery (unlike Dream A1/A2): committed blocks are FINAL when
warmed - the cached K/V are exact by construction (the model is TRAINED for this;
modeling.py does exactly this pass). No drift, no rewarm triggers, no cadence.

Expected: per-step cost flat ~7-8 ms (32-row forward, tiny model) vs v1's growing
23+ ms at 171 tokens -> ~3x at 171, more at 384+. Engine throughput target:
~200-300 tok/s raw (DiffusionGemma speed analysis doc gives the context).

### E3 MEASURED (2026-06-12, 5070 laptop, Q4_K_M, temp 0 argmax)

Equivalence gate: 96-token generation (3 blocks) BYTE-IDENTICAL cached vs uncached -
prefill WARM, DECODE mask, finalize offsets and the free AR step all verified. Longer
runs diverge on near-tie argmax flips (accumulated float differences between the
concat-store path and the square path - same acceptance criterion as Dream layer A;
the bench is the quality arbiter).

Speed (CLI, --diffusion-conf-threshold 0.9, GenServer KV-store prompt):

| output  | uncached v1            | cached (--diffusion-block-kv) | wall speedup |
|---------|------------------------|-------------------------------|--------------|
| 210 tok | 27.0 ms/step, 3.46 s   | 10.2 ms/step, 1.59 s          | 2.2x         |
| 338 tok | 36.7 ms/step, 8.00 s   | 10.3 ms/step, 2.17 s          | 3.7x         |
| ~440 tok| 43.4 ms/step, 11.85 s  | 10.4 ms/step, 3.02 s          | 3.9x         |

Cached per-step cost is FLAT (~10.3 ms) at every length - the concat read of the
F32 store is negligible against the 32-row forward. Raw engine throughput
132-156 tok/s (vs 35-61 uncached). The ~10 ms floor decomposes as ~7 ms forward
(layer D probe) + ~3 ms full-vocab CPU logits readback (152k vocab x 32 rows) -
backend sampling is the known next bite at it.

Caveat for the rig: pkv phase state lives on the MODEL - concurrent requests on
multiple contexts of one model (multi-replica server) would race. Single-replica
per model only, or per-context state later.

### E3 BENCH GATE + the p_reverse investigation (2026-06-12 late evening)

BENCH (profile fastdllm + --diffusion-block-kv server flag, ONE run, guards active,
zero incidents; 20260612T190731Z-e3-fastdllm-kv.jsonl):

| config | pass | deliverable | total wall |
|---|---|---|---|
| E2 uncached (t09 baseline) | 21/48 | 3.54 tok/s | 156 s |
| E3 block-kv | 19/48 | **8.17 tok/s (2.3x)** | 62 s (2.52x) |

Tier walls: m 1.74x, c 2.13x, h 2.02x faster; p-tier unchanged (drafts are 1-2
blocks - the cache has nothing to amortize there yet).

The 2 lost passes were BOTH p_reverse (seeds 11, 111; all 3 kv seeds burned 3
drafts). Investigated to ground truth:
- Text-level: cached output on that prompt shows duplicate-token artifacts
  ("do do") - looked like a shifted-commit bug.
- Geometry bisect: prompts < 32 tok (no prefill WARM) byte-identical at temp 0;
  aligned 64-tok prompt clean; only the partial-tail geometry showed the artifact
  -> suspicion of a tail-block bug.
- LOGIT-LEVEL ground truth (PROBE7, new env-gated probe in diffusion-batch-probe:
  PROBE_KV=1): [WARM|WARM|DECODE] vs UNIFIED square on identical committed
  content: max|dlogit| 1.47, argmax agree 32/32.
- ENVELOPE BASELINE (PROBE1, same model): pure batch-SHAPE change on the
  unchanged square path = max|dlogit| 1.54-1.59, argmax 103/103.

VERDICT: the cached path's deviation is AT/BELOW the model's own batch-shape
numerics envelope - E3 is mathematically faithful. p_reverse rides the quality
edge: +/-1.5-logit noise flips low-confidence commits, and the progress-guarantee
(force-commit best position) cascades them into visibly different (occasionally
degenerate) drafts. ANY numerics perturbation costs +/- a couple of bench passes
on edge cases; the 21 vs 19 delta is that sensitivity, not a cache defect.

RECOMMENDATION: --diffusion-block-kv ON for FastDLLM serving (2.3x deliverable,
2.5x wall; the rig's currency). Note for multi-replica rig serving: pkv phase
state is per-MODEL - single replica per model process only (same caveat as
Dream layer A). [CAVEAT RESOLVED 2026-06-13: the multi-replica server creates a
SEPARATE llama_model per replica, so block-kv + replicas compose fine - verified
live with 3 concurrent block_kv requests on 3 replicas (F1 probe, 07_layer_f.md).]

NEXT BITES at the ~10.3 ms step floor: backend sampling for block-AR (saves the
~3 ms full-vocab 32-row D2H per step), sub-block tuning, P106 validation.

## E4: backend (GPU) sampling for block-AR (started 2026-06-13; plan 07_layer_f.md)

PROGRESS LOG (resume from here after any crash):
- [x] Design written (this section), committed
- [x] Loop: diffusion_generate_block_ar backend path + warmup fallback
- [x] Build clean, 14/14 sampler tests (CPU and GPU)
- [x] Equivalence gate: temp-0 outputs backend vs CPU, 2-3 prompts (expect identical
      or near - confidence renormalizes over top-k, see design)
- [x] Speed: CLI ms/step, kv on/off, backend vs CPU sampling
- [x] Bench: same-process A/B (profiles e3kv / e4bs), compare vs E3 19/48 + 8.17 tok/s
- [x] Docs/memory updated, committed

### Design (decided after re-reading the masked path + reference semantics)

The 07_layer_f.md open question - "can the backend sampler chain express
plain-softmax commit confidence exactly?" - dissolves once the chain is built
WITHOUT temp/top_p: chain = top_k(K) -> dist(seed). dist's data->probs is then the
PLAIN softmax over the top-K candidates (untempered - temp only enters via a temp
sampler, which we omit), and the host replicates the reference sample_with_top_p
over those K candidates:
- temp <= 0: argmax over the K plain probs (top-1 of top-K == global argmax,
  exact); confidence = its plain prob.
- temp > 0: tempered q_j ~ p_j^(1/temp) over the K candidates, sort desc, nucleus
  top_p over q, draw from the loop's own rng (CPU-path RNG semantics preserved);
  confidence = PLAIN p_j of the sampled candidate (the de-temper lesson, by
  construction - no de-temper math needed since probs were never tempered).
The backend dist sampler's own sampled token is IGNORED (its RNG draw is paid but
unused); the chain exists to deliver candidates + plain probs at k floats/row
instead of 152k.

KNOWN approximation: confidence renormalizes over the top-K set, not the full
vocab - inflated by the tail mass beyond K (~1% at code positions with K=40). At
thr 0.9 a near-boundary commit can fire one step early vs the CPU path; this is
inside the E3-measured numerics envelope (+-1.5 logit batch-shape noise moves the
same commits), gate = bench.

Mechanics:
- Backend chain attached via llama_set_sampler(ctx, 0, ...) when
  params.backend_sampling && top_k > 0 (else CPU path, warn - mirroring the
  masked path's eligibility rules; bench/CLI/server all pass top_k 40).
- Warmup probe on the FIRST decode (sampled_ith(0) == NULL -> detach, redo the
  decode for raw logits, fall back to CPU) - masked-path precedent verbatim.
- forward_range returns success bool; CPU logits pointer kept only when
  !use_backend. Row for pos = (pos-1) - batch_base unchanged.
- AR-boundary predicts (kv finalize WARM doubles as the AR step) read the LAST
  decode's sampled rows via _ith - same lifetime as the current ar_logits host
  pointer (no decode intervenes between the WARM and the boundary predict).
- With the sampler attached, needs_raw_logits goes false: the 32-row x 152k x 4 B
  logits D2H disappears per step (the ~3 ms target of the 10.3 ms floor).

### E4 MEASURED (2026-06-13, 5070 laptop, Q4_K_M) - ADOPTED

Equivalence gate (temp 0, seed 7, kv on): BYTE-IDENTICAL backend vs CPU on all 3
prompts (poem 103 tok, GenServer 155 tok, Stack 472 tok). At 472 tokens the step
COUNTS differ (293 backend vs 311 CPU - the top-K renormalized confidence commits
near-boundary positions a step earlier) while the token sequence stays identical:
the approximation re-times commits without changing content on these prompts.

Speed (CLI, Stack prompt, temp 0, steps cap 384, 472 tok generated both modes):

| config            | CPU sampling      | backend sampling  | per-step | wall  |
|-------------------|-------------------|-------------------|----------|-------|
| kv on (block-kv)  | 10.24 ms, 3.18 s  | 8.42 ms, 2.47 s   | 1.22x    | 1.29x |
| kv off (square)   | 39.75 ms, 9.06 s  | 33.98 ms, 7.20 s  | 1.17x    | 1.26x |

Raw engine throughput kv+bs: 191 tok/s (was 148 CPU-sampled, 132-156 at E3).
The ~8.4 ms step that remains is ~7 ms forward (layer D probe floor) + ~1.4 ms
host/readback - the next bite is the forward itself (E6 graph check, sub-block
width E5a).

BENCH GATE (same server process, request-param profiles e3kv/e4bs; h-tier wall
noise investigated - see below):

| profile | pass | deliverable | notes |
|---|---|---|---|
| e3kv (= E3 config reproduced) | 19/48 | 6.76 tok/s | run 2: 19/48, 6.27 (host-wall drift) |
| e4bs (+ backend sampling)     | 19/48 | **9.24 tok/s (+37-47%)** | pass outcomes IDENTICAL per compare.exs |

m-tier median 3536 -> 2098 ms, c-tier 3004 -> 1526 ms (failing tiers burn 3 full
drafts each - step savings compound). compare.exs flagged h-tier +27.5% wall as a
"regression": h is the Credence-only tier (repairs=0, drafts=0 in every row, zero
GPU) and a THIRD run (e3kv again) showed 738 -> 941 -> 903 ms across identical
configs - host-side subprocess (runner VM boot) drift between back-to-back runs,
not an engine effect. Lesson for the gate: tier-wall comparisons only bind for
tiers that actually exercise the changed path.

Cross-process note: this session's e3kv reproduces E3's 19/48 exactly (pass-
exactness survives restarts, as calibrated); deliverable tok/s does NOT transfer
across processes (8.17 then, 6.27-6.76 now, same config) - the E4 claim is the
same-process +37-47%.

VERDICT: backend sampling ON for block-AR serving (it is the default via
params.backend_sampling; --no-backend-sampling / "backend_sampling": false to
disable). 07_layer_f.md open question 1 ANSWERED: the chain expresses plain-softmax
confidence EXACTLY (over the top-K set) by omitting temp/top_p from the chain and
sampling host-side from the K plain probs.

## E5: commit-rate levers (started 2026-06-13; plan 07_layer_f.md)

1.85 commits/step (code, thr 0.9) is the biggest no-training lever: commits/step
x2 ~= tok/s x2. Three independent bench-gated experiments on top of E4 (kv+bs).

PROGRESS LOG:
- [x] E5a sub-block sweep: sb 8 (reference) -> 16, 32. Server shape probe, bench.
- [x] E5b threshold sweep: 0.9 -> 0.85, 0.8. Same protocol.
- [x] E5c entropy-bound committer port (DG machinery, trained-in there - fail fast).
- [x] Docs/memory updated, committed

GATE (07_layer_f.md): bench pass count holds 19/48-class (accept +-2 = the
measured numerics band); walls + commits/step logged per config. KILL IF passes
drop >2 at every setting (1.85 is then the model's honest rate).
Method notes: probes ran THROUGH the server (CLI-vs-server GPU sharing rule),
single-seed = SHAPE only; verdicts from the bench, same server process as E4.

### E5a/E5b MEASURED (2026-06-13)

Shape probes (server, temp 0.2, seed 3, kv+bs; steps for 448-tok stack prompt /
commits-per-step incl. warm+AR steps):

| config | stack steps | commits/step | genserver steps |
|---|---|---|---|
| sb8 thr0.9 (E4 reference) | 273 | 1.64 | 177 |
| sb16 | 271 | 1.65 | 169 |
| sb32 | **205** | **2.19** | 170 |
| thr 0.85 | 250 | 1.79 | 161 |
| thr 0.8 | 240 | 1.87 | 153 |
| sb32 + thr 0.85 | 227 | 1.97 | - |

BENCH (same process as the E4 runs; reference e4bs = 19/48, 9.24 tok/s):

| profile | pass | deliverable | verdict |
|---|---|---|---|
| e5sb32 | 19/48 (p_reverse-11 GAINED, p_max-141 LOST) | 8.99 | pass-swap inside the numerics band, no deliverable gain |
| e5t085 | 18/48 (p -1) | 9.54 | +3% for -1 pass: noise-level trade, no adoption |
| e5t08 | 14/48 (p 5/18) | 8.38 | QUALITY CLIFF - killed |

READING: the commit-rate lever is QUALITY-BOUND, exactly as 07_layer_f.md
anticipated ("1.85 is the model's honest rate"). The threshold family maps the
cliff: 0.9 is calibrated, 0.85 starts bleeding passes for ~nothing, 0.8 collapses
p-tier. sb32's long-form step saving (1.32x wall on the 448-tok probe) does NOT
transfer to the bench's short drafts - same content-length split as Dream's
window-64/kv_block (Layers A/C): a knob for genuinely long output, not a default.
DEFAULTS UNCHANGED: sb 8, thr 0.9.

### E5c MEASURED (2026-06-13) - KILLED

Implemented minimally on E4's machinery (entropy over the top-K plain probs is
~free in predict(); the EB rule sorts the sub-block's masked positions by
ascending entropy and accepts while cumulative H <= block_eb, lowest always
commits). Engine param block_eb + server "block_eb"; default 0 = off; the
default path is untouched (e4bs re-run on the new binary/process: 19/48, 9.76
tok/s - also the in-process reference below).

Shape probe (stack-prompt steps): eb 0.1 = 348 (WORSE than thr 0.9's 273 - a
tight budget under-commits), 0.3 = 297, 0.6 = 281, 1.0 = 241. The budget rule
only beats the threshold rule at eb 1.0, i.e. at thr-0.8-class aggressiveness.

BENCH e5eb10 (eb 1.0, same process as the e4bs 19/48 / 9.76 reference):
17/48 - p_double 42+142 and p_max 41+141 LOST (p 10 -> 8), deliverable 9.37.
VERDICT: REGRESSION - killed. The EB rule samples the same quality-vs-rate curve
the threshold sweep mapped; without DG's training there is no better operating
point on it. block_eb stays in the tree default-off (canvas-model precedent).

### E5 LAYER VERDICT
All three levers measured, NONE adopted: 1.85 commits/step IS the model's honest
rate (the plan's KILL-IF outcome, now with data). Commit-rate gains for block-AR
models are training-side work (E1/F3 territory), not decode-side.

## E6: forward-cost floor - CUDA graph / llama-graph reuse check (2026-06-13)

Plan (07_layer_f.md): the fast-dllm DECODE path sets allow_reuse=true but
capture/replay was never measured here. A/B at fixed geometry (stack prompt,
temp 0, seed 7, kv+bs, steps 384 = the 8.42 ms/step E4 reference):
GGML_CUDA_DISABLE_GRAPHS=1 (CUDA-graph replay off) x LLAMA_GRAPH_REUSE_DISABLE=1
(llama graph-rebuild every decode). Expected 1.2-1.5x if launch-bound, possibly
nothing if already reusing. Results below.

### E6 MEASURED (2026-06-13, 5070 laptop AC, 2 reps each, 293 steps/472 tok fixed)

| config | ms/step (rep1 / rep2) |
|---|---|
| default (graphs + reuse)   | 8.48 / 8.52 |
| GGML_CUDA_DISABLE_GRAPHS=1 | 8.46 / 8.52 |
| LLAMA_GRAPH_REUSE_DISABLE=1| 8.60 / 8.65 |
| both off                   | 8.58 / 8.61 |

VERDICT: NOTHING TO GAIN on this hardware. CUDA-graph replay is worth exactly
zero on AC (the GPU is the bottleneck; launches hide behind the async queue -
same conclusion as the 2026-06-10 Dream A/B); llama-graph reuse IS engaging on
the DECODE path and is worth ~0.1 ms/step (~1.5%) - the can_reuse/phase-marker
machinery works as designed, the answer to the "was capture/replay measured"
question is now yes. The ~8.4 ms step is forward-bound (~7 ms model forward,
layer-D probe floor + ~1.4 ms host loop/readback). CAVEAT carried forward: this
null is 5070-on-AC-specific; launch-bound regimes (battery, Pascal Celeron rig)
are where graphs paid 1.5x before - the rig-day GGML_CUDA_FORCE_GRAPHS A/B
(p106 doc E3) is still owed and unaffected by this verdict.

## Catalog-E6 partial: quant ladder for the block-AR model (2026-06-13)

Catalog item E6 ("re-calibrate conf_threshold PER QUANT; parallel commits may be
MORE sensitive to logit noise") applied to FastDLLM-1.5B, cheapest first: the E3
p_reverse losses were measured numerics-edge flips, so LESS quant noise (Q8_0,
1.63 GB - still fits the 6 GB rig next to nothing else, or the 3 GB P106-090
does NOT hold it: Q4 stays the 3 GB option) may buy passes back, and IQ4_XS
probes the down-ladder. Protocol: quantize from the on-disk F16, ONE bench per
quant (e4bs profile, same process per pair where possible), CLI ms/step.
Results below.

### MEASURED (2026-06-13, e4bs profile, kv+bs)

| quant | size | bench | tier shape | deliverable | CLI ms/step (stack, temp 0) |
|---|---|---|---|---|---|
| Q4_K_M (shipped) | 986 MB | 19/48 | p 10/18, m 0/3 | 9.2-9.8 | 8.56 (293 steps) |
| Q8_0   | 1.65 GB | 18/48 | p 6/18, m 3/3 | 12.63 | 9.77 (290 steps) |
| Q8_0 + thr 0.85 | - | 15/48 | p 6/18, m 0/3 | 8.85 | - |
| IQ4_XS | 902 MB | 18/48 | p 6/18, m 3/3 | 14.13 | 8.22 (339 steps) |

FINDINGS:
1. Quant level is a PATH LOTTERY inside the same +-2 quality band for block-AR:
   totals 18-19/48 across an 1.8x size range, but the CASE MIX flips - both
   Q8_0 and IQ4_XS pass m_sumdoc 3/3 (Q4_K_M: 0/3) while losing 4 p-tier seeds.
   The "less quant noise buys numerics-edge passes back" hypothesis is REFUTED
   in its naive form: Q8 moves the noise, it does not remove the sensitivity.
2. Deliverable tok/s swings 9.2-14.1 purely by WHICH cases pass (m answers are
   long) - deliverable comparisons across quants need pass-set context.
3. Threshold 0.9 holds at every quant (Q8+0.85 strictly worse) - the catalog's
   "re-calibrate conf_threshold per quant" concern is ANSWERED: not needed in
   the Q4-Q8 range for this model.
4. Speeds: Q8_0 +14% step cost; IQ4_XS cheapest per step on Blackwell.
VERDICT: Q4_K_M stays the default (best p-tier shape, best reference-prompt
wall). IQ4_XS (902 MB) is the validated option when 986 MB does not fit (3 GB
P106-090 margins) at equal-band quality. W3/W2 cliffs below IQ4 remain
unmeasured (no current need).

## LAYER E PART-1 CLOSURE (2026-06-13)

07_layer_f.md PART 1 status:
- E4 backend sampling: ADOPTED (+37-47% deliverable, 9.2-9.8 vs 6.3-6.8 tok/s
  same-process; raw 191 tok/s; temp-0 byte-identical; default ON).
- E5 commit-rate levers: ALL MEASURED, NONE ADOPTED - 1.85 commits/step is the
  model's honest decode-side rate; further gains are training-side (E1/F3).
- E6 graph check: measured, nothing to gain on the 5070/AC; rig caveat recorded.
- E-rig P106 validation: BLOCKED on hardware (no P106 on hand) - runbook in
  docs/dllms/p106-mining-fleet.md sec 9 unchanged; note the pkv per-MODEL state
  caveat (one replica per model process) still applies.
- E1 SDTT: stays PARKED behind F2's oracle-probe verdict (07_layer_f.md).
Bench state at close: e4bs = 19/48, 9.24-9.76 tok/s deliverable (vs 3.54 at E2
uncached CPU-sampled = 2.6-2.8x cumulative from E3+E4). Regressions green:
14/14 sampler tests CPU+GPU; Dream masked path untouched (block-AR loop only);
defaults: thr 0.9, sb 8, block_kv per request, backend sampling on.
