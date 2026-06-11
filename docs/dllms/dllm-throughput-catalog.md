# The dLLM Throughput Catalog: every idea for fast diffusion on tiny hardware

Mission: acceptable-quality diffusion code generation on dramatically shrunken hardware
(6 GB Pascal mining cards up to laptop RTX 5070, models 1-8B, this llama.cpp fork +
kintsugi). Diffusion LLMs are non-negotiable. This document catalogs EVERY candidate
improvement - big, small, proven, speculative - to be investigated one by one. Nothing
was filtered out for being small: the target is the sum of a thousand improvements.

Provenance: deep-research workflow 2026-06-11 (108 agents, 26 primary sources, 25 claims
adversarially verified 3-0 unless noted) merged with everything measured in this fork
(docs/diffusion-gpu-sampling-plan.md, dllm-engine-improvements.md, dllm-elixir-harness.md,
p106-mining-fleet.md). Each entry: what / expected gain / evidence / applies to / floor /
effort / status.

Honest calibration up front (verified head-to-heads): published headline multipliers
(27.6x, 68.2x) are vs UNACCELERATED vanilla 256-1024-step decoding. Our stack already
runs Fast-dLLM-style confidence-threshold decoding, so the verified INCREMENTAL headroom
over what we have is roughly: feature caching ~1.4x+, adaptive threshold 1.07-1.21x,
suffix pruning ~1.7x, SSD self-speculation 2-3x (lossless), Streaming-dLLM bundle
1.5-2.3x over a Fast-dLLM-class baseline - and these STACK. A 5-15x further end-to-end
gain on existing models is evidenced, before any model-side work.

---

## Layer A - Step-to-step feature/KV caching (the biggest unexploited lever)

Our engine's core assumption - "masked dLLMs need a full-canvas forward every step, no
cache possible" - is FALSE per three independent verified papers. KV activations across
adjacent denoising steps have cosine similarity > 0.94.

- A1. Fast-dLLM v1 approximate dual cache (arXiv:2505.22618, github.com/NVlabs/Fast-dLLM)
  [verified 3-0]. Block-wise approximate KV cache for bidirectional attention: cache
  prefix AND suffix KV, reuse within a block, refresh at block boundaries. Up to 27.6x
  vs vanilla (8-shot prefill, 1024-token gen). Applies: masked (Dream/LLaDA/DiffuCoder).
  Floor: untested below datacenter GPUs - cache memory vs 6 GB is OUR experiment.
  Effort: large (engine: per-layer KV save/restore around the step loop). Status: the
  single highest-value engine item. Our docs item 6 (masked prefix-KV) is a subset.
- A2. dLLM-Cache adaptive feature caching + V-verify (arXiv:2506.06295, ICML,
  github.com/maomaocun/dLLM-cache) [verified 3-0]. Caches K, V, AttnOut, FFNOut per
  layer; prompt features refreshed every ~100 steps, response features every ~10, and in
  between recompute ONLY the ~25% of tokens whose Value vectors drifted most (cosine
  V-verify). 3.2-5.3x TPS, up to 9.1x FLOPs reduction; BEATS Fast-dLLM head-to-head
  (5.33x vs 3.83x, Dream GPQA, RTX 4090); accuracy sometimes IMPROVES. Caveat: only
  1.36x on Dream HumanEval - code commits change features faster; our threshold decoding
  already shortens runs, shrinking cache reuse windows. Applies: masked. Floor: cache of
  4 tensors/layer x canvas - quantify for 6 GB. Effort: large. Status: catalog.
- A3. dKV-Cache (arXiv:2505.15781). Delayed KV caching - cache a token's KV only AFTER
  it is committed (committed tokens' representations are stable; masked ones are not).
  Conceptually the cleanest fit for threshold decoding where commits are explicit.
  Applies: masked. Effort: large. Status: catalog - compare design vs A1/A2 before
  building any of them.
- A4. d2Cache (arXiv:2509.23094). Extends dLLM-Cache (certainty-prioritized decoding +
  two-level caching). Read after A2. Status: catalog.
- A5. Fast-dLLM v2 hierarchical cache (arXiv:2509.26328, ICLR'26) [verified 3-0].
  Block-level KV cache for completed blocks + sub-block cache for the partial block -
  EXACT (not approximate) caching, but requires block-diffusion architecture (their
  AR-converted models, or DiffusionGemma-style), not vanilla Dream. Applies: block.
  Status: catalog; pairs with E2.
- A6. Prompt-only prefix KV (our docs item 6, three patterns). Strict subset of A1 -
  if A1 lands, this is subsumed; if A1 proves too heavy for 6 GB, the prompt-only
  variant is the cheap fallback (prompt tokens never change; canvas re-decoded).
  Status: designed in dllm-engine-improvements.md.
- A7. Cache + quantized cache: store cached features in Q8/FP8 to halve cache VRAM on
  6 GB cards (llama.cpp already has K/V cache type flags for AR; reuse plumbing).
  Speculative. Effort: medium once A1/A2 exists.

## Layer B - Decoding schedules (cheap, pure-software, stack on everything)

- B1. Adaptive confidence threshold tau(t) = tau0 * (1 - alpha(1 - r_mask))
  (Streaming-dLLM, arXiv:2601.17917; corroborated CadLLM arXiv:2512.07173,
  arXiv:2511.02077) [verified 3-0]. Lower the threshold as fewer masks remain: +1.21x
  at alpha=0.6, no quality loss (alpha > 0.6 degrades). NEAR-DROP-IN: ~10 lines in our
  threshold branch. Effort: tiny. Status: top of the quick-win list.
- B2. SlowFast two-phase sampling (arXiv:2506.10848, ICLR'26,
  github.com/LiangrunFlora/Slow-Fast-Sampling) [verified 3-0]. Exploratory slow phase
  (monitor confidence convergence), then aggressive span-parallel fast phase. 2-5x
  typical alone (HumanEval 3.15x); 34.22x stacked with dLLM-Cache. Pure schedule change,
  GPU-agnostic. Effort: medium (step-loop state machine). Status: catalog, high value.
- B3. The three "golden principles" as standalone heuristics [verified 3-0]: certainty
  (high-confidence = stable), convergence (confidence converges over steps), positional
  (confident tokens cluster in contiguous spans). Each is independently exploitable:
  e.g. span-based unmasking (commit whole contiguous high-confidence spans), or
  convergence-based early commit. Effort: small each. Status: catalog as separate items.
- B4. Prophet early-commit (arXiv:2508.19982). Commit-all-and-stop once the
  top-1/top-2 confidence gap is large - "good enough, stop denoising". Effort: small.
- B5. Threshold operating point: Fast-dLLM v2 ablation shows tau=0.9 gives 2.6x
  (39.1 -> 101.7 tok/s) with marginal GSM8K drop [verified]. We default 0.6 (chosen on
  battery!). Re-sweep tau on the kintsugi bench per model. Effort: trivial (bench runs).
- B6. EB-style entropy-bound schedule for MASKED models: DiffusionGemma's
  entropy/stability/adaptive-stop machinery (already in our engine) ported to the
  masked path as an alternative to confidence thresholds. Speculative. Effort: medium.
- B7. Per-position step budgets: easy positions commit early (already true); cap TOTAL
  steps by canvas difficulty estimate (mask count, prompt length) instead of fixed 128.
  Effort: tiny. Speculative.
- B8. Remasking/self-correction schedules (ReMDM-style): allow re-masking committed
  tokens mid-run when confidence collapses (wrong commits poison neighbors; today only
  the harness can fix them post-hoc). Search: ReMDM arXiv:2503.00307. Effort: medium.
- B9. Seed-canvas initialization: initialize masked positions from a cheap prior
  (n-gram/retrieval/template, see H5) instead of all-mask - fewer steps to converge.
  Speculative, training-free. Effort: medium.

## Layer C - Canvas-compute reduction (extend what we started)

- C1. Suffix sliding-window pruning (Streaming-dLLM) [verified 3-0]: attention from
  active positions to DISTANT suffix masks is negligible - keep a window (w=128) plus
  the final anchor token; 1.73x with slightly IMPROVED accuracy. Generalizes our
  EOT-tail shrink from "committed EOT tail" to "any distant suffix mask". Corroborated:
  DPad (arXiv:2508.14148, 1.18-3.91x), Sparse-dLLM. Applies: masked. Effort: medium
  (window the batch like the shrink, keep anchor). Status: natural next engine step.
- C2. EOS early exit [verified 3-0; medium confidence]: stop generating ALL remaining
  blocks when EOS is predicted with high confidence (corroborated Prophet, SchED
  arXiv:2512.02892). Our EOT-tail shrink reduces compute; this ENDS the run. Caveat:
  Rainbow Padding (arXiv:2510.03680) - instruction-tuned dLLMs can emit premature
  high-confidence EOS at long max_length; guard with a min-tokens floor. Effort: small.
- C3. Prefix shrink: after the PROMPT, leading committed non-EOT tokens also stop
  changing - freeze-and-window the front like the tail (subsumed by A-layer caches but
  cheap standalone). Speculative. Effort: small.
- C4. Adaptive n_gen growth mid-run: start tiny (96), GROW the canvas only if no EOS
  materialized (cheaper than redraft-at-2x; needs engine support for canvas extension).
  Speculative. Effort: medium.
- C5. Prompt compression: shorten kintsugi's instruction wrapper (every prompt token
  rides every step); strip chat-template boilerplate for raw code tasks. Measured cost
  today: ~50 prompt tokens vs 192-canvas = ~20% of each forward. Effort: tiny.
- C6. Multi-hole single-canvas repair: the engine already fills ALL mask runs in one
  infill - kintsugi should batch independent diagnostics into ONE repair call instead
  of sequential repairs. Effort: small (harness only). Status: quick win.
- C7. Attention sparsity for bidirectional attention (Sparse-dLLM, arXiv:2509.xxxxx
  family): prune low-weight attention pairs per step. Pascal-friendly only if it maps
  to smaller dense GEMMs (block sparsity). Speculative for our engine. Effort: large.

## Layer D - Speculative / parallel step execution

- D1. SSD self-speculative decoding (arXiv:2510.04147) [verified 3-0]. The dLLM drafts
  tokens for all masked positions in one pass, then verifies a greedy chain of N+1
  candidates in ONE batched forward; accepts up to N+1 tokens/iteration. 2.0-3.46x,
  output IDENTICAL to stepwise decoding (lossless - zero quality risk, rare!). Built on
  Fast-dLLM dual cache; the free verification batch relies on memory-bound inference at
  batch <= 8 - MUST re-verify on Pascal's compute/bandwidth ratio (P106 is
  compute-poor: batch may not be free). Applies: masked. Effort: large (needs A1
  first). Status: catalog, after A-layer.
- D2. Spiffy (arXiv:2509.18085): tree-structured speculative drafts for dLLMs.
  Read alongside D1.
- D3. Batched multi-canvas decoding: decode several independent canvases (candidates,
  repairs) in one batch on ONE GPU. We measured pp batching flat on 5070 (AC) - but
  WITH caching (A-layer) inference becomes memory-bound and batching turns free (the
  SSD observation). Re-measure after A1. On the 9-GPU rig, cross-card farming already
  covers this. Effort: medium.
- D4. Two-model draft/verify: tiny dLLM (1B, E-layer) drafts the canvas; big model
  verifies/refines only low-confidence spans. Speculative (no paper found for
  dLLM-pairs); our infill machinery makes the refine step natural. Effort: medium,
  needs a good tiny model first.

## Layer E - Model-side (the path to TINY models; mostly training jobs)

- E1. SDTT step distillation (arXiv:2410.21035, ICLR'25, github.com/jdeschena/sdtt)
  [verified 3-0; one 2-1 sub-claim]. Distill 1024-step teachers into 16-32-step
  students; >= 32 tokens per forward; at 1.3B the distilled model is 8x FASTER THAN A
  KV-CACHED AR MODEL with NO caching at all - i.e. in exactly our no-cache engine
  regime. Demonstrated <= 1.3B on perplexity/LAMBADA (NOT code; quality sub-claim got
  2-1 on gameable metrics). THE training-side lever for the mission: a 1-3B
  SDTT-distilled code dLLM at 16 steps on a P106 = ~interactive. Effort: training run
  (feasible scale). Status: the strategic bet to scope.
- E2. AR-to-block-diffusion conversion with ~1B tokens (Fast-dLLM v2, arXiv:2509.26328)
  [verified 3-0]. Qwen2.5 1.5B/7B converted to block-dLLMs for ~1B fine-tuning tokens
  (500x less than Dream's 580B); 2.5x faster than the AR original at batch 1; PUBLIC
  1.5B checkpoint exists (HF: Efficient-Large-Model/Fast_dLLM_v2_1.5B). A 1.5B
  block-dLLM fits a 3 GB P106-090! Engine gap: needs Fast-dLLM-v2-style block decode
  in our fork (we have DiffusionGemma block-AR as a cousin). Status: download the
  1.5B checkpoint, convert to GGUF, measure - cheapest path to a tiny model TODAY.
- E3. CDLM consistency distillation (unverified - verifier died on rate limit, claims:
  3.4-7.9x fewer steps, block-causal attention enabling exact KV cache, Dream-7B
  HumanEval 50.0 vs 48.2 after distillation). Re-verify, then catalog next to E1.
- E4. Di4C distillation (arXiv:2410.08709, unverified): distills the JOINT
  (correlations between positions) not just marginals - attacks the conditional-
  independence error that forces small commit batches; stacks on other distillation
  (claimed ~2x more). Catalog.
- E5. DiDi-Instruct (unverified, claimed up to 64x fewer steps): instruction-style
  distillation of masked dLLMs. Catalog, re-verify.
- E6. Quantization ladder for dLLMs: we run Q4_K_M. Investigate W3/W2 (IQ3/IQ2 GGUF)
  quality cliffs on dLLMs specifically (parallel commits may be MORE sensitive to logit
  noise - confidence calibration shifts!); QAT'd dLLM checkpoints; FP8 activations on
  modern cards; dp4a INT8 paths are already our Pascal regime. Also: re-calibrate
  conf_threshold PER QUANT (quant changes confidence distributions). Effort: bench runs.
- E7. LM-head + embedding compression: 152k-vocab head is ~0.5-1 GB and runs EVERY
  step over EVERY canvas row. Ideas: quantize head harder than body; low-rank factorize
  (SVD) the head; skip head for non-active rows (engine: we documented lm_head-over-
  all-rows waste - encode() logits-flags fix, docs item 4b). Status: 4b is designed;
  the rest speculative.
- E8. Vocabulary pruning for code: most of 152k tokens never appear in Elixir code.
  A pruned 32-48k head shrinks the head GEMM 3-4x and VRAM by ~700 MB. Risky (model
  was trained with full vocab; logits renormalize) but training-free to TRY via logit
  masking first (H4), then physically prune. Speculative. Effort: medium.
- E9. Layer skipping / early exit per denoising step: early steps (noisy canvas) may
  not need all 28 layers; late steps refine details. Per-step depth schedules - image
  diffusion precedent (DeepCache works BECAUSE late layers change slowly). Speculative
  for text. Effort: large.
- E10. Existing small dLLMs inventory (to test as-is): Fast-dLLM v2 1.5B (E2);
  LLaDA-MoE (arXiv:2509.24389 - MoE dLLM, experts-on-CPU pattern like DiffusionGemma);
  any Dream/DiffuCoder distills that appear. Keep scanning HF.
- E11. LoRA/IA3 adapters at inference for task specialization of a tiny model (Elixir
  adapter on a 1.5B base): cheaper than full fine-tune, swappable per language.
  llama.cpp LoRA support exists (untested with diffusion models - smoke test it).

## Layer F - Engine/system level (llama.cpp fork specifics)

- F1. [DONE this fork] GPU backend sampling multi-row (~2 ms/step flat), threshold
  decoding + de-tempered calibration, EOT-tail shrink, right-sized canvases (n_gen),
  k-stride pinned buffers (1997->783 MB), multi-replica server (~52 MB/replica),
  GGML_CUDA_FORCE_GRAPHS, --host/--port. Listed so nobody re-researches them.
- F2. encode() logits-flags fix (docs item 4b): stop computing lm_head over ALL rows
  on the canvas decode - ~2 GB VRAM + meaningful per-step FLOPs at long canvases.
  Designed, not built. Effort: medium, core-risky.
- F3. EB backend-sampling graft (docs item 4): DiffusionGemma's full-vocab CPU entropy
  loop -> GPU. Matters when forward gets fast. 3090-gated historically; revisit.
- F4. sc_embT quantization (DiffusionGemma): the 1.48 GB F16 self-conditioning matrix
  -> Q8/Q4 (~0.7-1.1 GB saved). Effort: medium.
- F5. Chunked prefill for canvas models: without it max prompt = one ubatch. Effort:
  medium.
- F6. Step-loop CUDA graph capture: capture the ENTIRE denoising step (decode +
  sampling chain) as one graph, replayed N times - kills per-step launch overhead
  everywhere, not just Ampere+ (pair with GGML_CUDA_FORCE_GRAPHS on Pascal; we measured
  1.5x from graphs when launch-bound). Effort: medium-large (graph capture across our
  step loop's host logic requires restructuring the loop to be shape-stable - EOT
  shrink changes shapes! tension to resolve: maybe re-capture on shrink).
- F7. Overlap host work with device work: detokenize/HTTP/JSON of request N while
  request N+1 computes (server-level pipelining per replica); also overlap the
  sampling-readback (already async) with the next step's batch setup. Effort: small.
- F8. Pascal micro-tuning (rig-day): lock clocks (nvidia-smi -lgc), persistence mode,
  compile 61-real (not JIT from virtual), FA on/off per-arch bench, MMQ vs cuBLAS
  forced paths, ubatch sweep per card, quant sweep (Q4_0 vs Q4_K_M dp4a paths). Each
  worth 2-10%. Status: runbook E2 in p106 doc.
- F9. Server-side prompt caching across requests: kintsugi re-sends near-identical
  prompt wrappers; with A-layer caching, prompt features could persist ACROSS requests
  keyed by prompt hash (the kintsugi loop reuses the same instruction template
  constantly). Effort: medium, after A1.
- F10. Speculative-canvas warm start: repair requests re-decode a canvas 95% identical
  to the draft the server JUST decoded - cache last canvas features per session and
  V-verify-style partial-update only changed positions (the hole). This is dLLM-Cache
  applied ACROSS REQUESTS - potentially huge for kintsugi's repair loop (repairs are
  small deltas by construction!). Speculative but uniquely suited to our workload.
  Effort: large, after A2. Status: flagship original idea worth a design doc.
- F11. Reduce step-loop host overhead: batch setup memcpy per step (cur_length tokens)
  - keep the batch persistent and update only committed positions. Micro (~us). Listed
  per "nothing too small".

## Layer G - Harness/kintsugi co-design (GPU calls avoided = infinite speedup)

- G1. [DONE] Credence as repair round 0 (heal_fib: 2 GPU repairs -> 0, 35.8 -> 136
  tok/s); module-name alignment; body remask on check failures; token-sized hole
  sweeps; draft gating; autofix extraction.
- G2. Credence rule expansion as a PROCESS: every recurring model quirk becomes a rule
  (each converts a GPU repair class to free). Mine the bench/production failure logs
  weekly for new rules. The asymptotic goal: GPU only for semantic content, never
  syntax. Effort: continuous small.
- G3. Confidence-driven proactive remasking: use output_confidences (already exported)
  to remask lowest-confidence spans BEFORE compiling - catch likely-wrong code without
  a verify round-trip. Threshold to calibrate per model. Effort: small. Status: the
  confidence export was built for exactly this; not yet used by kintsugi!
- G4. Semantic repair cache: hash (error message class + masked span context) ->
  previously successful fill; replay deterministically before GPU. Grows like Credence
  but data-driven. Effort: medium.
- G5. Parallel hole-size variants across replicas: the {n, n+2, 1.4n} sweep runs
  sequentially today; on multi-GPU (rig) fire all three concurrently, first compiling
  wins, cancel rest. Effort: small (Task.async_stream). Rig-relevant.
- G6. Parallel seeds-and-checks: forge fires 2-3 seeds concurrently across replicas;
  first to pass check wins (system-level best-of-N - the candidate farm pattern).
  Effort: small. Rig flagship.
- G7. Verifier-streamed early abort: kintsugi's step_callback/SSE could stream partial
  canvases; compile-check partial code mid-generation and ABORT hopeless drafts at
  step 5 instead of step 17. Needs server streaming endpoint. Effort: medium.
- G8. Hole-size learning: log (replaced-tokens, chosen-variant, success) and fit the
  initial hole size per error class - kill the sweep's wasted variants. Effort: small.
- G9. Canvas seeding from repo context (retrieval): pre-commit OBVIOUS tokens (module
  header, function signatures from the check, imports) deterministically before step 1
  - the model only diffuses the body. Fewer masks = fewer steps. Our infill already
  supports this (fixed text + holes)! kintsugi can template-skeleton EVERY draft:
  "defmodule X do\n  <masks>\nend" - turning forge into structured infill. Effort:
  small. Status: high-value quick win, uses only existing engine features.
- G10. Logit-bias bans of known-bad patterns (uses our existing backend logit_bias
  sampler!): ban/penalize the exact token sequences behind FixDoBlockFusion (", do"
  fusions) AT DECODE TIME - prevent the quirk instead of fixing it. Per-model ban
  lists derived from Credence rule hit counts. Effort: small-medium (sequence-level
  bias needs care; single-token bans are trivial). Status: original idea, catalog.
- G11. Grammar-constrained decoding for dLLMs (research-grade): llama.cpp grammars
  assume left-to-right; diffusion commits any-order, so full GBNF is hard - but
  POSITION-LOCAL constraints (valid identifier chars, balanced-delimiter counts as a
  canvas-level filter) could mask obviously-invalid tokens per step on the backend.
  Speculative, possibly novel research. Effort: large.
- G12. Multi-diagnostic batch repair (= C6 harness side). Quick win.
- G13. Check-first prompting: include the check in the draft prompt (model sees the
  test = fewer semantic misses; classic but unmeasured here). Effort: trivial.

## Layer H - Left-field (catalog everything)

- H1. Cross-request semantic caching: cache (instruction-embedding -> final code) and
  serve near-duplicates instantly; kintsugi workloads repeat heavily. Effort: medium.
- H2. Token merging for text (ToMe-style): merge redundant canvas positions (long
  whitespace/comment runs) into super-tokens during attention. Image-diffusion
  precedent; unproven for text dLLMs. Speculative research. Effort: large.
- H3. RAM/VRAM offload of cold layers (--n-cpu-moe precedent): for 6 GB cards +
  bigger models, keep coldest tensors (head?) in RAM over PCIe - x1 makes this brutal
  on the rig (250 MB/s) but viable on x16 hosts. Catalog with caveat.
- H4. Logit masking to a code-subset vocabulary at SAMPLING time (precursor to E8):
  backend logit_bias with -inf on never-valid-in-Elixir tokens; free quality, maybe
  free speed via top-k behavior. Effort: small. Try before E8.
- H5. n-gram / retrieval speculative seeding (= B9/G9 family): mine the repo being
  edited for likely spans (function names, common idioms) and pre-commit them. The
  diffusion fills gaps - literally its native operation. Effort: small-medium.
- H6. "1000 small things" measurement discipline: every idea above gets a bench number
  on the kintsugi fixed-seed bench (tok/s aggregate + pass rate) before/after, recorded
  in this file. The bench IS the instrument (kintsugi/bench/bench.exs).

## Refuted / corrected during verification (do not re-cite naively)

- "27.6x/68.2x apply to our stack" - NO: baselines are vanilla multi-step decoding;
  our stack already has the biggest single component (threshold decoding).
- dLLM-Cache on CODE: only 1.36x on Dream HumanEval (vs 5x on knowledge tasks) -
  code's fast-changing features cut cache reuse; temper A2 expectations for kintsugi.
- SDTT "exceeds AR quality": 2-1 split vote - perplexity/LAMBADA metrics are gameable;
  treat quality claims as unproven for code until we bench.
- Several claims (CDLM, Di4C, DiDi-Instruct details, SSD sub-claims) were never
  adversarially completed due to rate limits - marked unverified above; re-verify
  before acting.

## Suggested investigation order (we will go through ALL of them; this is just a start)

1. Quick wins, this week: B1 (adaptive tau), B5 (tau re-sweep), G9 (skeleton-seeded
   drafts), G3 (confidence-driven remask), C6/G12 (multi-hole repair), G13, C5, G8.
2. E2: pull Fast_dLLM_v2_1.5B, GGUF it, measure - the tiny-model existence proof.
3. C1 (suffix window pruning) - natural extension of our EOT shrink, engine-local.
4. A-layer bake-off design review (A1 vs A2 vs A3 on 6 GB budgets), then build one.
5. D1 SSD (lossless) once A-layer exists; F10 cross-request canvas cache design doc.
6. E1 SDTT scoping: cost a 1-3B distillation run for a code dLLM.
7. Rig-day items: F8, G5, G6 (the farm).

## Primary sources

arXiv:2505.22618 (Fast-dLLM) | arXiv:2506.06295 (dLLM-Cache) | arXiv:2505.15781
(dKV-Cache) | arXiv:2509.23094 (d2Cache) | arXiv:2506.10848 (SlowFast) |
arXiv:2508.19982 (Prophet) | arXiv:2509.26328 (Fast-dLLM v2) | arXiv:2510.04147 (SSD) |
arXiv:2509.18085 (Spiffy) | arXiv:2601.17917 (Streaming-dLLM) | arXiv:2508.14148 (DPad)
| arXiv:2512.02892 (SchED) | arXiv:2510.03680 (Rainbow Padding) | arXiv:2512.07173
(CadLLM) | arXiv:2410.21035 (SDTT) | arXiv:2410.08709 (Di4C) | arXiv:2509.24389
(LLaDA-MoE) | github.com/NVlabs/Fast-dLLM | github.com/maomaocun/dLLM-cache |
github.com/LiangrunFlora/Slow-Fast-Sampling | github.com/jdeschena/sdtt |
HF Efficient-Large-Model/Fast_dLLM_v2_1.5B | NVIDIA llama.cpp CUDA-graphs blog |
johannesgaessler.github.io/llamacpp_performance
