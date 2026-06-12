# Layer F: post-E3 speedup roadmap - PLAN (written 2026-06-12, start by FINISHING E)

## Context

State after E3 (05_layer_e.md): FastDLLM-1.5B with committed-block KV cache runs at a
FLAT ~10.3 ms/step at any length, 132-156 tok/s raw, 8.17 deliverable tok/s on bench
(2.3x over uncached). Layers A-D CLOSED (see their docs for verdicts incl. rejected
branches). Layer E is NOT finished: E2+E3 delivered, E1 (distillation) scoped only.

The 10.3 ms step decomposes as ~7 ms forward + ~3 ms CPU full-vocab logits readback
(152k vocab x 32 rows x 4 B/step). The commit rate is 1.85 tokens/forward (code,
thr 0.9) vs DiffusionGemma's trained-in 5-15. Deliverable tok/s is dominated by
FAILED cases burning 3 SEQUENTIAL redrafts (p-tier fail = 3 full drafts, no repairs
possible on block-AR).

RULES (unchanged): one bench at a time; bench guards stay on; write findings to
the doc BEFORE heavy GPU work; every claim goes through bench v2 or a probe with
numbers in this file.

---

## PART 1 - finish Layer E first (each is ~one session, no training)

### E4. Backend (GPU) sampling for block-AR  [DO FIRST - known machinery]
- WHAT: route diffusion_generate_block_ar's predict() onto the fork's multi-row
  backend sampling (built for Dream; 175x sampling reduction there). Today the loop
  reads full-vocab logits to host every step and does CPU softmax/nucleus per position.
- WHY: ~3 ms of the 10.3 ms step is readback+CPU sampling. Expected ~1.3-1.4x
  -> ~190-210 tok/s raw.
- HOW: the backend sampler chain already does multi-row top-k/top-p/temp + confidence
  export. Needs: plain-softmax commit confidence (the de-temper lesson - confidence
  MUST be the UN-tempered prob of the sampled token, see 05_layer_e.md fix #3) and
  the row->position shift (row i predicts pos i+1) preserved.
- GATE: temp-0 equivalence vs CPU path on 2-3 prompts; ONE bench run (label
  e4-fastdllm-bs); ms/step before/after.
- KILL IF: backend confidence cannot reproduce plain-softmax semantics (then document
  and keep CPU sampling; partial win possible via top-k-only readback).

### E5. Commit-rate levers (1.85 commits/step is the biggest no-training lever)
Three independent, cheap, bench-gated experiments (~one bench each; can share a session):
- E5a sub-block sweep: sb=8 (reference) -> 16, 32. Wider commit scope per step.
- E5b threshold sweep: 0.9 -> 0.85, 0.8. We know 0.6 corrupts (model-scale cliff);
  the edge between 0.8 and 0.9 is unmeasured.
- E5c entropy-bound committer port: DG's cumulative-entropy budget (entropy_bound,
  temp schedule) expressed over our existing confidence machinery as an alternative
  accept rule. DG's is TRAINED-IN - expect modest gains here, fail fast.
- WHY: commits/step x2 ~= tok/s x2 (step count halves). DiffusionGemma proves the
  decode-shape supports 5-15 with the right committer.
- GATE: bench pass count must hold (21/48-class; accept +-2 = the measured
  numerics-sensitivity band, see E3 p_reverse investigation). Walls + commits/step
  logged per config.
- KILL IF: passes drop >2 at every setting (then 1.85 is the model's honest rate;
  document and move on).

### E6. Forward-cost floor (7 ms vs ~2.6 ms bandwidth floor)
- WHAT: one session checking CUDA-graph capture + llama.cpp graph reuse on the cached
  path. The Dream phase-marker work measured ~3x per-step from graph reuse on small
  canvases - verify the fast-dllm DECODE path actually reuses (same phase marker
  machinery; allow_reuse=true is set, but capture/replay was never measured here).
- HOW: GGML_CUDA_DISABLE_GRAPHS A/B + nsys/step timing at fixed geometry; check
  can_reuse hit rate (a LOG_DBG counter is enough).
- EXPECTED: 1.2-1.5x if launch-bound; possibly nothing if already reusing. Cheap to
  know.

### E-rig. P106 validation run (deployment gate, not a speedup)
- FastDLLM Q4 (986 MB) + block-kv on one P106-100: measure ms/step, tok/s, VRAM.
- CAVEAT (documented in 05_layer_e.md): pkv state is per-MODEL - one replica per
  model process; the multi-replica single-process server does NOT compose with
  block-kv yet. Either run one process per card on the rig, or move pkv to
  per-context state first (small refactor; do it only if the rig numbers justify).

### E1. SDTT distillation - STAYS PARKED until F2 evidence lands
Bar unchanged: beat 21/42 draft-only at <= 1 GB. F2's oracle probe (below) may
REPLACE E1 entirely: if drafting-for-AR-verify works, the drafter does not need
standalone quality - reassess after F2.

---

## PART 2 - Layer F proper (start when Part 1 E4+E5 are measured)

### F1. Parallel seed racing (catalog G5/G6) - the layer D re-entry trigger has FIRED
- PREMISE: layer D was parked with re-entry condition "a genuinely different decode
  regime". E3 IS that regime: flat 10 ms steps, 32-row batches, 1.5B model.
- WHAT: race N seeds of the same draft as ONE batched decode (N sequences x 32 rows).
  Failed cases currently burn 3 SEQUENTIAL redrafts; racing makes a failure cost
  ~1 draft of wall. This attacks the DELIVERABLE metric where it actually bleeds
  (13 hard cases dominate bench wall).
- DE-RISK (probe before building): batched-step cost at N=2,3,4 x 32 rows on the
  CACHED path (the 1.5B probe measured tiny-canvas K=8 at 2.35x cost - racing wins
  if N seeds cost < N sequential drafts, i.e. ratio < N). Cross-seq isolation is
  already proven bit-exact (layer D finding 1).
- ENGINE NEEDS: multi-sequence block-AR decode (per-seq pkv stores or one store per
  seq slot - design before coding), server N-seeds-per-request param, kintsugi
  first-success pick.
- EXPECTED: bench wall on failing tiers /2-3; deliverable tok/s up correspondingly.
- KILL IF: batched step cost ratio >= N (no win) or quality of batched decode
  diverges beyond the numerics envelope (PROBE7 methodology).

### F2. DFlash-style pairing: FastDLLM drafts -> AR model verifies (LOSSLESS quality)
- PAPER: DFlash (arXiv 2602.06036, ICML 2026; in vLLM speculators) - block-diffusion
  drafter + AR target verification, >6x lossless. Their drafter is trained
  context-conditioned; OURS comes free: FastDLLM-1.5B IS Qwen2.5-1.5B lineage ->
  pair against Qwen2.5-7B-Instruct (SAME family/tokenizer) with llama.cpp's existing
  speculative-decoding infra.
- WHY THIS CHANGES THE GAME: output distribution = Qwen2.5-7B's (a different and
  likely HIGHER quality ceiling than Dream's 35/48), at block-diffusion drafting
  speed. Also reframes E1: the drafter no longer needs standalone quality.
- STEP 1 (cheap, decides everything): ORACLE ACCEPTANCE PROBE (layer D PROBE6
  methodology) - draft 32-token blocks with FastDLLM (cached, greedy + t0.2),
  measure how many consecutive tokens Qwen2.5-7B-Instruct (greedy) accepts per
  block, on the bench p-tier prompts. Acceptance L tokens/block at draft cost
  ~6 steps x 10 ms + verify 1 x ~40 ms forward -> speedup ~ L / (AR per-token
  ~35 ms x ...) - compute the actual ratio from measured numbers, the formula
  goes here after the probe.
- STEP 2 (only if L >= ~8): wire as llama.cpp speculative pair (target Qwen2.5-7B
  Q4 ~4.7 GB + drafter 1 GB = fits 6 GB rig card AND the 8 GB laptop).
- KILL IF: oracle acceptance < ~4-6 tokens/block off-the-shelf (then it needs
  DFlash-style drafter training = E1-class effort; document the measured L and park).
- NOTE: this is a NEW product shape (AR-quality engine) - kintsugi still applies
  (compiler verify on top), Dream repair ladder stays for infill.

### F3. MTP heads / drafter training - LAST (training tier)
- Multi-token heads at the block boundary (seed next block's first k tokens instead
  of 1), FastMTP-style self-speculation, or DFlash drafter training proper.
- Only after E4/E5/F1/F2 are spent AND F2's probe says training would flip its
  verdict. Bar and bench protocol per E1.

---

## Order for tomorrow
1. E4 backend sampling - DONE 2026-06-13, ADOPTED (+37-47% deliverable, 191 tok/s
   raw; open question 1 answered - see 05_layer_e.md E4)
2. E5a/b/c commit-rate sweeps - DONE 2026-06-13, ALL KILLED (1.85 commits/step is
   the model's honest rate; open question 2 answered: NO setting beats it without
   dropping passes - see 05_layer_e.md E5)
3. E6 CUDA-graph check - DONE 2026-06-13, null on 5070/AC (reuse already engaged;
   step is forward-bound; rig caveat recorded - see 05_layer_e.md E6)
4. F2 STEP 1 oracle probe (cheap, strategic - decides F2 vs E1 vs F3) <- NEXT
5. F1 de-risk probe, then build if ratio < N
6. E-rig P106 run when an engine config stabilizes (hardware-gated)
The 1-3 compounded ceiling estimate (~2.5-3x over E3) was NOT reached: only E4
landed (1.22-1.29x engine, +40% deliverable); E5/E6 headroom did not exist.
Decode-side exhaustion on this model is now measured, which is exactly why F2
(verify-against-AR, a different axis) is the next move.

## F2 STEP 1 work log (started 2026-06-13)

Oracle acceptance probe design (PROBE6 methodology, two-phase to keep one model
on the GPU at a time):
1. DRAFT phase: FastDLLM-1.5B server (kv+bs, the E4 config) generates drafts for
   the bench p-tier instructions (kintsugi forge wrapper, the production prompt
   shape) + m_sumdoc + c_stack (longer = more blocks), greedy AND t0.2, seeds
   {3,103,203} for the sampled variant. Drafts + the wrapped user content saved
   as JSONL.
2. VERIFY phase: new probe tool llama-oracle-probe (examples/diffusion): loads
   the AR verifier (Qwen2.5-7B-Instruct Q4_K_M, SAME tokenizer family), applies
   its own chat template to the user content (identical Qwen2.5 template), one
   causal decode over [prompt | draft] with all logits, then per 32-token block
   of the draft: L = consecutive positions from block start where the verifier's
   greedy argmax (row pos-1) equals the draft token. Reports per-block L, mean L,
   per-position agreement.
   Retokenization note: drafts cross as TEXT (shared tokenizer; canonical
   re-tokenization) - acceptable for an oracle estimate, exact token plumbing
   only matters in STEP 2.
3. Speedup model with measured numbers (filled after the probe): draft cost/block
   ~ (32/1.85 commits-per-step) x 8.4 ms + verify 1 forward; AR baseline ~ Qwen-7B
   ~35-40 ms/token tg. Speedup ~ L / (draft+verify cost expressed in AR-tokens).
GO bar: L >= ~8 -> wire as llama.cpp speculative pair (STEP 2). KILL: L < ~4-6
off-the-shelf -> document L, park F2 (drafter training = E1-class effort).

### F2 STEP 1 MEASURED (2026-06-13) - acceptance GO, economics RE-BASED

Probe: 32 drafts (8 cases x greedy + 3 t0.2 seeds; FastDLLM E4 config) verified
by Qwen2.5-7B-Instruct Q4_K_M greedy (llama-oracle-probe, new tool in
examples/diffusion):

- 70 blocks | mean L 12.01 | MEDIAN L 8 | mean first-block L 12.25 | overall
  token agreement 84.9% | blocks with L>=8: 38/70 | fully accepted: 12/70.
- Distribution is bimodal: the 12 L=32 blocks come from two 256-token repetitive
  drafts (repetition is trivially predictable) - the honest center is median 8 /
  first-block 12.25. Per-case first-block L: double 29(all of it), even 5,
  sum 14, reverse 8, max 13, swap 6, sumdoc 12, stack 11.
- Probe caveat: blocks after the first are conditioned on the DRAFTER's own
  continuation; a real spec loop re-drafts from the verifier's corrected prefix
  (first-block L is the cleanest per-round proxy).

THE ECONOMICS CHANGED UNDER US: measured Qwen2.5-7B Q4_K_M on the 5070 =
52.6 tok/s tg (19.0 ms/token; the plan assumed ~35), pp512 2808 t/s. Round
arithmetic with measured numbers: draft a 32-block ~145 ms (17.3 steps x 8.4 ms,
1.85 commits/step) + verify ~15 ms (32 rows vs cache) vs gain (L+1) x 19 ms ->
1.54x at L=12, 1.07x at median L=8. Off-the-shelf pairing is REAL but NOT the
paper's >6x (their drafter is trained AND cheap). Draft-length tuning is the
obvious lever: acceptance dies by ~12-14, so 16-token drafts (~72 ms) project
~1.8-2x. RIG NOTE: economics INVERT on Pascal (drafter forward is compute-shaped
= 11x slower; AR verify is bandwidth-shaped = barely slower) - F2 is a
laptop/modern-card play only.

VERDICT: GO bar met (median 8, first-block 12.25) -> STEP 2 v0 prototype
justified, with expectations re-based to ~1.5-2x over AR (= ~80-100 tok/s at
Qwen-7B quality - still a NEW QUALITY CEILING vs Dream's 35/48 system). The v0's
real job: measure TRUE closed-loop acceptance (oracle is approximate in both
directions) and end-to-end tok/s vs the same-process AR baseline.
Drafts: /tmp/f2_drafts.jsonl (regenerate via the doc's recipe; bench prompts).

## Unresolved questions (answer as they land)
1. E4: can the backend sampler chain express plain-softmax commit confidence
   exactly? (If only approximately - is the approximation inside the numerics
   envelope measured in E3?)
2. E5: does ANY committer setting beat 1.85 commits/step without dropping >2 bench
   passes?
3. F1: per-seq pkv store design - N independent stores vs one store with seq-strided
   rows? (Decide on memory: 59 MB/seq at 1024 ctx is cheap; independence is simpler.)
4. F2: oracle acceptance L on p-tier prompts = ? (THE number; everything branches
   on it.)
5. Rig: is per-context pkv worth the refactor, or is one-process-per-card fine for
   9 cards on 4 GB host RAM? (Measured: ~52 MB marginal replica host cost in the
   multi-replica design - one process per card costs more; measure actual.)
