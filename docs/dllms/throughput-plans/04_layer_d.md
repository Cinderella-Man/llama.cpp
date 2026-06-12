# Layer D: speculative / parallel step execution - PLAN / IMPLEMENTATION GUIDE

## Context

Layer D of docs/dllms/dllm-throughput-catalog.md: spend BATCH dimension to buy STEPS.
Critical calibration before believing any paper number: both reference methods are
"lossless" accelerators of STEPWISE decoding (1 token/step) - the baseline we abandoned
in Layer 0. Our threshold parallel commits ARE the thing they speculate toward; Prophet
early-commit (B2) is already the degenerate "accept everything" special case. So
headline multipliers (3.46x SSD, 2.8-3.1x Spiffy) do NOT transfer. The honest headroom:
1. Spiffy's OWN composability table: threshold-0.7 + Spiffy = 7.88x vs vanilla, i.e.
   a ~1.5-2x MARGINAL on top of threshold decoding - the steps where our threshold
   commits only 1-3 tokens (mid-game stalls, end-game) are the recoverable cost.
2. The fixed-cost floor measured in Layer C grilling (2026-06-12, 5070 laptop):
   per-step wall ~25-30 ms at <100 rows, ~87 ms at ~200 rows, ~110 ms at ~440 rows.
   Below ~100 rows the GPU is overhead-dominated -> batching K small forwards into one
   approaches K-times-free. This is the engine-local fact that makes Layer D real
   HERE, independent of paper claims (their "memory-bound at batch<=8" story is the
   A100 version of the same observation; P106 must be re-measured - catalog caveat).
3. Kintsugi's repair ladder is SEQUENTIAL today (try_hole_variants: up to 3 infill
   calls back-to-back, each paying the floor) - the only Layer D item that is pure
   harness-visible win with no speculation math at all.

Status up front:
- D1 SSD + D2 Spiffy: same family, plan together. NO PUBLIC CODE for either (checked
  2026-06-12; SSD "code will be made available", Spiffy none) - paper-only planning
  (precedent: Prophet B4). Mechanical details below from full texts; re-derive during
  implementation, do not trust my paraphrase.
- D3 multi-canvas batching: MASSIVELY de-risked by a code fact (sec "Engine reality").
- D4 two-model draft/verify: GATED on E2 (a usable tiny dLLM). Design notes only.

## Reference algorithms (paper-only, no clones)

### D1. SSD (arXiv:2510.04147) - self-speculative draft + linear-chain verify
- SelfDraft: the NORMAL forward already predicts every masked position - draft = top-1
  token for the k highest-confidence masked positions (k = draft length, paper best
  3-5). Zero extra cost: we already read sorted top-k probs per row (backend sampler).
- Verify: batch of k+1 sequence COPIES; copy j has the first j draft tokens placed
  (priority order). One batched forward. Acceptance = exact match, child accepted only
  if every parent prediction matched ("linear chain"); up to k+1 commits/iteration.
- Lossless vs stepwise semi-AR greedy because acceptance == "what stepwise would do".
- Dream-Instruct 3.46x (6.4 -> 22 TPS), LLaDA 2.11x, 50-77% step reduction - ALL vs
  stepwise baseline. Built on Fast-dLLM dual cache, block 8, refresh 8 (our kv_block
  is the analog; ours is 32/rewarm 6).
- OUR mapping: draft = predicted NEXT-THRESHOLD-STEP commits, not next single token.
  Acceptance = the engine's own commit rule replayed on the verified logits. Lossless
  RELATIVE TO OUR PATH by construction; byte-identity is the gate, not the bench.

### D2. Spiffy (arXiv:2509.18085) - directed draft GRAPHS + offline calibration
- Generalizes SSD's chain: D draft blocks = speculated future block-states, graph not
  tree (multi-parent: bidirectional attention lets one verified state confirm several
  ancestors). All D verified in ONE forward via custom block-diagonal-ish attention
  mask + per-draft positions ("single model call with a custom attention mask").
- Draft formulas = (position-rank, vocab-rank) pairs; CALIBRATED OFFLINE: run vanilla
  decode on ~50 samples, record which (i,j) actually got unmasked k steps ahead, pick
  the D-subgraph maximizing accumulated acceptance counts. ~30 min on one A100;
  converges <50 samples. Acceptance rule: draft block == actual next block state ->
  swap in the draft's cached distribution, recurse. Lossless (Appendix A.1).
- LLaDA 2.8-3.1x vanilla; composes: +hardcode-4/iter = 5.28x, +threshold0.7 = 7.88x.
  D in {3,5,8,10}; gains saturate ~D=5-10. Block 32, gen 256 (= our shapes).
- OUR mapping: calibration harness = bench prompts (we have 48 cases + seeds); the
  graph is small (D<=5 states of one 32-block); per-step the batch is [current block
  + D speculated blocks] x 32 rows vs the frozen store - EXACTLY the fork's PKV_BLOCK
  phase with a block-diagonal batch-side mask.

### D3. Batched multi-canvas decoding - no paper needed
Independent canvases in one forward; value = the fixed-cost floor, not memory-bound
theory. Targets, by value: (a) repair hole-size variants (today 3 SEQUENTIAL infills
per repair round), (b) multi-seed draft racing (redraft attempts 2-3 as a parallel
race instead of sequential), (c) cross-request server batching (rig already covers
via replicas; lowest priority).

### D4. Tiny-drafts-big-verifies - design notes only (E2-gated)
Tiny dLLM (Fast_dLLM_v2_1.5B once GGUF'd, or future E1 distill) drafts the full
canvas; big model runs ONE forward over the drafted canvas; positions where big
model's threshold rule agrees stay, disagreements re-masked -> our existing infill
repairs them. This is exactly our draft->verify->infill loop with the draft
outsourced. No new engine machinery beyond D0/D3. BLOCKED until E2 measurement.

## Engine reality (verified against code, 2026-06-12)

- THE DE-RISK: src/llama-graph.cpp:443 (non-causal mask path, no KV) already masks
  cross-sequence attention (s0 != s1 -> -inf). Multi-canvas batching on the SQUARE
  path needs ZERO graph surgery: give each canvas a distinct seq_id, keep sum of
  lengths <= n_ubatch, map rows per-seq in example code. The entire D0 enabler on the
  square path is example-level batch construction + row bookkeeping (the row_of_pos
  vector idea from C1b, now actually needed).
- KV modes: the store is OUTSIDE ubatch; PKV_BLOCK/DECODE masks are custom rectangular
  (batch x [store|batch]) built in the fork's graph path - multi-variant there needs
  the batch-side submask block-diagonal while keeping full store visibility. Moderate
  surgery, A-layer 3-way-concat class, NOT free. Sequence AFTER square-path wins.
- Backend sampling is per-row (llama_get_sampled_*_ith) - works unchanged for
  multi-seq batches. Confidence/commit logic is example-side, already per-position.
- n_ubatch 512 today: 3 repair variants (~60-150 tok canvases) fit one batch; 2+
  full draft canvases (192-gen + prompt) do NOT - draft racing needs ub 1024 (VRAM
  check on 8 GB: weights 4.4 GB + ctx; measure, may force ub 768 or fp16-kv).
- Server is one-request-one-generate; D3a lives INSIDE one /generate (infill request
  carries variant list), no concurrency model change. Proposed API: "canvases": [str]
  (or "hole_sizes": [n1,n2,n3] server-side expansion), response: per-variant texts.

## Implementation order (each: implement -> probe -> code KPI 3 seeds {3,103,203} ->
## bench v2 gate -> doc log; builds judged by EXIT CODE; same-session comparisons)

1. D0 microbench (GO/NO-GO for everything): square path, K identical canvases as K
   seq_ids, K in {1,2,3,4,8}; report ms/step vs K and vs K separate runs; THEN same
   on a 32-row block shape (the D1/D2 regime). Also the correctness probe: 2 canvases
   batched == 2 sequential runs byte-identical. ~1 day. NO-GO if batched K=3 costs
   >= 2.2x single (floor amortization absent on this GPU) -> D3a still viable check
   at exactly 3-variant shape before killing.
2. D3a batched repair variants (kintsugi + server + example): infill request carries
   all hole sizes; engine decodes them as one multi-seq batch; kintsugi takes first
   compiling fill (same selection semantics as today's sequential ladder - early-exit
   saving traded for floor amortization). Gate: i-tier latency + cascade-case walls +
   repairs-per-task unchanged + pass count (expect neutral pass, -30-50% repair wall).
3. D3b draft racing (OPTIONAL, only if ub/VRAM check passes): redraft attempts as
   2-seed parallel race at 384. Gate: bench pass + wall on the 16/48 redraft cases.
   NOTE Layer C lesson: geometry changes shift draft quality - racing changes NOTHING
   about geometry (same canvases, just batched) so quality must be IDENTICAL per
   seed; verify byte-identity per branch first.
4. D1-lite SSD chain on the SQUARE path first (no kv store interaction): draft k=3
   next-commits, k+1 canvas copies in one batch (fits ub for <=128-tok canvases ONLY
   - i.e. repairs and short drafts; fine, that is where threshold stalls bite), exact
   match acceptance. Gate: byte-identity vs non-speculative run (MANDATORY, the
   lossless claim), then step-count + wall on code KPI, then bench.
5. D1/D2 on kv_block (the real prize: 32-row copies are floor-cheap): block-diagonal
   batch submask in PKV_BLOCK phase, draft graph D<=5, START with SSD chain (graph
   degenerate case), add Spiffy calibration ONLY if chain acceptance < ~60% (paper
   says calibration is what lifts acceptance; our threshold commits may already be
   predictable enough that the chain suffices).
6. D4: revisit after E2 lands a measured tiny model. Not this layer's work.

## Risks (pre-resolved from prior layers + new)
- Paper multipliers vs OUR baseline: addressed in Context; expectations set at
  1.3-2x marginal for D1/D2, not 3.46x. If D0 shows no floor amortization: layer
  shrinks to D3a-or-nothing - that is a valid outcome, write it down and close.
- "Lossless" lives or dies on acceptance-rule fidelity: byte-identity gate per phase
  (we keep deterministic CLI refs from Layer C; extend with a threshold-mode
  multi-seed identity set). A bench run can NEVER bless a speculative path that
  fails identity - identity first, bench second.
- Multi-seq batches + degeneracy guard / EOT-shrink / window: all index absolute
  positions of ONE canvas. Per-seq state must be per-variant (cur_length, guard
  counters, mask_positions per seq). The B-4 row-bookkeeping crash is the cautionary
  tale: build row_of_pos PER SEQUENCE from day one, no arithmetic mapping.
- VRAM on 8 GB laptop (and 3-6 GB P106s): bigger ub = bigger compute buffers; measure
  llama_context memory at ub {512,768,1024} BEFORE promising draft racing. Server
  must keep working at ub 512 with D3a (3 small variants fit).
- Thermal/crash discipline (post-mortem 03_layer_c.md): bounded runs, temp checks
  between benches, findings written to disk before long GPU phases, never CLI while
  the server holds the GPU.
- Calibration (D2) is per-model, per-decode-config: a calibrated graph for Dream
  threshold-0.6 is invalid for DiffuCoder or tau-enabled runs. Store graphs keyed by
  (model, decode params); re-calibrate on config change. Skip D2 entirely if D1-lite
  acceptance is already high (Occam).

## Verification
- D0 probe: batched == sequential byte-identity (2 canvases, then 3 mixed-length).
- Speculative phases: token-for-token identity vs non-speculative path on the
  3-prompt matrix x 3 seeds BEFORE any wall numbers are reported.
- 14/14 sampler tests CPU+GPU; flags-off byte-identity on both CLI refs (Layer C
  protocol).
- Code KPI 3 seeds same-session; bench v2 vs same-day baseline (35/48 Dream at HEAD,
  33/48 DiffuCoder); i-tier latency tracked specifically for D3a.
- DG regression only if shared graph files (src/llama-graph.cpp, fork phase
  machinery) are touched - D5 kv-path work WILL touch them; budget one canvas run.

## EMPIRICAL GRILLING (2026-06-12): D0 probe built and run - plan substantially revised

Probe tool: examples/diffusion/diffusion-batch-probe.cpp (llama-diffusion-batch-probe;
committed as the D0 deliverable). Dream-7B Q4_K_M, 5070 laptop, ngl 99, ub 1024
(loads fine on 8 GB). Three probe-side bugs were found and fixed before the numbers
below became trustworthy - recorded because they are reusable traps:
(1) llama_decode is ASYNC - time through llama_synchronize or you measure kernel
    launches (0.6 ms "forwards" of a 7B);
(2) raw llama_get_logits ordering is unmapped for multi-seq batches - use _ith;
(3) diffusion.h defaults top_p=0 and the sampler chain adds top-p when < 1.0 - a
    direct diffusion_generate caller gets a top-p(0) chain = only top-1 survives =
    conf 1.0 = one-step flood. Pass 0.95 explicitly (the CLI/server always did).

### Finding 1: cross-seq isolation is EXACT - the de-risk is real
PROBE1d (killer test): seq0 = canvas A, seq1 swapped A -> random junk; seq0 logits
BIT-IDENTICAL (max|dlogit| 0.0, 0 argmax flips). The no-cache mask path
(llama-graph.cpp:443) is consumed by Dream's unified phase and isolates sequences
perfectly. Multi-canvas batching on the square path is SAFE. (Control PROBE1c with
both copies under ONE seq id shows ~7.8 dlogit contamination, as expected.)

### Finding 2: "byte-identity" across batch shapes is IMPOSSIBLE - revise the gate
The same canvas forwarded alone vs inside a 2-seq batch differs by up to 7.1 LOGITS
(1-2 argmax flips per 103 rows; FA on AND off; even the two identical copies inside
ONE batch differ from each other by ~6 - row offset changes quantized-matmul kernel
tiling/accumulation order). This is not a bug, it is Q4 CUDA numerics. CONSEQUENCE:
any speculative path that verifies in a different batch shape CANNOT be token-exact
vs the sequential path near ties. The plan's identity-first gate is revised:
identity within fixed shape only; batch-shape-crossing features gate on KPI + bench
(the Layer C protocol), and "lossless" claims are downgraded to "approximately
lossless" on this stack.

### Finding 3: square-path batching does NOT amortize at production sizes - NO-GO
Corrected timing (sync'd, 16 reps): 103-token canvas: K in {1,2,3,4,8} batched vs
sequential = 0.99x/1.00x/1.01x/1.00x/1.02x - NOTHING. Cost is linear in rows from
~100 rows up (fixed floor ~20 ms + ~0.25 ms/row + quadratic attention creep).
Tiny canvas (25 tok): K=2 1.44x, K=4 2.35x, K=8 2.40x - real amortization ONLY
under ~32-50 rows. VERDICTS:
- D3a repair-variant batching: REFUTED as planned (repair canvases are 60-150
  tokens = the no-gain regime). Do not build.
- D3b draft racing: DEAD (192+ token canvases, plus ub pressure).
- D1/D2 on the SQUARE path: DEAD (k+1 verify copies cost (k+1)x - speculation can
  never pay).
- D1/D2 on the KV-BLOCK path: ALIVE and now the layer's only real target - 32-row
  block batches are exactly the regime where K=4 costs 2.35x less than sequential.
  The kv-path microbench (K block-copies vs shared frozen store) is the new D0
  step 1; requires the custom-mask surgery first (block-diagonal batch submask).

### Finding 4: the speculation headroom is real and quantified
Commits-per-step histogram (3 chat-templated prompts, threshold 0.6, seed 3,
73 steps total): avg 7.9 commits/step BUT 53% of steps commit <= 3 tokens (26%
commit exactly 1). The end-game stall the papers attack exists in OUR decode too.
If chain-speculation converted even half the stall steps into 3-4-commit steps,
total steps drop ~25-35% - consistent with the 1.3-2x marginal estimate IF the
verify batch is near-free, which (Finding 3) it is only on the kv-block path.

### Revised implementation order (supersedes the list above)
1. KV-path microbench: K speculative 32-row block-copies vs the shared store
   (needs the block-diagonal batch submask in PKV_BLOCK phase - the same surgery
   D1 needs anyway; build the mask first, measure, THEN decide). GO if K=4 copies
   cost <= 2x one block step.
2. D1-lite SSD chain on kv_block: draft k=3 from stall-step probs (we already read
   them), k+1 block-copies, accept by replaying the commit rule. Gate: code KPI +
   bench (NOT byte-identity - Finding 2).
3. D2 Spiffy calibration only if D1-lite acceptance < ~60%.
4. D3a/D3b: CLOSED (Finding 3). D4: unchanged, E2-gated.

## SECOND GRILLING PASS (2026-06-12, same day): the kv question answered - layer PARKED

Probes 1e/5/6 added to the tool; all run on the 5070, Dream Q4_K_M.

### Finding 5: batch-shape numerics do NOT touch commit decisions (1e)
The single argmax flip in the 103-row identity test sits at confidence 0.128 - far
below any commit threshold. Zero flips at conf >= 0.5. The Finding-2 drift flips
only near-tie argmaxes that the threshold rule never commits; "approximately
lossless" is in practice "commit-stable across batch shapes". The bench gate
remains mandatory but is expected to be a formality.

### Finding 6: kv-path row-scaling is LINEAR - K=4 verify is NO-GO by the set bar (5)
Measured WITHOUT mask surgery (timing is mask-content-independent): WARM a 256
canvas, then BLOCK-phase decodes at W rows vs the shared store:
W=32 28.0 ms, W=64 43.0 (1.53x), W=96 55.1 (1.97x), W=128 67.9 (2.42x).
~18 ms floor + ~0.31 ms/row. The plan's GO bar ("K=4 copies <= 2x one block
step") FAILS at 2.42x. Only K=2 (1.53x) amortizes meaningfully.

### Finding 7: chain acceptance is strong at depth 1, dead beyond (6)
Oracle probe (greedy threshold replay of the counter task; predictions recorded
per step, scored when positions commit): Delta1 = 86% (122/142), Delta2 = 40%,
Delta3 = 32%. One-step-ahead drafts are predictable; two-plus are coin flips.
Spiffy's multi-level draft graphs (whose value is depth) are DEAD on this decode;
only k=1 chain speculation has fuel. Caveats: single task, greedy approximation,
and acceptance measured on positions that DID commit - an optimistic bound for
the stall steps speculation actually targets.

### Layer verdict: PARKED
The only configuration left standing is k=1 chain on kv_block with K=2 verify:
cost 1.53 steps, expected extra commits 0.86 -> ~1.2x on stall plateaus (53% of
steps), diluted to ~1.05-1.15x overall - BELOW every shipped layer's win, before
paying the block-diagonal mask surgery and per-variant bookkeeping it requires.
On the P106 target the economics worsen (compute-poor -> linear term dominates ->
less amortization, not more). DECISION: park Layer D; the catalog's E2 (tiny
model existence proof) is the higher-ROI next move, and a future model/hardware
where the floor share grows (or a decode whose stall fraction rises) is the
re-entry trigger. The probe tool stays in the tree - re-running it on new
hardware/models is the cheap re-evaluation path (one command, all seven probes).

## Unresolved questions (post-grilling)
1. D0 NO-GO threshold: ANSWERED EMPIRICALLY on the 5070 - square path NO-GO at
   production sizes, tiny-row (<~50) regime GO (Finding 3). P106 re-verify still
   owed before promoting anything to rig defaults.
2. D3a API shape: MOOT - D3a closed (Finding 3).
3. D3b racing scope: MOOT - D3b closed (Finding 3).
4. D1-lite draft length k: ANSWERED by the second pass - k=1 is the only economic
   config (K=4 verify costs 2.42x, depth-2+ acceptance 40%/32%). Moot while parked.
5. Chain order: MOOT while parked (a depth-1 chain has no ordering question).
6. kv-path block-diagonal submask: MOOT - the microbench was achieved WITHOUT the
   surgery (timing is mask-content-independent, Finding 6); the surgery is needed
   only if the layer is un-parked.
7. NEW - re-entry triggers for the parked layer: hardware/models where the fixed
   floor share grows, or decode regimes with higher stall fractions. CONCRETE
   CHECKPOINT: when E2's 1.5B tiny model lands, re-run the probe tool - per-row
   cost shrinks ~5x but the ~18 ms floor does not, so the amortizing regime may
   extend well past 128 rows, reviving D1-lite AND multi-canvas batching in one
   stroke. One command, all seven probes.
   CHECKPOINT EXECUTED (2026-06-12, Fast_dLLM_v2_1.5B Q4_K_M - see 05_layer_e.md):
   hypothesis REFUTED - the floor shrank WITH the per-row cost (tiny forward
   7.2 ms vs 27 ms; 103-tok K=2-4 batching 1.20-1.23x; W=128 = 2.78x of W=32).
   Layer D REMAINS PARKED on both models. Next trigger: a genuinely different
   hardware class (P106 measurement) or decode regime.
