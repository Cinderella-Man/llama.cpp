# Layer C: canvas-compute reduction - PLAN / IMPLEMENTATION GUIDE

## Context

Layer C of docs/dllms/dllm-throughput-catalog.md: stop feeding positions through the
model that cannot influence the answer this step. We already shipped two members of
this family (EOT-tail shrink; EOS early-exit in kv mode) and learned hard lessons from
two adjacent failures (Layer A's kv_window lost via commit restriction; Layer B-4's
detached anchor rows broke row bookkeeping). Layer C finishes the family on corrected
foundations: a contiguous suffix window on the SQUARE path, the row-map refactor that
makes non-contiguous batches legal, in-run canvas growth, and two harness reductions.

Status of catalog items up front (no re-research):
- C2 EOS early-exit: DONE (kv mode) + EOT-shrink covers the non-kv tail. CLOSED.
- C3 prefix shrink: only valid WITH a cache (dropping committed prefix from a square
  forward loses real context) - SUBSUMED by Layer A prefix mode. CLOSED.
- C7 attention sparsity: research-grade, no fit for dp4a/Pascal target. CLOSED.
- C8 Rainbow-padding caveat (premature EOS at long max_length): our EOG quarantine +
  pressure-warm machinery already guards the cached path; non-kv relies on the
  degeneracy guard. NOTE carried into validation (watch short-canvas EOS behavior).

## Reference algorithms (read from cloned source)

### DPad (arXiv:2508.14148) - /home/car/projects/dllm-references/dpad/dream/model/generation_utils_block.py:87
    suffix_dropout(x, sampler, block_end):
      q_indices = [0..block_end) DENSE  ++  sampler.sample([block_end..L))
- The suffix is kept STOCHASTICALLY, not hard-cut: GaussianSampler (sampler.py:40)
  keeps suffix positions with probability decaying over a window (sigma/scale knobs,
  defaults window=128); UniformSampler keeps N random within the window. Everything
  before block_end stays dense; queries == keys == the kept subset.
- Mechanically: a NON-CONTIGUOUS gather with explicit position indices - the third
  reference (after streaming-dllm tok_idx and Fast-dLLM replace_position) that keeps
  row identity via explicit per-row position ids. Our engine's graphs are already
  position-explicit (batch.pos drives rope; non-kv masks are square over the batch);
  ONLY our example's row<->pos arithmetic (get_row_for_pos) assumes contiguity.
  B-4's crash was an example-code limitation, not an engine one.

### Streaming-dLLM suffix window - generation_utils_block.py:463
  tok_idx2 = cat(dense_prefix, window_tail_sampler(arange(block_end, L), window, tail_keep=1))
Same pattern: dense up to the active block, windowed suffix + 1 tail anchor, explicit
indices. Their +1.73x (and DPad's 1.18-3.91x) are vs THEIR baselines; our EOT-shrink
already harvests the committed tail, so our headroom is the UNCOMMITTED distant masks
on long canvases (early steps of the 512-canvas code case: ~460 masks, only ~128 near
the frontier matter).

## Engine integration map (verified against our code)

All in examples/diffusion/diffusion.cpp; the graphs need NO changes for C1a/C4 (the
square no-cache path takes any contiguous batch; rope uses batch.pos).

- C1a contiguous suffix window (square path, kv OFF):
  batch_last = min(cur_length, max(first_mask + W, last_nontail_committed + 1))
  where first_mask = mask_positions.front() (sorted ascending) and
  last_nontail_committed = highest committed non-EOG position (committed islands beyond
  the frontier must STAY in the batch - they are real context; the contiguous rule
  extends the window to cover them rather than dropping them). Commits scan the whole
  batch (scan_hi = batch_last) - whole-window commits, the Layer B lesson (restricting
  commits inflated 15-step runs to 128). EOT-shrink composes: it shrinks cur_length
  (the committed tail), the window shrinks the UNcommitted frontier - different ends.
  Flag: --diffusion-window W (0 = off), independent of kv flags; v1 mutually exclusive
  with kv modes (their batch windows already exist), lift later if C1a wins.
- C1b row-map refactor (the enabler): replace get_row_for_pos arithmetic with a
  row_of_pos vector filled during batch construction (pos -> batch row; -1 = absent)
  and use it in get_logits_for_pos/sampled reads/remask/early-commit. ~30 lines,
  unlocks non-contiguous batches (DPad gaussian sampling, true far anchors, committed
  islands without window extension). Implement AFTER C1a is measured - only if the
  contiguous approximation leaves measurable headroom (count: how often does
  last_nontail_committed extend the window far beyond first_mask + W?). Instrument
  C1a to LOG that distance.
- C4 in-run canvas growth: split "active length" from capacity. params.max_length
  stays the allocation (n_input + n_gen, capped by ub); new params.gen_initial
  (0 = start full): cur_length starts at n_input + gen_initial and GROWS by +64 when
  (remaining masks < 8 && no committed EOG anywhere) until max_length. Arrays are
  already sized max_length (output_tokens, confidences); output_tokens beyond
  cur_length are pre-filled with masks at init - growth is just raising cur_length.
  Interplay: EOT-shrink may shrink below gen_initial (fine); kv modes: defer (geometry
  churn), v1 square-path only. Harness: kintsugi can then send n_gen = 384 with
  gen_initial 96 instead of redraft-doubling (a full redraft costs a draft + ladder;
  growth costs ~nothing). Server params: "gen_initial".
- C5 prompt slimming (kintsugi): the forge wrapper is
  "Write Elixir code for the following task. Reply with ONLY a single ```elixir code
  block, no explanation.\n\nTask: " (~25 tokens) + chat template overhead (~25).
  Probe actual token counts via /tokenize; try a ~10-token wrapper ("Reply with only
  an ```elixir code block.\nTask: ..."); gate on bench pass-rate (prompt changes can
  shift draft quality!). Expected: ~10-20% of per-step rows on n_gen 192 drafts.
- C6 multi-hole single-canvas repair (kintsugi): the engine fills ALL mask runs in one
  infill call (verified long ago - mask markers anywhere). Today repair/4 fixes ONE
  diagnostic per round; with multiple independent diagnostics (different lines),
  mask all their lines in ONE canvas, one infill, one verify. Care: Verifier returns
  the FIRST error; collecting multiple requires compile diagnostics list (it already
  returns a list!) - use all distinct lines (cap 3), skip if lines adjacent (one hole).
  Gate: bench h-tier + repairs-per-task; expect fewer GPU roundtrips on multi-error
  drafts (the p_double cascade case).

## Implementation order (each: implement -> 3-prompt matrix + code KPI multi-seed ->
## bench v2 gate -> doc log; builds judged by EXIT CODE; same-session comparisons)
1. C-0 baselines refresh at HEAD (matrix + bench).
2. C1a window (engine flag + scan; instrument window-extension distance for the C1b
   decision). Sweep W {64, 128, 256} on code KPI; haiku/story sanity; bench gate.
3. C4 canvas growth (engine + server param); kintsugi switches draft policy to
   gen_initial 96 / n_gen 384; bench gate (drafts may change paths - expect verdict
   noise review, pass-count is the gate).
4. C5 prompt slimming (kintsugi-only; bench gate).
5. C6 multi-hole repair (kintsugi-only; bench gate + repairs-per-task metric).
6. C1b row-map refactor ONLY if C1a's instrumentation shows the contiguous rule
   regularly forfeits > ~25% of prunable rows; then optionally DPad-gaussian keep.

## Risks (pre-resolved from prior layers)
- Commit restriction == step inflation: avoided by whole-window commits + the
  window>=frontier+W rule (the model always has W tokens of open frontier).
- Non-contiguity: forbidden until C1b (B-4 lesson recorded in 02_layer_b.md).
- Degeneracy guard + EOS machinery index absolute positions - window only changes
  batch_last, scan_hi; guard_end stays as-is for non-kv (max_length) - VERIFY during
  implementation that a shrunken batch never feeds guard counters with stale data.
- Window + threshold + de-temper unchanged (same commit machinery).
- C4 growth must NOT fight the degeneracy guard (growing adds masks - the guard's
  masked-count denominator changes; re-check its bounds with growth on).
- Single-seed jaggedness: finals always 3 seeds; bench is the quality gate (Layer B's
  blood-won process rules apply unchanged).

## Verification
- 14/14 sampler tests CPU+GPU; baseline byte-identical with all new flags off.
- Code KPI 3 seeds {3,103,203} per config, same session as its baseline.
- kintsugi bench v2 vs committed baselines per phase (pass gates; m_sumdoc watched).
- DG regression once (shared engine files untouched in theory - verify).
- Guide updated per phase with findings, failures included.

## Unresolved questions
1. C4 default gen_initial for the server when client does not specify: 0 (off,
   current behavior) recommended - kintsugi opts in explicitly?
2. C6 cap of 3 simultaneous holes per repair round: reasonable, or start with 2
   (recommended: 2 - hole interaction in one canvas is unmeasured)?
3. C1a + kv mutual exclusion in v1: acceptable (recommended: yes - kv modes already
   window their batches; composing two window systems risks subtle geometry bugs for
   marginal gain)?


## EMPIRICAL GRILLING (2026-06-12): C1a pre-implemented; guard generalized (v3); probes

### Offline probes (cheap kills first)
- C6 VIABLE WITH CORRECTION: multi-error files yield MULTIPLE diagnostics for SEMANTIC
  errors (probe: 3 undefined fns -> 4 diagnostics, lines [4,3,2,0] - filter the line-0
  generic entry) but only ONE for PARSE errors (the parser stops). Multi-hole repair
  targets semantic multiplicity; parse cascades stay one-hole-at-a-time.
- C4 init claim VERIFIED in code: diffusion.cpp:186-188 pre-fills [n_input, max_length)
  with masks and confidences with -1 - growth really is "raise cur_length".

### C1a implemented and fought through three bugs (all now guarded)
1. Degeneracy guard false positive #3 (after Layer A's and the kv variant): the
   suffix-tail exclusion assumes full-canvas commits; ANY reduced-decode mode grows its
   EOT tail incrementally (haiku w128 aborted at step 0: "114/114 end tokens, 180/294
   masked" - all legitimate). FIX = GUARD v3, a structural generalization: an EOG token
   is "scattered" (degenerate signal) ONLY if committed TEXT exists at a later position;
   EOG runs bordering masks/undecoded space are tails in progress in every decode mode.
   One backward pass; works for plain/kv/window uniformly. TRADE: the all-EOT flood
   true-positive no longer aborts early - measured cost +1.8 s on the pathological
   config (EOT-shrink contains it); 3 false-positive classes fixed for one cheap loss.
2. Empty-scan termination: when the window's masks complete, the run ended with masks
   beyond batch_last. Fix: empty scan + masks-beyond -> continue (the window SLIDES on
   the recomputed frontier next step).
3. Instrumentation answered the C1b question STRUCTURALLY: committed islands beyond
   the window can never form (commits only come from decoded rows) - "max island
   extension" is 0 by construction. C1b (row-map refactor) is therefore NOT needed for
   island coverage; its only remaining motivation would be DPad-style stochastic
   sampling, which is now low-priority.

### Measured results
- W sweep (code KPI, s=3): off 22.6 s/96 st; w64 6.1 s/60; w128 7.0 s/59; w256
  degenerate-ish 12 steps (window ~= canvas -> pure path perturbation, pointless).
- MULTI-SEED FINALS (3 seeds): off 73.3 s, w64 20.0 s = 3.67x - THE BEST single-config
  result of the project so far (beats kv_block 32's 3.08x), on the plain square path
  with no cache machinery.
- Short-form: haiku w128 = 1.31 s/14 steps vs 0.34 s/8 off - the universal
  content-length split, again.
- BENCH GATE: win64 profile = 23/48 vs baseline 35/48 - small-canvas drafts collapse
  (the 64-window restricts commits on a 190-token region). VERDICT: window ships
  DEFAULT OFF, content-gated exactly like kv flags and tau: long generations only.
- Fresh baseline re-committed (guard v3 changes some paths; 35/48 incl. m-tier).

### Updated plan status
- C1a: DONE (flag --diffusion-window; recommendation: 64 for >=384-token generations).
- C1b: CLOSED (structurally unnecessary; see instrumentation finding).
- C4/C5/C6: probes done (C4 verified, C6 corrected to semantic-only, C5 pending the
  tokenize count) - implementation remains, now de-risked.
- The recommendation table consolidates across layers: long-code generation wants
  window 64 (3.67x) OR kv_block 32 (3.08x) - composition of the two is the obvious
  next experiment when implementation resumes (currently mutually exclusive in v1).
