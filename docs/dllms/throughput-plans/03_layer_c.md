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

### Updated plan status (final, all phases measured)
- C1a: DONE (flag --diffusion-window; recommendation: 64 for >=384-token generations).
- C1b: CLOSED (structurally unnecessary; see instrumentation finding).
- C4: DONE as engine feature (--diffusion-gen-initial / "gen_initial", default off);
  REJECTED for kintsugi drafts (grow 27/48, big384 30/48 vs baseline 35/48 - canvas
  geometry is a quality knob; n_gen 192 + redraft-doubling stays). See C4 section.
- C5: REJECTED (slim 28/48, mid catastrophic; full wrapper stays). See C5 section.
- C6: DONE as opt-in ("multi_hole"); pass-neutral and DORMANT on Dream bench
  (parse cascades produce one diagnostic at a time). See C6 section.
- Layer C net (after second pass): ONE production win (C1a window 64 = 3.67x on
  long code), one engine capability banked default-off (C4 - mechanism-mismatched,
  see second-pass section), three hypotheses refuted cross-checked (C5; C6 dormant
  on BOTH models; window-x-kv composition worse at HEAD too). Final recommendation:
  long-code generation wants window 64 (3.67x) OR kv_block 32 (3.08x), NOT both -
  the exclusion is empirical now. LAYER C IS CLOSED; bench profiles slim/mid are
  retired from the runnable set (crash hazard + verdict recorded).

### Adoption attempt: window-64 content routing in kintsugi - REJECTED by bench
The last executable piece: nothing in production sent window 64, and the one place
kintsugi reaches >=384-token canvases is redraft-doubling (16/48 bench cases reach
a doubled draft). Implemented as opt-in routing ("win_route": retry attempts whose
effective n_gen >= 384 add window 64; first drafts and explicit "window" callers
untouched), verified live (window engaged on attempts 2-3 only), bench-gated
same-process: baseline 35/48 (4th reproduction today) vs winroute 33/48 -
p_reverse flips to fail on BOTH seeds, and the redraft population's wall INFLATES
61% (119.6 s -> 192.6 s). Mechanism: the KPI's 3.67x needs output that actually
FILLS the long canvas; a doubled redraft still writes a ~100-token answer, so the
window restricts commit visibility on a canvas EOT-shrink would collapse anyway -
step inflation, no row savings that matter. VERDICT: routing stays opt-in default
OFF ("win_route" kept for future models); kintsugi adopts NOTHING from Layer C.
window 64's win is real but belongs to genuinely long OUTPUT (CLI / raw server
long-form code generation), not to kintsugi's short-answer draft workload. The
recommendation table's "content-aware" gate means OUTPUT length, which the harness
cannot know in advance - the engine-side flag remains the right delivery vehicle.

### C5 prompt slimming: MEASURED, REJECTED (recovered from crashed sessions)
Two sessions died mid-C5 (see crash post-mortem below); results recovered from
kintsugi/bench/results/*c5-*.jsonl and the surviving working-tree diff (forge_wrapper
variants in kintsugi.ex + slim/mid bench profiles).
- Wrapper token counts (from the implementation comment, /tokenize-probed): full
  wrapper 26 tokens -> slim 13; full templated prompt 58 -> 45 for a typical bench
  task (~5% fewer rows/step on a 192-token draft). The saving is real but small.
- BENCH GATE (48 cases): baseline 35/48, reproduced TWICE same-session (matches the
  committed baseline exactly). slim ("Reply with only an ```elixir code block.")
  = 28/48: p-tier 18->13, m-tier 2->0, wall 148 s -> 255 s (failed drafts cascade
  into repairs). mid (full sentence, dropped "no explanation") = 8/18 p-tier at
  truncation, IDENTICALLY in both runs - deterministic collapse, not noise.
- VERDICT: REJECTED. Draft quality is extremely sensitive to wrapper phrasing; the
  ~5% row saving cannot survive a 7-pass quality loss. The full 26-token wrapper
  stays. CLOSED (do not revisit without a fundamentally different prompt strategy;
  any retest must avoid the mid profile - see post-mortem).

### C4 in-run canvas growth: IMPLEMENTED, MEASURED - growth REJECTED for drafts,
### but it exposed a better flag-free policy (session 3, after recovery)
Implemented as planned (--diffusion-gen-initial / server "gen_initial"; square
threshold path only, +64 growth while masks < 8 and no committed EOG; pre-filled
masks make growth a pure bound raise). Verification: baseline byte-identical with
the flag off (deterministic AND threshold CLI refs), 14/14 sampler tests CPU+GPU,
kv-gating warning fires, EOT-shrink composes (haiku: started 121, shrunk to 33, no
guard abort), no mask leak (mask piece detokenizes to empty with special=false),
growth fires when needed (gen_initial 48 counter: grew once 91->155, complete code).
- KPI probes (server-side, 3 seeds, same session) split BOTH ways by content:
  counter task: n_gen 384 baseline 1.66 s vs gi96 2.10 s (+25% - step inflation, 24
  vs 15 steps; the small canvas commits slower). Stack task: baseline 5.53 s vs gi96
  2.74 s (-50%). Single prompts cannot settle it; bench only.
- KEY MECHANISM FINDING: allocation is already nearly FREE - n_gen 192 (1.72 s) ==
  n_gen 384 (1.66 s) on the same task, because EOT-shrink collapses the canvas a few
  steps in. The C4 premise ("the allocation sets per-step cost") is FALSE at HEAD;
  EOT-shrink already harvested that win.
- BENCH GATE grow (n_gen 384 + gen_initial 96): 27/48 vs baseline 35/48 - REJECTED
  for drafts (p-tier 12/18 vs 18/18; the universal content-length split again: a
  96-start draft is a worse draft, exactly like slim prompts and win64 drafts).
- COROLLARY tested and ALSO REJECTED: "big384" (n_gen 384 flat, no growth) = 30/48
  vs 35/48 - it breaks the same cases (p_sum 3/3 seeds, m_sumdoc 2) minus p_max. The
  "allocation is free" finding holds for WALL but not QUALITY: the model sees the
  mask count and plans against it; draft geometry is a quality knob and n_gen 192 is
  the tuned spot. Failures are seed-deterministic collapses (1 s passes -> 10-28 s
  repair cascades), the same signature as slim/mid.
- C4 VERDICT: engine feature DONE and correct, ships DEFAULT OFF like window/kv;
  kintsugi draft policy UNCHANGED (n_gen 192 + redraft-doubling stays). Remaining
  plausible use: long generations (>=384 wanted tokens) where the stack-task probe
  showed 2x - same content-gated family as window 64 / kv_block 32; composition with
  window is the open experiment.

### C4 second-pass review (same day): verdict UPGRADED to "mechanism mismatched"
A deeper probe round invalidated even the residual "long generations" hope:
- Across ALL 76 grow-enabled server generations (grow bench + probes): grew 0
  time(s), every single run. The stack-task "2x win" above also grew 0 - it was
  path perturbation from the smaller start, not growth.
- STUB ADAPTATION, the root cause: gen_initial 32 on the 7-function Stack task
  produces a complete-LOOKING module whose bodies are "# stack implementation"
  comment stubs, EOG'd neatly inside the 32-token window (2 seeds, identical
  behavior). The model PLANS THE ANSWER TO FIT THE VISIBLE CANVAS - canvas size is
  effectively a length prompt. Growth-by-frontier-pressure assumes autoregressive
  overflow demand; masked diffusion shapes demand to supply, so the trigger
  (frontier near end, no EOG) is starved by construction. Prose prompts EOT-bail
  even on the full canvas (story task: 0 words at 512), so no regime exercises it.
- The ONLY observed firing remains the gi-48 counter probe (structural momentum of
  unclosed code outran the plan by a hair: one grow, 91->155).
- Degeneracy guard with growth: 0 aborts in all 76 grow-enabled bench generations
  (plan risk "C4 must not fight the guard" - no false positives observed; the v3
  criterion is bound-independent by design).
- window + gen_initial compose without geometry errors (244 rows pruned + growth
  bookkeeping + EOT-shrink in one run, clean termination).
- FINAL: C4 stays in the tree as a correct, default-off curiosity. Do NOT spend
  further tuning effort on growth triggers; any future "small active canvas" work
  must counter stub adaptation FIRST (e.g. length-hint prompting or schedule-side
  length control), which is a different research problem.

### C6 multi-hole repair: IMPLEMENTED, MECHANISM VERIFIED - DORMANT on bench v2
Implemented opt-in ("multi_hole" => cap, bench profile mh2; cap 2 per unresolved
Q2): on a FRESH error (same-line streak 0), >= 2 semantic error diagnostics on
distinct non-adjacent lines (line >= 2 - the header rule keeps line 1; check-script
lines fall out via the > code-length filter; the rescued generic CompileError lands
on line 0 and falls out too) get masked in ONE canvas - per-line tokenized hole
sizes - one infill, one verify; broken results fall back to the single-hole ladder.
- E2E probe (3 undefined helpers): fired as designed (history "multihole:[2, 4]",
  one infill, 1.8 s vs 2.7 s single-hole on the same state).
- BENCH GATE mh2: 35/48 == baseline, wall 146.7 s vs 145.5 s (noise), total repairs
  62 == 62, ZERO per-case repair-count changes -> multi-hole NEVER FIRED across 48
  cases. Dream-7B bench failures are PARSE-cascade dominated (the parser stops at
  the first error, one diagnostic per round); semantic multiplicity - the C6
  precondition - is absent from this model's real failure distribution. The offline
  probe that motivated C6 (3 undefined fns -> 4 diagnostics) was synthetic.
- VERDICT: ships as opt-in, default off, zero risk (pass-identical when dormant).
  Re-check on DiffuCoder (different failure modes) before investing further; the
  "fewer roundtrips on p_double cascades" expectation is REFUTED for Dream (those
  cascades are parse errors, by construction outside C6's reach).
- DIFFUCODER RE-CHECK DONE (second pass, same day): fresh baseline at HEAD 33/48
  (committed to baselines/, replaces the stale 45-case one), mh2 33/48 - repairs
  63 == 63, zero flips, zero per-case repair-count changes. Multi-hole NEVER FIRED
  on DiffuCoder either. The dormancy is now CROSS-MODEL: by the time a real draft
  PARSES, at most one semantic error line remains (truncation/fence damage fails at
  the parser, which reports one error per round and serializes the ladder). C6's
  precondition is structurally absent from dLLM draft pathology, not just Dream's.
  CLOSED - keep the opt-in, spend nothing more here.

### Verification record (this session)
- 14/14 sampler tests CPU+GPU at HEAD and again (GPU) after the C4 engine change.
- Baseline byte-identity with new flags off: deterministic CLI ref (eps schedule,
  temp 0, seed 7) AND threshold-mode ref both byte-identical pre/post C4. At bench
  level: mh2 run reproduced baseline walls per-case through the C4-modified binary.
- C5 wrapper token counts confirmed via /tokenize: full 26, slim 13.
- Mask piece detokenizes to empty (special=false) - C4's never-grown tail cannot
  leak into response text (probed directly: no "<|mask|>" in grown outputs).
- DG regression: satisfied structurally - git diff shows zero shared-engine changes
  (src/, ggml/ untouched; diffusion_generate_entropy_bound byte-identical; DG
  binaries rebuilt cleanly). No behavioral DG run needed this round.
- Baselines: c5-baseline (35/48, 148 s) and c5-baseline2 (35/48, 145 s) from the
  crashed sessions match the committed baseline - reproducibility across crash
  boundaries confirmed.

### Unresolved questions: ANSWERED
1. C4 server default: 0 (off) - implemented; moot for kintsugi (growth rejected
   for drafts entirely).
2. C6 cap: 2 - implemented; hole-interaction risk unmeasurable on Dream bench
   (never fires); revisit with DiffuCoder.
3. C1a+kv mutual exclusion: kept for v1; window-x-kv composition remains the top
   open experiment for long-code generation (3.67x vs 3.08x candidates).
   SECOND PASS: composition MEASURED at HEAD and CLOSED. Motivation for re-test:
   the Layer A "dual+w64 worse" verdict predated guard v3, and guard v2 was a known
   false-positive source under kv+window combos. Result at HEAD (CLI, stack task,
   seed 3): blk32 2.46 s/40 steps, blk32+kvw64 4.05 s/47, blk32+kvw128 6.22 s/63 -
   monotonically worse with window size, same direction as the old verdict, now
   clean of the guard-v2 confound. Mechanism: dual mode's per-step batch is the
   BLOCK (tiny); a lookahead window multiplies per-step rows for steps it rarely
   saves. The window family's value is exclusive to the square path; kv dual stays
   pure-block; the mutual exclusion is now EMPIRICALLY justified, not v1 caution.
   (Caveat: raw-CLI prompts are path-jagged on Dream - off/win64 EOT-bailed on this
   task - but the dual-mode ordering is internally consistent and matches history;
   not worth 3-seed finals on a flaky elicitation.)

### Crash post-mortem (2026-06-12, two dead sessions)
Both deaths occurred during C5 `mid`-profile bench runs, ~4-5 min into sustained
back-to-back repair cascades (run 1: 285 s of cascades, died entering h_fib; run 2:
259 s, died during the third c_stack). One crash took the whole machine down
(reboot 15:14). The desktop shares the 8 GB GPU, so a driver hang/reset kills
VSCode and the session with it. Normal bench runs (~150 s, natural idle gaps
between requests) completed many times the same day without incident: the hazard is
SUSTAINED pathological load (low-pass-rate profiles whose every task spirals into
max-length repair loops), not any single case. Rules going forward: never re-run
the C5 mid/slim profiles; write findings to disk before heavy GPU work; spot-check
GPU temperature during long runs.
