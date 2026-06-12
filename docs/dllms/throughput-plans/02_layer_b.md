# Layer B: decoding schedules - PLAN / IMPLEMENTATION GUIDE

## Context

Layer B of docs/dllms/dllm-throughput-catalog.md: make the denoising SCHEDULE smarter -
when to commit, how many, where, and when to stop. We already own the single biggest
schedule win (fixed-threshold parallel commits + de-tempered calibration + EOT-shrink +
EOS early-exit + Layer A warm/quarantine machinery), so headline multipliers from papers
do NOT transfer; the verified incremental headroom is ~1.07-1.21x from adaptive tau plus
end-game savings from early-commit, plus a structural prize: SlowFast's span detection
is exactly the DYNAMIC BLOCK SIZING that Layer A's kv_block mode wants.

All references read from CLONED SOURCE (same method as Layer A):
/home/car/projects/dllm-references/streaming-dllm (xiaoshideta/Streaming-dLLM), /home/car/projects/dllm-references/slowfast
(LiangrunFlora/Slow-Fast-Sampling), /home/car/projects/dllm-references/remdm (kuleshov-group/remdm).

## Reference algorithms as actually implemented

### B1. Adaptive threshold tau(t) - streaming-dllm Dream/model/generation_utils_block.py
:157-165 (context_aware_threshold), used at :539:
    mask_factor = 1.0 - confidence_alpha * (1 - mask_ratio)
    tau_eff     = tau0 * mask_factor
- mask_ratio = remaining-mask fraction measured FROM THE CURRENT BLOCK START to the end
  of the generation region (:538: current_masks / x[:, block_start:].shape[1]) - NOT
  whole-canvas. Decays tau from tau0 (all masked) toward tau0*(1-alpha) (none masked).
- Their commit loop: top-1 always commits; candidates k>=1 are DROPPED below tau_eff
  (:545-547) - exactly our threshold branch with a sliding tau.
- Paper sweet spot: alpha 0.6 (+1.21x), alpha > 0.6 degrades; their initial tau0 0.9.
- OUR mapping: replace the params.conf_threshold read with a per-step tau_eff; r_mask =
  mask_positions.size() / (scan_hi - kv_s_or_n_input). Our confidences are DE-TEMPERED
  probabilities in (0,1] - same scale as theirs (softmax probs); formula transfers as-is.

### B2. SlowFast - slowfast/sampling_utils/generate_slow_fast_sampling.py
Three phases per block ("sub-cycle"):
- Phase 1 EXPLORE (k_exploration_steps=6): commit 1 token/step (+ extra fills above
  high_confidence_threshold); each step estimate the span length = last index above
  cycle_len_confidence_threshold=0.3 (contiguous-ish from block start); declare the
  span DETERMINED when the estimate is stable (std-dev over a 2-step window < 1.0).
- Phase 2: cache features for the determined span (their dLLM-Cache integration).
- Phase 3 ACCELERATE: aggressively fill the determined span in parallel.
- THE INSIGHT FOR US: SlowFast is dynamic block sizing by confidence-span detection.
  Our fixed kv_block=32 becomes a DETECTED span: at each WARM step, span = contiguous
  run from kv_s with confidence >= span_thresh; kv_e = kv_s + clamp(span, 8, 64).
  Their exploration/stability machinery is largely subsumed by our warm steps (exact
  forwards) already producing full-canvas confidences for free.

### Suffix tail-anchor (catalog C1 correction) - streaming-dllm Dream/model/sampler.py:3
    deterministic_window_tail_sampler(tokens, window=32, tail_keep=3)
Their suffix pruning keeps the first 32 suffix tokens PLUS THE LAST 3 ("final anchor").
Our --diffusion-kv-window drops the anchor entirely - their ablations say the anchor
matters (the model needs to see "the end exists"). Cheap fix: always include the last
A tokens of the canvas in windowed batches (A=3).

### B4. Prophet early-commit - arXiv:2508.19982 (paper only, no repo needed)
When the decision is "already known" - top-1/top-2 confidence gap large for ALL
remaining masks - commit everything and stop. We have sorted top-k probs per masked row
(backend sampler: candidates sorted by logit, probs aligned) so gap = p[0] - p[1] is
free. End-game steps are pure overhead today; this kills them.

### B8. ReMDM remasking - remdm/diffusion.py:735-760 (remdm-conf / remdm-loop)
sigma (remask prob) per COMMITTED position = softmax over negative commit-confidences
x sigma_max; masked positions sigma=0; remasked positions get conf=-inf (will be
re-predicted). The loop variant gates remasking to a mid-schedule window (t_on/t_off).
OUR adaptation (simpler, deterministic): on steps where a committed position's CURRENT
top-1 token differs from its committed token with margin p1 - p_committed >= m, remask
it (budget <= N/step, never the prompt, never EOG-tail). Quality play: catches early
wrong commits that today only kintsugi repairs post-hoc. Metric = bench C-tier/pass
movement + repairs-per-task, not raw speed.

## Engine integration map (all in examples/diffusion/diffusion.cpp threshold branch)
- tau_eff: computed right before the commit loop (where conf_threshold is read);
  needs mask_positions.size() (have) and the region size (scan bounds in scope).
- Prophet gap: in the confidence loop we already read sampled probs per masked row;
  track min_gap = min(p0 - p1); after loop: if min_gap >= gap_thresh -> commit ALL
  sampled tokens (respecting EOG quarantine: only on warm/uncached steps - on cached
  steps set kv_warm_needed and let the warm confirm) -> loop ends naturally.
- Span sizing: in the kv block-advance section, after a WARM step's confidences are
  available (cache them from the commit loop), set kv_e dynamically instead of
  kv_s + kv_blk_sz.
- Remask: after the commit loop, scan committed non-prompt positions present in the
  batch (prefix mode decodes the whole suffix - rows exist); compare row top-1 vs
  committed token; remask up to budget. mask_positions/conf arrays already absolute.
- Tail anchor: batch construction (kv_batch_end section) gains "+ last A rows".
New flags (diffusion.h + arg.cpp + server passthrough, defaults OFF):
  --diffusion-tau-alpha F (0=fixed tau), --diffusion-early-commit F (top1-top2 gap,
  0=off), --diffusion-kv-span F (span threshold, 0=fixed blocks),
  --diffusion-kv-anchor N (default 3 when kv_window>0), --diffusion-remask F (margin,
  0=off) + --diffusion-remask-budget N (default 2).

## Implementation phases (each: implement -> 3-prompt matrix -> multi-seed finals ->
## bench v2 gate -> guide log; Layer A lesson: NEVER trust single-seed deltas)
- B-0: refresh baselines at HEAD (matrix + bench, both models).
- B-1: adaptive tau. Sweep alpha {0.3, 0.6, 0.8} x tau0 {0.6, 0.75, 0.9} on the matrix;
  finals on code KPI; bench gate (expect small wall win, pass-rate must hold).
- B-2: Prophet early-commit. Sweep gap {0.3, 0.5, 0.7}; compose with B-1 winner.
  Expect end-game step savings, biggest on short/structured outputs.
- B-3: SlowFast span-sizing for kv_block. Code-KPI focus; compose with Layer A champion
  (kv_block 32 -> kv_block auto). This is the headline experiment of the layer.
- B-4: tail anchor for kv_window (quick; re-test the window configs that lost in
  Layer A sec 14.3 - the missing anchor may be WHY they lost).
- B-5: remask budget (quality play; gate = bench pass-rate/C-tier + kintsugi
  repairs-per-task; wall may be allowed to regress slightly if quality/sec wins).
- B-6 (deferred unless headroom remains): EB-schedule port to masked models.

## Risks / interactions (pre-resolved)
- tau decay must NOT bypass the EOG quarantine (quarantine is independent of tau ✓).
- Early-commit-all on a CACHED step would commit drift-tainted tokens: route through a
  warm (set kv_warm_needed, commit on the exact forward).
- r_mask region: use the active scan window (matches reference semantics under blocks).
- De-temper compatibility verified (same probability scale).
- Span sizing changes graph shapes per block -> phase-marker reuse machinery already
  handles it (Layer A 13.7).
- Single-seed sweeps are path-luck (Layer A 14.1) - finals always 3 seeds x {3,103,203}.

## Verification
- 14/14 sampler tests CPU+GPU after each phase; baseline path byte-identical with all
  new flags at 0.
- 3-prompt CLI matrix (haiku/story/code) per phase; multi-seed finals for any claimed
  winner; kintsugi bench v2 compare vs committed baselines (gates: no stable-tier pass
  loss, tier wall +10%).
- DG regression once per phase (shared files).
- Guide updated per phase with findings (the Layer A discipline).

## Unresolved questions
1. B-5 remask is a quality play inside a "throughput" layer - in scope now (recommended:
   yes, quality/sec is the mission metric and it may CUT kintsugi repair roundtrips) or
   defer to a quality-focused pass?
2. B-3 span sizing: clamp range {8..64} reasonable, or allow spans up to 128 for code?
   (recommend start {8..64}, widen if the detector saturates)
3. Sweep budget: full alpha x tau0 grid is 9 configs x 3 prompts - run all single-seed
   for shape, finals only for the best 2 (recommended) vs 3-seed everything (3x cost)?


## EMPIRICAL GRILLING (2026-06-12): B-1 pre-implemented as the probe; plan corrected

### Verified by code reading
- Prophet gap IS free: ggml_top_k = descending argsort (src/llama-sampler.cpp:1314 ->
  backend gather), so candidates[0]/probs[0] is the argmax and gap = probs[0]-probs[1].
- Sampled tokens/probs exist for ALL output rows (the backend chain samples every row;
  we only read masked rows today) - the remask prerequisite (B8) holds.
- CAUGHT PRE-IMPLEMENTATION: the plan said r_mask uses "the active scan window" - in
  non-kv mode the scan starts at 0 and includes the PROMPT, deflating r_mask and
  over-decaying tau from step 0. Correct denominator: the GENERATION region
  ([n_input, cur_length) non-kv; [kv_s, batch_end) kv). Fixed in the implementation.

### B-1 implemented (--diffusion-tau-alpha, default 0) and probed hard
- Code KPI, multi-seed {3,103,203}: alpha 0.3 = 21.4 s vs 73.1 s baseline (3.4x!),
  alpha sweep single-seed: 0.3 -> 7.2 s/31 steps, 0.6 -> 10.8 s, 0.8 -> 7.9 s
  (jagged = path-luck, as always). Haiku: neutral (short runs never reach deep decay).
- THEN THE QUALITY GATE: full bench v2 at alpha 0.3 = 13/45 vs baseline 33/45 -
  P-tier COLLAPSED 18/18 -> 2/18, infill 6/6 -> 2/6. VERDICT: REGRESSION (20 passes
  lost). The 3.4x code wall is worthless - completeness checks passed but correctness
  died. GLOBAL tau decay is poison.
- ROOT CAUSE (re-read of the reference with this lens): streaming-dllm applies the
  decay PER 32-TOKEN BLOCK - mask_ratio resets to ~1 at each block start, so tau only
  dips briefly at each block's end. Our global r_mask spends most of a long run deeply
  decayed -> sloppy commits everywhere. The naive global transfer of a block-scoped
  formula was the error.
- REVISED B-1 DESIGN: (a) block-scoped decay - apply tau decay with r_mask computed
  over the ACTIVE KV BLOCK (kv modes already have the structure; non-kv gets a virtual
  32-token window from the first remaining mask), and/or (b) tau floor
  (tau_min = tau0 * (1 - alpha) >= ~0.45). The flag stays (default 0) as substrate.
- Bench discipline note: the wall-only 3-prompt matrix can NEVER bless a schedule
  change - tau results must pass bench v2 BEFORE any multi-seed wall finals are even
  interesting. Phase ordering in this plan updated accordingly (quality gate FIRST).

### Infrastructure vulnerability found and fixed (would have poisoned all Layer B work)
At alpha 0.3 the model emitted `defmodule List` - Verifier.compile loaded it into the
harness VM, clobbering the Elixir stdlib; purge_candidate_modules then DELETED List and
the VM died mid-bench ({undef, List.ascii_printable?}). Fix shipped: Verifier.compile
now rejects any candidate defining an ALREADY-LOADED non-candidate module (AST prewalk
before compile; probe: `defmodule List` -> :error, fresh module -> :ok). The hardened
verifier survived a full 45-run bench of heavily degraded generations. Lesson recorded:
aggressive decoding states produce core-module redefinitions - the in-VM compile
shortcut needs this guard everywhere it exists.

### Status after grilling
- B-1: flag shipped (off), naive design REFUTED with data, revised design specified.
- B-2 (Prophet): prerequisites verified; unblocked.
- B-3 (span sizing): unchanged; warm-step confidences confirmed available.
- B-4 (tail anchor): unchanged.
- B-5 (remask): prerequisite verified (all-row sampling).
- Bench infrastructure: hardened (the real win of this grilling).


## EMPIRICAL GRILLING ROUND 2 (2026-06-12 afternoon): faithful B-1/B-2, three traps, real verdicts

### The traps this round caught (each now has a guard rail)
1. SILENT BUILD FAILURES: `grep -cE " error "` does not match gcc's "error:" - a failed
   build masqueraded as success and a STALE binary (new-layout lib, old objects)
   segfaulted in arg parsing, then later a stale SERVER silently ignored new request
   params. Rule now: builds are judged by EXIT CODE only; flag changes are smoke-tested
   with the flag before any measurement (the tell for the stale server: three "different"
   configs returning byte-identical token totals).
2. THE defmodulelerler RED HERRING: hours chased a "corruption" that was (a) reproduced
   via a CLI command that DOUBLE-TEMPLATES the prompt and (b) actually seed-42's normal
   ugly draft, repaired by the harness since the morning baselines. Rule: reproduce
   server-path issues with the server, byte-identical request.
3. THE GUARD FALSE-POSITIVE (real cause of the bench collapse): in-memory compiled
   modules report :code.which == [] - so (a) purge_candidate_modules had NEVER purged
   anything since kintsugi's first day, and (b) round-1's stdlib-redefinition guard
   rejected every RE-compile of a candidate, killing all repair ladders: bench 33/45 ->
   13/45 on the BASELINE profile. Fix: detect candidates via
   module_info(:compile)[:source] (carries the marker filename); repeat-compile
   regression test added. Probes: first/second/third compile :ok/:ok/:ok, defmodule
   List still rejected. Bench restored to 33/45 VERDICT OK.
4. INSTRUMENT GAP: Credence (path dep) had moved 5 commits mid-day and the bench header
   recorded only the llama.cpp rev - verdicts were not attributable. Header now records
   credence_rev + credence_dirty. (Credence itself was exonerated: identical 33/45 at
   its new tip once the guard was fixed.)

### REAL B-1/B-2 verdicts (correct binary, fixed guard, fresh baseline)
- Block-scoped tau (tau_alpha 0.6): code KPI 63.7 s vs 73.8 s multi-seed (-14%) BUT
  bench REGRESSION: p_max 3/3 + p_sum 3/3 lost (27/45). On small canvases the 32-token
  window covers most of the active region - effectively global decay again. Same
  content-length split as Layer A: helps long, hurts short.
- Prophet early-commit (0.5): bench REGRESSION on p_sum 3/3 (30/45) - commits a
  confidently-wrong draft the ladder cannot save. BUT: infill tier -35% wall at 6/6
  passes - PROPHET IS A REPAIR-PATH WIN. Action: kintsugi should set early_commit on
  REPAIR/infill requests only (they are short, hole-bounded, and verified afterwards);
  drafts keep it off.
- Both flags ship DEFAULT OFF. Status: B-1 = available, content-gated like kv flags;
  B-2 = adopt for infill in kintsugi (follow-up), off for drafts.
- BENCH GAP exposed: no passing-tier LONG-form case exists, so long-code quality under
  these flags is unverifiable by the bench (the C tier fails at baseline too). Before
  any code-targeted schedule flag is blessed, the bench needs a long-form case the
  models can pass (e.g. multi-clause but single-function). Added to the bench backlog.

### Where Layer B stands after two grilling rounds
B-1 implemented (block-scoped, floored, off); B-2 implemented (off; repair-path
candidate); B-3 span-sizing and B-4 tail anchor remain to implement with the same
gates; B-5 remask unchanged. The instrument survived two self-inflicted wounds and is
sharper for it: exit-code builds, dependency revs in headers, repeat-compile tests.


## EMPIRICAL GRILLING ROUND 3 (2026-06-12): adoption, B-3 verdict, the long-form limit

### Prophet-on-repairs: SHIPPED and gated
kintsugi's repair infill calls now pass early_commit 0.5 (lib/kintsugi.ex repair/4);
drafts keep it off. Full bench: 33/45, VERDICT OK, aggregate deliverable throughput
8.03 -> 9.02 tok/s (+12%) - round 2's discovery converted into a real harness win.

### B-3 SlowFast span sizing: implemented, measured, REJECTED as default
--diffusion-kv-span F (dynamic kv-block size = contiguous confident span at warm steps,
clamped [8,64]). Code KPI, 3 seeds, same session: fixed blk32 = 27.2 s, span 0.3 =
28.3 s, span 0.5 = 28.0 s. No win - our threshold commits + warm-step machinery already
capture what SlowFast's exploration phase buys; their published gains are vs quota
baselines. Flag stays (other models/shapes may differ); default 0.
Environment note: blk32-fixed measured 27.2 s here vs 24.9 s hours earlier - ~9%
inter-session drift (thermals). The same-session-comparisons rule holds for CLI
matrices, not just the bench.

### B-4 tail anchor: design constraint discovered by analysis (implementation deferred)
Anchor rows in BLOCK mode conflict with the store write (the graph writes batch rows
into store[s..s+n_tokens) contiguously - appended anchor rows would land at wrong
offsets). Anchor is therefore PREFIX-mode-only (no store write of batch rows) unless
the block write learns row indirection. Recorded for the implementer; the windowed
configs it would rescue were marginal anyway (Layer A 14.3).

### The long-form bench gap: probed, and the honest answer is a MODEL limit
Candidates at 3 seeds: two-trivial-functions 0/3 (the multi-function hole is absolute -
even c_to_f + f_to_c together fail); guarded sign/1 0/3; single-function-with-docs
(sum_to + @doc + @spec, ~58 tokens) 2/3 - the ONLY long-ish candidate Dream sometimes
passes. Added to the suite as m_sumdoc (tier :m, boundary; additive change - compare.exs
pairs by id, old baselines remain valid; self-test 14/14). Conclusion: Dream-7B has no
reliably-passing long-form tier; full long-form quality gating awaits DiffuCoder
baselines for tier m/c or a better model. Until then code-targeted schedule flags are
gated by m_sumdoc + C-tier movement + manual review.

### Layer B exhaustion status after three rounds
- B-1 adaptive tau: implemented (block-scoped + floor), default off; -14% long code,
  small-canvas regression; content-gated use only.
- B-2 Prophet: implemented, default off; ADOPTED on kintsugi repair path (+12%
  aggregate); regresses drafts.
- B-3 span sizing: implemented, measured, no win, default off.
- B-4 tail anchor: deferred with a recorded design constraint.
- B-5 remask: not implemented (quality play; prerequisites verified; next candidate).
- B-6 EB port: deferred.
Layer B's honest yield: the schedule layer was largely ALREADY CAPTURED by our existing
threshold machinery; the real wins were Prophet-for-repairs (+12% aggregate) and the
instrument hardening that the hunt forced (exit-code builds, dep-rev headers,
repeat-compile guard, boundary tier).


## LAYER B COMPLETE (2026-06-12): B-4 + B-5 implemented, layer closed

### B-4 tail anchor: implemented twice, hypothesis REFUTED
First implementation (appended detached anchor rows) broke instantly: non-monotonic
batch positions violate the example's row==batch-index assumption (engine may reorder;
observed garbage commits + guard abort at step 1 - "```imiterdociter"). RULE FOR
IMPLEMENTERS: batches must stay position-contiguous unless row indirection is added.
Second implementation (contiguity-safe: the window SNAPS to the canvas end when within
kv_anchor+8): measured 39.7 s vs 39.7 s without anchor (code, 3 seeds) - identical.
The anchor hypothesis ("end-visibility is why windows lost") is REFUTED for our stack;
the windowed prefix mode's loss vs plain prefix (39.7 vs 26.3 s) has other causes
(restricted commits). --diffusion-kv-anchor ships (default 3, only active with
kv_window) as the safe variant.

### B-5 remask self-correction: implemented, gated OK, quality-neutral here
--diffusion-remask F (margin) + --diffusion-remask-budget N (2/step, 16/run cap).
Mechanics: exact steps only; positions masked at the step's start exempt (their commits
reflect current logits - prevents dist-sampling thrash); EOG/pad never remasked; row
bounds guarded. Bench at margin 0.3: VERDICT OK, 35/48 (incl. new m-tier 2/3), p-tier
walls -13.8% median. No measurable quality GAIN on the current suite - the suite's
failures are capability-bound, not wrong-commit-bound. Ships default OFF; its expected
home is as a safety net under aggressive schedule configs (tau/early-commit) and weaker
models - re-evaluate when those are in play.

### B-6 EB-schedule port: CLOSED as not-pursued
Three rounds established the schedule layer's headroom is thin on top of our threshold
machinery; the EB port is medium-effort with no evidence of upside. Revisit only if a
canvas-model (DiffusionGemma-style) workload becomes primary.

### Final Layer B scorecard
| item | status | measured outcome |
| B-1 adaptive tau (block-scoped + floor) | shipped, off | -14% long code; small-canvas regression; content-gated |
| B-2 Prophet early-commit | shipped, off; ON for kintsugi repairs | +12% bench aggregate (8.03->9.02 tok/s) |
| B-3 span sizing | shipped, off | no win vs fixed blocks (28.3 vs 27.2 s) |
| B-4 tail anchor (snap variant) | shipped, default 3 w/ window | refuted hypothesis; neutral |
| B-5 remask | shipped, off | quality-neutral safety net; gated OK |
| B-6 EB port | closed | not pursued |
Durable yield: Prophet-on-repairs (+12% aggregate - the layer's one production win),
five default-off tuning knobs with honest measurements, and another round of
instrument/methodology hardening. Regressions at close: 14/14 sampler tests, 12/12
offline kintsugi tests, bench baseline intact.
