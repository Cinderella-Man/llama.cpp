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

> NAMING COLLISION (read first): this file's PART 1/PART 2 below use F1/F2/F3 for
> the FastDLLM post-E3 follow-ups (seed racing / DFlash pairing / MTP). The
> THROUGHPUT CATALOG (dllm-throughput-catalog.md) Layer F uses F1..F11 for
> engine/system items (CUDA graphs, encode logits-flags, prompt/canvas caching,
> ...). The 2026-06-13 catalog-F planning round below uses CATALOG numbering and
> is prefixed "catalog-F" everywhere to disambiguate. The two namespaces do NOT
> map onto each other.

---

## CATALOG-F PLANNING ROUND (2026-06-13) - engine/system layer, kintsugi-first

Scope (user-confirmed): the catalog's Layer F engine/system items as they serve
the Dream + FastDLLM kintsugi workload. Flagship candidate was catalog-F10
(cross-request canvas cache). DG-specific items (catalog-F3/F4/F5) and rig-day
F8 parked unless DG/rig becomes an active target.

### Probe 0: the kintsugi-Dream wall decomposition (the input to ALL F decisions)

Method: env-gated request trace added to kintsugi Engine.post (KINTSUGI_TRACE=path
-> one JSONL line per /generate: infill?, prompt, n_gen, seed, ms_total,
n_prompt_tokens, text). Ran the full 48-case x 3-seed bench (Dream-7B Q4_K_M,
5070 AC, baseline profile) = 35/48, 6.46 tok/s deliverable, 421 engine calls,
125.9 s engine wall (147 s bench wall; the ~21 s gap is Credence/runner/verify).
Trace at /tmp/ktrace.jsonl; analysis inline below.

| class  | calls | engine wall | share | median ms |
|--------|-------|-------------|-------|-----------|
| DRAFT  |  58   | 52.1 s      | 41%   | 637       |
| INFILL | 363   | 73.8 s      | 59%   | 111       |

Infill ms distribution: min 32, p10 40, median 111, p90 554, max 958.
Draft prompt tokens: median 63, max 75 (n_gen default 192 -> ~255-tok canvas;
prompt ~25% of each forward).

### Probe 1: infill wall vs canvas size (the F10 "cacheable compute" number)

Controlled infill, 4-mask hole, fixed code context of N filler lines (server
direct, seeds 3/13/23, median):
- ~23 prompt tok  -> 133 ms
- ~292 prompt tok -> 782 ms   (120-line case 500'd: canvas > ubatch 512)
=> ~2.3 ms/token ABOVE an ~80 ms fixed floor. The UNCHANGED code (not the hole)
dominates a large repair; small repairs are floor-bound.

### Probe 2: cross-request redundancy (F9 prompt reuse / F10 canvas delta)

- F9: 58 draft calls, only 11 UNIQUE prompts; 47 calls re-send a seen prompt
  (the 3-redraft cascade on failing cases), 51.9 s of the 52.1 s draft wall.
- F10: 354 consecutive-infill pairs, char-overlap median 0.92 / p90 0.98;
  208/354 (59%) are >90% identical (hole-variant sweeps {n,n+2,1.4n} + repair
  retries on the same code).

### What the data says about each catalog-F item (VERDICTS)

- **catalog-F9 (cross-request prompt cache): DEAD.** Three independent blockers,
  all measured/established: (a) prompt is only ~25% of each forward (63/255 tok);
  (b) Dream is BIDIRECTIONAL - no exactly-static prefix KV exists (Layer A sec 1),
  so any prompt cache is approximate; (c) the approximate version IS kv_prefix,
  which Layer A measured as a NET LOSS on small canvases (bench +38% p-tier).
  F9 only persists that net-loss cache across requests - it cannot beat a thing
  that already loses. The 47 redrafted prompts are real redundancy but live on
  the wrong (small, bidirectional) canvas to exploit.
- **catalog-F10 (cross-request canvas cache): MARGINAL, ceiling ~10%, structurally
  hard.** Headroom exists ONLY on large repair canvases (p90 554 ms+, the ~2.3
  ms/tok regime) which come from FAILING cases; production median repair (111 ms)
  is floor-bound and uncacheable. The 59%-identical pairs are hole-VARIANT sweeps,
  but variants change mask COUNT -> shift the suffix's absolute positions -> rope
  KV differs -> naive position-keyed caching misaligns. The only clean F10 slice
  is same-position REPAIR-ROUND re-decodes (failed compile -> new seed, identical
  canvas geometry), a small subset. Needs A2 (dLLM-Cache feature caching) which
  was never built precisely because code workloads cache worst (HumanEval 1.36x).
  Best-case capture ~10% of bench wall for a whole new cross-request subsystem.
- **catalog-F2 (encode logits-flags fix): NOT a kintsugi-Dream item.** encode()
  forces output_all=true (llama-context.cpp:1531) + n_outputs=n_tokens (:1566) =
  lm_head over every row. But needs_raw_logits already skips the D2H COPY under
  backend sampling (:1595); the remaining win is the lm_head COMPUTE over prompt
  rows, marginal at Dream's 152k vocab / 63-tok prompts, and ZERO on infill
  (n_input==max_length there, all rows are output). It was a DiffusionGemma item
  (262k vocab, 2.3 GB compute buffer). Park unless DG is an active target.
- **catalog-F6/F7/F11 (CUDA graphs / host overlap / step-loop micro): marginal.**
  E6 already measured CUDA graphs = ZERO on 5070/AC (GPU-bound; launches hidden).
  Host overlap/micro are single-digit-% and do not touch the deliverable headline.

### Probe 3: per-step cost mechanics (CORRECTS an earlier wrong assumption)

Initially inferred (from tiny-repair timings) that Dream-7B per-step is
weight-bandwidth-bound and canvas-INDEPENDENT. DIRECT MEASUREMENT REFUTED IT.
Full-canvas steps (thr 0.95, canvas-filling prompt so EOT-shrink can't collapse
it), 40 steps, per-step avg:
- 64-canvas:  65 ms   | 128-canvas: 101 ms | 256-canvas: 130 ms | 448: 125 ms
=> per-step ~= ~15 ms floor + ~0.4 ms/token. CANVAS-DEPENDENT (attention O(L^2) +
FFN/lm_head O(L) over the canvas rows). Lesson recorded: never infer the cost
model from one canvas size; sweep it. CONSEQUENCE: canvas-reduction F-items
(window, kv-cache, F10) are NOT architecturally dead - they have real headroom on
LARGE canvases. The kintsugi workload is small-canvas-dominated BY DESIGN
(EOT-shrink collapses drafts; repairs are 60-150 tok), which is precisely why
Layer A measured kv-cache as a net loss HERE. Workload-mismatched, not impossible.

Step-count mechanics (small canvas, the kintsugi regime):
- repair, 3-mask hole: 4 steps; 6-mask: 7 steps; 12-mask: 13 steps (thr 0.9).
- WITH early_commit 0.5 (Prophet, already adopted on the repair path): a 12-mask
  hole collapses to 4 steps. Repairs are ALREADY near-optimal on step count.
- threshold sweep on a 12-mask repair (early_commit on): thr 0.9/0.7/0.5 ~= 4
  steps/146 ms; thr 0.3 -> 2 steps/80 ms but quality-risky. No free win.

### Probe 4: backend sampling HURTS tiny repairs (attach overhead)

Per-request, diffusion_generate builds+attaches+detaches a fresh backend sampler
chain (llama_set_sampler), forcing a sched re-reserve. On few-step repairs this
overhead exceeds the sampling saving:
- 3-mask hole: backend ON 109 ms vs OFF 87 ms (+23 ms)
- 6-mask hole: ON 197 vs OFF 160 (+37 ms)
- 12-mask hole: ON 364 vs OFF 383 (-19 ms, backend finally wins)
Crossover ~10-12 masks. Production repair median is 111 ms (~3-6 mask regime) =
the LOSING regime. BANKABLE (small, free): route small infills (<= ~8 masks) to
CPU sampling, OR (better, the gpu-sampling-plan.md deferred item) attach the
sampler ONCE at context creation so backend wins at every size. Est. ~3-5% of
bench wall; noisy, workload-dependent - measure before banking.

### Probe 5: the draft "wide-canvas tax" (no free win - quality-bound)

A 71-char "double" answer still costs 493 ms = 8 steps x 62 ms: the 192-canvas
(~255 tok) runs wide steps before/without aggressive EOT-shrink. Right-sizing to
~64 tok would give ~28 ms/step (~224 ms), BUT Layer C measured every draft-canvas
reduction (gen_initial, big384, slim, window) as a BENCH REGRESSION - canvas
width is a quality knob (the model plans its answer to fit the visible canvas;
stub adaptation). So the wide-canvas tax is REAL but unrecoverable without losing
draft quality. Confirmed dead, fresh data.

### Bankable engine deliverable found: expose steps_done in /generate

The server response lacks the step count (only ms_total). The harness needs it
for G8 (hole-size learning -> cut the 3-variant sweep that is 59% of infill
calls). diffusion_generate already computes n_steps_done; plumb it out as an
out-param + "steps_done" in the response. Trivial, zero-risk, enables a real
HARNESS lever (fewer calls). The single clean engine-side item this round yields.

### Probe 6: global backend-sampling A/B (kills the "route to CPU" idea as global)

Full bench, server --no-backend-sampling vs default:
- backend ON:  35/48, 6.46 tok/s, 147 s (the baseline)
- backend OFF: 33/48, 4.34 tok/s, 186 s  (-2 pass, -33% deliverable, +27% wall)
Backend ON is correct globally - the draft + large-repair savings dwarf the
tiny-repair attach penalty (Probe 4). The CPU-routing win exists ONLY for the
small-infill subset and would need surgical per-call gating for a ~3-5% sliver
while risking the draft gains. NOT worth a flag. (attach-once-at-creation remains
the only clean way to capture it without the downside - a real but modest item.)

### THE STRATEGIC FINDING (why this matters)

The deliverable metric bleeds on FAILING cases (c-tier 0/9 @ ~8 s, a-tier 0/3 @
~2.9 s, m-tier 1/3) burning draft+repair cascades that never converge - a MODEL
CAPABILITY problem. No engine cache changes the pass count, and the cacheable
redundancy sits on small/bidirectional canvases where every prior caching
experiment (Layer A) already showed no win. Catalog Layer F (engine/system) is
therefore largely SPENT for Dream-on-laptop, the same verdict Layers B/C/D
reached about their own headroom. The remaining real lever against the
failing-case wall is HARNESS-side (Layer G): early-abort of hopeless drafts
(catalog G7), semantic repair cache (G4), fewer sweep variants (G8 hole-size
learning) - none of which are engine work.

### Probe 7: the "sampling time/step" metric is a RED HERRING (host work is ~0)

Chased a promising-looking lead: the diffusion loop's logged "sampling time per
step" was 58 ms (threshold) to 155 ms (schedule) on a 256-canvas - and looked
like host-side per-position readback overhead (each masked position calls 5
syncing `llama_get_sampled_*_ith`, each a full ggml_backend_sched_synchronize).
HYPOTHESIS: ~600 us/position of redundant syncs; fix by syncing once + nosync
reads.

BUILT IT, MEASURED IT, IT WAS WRONG. Decisive test (env DIFF_SYNC_AFTER_DECODE:
one llama_synchronize right after llama_decode, before the sampling timer):
sampling time/step 154.94 ms -> **0.06 ms**, per-step total UNCHANGED at 160 ms.
ROOT CAUSE: llama_decode is ASYNC. The decode submits the GPU forward and
returns; the first synchronize() inside the sampling-timer region (the first
_ith call) is where the CPU actually WAITS for the ~155 ms GPU forward. The
"sampling time" was 100% mis-attributed GPU-forward wait; true host sampling
(readback loop + confidence) is 0.06 ms/step.
CONSEQUENCES:
- The nosync-accessor idea saves NOTHING (the redundant idle-stream syncs are
  ~us; the real time is the single unavoidable GPU-forward wait). Reverted; no
  API added.
- Per-step IS the GPU forward (canvas-bound, Probe 3 confirmed from the other
  side). There is no host-side waste to reclaim anywhere in the step loop.
- LESSON for future probing: llama_decode is async - any "host" timer that
  contains the first post-decode sync is really measuring GPU-forward wait. The
  diffusion-batch-probe.cpp header already warned this (bug #1 there); it bit
  again. Always sync-after-decode before attributing host vs device time.
- Layer F micro-tier (F7 host overlap, F11 step-loop micro) is therefore DEAD on
  arrival: there is no meaningful host work to overlap or shave (~0.06 ms/step).

### CATALOG-F VERDICT (2026-06-13, after 7 probes + 2 full benches)

The engine cost model, measured not assumed:
  per-step ~= 15 ms + 0.4 ms/token (canvas-dependent);
  wall = sum over calls of (warmup + n_steps x per_step);
  per-request fixed overhead is small once amortized (>=7 steps);
  backend sampling ON is globally correct (OFF = -2 pass, -33% deliverable).
Every term's lever lives OUTSIDE the engine:
  - n_steps: already minimized (threshold + Prophet on repairs; Layer B);
  - per_step: canvas-bound, but kintsugi keeps canvases small by design
    (EOT-shrink + tiny repairs) so there is little to cut, and draft-canvas
    reduction is quality-bound (Layer C, rejected);
  - calls: the 3-variant hole sweep + 3-redraft cascade dominate - HARNESS (G);
  - pass count: model capability - MODEL (E).

ITEM-BY-ITEM (catalog numbering):
| item | verdict | basis |
| F2 encode logits-flags | PARK (DG-only) | D2H already skipped; compute marginal at 152k vocab; zero on infill |
| F6 CUDA graphs | CLOSED null | E6: zero on AC; Probe-0 reconfirms |
| F7 host overlap | DEAD | Probe 7: host work is 0.06 ms/step - nothing to overlap |
| F9 cross-req prompt cache | DEAD | prompt 25% of forward; Dream bidirectional; kv_prefix already net-loss small-canvas |
| F10 cross-req canvas cache | PARK (~10% ceiling) | headroom only on large failing-case repairs; variant sweeps shift positions; needs unbuilt A2 |
| F11 step-loop micro | DEAD | Probe 7: host step is 0.06 ms; per-step is pure GPU forward |

TWO bankable engine deliverables (small, real, low-risk):
1. Expose steps_done in /generate (trivial) - unblocks G8 hole-size learning,
   which cuts the 3-variant sweep (59% of infill calls). The one clean win.
2. Attach the backend sampler ONCE at context creation (gpu-sampling-plan.md
   deferred item) - removes per-request re-reserve, makes backend sampling win at
   every repair size (Probe 4). ~3-5% ceiling; do only if #1's harness work shows
   the repair path still matters after variant-cutting.

CONCLUSION: catalog Layer F engine/system work is SPENT for the Dream-on-laptop
kintsugi workload (same shape as the B/C/D closures). Bank deliverable #1; treat
#2 as optional. The real post-E headroom is HARNESS (Layer G: kill wasted calls)
and MODEL (Layer E: a model that passes more, or streams fewer bytes/step) - both
already on the roadmap. F10 is the only engine item worth UN-parking, and only IF
a future workload sends genuinely large repair canvases (it does not today).

### The remaining cost IS the GPU forward - and its only levers are E/G

After Probe 7, the per-step is provably 100% GPU forward (host work 0.06 ms).
Levers on the forward itself, all OUT of Layer F scope:
- Flash attention: already auto-ON for Dream (~7% per prior measurement); no win
  left to capture (it is the default).
- Small canvas (the kintsugi repair regime, ~25 ms/step at ~25 tok): weight-
  bandwidth-bound (stream 4.7 GB Q4/step) + fixed launch overhead - only a
  SMALLER model helps (E-tier: FastDLLM-1.5B, already integrated).
- Large canvas (>=256 tok, ~130 ms/step): compute-bound (attention O(L^2) + lm_head
  O(L)); canvas-reduction would help but the workload avoids it (EOT-shrink).
- lm_head over prompt rows (catalog-F2): part of the GPU forward, ~25% of lm_head
  rows on a draft, but lm_head is a small fraction of the 7B forward -> low single
  digit %, and zero on infill. Not worth the encode() core surgery for Dream.
None of these is an engine/system (Layer F) item that moves the kintsugi number.
The hunt is complete: there is no hidden host-side waste, and the GPU forward is
already near-optimally configured. Next headroom is E (smaller/better model) and
G (fewer calls). LAYER F CLOSED for Dream-on-laptop.

> SCRUTINY ADDENDUM (2026-06-13, second-pass red-team; data in
> throughput-plans/07_layer_f/permutation-batched-infill/). One factual claim
> behind the F1/batching closure is REFUTED: "the 5070 has no spare arithmetic at
> these shapes / pp batching flat" holds at 103-192 tok (measured 1.0x) but is
> FALSE at the real tiny-infill canvas sizes - batched-3 is 1.96x free at 32 tok,
> 1.79x at 40 tok (Dream-7B, batch-probe PROBE3c). 75% of infill wall lives at
> 0-64 tok. This opens a LIVE, LOSSLESS (bit-exact, 35/48 holds) engine lever -
> batched variant-sweep infill - projected +6-12% deliverable, and it attacks the
> failing-case wall where harness G8 cannot (no winning hole-size to predict when
> the repair fundamentally fails; 75% of sweeps end best_effort, E[variants
> tried]=2.151). The strategic conclusion (real headroom is E+G) is WEAKENED not
> overturned: this is an engine lever, but modest, surgery-gated (square no-cache
> infill multi-seq; see BUILD-SPEC.md), and best paired with G8. "Layer F engine
> SPENT" -> "Layer F has one live surgery-gated lever (~+6-12%) the closure
> mis-measured." The rest of the catalog-F round (host work 0.06ms, F9/F10/F11,
> per-step cost model) re-verified and stands - see 07_layer_f/VERIFICATION.md.

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
- CAVEAT RESOLVED (2026-06-13, F1 de-risk probe): the multi-replica server DOES
  compose with block-kv - each replica is its OWN llama_model instance
  (mmap-shared weights, separate pkv store/phase state), and 3 concurrent
  block_kv requests across 3 replicas on one GPU ran correctly. The per-MODEL
  pkv caveat only forbids sharing ONE model object across concurrent contexts,
  which the server never does. No per-context refactor needed for the rig.

### E1. SDTT distillation - STAYS PARKED until F2 evidence lands
Bar unchanged: beat 21/42 draft-only at <= 1 GB. F2's oracle probe (below) may
REPLACE E1 entirely: if drafting-for-AR-verify works, the drafter does not need
standalone quality - reassess after F2.

---

## PART 2 - Layer F proper (start when Part 1 E4+E5 are measured)

### F1 MEASURED (2026-06-13) - KILLED on single-GPU; rig racing needs no engine work

De-risk probe ran WITHOUT engine surgery, via the multi-replica server (each
replica = own llama_model instance = own pkv store, so block_kv + concurrency is
safe): --diffusion-replicas 3 on the 5070 (3.9 GB VRAM), stack-task drafts at
seeds {3,103,203}, kv+bs E4 config.

- sequential 3 drafts: 1.38 s wall (0.43/0.39/0.56 s each)
- concurrent 3 drafts: 1.21 s wall, but EACH request slows to 1.07-1.21 s
- ratio vs one draft: 2.63 (bar was < 3)

2.63 < 3 passes the formal bar but kills the use case: first-success racing
SLOWS every case whose first seed passes (0.43 -> 1.07 s, the majority of
p-tier) and saves only 12% on triple-fail cases. Race-only-redrafts (seed 1
sequential, 2+3 raced on failure) saves ~10% on double-redraft cases only -
noise at bench level. The batched-multi-seq variant (one N x 32-row batch,
per-seq pkv stores) would land ~2.0-2.2x by the E2-checkpoint row-scaling
numbers - same conclusion, plus the engine surgery the plan costed. The 5070
has no spare arithmetic at 32-row 1.5B shapes; this is Layer D findings 3/6
again on the new model.

DECISION: F1 CLOSED for single-GPU serving. Cross-CARD racing on the rig is the
already-designed farm pattern (catalog G5/G6, kintsugi fires seeds at replicas;
zero engine work) - implement it AS HARNESS WORK on rig day, where N cards make
the ratio exactly 1.

Original premise kept for the record:

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

### F2 STEP 2 v0 MEASURED (2026-06-13) - PARKED: closed-loop acceptance kills it

Built llama-dflash (examples/diffusion/dflash.cpp): both models in one process
(5.4 GB VRAM), drafts cross as TOKENS (shared Qwen2.5 vocab - no retokenization),
verifier greedy walk + cache rollback per round, drafter stateless per round
(re-prefills its pkv store; ~8 ms/32-block, minor at probe lengths). --ar mode =
same-process pure-AR baseline. 8 bench cases, max 256 tokens, output IS the
verifier's greedy decode by construction.

| config | tok/s | accepted/round | drafted-token efficiency |
|---|---|---|---|
| AR baseline (Qwen2.5-7B greedy) | **54.8** | - | - |
| dflash draft-len 32 | 26.1 | 5.79 | 18.1% (draft = 85% of wall) |
| dflash draft-len 16 | **40.2** | 5.69 | 35.6% |
| dflash draft-len 8  | 34.6 | 4.11 | 51.3% |

THE ORACLE WAS OPTIMISTIC, AS WARNED: closed-loop acceptance is 5.7/round, not
the oracle's 8-12. Mechanism: the oracle's first blocks scored the easy chat
preamble ("Sure, I'd be happy to help...") at L=29-30; INSIDE code the drafter
diverges from Qwen-7B every ~4-6 tokens. Re-drafting from corrected prefixes
does not lift it (5.79 at dl 32 vs 5.69 at dl 16 - acceptance dies early and
re-basing does not save it). Acceptance is capability-bound (1.5B vs 7B
disagreement rate on code), not context-bound.

Ceiling arithmetic for a PERFECT harness (persistent cross-call pkv, no attach,
no re-prefill: ~8.6 steps x 8.4 ms + 12 ms verify per round of 6.7 tokens):
~80 tok/s = 1.45x over AR - the maximum a multi-session engine effort could
recover, on the laptop only (rig economics invert: drafting is compute-shaped,
AR verify bandwidth-shaped). NOT worth it.

DECISION: F2 PARKED. Off-the-shelf FastDLLM-1.5B cannot draft for Qwen2.5-7B at
useful acceptance; DFlash-class wins require the trained context-conditioned
drafter (their whole contribution, in retrospect). E1's reframing resolves the
OTHER way: the drafter DOES need training either for standalone quality (bar:
beat 21/42 at <= 1 GB) or for acceptance (bar: closed-loop L >= ~8-10) - both
training-tier, both parked together. The llama-dflash tool stays in the tree:
re-run it the day a trained/better drafter exists (one command, full closed-loop
measurement).

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
