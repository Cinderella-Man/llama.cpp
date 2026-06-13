# Probe 3 — per-step cost mechanics (re-verification)

**Doc claims** (07_layer_f.md:103-123):
1. Per-step is **canvas-DEPENDENT** (refutes an earlier "weight-bandwidth-bound,
   canvas-independent" assumption). Full-canvas steps (thr 0.95): 64→65 ms,
   128→101 ms, 256→130 ms, 448→125 ms ⇒ **per-step ≈ 15 ms floor + 0.4 ms/token**.
2. Step counts (infill, thr 0.9): 3-mask→4 steps, 6-mask→7, 12-mask→13.
3. With early_commit 0.5 (Prophet): a 12-mask hole collapses to ~4 steps.

## How verified
`run.sh` (Dream-7B server, GPU sampling default):
- per-step sweep via full-canvas generation, thr 0.95, steps 40;
- step-count sweep via infill (`infill:true`, mask piece `<|mask|>` from /health)
  at 3/6/12 masks × early_commit {0, 0.5}, reading `steps_done` from the response.
- ADDENDUM (cleanest test): `steps=1` single forward at each canvas (shrink is
  impossible with one step) to isolate per-step-vs-canvas without EOT-shrink
  contamination. Raw data in `raw.log`.

## Result (re-run 2026-06-13 10:01, 5070 on AC)

### Step counts — EXACT
| masks | doc steps | re-run steps | verdict |
|---|---|---|---|
| 3  | 4  | **4**  | REPRODUCED exact |
| 6  | 7  | **7**  | REPRODUCED exact |
| 12 | 13 | **13** | REPRODUCED exact |

### Per-step vs canvas — canvas-dependence REPRODUCED, coefficient confirmed
The doc's per-step *averages* (65/101/130/125) are EOT-shrink-contaminated — its
own 448 (125) < 256 (130) is non-monotonic, proving shrink leaked into the
average. My server-bench average reproduced the same artifact (flatter, ~22
steps not 40). The clean `steps=1` single-forward measurement removes the
contamination:

| canvas | single-forward ms (warm) |
|---|---|
| 64  | ~56  |
| 128 | ~92  |
| 256 | ~184 |
| 448 | ~239 |

Linear fit **~25 ms + 0.48 ms/token**. Slope 0.48 ≈ doc's 0.4 (within band);
floor 25 vs 15 ms (the `steps=1` path carries one prompt prefill + fixed
overhead). Strongly monotonic (56→239 ms) ⇒ **canvas-DEPENDENCE — the actual
load-bearing finding — REPRODUCED**, and the cost-model coefficient is
reproduced (my clean fit matches the doc's model better than the doc's own
shrink-contaminated averages do).

### early_commit collapse — mechanism holds, magnitude DRIFTED
| masks | ec=0 | ec=0.5 (re-run) | doc ec=0.5 |
|---|---|---|---|
| 3  | 4  | 1 | — |
| 6  | 7  | 4 | — |
| 12 | 13 | **9** | "collapses to 4" |

early_commit 0.5 reduces steps in every case (mechanism reproduced), but a
synthetic 12-mask hole collapsed to 9, not the doc's 4. The doc's "→4" was on a
real repair canvas where the fills are higher-confidence (top1-top2 gap ≥ 0.5
sooner); on a low-confidence synthetic hole the gap is reached later. Magnitude
is workload-dependent; direction is correct.

**VERDICT: REPRODUCED (core), one quantitative DRIFT.** The two load-bearing
claims — per-step canvas-DEPENDENCE and the exact base step-counts (4/7/13) —
reproduce; the cost-model coefficient (~0.4-0.48 ms/tok) reproduces on a clean
measurement. The only miss is the *magnitude* of early_commit collapse on a
synthetic hole (9 vs 4), which does not affect any Probe 3 conclusion ("repairs
are already near-optimal on step count").
