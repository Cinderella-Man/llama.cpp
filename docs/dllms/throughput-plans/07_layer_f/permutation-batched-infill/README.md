# Permutation: batched variant-sweep infill (a live engine lever the doc missed)

Scrutiny target: 07_layer_f.md's verdict that "Layer F engine/system is SPENT for
Dream-on-laptop" and its basis for killing the batching family (F1/D-layer):
*"the 5070 has no spare arithmetic at these shapes"* / catalog *"pp batching flat
on 5070 (AC)."*

## The hole
That claim is TRUE at 103-192 tok (compute-bound) but **FALSE at the real
tiny-infill canvas sizes** (bandwidth-bound, weights amortize across the batch).
Measured on Dream-7B Q4_K_M, 5070 AC (`llama-diffusion-batch-probe`, PROBE3c —
see `batchprobe-realsizes.diff`, `amortization-realsizes.log`):

| canvas | batched-3 vs 3-sequential |
|---|---|
| 32 tok | **1.96× free** |
| 40 tok | **1.79× free** |
| 48 tok | 1.35× |
| 64 tok | 1.22× |
| 96 tok | 1.05× (flat — the regime the doc measured) |
| 103 tok | 1.04× (original probe) |

The doc generalized from 103-192 tok (flat) and missed that the kintsugi infill
workload lives at 32-48 tok, where batching is 1.4-2× free.

## Why it matters (the opportunity is real and on the dominant cost)
- Probe 0 trace: **75% of infill wall (55.4 s) is in 0-64 tok canvases**
  (wall-weighted mean 41 tok) — exactly the bandwidth-bound, batchable regime.
- The 3-variant hole sweep `{n, n+2, 1.4n}` (kintsugi.ex:337) fires these
  sequentially with short-circuit. Instrumented exhaust rate
  (`sweep-instrumentation.diff`, `sweep-trace.csv`, full bench):
  **166 sweeps, E[variants tried] = 2.151** (>1.67 break-even), 84% try ≥2
  variants, **75% end best_effort (no variant compiles)**.
- So the sweep is largely a FAILURE path — and the failing-case wall is exactly
  the bleed the doc's strategic finding identifies. Batching makes it cheaper
  where the doc's preferred harness lever (G8 hole-size prediction) CANNOT help:
  when the repair fundamentally fails, there is no winning size to predict.
  → **complementary to G8, not dominated by it.**

## Projection (all inputs measured)
`run.sh` recomputes. Per-call, gate-in only where batching wins (≤~56 tok):

- projected saving **16.3 s of 147 s bench wall**
- deliverable **6.46 → 7.26 tok/s (+12%)**
- **pass count UNCHANGED (35/48)** — batched multi-seq is bit-exact (Layer D
  finding 1: cross-seq isolation), so the lever is **LOSSLESS**.

### Honest caveats (why the real number is lower)
1. **Step-count divergence**: the 3 variants have different mask counts → different
   step counts. A real batched decode runs until the LONGEST variant finishes,
   wasting rows on early-finishers. The +12% is a per-step upper bound; realistic
   is ~**+6-10%**. Only the actual multi-step build measures this.
2. **Extra compile checks**: batched checks all `total` variants (E[total]=2.464)
   vs sequential's short-circuit (E[tried]=2.151) — ~0.3 extra CPU compile checks
   per sweep. Minor, CPU-side.
3. **Capture cost**: needs F1-class multi-seq infill decode (per-seq pkv/EOT/
   degeneracy state) + harness batched-sweep + size-gating. The doc scoped this as
   multi-session engine surgery; that judgment stands.

## Verdict
The doc's *factual* claim ("batching flat / no spare arithmetic on the 5070") is
**REFUTED** at the workload's real canvas sizes (1.96× free at 32 tok, measured).
A genuine, **lossless, ~+6-12% engine lever exists** (batched variant-sweep
infill) that the doc missed and that attacks the dominant failing-case wall.

The doc's *strategic* conclusion ("the real post-E headroom is harness G + model
E") is **weakened but not overturned**: this is an ENGINE lever with real headroom,
but it (a) is modest, (b) needs multi-session surgery the doc reasonably deferred,
and (c) is best deployed alongside G8, not instead of it. "Layer F engine SPENT"
should be downgraded to "Layer F has one live, lossless, surgery-gated lever
(~+6-12%) on the tiny-canvas batching the doc mis-measured."

## Files
- `batchprobe-realsizes.diff` — PROBE3c (K=3 amortization at 32/40/48/64/96 tok)
- `amortization-realsizes.log` — its output
- `sweep-instrumentation.diff` — harness KINTSUGI_SWEEP_TRACE exhaust logging
- `sweep-trace.csv` — 166 sweeps (tried,total,outcome) from a full bench
- `run.sh` — recomputes the projection from the measured inputs
