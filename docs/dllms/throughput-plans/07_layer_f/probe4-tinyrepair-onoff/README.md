# Probe 4 — backend sampling on tiny repairs (re-verification) — **DRIFTED**

**Doc claim** (07_layer_f.md:125-137): per-request backend sampler
attach/detach overhead exceeds the sampling saving on few-step repairs:
- 3-mask: ON 109 / OFF 87 (+23 ms)
- 6-mask: ON 197 / OFF 160 (+37 ms)
- 12-mask: ON 364 / OFF 383 (−19, backend finally wins)
- Crossover ~10-12 masks; production repair median (111 ms, ~3-6 mask) = the
  LOSING regime. ⇒ bankable item: route small infills to CPU, or attach-once.

## How verified
`run.sh`: two server arms (default = backend ON; `--no-backend-sampling` = OFF),
same Dream-7B config. Infill `defmodule M do … HOLE … end` with 3/6/12 masks
(mask piece `<|mask|>`), steps 16, thr 0.9. **Warmup + median-of-5 per cell**
(tiny-repair timings are noisy; the doc's single-shot ON numbers look cold).
Backend-active confirmed from server logs: ON arm = 16× "sampling on the
backend", OFF arm = 0×; identical step counts (e.g. 12-mask = 13 steps both arms).

## Result (re-run 2026-06-13 10:07, 5070 on AC)

| masks | ON (re-run) | OFF (re-run) | ON−OFF | doc ON | doc OFF | doc ON−OFF |
|---|---|---|---|---|---|---|
| 3  | **81.3** | 89.8 | **−8.5** | 109 | 87 | +23 |
| 6  | **164.2** | 170.5 | **−6.3** | 197 | 160 | +37 |
| 12 | **344.3** | 356.9 | **−12.6** | 364 | 383 | −19 |

- The **OFF arm reproduces** (3-mask 89.8 ≈ doc 87; 6-mask 170.5 ≈ doc 160).
- The **ON arm is much faster than the doc's ON** (3-mask 81 vs 109; 6-mask 164
  vs 197). The per-request attach penalty the doc measured is **absent** here:
  backend ON is **uniformly ≤ OFF at every mask count**, including the tiny ones.

## Verdict & interpretation

**VERDICT: DRIFTED (the load-bearing sub-claim is REFUTED at HEAD).** The doc's
central Probe 4 finding — "backend sampling HURTS tiny repairs; crossover at
10-12 masks; production median repair is in the LOSING regime" — does **not**
reproduce. Backend ON wins at 3, 6 and 12 masks. The most likely cause: the
doc's ON column is a cold/single-shot measurement that captured one-time costs
(graph capture, first sched reserve); under warm median-of-5 the per-request
attach cost is ≤ the sampling saving at every size.

**Consequence for the doc's conclusions (no harm to the headline):** this
actually *strengthens* the doc's main verdict — backend sampling ON is globally
correct (Probe 6) — and makes the two Probe-4-derived levers **moot**: there is
no tiny-repair regime where ON loses, so "route small infills to CPU" is
unnecessary and bankable deliverable #2 ("attach-once to make backend win at
every repair size") describes a state that already holds. No Layer-F verdict
flips; one optional optimization is retired as already-achieved.

(See `../static-code-claims/` S5 — the per-request attach/detach machinery is
present in the source exactly as described; it is simply cheap, not free-lunch
absent.)
