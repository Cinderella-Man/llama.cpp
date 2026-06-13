# Probe 1 — infill wall vs canvas size (re-verification)

**Doc claim** (07_layer_f.md:55-62): controlled infill, 4-mask hole, fixed code
context of N filler lines (server direct, median): ~23 prompt tok → 133 ms;
~292 prompt tok → 782 ms (120-line case 500'd: canvas > ubatch 512) ⇒
**~2.3 ms/token above an ~80 ms fixed floor**. The UNCHANGED code (not the hole)
dominates a large repair; small repairs are floor-bound.

## How verified
`run.sh`: Dream-7B server, 4-mask hole inside `defmodule M do … end` with a
growing block of fixed filler lines (1/20/70/130). Warmup + median-of-3.
Reports server-reported `n_prompt_tokens` and `ms_total`; detects HTTP 500 when
the canvas exceeds ubatch 512.

## Result (re-run 2026-06-13 10:09, 5070 on AC)

| filler lines | prompt tokens | median ms | doc reference |
|---|---|---|---|
| 1   | 35  | 118.8 | ~23 tok → 133 ms |
| 20  | 302 | 767.0 | ~292 tok → 782 ms |
| 70  | —   | **HTTP 500** | 120-line case 500'd (canvas > ubatch 512) |
| 130 | —   | **HTTP 500** | "" |

- **Slope** = (767.0 − 118.8)/(302 − 35) = **2.43 ms/token** vs doc 2.3 → REPRODUCED.
- **Large-canvas point**: 302 tok → 767 ms vs doc 292 tok → 782 ms → REPRODUCED.
- **ubatch-512 overflow → HTTP 500**: reproduced exactly (70-line and up 500;
  the doc saw it at 120 lines — the threshold depends on tokens/line, but the
  failure mode "canvas > ubatch 512 → 500" is confirmed).
- **Floor**: my two points fit a ~33 ms floor (118.8 − 2.43·35 ≈ 34). This is
  *lower* than the doc's stated "~80 ms" but **consistent with Probe 0's measured
  infill min of 32 ms** — the doc's Probe 1 floor estimate was loose; 33 ms is
  the better value and does not change the conclusion.

**VERDICT: REPRODUCED.** The F10 load-bearing number — ~2.3 ms/token, so the
unchanged surrounding code dominates a large repair while small repairs are
floor-bound — reproduces (2.43 ms/tok), as does the ubatch-512 ceiling. Only the
absolute floor differs (33 vs 80 ms), and the lower value is the self-consistent
one. The F10 verdict (headroom only on large repair canvases) stands.
