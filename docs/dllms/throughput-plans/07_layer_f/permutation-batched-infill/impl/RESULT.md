# Layer F implementation — batched variant-sweep infill: BUILT, MEASURED, NET-NEGATIVE

The 07_layer_f scrutiny found one engine lever that looked alive: batching the 3-variant
hole sweep into one decode, where tiny canvases are bandwidth-bound and batching is up to
1.96× free (measured). This is the full implementation + A/B. **Outcome: the lever does
NOT produce a bench win — flat at best, −9% at worst. Layer F engine stays SPENT.**

## What was built (all real, working code)
- **Engine** `diffusion_generate_infill_batch` (examples/diffusion/diffusion.cpp): decodes
  N infill canvases in one batched forward (seq_id = canvas index), mirroring the infill
  config (threshold + Prophet early-commit + degeneracy guard) per-seq. **N=1 is
  byte-identical to diffusion_generate (6/6)** — logic verified (impl/CORRECTNESS.md).
- **Server** `POST /infill_batch` + `n_seq_max=4` for masked models (diffusion-server.cpp).
- **Harness** `Engine.infill_batch/3`, batched `try_hole_variants` path, `batchsweep`
  bench profile (opt-in; OFF by default).

## A/B (Dream-7B Q4_K_M, 5070 AC, full 48-case bench, same build)
| config | baseline | batchsweep | Δ deliverable |
|---|---|---|---|
| byte gate (canvas ≤230 B) | 35/48, 6.45 tok/s, 147.1 s | 35/48, **6.40**, 146.1 s | **flat (−0.8%)** |
| hole-size gate (n_tokens ≤8) | 35/48, 6.41 tok/s, 148.1 s | 34/48, **5.82**, 155.7 s | **−9%** |

Projection was +6–12%. Realized: flat-to-negative.

## Why the projection failed (the two effects the per-step model missed)
Direct timing on real sweep canvases (impl, batched vs sequential):

| variants (masks) | tokens | seq (steps) | batched (steps) | ratio |
|---|---|---|---|---|
| [6,8,8]    | 52–54 | 311 ms (3,2,2)  | 233 ms (3,2,2)    | 1.34× win |
| [10,12,14] | 56–60 | 783 ms (6,10,2) | 738 ms (6,10,**5**) | 1.06× flat |
| [20,22,28] | 66–74 | 1478 ms (7,8,16)| 1778 ms (**16,15,16**) | 0.83× SLOWER |

1. **Wrong canvas-size zone.** Real sweep canvases are 50–60 tok (the surrounding code
   dominates; the hole is small but the canvas isn't). PROBE3c amortization there is only
   1.2–1.35×, not the 1.96× at 32 tok. The juicy zone is nearly empty in production.
2. **Multi-seq FP noise breaks Prophet early-commit.** Batching changes the GEMM
   reduction order → logits differ in the last bits → near-tie positions flip. Prophet
   early-commit requires EVERY remaining mask to clear the top1–top2 gap; with many masks,
   one batched-noise-perturbed position fails the gap every step, so batched variants run
   to the 16-step cap instead of converging in 7–8. Batched ends up doing MORE total
   decode work than the sequential short-circuit, not less.

The per-step projection assumed equal step counts batched vs sequential; in reality
batching INCREASES steps for the larger holes, and the small holes sit in the weak
amortization zone. Net: no win.

## Correctness note (the build is sound; the lever is what's dead)
- N=1 batched == diffusion_generate, 6/6 byte-identical → indexing/logic correct.
- N≥2 diverges ~15% of canvases by one near-tie token (inherent multi-seq GPU FP noise,
  same class as Probe 0's m-tier 2/3-vs-1/3). Pass count still held (35→35) on the byte
  gate, so quality is robust; the problem is purely that it's not FASTER.

## Verdict
**Layer F engine is confirmed SPENT for Dream-on-laptop — now by construction, not just
analysis.** The one lever the scrutiny kept alive was built end-to-end and measured net
flat-to-negative. The doc's strategic conclusion (real headroom is harness G + model E)
stands fully. The code is retained as a correct, opt-in (default-OFF) path: re-run the
`batchsweep` profile the day a model/quant/early-commit rule changes the convergence
dynamics (e.g. a model that doesn't rely on all-mask early-commit, or true tiny-canvas
repairs). Until then it is not enabled.

## Reproduce
`impl/ab.sh` (byte gate) — baseline vs batchsweep on one server. Timing breakdown by the
inline python in the session log. Diffs: the engine/server/harness changes are in the
working tree (or `impl/*.diff` if reverted).
