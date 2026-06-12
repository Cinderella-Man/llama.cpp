# DiffusionGemma speed analysis (2026-06-12) - why 1000 tok/s, and what transfers

Question driving this: DiffusionGemma claims 800-1000+ tok/s with Unsloth-tweaked
llama.cpp while our 3-4 GB diffusion models (Dream-7B Q4, Fast_dLLM_v2-1.5B Q4) deliver
tens of tok/s on the RTX 5070 laptop. What did they do, and would applying it to our
tiny models give thousands of tok/s locally?

Companion doc: [diffusiongemma-support.md](diffusiongemma-support.md) (architecture,
PR #24423 analysis, support plan). This doc is the throughput comparison only.

## TL;DR

**The 1000 tok/s is the H100, not the algorithm.** Independently measured, the same
model on hardware in our class does 14 tok/s (Strix Halo) and 5.6 tok/s (4x Tesla
P40) - 2-3% of the H100 headline, same model, same llama.cpp PR. Our FastDLLM-1.5B
already does ~90 tok/s raw engine throughput on the 5070 laptop; Dream-7B ~21 tok/s
raw. **Per unit of hardware, our tiny models are already faster than DiffusionGemma.**
Nobody does thousands of tok/s on a laptop; Google's own desktop RTX 5090 number is
"700+".

## Calibration: measured DiffusionGemma numbers by hardware

| Hardware | Quant | tok/s | per-step | steps/256-tok block | source |
|---|---|---|---|---|---|
| H100 (FP8) | FP8 | 1008 | ~13 ms (derived) | ~17-48 | Google claim |
| H200 (FP8) | FP8 | 1288 | - | - | Google claim |
| RTX 5090 | - | 700+ | - | - | Google claim |
| RTX 6000 (Blackwell) | - | 2000+ | - | - | Unsloth docs |
| Strix Halo (Radeon 8060S) | Q8_0 (26.9 GB) | **14** | 1025-1031 ms | 17 (256 tok) / 36 (512) | tinycomputers.io |
| 4x Tesla P40 | Q8_0 | **5.6-6** | 2235-2423 ms | 19 / 38 | tinycomputers.io (KV cache off, no FA) |

Our own numbers on the RTX 5070 laptop (8 GB, AC), for comparison:

| Model | raw engine tok/s | per-step | commits/step |
|---|---|---|---|
| Dream-7B Q4_K_M (threshold 0.6, 512 canvas, no cache) | ~21-70 | ~110 ms (512 canvas) -> 15-30 ms post-shrink | 7.9 |
| Fast_dLLM_v2-1.5B Q4_K_M (block-AR, **uncached square**) | ~90 | 23.2 ms @ 171-tok output | 1.85 |
| Kintsugi full system (Dream, verified deliverable) | 6.22 deliverable | - | - |

METRIC MISMATCH WARNING: our headline 6.22 tok/s is *deliverable* throughput
(compile+test-verified code, failed runs and repair ladders counted in the
denominator). Google's 1000 is raw decode. Comparing them is apples to oranges; our
raw numbers are 21-90.

## Where the H100 number comes from (three stacked multipliers)

DiffusionGemma = 26B MoE with **3.8B active params**, block-AR over 256-token canvases,
~17-48 denoising steps per block, entropy-bound sampler committing ~5-15 tokens per
forward. Each forward is a 256-row batch -> decode becomes ARITHMETIC-bound, not
bandwidth-bound (the regime AR decode lives in).

1. **Cheap forwards**: MoE 3.8B active = a 4B-class model's FLOPs per pass despite the
   26B footprint.
2. **Few steps, trained-in**: USD training ("The Diffusion Duality", arXiv 2506.10892)
   makes aggressive parallel commits safe - the quality cliff we measured on Dream
   (conf-threshold 0.9 corrupts FastDLLM output; adaptive tau lost 20 bench passes) is
   trained away, plus native re-noise self-correction and a *trained* KV cache for
   prompt + committed blocks (no square recompute, no approximation error).
3. **The card**: H100 FP8 tensor cores + FlashAttention do a 256-row x 3.8B forward in
   ~13 ms -> 256 tok / (17 steps x 13 ms) ~= 1000 tok/s. Strix Halo needs 1031 ms for
   the same forward: that one factor is ~70x and is the entire headline. The
   tinycomputers post calls it out directly: this workload rewards raw arithmetic
   throughput, which consumer/older GPUs don't have.

Our layer D probes (04_layer_d.md) independently confirmed point 3 from the other
side: on the 5070, canvas batching does NOT amortize at production sizes (1.00-1.02x,
~18 ms step floor + ~0.31 ms/row) - the laptop lacks the spare arithmetic that makes
DG's regime fly.

## What transfers to our models

1. **KV-cached block-AR decode for FastDLLM-1.5B (E3) - the real prize.** FastDLLM is
   trained block-causal exactly like DG, but our v1 integration recomputes the full
   square canvas every step (23 ms/step at 171 tokens, growing with output length). A
   standard causal cache for committed blocks keeps steps flat (~8-12 ms expected).
   Expected: **2-3x raw -> ~200-300 tok/s engine** on the 5070. Already first on the
   layer E NEXT list (05_layer_e.md). That would be ~15-20x DiffusionGemma's measured
   throughput on comparable consumer silicon.
2. **Entropy-bound committer experiment (cheap, fail-fast).** Port DG's
   cumulative-entropy budget (entropy_bound 0.1, temp schedule 0.8->0.4, adaptive stop)
   into our existing confidence machinery as an alternative to fixed conf-threshold
   0.9; might lift FastDLLM's 1.85 commits/step. Risk: trained-in for DG, not for
   FastDLLM - our measured threshold cliffs say expect modest gains.
3. **DiffusionGemma itself as a QUALITY play, not speed.** Q4_K_M is 16 GB (doesn't
   fit 8 GB VRAM) but `--n-cpu-moe` with 3.8B active is plausible on 65 GB host RAM
   (support doc Path A). Expect 5-15 tok/s locally - slower than FastDLLM, but 256K
   context and natively-trained re-noise repair = a trained version of what kintsugi
   does by hand.

**What does NOT transfer**: H100/FP8 arithmetic; USD few-step training (training-time
property - reaching it ourselves is E1-style distillation with its own bar).

## Bottom line

Thousands of tok/s on this laptop is not achievable by any open model today - compute
is the binding constraint, and DG's own consumer-hardware numbers prove it. The
realistic measured-headroom target is FastDLLM-1.5B + committed-block KV cache at
~200-300 tok/s raw on the 5070. Our tiny models are not embarrassingly slow; they are
ahead of the new hotness once you control for the card.

## Sources

- https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF
- https://tinycomputers.io/posts/running-diffusiongemma-on-strix-halo-and-tesla-p40s.html
  (Strix Halo / P40 measurements, no-FA caveat)
- https://unsloth.ai/docs/models/diffusiongemma (settings, RTX 6000 claim, VRAM table)
- https://diffusiongemma.org/how-to-run
- https://www.analyticsvidhya.com/blog/2026/06/diffusiongemma-diffusion-based-open-model-for-faster-text-generation/
  (H100 FP8 1008 tok/s, H200 1288 tok/s)
- llama.cpp PR #24423 (Unsloth draft; analyzed in diffusiongemma-support.md)
- Local measurements: throughput-plans/01-06, dllm-throughput-catalog.md
