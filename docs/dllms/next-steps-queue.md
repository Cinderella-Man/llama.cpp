# dLLM/Elixir next-steps queue — laptop-runnable, work one-by-one

## Context

Why: three research streams in `docs/dllms/` now disagree, and the laptop diffusion roadmap
(throughput Layers A–G) is mostly MEASURED-CLOSED. The overnight `Usable_Elixir_on_a_9×P106-100.md`
claims throughput is already solved on AR (30.4 tok/s) and diffusion is dead on Pascal; yesterday's
`elixir-research-integration.md` says the real wall is CAPABILITY not speed; the `dllm-throughput-catalog.md`
still assumes "diffusion non-negotiable." Goal of this file: a deduplicated, ranked list of what is
genuinely UNTRIED, with kill-gates, so we execute one-by-one. Built from a 6-agent read of all
~6k lines + 3 crux docs, reconciled against measured closures.

## HARD RULE (operator steer 2026-06-14)

- **tok/s alone is REJECTED.** "Random letters on a toaster isn't code." Every throughput/engine item
  reports **pass-rate X/48 WITH tok/s**, or it doesn't count. The "20 tok/s AR already solves it"
  claim is NOT accepted — that number is Llama-2-7B raw `llama-bench`, quality-blind. Re-measure
  quality-gated before believing throughput is solved.
- **Fork decisions are pass-rate first, speed second.**
- Report on BOTH aggregate AND multi-function (c_shout/c_stack) — keep both framings live.
- Instrument = `kintsugi/bench/bench.exs` (16 cases × 3 seeds). Diffusion `/health` must show
  `family: masked`, `mask_token_id` (Dream=151666).
- Rig (9×P106) NOT in hand → rig-gated probes parked below, not in the working queue.

## Strategic forks (these gate the list; resolve via the probes, not argument)

- **F-1 diffusion vs AR engine.** Catalog "diffusion non-negotiable" vs overnight "AR wins on Pascal."
  Diffusion is the only path that can INFILL/repair (carries 3/30→35/48); AR (block-causal) cannot.
  DECIDE on pass-rate via #4 (Dream full-loop vs Qwen-Coder-7B best-of-N on the bench). Until decided,
  do NOT spend on new diffusion-engine work OR abandon diffusion-repair.
- **F-2 throughput vs capability.** Binding constraint is multi-function/OTP reasoning
  (c_shout 0/189, c_stack 0/188 across both models + every profile) — a low-resource pretraining
  ceiling, not speed. Reallocate to capability, but EVERY capability bet carries the c-tier kill-gate.
- **F-3 train-where / single-vs-multi-function.** Laptop is the only train box now. OTP is likely
  unreachable by Python pivot; MultiPL-T "eats itself" (lifts what we already pass). Keep both bars.

---

## WORKING QUEUE (laptop, now) — all three tracks interleaved, cheap→expensive

Tags: [CAP]=capability [THR]=throughput [FARM]=harness/candidate-farm [ENG]=engine [PROBE]=falsification

### Wave 0 — hours each, zero/low code (do first)
1. **Dream-Coder-7B-Instruct drop-in swap** [CAP] tiny ~0.5–1d.
   Download mradermacher/Dream-Coder-v0-Instruct-7B-GGUF Q4_K_M; byte-identical arch reuses
   `src/models/dream.cpp` (zero C++). Verify `/health` family:masked mask 151666; run bench + c_shout/c_stack.
   First-ever diffusion-code Elixir number. **KILL:** c-tier 0/3 AND Δpass ≤+1 → model-instance dead, ceiling is scale.
2. **Degeneracy/repetition control sweep** [CAP] tiny <1d.
   Sweep repetition-penalty/min-p to kill `defmodulelerler`-class drafts. **KILL:** 0 pass change AND no
   repair-round drop → not a material slice.
3. **Reconcile BankF-1 `steps_done`** [ENG] tiny <0.5d.
   Static audit (static-code-claims S3) says `steps_done` already in `/generate`. VERIFY; if present,
   it's done → value moves to G8 hole-size learning. Else trivial plumb. **KILL:** already shipped → close, open G8.

### Wave 1 — the three forks, one probe per track (decision-grade, cheap)
4. **Diffusion-vs-AR capability head-to-head** [PROBE] medium ~3–5d (capability half laptop-now; speed half rig-parked).
   Dream-7B full repair-loop vs Qwen2.5-Coder-7B AR best-of-N on the bench — PASS-RATE first.
   **This is the F-1 decider.** **KILL:** AR best-of-N matches diffusion pass set → deprioritize entire
   diffusion-engine corpus to research-only, rig pivots AR. (download Qwen2.5-Coder-7B GGUF)
5. **n-gram / prompt-lookup ACCEPTANCE probe** [THR] small ~2–3d (probe only, build later).
   Measure accepted-tokens/round + BLEU-4 overlap on templated OTP specs (AR path, zero VRAM). Quality-gated.
   **KILL:** overlap/acceptance low (OTP boilerplate is small fraction of spec) → speed lever can't pay, park.
6. **AR candidate-farm best-of-N (laptop proxy)** [FARM] medium ~3–5d.
   kintsugi dispatcher, N seeds on one card, verifier picks first/best passing; metric =
   wall-clock-to-first-passing. `--diffusion-replicas` + max_drafts already exist (kintsugi.ex:156).
   **KILL:** lifts pass set ≤+1 beyond shipped max_drafts=3 → farm buys latency only, offline-throughput-only.

### Wave 2 — the capability bet + remaining escape hatches
7. **Stage1 Pylixir + QLoRA Qwen2.5-Coder-1.5B (kill-gated)** [CAP] medium ~3–5d (overnight QLoRA).
   Minimal Python-assert→ExUnit transpiler (~50 seeds; Elixir NOT in MultiPL-E, infra unbuilt) →
   tiny validated corpus → QLoRA 1.5B → measure ONLY c_shout/c_stack. **HARD KILL:** needles don't move →
   STOP all coverage-data work, ceiling is reasoning/scale.
8. **D4 hybrid with STRONG drafter** [CAP] small-med ~2–3d.
   Qwen2.5-Coder-7B AR drafter (AutoCodeBench-Elixir 47.0) + Dream-Coder repairer, time-shared (dodge
   ~9.4GB co-residency). Prior D4 reject used a WEAK 1.5B drafter — distinct config. **KILL:** pass set
   ==35/48 OR multi-fn 0/3 → D4 closed for all drafters.
9. **Grammar-constrained REPAIR only** [PROBE] small-med ~1–2d.
   GBNF (p4b-engine.diff exists) scoped to the masked-infill hole only (committed text pins context) —
   the ONE untested grammar slice (draft/full-decode CLOSED). Forces backend_sampling:false (CPU grammar).
   **KILL:** 0 pass gain / latches off → Layer G fully closed.
10. **Native FIM-channel scaffold probe** [PROBE] small ~1d.
    Use Dream's native `<|fim_*|>` tokens for scaffold/infill vs raw mask-runs. **KILL:** matches P1c
    scaffold loss → channel irrelevant.

### Wave 3 — engine (lossless / conditional) + deeper capability
11. **F-batched-infill: real multi-step variant-sweep batch** [ENG] med-large ~3–6d. LOSSLESS.
    Fire {n,n+2,1.4n} sweep (kintsugi.ex:337) as ONE multi-seq infill batch w/ per-seq state, gate ≤~56 tok.
    Refutes corpus's "pp batching flat" claim (1.96x @32tok measured, where 75% of infill wall lives).
    Stubs: `07_layer_f/permutation-batched-infill/` (batchprobe-realsizes.diff, sweep-trace.csv).
    Expect +6–10% deliverable, pass set unchanged. **KILL:** <+5% after step-divergence OR breaks pass-exactness.
12. **BankF-2: attach backend sampler once at ctx creation** [ENG] small ~1–2d.
    Removes ~11ms re-reserve + ≤8-mask attach penalty. **KILL:** repair path no longer dominant after #11/G8 → skip.
13. **Fine-tune-the-repairer (CYCLE/LeDeX LoRA)** [CAP] large ~5–10d. Depends on #7 corpus infra.
    Repair-specialized LoRA on the 7B masked repairer (the 3/30→35/48 workhorse, never tuned).
    **KILL:** FT'd repairer + compile/test feedback still can't pass c-tier → feedback can't grant capability, stop.
14. **Python-derived oracle selection (CodeChemist/TransCoder-ST)** [FARM] medium ~3–4d. After #6.
    Derive language-agnostic oracles in Python, select Elixir candidates by pass-rate; STATELESS only.
    **KILL:** oracle best-of-N ≤ seed-only best-of-N on stateless → adds nothing.
15. **LoRA-on-Dream e2e smoke test** [CAP] small ~1–2d. Run ONLY if #7 shows diffusion-FT worth it.
    100-step PEFT → convert_lora_to_gguf → llama-diffusion-cli --lora A/B. De-risk diffusion adapter tier.

### Deferred / conditional (laptop-buildable, low priority or gated)
16. **F5 chunked prefill** [ENG] med — lifts canvas>512 HTTP-500 ceiling. Correctness, not speed. Build only if a workload sends >512-tok canvases.
17. **Item-4b encode() logits-flags** [ENG] med — ~2GB VRAM relief. Likely CLOSE: Dream Q4 already fits 6GB (4946 MiB, ~1.2GB free); only helps models that don't fit anyway.
18. **Item-5 real masked prefix-KV cache (Fast-dLLM DualCache)** [ENG] LARGE weeks. **DO NOT START** until F-1=diffusion-stays AND diffusion-speed-matters-on-Pascal (R1 rig). Threshold decode already harvested most headroom (191 tok/s raw shipped).
19. **F2 closed-loop speculation w/ trained drafter** [THR] large >1wk. Permanently parked unless throughput becomes the goal AND a trained drafter exists free from #13.

---

## RIG-PARKED (run when 9×P106 in hand) — re-measure laptop verdicts on sm_61, quality-gated
- **R1 diffusion-vs-AR SPEED half** (#4 speed): per-step diffusion vs AR tok/s on one P106 — F-1 speed evidence.
- **R2 E-rig deployment validation**: FastDLLM-1.5B Q4 (986MB)+block-kv AND Dream-7B Q4 (4946 MiB) on one P106; ms/step, VRAM, llama-bench pp512/tg128, quant sweep Q4_0/Q4_K_M/IQ4_XS, 14 sampler tests. Runbook `p106-mining-fleet.md` sec 9.
- **R3 GGML_CUDA_FORCE_GRAPHS A/B** on sm_61 (flag shipped) — Celeron host plausibly launch-bound (1.5x on battery).
- **R4 9-card candidate-farm payoff** (#6 at N=9): wall-clock-to-first-passing N=1 vs N=9.
- **R5 quality-gated AR baseline** on real P106 — the overnight doc's Stage-0 quant sweep, but reported WITH Elixir pass-rate (reject raw tok/s).

---

## DO NOT REDO (measured-closed — listed so we don't re-propose)
- Catalog G9 canvas/scaffold-seed (−9 pass, −47% tok/s, p<0.002) · G11 grammar decode (7→4, 21× slower; also NOT novel, arXiv:2508.10111) · G13 check-first (0 fixed, NL −7 real loss).
- Layer B all levers (tau-decay poison, Prophet-on-drafts refuted, span/anchor killed) · Layer C all but C1a (C4/C5/C6/win_route refuted; multi_hole NEVER fired across 48 cases both models).
- Layer D4 hybrid with WEAK 1.5B drafter (35/48 same set, half tok/s) — strong-drafter variant (#8) is distinct.
- Layer F1 single-GPU racing (ratio 2.7<3) · catalog F6/F7/F9/F11 (graphs zero on AC, host 0.06ms, no static prefix-KV).
- Layer E decode-side complete (E2/E3/E4 shipped; E5a/b/c killed — 1.85 commits/step is model's honest rate; quant ladder ±2).
- Multi-card tensor/row split to speed ONE stream (PCIe1.0x1 = 250MB/s; tg bandwidth-bound) — only independent-streams (farm) helps.
- Naive deterministic Python→Elixir transpiler-at-inference (none exists; fixes syntax not logic) — Pylixir-as-DATA-GEN (#7) is different.
- Bigger diffusion model as rig engine (DG 16GB, RND1 31GB, LLaDA2.0 overflows, Stable-DiffCoder can't infill).
- Shipped & done: backend GPU sampling, degeneracy guard, multi-replica server, k-stride RAM fix (1997→783MB).

## Verification (how each item is judged)
- Build CUDA `61-real` (CUDA 12.x), FA off, INT8 dp4a/MMQ path. Start diffusion server, `curl /health`.
- Run `kintsugi/bench/bench.exs`; record **X/48 pass + tok/s + c_shout/c_stack** before/after. Throughput items
  MUST show pass-rate unchanged (lossless) or improved — never tok/s alone.

## Unresolved questions
1. Which model GGUFs are already on disk vs need downloading? (Dream-Coder-7B, Qwen2.5-Coder-7B, Qwen2.5-Coder-1.5B base for QLoRA all appear needed.)
2. Is the laptop free for overnight QLoRA without contending with daytime bench runs? (#7/#13/#15 assume it.)
3. Confirm cards on order are 1280-core P106-100 (6GB) not 768-core P106-090 — changes rig projections ~1.6×.
4. Target deliverable: interactive single-stream latency, or offline throughput-to-first-passing? (re-ranks THR vs FARM).
5. Apply the doc-corrections the integration memo lists (target-rig-hardware 6GB, survey fine-tune section, VERDICT refs) now, or keep docs frozen until probes land?
