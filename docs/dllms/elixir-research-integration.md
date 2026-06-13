# Elixir codegen: integrated path forward (read this first tomorrow)

Fuses TWO research efforts into one actionable plan:
1. Our MEASURED Layer-G campaign (`docs/dllms/dllm-grammar-scaffold-research/`, start at
   `VERDICT.md`): scaffold / grammar / check-first all measured -> all regress; the
   bottleneck is model CAPABILITY, not syntax; the GPU masked-infill repair loop is the
   workhorse (carries 3/30 first-shot to 35/48).
2. The user's PARALLEL research (Pascal hardware, MultiPL-T fine-tuning, Python-pivot,
   grammar format-tax, self-repair literature), pasted in 2026-06-14.

Produced + cross-verified by a multi-agent workflow (`elixir-research-integration`). It
CORRECTED an earlier draft on three points: the deploy tier is 6GB not 4GB; MultiPL-T may
"eat itself" (fine-tunes on single-function shape = what we already pass); and the 35/48 is
carried by GPU infill-repair, not Credence's regex layer.

================================================================================
## START HERE TOMORROW (the one experiment that matters)

STAGE 1 de-risk pilot, ~1 week, run on the 8GB RTX 5070 laptop (NOT Pascal):
  1. Build a MINIMAL "Pylixir": Python-assertion -> ExUnit transpiler, ~50 seeds only
     (this is the missing piece - Elixir is NOT in MultiPL-E, so the test-translation infra
     does not exist yet). Scope to stateless functions; do NOT attempt OTP/GenServer.
  2. Generate a tiny synthetic Elixir corpus with your 16-20B model, FILTER to only
     translations whose ExUnit tests pass (reuse kintsugi `Verifier` for compile+run).
  3. QLoRA fine-tune Qwen2.5-Coder-1.5B overnight (~5-20 GPU-hrs, fits the 5070).
  4. Measure ONLY on c_shout / c_stack / multi-function specs - NOT aggregate (aggregate
     hides the single-function trap).
  KILL-GATE: if the multi-function needles do not move, STOP - the ceiling is reasoning/
  scale, not fine-tune coverage, and every later stage is foreclosed. This ~1 week
  FALSIFIES the top move before committing the 3-6 week full build.

In PARALLEL (free, ~30 min): the Dream-Coder-7B drop-in smoke test (Stage 5 kill probe) -
download mradermacher/Dream-Coder-v0-Instruct-7B-GGUF Q4_K_M, point the server at it,
`/health` should show family: masked / mask 151666, run `mix run bench/bench.exs`. It fits
the 6GB rig (4946 MiB) and gives the FIRST-EVER Elixir number for a diffusion code model.
Expected: marginal. Settles the diffusion-engine question.
================================================================================

# INTEGRATED PATH FORWARD: Elixir Codegen on the P106 Deployment Tier

Fuses the user's parallel research (MultiPL-T adversarial assessment, hardware reality probe, our-evidence reconciliation) with our measured Layer-G campaign (VERDICT.md P0-P4b, elixir-model-survey, p106-mining-fleet, target-rig-hardware). Decision-grade. ASCII only.

---

## 0. ONE HARD CORRECTION UP FRONT: the deployment tier is 6GB, not 4GB

The three input research streams disagree on the target VRAM. This must be resolved before any sizing, because it changes which models fit.

- The "OUR-EVIDENCE RECONCILIATION" stream asserts **4GB Pascal** and demands we "re-size everything."
- The "HARDWARE REALITY" stream asserts **9x Manli P106-L cards, 6GB each**, and explicitly calls the 4GB claim incorrect.
- Our own checked-in doc `p106-mining-fleet.md` sec 0 settles it: P106-090 = 3GB, **P106-100 = 6GB (GTX 1060 silicon)**, and "the 6 GB plan needs P106-100s." Sec 9 measures Dream-7B Q4_K_M at **4946 MiB at -ub 512 with ~1.2GB headroom on a 6GB card.**

**Resolution: the deployment tier is 6GB per card, compute capability 6.1 (sm_61).** The 4GB premise in the reconciliation stream is wrong and must NOT be propagated into the docs. This matters: at 6GB a 7B-Q4 model fits with headroom; at 4GB it would not. The reconciliation stream's strategic conclusions that depend specifically on "7B does not fit" are therefore void; its conclusions that depend on the *Elixir capability ceiling* still stand (those are hardware-independent).

What is true regardless of 4 vs 6 GB (verified against `p106-mining-fleet.md` sec 2, our CUDA audit): no FlashAttention MMA path on sm_61 (scalar/vec fallback only), no bf16, FP16 at 1/64 rate (avoid F16-compute), INT8 dp4a + MMQ quantized matmul is the good path, GPU backend sampling fully supported (critical: mining hosts are Celeron-class), CUDA graphs gated off below Ampere, build needs CUDA 12.x (not 13) with `61-real`.

---

## 1. WHERE THE TWO EFFORTS CONVERGE, AND WHERE THEY CONFLICT

### Strongest mutually-confirmed conclusions (high confidence)

1. **The bottleneck is semantic/capability, not syntax.** Our P1 is the decisive proof: scaffold drove parse errors 12->0 (100% of syntax eliminated) but pass stayed 3->3 (VERDICT.md). The user's translation-literature claim ("logic errors are language-independent; pivot helps syntax not logic") reaches the identical conclusion from an independent angle. This is the single strongest cross-confirmation. DiffuCoder cross-check generalizes it (parse 27/30->0, pass unmoved; `dc-results.md`).

2. **A 7B-class Elixir capability floor exists and is real.** Our c-tier ceiling: c_shout 0/189, c_stack 0/188 across every profile and both models (Dream + DiffuCoder). The user's external coordinates corroborate: AutoCodeBench (arXiv:2508.09101, 200 Elixir problems) and McEval (Qwen2.5-Coder-32B 65.9 avg across 40 langs, Elixir below avg) both show the dependable jump needs 14B+reasoning / 32B-coder / frontier. Caveat both sides agree on: those external benches are *translated single-function* problems, strictly EASIER than our multi-function failure mode, so external scores understate our ceiling.

3. **Grammar/constraint on the whole generation hurts.** Our P4b: grammar-constrained decode pass 7->4, ~21x slower (VERDICT.md). The user's CRANE (arXiv:2502.09061) / Tam et al. (EMNLP 2024) "format tax" literature predicts exactly this. Their prescription (constrain only final emission, leave reasoning free) is the untested escape hatch our negative implies.

4. **Execution-feedback repair is the right regime; self-critique is not.** Olausson (ICLR 2024, arXiv:2306.09896) says small-model self-critique repair is weak but execution/test feedback works. Our kintsugi repair loop uses compile+test feedback by construction (carries 3/30 first-shot to 35/48), so we are on the right side of the literature.

5. **A raw model swap is marginal.** The user's pessimism ("3B-at-Q4 hallucinates; prefer 1.5-3B Q4 over 7B-Q2") and our survey's verdict ("a swap is most probably marginal, a handful of extra specs, not a category change") agree.

### Genuine conflicts / tensions to flag

- **CONFLICT (hardware): 4GB vs 6GB.** Resolved above to 6GB. The reconciliation stream's "re-size everything to 4GB" action item is rejected.

- **TENSION (the MultiPL-T premise eats itself): the method most improves what we already pass.** The user's TOP MOVE (MultiPL-T Python-pivot fine-tune) is premised on pivot being valuable, but by their OWN "logic is language-independent" claim AND our P1, pivot fixes the part that was never our bottleneck. MultiPL-T's validated gains (Racket/OCaml/Lua/R/Julia) are on expression-oriented, single-function-shaped languages close to Python's evaluation model, on HumanEval-style single-function problems. Our measured failures (c_shout/c_stack, multi-function/stateful) are the exact opposite shape. So a synthetic single-function-pure-Elixir corpus teaches the model more of what it already passes. This is the sharpest risk and both the user's adversarial stream and our P0 taxonomy independently land on it.

- **CONFLICT (decisive, on "reuse the recipe"): Elixir is not in MultiPL-E.** MultiPL-T's test-validation step depends on MultiPL-E's Python-assertion->target "little compilers." MultiPL-E (22 langs incl. Ada) has no Elixir/Erlang. So step 0 (translate Python unit tests -> runnable ExUnit assertions) is UNBUILT. "Reuse a validated recipe" is actually "build the missing infrastructure first, on a language MultiPL-T was never validated on." Verified against arXiv:2308.09895 and the nuprl repos.

- **NUANCE (keep Claim 3 honest): the deterministic Credence layer rescues 0/30 real draft failures (P0).** The 35/48 is carried by GPU masked-infill repair with compile+test feedback, NOT by Credence's regex/rule layer. And execution feedback does NOT grant missing capability (the 0/189 floor proves it). Do not overclaim "Credence reaches 35/48."

---

## 2. THE REFRAMED PROBLEM

"Swap in a bigger diffusion model" is dead as a strategy, for three independently sufficient reasons:

1. **No bigger masked-diffusion model exists at the tier.** There is no 12-14B masked-diffusion model that is both better-at-Elixir and loadable by the full harness (survey sec 1). Going down (1.5-3B) buys nothing: Fast-dLLM-v2-1.5B is block-AR (block-causal attention = no infill), and there is ZERO open 1.5-3B pure-masked model with any Elixir evidence (HARDWARE REALITY sec 4). Going up (RND1-30B, LLaDA2.0) overflows 6GB or needs a new loader.

2. **The ceiling is a model-representation problem, not a harness problem.** Elixir sits in the long tail of The Stack (smol-xl shard ~51MB), absent from small-StarCoder2's language set. Swapping Dream-Coder-7B for DiffuCoder-7B adds zero masked-diffusion Elixir evidence and cannot move a pretraining-representation deficit.

3. **The deployment hardware reshapes the engine even when it does fit.** Dream-7B Q4_K_M *does* fit 6GB (4946 MiB at -ub 512), so the diffusion harness is architecturally alive at the tier. But the 9-card farm is "embarrassingly parallel independent requests over PCIe 1.0 x1" (fleet sec 3) -- a candidate-farm, not a tensor-split cluster. The hardware's real gift is throughput-via-replication for offline GENERATION, not a smarter single forward.

**The reframe:** the only lever that attacks the binding constraint (semantic capability on multi-function/stateful Elixir) is changing the MODEL'S knowledge, i.e., fine-tuning -- and even that is bet, not guarantee, because the cheap synthetic data is single-function-shaped. The harness assets (verification ladder, execution-feedback repair, candidate farm, GPU sampling) are model-agnostic and survive any model decision. So the strategy splits cleanly: **invest in capability (fine-tune) + preserve the verification loop; stop investing in draft-side inference tricks and model swaps.**

---

## 3. STAGED PLAN, RANKED BY ROI

ROI = (probability of moving the multi-function ceiling) x (gain if it moves) / (effort + compute). Each stage has an explicit kill-criterion so we stop fast on the most likely failure (the single-function trap).

### STAGE 1 (HIGHEST ROI -- DE-RISK PILOT): minimal Pylixir + tiny corpus + overnight QLoRA, measured ONLY on the failing specs

- **What:** Build a minimal Python-assertion -> ExUnit transpiler ("Pylixir") for ~50 seeds only (the missing MultiPL-E piece). Generate a tiny validated corpus (generate -> compile -> ExUnit -> filter), reusing the Tunex/Credence harness. QLoRA Qwen2.5-Coder-1.5B overnight on the 5070 laptop. Measure specifically on c_shout / c_stack and the multi-function specs -- NOT on aggregate pass rate (aggregate would hide the single-function trap).
- **Evidence basis:** Ours -- P0/P1 say the ceiling is semantic and single-function syntax is already solved, so the only honest test is whether fine-tune moves the *multi-function* needles. Theirs -- MultiPL-T (arXiv:2308.09895) verified gains on single-function shape; QLoRA (arXiv:2305.14314) 1.5B is trivial VRAM (4-6GB, fits 5070); their own recommended de-risk sequence (a-d).
- **Effort/compute:** Pylixir-minimal ~3-5 person-days (scoped to assertions on stateless functions, NOT full OTP). Pilot pipeline ~2-3 days. QLoRA ~5-20 GPU-hours on the 5070 (overnight). Total ~1-1.5 weeks. Train on the 5070, NOT Pascal (no bf16, FP16 1/64, no FA -> pathologically slow training).
- **Expected gain:** If the bet is right, a few multi-function specs move. Realistically (per both streams' analysis) +10-20 pass@1 on single-function Elixir, little movement on multi-function.
- **KILL-CRITERION:** if the c_shout/c_stack/multi-function needles do not move on the pilot, the full pipeline will not move them either. STOP -- the ceiling is reasoning/scale, not fine-tuning coverage. This kill gate is the entire point of staging: it costs ~1 week to falsify the user's TOP MOVE before committing the 3-6 week full build.

### STAGE 2 (CONDITIONAL on Stage 1 GO): scale the fine-tune corpus + pivot for STATELESS logic only + Python-derived oracles

- **What:** Only if Stage 1's needles moved. Expand Pylixir to fuller coverage (still scoping OTP as a deliberate decision -- you cannot pivot a Python function into a GenServer; there is no source-side construct). Scale seed selection (fan generation across the 9-card P106 farm -- generation is embarrassingly parallel, fleet sec 3 -- not the laptop, which is multi-day wall-clock). Use the pivot ONLY for stateless/single-function logic, and use Python-derived test oracles (CodeChemist arXiv:2510.00501, TransCoder-ST) to strengthen the verification signal our repair loop consumes.
- **Evidence basis:** Ours -- the pivot's bounded prize for us is idiom/API knowledge (kills our measured `String.count`, hallucinated `:lang.to_list`), NOT logic (P1). Theirs -- pivot helps syntax/idiom not logic; CodeChemist "up to 69.5% on Lua" via Python-derived oracles; "no deterministic Python->Elixir transpiler exists" (explains why our purely-syntactic grammar/scaffold could not bridge semantics).
- **Effort/compute:** Pylixir-full + pipeline ~12-22 person-days (the un-budgeted bulk the user's adversarial stream flags as ~5-8x the implied "QLoRA a 1.5B" cost). Generation = multiple days wall-clock on laptop, OR hours fanned across the P106 farm.
- **Expected gain:** Solidify single-function Elixir competence + idiom/API fixes. Multi-function gain only as far as Stage 1 demonstrated.
- **KILL-CRITERION:** if expanding the corpus only lifts single-function pass while multi-function stays flat (the single-function trap realized at scale), freeze the corpus at the single-function-competent point and do not chase OTP synthesis -- it is unbuildable from a Python pivot.

### STAGE 3 (CHEAP, ORTHOGONAL): grammar on FINAL emission only

- **What:** Constrain ONLY the final code emission, leaving reasoning free -- NOT decode-wide grammar (we measured decode-grammar loses).
- **Evidence basis:** Ours -- P4b decode-grammar pass 7->4, ~21x slower; the untested escape hatch. Theirs -- CRANE / Tam et al. "constrain only final emission." Hard caveat (ours): our JSON repro shows frontier-grammar cannot enforce structure under diffusion's out-of-order commits without the full eth-sri any-order CFG-intersection (Muendler et al. 2025, arXiv:2508.10111). So "final-emission-only" is structurally harder under diffusion than in CRANE's AR setting.
- **Effort/compute:** Low to try the concept; HIGH if it requires the eth-sri port. Compute negligible.
- **Expected gain:** Marginal at best -- syntax was never the bottleneck (P1). This is a one-experiment "untested escape hatch," not a capability lever. Notably, if Stage 5 moves to a small AR model, final-emission grammar becomes tractable (left-to-right, no out-of-order problem) -- so this stage is far cheaper AFTER an AR decision.
- **KILL-CRITERION:** if it does not lift pass on the first AR-compatible try, drop it -- the JSON repro already shows the diffusion path needs a large port for no Elixir benefit.

### STAGE 4 (ALWAYS-ON, PRESERVE + EXTEND): execution-feedback repair -- what to ADD to Credence

- **What:** Keep the full verification ladder (parse/format/Credence/compile/ExUnit/best-of-N seed-retry, ModelServer serialization) -- these are model-agnostic and are the project's real assets. ADD: (a) fine-tune-the-repairer (CYCLE / LeDex) -- our repairer is currently the same capability-bound 7B, which the 0/189 floor shows feedback cannot rescue; this is a genuinely untested lever our docs don't list. (b) Python-derived oracle generation (CodeChemist/TransCoder-ST) to strengthen the test signal the loop consumes. (c) Extend deterministic Credence rules (module alignment, body remask, logit-bias bans) which are FREE relative to a weak GPU repairer.
- **Evidence basis:** Ours -- the 35/48 is carried by GPU masked-infill repair with compile+test feedback (P0); the deterministic Credence layer rescues 0/30 real draft failures, so the headroom is fine-tuning the repairer, not more rules alone. Theirs -- Olausson (execution feedback works for small models); CYCLE/LeDex (fine-tune-for-repair is the documented small-model unlock).
- **Effort/compute:** CYCLE/LeDex repairer fine-tune ~ same magnitude as Stage 1 QLoRA. Oracle generation reuses Stage 2 pipeline. Rule extensions ~days.
- **Expected gain:** Routes around syntax (already mostly handled) and, if the repairer is fine-tuned, may extend reach into semantic repairs the current 7B cannot do.
- **KILL-CRITERION:** if a fine-tuned repairer's compile+test feedback still cannot pass the c-tier specs, this confirms feedback cannot grant missing capability -- stop adding repair machinery and accept the ceiling.

### STAGE 5 (ARCHITECTURE DECISION): does the diffusion harness survive at the deployment tier?

- **What:** Decide between (i) KEEP diffusion -- Dream-Coder-7B Q4_K_M as drop-in repairer (fits 6GB at 4946 MiB, -ub 512, zero C++), optionally with a fine-tuned Qwen2.5-Coder-1.5B AR drafter (D4 hybrid; 1.5B Q4 ~1GB + Dream-7B Q4 ~4.7GB ~= 5.7GB+buffers, borderline on 6GB -> needs Q4_0/IQ4 + small ubatch or time-share); vs (ii) MOVE to small AR -- fine-tuned Qwen2.5-Coder-1.5B as both drafter and prompt-regenerate-on-test-failure self-repair (drops masked infill entirely, keeps the verification ladder).
- **Evidence basis:** Ours -- diffusion fits 6GB and the farm/threshold-decode/GPU-sampling all work on Pascal (fleet sec 9, sec 2), so the harness is alive at the tier; but DiffuCoder/Dream are measured multi-function failures, so the diffusion *engine* adds no Elixir capability. Theirs -- a fine-tuned AR 1.5B cannot infill (so it cannot be the diffusion repairer), but it can slot in as the AR drafter; on a strict-budget box AR-only with execution-feedback regenerate is the config that fits cleanest. Diffusion repair becomes dead code in production, surviving only on the 8GB dev box.
- **Effort/compute:** KEEP = low (Dream-Coder is a byte-identical drop-in). MOVE = the Stage 1 fine-tune plus a regenerate-on-failure loop (interface change to ModelServer, not a rewrite).
- **Expected gain:** The decision is downstream of Stage 1. If fine-tuning a 1.5B AR model is what moves the needle, the AR drafter + execution-feedback regenerate path is the natural production architecture, and the diffusion engine is retired to the dev box. If nothing moves the needle, neither architecture matters.
- **KILL-CRITERION:** if Dream-Coder-7B (the free drop-in repairer probe, survey sec 4 step 1) shows no multi-function Elixir improvement -- the expected outcome -- then the diffusion-engine question is settled: keep diffusion only on the dev box, ship AR + verification ladder + (Stage 1) fine-tune on the rig.

---

## 4. HONEST BOTTOM LINE

**Realistic best-case Elixir competence on the 6GB P106 tier:** solid single-function Elixir (idiom, syntax, API correctness; plausibly +10-20 pass@1 on single-function/McEval-style problems via a fine-tuned 1.5B AR coder), with the multi-function/stateful ceiling (c_shout/c_stack, OTP/GenServer) largely INTACT. The method most improves the problems we already pass and least improves the ones we fail. OTP is not reachable by a Python pivot at all -- there is no source-side construct to translate from. The deployment box runs this fine: a fine-tuned Qwen2.5-Coder-1.5B Q4 (~1GB) leaves ~5GB headroom on a 6GB card, fits with room for context, and drafts well on Pascal (pp-shaped, dp4a/MMQ path); diffusion repair, if kept, survives only on the 8GB dev box. The honest framing: this lifts the floor, it does not raise the ceiling. Raising the ceiling needs 14B+reasoning / 32B-coder, which is architecturally and physically out of the diffusion-repairer role and out of the 6GB budget.

**Single highest-ROI next experiment:** STAGE 1 -- the de-risk pilot. Build a minimal Pylixir for ~50 seeds, generate a tiny validated corpus, QLoRA Qwen2.5-Coder-1.5B overnight on the 5070, and measure ONLY on c_shout/c_stack and the multi-function specs. This costs ~1 week and directly falsifies (or validates) the user's TOP MOVE before committing the 3-6 week full build. If those needles do not move, every later stage is foreclosed and the project's honest conclusion is that the ceiling is reasoning/scale, not fine-tuning coverage. Run it in parallel with the free Dream-Coder-7B drop-in smoke test (Stage 5 kill probe), since that costs almost nothing and settles the diffusion-engine question simultaneously.

---

## 5. CORRECTIONS THIS FORCES TO EXISTING DOCS

These are the doc edits the integrated analysis requires (I have not applied them; listing for decision):

1. **`target-rig-hardware.md` (memory): KEEP 6GB, add Pascal capability facts.** The 6GB figure is CORRECT and confirmed by `p106-mining-fleet.md` sec 0/9 -- do NOT change to 4GB (the reconciliation stream's 4GB premise is wrong). ADD: the rig is 9x Manli P106-100-class cards, 6GB each, compute 6.1; no FlashAttention MMA, no bf16, FP16 1/64, INT8 dp4a/MMQ is the path, GPU sampling works, build needs CUDA 12.x + `61-real`.

2. **`elixir-model-survey.md`: ADD a fine-tuning lever section.** Currently the survey only covers model SWAPS (judged marginal). Add MultiPL-T QLoRA fine-tune (arXiv:2308.09895, arXiv:2305.14314) of Qwen2.5-Coder-1.5B as the named #1 capability lever distinct from swapping a pretrained model -- WITH the load-bearing caveats: (a) Elixir is NOT in MultiPL-E, so the Python->Elixir test transpiler ("Pylixir") is unbuilt and is a hard prerequisite, ~8-15 person-days; (b) MultiPL-T's validated gains are on single-function shape, which is exactly what we already pass -- so it lifts the floor, not the ceiling. Cite AutoCodeBench (arXiv:2508.09101) and McEval (Qwen2.5-Coder-32B 65.9 avg, Elixir below avg) as external corroboration of the floor.

3. **`elixir-model-survey.md`: NOTE no <=3B masked-diffusion model exists.** Fast-dLLM-v2-1.5B is block-AR (block-causal, no infill); there is zero open 1.5-3B pure-masked model with Elixir evidence. This forecloses "go smaller on the diffusion side."

4. **`VERDICT.md`: KEEP the core conclusion (externally corroborated); add references + four nuances.** KEEP "do NOT keep optimizing the draft" and "headroom is model capability + repair loop." ADD external refs: Olausson (arXiv:2306.09896), CYCLE/LeDex, CRANE (arXiv:2502.09061), Tam et al. (EMNLP 2024), AutoCodeBench, McEval. ADD to the "model capability" item: fine-tuning (MultiPL-T QLoRA) is the named #1 lever, distinct from "swap a pretrained model." ADD to the "repair loop" item: fine-tune-the-repairer (CYCLE/LeDex) and Python-derived oracle generation (CodeChemist arXiv:2510.00501, TransCoder-ST). ADD one sentence to the grammar section: CRANE's "constrain final emission only" is the untested escape hatch, but our JSON repro shows it is structurally hard under diffusion's out-of-order commits without the eth-sri intersection. KEEP the honest framing that the deterministic Credence layer rescues 0/30 real draft failures (P0) and execution feedback does not grant the missing c-tier capability (0/189) -- do not let any "execution feedback wins" reading overclaim.

5. **`dllm-throughput-catalog.md`: the existing G9/G11/G1 corrections in VERDICT.md still stand** (canvas-seeding REFUTED, grammar published-not-novel and pass-negative, Credence-converts-syntax-free holds only for crafted heal cases). No change from this analysis; just confirming they remain valid.

Relevant paths: `/home/car/projects/llama.cpp/docs/dllms/dllm-grammar-scaffold-research/VERDICT.md`, `/home/car/projects/llama.cpp/docs/dllms/elixir-model-survey.md`, `/home/car/projects/llama.cpp/docs/dllms/p106-mining-fleet.md`, `/home/car/.claude/projects/-home-car-projects-llama-cpp/memory/target-rig-hardware.md`, `/home/car/projects/llama.cpp/docs/dllms/dllm-elixir-harness.md`.

Sources: arXiv:2308.09895 (MultiPL-T, OOPSLA 2024); nuprl/MultiPL-T + nuprl/MultiPL-E (no Elixir); arXiv:2305.14314 (QLoRA); arXiv:2306.09896 (Olausson, self-repair); CYCLE / LeDex (fine-tune-for-repair); arXiv:2502.09061 (CRANE) + Tam et al. EMNLP 2024 (format tax); arXiv:2508.10111 (eth-sri constrained-diffusion); arXiv:2510.00501 (CodeChemist) + TransCoder-ST; arXiv:2508.09101 (AutoCodeBench, 200 Elixir) + McEval. Our measured evidence: VERDICT.md (P0-P4b), elixir-model-survey.md, p106-mining-fleet.md (sec 0/2/3/9), target-rig-hardware.md.
