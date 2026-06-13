# Elixir-capable models worth swapping into the kintsugi harness

Survey result (workflow `elixir-model-survey`, 6 agents, mid-2026). Companion to the
Layer G verdict (the Elixir ceiling is MODEL CAPABILITY); this surveys what to swap in.

---

# Elixir-Capable Models Worth Swapping Into the Kintsugi Diffusion Harness
*Decision-grade survey, mid-2026. Synthesized from 5 research threads: diffusion landscape, Elixir capability, AR drafters, harness fit, VRAM/quant.*

## 1. The Core Tension

The kintsugi harness needs a **masked-diffusion** model (one with a `mask_token_id`) to run its full DRAFT+INFILL repair loop. That requirement collides head-on with where Elixir competence actually lives:

- **The diffusion models the fork can load are all ~7-8B and weak at low-resource Elixir.** Every supported masked arch (`dream`, `llada`, `llada-moe`, `rnd1`) tops out around 7-8B in something that fits VRAM. None has *any* published Elixir number -- all diffusion code models (DiffuCoder, Dream, Dream-Coder, LLaDA, RND1, Fast-dLLM) are evaluated Python-only. So Elixir-on-diffusion is genuinely uncharted.

- **The models that are actually good at Elixir are large AR models that break the premise.** The only public per-model Elixir benchmark (AutoCodeBench, arXiv:2508.09101, 200 Elixir problems) shows decent Elixir at <=8B is reachable ONLY by code-specialist 7Bs (Hunyuan-Coder-7B 60.1, Seed-Coder-8B 57.1, Qwen2.5-Coder-7B 47.0), and the dependable jump (65-82) needs 14B+reasoning or 32B-coder or frontier proprietary. ALL of these are autoregressive -- they cannot be the diffusion repairer. They can only serve as an AR drafter in the D4 hybrid.

- **VRAM closes the trap.** On the 8GB laptop only ~3GB is free in practice; usable resident footprint is ~6-6.5GB (8GB) / ~4.7-5GB (6GB rig). That caps you at one 7-8B-Q4 model resident. A 14B coder (8.99GB Q4_K_M) does not co-reside with a 7B repairer, and dense 14-32B offload drops the repair loop to ~1-4 tok/s (from the resident ~9-10 tok/s deliverable). 32B-coder -- the real Elixir jump -- is AR, doesn't fit, and can't infill.

- **Conversion caveat compounds it.** This fork's `convert_hf_to_gguf.py` has ZERO diffusion references (grep-clean). Any new diffusion model must arrive as a pre-made GGUF whose `general.architecture` already equals a supported tag AND carries `tokenizer.ggml.mask_token_id`, or you convert with external/patched tooling. A plain `qwen2`/`llama`-tagged GGUF loads but silently takes the AR path, not the masked path.

**Net:** there is no 12-14B masked-diffusion model that is both better-at-Elixir and loadable by the full harness today. The 7B Elixir ceiling is a model-capability ceiling that only breaks near 32B-coder, which is architecturally and physically out of reach for the diffusion role.

## 2. Ranked Shortlist

Arch column legend: **MD** = diffusion-maskable (full harness), **BAR** = block-AR drafter-only (no infill), **AR** = autoregressive drafter-only.

| # | Model | Arch role | Params + Quant | VRAM 8GB? / 6GB? | Elixir evidence | Harness role | GGUF availability | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | **Dream-Coder-7B-Instruct** | MD (`dream`) | 7B Q4 ~4.7GB | Yes / Yes | None direct; code-tuned, LiveCodeBench 21.4, HumanEval 82.9 (vs Dream-7B 63.4) | **Drop-in full harness** (byte-identical arch to on-disk Dream-7B) | Yes: mradermacher base/instruct, +i1, DevQuasar | **TRY NOW** |
| 2 | **Qwen2.5-Coder-7B-Instruct** | AR | 7.6B Q4_K_M 4.68GB | Yes / borderline | Strongest small-model Elixir data point; AutoCodeBench-Elixir 47.0; CodeQwen1.5 lineage ~38-40 (McEval) | **Hybrid drafter** (Dream repairer) | Yes: official Qwen | **TRY NOW** |
| 3 | **DiffuCoder-7B-cpGRPO** | MD (`dream`-compat) | 7B Q4_K_M 4.6GB | Yes / Yes | None direct; on-disk; already a measured ceiling failure on multi-fn specs | Drop-in full harness (baseline, already tested) | Yes: Mungert | already shipped |
| 4 | **LLaDA-1.5** | MD (`llada`) | 8B Q4 ~5GB | Yes / tight | None direct; code is its weak axis even post-VRPO | Drop-in full harness (shift_logits=false) | Convertible via LLaDA-8B tooling; no dedicated GGUF | TRY LATER |
| 5 | **Seed-Coder-8B-Instruct** | AR (`llama`-class) | 8B Q4 ~5GB | Yes / tight | AutoCodeBench-Elixir 57.1 (2nd-best <=8B) | Hybrid drafter | Check availability | TRY LATER |
| 6 | **Granite-3.x-8B-Code** | AR (`granite`) | 8B Q4_K_M 4.88GB | Yes / Yes | No isolated #; weak on functional langs | Hybrid drafter (Apache-2.0 fallback) | Yes: bartowski | TRY LATER |
| 7 | **Qwen2.5-Coder-14B-Instruct** | AR (`qwen2`) | 14.7B Q3_K_M 7.34GB | tight (no co-resident) / No | No isolated #; > Coder-7B McEval avg | Hybrid drafter (ceiling probe, 8GB only) | Yes: bartowski | TRY LATER |
| 8 | **LLaDA-MoE-7B-A1B-Instruct** | MD (`llada-moe`) | 7B/1.4B-act Q4_K_M 4.52GB | Yes / tight | None direct; paper: only ~Qwen2.5-3B-class code | Drop-in full harness (low expectation) | Yes: markldn Q4_K_M, rockon1095 Q4_0 | TRY LATER (low odds) |
| 9 | **Stable-DiffCoder-8B** | BAR (custom) | 8B Q4 ~5GB | Yes / tight | None direct; strongest ~8B code dLLM (~Qwen2.5-Coder-32B on BigCodeBench/LCB) | **Needs new C++ loader**; block-diffusion likely DRAFT-only (infill questionable) | Not found | TRY LATER (high value/high effort) |
| 10 | **DreamOn-7B** | MD (`dream`-family, new infill mech) | 7B | Yes / Yes | None direct; variable-length infill via `<expand>`/`<delete>` | Drop-in arch but new infill mechanism harness doesn't speak | Check | TRY LATER |
| 11 | **RND1-Base (30B-A3B)** | MD (`rnd1`) | 30B/3.3B-act; Q8-only GGUF ~31GB | No / No (offload-only) | None direct; base-only, "SOTA among open DLMs for code" | Full harness via `--n-cpu-moe`; needs Q4 quant produced | Q8_0 only (vikramkr); no Q4 | SKIP (for now) |
| 12 | **LLaDA2.0-mini** | MD (custom, != llada-moe) | 16B/1.4B-act Q4 ~9-10GB | No / No | Yes (code+math strong for size) | Needs new loader; overflows 8GB | Not found | SKIP |
| 13 | **LLaDA2.0-flash** | MD (custom) | 100B/~6B-act | No / No | SOTA dLLM code | Needs new loader; too big | Not found | SKIP |
| 14 | **Qwen3-Coder-30B-A3B** | AR (`qwen3moe`) | 30.5B Q4 18.6GB | No / No (offload) | Strongest code of set; Qwen3-8B-Elixir 43.9 (dense ref) | Hybrid drafter only via offload (kills draft speed) | Yes: unsloth | SKIP (offload too slow) |
| 15 | **Codestral-22B** | AR | 22B Q4 12.6GB | No / No | 80+ langs incl Elixir, no isolated #; non-commercial MNPL | Hybrid drafter, out of budget | Yes: bartowski | SKIP (size+license) |
| 16 | **DiffusionGemma-26B-A4B** | canvas (`diffusion-gemma`) | 26B/A4B | offload / No | None | Forge/canvas only, CANNOT infill | Yes (on disk) | SKIP (no repair) |
| 17 | **Open-dCoder-0.5B** | MD (tiny) | 0.5B | trivial / trivial | None; too small to beat 7B ceiling | Loader work + too weak | Not found | SKIP |

## 3. Candidate Buckets

### (a) DROP-IN full-harness diffusion swaps (zero C++, loadable today)
- **Dream-Coder-7B-Instruct** -- byte-identical to on-disk Dream-7B (`DreamModel`/`Dream`, 3584h/28L, mask=151666), so it reuses `src/models/dream.cpp` with no loader work. The ONLY candidate that is simultaneously (a) code-tuned beyond the current ceiling, (b) loadable today with no C++, (c) fits 6GB at Q4. Confirm `general.architecture == dream` + `tokenizer.ggml.mask_token_id` post-download.
- **LLaDA-1.5** -- native `llada`, GGUF-convertible via LLaDA-8B tooling, fits 8GB (tight on 6GB). Code is its weak axis; lower odds than Dream-Coder.
- **LLaDA-MoE-7B-A1B-Instruct** -- native `llada-moe`, GGUF exists (markldn Q4_K_M 4.52GB), fits. But ~1.4B active -> only ~Qwen2.5-3B code class; likely below the 7B-dense ceiling. Quick low-expectation eval only.
- **DiffuCoder-7B-cpGRPO** -- already on disk and already a measured failure on multi-function specs; listed for completeness, not a new lever.

### (b) HYBRID-drafter swaps (stronger AR drafter + Dream/Dream-Coder repair)
The D4 hybrid was previously REJECTED using a weak Fast-dLLM-1.5B drafter. The untested, evidence-backed lever is a code-specialist 7B drafter:
- **Qwen2.5-Coder-7B-Instruct (Q4_K_M, 4.68GB)** -- best code-per-byte that fits; strongest small-model Elixir data point (AutoCodeBench 47.0). Primary hybrid bet. NOTE the on-disk drafter is the *generalist* `Qwen2.5-7B-Instruct`, NOT the Coder -- the Coder GGUF must be downloaded to test D4 properly.
- **Seed-Coder-8B-Instruct** -- AutoCodeBench-Elixir 57.1, 2nd-best <=8B; alternative drafter.
- **Granite-3.x-8B-Code (Q4, 4.88GB)** -- Apache-2.0 fallback, fits both boxes, weaker on functional langs.
- **Qwen2.5-Coder-14B (Q3_K_M, 7.34GB)** -- ceiling-probe drafter for the 8GB dev box only; measures the AR-draft upper bound. Won't co-reside with a repairer.

**Hard co-residency blocker:** Dream-7B Q4 (~4.7GB) + Coder-7B Q4 (4.68GB) ~= 9.4GB > 8GB. D4 hybrid requires sequential load/swap (accept reload latency), one model on CPU, or a smaller drafter. The HIGHER-leverage half-step that costs nothing: pair ANY drafter with **Dream-Coder-7B as the repairer** instead of base Dream-7B, since the repairer is what's capped and Dream-Coder is a drop-in.

### (c) NOT viable (why)
- **Stable-DiffCoder-8B** -- best ~8B code dLLM but custom `StableDiffcoderForCausalLM` block-diffusion arch -> new fork loader (~200-300 LOC) AND infill likely doesn't work (block-causal attention structurally prevents it, like fast-dllm). High value, gated.
- **RND1-30B-A3B** -- native `rnd1` but 30B resident; only Q8 GGUF published (~31GB), base-only (no chat tuning). Viable only via `--n-cpu-moe` after producing a Q4 quant -- research bet, not a swap.
- **LLaDA2.0-mini/flash** -- newer inclusionAI arch != `llada-moe`, needs new loader; even mini (16B) overflows 8GB at Q4.
- **Qwen3-Coder-30B / Codestral-22B / Qwen2.5-Coder-32B** -- AR-only (can't infill); too big without offload, and dense offload kills the repair loop (~1-4 tok/s). Codestral also non-commercial.
- **DiffusionGemma-26B** -- canvas/forge only, CANNOT infill -- excluded from the full harness by design.
- **Open-dCoder-0.5B** -- too small to beat the 7B ceiling; needs loader work anyway.

## 4. Concrete Empirical Next Step

Download and smoke-test these, in order:

1. **mradermacher/Dream-Coder-v0-Instruct-7B-GGUF (-i1 imatrix preferred), Q4_K_M.** This is the single highest-value, lowest-cost experiment.
   - Verify: `general.architecture == dream` and `tokenizer.ggml.mask_token_id` present (expect 151666), then `curl /health` should show `family: masked`, `mask_token_id != null`.
   - Run the full DRAFT+INFILL harness as both drafter AND repairer; re-run your multi-function Elixir spec suite (the 35/48-pass set from the D4 memo).
   - **Expected outcome:** modest improvement over Dream-7B/DiffuCoder on Python-shaped Elixir; UNPROVEN on multi-function specs. There is no Elixir number for any diffusion model, and AutoCodeBench's best <=8B coders (47-60) are on *translated single-function* problems -- easier than your failure mode. Plausible to pass a handful more specs; unlikely to be a category change.

2. **Qwen/Qwen2.5-Coder-7B-Instruct-GGUF, Q4_K_M (4.68GB)** -- the real D4 hybrid test the memo flagged as untested.
   - Run as AR drafter + **Dream-Coder-7B** repairer (time-shared/swapped to dodge the 9.4GB co-residency overflow on 8GB).
   - **Expected outcome:** this is the highest-upside path because the *drafter's* Elixir competence is the variable that actually improved (47.0 AutoCodeBench-Elixir vs the rejected Fast-dLLM-1.5B drafter). Still bounded: even Qwen3-8B hits only 43.9 -- the bet is that masked-infill REPAIR closes the residual gap, not that the drafter clears it alone.

3. (Optional, low expectation) **markldn/LLaDA-MoE-7B-A1B-Instruct-Q4_K_M-GGUF (4.52GB)** -- a cheap eval of a different masked family. Expect it to NOT clear the ceiling (~Qwen2.5-3B code class); run only if 1-2 are inconclusive and you want a second native-diffusion data point.

Do #1 and #2 first; they are the two moves with real upside. #2 requires the swap/time-share workaround on the 8GB box.

## 5. Honest Bottom Line

**A model swap is unlikely to decisively break the multi-function-Elixir ceiling; it is most probably marginal -- a handful of extra specs, not a category change.** Specifics:

- The ceiling is a **low-resource pretraining-representation** problem, not a harness problem. Elixir sits in the long tail of The Stack v1 (below Lua's ~6.58GB / Haskell's ~6.95GB; the-stack-smol-xl Elixir shard is ~51MB) and is absent from small-StarCoder2's 17-language set (arXiv:2211.15533; arXiv:2402.19173). No 7B fixes that by architecture choice.
- AutoCodeBench (arXiv:2508.09101) is the only multi-model Elixir benchmark, and it shows the dependable jump is **~32B-coder or 14B+reasoning or frontier** -- all AR, none loadable as the diffusion repairer, none fitting VRAM (Qwen2.5-Coder-7B 47.0 vs Qwen2.5-Coder-32B 59.6 vs Claude Opus 4 82.3). And those scores are on *translated single-function* problems, easier than your measured multi-function failure mode -- so even a strong AutoCodeBench number does not guarantee multi-function success.
- Zero diffusion model has ANY published Elixir number (DiffuCoder arXiv:2506.20639, Dream arXiv:2508.15487, Dream-Coder arXiv:2509.01142, LLaDA, RND1 all Python-only). You are operating where there is no baseline; any gain is plausible-but-unproven.

**What the swap CAN deliver, and why it's still worth doing:** Dream-Coder-7B is a free, zero-C++ upgrade of the repairer (a much stronger Python coder than the Dream-7B you ship: HumanEval 82.9 vs 63.4), and the Coder-7B-drafter D4 hybrid is the one untested high-upside lever from the D4 memo. Both are low-cost. The realistic expectation is incremental improvement plus, importantly, **the first actual Elixir data points for diffusion code models** -- which is itself decision-grade information you currently lack.

**What will NOT work:** "just load a bigger diffusion model on the same box" (none exists in 12-14B that is both better-at-Elixir and full-harness-loadable); dense 14-32B offload (too slow for the iterative repair loop, ~1-4 tok/s); and any AR-coder-as-repairer (architecturally impossible -- they can't infill).

Relevant local paths: `/home/car/projects/llama.cpp/src/models/dream.cpp` (loader Dream-Coder reuses), `/home/car/projects/llama.cpp/gguf-py/gguf/constants.py` (arch registry ~lines 509-518, 1068-1078), `/home/car/projects/llama.cpp/convert_hf_to_gguf.py` (no diffusion support -> conversion caveat), `/home/car/projects/llama.cpp/examples/diffusion/diffusion-server.cpp` (metadata read + /health), on-disk drafter `/home/car/models/qwen7b/Qwen2.5-7B-Instruct-Q4_K_M.gguf` (generalist, NOT the Coder -- Coder-7B GGUF must be downloaded for D4).
