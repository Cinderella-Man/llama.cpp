# D4 hybrid: FastDLLM-1.5B drafts + Dream-7B repairs - PLAN / WORK LOG

## Context

Catalog item D4 (two-model draft/verify), reached from measurement, not theory:
05_layer_e.md E2 verdict showed FastDLLM-1.5B out-drafts Dream-7B (21/42 vs
15/42 draft+Credence-only) at ~3x lower draft latency, while Dream's full-system
lead (35/48) is entirely its repair ladder - which FastDLLM structurally cannot
have (block-causal attention = no infill). The complementary shape IS D4.
04_layer_d.md stays parked; this is its one live item, executed as pure
kintsugi harness work (no engine changes).

## Design (grilled 2026-06-12, all decisions user-confirmed)

- KPI: SPEED IS KING. Materially lower wall than Dream-alone; score floor held
  by the final rung (Dream full system = the proven 35/48). Credence keeps
  absorbing fix classes over time, so the deterministic layer carries quality.
- Topology: BOTH models resident, two server processes - NO load/unload
  swapping (a 4.7 GB reload costs seconds vs 0.3-0.5 s drafts). Dream Q4_K_M
  4.68 GB + FastDLLM Q4_K_M 0.99 GB + buffers ~= 6.4 GB: fits the 8 GB dev
  GPUs, NOT the 6 GB rig target (user correction 2026-06-12: rig budget is
  6 GB VRAM, not the 3 GB P106 assumed in 05). Rig-fit is a separate later
  step: Dream Q3_K_M (~3.6 GB, repair quality unmeasured) vs partial offload.
- Ladder (per case): FastDLLM draft -> 1 reseeded FastDLLM retry (canvas
  doubling as in generate_with_stats) -> 2 Dream cross-repairs on the failed
  draft -> full Dream redraft WITH its own capped ladder.
- Cross-repair is text-level: the masked FastDLLM draft is an ordinary infill
  request to the Dream server; provenance does not matter mechanically. The
  open question D4 answers: does Dream repair foreign drafts as well as its
  own?
- HARD CAP rationale (user concern, on record): a LOGICALLY flawed draft
  cannot be fixed by inpainting - infill is conditioned on the surrounding
  draft, so a poisoned global structure yields locally-plausible fills that
  serve the wrong algorithm. Bounded repairs (2) make hopeless drafts cost at
  most two cheap infills before escaping to a redraft. Unbounded repair
  cascades are also the machine-crash signature (03_layer_c.md post-mortem).
- Bench accounting: full 48-case run, profile "d4". Tier routing mirrors the
  deployed system: p-tier (forge) = hybrid ladder; h-tier (heal) and i-tier
  (infill) = Dream directly (FastDLLM cannot infill; heal is repair work =
  Dream's job). Comparable head-to-head with Dream-alone 35/48 and FastDLLM
  draft-only 21/48.
- Comparison baselines (same machine, same day): 20260612T152720Z-wr-baseline
  (Dream full system) and 20260612T170845Z-e2-fastdllm-t09 (draft-only).

## Implementation plan

1. lib/kintsugi.ex: generate_hybrid(draft_eng, repair_eng, instruction, opts)
   - rung 1-2: attempt FastDLLM drafts (conf_threshold 0.9 - the model's own
     scale, 05_layer_e.md de-temper lesson) with NO model-side repairs
     (max_repairs 0); verify + free Credence round as usual.
   - rung 3: heal(repair_eng, failed_draft, max_repairs: 2) - Dream params
     (conf_threshold 0.6).
   - rung 4: generate_with_stats(repair_eng, instruction, max_drafts: 2) -
     the standard Dream system.
   - stats: per-rung wall + which rung delivered (rung field in bench rows).
2. bench/bench.exs: optional 4th arg = draft engine URL; profile "d4"; when a
   draft engine is connected, :forge cases dispatch via generate_hybrid,
   :heal/:infill via the repair engine unchanged.
3. Smoke: 1-2 single cases, temp spot-check, VRAM headroom check
   (nvidia-smi) with both servers resident.
4. Full 48-case bench, temp checks before/after, everything committed first.

## Safety preconditions (machine-crash rules)

- All work committed before GPU load; doc written before implementation.
- Both-resident VRAM verified before benching; display GPU shares the 8 GB.
- No other Elixir VSCode workspaces open (2026-06-12 OOM was ElixirLS in
  opc-sft-stage2-elixir at 14 GB RSS, NOT the bench - exonerated, see
  memory/gpu-sustained-load-crash.md).

## Work log

(to be filled as executed)

## Unresolved questions

1. Which failed draft does rung 3 repair - the retry (current plan: the last
   one) or the better-scoring of the two? (v1: last; revisit if rung-3 yield
   is low.)
2. Rig-fit at 6 GB: Q3_K_M Dream vs partial offload - separate measured step.
