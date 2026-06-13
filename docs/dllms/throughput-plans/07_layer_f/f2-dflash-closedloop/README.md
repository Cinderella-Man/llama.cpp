# F2 STEP 2 — dflash closed-loop (re-verification)

**Doc claim** (07_layer_f.md:482-519): `llama-dflash` (FastDLLM-1.5B drafter +
Qwen2.5-7B verifier, one process, 8 bench cases, max 256 tokens):

| config | tok/s | accepted/round | drafted-token efficiency |
|---|---|---|---|
| AR baseline (Qwen2.5-7B greedy) | 54.8 | — | — |
| dflash draft-len 32 | 26.1 | 5.79 | 18.1% |
| dflash draft-len 16 | 40.2 | 5.69 | 35.6% |
| dflash draft-len 8  | 34.6 | 4.11 | 51.3% |

Decision: F2 PARKED — closed-loop acceptance 5.7/round (not the oracle's 8-12),
AR baseline beats every dflash config.

## How verified
`run.sh`: extract the 8 unique cases from `/tmp/f2_drafts.jsonl` →
`cases8.jsonl` (ids: p_double, p_even, p_sum, p_reverse, p_max, p_swap,
m_sumdoc, c_stack — exactly the doc's 8), then run
`llama-dflash -m Qwen2.5-7B-Instruct-Q4_K_M --model-draft
fast-dllm-v2-1.5b-Q4_K_M -ngl 99 --in cases8.jsonl` in `--ar` mode and at
`--draft-len {32,16,8}`. Tool prints per-case tok/s + summary (rounds,
accepted/round, draft/verify share, efficiency). Output → `raw.log`.

## Result (re-run 2026-06-13 10:10, 5070 on AC)

| config | doc tok/s | re-run tok/s | doc acc/rd | re-run acc/rd | doc eff | re-run eff |
|---|---|---|---|---|---|---|
| AR baseline | 54.8 | **54.9** | — | — | — | — |
| dl 32 | 26.1 | 26.4 | 5.79 | **5.79** | 18.1% | **18.1%** |
| dl 16 | 40.2 | 40.3 | 5.69 | **5.69** | 35.6% | **35.6%** |
| dl 8  | 34.6 | 35.0 | 4.11 | **4.11** | 51.3% | **51.3%** |

accepted/round (5.79 / 5.69 / 4.11) and drafted-token efficiency
(18.1% / 35.6% / 51.3%) reproduce to the **exact digit**; tok/s within <1.5%.
Draft share dl-32 85.3% (doc "85% of wall").

**VERDICT: REPRODUCED (exact).** The closed-loop measurement that drove the F2
PARK is faithful in every column. The decision holds: best dflash (40.3 tok/s,
dl-16) < AR baseline (54.9 tok/s); off-the-shelf FastDLLM-1.5B drafting for
Qwen2.5-7B is net-negative, so F2 stays parked pending a trained drafter.
