# F2 STEP 1 — oracle acceptance probe (re-verification)

**Doc claim** (07_layer_f.md:448-480): `llama-oracle-probe`, 32 FastDLLM-1.5B
drafts (8 cases × greedy + 3 t0.2 seeds) verified by Qwen2.5-7B-Instruct Q4_K_M
greedy: **70 blocks, mean L 12.01, median L 8, mean first-block L 12.25, overall
token agreement 84.9%, blocks L≥8: 38/70, fully accepted: 12/70.** Per-case
first-block L: double 29, even 5, sum 14, reverse 8, max 13, swap 6, sumdoc 12,
stack 11. GO bar met (median 8, first-block 12.25).

## How verified
`run.sh`: `llama-oracle-probe -m Qwen2.5-7B-Instruct-Q4_K_M -ngl 99 -ub 1024 --in
f2_drafts.jsonl` over the original 32-draft file (`/tmp/f2_drafts.jsonl`, copied
into this dir). Tool re-tokenizes each draft under Qwen's chat template, one
causal decode over [prompt|draft] with all-row logits, reports per-32-token-block
L = consecutive argmax matches. Output → `raw.log`.

## Result (re-run 2026-06-13 10:11, 5070 on AC)

| metric | doc | re-run | verdict |
|---|---|---|---|
| blocks | 70 | 70 | exact |
| mean L | 12.01 | **12.01** | exact |
| median L | 8 | **8** | exact |
| mean first-block L | 12.25 | **12.25** | exact |
| overall agreement | 84.9% | **84.9%** | exact |
| blocks L≥8 | 38/70 | **38/70** | exact |
| fully accepted (L=32) | 12/70 | **12/70** | exact |

Per-case first-block L (re-run): double 29, even 5, sum 14, reverse 8, max 13,
swap 6, sumdoc 12, stack 11 — **all 8 match the doc exactly**. The two L=32
repetitive-draft blocks (p_sum-t02-s203, p_reverse-g-s3) reproduce (32 32 32…),
confirming the doc's note that the 12 fully-accepted blocks come from repetitive
drafts and inflate the mean over the honest median of 8.

**VERDICT: REPRODUCED (exact).** Every oracle statistic and all 8 per-case
first-block L values match to the digit. The GO bar (median 8 / first-block
12.25) is confirmed — and, as the doc warned and STEP 2 (dflash) then showed,
the oracle was optimistic vs the 5.7/round closed-loop reality.
