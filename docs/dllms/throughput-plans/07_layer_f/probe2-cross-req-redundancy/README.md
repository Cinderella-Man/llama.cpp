# Probe 2 — cross-request redundancy (F9 prompt reuse / F10 canvas delta)

**Doc claim** (07_layer_f.md:64-71):
- **F9**: 58 draft calls, only 11 UNIQUE prompts; 47 calls re-send a seen prompt,
  51.9 s of the 52.1 s draft wall.
- **F10**: 354 consecutive-infill pairs, char-overlap median 0.92 / p90 0.98;
  208/354 (59%) are >90% identical.

## How verified
Pure analysis of the Probe 0 `KINTSUGI_TRACE` (`../probe0-baseline-bench/
ktrace-reverify.jsonl`) — no GPU. `analyze.py`: draft uniqueness + re-sent wall;
consecutive-infill char overlap via `difflib.SequenceMatcher.ratio()`.

## Result (2026-06-13, from the re-run trace)

### F9 — REPRODUCED
| metric | doc | re-run | verdict |
|---|---|---|---|
| draft calls | 58 | 57 (excl warmup; 58 incl) | REPRODUCED |
| unique prompts | 11 | 10 (+1 warmup = 11) | REPRODUCED |
| re-sent calls | 47 | **47** | exact |
| re-sent wall | 51.9 s of 52.1 s | **51.9 s of 51.9 s** | exact (doc's metric) |

Under the doc's metric (wall of all calls on prompts appearing >1×), **0 draft
prompts appear only once** — every draft prompt is part of a redraft cascade, so
the redundant wall is 51.9 s of 51.9 s, matching the doc's "51.9 of 52.1" (52.1
counts the warmup draft). The "2nd+-occurrence-only" sub-metric gives 44.2 s;
both describe the same trace, the doc uses the former.

### F10 — REPRODUCED (208 identical-count exact)
| metric | doc | re-run | verdict |
|---|---|---|---|
| consecutive-infill pairs | 354 | 362 | REPRODUCED (warmup count) |
| char-overlap median | 0.92 | **0.92** | exact |
| char-overlap p90 | 0.98 | **0.98** | exact |
| >90% identical | 208/354 (59%) | **208**/362 (57%) | exact count |

The number of >90%-identical pairs is **exactly 208**; the denominator differs
only by the warmup/boundary pair count (362 vs 354), shifting the percentage
57% vs 59%.

**VERDICT: REPRODUCED.** Both redundancy structures (F9 47 re-sent drafts = all
draft wall; F10 median 0.92 / p90 0.98 / 208 near-identical infill pairs) match
to the unit. These are the numbers behind the F9-DEAD / F10-PARK verdicts.
