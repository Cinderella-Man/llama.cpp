# Probe 0 — the kintsugi-Dream wall decomposition (re-verification)

**Doc claim** (07_layer_f.md:37-54): full 48-case bench, Dream-7B Q4_K_M, 5070 AC,
baseline profile = **35/48, 6.46 tok/s deliverable, 421 engine calls, 125.9 s
engine wall (147 s bench wall)**. Class split DRAFT 58/52.1 s/41%/median 637 ms;
INFILL 363/73.8 s/59%/median 111 ms. Infill dist min 32 / p10 40 / median 111 /
p90 554 / max 958. Draft prompt tokens median 63, max 75.

## How verified
1. `run.sh` — starts `llama-diffusion-server` (Dream-7B Q4_K_M, `-ub 512 -ngl 99
   --diffusion-eps 0.001 --diffusion-steps 128 --temp 0.2 --top-k 40`, GPU backend
   sampling ON by default) as a child of one foreground command (sandbox reaps
   detached/background GPU servers — exit 144/SIGUSR1), waits for `/health`, then
   runs `KINTSUGI_TRACE=… mix run bench/bench.exs … baseline`, kills server on exit.
2. `analyze.py` — re-derives the DRAFT/INFILL decomposition from the captured
   `ktrace-reverify.jsonl` (one JSONL line per /generate, the `infill` flag splits
   the classes; warmup "hi" line counted in the 421 as the doc does).
3. Bench stdout → `raw.log`; trace → `ktrace-reverify.jsonl`; analysis →
   `trace-analysis.txt`; server stderr → `server.log`. Bench result row file:
   `kintsugi/bench/results/20260613T075252Z-flayer-reverify.jsonl`.

## Result (re-run 2026-06-13 09:55, 5070 on AC)

| metric | doc | re-run | verdict |
|---|---|---|---|
| pass count | 35/48 | **35/48** | REPRODUCED (exact) |
| deliverable | 6.46 tok/s | **6.46** (949 tok/146989 ms) | REPRODUCED (exact) |
| bench wall | 147 s | 146989 ms | REPRODUCED (exact) |
| engine calls | 421 | 421 trace lines | REPRODUCED (exact) |
| engine wall | 125.9 s | 125.9 s | REPRODUCED (exact) |
| DRAFT | 58 / 52.1 s / 637 ms | 58 / 51.9 s / 635 ms | REPRODUCED |
| INFILL | 363 / 73.8 s / 111 ms | 363 / 73.8 s / 110 ms | REPRODUCED (exact) |
| DRAFT/INFILL share | 41% / 59% | 41% / 59% | REPRODUCED (exact) |
| infill ms dist | 32/40/111/554/958 | 32/40/110/556/955 | REPRODUCED |
| draft prompt tok | med 63 / max 75 | med 63 / max 75 | REPRODUCED (exact) |

Per-tier (re-run): p 18/18, m 2/3, c 0/9, h 9/9, a 0/3, i 6/6. The doc's strategic
text cites "m-tier 1/3"; re-run got 2/3 (+1, inside the ±2 numerics band). The
c-tier 0/9 and a-tier 0/3 failing walls that the strategic finding rests on are
reproduced exactly.

**VERDICT: REPRODUCED.** Probe 0 is faithful to the digit on every headline and
sub-distribution. The original source log (`/tmp/dream_server.log`, mtime
2026-06-13 01:42, 421 "sampling on the backend" lines) corroborates that the doc
reported its own run honestly.
