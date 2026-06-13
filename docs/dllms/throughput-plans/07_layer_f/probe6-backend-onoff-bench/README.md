# Probe 6 — global backend-sampling A/B (re-verification)

**Doc claim** (07_layer_f.md:157-166): full bench, server `--no-backend-sampling`
vs default. backend ON = 35/48, 6.46 tok/s, 147 s (= Probe 0). backend OFF =
**33/48, 4.34 tok/s, 186 s** (−2 pass, −33% deliverable, +27% wall). Conclusion:
backend ON is globally correct.

## How verified
- ON arm = Probe 0 re-run (see `../probe0-baseline-bench/`).
- OFF arm = `run.sh` here: same server launch + `--no-backend-sampling`, baseline
  profile, full 48-case bench. (`--no-backend-sampling` confirmed in
  `llama-diffusion-server --help`: `-bs, --backend-sampling, --no-backend-sampling`.)
- stdout → `raw.log`; bench rows →
  `kintsugi/bench/results/20260613T...-flayer-cpusample-reverify.jsonl`.

## Result (re-run 2026-06-13 10:00, 5070 on AC)

| arm | doc | re-run | verdict |
|---|---|---|---|
| backend ON | 35/48, 6.46 tok/s, 147 s | 35/48, 6.46, 147 s | REPRODUCED (Probe 0) |
| backend OFF | 33/48, 4.34 tok/s, 186 s | **33/48, 4.33, 186.7 s** | REPRODUCED |
| Δ pass | −2 | −2 (35→33) | exact |
| Δ deliverable | −33% | −33% (6.46→4.33) | exact |
| Δ wall | +27% | +27% (147→186.7 s) | exact |

OFF per-tier: p 17/18, m 1/3, c 0/9, h 9/9, a 0/3, i 6/6 (809 tok / 186670 ms).

**VERDICT: REPRODUCED.** Backend sampling OFF costs exactly −2 passes, −33%
deliverable, +27% wall as claimed; backend-ON-is-globally-correct holds.
