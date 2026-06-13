# Probe 7 — the "sampling time/step" is mis-attributed GPU-forward wait (re-verification)

**Doc claim** (07_layer_f.md:181-210): the diffusion loop's logged "sampling time
per step" (58-155 ms on a 256-canvas) LOOKED like host-side readback overhead.
Decisive test: env `DIFF_SYNC_AFTER_DECODE` adds one `llama_synchronize` right
after `llama_decode`, before the sampling timer. Result: **sampling time/step
154.94 ms → 0.06 ms, per-step total UNCHANGED (~160 ms)**. Root cause:
`llama_decode` is async; the first sync inside the sampling-timer region is where
the CPU waits for the GPU forward. True host sampling is ~0.06 ms/step ⇒ F7/F11
(host overlap / step-loop micro) DEAD on arrival. "Reverted; no API added."

## How verified (the change is REQUIRED — this probe needs a code edit + rebuild)
1. `probe7.diff` — reconstructed the env gate in `examples/diffusion/diffusion.cpp`:
   a `static bool diff_sync_after_decode = getenv("DIFF_SYNC_AFTER_DECODE")` that,
   when set, calls `llama_synchronize(ctx)` immediately before
   `int64_t time_start_sampling = ggml_time_us();` (the masked step loop). This is
   the exact mechanism the doc describes.
2. Rebuilt `llama-diffusion-server`, then ran `run.sh`: two server arms (env unset
   vs `DIFF_SYNC_AFTER_DECODE=1`), each one 256-canvas generation (thr 0.95, steps
   40), reading the `total time / time per step / sampling time per step` log line.
3. After measuring, **reverted** the source (`git checkout`) — matching the doc's
   "reverted; no API added". `probe7.diff` is retained as the reproduction recipe.

## Result (re-run 2026-06-13 10:04, 5070 on AC)

| arm | per-step total | sampling time/step | verdict |
|---|---|---|---|
| OFF (unset) | 55.08 ms | **37.32 ms** | baseline (mis-attributed) |
| ON (`DIFF_SYNC_AFTER_DECODE=1`) | 54.57 ms | **0.07 ms** | the collapse |

- sampling time/step: 37.32 → **0.07 ms** — doc says 154.94 → 0.06; the **collapse
  to ~0.06-0.07 ms is reproduced exactly**.
- per-step total: 55.08 → 54.57 ms — **UNCHANGED** (within 1%), as claimed.

The absolute *pre-sync* magnitude differs (37 ms here vs 155 ms in the doc):
my 256-canvas EOT-shrank to a smaller effective canvas, so the GPU-forward wait
mis-attributed as "sampling" was ~37 ms not ~155 ms. The MECHANISM and the
load-bearing result are identical: that pre-sync number is 100% GPU-forward wait
(it vanishes when the sync is moved before the timer; the total is unchanged),
and true host sampling work is ~0.06-0.07 ms/step.

**VERDICT: REPRODUCED.** The decisive finding — "sampling time" is mis-attributed
async GPU-forward wait, true host step work ≈ 0.06 ms — reproduces exactly. The
F7/F11-DEAD conclusion (no host work to overlap/shave) stands.
