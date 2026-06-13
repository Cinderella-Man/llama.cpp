# Build spec: batched variant-sweep infill (scoped follow-up)

Goal: capture the measured tiny-canvas batching amortization (32 tok 1.96× free)
by decoding the 3 hole-size variants `{n, n+2, 1.4n}` in ONE batched infill
instead of sequentially. Projected +6-12% deliverable, lossless (35/48 holds).

## Why this is more tractable than the doc's F1 estimate
Dream INFILL runs the **square (no-cache) path** — kv is OFF for infill (Layer A
sec 16; 01_layer_a.md:280,464). So there is **no per-seq pkv/block state** to
replicate — the hard part of the F1 multi-seq design (per-seq pkv stores) does
NOT apply here. And `llama-diffusion-batch-probe` already proves multi-seq square
decode is **bit-exact** (Layer D finding 1) and reuses one weight-stream
(`forward_batched`). The build is: lift that multi-seq forward into a per-seq
commit loop + a server endpoint + a harness call. Bounded, not open-ended.

## Engine
New function (do NOT touch the single-canvas path):
`diffusion_generate_infill_batch(ctx, canvases[N], opts, out[N])` in
`examples/diffusion/diffusion.cpp`.
- Build one batch of the N canvases under N seq_ids (as `forward_batched` does in
  diffusion-batch-probe.cpp:40).
- Per step: one `llama_decode`; sample per-seq (backend sampling, multi-row — the
  E4 machinery already does multi-row); commit per-seq using the EXISTING
  threshold/confidence/Prophet/degeneracy logic, indexed per canvas.
- Per-seq completion: a canvas is done when all its masks are committed OR its
  degeneracy guard fires. **Drop finished canvases from the batch** and rebuild
  for the next step — this is what bounds the step-count-divergence penalty
  (variants with fewer masks finish first and stop consuming rows).
- Return each canvas's filled text + per-canvas ms (or one shared wall).

Per-seq state to carry (arrays of length N of the existing scalars/buffers):
`output_tokens`, mask/commit bookkeeping, EOT-shrink frontier, degeneracy
counters, confidence buffers. No kv state (square path).

## Server
`diffusion-server.cpp`: new endpoint `POST /infill_batch`
`{ "canvases": [str,...], <infill opts> } -> { "results": [{text, ms_total, steps_done, degenerate}, ...] }`.
Reuse the request-param plumbing; route to `diffusion_generate_infill_batch`.

## Harness
`kintsugi.ex try_hole_variants`: add a batched path.
- If ALL variants' canvas ≤ GATE (default 56 tok — above it batching loses, per
  the amortization table) AND `batch_sweep` opt on: build the N variant canvases,
  call `Engine.infill_batch(eng, canvases, opts)`, then compile-check results in
  variant order (n, n+2, 1.4n), return first compiling (else best_effort).
- Else: existing sequential short-circuit (unchanged).
`Engine.infill_batch/3` posts to `/infill_batch`.
Bench profile: add `"batchsweep" => %{"batch_sweep" => true}` to bench.exs.

## A/B measurement (the GO/KILL)
- Baseline: 35/48, 6.46 tok/s (probe0-baseline-bench, already measured).
- Treatment: `mix run bench/bench.exs URL batchsweep batchsweep` (same server,
  batched sweep on).
- **GO**: pass count == 35 (±0 expected — bit-exact; ±2 is the noise band) AND
  deliverable up (target +6-12%, real win if ≥ +5%). Log infill wall before/after.
- **KILL**: pass count drops >2 (batching/commit bug — bit-exactness violated) OR
  deliverable flat/down (step-count divergence ate the amortization; the
  drop-finished-canvases mitigation failed).

## Validation gates (correctness-first)
1. Bit-exactness: a batched 3-variant infill must produce byte-identical fills to
   the 3 sequential infills (assert in a unit probe before the bench). If not,
   the per-seq commit indexing is wrong.
2. Re-run `llama-diffusion-batch-probe` IDENTITY section to confirm multi-seq
   isolation still holds on this build.

## Effort & risk
~1-2 sessions. Main risk: per-seq commit/shrink bookkeeping bugs (caught by the
bit-exactness gate). No new numerics (lossless), so quality cannot silently
regress — the pass count is a hard tripwire.
