# Methodology audit — do the probe tools do what the doc claims?

The doc's empirical claims rest on three custom tools + the kintsugi bench. This
dir audits that each tool's *code* implements the methodology the doc describes
(the runtime *numbers* are re-verified in the per-probe dirs).

## llama-oracle-probe (`examples/diffusion/oracle-probe.cpp`)
Doc (F2 STEP 1) says: verifier applies its own Qwen2.5 chat template to the
`user` content, one causal decode over [prompt|draft] with all-row logits, then
per 32-token block L = consecutive argmax matches from block start.
- Header (`oracle-probe.cpp:1-18`) states exactly this; reads `{"id","user",
  "draft"}` lines (`:139-155`), applies the template, decodes, counts L per block.
- **AUDIT PASS** — implements the stated oracle methodology. Re-run reproduced
  every statistic (see `../f2-oracle-probe/`).

## llama-dflash (`examples/diffusion/dflash.cpp`)
Doc (F2 STEP 2) says: both models one process; drafter block-AR over
[committed | draft_len masks]; verifier greedy walk + cache rollback per round;
`--ar` = pure AR baseline; output IS the verifier's greedy decode.
- `dflash.cpp:8-12` usage + `--ar`; `:33-44` draft_len/max_tokens; `:170-223`
  the draft → verify-argmax → accept-run → rollback loop; verifier ctx is causal
  KV (`:75-78`). Cross-model tokens via shared Qwen2.5 vocab (no retokenization).
- **AUDIT PASS** — implements the closed-loop draft/verify the doc describes.
  Re-run reproduced the tok/s + accepted/round + efficiency table exactly
  (see `../f2-dflash-closedloop/`).

## llama-diffusion-batch-probe (`examples/diffusion/diffusion-batch-probe.cpp`)
Referenced by Probe 7 for its "bug #1" async-decode warning.
- `:60` `llama_synchronize(ctx); // llama_decode is async; sync before stopping
  the clock` — the exact lesson Probe 7 re-learned. **AUDIT PASS.**

## kintsugi bench (`kintsugi/bench/bench.exs`)
Doc's Probe 0/6 driver.
- `:1-21` usage + honest-denominator note (walls measured around each call, "ALL
  runs" denominator); `:26-58` profiles incl. `baseline` (empty overlay), `e4bs`
  etc.; warmup request `:81`; battery guard `:76`.
- **AUDIT PASS** — the `baseline` profile is the unmodified default the doc used;
  the deliverable tok/s is `total_tokens / total_wall_ms` over all 48 cases.

## Conclusion
All four instruments implement the methodology their claims assume; none contains
a shortcut that would inflate the reported numbers. Combined with the per-probe
re-runs, the doc's measurement chain is sound.
