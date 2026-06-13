# Static-code claims audit (no GPU)

Every claim in `07_layer_f.md` that asserts something about the *source code*
(line references, presence of a flag, a tool, an env var). These are verifiable
exactly and definitively by reading the tree at the HEAD the doc was written
against. Verdict band for this class: **EXACT** (line ref + semantics must both
match) or the claim is marked DRIFTED (right code, wrong line) / REFUTED.

- Repo HEAD at verification time: `53953f4b4`
- Method: `grep`/`sed`/`awk` over `src/`, `examples/diffusion/`, `kintsugi/`.
- Raw commands re-runnable via `./run.sh` in this dir.

---

## Claim S1 — catalog-F2 / Probe (F2 section): "encode() forces output_all=true (llama-context.cpp:1531)"

Doc text (07_layer_f.md:96): *"encode() forces output_all=true
(llama-context.cpp:1531) + n_outputs=n_tokens (:1566) = lm_head over every row."*

Evidence:
- `src/llama-context.cpp:1531`:
  ```
  if (!balloc->init(batch_inp, model.vocab, nullptr, n_embd, cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, true)) {
  ```
- The trailing `true` is the `output_all` parameter — confirmed from the
  signature at `src/llama-batch.h:78-84`:
  ```
  bool init(
          ...
          bool output_all);
  ```
  and its use at `src/llama-batch.cpp:121,131` (`if (output_all) ...`).
- `src/llama-context.cpp:1566`: `n_outputs = n_tokens;`

**VERDICT: REPRODUCED (EXACT).** Line 1531 is the `init(..., output_all=true)`
call; 1566 sets `n_outputs = n_tokens`. Both line numbers and semantics match.

## Claim S2 — catalog-F2: "needs_raw_logits already skips the D2H COPY under backend sampling (:1595)"

Doc text (07_layer_f.md:96-97).

Evidence — `src/llama-context.cpp:1594-1595`:
```
    // skip the copy when every output token is sampled on the backend
    if (logits.size > 0 && t_logits && needs_raw_logits(ubatch, sampling.samplers)) {
```
The guarded block is the device->host logits copy; it is skipped (the `if`
gates the copy) when `needs_raw_logits` is false, i.e. when every output token
is sampled on the backend. The analogous decode-path guard is at `:1908`.

**VERDICT: REPRODUCED (EXACT).** Line 1595, semantics as described (D2H copy is
conditional on `needs_raw_logits`).

## Claim S3 — "diffusion_generate already computes n_steps_done; plumb it out ... + steps_done in the response" (the bankable deliverable #1)

Doc text (07_layer_f.md:149-155, 237). The doc proposes exposing `steps_done`;
git log shows it was then implemented (commit "expose steps_done in /generate").
So at HEAD the deliverable is DONE, not pending.

Evidence:
- Computed: `examples/diffusion/diffusion.cpp:319` `int32_t n_steps_done = 0;`,
  incremented `:599`, floored `:1166`.
- Plumbed via out-param: `examples/diffusion/diffusion.h:122`
  `int32_t * out_steps_done = nullptr;  // [out, optional] denoising steps actually run (for harness`
  written at `diffusion.cpp:1167-1168` and `:1661-1662`.
- Surfaced in server response: `examples/diffusion/diffusion-server.cpp:215`
  (`int32_t steps_done = 0;`), `:237` (`dp.out_steps_done = &steps_done;`),
  `:350` (`res["steps_done"] = steps_done;`).

**VERDICT: REPRODUCED (EXACT) + status note.** The claim ("already computes
n_steps_done; plumb it out") is true AND the plumb-out was subsequently shipped
(`steps_done` is in the JSON response at HEAD). The GPU re-run of Probe 0 will
confirm the field is actually returned at runtime.

## Claim S4 — Probe 0: "env-gated request trace added to kintsugi Engine.post (KINTSUGI_TRACE=path -> one JSONL line per /generate)"

Doc text (07_layer_f.md:38-44).

Evidence — `kintsugi/lib/kintsugi/engine.ex:90-94`:
```
  # KINTSUGI_TRACE=<path> appends one JSONL line per /generate so cross-request
  ...
    case System.get_env("KINTSUGI_TRACE") do
```

**VERDICT: REPRODUCED (EXACT).** The env-gated trace exists in
`Kintsugi.Engine` keyed on `KINTSUGI_TRACE`. (Runtime behaviour — fields
infill?/prompt/n_gen/seed/ms_total/n_prompt_tokens/text — re-exercised live in
`../probe0-baseline-bench/`.)

## Claim S5 — Probe 4: "diffusion_generate builds+attaches+detaches a fresh backend sampler chain (llama_set_sampler), forcing a sched re-reserve"

Doc text (07_layer_f.md:127-129).

Evidence — per-request attach/detach in `examples/diffusion/diffusion.cpp`:
- Attach: `:263` and `:1252` `if (!llama_set_sampler(ctx, 0, backend_sampler)) {`
- Detach (back to nullptr): `:578`, `:1182`, `:1657` `llama_set_sampler(ctx, 0, nullptr);`
- Implementation: `src/llama-context.cpp:3772` `llama_set_sampler(...)` ->
  `ctx->set_sampler(:1153)`; the sched re-reserve is internal to attaching a new
  per-seq sampler.

**VERDICT: REPRODUCED (EXACT).** The attach-then-detach-per-request structure is
present exactly as described; this is the mechanism behind Probe 4's attach
overhead. (The *cost* of that overhead is the GPU claim, re-run in
`../probe4-tinyrepair-onoff/`.)

## Claim S6 — Probe 7: "the diffusion-batch-probe.cpp header already warned this (bug #1 there)" — llama_decode is async

Doc text (07_layer_f.md:205-207): the lesson that any host timer containing the
first post-decode sync measures GPU-forward wait.

Evidence — `examples/diffusion/diffusion-batch-probe.cpp:60`:
```
    llama_synchronize(ctx);  // llama_decode is async; sync before stopping the clock
```

**VERDICT: REPRODUCED.** The async-decode warning is in the batch-probe source
exactly as cited (the comment is the "bug #1" lesson the doc refers to).

## Claim S7 — E-rig / F1: "each replica is its OWN llama_model instance (mmap-shared weights, separate pkv store/phase state)"

Doc text (07_layer_f.md:316-319, 332-334). This underpins the F1 multi-replica
de-risk. Verified at the server-construction level in
`../methodology-audit/` (where the replica/model wiring is read); recorded here
as a static claim, GPU-confirmed by the F1 racing re-run in
`../f1-multireplica-racing/`.

---

## Summary

| claim | doc line | verdict |
|-------|----------|---------|
| S1 output_all=true @1531 + n_outputs=n_tokens @1566 | :96 | REPRODUCED exact |
| S2 needs_raw_logits D2H skip @1595 | :96-97 | REPRODUCED exact |
| S3 steps_done computed + plumbed to response | :149-155,237 | REPRODUCED exact (+shipped) |
| S4 KINTSUGI_TRACE env trace in Engine.post | :38-44 | REPRODUCED exact |
| S5 per-request llama_set_sampler attach/detach | :127-129 | REPRODUCED exact |
| S6 batch-probe async-decode warning (bug #1) | :205-207 | REPRODUCED |
| S7 replica = own llama_model instance | :316-334 | static OK; GPU in f1 dir |
