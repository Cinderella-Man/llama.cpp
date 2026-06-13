# Layer G grammar/scaffold - experiment log

Running log of the "go ballistic" investigation into grammar-constrained and
scaffold-seeded generation for Dream Elixir codegen. Parent doc:
`../dllm-grammar-scaffold-research.md`. Every experiment here is reproducible: the probe
script, the raw output, and a findings md live together.

Rule: no claim without a number. Losing prototypes are kept and labelled, not deleted.

## Verified substrate (firsthand, 2026-06-13)

- GPU: RTX 5070 Laptop, 8 GB, idle. AC power confirmed.
- Models on disk: `Dream-v0-Instruct-7B-Q4_K_M` (shipped SUT, owns the 35/48 baseline),
  `DiffuCoder-7B-cpGRPO-Q4_K_M` (a model the eth-sri grammar paper measured gains on),
  Fast-dLLM-v2-1.5B, DiffusionGemma.
- Binaries built: `llama-diffusion-server`, `llama-diffusion-cli`, ...
- Server start (background): `llama-diffusion-server -m <dream> -ub 512 -ngl 99
  --diffusion-eps 0.001 --diffusion-steps 128 --temp 0.2 --top-k 40`. /health reports
  family "masked", mask_token_id 151666, mask_piece `<|mask|>`, n_ctx 131072.
- Dream GGUF carries FIM tokens (`<|fim_prefix|>`/`<|fim_suffix|>`/`<|fim_middle|>`) - a
  potential native scaffold/infill channel beyond raw mask-runs. (Note for P1.)
- Confirmed by reading source:
  - canvas init is all-`<mask>`, NOT random noise (`examples/diffusion/diffusion.cpp:186-188`).
  - main diffusion path uses the standard sampler chain (`diffusion.cpp:211-221`).
  - `Engine.infill` = `/generate` with `"infill": true`; only mask pieces regenerate, rest
    byte-identical. `opts` pass through verbatim.
  - Draft path (`kintsugi.ex:222-248`): `prompt = wrapper <> instruction` ->
    `Engine.generate` -> `extract_code |> Autofix.run |> normalize_draft |>
    align_module_name`. Default draft opts: steps 128, conf_threshold 0.6, temp 0.2,
    top_k 40, eps 0.001, n_gen 192.
  - The free deterministic layer is Credence (path dep `../../credence`): `Credence.Syntax`
    (cheap, regex) runs in repair round 0; full `Credence.fix` (117 rules, compiles
    internally) runs once on broken code.
- Bench = 16 cases x 3 seeds = 48 runs (`kintsugi/bench/cases.exs`); KPI = check-pass + tok/s
  (`bench/bench.exs`). Forge cases (model drafts): 6 p + 1 m + 3 c = 10.

Key observation from stored baselines: even trivial `p_double` passes only after
`drafts:1, repairs:2` - the raw draft does not pass first try. Whether those repairs fix
SYNTAX (grammar/scaffold can kill them) or SEMANTICS (cannot) is the whole question -> P0.

## Experiments

### P0 - raw-draft failure taxonomy (DONE) -> `p0-results.md`
Probe `p0_draft_taxonomy.exs`, 30 forge drafts. Result: only 3/30 pass first-shot;
**40% fail at PARSE**, 43% compile(semantic), 7% check. **Credence rescues 0** draft
failures (B==C==D). Parse failures decompose into skeleton (missing do/end),
balanced-delim/keyword, and degeneracy/typo. ~50% of all failures are semantic - a hard
ceiling no grammar/scaffold can move.

### P1 / P1b - scaffold-seeding (DONE) -> `p1-results.md`
Probes `p1_scaffold_seed.exs` (module shell), `p1b_signature_seed.exs` (+ signature).
- P1 module shell: **parse 12 -> 0** (syntax solved) but pass 3 -> 3 (failures moved to
  semantic). Confirms scaffold kills syntax; confirms syntax was NOT the bottleneck.
- P1b signature seed: **pass 3 -> 7** (doubled) by killing wrong-function-name; but
  reintroduces parse on multi-clause funcs and breaks p_max. The lever is SEMANTIC seeding
  (contract), not syntactic structure.
KEY VERDICT: a pure grammar (P4, large) is predicted PASS-NEUTRAL here; the cheap win is
signature-scaffold. Decisive next test = full-bench A/B (check-pass + tok/s with the
repair loop), pending a forge code change.

### P1c - scaffold full-bench A/B (DONE) -> `p1c-scaffold-bench.md`
Wired signature-scaffold into forge (opt-in), full bench. **REGRESSED: 35->26 pass,
5.29->2.83 tok/s, +55% wall. 9 regressions, 0 gains.** Single-clause skeleton fights
recursive funcs; repair loop already handles syntax; constraint triggers expensive
cascades. Diff saved `scaffold-forge.diff`, source reverted (kept-as-killed). First-shot
win did NOT survive the repair loop.

### P3 - offline grammar acceptor (DONE) -> `p3-results.md`
Hand-wrote `elixir-subset.gbnf`, validated via llama.cpp `test-gbnf-validator`. Grammar
accepts 14/14 references (0 false-reject), rejects 12/12 syntax-broken drafts (100%
coverage), accepts 9/13 semantic drafts (can't judge semantics; 4 rejected = subset too
strict). Confirms a grammar WOULD catch syntax - but P1/P1c show that doesn't help the KPI.

### Converging verdict (after P0,P1,P1c,P3)
Grammar/scaffold reliably eliminate syntax errors but DON'T improve pass rate (semantics
bind) and scaffold actively HURT speed (repair loop already covers syntax). P4
(grammar-constrained decode) predicted pass-neutral + speed-negative on Dream-Elixir.
Open: is this Dream-specific? DiffuCoder-7B (the paper's model) is on disk - cheap cross-check.

### Decision point 2 -> re-grilling on P4 scope: user chose "full send everything"

### P4b - grammar-constrained diffusion decode (BUILT) -> `p4b-results.md` (A/B running)
Implemented a tractable frontier realization in the engine: `examples/diffusion/diffusion.cpp`
+ `diffusion.h` + `diffusion-server.cpp` (diff `p4b-engine.diff`). Each step builds a fresh
grammar sampler, accepts the committed prefix [0, frontier), masks the frontier masked
position to grammar-valid tokens. Plumbed via a `grammar` request param. Built clean.
Sanity findings (decisive already):
- Grammar is NOT wired into the GPU backend sampler - only the CPU sampling path. Forcing
  `backend_sampling:false` is REQUIRED for it to engage (production default is backend/GPU).
- It engages (changes output) but costs ~16x latency on a micro-test (493ms -> 8158ms):
  per-step grammar rebuild + full-vocab (151k) grammar masking. Plus losing the GPU
  backend sampler is itself a big slowdown.
- Only the frontier position is constrained (out-of-order commits beyond it are free) -
  the tractable limit vs full eth-sri any-order CFG-intersection.
Full A/B (pass/parse/compile + latency) measured in p4b-results.md.

P4b A/B RESULT (latched/fair): grammar ON pass 7->4, parse not reduced, **~21x slower**.
Un-latched: engaged more (parse 9->5) but errored 15/30 (subset-strictness) and pass->1.
Either way a decisive loss. -> `p4b-results.md`.

### FINAL VERDICT -> `VERDICT.md`
No: grammar/scaffold reliably kill syntax errors but syntax is not the bottleneck on this
stack. scaffold draft = -9 pass/-47% tok/s; grammar decode = -3 pass/21x slower. Bottleneck
is SEMANTIC; the repair loop already handles syntax cheaply. Catalog G9/G11/G1 corrections
noted. One open branch: reproduce the paper's setting on DiffuCoder-7B (on disk).

### Cleanup
Engine P4b changes reverted (kept as `p4b-engine.diff`); scaffold reverted (`scaffold-forge.diff`).
Production source clean. All probes/results/diffs preserved in this directory.

### DiffuCoder cross-check (open branch CLOSED) -> `dc-results.md`
Re-ran P0/P1/P1b on DiffuCoder-7B (`dc_crosscheck.exs`, `dc-samples.jsonl`). Verdict
GENERALIZES: scaffold parse 27->0 but pass raw 3 -> scaffold 0 -> sig 3 (all module-scaffold
-> `check`). Raw parse-fails are real (do-fusion, stray `}`), not extraction artifacts.
DiffuCoder benefits even less than Dream. Across both diffusion models on Elixir: grammar/
scaffold remove syntax, don't move the KPI. The eth-sri +7% (C++/JSON) setting not
reproduced (out of scope).

### G13 - check-first prompting (DONE) -> `g13-results.md`
Put the check in the draft prompt (raw + natural-language variants), full-bench A/B on
Dream. **Both REGRESS: baseline 35/48 -> raw 32/48 (3.81 tok/s) -> NL 28/48 (2.66 tok/s),
0 cases fixed.** Capture probe (`g13_capture.exs`): the test doesn't grant capability
(c-tier 0/9 unchanged), perturbs drafts into new syntax errors (repairs 50->59->76), and
invites example-overfitting (`number*number` passes double(2)=4). Diff `g13-checkfirst.diff`,
reverted.

### META-FINDING (campaign conclusion)
EVERY draft-side intervention (scaffold P1c, grammar P4b, check-first G13) regresses, on both
Dream and DiffuCoder. Baseline draft+repair is already at the model's capability ceiling on
this bench. Headroom = MODEL CAPABILITY + repair loop, NOT draft prompting/constraint.

### Robustness re-grill (DONE) -> `robustness-analysis.md`
Adversarially re-examined the verdict: bench HAS dynamic range (8/10 forge cases flip;
p_sum 97/93) so negatives are real; n=3 seeds = low power so check-first -3 is within noise
(softened) while scaffold -9 and grammar 21x are robust; capability floor confirmed
(c_shout/c_stack 0/189 across both models); best-of-N already shipped (max_drafts=3).

### Paper-setting reproduction: JSON (DONE) -> `json-reproduction-results.md`
Re-applied + extended grammar to non-infill; tested on DiffuCoder. JSON HAS syntactic
headroom (raw-primed baseline ~50% valid; failures = `true/false`, stray punct, truncation)
- UNLIKE Elixir's semantic ceiling. So the Elixir negative is DOMAIN-specific, not "grammar
never helps." But the tractable frontier-grammar can't enforce under out-of-order commits
(A/B 0/30, corrupts); reproducing the paper's +7% needs the FULL eth-sri algorithm (large),
which does NOT help the Elixir goal. Engine re-reverted; full diff in `p4b-engine.diff`.

### Final server state
DiffuCoder server running on :8080 (pid 826650, grammar-capable binary, grammar off by
default). Source clean. kintsugi production default model is Dream - restart Dream
(`./build/bin/llama-diffusion-server -m ~/models/dream7b/...`) for production. Stop the
server with `kill -9 826650`.
