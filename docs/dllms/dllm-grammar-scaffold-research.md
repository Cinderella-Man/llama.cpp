# Layer G: grammar + scaffold-seeding for Dream Elixir codegen (research + plan)

Status: PROTOTYPED + MEASURED (2026-06-13). Empirical verdict and full experiment trail
(probes, raw samples, diffs, per-experiment write-ups) live in
`dllm-grammar-scaffold-research/` - start with `VERDICT.md`, then `00-session-log.md`.
TL;DR: NO on this stack - scaffold draft = -9 pass / -47% tok/s (P1c); grammar decode =
-3 pass / ~21x slower (P4b). Syntax is not the bottleneck; the repair loop already handles
it; the bottleneck is semantic. This doc below is the original research/plan (prior art +
engine reality + ladder) that the experiments then tested.

## Question

Would a grammar (GBNF-style) and/or NOT starting from an all-mask canvas (pre-seeding
`defmodule Solution do ... end`) help the Dream diffusion engine produce valid Elixir?
We know the problem space: it is Elixir, it must parse and compile, the shape is known
(first line `defmodule ... do`, last line `end`, function shells). Two candidate loci:
(a) kintsugi/Credence side (seed/repair as it reads the snippet) or (b) directly in
llama.cpp (define a starting point / constrain decode).

## TL;DR

- The grammar idea is NOT novel - it is published, on this exact model class, with a
  reference implementation. Catalog G11's "possibly novel" label is wrong; correct it.
- "Don't start from random noise" is a misread of the mechanism: Dream starts from an
  all-`<|mask|>` canvas, not random noise. Seeding the module shell = infill, which the
  engine ALREADY supports. This is catalog G9 ("high-value quick win").
- A grammar guarantees PARSE-validity, not COMPILATION. It cannot fix Elixir semantic
  errors (undefined vars/functions). Realistically a grammar == "Credence's Syntax phase
  enforced at decode time" - it prevents the quirk instead of fixing it post-hoc.

## Prior art (established, with citations)

- **Muendler et al. 2025, "Constrained Decoding of Diffusion LLMs with Context-Free
  Grammars"** (arXiv:2508.10111; repo `eth-sri/constrained-diffusion`, MIT). First
  generalized constrained-decoding method for multi-region-infilling / out-of-order
  generation models. Core idea: reduce constrained decoding to the *additive infilling*
  problem - can a partial output still be completed to a valid word in the target
  language? - which reduces to deciding whether `CFG INTERSECT (regular language of the
  committed context)` is non-empty. Accept a token at a masked slot only if a valid
  completion still exists. Guarantees syntactic correctness w.r.t. the grammar. Result:
  **+up to 7% functional correctness with minimal overhead**. Tested on LLaDA-8B,
  DiffuCoder-7B, Dream-Coder, StarCoder-FIM, DeepSeek-Coder, CodeGemma; tasks C++
  (HumanEval multi-region infilling), JSON schema, SMILES. Explicitly SUBSUMES
  multi-region infilling = the "static text length between two snippets" concern.
- **Lookahead-then-Verify (LAVE)** (arXiv:2602.00612). Identifies the key pitfall: naive
  any-order grammar commits can reach states impossible to complete (grammatical
  dead-ends). LAVE uses the model's parallel per-position distributions to verify a
  proposed token can extend to a valid completion. Negligible overhead; 4 dLLMs, 3
  benchmarks.
- **Beyond Autoregression** (arXiv:2509.11252). Canvas length is the dominant
  hyperparameter: usability rises with length then declines; optimum is model/task
  specific (DiffuCoder best at 128 on HumanEval; LLaDA-8B best at 1024). In-place
  prompting (prefix + suffix + masked middle) is flagged as promising but UNMEASURED. No
  syntax-vs-semantic failure breakdown. Confirms the length worry is real.

## Engine reality (this fork)

- Main diffusion loop `examples/diffusion/diffusion.cpp` (`diffusion_generate`,
  ~160-1187) initializes canvas as prompt + `std::fill(... mask_token_id)` (186-188):
  all-MASK, NOT random noise. Mask id from `llama_vocab_mask`.
- Sampling on the main path already uses the standard sampler chain
  (`llama_sampler_apply`, ~745-787) -> a grammar sampler COULD hook there. The block-AR
  path (`diffusion_generate_block_ar`, 1194-1667) uses CUSTOM sampling (`predict`,
  ~1321-1448) -> separate hook.
- BUT `src/llama-grammar.cpp` is strictly left-to-right and destructive: `accept_token`
  advances stacks codepoint-by-codepoint in order and overwrites state; no
  arbitrary-position query, no resumable/forkable parse. So it is incompatible with
  any-order commits without per-position state - the exact hard problem the papers solve.
- **Infill already supported**: canvas = fixed text + `<|mask|>` runs, only masks
  regenerated (`kintsugi/lib/kintsugi.ex:349-366`, `Masker.mask_lines`, `Engine.infill`).
  Scaffold seeding needs ZERO engine change.
- **logit_bias backend sampler already exists** (catalog G10/H4) -> single-token bans are
  free; sequence-level bans need care.
- Verifier (`kintsugi/lib/kintsugi/verifier.ex`) does parse (`Code.string_to_quoted`),
  then compile (`Code.compile_string`), then isolated-process run. Bench
  (`kintsugi/bench/bench.exs`) = 24 cases x ~3 seeds, currently 35/48.
- Catalog `docs/dllms/dllm-throughput-catalog.md:246-292` already lists G9 (scaffold),
  G10 (logit bans), G11 (grammar), H4 (code-subset logit mask).

## Honest limits

Grammar guarantees PARSE-validity, not COMPILATION. Elixir surface grammar is small
(`def`/`if`/`defmodule` are macros, not syntax) but has context-sensitive parts (sigils,
heredocs, string interpolation) that are not pure CFG even in tree-sitter-elixir (custom C
scanner). Semantic errors - undefined vars/functions, the bench a-tier - are uncatchable
by any CFG. Therefore: bar = check-pass; compile-rate is a diagnostic, not the bar.

## Prototype ladder (full, committed; run cheap -> expensive, each informs the next)

Bar = check-pass rate (X/48) + tok/s aggregate (existing KPI). Compile-rate (parse+compile)
= secondary diagnostic that explains why the KPI moved. Record every number in the catalog
(H6 discipline). Losers recorded as killed, not deleted.

- **P0 - Failure taxonomy (offline).** Classify every bench/production failure: parse vs
  compile/semantic vs check(logic). Run `Code.string_to_quoted` over logged RAW drafts ->
  % parse-invalid before repair. Sizes the prize; if syntax is a tiny slice, grammar is
  low-value. Reuse Verifier's parse-vs-compile split.

- **P1 - G9 scaffold seeding via existing infill (no engine change).** Forge builds canvas
  `defmodule <Name> do\n  <MASK*n>\nend` via `Masker` + `Engine.infill`; reuse the existing
  hole-size sweep `{n, n+2, 1.4n}`. Two variants: (1a) prefix-only (no `end` anchor) and
  (1b) full bracket (prefix + `end` suffix anchor - tests the fixed-interior length case).

- **P2 - G10 logit-bias quirk bans (existing backend sampler).** Ban single tokens behind
  the top Credence Syntax rules (e.g. `, do` fusion); measure quirk recurrence, pass rate,
  repair-rounds saved.

- **P3 - Elixir GBNF subset + offline acceptor study.** Author a minimal Elixir GBNF
  subset (module / def / do-end / balanced delimiters; template off `grammars/c.gbnf`).
  Run it as a POST-HOC acceptor over logged drafts via the existing `llama_grammar_*`
  engine. Measure accept/reject vs Verifier parse result AND false-reject rate on
  KNOWN-GOOD passing solutions (a too-strict grammar rejecting valid Elixir is the real
  risk). De-risks P4 before building it.

- **P4 - CFG-constrained dLLM decode (engine change). Two increments:**
  - **P4a position-local filter (first):** balanced-delimiter counts + identifier/keyword
    char classes as a backend canvas filter masking obviously-invalid tokens per step. Hook
    the main path (~745-787) and the block-AR sampler (~1321-1448). Cheap; no
    CFG-intersection machinery.
  - **P4b full CFG (then):** port the eth-sri additive-infilling approach - accept a token
    at a masked slot only if `CFG INTERSECT committed-context` is non-empty - plus
    LAVE-style lookahead (reuse per-position parallel distributions) to avoid dead-ends.
    Reuse `llama_grammar_*` where possible; the any-order gap is the new code.
  - Compare P4a vs P4b vs P1/P2 on the same bench.

## Files (for execution)

- `kintsugi/lib/kintsugi.ex` - forge scaffold path (P1); reuse `Masker.mask_lines`,
  `Engine.infill`, hole sweep.
- `kintsugi/lib/kintsugi/engine.ex` - confirm/extend `logit_bias` param (P2).
- `kintsugi/bench/bench.exs` - add `scaffold` / `bans` profiles; reuse summarize / tok-s KPI.
- `grammars/elixir-subset.gbnf` (new) + small acceptor harness (P3).
- `examples/diffusion/diffusion.cpp` - P4 only (main + block-AR sampler hooks).
- Catalog update (G9/G11 corrections) + this doc.
- Reference: clone `eth-sri/constrained-diffusion` (MIT) into `~/projects/dllm-references`
  (matches existing reference-clones pattern) for P3/P4 study.

## Verification

Run `kintsugi/bench/bench.exs` before/after each prototype; bar = pass (X/48) + tok/s; also
log compile-rate. Temp-0 byte-identical checks where applicable. P0/P3 are offline
(parse/grammar acceptors over logged drafts) - no GPU regression risk.

## Decisions locked / defaults

- Full ladder P0 -> P4 committed; run cheap -> expensive.
- Bar = check-pass (X/48) + tok/s; compile-rate = secondary diagnostic.
- Default: clone the MIT reference repo; run baseline + P0-P4 on shipped Dream-7B for
  apples-to-apples with the 35/48 baseline. Optionally cross-check P4 on a code-tuned
  diffusion model (Dream-Coder / DiffuCoder) the papers measured.

## Open questions

1. P4 ordering: P4a (position-local) before P4b (full CFG), or straight to P4b? (P4a may
   capture most of the gain at a fraction of the cost.)
2. For P4b, accept per-candidate CFG-intersection cost on a 7B at interactive latency, or
   restrict to the P3 Elixir subset to keep overhead "minimal"?

## Sources

- arXiv:2508.10111 - Muendler et al., Constrained Decoding of Diffusion LLMs with CFGs
  (repo eth-sri/constrained-diffusion, MIT)
- arXiv:2602.00612 - Lookahead-then-Verify (LAVE)
- arXiv:2509.11252 - Beyond Autoregression: Diffusion LLMs for Code Generation
- llama.cpp `grammars/` + `src/llama-grammar.cpp` (GBNF engine);
  `elixir-lang/tree-sitter-elixir` (Elixir grammar, custom C scanner)
