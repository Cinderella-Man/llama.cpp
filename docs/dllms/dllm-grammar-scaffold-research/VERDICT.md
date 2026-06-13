# VERDICT: would grammar / scaffold-seeding help Dream Elixir codegen?

Empirical answer from six experiments (P0-P4b), Dream-7B Q4_K_M on the kintsugi bench,
RTX 5070, 2026-06-13. All probes, raw samples, diffs, and per-experiment write-ups are in
this directory. Bar = check-pass rate (X/48) + tok/s (the existing KPI).

## Bottom line

**No - not on this stack.** Grammar and scaffold-seeding reliably eliminate SYNTAX errors,
but syntax is not the bottleneck, and forcing structure/grammar HURTS the KPI:

| approach (vs baseline 35/48, 5.29 tok/s)        | result                              |
|-------------------------------------------------|-------------------------------------|
| scaffold-seed draft (full bench, P1c)           | **26/48, 2.83 tok/s** (-9, -47%)    |
| grammar-constrained decode (infill A/B, P4b)    | **pass 7->4, ~21x slower**          |

The user's instinct ("we know the shape; constrain it") is mechanically correct but
strategically wrong here: the shape was never the thing the model gets wrong that matters.

## The evidence chain (each step earns the conclusion)

| # | experiment | finding |
|---|------------|---------|
| P0 | draft failure taxonomy | 40% of drafts fail at PARSE, 43% compile/semantic, 7% logic. **Credence rescues 0** of them. ~50% of all failures are semantic - a hard ceiling. |
| P1 | scaffold (module shell), first-shot | parse 12->0 (syntax solved) but pass 3->3. **Removing 100% of syntax errors did not create one new pass** - failures just moved parse->semantic. |
| P1b | scaffold + signature, first-shot | pass 3->7. The gain came from seeding the function CONTRACT (semantic), not structure. |
| P1c | scaffold as draft, FULL bench | **35->26 pass, 5.29->2.83 tok/s.** First-shot win inverted: the repair loop already fixes syntax cheaply, and a rigid skeleton fights recursive/multi-clause solutions + triggers expensive cascades. |
| P3 | offline GBNF acceptor | grammar catches 100% of syntax errors and 0 of semantics; 0 false-reject on references BUT rejects 4/13 parse-valid drafts (subset too narrow). |
| P4b | grammar-constrained decode (built in engine) | pass 7->4, ~21x slower; not wired into GPU sampler; subset-strictness crashes/latches it off. Decisive loss. |

## Why syntax constraints can't win here

1. **Semantics bind, not syntax.** P1 is the proof: eliminate every parse error and the
   pass rate doesn't move, because the remaining failures are wrong logic / undefined
   symbols / wrong API (e.g. `String.count`, hallucinated `:lang.to_list`). No CFG can fix
   those.
2. **The repair loop already neutralizes syntax for free-ish.** Baseline reaches 35/48
   from a 3/30 first-shot purely via masked-infill repairs. Syntax errors are the CHEAP
   ones to repair. So preventing them at decode time removes little cost while adding
   constraint damage.
3. **Constraint removes useful model freedom.** Both scaffold (P1c) and grammar (P4b) push
   the model into worse solution basins; recursive/multi-clause Elixir is especially hurt
   by a single-clause skeleton, and a narrow grammar forces odd tokens.
4. **Elixir is genuinely hard to constrain.** A subset GBNF is too strict (drift ->
   crash/latch-off, P3/P4b); a full CFG is impractical (macros, sigils, heredocs,
   interpolation are partly context-sensitive). The grammar engine is also strictly
   left-to-right (incompatible with diffusion's out-of-order commits without the heavy
   eth-sri intersection) and not wired into the GPU sampler.

## On the specific sub-questions asked

- **"Don't start from random noise."** Dream starts from all-`<mask>`, not noise; seeding
  known tokens = infill, already supported. Tested thoroughly (P1/P1b/P1c): it works for
  syntax but loses on the KPI.
- **"Static text length between two snippets."** This is multi-region infilling; the engine
  already handles fixed text + holes byte-identically. It was never the blocker - the hole
  SIZE matters (sweep already exists), but the binding issue is semantic content, not the
  fixed-length frame.
- **"Credence side vs llama.cpp side?"** Both were tried. Credence/kintsugi side =
  scaffold-seed (P1c, lost). llama.cpp side = grammar-in-decode (P4b, lost worse).

## Catalog corrections (please fix `dllm-throughput-catalog.md`)

- **G9 "canvas seeding ... high-value quick win"** is REFUTED: measured -9 pass, -47% tok/s
  (P1c). Not a quick win; a regression.
- **G11 "grammar ... possibly novel"** is WRONG on novelty: published (Muendler et al. 2025,
  arXiv:2508.10111, eth-sri/constrained-diffusion). And measured here as pass-negative + 21x
  slower (P4b).
- **G1 "Credence converts syntax repairs to free"** holds only for crafted heal cases;
  Credence rescues 0 real draft parse failures (P0).

## What the data says to do instead (UPDATED - G13 now tested, also a loss)

The bottleneck is the model's SEMANTIC capability, and EVERY draft-side intervention tried
regresses:
- **Check-first prompting (G13): TESTED, NO IMPROVEMENT** (35 -> 32 raw, -> 28 NL; 0 cases
  fixed; see `g13-results.md`). NB `robustness-analysis.md`: at n=3 the raw -3 is within
  seed noise (p~0.13); the NL -7 is a real loss (p~0.008). Showing the test doesn't grant
  capability (c-tier ceiling), perturbs drafts into new syntax errors, invites overfitting.
- So do NOT keep optimizing the draft. The remaining headroom is:
  1. **Model capability** - a stronger / larger / Elixir-tuned model (the c-tier ceiling is
     a 7B Elixir limit, hit by Dream AND DiffuCoder).
  2. **The repair loop / verification** - already the workhorse that carries 3/30 first-shot
     to 35/48; invest there, not in draft tricks.
  3. (Minor) degeneracy control in sampling - some failures were token repetition
     (`defmodulelerler`), a quality knob, not a prompt/constraint.

## Cross-check (open branch, now CLOSED) -> `dc-results.md`

Re-ran P0/P1/P1b on **DiffuCoder-7B-cpGRPO** (the code model the paper measured +7% on),
same bench/params. The verdict GENERALIZES, more strongly: scaffold eliminates 100% of
parse errors (27/30 -> 0) but pass rate does not improve (raw 3 -> scaffold 0 -> sig 3);
all 30 module-scaffold drafts become `check` (compile-but-wrong). Signature scaffold helps
DiffuCoder even less than Dream (3 -> 3 vs 3 -> 7). So across BOTH diffusion models on this
Elixir bench, syntactic constraint removes syntax errors and does not move the KPI.

## Paper-setting reproduction (JSON) -> `json-reproduction-results.md`

Tested grammar on JSON (a domain DiffuCoder IS competent at). Key contrast: with raw
priming, DiffuCoder emits valid JSON ~50%, and the failures are SYNTACTIC (`true/false`
literal, stray punctuation, truncation) - exactly what a grammar prevents. So JSON HAS the
headroom Elixir lacks (Elixir fails for SEMANTIC reasons no CFG can fix). This confirms the
Elixir negative is DOMAIN/CAPABILITY-specific, not "grammar never helps."

But we did NOT get a positive grammar number: the tractable frontier-grammar can't enforce
structure under diffusion's out-of-order commits (it forced `{` then prose escaped ->
corrupt; A/B 0/30). Realizing the paper's +7% needs the FULL eth-sri any-order
CFG-intersection (or forced left-to-right AR) - a large port, and one that does NOT help the
Elixir goal (Elixir has no syntactic headroom to capture).

For **Elixir on 7B diffusion models, the answer is settled: grammar/scaffold do not help.**
The mechanism is real in principle (JSON), just not applicable to this capability-bound stack.
