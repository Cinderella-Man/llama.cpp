# P0 - raw-draft failure taxonomy (Dream-7B Q4_K_M)

Probe: `p0_draft_taxonomy.exs`. Raw drafts: `p0-samples.jsonl`. 30 forge drafts (10 forge
cases x 3 seeds), real draft config (steps 128, conf 0.6, temp 0.2, top_k 40, n_gen 192).
Run 2026-06-13.

## Headline

| stage | pass | parse-fail | compile-fail (semantic) | check-fail (logic) |
|-------|------|-----------|-------------------------|--------------------|
| B raw draft (extract+autofix+normalize+align) | 3 | **12 (40%)** | 13 (43%) | 2 (7%) |
| C + Credence.Syntax (free regex layer)         | 3 | 12 | 13 | 2 |
| D + full Credence.fix (117 rules)              | 3 | 12 | 13 | 2 |

Two load-bearing findings:

1. **Only 3/30 drafts pass first shot.** The whole 35/48 bench result is carried by the
   GPU repair loop, not the draft. Consistent with stored baselines (`p_double`:
   drafts 1, repairs 2).

2. **The free Credence layer rescues ZERO draft failures** (B == C == D, exactly). The
   catalog claim "Credence converts syntax repairs to free" is true ONLY for the crafted
   `h_*` heal cases (do-fusion on hand-written input). On real Dream draft output,
   Credence's syntax pipeline runs, fails to make the source parse, and bails
   ("source still does not parse" in the debug trace). So today every one of the 12 parse
   failures costs GPU repair rounds.

## What the 12 parse failures actually are (root cause, from p0-samples.jsonl)

| case (seeds)      | error                                            | root cause                         | addressable by                         |
|-------------------|--------------------------------------------------|------------------------------------|----------------------------------------|
| c_vowels (x3)     | `defmodule Vowels` with **no `do`**              | missing block opener (skeleton)    | **scaffold-seeding** (clean kill)      |
| c_stack (x1)      | `ends` instead of `end`                          | keyword near-miss typo             | logit-ban / keyword grammar            |
| c_shout (x3)      | `do: String.upcase) string <> "!" end`           | do-fusion + stray `)`              | logit-ban / balanced-delimiter grammar |
| p_swap (x3)       | body is `b, a` (should be `{b, a}`)              | missing tuple braces               | grammar (borderline; expr validity)    |
| m_sumdoc (x2)     | `Enum.reduce(1..n, 0, +)` - bare `+` as arg     | operator used as argument          | grammar (operator-not-an-expression)   |

The 13 compile (semantic) failures are NOT addressable by grammar or scaffold, e.g.:
- `p_double`: `defmodulelerler do ... num 2` - degeneracy (token repetition) + missing `*`.
- `p_sum`: `def def_list(...)` typo + undefined var `t` (ref body uses `tail`).

## Interpretation (the truth this establishes)

- The syntactic prize is **real and large** (40% of drafts) and is **currently unclaimed**
  by the free layer - so it is genuinely paid for in GPU repair rounds today.
- But it **decomposes**, and only part is grammar's job:
  - skeleton errors (missing `do`/`end`) -> **scaffold-seeding** kills these outright;
  - balanced-delimiter / keyword / operator errors -> a grammar or logit-bans could catch;
  - degeneracy / typos (`defmodulelerler`, `ends`, `def_list`) -> partly a model-QUALITY
    problem; a CFG cannot stop a model emitting a valid-but-wrong identifier.
- A **hard ceiling**: ~50% of all draft failures are semantic/degeneracy that neither
  grammar nor scaffold can touch. Grammar/scaffold improve the SYNTACTIC gate only; the
  repair loop (or better drafting) still owns the rest.

## Implication for ordering

Expected value, cheapest first:
1. **P1 scaffold-seeding** - directly kills the skeleton class (c_vowels), constrains
   length (helps truncation like c_stack), zero engine change. Test next.
2. **P2 logit-bans** - target `ends`, do-fusion `, do`, stray-paren classes.
3. **P4 grammar** - catches balanced-delimiter/operator classes, but entangled with
   degeneracy, so marginal ceiling is lower than the raw 40% suggests.

Caveat to verify in P1: some parse failures may be n_gen=192 truncation, not malformed
syntax (`c_stack: missing terminator: end`). Truncation is a LENGTH problem - scaffold's
fixed `end` anchor and the hole-size sweep address it, a grammar does not.
