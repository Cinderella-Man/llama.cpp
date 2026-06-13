# P1 / P1b - scaffold-seeding (Dream-7B Q4_K_M)

Probes: `p1_scaffold_seed.exs` (module shell), `p1b_signature_seed.exs` (+ function
signature from the check). Samples: `p1-samples.jsonl`, `p1b-samples.jsonl`. 30 forge
cases x seeds, first-shot (no repair loop). Run 2026-06-13.

## Results

| variant                         | pass | parse | compile | check | note |
|---------------------------------|------|-------|---------|-------|------|
| baseline free-form draft (P0 B) | 3    | 12    | 13      | 2     | |
| P1 module-shell seed            | 3    | **0** | 10      | 17    | parse eliminated, failures -> semantic |
| P1b signature seed              | **7**| 3     | 4       | 16    | doubles pass; reintroduces some parse |

Canvas: P1 `defmodule <Name> do\n <MASK*n>\nend`; P1b adds `def <fname>(<a,b..>) do ...`,
fname/arity parsed from the check. Body-hole swept (P1 [16,24,32,48], P1b [12,16,24]),
best-of taken. Infill config steps 48, conf 0.6, temp 0.2, top_k 40.

## What we learned (truth)

1. **Module-shell seeding eliminates 100% of parse errors (12 -> 0).** Seeding
   `defmodule X do ... end` removes the entire syntactic-skeleton failure class (e.g.
   c_vowels' missing `do`). The user's "don't start from all-mask, seed the shape"
   hypothesis is CONFIRMED for syntax.

2. **But syntax was not the binding constraint.** Module-shell pass is unchanged (3 -> 3):
   the 12 parse failures became compile/check (semantic) failures, not passes. Examples
   (p1-samples): p_swap -> invented `all_pairs/1` instead of `swap/1`; c_vowels ->
   `String.count(s,"aeiou")` (no such fn); p_max -> hallucinated `:lang.to_list`. Seeding
   structure alone lets the model write syntactically-clean wrong code.
   => A pure-syntax constraint (a GBNF/CFG grammar, P4) is predicted PASS-NEUTRAL here.
   This is the central finding for the "would a grammar help?" question.

3. **The real lever is SEMANTIC seeding.** Putting the function contract (name + arity
   from the check) into the canvas doubled first-shot pass (3 -> 7) by killing the
   wrong-function-name class (p_swap parse -> pass). This is scaffold/G9 + check-first/G13
   territory, NOT grammar/G11.

4. **Signature seeding has a sharp edge.** A rigid single-clause `def f(a) do <body> end`
   skeleton FIGHTS multi-clause / recursive solutions: p_sum (recursive `sum_list`) went
   compile -> parse under P1b because the model tried to express two clauses inside one
   seeded def. And both P1 and P1b broke p_max (free-form got `Enum.max` right; the fixed
   canvas pushed it into a worse basin). Scaffold is not free - it can constrain the model
   out of a correct solution. Matches the literature: canvas shape/length is a sensitive
   hyperparameter (arXiv:2509.11252).

5. **Speed angle (unconfirmed):** the winning scaffold infill is ~2x cheaper per attempt
   than a 192-token free-form draft (median 416 vs 617 ms) because the canvas is tiny -
   BUT the hole-size sweep (3-4 infills) multiplies that. Net speed needs the full
   repair-loop bench, not first-shot.

## Caveats

- First-shot only. Production uses a repair loop (35/48); these numbers do not include it.
  The decisive KPI test is: wire signature-scaffold as the DRAFT step and run the full
  bench for check-pass + tok/s (A/B same server process).
- Tiny sample (30). Directional, not a verdict.
- Single-function arity parser; multi-function checks (c_stack) seed only the first fn.

## Bottom line so far

For the user's two ideas:
- "Use a grammar (GBNF/CFG)": predicted pass-NEUTRAL on this stack (P1 proves syntax is
  not the bottleneck). The large P4 build is not justified by pass-rate. A grammar's only
  remaining value is SPEED (fewer repair rounds) - to be tested, and scaffold likely
  captures that more cheaply.
- "Seed the known shape instead of all-mask": YES, but seed the CONTRACT (signature), not
  just the skeleton. Cheap (existing infill), doubles first-shot pass, with caveats
  (multi-clause, can break free-form wins). Worth a full-bench A/B.
