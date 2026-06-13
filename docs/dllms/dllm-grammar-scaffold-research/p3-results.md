# P3 - offline Elixir-grammar acceptor study

Grammar: `elixir-subset.gbnf` (hand-written subset). Engine: llama.cpp
`build/bin/test-gbnf-validator` (the SAME char-level grammar engine used at decode time -
strict left-to-right `llama_grammar_accept` per codepoint). Harness:
`p3_grammar_acceptor.py`. Data: 14 bench references + 30 P0 drafts. Run 2026-06-13.

## Authoring friction (a finding in itself)

First version rejected 100% of inputs - the GRAMMAR FILE failed to parse: llama.cpp GBNF
does not allow a rule's alternation to continue on a line that STARTS with `|` (the `|`
must be at the end of the previous line), and `.` is not a wildcard. After fixing those,
the grammar accepts all references. Hand-writing even a SUBSET of Elixir as GBNF is
fiddly; a full grammar (sigils, heredocs, interpolation, the macro surface) would be much
harder and is partly context-sensitive (not pure CFG).

## Result

| input class                      | grammar-accepts | wanted  | reading |
|----------------------------------|-----------------|---------|---------|
| 14 references (valid Elixir)     | **14/14**       | accept  | 0 false-reject - grammar calibrated to real code |
| 12 P0 syntax-broken drafts       | **0/12**        | reject  | catches 100% of the parse-error class |
| 13 P0 compile (semantic) drafts  | 9/13            | accept  | correctly passes parse-valid-but-wrong code |
| 2 P0 check (logic) drafts        | 2/2             | accept  | correct |
| 3 P0 pass drafts                 | 3/3             | accept  | correct |

## What this establishes

1. **A grammar reliably catches the syntax-error class** (0/12 broken drafts slipped
   through). A grammar-constrained decoder WOULD prevent exactly the parse failures P0
   found - confirming the mechanism works.
2. **A grammar cannot judge semantics** (9/13 semantically-broken drafts are accepted
   because they parse). It moves failures parse -> semantic, identical to scaffold (P1).
3. **Subset-strictness is a real risk.** 4/13 compile-class drafts were REJECTED by the
   grammar even though Elixir's own parser accepts them - they fall outside my subset
   (e.g. degeneracy like `defmodulelerler`, `def def_list`). On code beyond the bench's
   narrow shape, a hand-written subset grammar WILL false-reject valid Elixir, which at
   decode time would force the model off correct tokens. A fuller grammar reduces this but
   is much harder to write and partly not expressible as a CFG.

## Bearing on P4 (grammar-constrained decode)

Converging with P0/P1/P1c:
- P3: a grammar catches 100% of syntax errors (mechanism confirmed), but
- P1: removing 100% of syntax errors did NOT raise pass rate (3->3); and
- P1c: doing it at draft time (scaffold) REGRESSED the full bench (35->26, 5.29->2.83),
  because the repair loop already handles syntax cheaply and the constraint damages
  solution quality.

Prediction for P4 (grammar-constrained decode): pass-NEUTRAL at best on this stack, with
added per-step decode overhead (CFG-intersection per candidate) and subset-strictness risk
-> likely a speed LOSS. The empirical case against the syntax-constraint line is now strong.

Scope caveat: all of this is Dream-7B on a small, recursion-heavy Elixir bench. The
published +7% (Muendler et al.) was on DiffuCoder/LLaDA and C++/JSON/multi-region infilling
- a different model and task. We HAVE DiffuCoder-7B locally; a cheap cross-check there would
tell us whether the negative is Dream-specific before declaring a general verdict.
