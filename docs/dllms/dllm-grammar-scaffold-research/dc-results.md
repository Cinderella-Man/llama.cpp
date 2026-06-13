# DiffuCoder-7B cross-check (the one open branch from VERDICT.md)

Question: is the "syntax is not the bottleneck; scaffold/grammar don't raise pass" verdict
Dream-specific, or does it hold for a CODE-tuned diffusion model? DiffuCoder-7B-cpGRPO is
the model Muendler et al. measured +7% on (C++/JSON), and it's on disk. Re-ran P0 + P1 + P1b
on it, same bench/params/harness as Dream. Probe: `dc_crosscheck.exs`, samples
`dc-samples.jsonl`. DiffuCoder server, same flags as Dream, mask `<|mask|>`. 2026-06-13.

## Result (30 forge drafts, first-shot)

| stage                     | pass | parse-fail | compile | check | (Dream for comparison)      |
|---------------------------|------|-----------|---------|-------|-----------------------------|
| raw draft (P0)            | 3    | **27**    | 0       | 0     | Dream: pass 3, parse 12     |
| module-shell scaffold (P1)| 0    | **0**     | 0       | 30    | Dream: pass 3, parse 0      |
| signature scaffold (P1b)  | 3    | 8         | 12      | 7     | Dream: pass 7, parse 3      |

## Findings

1. **The verdict GENERALIZES - more strongly.** Scaffold eliminates 100% of parse errors
   (27 -> 0) just like Dream (12 -> 0), but pass rate does NOT improve (raw 3 -> scaffold 0
   -> sig 3). For DiffuCoder, module-scaffold turns ALL 30 drafts into `check` failures:
   they compile but compute the wrong thing. Semantics bind for the code model too.
2. **DiffuCoder is syntactically MESSIER raw, not cleaner.** 27/30 raw drafts fail to parse
   (vs Dream's 12/30) - verified real, not an extraction artifact: e.g.
   `def double(number) do: number * 2 end` (do-fusion) and `def swap({a, b}), do: {b, a}`
   followed by a stray `}`. Yet DiffuCoder still reaches 33/48 on the full bench - because
   the repair loop + Credence rules clean the syntax. Same lesson as Dream: raw syntax
   quality is poor but irrelevant to final pass; the loop owns it.
3. **Signature scaffold helps DiffuCoder even LESS than Dream** (pass 3 -> 3, vs Dream
   3 -> 7). A code model commits code tokens at high confidence and fights a forced
   skeleton harder; the structure it would have chosen is overwritten.

## Interpretation

The negative is not Dream-specific: across BOTH diffusion models available, on this Elixir
bench, syntactic constraint (scaffold) reliably removes syntax errors and does not raise
pass rate, because the bottleneck is semantic and the repair loop already handles syntax.

## The honest boundary of this cross-check

This is still ELIXIR. DiffuCoder's native strength (and the paper's +7%) is C++/Python/JSON
- languages with far more training data and where the model's semantic competence is high,
so the residual failures there are more likely syntactic (where a grammar CAN help). We did
NOT reproduce the paper's C++/multi-region-infilling setting; doing so would require a
different harness and is out of scope for "does this help OUR Elixir stack." For Elixir on
7B-class diffusion models, the answer is settled: no.
