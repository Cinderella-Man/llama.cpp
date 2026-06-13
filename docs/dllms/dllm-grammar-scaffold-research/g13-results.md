# G13 - check-first prompting (show the model the test at draft time)

The VERDICT's recommended "untested semantic lever": put the check (test assertions) into
the draft prompt so the model targets the right contract/behavior - without scaffold's
structural constraint. Diff: `g13-checkfirst.diff` (reverted). Probe (capture):
`g13_capture.exs`. Full-bench A/B on Dream-7B, same process. 2026-06-13.

## Result: REGRESSES (both phrasings), 0 cases fixed

| arm                         | total | p     | m   | c   | tok/s | fixed | broke |
|-----------------------------|-------|-------|-----|-----|-------|-------|-------|
| baseline                    | 35/48 | 18/18 | 2/3 | 0/9 | 5.59  | -     | -     |
| check-first (raw assertion) | 32/48 | 17/18 | 0/3 | 0/9 | 3.81  | 0     | 3     |
| check-first (natural lang)  | 28/48 | 13/18 | 0/3 | 0/9 | 2.66  | 0     | 7     |

forge-tier repairs: baseline 50 -> raw 59 -> NL 76 (the check text makes drafts WORSE, so
the repair loop works harder). NL (more verbose) perturbs more than raw. c-tier stayed 0/9
under both: showing the test did not unlock a single ceiling case.

## Why it fails (from the capture probe, `g13_capture.exs`)

Drafting WHILE shown the check, the model:
1. **Does not gain the missing capability.** c-tier still wrong: `string.upper!` for shout,
   `Enum.drop(1, stack)` for pop, a broken vowel counter. The test states WHAT, not HOW;
   a 7B model that can't write the algorithm still can't (c-tier is a known capability
   ceiling - both Dream AND DiffuCoder fail it, dc-results.md).
2. **Gets perturbed into NEW syntax errors** the baseline didn't make: `def sum_list([])
   when do: 0` (spurious guard), `defmodule Acc` missing `do`, `@spec sum_to(integer))`
   extra paren. The extra prompt text shifts the draft distribution downward - the same
   fragility the C5 prompt-slimming experiments hit (prompt changes are pass-rate-gated).
3. **Risks example-overfitting:** `def double(number) do number * number` - `2*2 = 4`
   satisfies the single check example but is squaring, not doubling. Minimal checks (one
   I/O pair) invite gaming the specific value rather than generalizing. (Here a structural
   error happened to mask it; on a clean structure it would be a FALSE pass.)

## Verdict

(See `robustness-analysis.md`: at n=3 seeds the raw-phrasing -3 is within seed noise -
paired 3-worse/0-better, p~0.13 - so raw check-first shows NO IMPROVEMENT rather than a
robust loss; the NL -7 IS robust, p~0.008. Either way, 0 cases fixed.)

Check-first prompting does not help on this stack (35 -> 32 raw, -> 28 NL; 0 fixed). The
bottleneck is the model's SEMANTIC capability (c-tier), which no prompt grants; the p-tier
is already maxed so check-first can only perturb it down; and minimal checks invite
overfitting. This closes the last draft-side lever the VERDICT had flagged.

## The meta-finding (the real conclusion of the whole campaign)

Every draft-side intervention tried - scaffold (P1c), grammar decode (P4b), check-first
(G13) - REGRESSES, on both Dream and DiffuCoder. The baseline draft+repair loop is already
near the model's capability ceiling on this bench; perturbing or constraining the draft
costs more than it gains. The remaining headroom is NOT in how we prompt/constrain the
draft - it is MODEL CAPABILITY (a stronger/larger or Elixir-tuned model) and the repair
loop (already the workhorse). Stop optimizing the draft; change the model or invest in
repair/verification.
