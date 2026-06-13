# Robustness analysis - grilling the verdict (do the negatives actually hold?)

After the campaign concluded "draft-side interventions don't help," this is an adversarial
re-examination of that conclusion's validity, from the 64 stored bench runs + the kintsugi
source. Four findings; two harden the verdict, one softens a claim, one corrects an idea.

## 1. The bench HAS dynamic range (negatives are real, not blind) - HARDENS

Across 64 runs, 8/10 forge cases flip outcome (pass in some runs, fail in others):

| case      | tier | pass | fail | per-attempt p_hat (25 baseline runs) |
|-----------|------|------|------|--------------------------------------|
| p_swap    | p    | 163  | 26   | 0.83 |
| p_max     | p    | 145  | 44   | 0.83 |
| p_double  | p    | 151  | 41   | 0.80 |
| p_reverse | p    | 128  | 61   | 0.80 |
| p_sum     | p    | 97   | 93   | 0.77 |
| p_even    | p    | 114  | 78   | 0.72 |
| m_sumdoc  | m    | 47   | 97   | 0.48 |
| c_vowels  | c    | 9    | 180  | 0.08 |
| c_shout   | c    | 0    | 189  | 0.00 |
| c_stack   | c    | 0    | 188  | 0.00 |

The bench is NOT saturated - p_sum is a near coin-flip. So it CAN detect a draft-quality
change; the interventions genuinely moved it DOWN, they weren't invisible to the instrument.

## 2. n=3 seeds = low power; SOFTEN the check-first claim - CORRECTS

Each A/B used 3 fixed seeds against cases that are ~0.5-0.8 per attempt. Paired-flip counts:
- **scaffold (P1c): 9 worse, 0 better** -> sign-test p < 0.002. ROBUST loss.
- **grammar (P4b): ~21x slower** -> unambiguous regardless of n.
- **check-first (G13): 3 worse, 0 better** -> p ~ 0.13. WEAK. Honest restatement: check-first
  shows NO improvement and a slight, NOT-robustly-significant harm at n=3. (The NL variant's
  7-worse is stronger, p ~ 0.008 - that one is a real loss.)

Correction to `g13-results.md`/`VERDICT.md`: "check-first regresses" should read "check-first
does not help; raw-phrasing harm is within seed noise, NL-phrasing harm is real."

## 3. The floor is CAPABILITY, confirmed - HARDENS

c_shout and c_stack passed 0 / ~189 times across 64 runs spanning every profile AND both
models (Dream + DiffuCoder). a_undef 0/183. This is a 7B-Elixir capability wall, not a
harness or syntax problem. No draft-side trick (or grammar) can move it; only a stronger /
Elixir-tuned model can.

## 4. best-of-N seed racing is ALREADY in production - CORRECTS

The variance in #1 looked like an untested lever (best-of-3 projects forge 5.3 -> 7.0/10).
But `attempt_drafts` (kintsugi.ex:156) ALREADY redrafts on failure with a bumped seed
(max_drafts=3) - so the baseline p_hat values are POST-retry. The residual variance is what
remains after best-of-3. Pushing further (G6 parallel racing, more seeds) gives diminishing
returns at N x compute - a rig lever, not a clean new win. Not the answer it first appeared.

## Net

The verdict's load-bearing claims survive grilling and are now better grounded:
- scaffold-as-draft: robust LOSS (paired p<0.002).
- grammar decode: robust LOSS (21x slower, pass-neutral-at-best).
- check-first: NO improvement (harm within noise for raw; real for NL).
- The system is CAPABILITY-bound; the mature draft+repair+seed-retry harness already
  extracts most of this model's headroom. Real remaining headroom = a better model.

Untested branches that remain (ranked by likely value, all probably small):
1. grammar-constrained REPAIR (not draft) - the one place grammar could still pay, untested.
2. more-seeds hardening of the check-first number - rigor only.
3. eth-sri C++/JSON reproduction - tests grammar generality off our stack.
