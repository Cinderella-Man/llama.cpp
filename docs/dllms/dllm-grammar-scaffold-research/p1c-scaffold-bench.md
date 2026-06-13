# P1c - scaffold-seeding, FULL-LOOP bench A/B (the decisive KPI test)

Wired signature-scaffold into `forge` as an opt-in draft mode (`opts["scaffold"]`), added a
`scaffold` bench profile, ran the full 48-case bench against the same server process as the
baseline. Diff: `scaffold-forge.diff` (reverted after measuring - kept-as-killed). Run
2026-06-13, Dream-7B Q4_K_M, RTX 5070.

## Result: scaffold REGRESSES both pass and speed

| tier  | baseline | scaffold |
|-------|----------|----------|
| p     | 18/18    | **10/18** |
| m     | 2/3      | 1/3 |
| c     | 0/9      | 0/9 |
| h     | 9/9      | 9/9 |
| a     | 0/3      | 0/3 |
| i     | 6/6      | 6/6 |
| TOTAL | **35/48, 5.29 tok/s** | **26/48, 2.83 tok/s** |

Wall total 179 s -> 278 s (+55%). 9 regressions, **0 improvements**.

## Why (per-case diff)

Every regression is a case the baseline handled easily, turned into a max-escalation
failure (drafts 3, repairs 4):

| case            | baseline                | scaffold            | cause |
|-----------------|-------------------------|---------------------|-------|
| p_sum (x3)      | pass, 1 draft 1 repair  | fail, 3 drafts 4 rep | recursive `sum_list` needs 2 clauses; the seeded single-clause `def sum_list(a) do ... end` cannot express it |
| p_reverse (x2)  | pass                    | fail                 | recursive helper / multi-clause, same skeleton conflict |
| p_max (x3)      | pass instantly (~970ms) | fail (~12 s)         | `Enum.max(l)` - scaffold pushed the model into wrong basins (matches P1b breaking p_max) |
| m_sumdoc (x1)   | pass                    | fail                 | needs @doc/@spec the signature-only scaffold omits |

Mechanism:
1. **Rigid structural seeding fights the model's structural choices.** A single-clause
   `def f(a)` skeleton is actively hostile to multi-clause / recursive solutions (half the
   p-tier). The model cannot recover because the wrong shape is fixed from step 1.
2. **The repair loop already neutralizes syntax errors cheaply** (baseline: most p-tier
   pass in 1 draft + 0-2 repairs). Scaffold removes the freedom the model needs while
   solving a problem (syntax) the loop already solves.
3. When a scaffolded draft is wrong, it triggers the FULL escalation cascade (3 drafts x
   4 repairs) - far more expensive than a free-form redraft. Hence the +55% wall.

## Verdict

Scaffold-seeding (G9), as a draft-replacement, is a **measured LOSS** on this stack
(-9 pass, -47% tok/s). The first-shot win (P1b 3->7) was an artifact of (a) best-of
hole-sweep and (b) ignoring the repair loop. Combined with P1 (pure syntax constraint =
pass-neutral), the evidence now says structural/syntactic seeding does not help this
pipeline's KPI.

## Honest caveats (what could change the verdict, for the record)

- One scaffold design tested: module + SINGLE-clause signature, fixed hole 24. A
  multi-clause-aware skeleton, or scaffold-as-FALLBACK (free-form first, scaffold only on
  repeated failure), is untested and might regress less. But the core lesson (structure
  fixed too early + repair already covers syntax) is likely general.
- This bench is recursion-heavy (p_sum/p_reverse), somewhat adversarial to single-clause
  scaffolding. A single-function-problem suite would regress less - but that is
  cherry-picking the workload to fit the technique.
