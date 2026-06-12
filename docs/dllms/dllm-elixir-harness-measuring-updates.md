# Fixing the measuring instrument: kintsugi bench v2


## Context

Every engine and harness decision (threshold sweeps, Layer A kv defaults, model choice,
quick-wins) is gated by kintsugi's bench. The current bench (kintsugi/bench/bench.exs) is
4 cases x 1 seed with a single aggregate number - it swung 16.8 -> 9.1 tok/s on one
redraft coin-flip during Layer A work, making a 1.96x engine win unreadable. The
instrument must be fixed before any further optimization is trusted.

## MEASURED findings driving the design (probe: Dream-7B, 2026-06-12, AC power)

1. Per-seed WALL variance on passing cases is TINY: medium case across seeds {5,105,205}
   = 1233/1169/1173 ms (+-3%). The instrument's noise is NOT timing - it is pass/fail
   flips and draft-count cascades. Conclusion: seeds exist to measure PASS-RATE; timing
   needs only a median over the passing runs.
2. ACCOUNTING BUG, quantified: failing generate calls report ms_wall of the LAST attempt
   only - measured true wall ~10,000 ms vs reported 1,713-2,432 ms (4-6x under-billing).
   Root: kintsugi.ex attempt_drafts error path returns last_error stats without
   restamping (lib/kintsugi.ex:65-86; the restamp/5 fold only runs on success). The old
   bench's "forge_long FAIL 1725ms" actually burned ~10 s of GPU. Any aggregate built on
   reported ms_wall flatters failure-heavy configs. Fix in BOTH places: kintsugi restamps
   failures too; the bench ALSO measures wall externally around each call (trust nothing).
3. The hard-case failure is STABLE across seeds (Stack: 0/3 fail, ~10 s each) - good
   anchor case; the flip-prone zone is mid-difficulty. Tier the task set so pass-rate has
   dynamic range.
4. Runtime budget (measured basis: pass ~1.2 s, fail ~10 s, heal 0.4-2 s, infill ~0.4 s):
   15 cases x 3 seeds x 1 model ~ 2 min. Two models + two configs ~ <10 min. Acceptable.
5. Environment poisons results (battery vs AC measured at up to 5x earlier in the
   project; CPU sampling 24->1924 ms/step) - run metadata MUST capture power state.

## Design

### Files (all under kintsugi/)
- bench/cases.exs        - pure data: the task list (returns a list of maps)
- bench/bench.exs        - REWRITTEN runner (same entrypoint name; docs stay valid)
- bench/compare.exs      - paired A/B diff of two result files
- bench/results/         - JSONL outputs, git-tracked baselines under results/baselines/
- lib/kintsugi.ex        - failure-path restamp fix (see finding 2)
- README.md              - bench protocol section

### Task set: 15 cases x 3 seeds (base, +100, +200), 5 tiers
- T1 trivial forge (3): Doubler.double/1; Greeter.hello/0 returning a string;
  Parity.even?/1. Checks: exact assertions.
- T2 medium forge (4): Sums.sum_list/1 recursive; Rev.reverse/1 without Enum.reverse;
  Vowels.count/1; Fizz.fizzbuzz/1 -> list of 15. (The flip-prone tier - this is where
  3 seeds earn their keep.)
- T3 hard forge (3): Stack push/pop/size (known stable-fail anchor); CounterServer
  (GenServer, start_link + increment + get); Point struct + move/3.
- T4 heal (3): broken fib (", do" midline class); undefined-variable body; missing "end"
  multi-line module. All with functional checks. (Measures the Credence+infill path;
  expected mostly 0-GPU after FixDoBlockFusion - that is itself a tracked metric:
  credence_fixes count.)
- T5 infill micro (2): direct Kintsugi.Engine.infill calls (no harness loop): 3-mask
  one-liner, 12-mask function body; asserts byte-identity of fixed text + compile.
  (Measures raw engine repair latency; isolates engine changes from harness changes.)
Case map fields: %{id, tier, kind: :forge|:heal|:infill, instruction/code/canvas, check,
seeds: [s, s+100, s+200], opts (n_gen etc. when non-default)}.

### Metrics + output
Per (case, seed) JSONL line:
  {id, tier, kind, seed, ok, wall_ms_external, reported: {ms_wall, ms_total, drafts,
   repairs, credence_fixes, tokens, tokens_per_second}, history_len}
Run-header JSONL line (FIRST line):
  {label (CLI arg), timestamp, engine: full /health dump, git_rev (llama.cpp +
   kintsugi via System.cmd("git", ["rev-parse", "--short", "HEAD"])),
   power_ac: read /sys/class/power_supply/A*/online, gpu: nvidia-smi --query-gpu=
   name,power.draw,clocks.sm --format=csv,noheader, elixir/otp versions}
Summary table (stdout): per tier - pass k/n, median wall (passing), worst wall; per
kind - deliverable tok/s = sum(tokens of passes) / sum(EXTERNAL wall of ALL runs incl
failures - the honest denominator), pass-rate. Plus a NOISE line: for each case, spread
across seeds (max-min wall on passes; pass-count).
Gate definition (written into README): a config change is a REGRESSION if aggregate
pass-count drops by >=2 of 45 run-units OR any tier's median wall worsens >15% with
pass-count not improving. Single-run deltas inside those bands = noise, do not act.

### compare.exs old.jsonl new.jsonl
Pairs lines by (id, seed). Reports: pass transitions (counts + which), per-kind
deliverable tok/s delta, per-case median wall delta (passing pairs only), and a verdict
per the gate definition. Exit code 1 on regression (CI-friendly later).

### Runner mechanics
- Caller starts the server (documented standard commands for: baseline config, kv_prefix
  config); runner takes [url] [label] argv. Runner REFUSES to run (clear error) if
  /health fails or power_ac reads "0" without --allow-battery (the battery lesson,
  enforced).
- Warmup: one throwaway T1 generate before timing (first request pays sampler attach +
  graph capture; measured earlier at ~150 ms extra).
- Cases run sequentially (GPU is serial; keeps timing clean).
- Wall measured around the Kintsugi call with System.monotonic_time - reported stats
  recorded but never used as the denominator.

### kintsugi accounting fix (lib/kintsugi.ex)
In attempt_drafts budget-exhausted branch: restamp the error stats - ms_wall =
now - t0, drafts = spent (currently last attempt's 1), keep last reason. ~6 lines.
Unit test: stub... simplest honest test: heal with max_repairs 0 vs 1 wall ordering is
flaky; instead assert via the public contract: generate_with_stats on an impossible
check with max_drafts 3 reports ms_wall >= sum of attempts (integration test, tagged
:engine).

## Implementation steps
1. lib fix + test (finding 2).
2. bench/cases.exs (the 15 cases; verify every check string compiles + passes against a
   KNOWN-GOOD hand-written solution first - a self-test mode: `mix run bench/cases.exs`
   compiles each reference solution against its check so a broken CHECK never poisons
   the bench).
3. bench/bench.exs rewrite (runner + JSONL + summary + env capture + battery guard).
4. bench/compare.exs.
5. Baseline runs: Dream baseline + Dream kv_prefix-32 + DiffuCoder baseline (3 files
   committed under results/baselines/ with the run commands in README).
6. README protocol section + docs/dllms/dllm-elixir-harness.md log entry.

## Verification
- cases self-test: all 15 reference solutions pass their own checks.
- mix test (offline 10 + engine-tagged) green.
- Run bench twice back-to-back on identical config: compare.exs must declare NO
  regression (the instrument agrees with itself - the actual acceptance test).
- Deliberately rerun with kv_prefix 32: compare.exs output is readable per-tier (the
  Layer A question this whole plan exists to answer cleanly).
- Failure accounting: forge_long-class case shows external wall ~= reported wall after
  the lib fix (within HTTP overhead).

## Unresolved questions
1. Commit baseline JSONLs to git (recommended: yes, under results/baselines/ - they are
   the longitudinal record) or keep untracked?
2. DiffuCoder in the default matrix (adds ~2-3 min + a server restart) or Dream-only by
   default with DiffuCoder weekly? (recommended: both for baselines, Dream-only for
   iteration runs)
3. T3 GenServer case: checks need :timer.sleep-free synchronous API - acceptable to
   require start_link returning {:ok, pid} + GenServer.call (recommended) vs simpler
   Agent-based case?


## EMPIRICAL GRILLING APPENDIX (2026-06-12): every claim probed, several corrected

All probes: Dream-7B Q4_K_M, RTX 5070, AC power (ADP1/online=1), standard server flags.

### A. Environment-capture commands - VERIFIED with corrections
- Power: /sys/class/power_supply/ADP1/online exists and reads 1 on AC (the A* glob works).
- nvidia-smi --query-gpu=name,power.draw,clocks.sm --format=csv,noheader -> "NVIDIA
  GeForce RTX 5070 Laptop GPU, 23.57 W, 2782 MHz" - parseable as planned.
- CORRECTION: kintsugi has NO own git repo - it lives inside the llama.cpp checkout.
  ONE git_rev covers both (plan said two).
- Warmup delta MEASURED: first request 233 ms vs second 148 ms (= 85 ms attach+capture
  cost). The runner's throwaway warmup is confirmed worthwhile.

### B. The self-test caught its first broken check BEFORE the bench shipped
The planned struct case check used %Point{x: 1, y: 2} literal syntax - it FAILS against
a CORRECT reference solution: in a single .exs script, top-level code compiles before
the module loads, so struct expansion raises "cannot access struct Point" (a genuine
Elixir gotcha, not a model failure - it would have scored every model 0 on a case no
code could pass). Fixed form: runtime struct/2 - "p = struct(Point, x: 1, y: 2); ...".
All other 12 planned checks passed their references through Kintsugi.Verifier.run.
LESSON: the self-test mode is not optional ceremony; run it whenever a case changes.

### C. Tier composition by INTUITION was wrong - cases must be graded empirically
GPU probe results (3 seeds each unless noted):
| candidate                  | pass | wall (ms)         | verdict |
| double/1                   | 3/3  | 1696/1640/4528    | PASS tier (one repair-cascade seed - healthy variance) |
| even?/1                    | 3/3  | 1122/1123/1121    | PASS |
| sum_list/1 (recursive)     | 3/3  | 1233/1169/1173    | PASS |
| reverse/1 (no Enum)        | 3/3  | 1236/1033/1020    | PASS |
| max_of/1                   | 3/3  | 918/892/896       | PASS |
| swap/1 (tuple)             | 3/3  | 1099/967/961      | PASS |
| count vowels (string)      | 0/3  | ~4100             | CEILING (string hole) |
| shout/1 (upcase string)    | 0/3  | ~5400             | CEILING (string hole) |
| hello/0 (CONSTANT string)  | 0/3  | ~4800             | dropped (string hole, redundant) |
| fizzbuzz/1                 | 0/3  | ~17900 (!)        | dropped (stable-fail at 18 s = budget killer) |
| len/1 (recursive, no length)| 0/3 | ~13600            | dropped (cost) |
| Stack (multi-function)     | 0/3  | ~10000            | CEILING (multi-function hole) |
| Point struct + move/3      | 0/1  | 8012              | dropped (redundant with Stack, expensive) |
| GenServer counter          | 0/1  | 7378              | dropped (ditto) - resolves open question 3 |
MODEL INSIGHT (free finding): Dream-7B's Elixir failure modes are SYSTEMATIC - any
string manipulation fails (even returning a constant string), and any multi-function
module fails; arithmetic, list recursion, tuples and pattern matching pass reliably.
Ceiling cases are therefore not "hard versions of pass cases" - they probe two
specific capability holes, which is exactly what a model-upgrade decision (DiffuCoder
default? E2 1.5B? SDTT) needs to see move.

### D. Planned heal cases were UNHEALABLE - the heal tier must track stack capabilities
- undefined-variable heal: FAILS (4 GPU repairs, credence_fixes=0 - no Credence rule
  for it, and masked infill does not recover it either). KEPT as the single
  ASPIRATIONAL case - it is the progress meter for catalog G3/G4 work.
- missing-end heal: FAILS the same way (no Credence rule, infill flails). DROPPED.
- do-fusion family heals (the classes FixDoBlockFusion handles): fib ", do" midline,
  "do: ... end" fusion, ", do" EOL - all heal in ~315 ms with credence_fixes=1 and
  ZERO GPU repairs. These 3 form the heal tier and double as a regression test for the
  deterministic path (if credence_fixes drops to 0 or repairs rise, Credence broke).
### E. Infill micro-cases - timing confirmed
3-mask one-liner: 73 ms; 12-mask function body: 329 ms. Clean engine-latency probes.

### F. FINAL EMPIRICALLY-GRADED SUITE (15 cases, measured cost ~100 s/config at 3 seeds)
- P pass tier (6, ~25 s): double, even?, sum_list, reverse, max_of, swap - measures
  draft speed + repair-cascade variance.
- C ceiling tier (3, ~59 s): vowels, shout, Stack - measures the two capability holes;
  expected 0/9 today, every pass gained is real quality movement.
- H heal-deterministic tier (3, ~3 s): fib + 2 do-fusion variants - measures the
  Credence path; expected 9/9 at ~320 ms.
- A aspirational heal (1, ~10 s): undefined-variable - expected 0/3 today; the G3/G4
  progress meter.
- I infill micro (2, ~1 s): 3-mask, 12-mask - raw engine repair latency.
Runtime: ~1.7 min/config measured basis -> Dream baseline + kv_prefix + DiffuCoder
baseline well under 10 min including server restarts. Budget CONFIRMED.

### G. Accounting bug (finding 2) re-confirmed at suite scale
Failing forge runs reported 1.7-2.4 s while externally measured at 9.5-10.2 s (3 drafts
x repair ladders). With the ceiling tier expected-failing by design, the bench MUST use
external wall as the only denominator (and the lib restamp fix ships with this work).

### Open questions - RESOLVED by probing
1. Baselines in git: YES (results/baselines/, they are small JSONL).
2. DiffuCoder: in baselines, not in iteration runs (unchanged recommendation).
3. GenServer vs Agent for T3: NEITHER - both graded out empirically; the ceiling tier
   is vowels/shout/Stack. GenServer-class capability is measured implicitly by Stack
   (multi-function hole) at half the cost.
