# Kintsugi - dLLM Elixir harness design

Design document for "kintsugi" (~/projects/kintsugi): an Elixir project that wraps the
upgraded llama.cpp fork and runs the verify-and-remask loop - draft Elixir code with a
diffusion LLM fast, fix deterministically, verify with the compiler/tests, re-mask only the
broken spans, repair by infill, repeat until verified. A task = natural-language instruction
+ optional scaffold + optional ExUnit tests (the oracle).

## 1. Context

The llama.cpp fork (committed) provides everything the loop needs: GPU backend sampling
(2.1 ms/step flat), `--diffusion-infill` with `<|mask|>` markers (fixed text stays
byte-exact), and `--diffusion-conf-threshold` (128 -> 17-47 steps). Measured on the RTX 5070
Laptop: 512-token draft ~4.6 s (threshold 0.6) / ~12 s (block 0.9 near-greedy, higher
quality); 78-token infill repair at 36 ms/step (~0.3-1 s); model load 2-4 s.

## 2. Verified foundations (everything below was checked first-hand on this machine)

Toolchain: Elixir 1.19.5 / OTP 28 (asdf), Hex 2.4.2, built-in `JSON` module (no Jason dep).

Credence v0.7.1 at ~/projects/credence (production-ready, 117 rules, deps: sourceror ~1.11):
- `Credence.fix(code, opts) :: %{code: String.t(), issues: [Issue.t()], applied_rules: [...]}`
- `Credence.analyze(code, opts) :: %{valid: boolean(), issues: [Issue.t()]}`
- `%Credence.Issue{rule, message, meta: %{line: ...}}`; three phases (syntax/semantic/pattern),
  compile-gated, fixes that break compilation are auto-reverted (`:reverted` in trace).
- Precedent: ~/projects/opc-sft-stage2-elixir (Tunex) already orchestrates LLM->Credence loops
  with req + supervised processes.

Elixir API shapes (verified by execution, exact):
- `Code.string_to_quoted(code, columns: true)` error:
  `{:error, {[opening_delimiter: :do, expected_delimiter: :end, line: 1, column: 12,
  end_line: 1, end_column: 24], "missing terminator: end", ""}}` - parse errors carry SPANS.
- `Code.with_diagnostics(fn -> Code.compile_string(...) end)` diagnostic:
  `%{message: "undefined variable \"b\"", position: {3, 5}, span: {3, 6}, severity: :error,
  file: "nofile", ...}` - structured, in-process, no mix project needed.
- `Code.format_string!` on invalid code raises `TokenMissingError` (also SyntaxError family).
- `ExUnit.start(autorun: false)` + `Code.compile_string(tests)` + `ExUnit.run()` returns
  `%{total: 2, failures: 1, excluded: 0, skipped: 0}`; failure text includes `nofile:LINE`.
  Per-test detail via custom formatter GenServer (`{:test_finished, test}` casts).
- Isolated runner VM boot: `elixir -e ''` = 0.47 s; with a compile = 0.69 s.

Web-verified libraries: Req 0.6.1 (Finch pooling, `Req.post!(url, json: ...)`,
receive_timeout default 15 s); MuonTrap 1.7.0 (`MuonTrap.Daemon` in supervision tree,
SIGTERM -> SIGKILL `:delay_to_sigkill`, kills the external process when the BEAM dies - the
zombie-prevention standard); `Task.async_stream(enum, fun, max_concurrency:, timeout:,
on_timeout: :kill_task)`.

Engine output contract (current CLI): ALL output on stderr; generated text follows the line
matching `total time: .*steps: N`; mask piece is `<|mask|>` (id 151666, logged at startup,
same for Dream-7B and DiffuCoder).

## 3. Architecture

Three kinds of OS processes:

```
[kintsugi BEAM]  --HTTP(M1) / argv(M0)-->  [llama-diffusion engine, model resident in VRAM]
      |
      +--stdio JSON--> [runner VM (ephemeral `elixir runner.exs`) - the ONLY place
                        generated code executes]
```

Supervision tree (Kintsugi.Application):
```
Supervisor (one_for_one)
|- Kintsugi.ModelServer        GenServer; owns engine access; serializes GPU requests
|  `- MuonTrap.Daemon          supervises the engine binary (M1); restart -> model reload ~4s
|- Task.Supervisor (Kintsugi.RunnerSup)   verification subprocesses
|- Kintsugi.Metrics            ETS-backed counters + JSONL sink
`- Kintsugi.LoopSup            DynamicSupervisor; one Kintsugi.Loop process per task
```

Hard rule: generated code NEVER executes in the harness BEAM. `Code.compile_string` runs
module-body code at compile time and ExUnit runs arbitrary code - both live only in the
runner VM (and Credence.fix, whose semantic phase compiles, runs there too).

## 4. Engine interface

### M0 - CLI per call (validates the loop; +2-4 s model load per call)
ModelClient builds argv and runs via System.cmd (binary:
~/projects/llama.cpp/build/bin/llama-diffusion-cli):
- draft: `-m <gguf> -p <prompt> -ub 512 --diffusion-eps 0.001 --diffusion-algorithm 4
  --diffusion-steps 128 --diffusion-conf-threshold 0.6 --temp 0.2 --top-k 40 -ngl 99 --seed S`
- repair: `--diffusion-infill -p <canvas-with-markers> -ub <fit> --diffusion-eps 0.001
  --diffusion-algorithm 4 --diffusion-steps 32 --diffusion-conf-threshold 0.9 --temp 0.2
  --top-k 40 -ngl 99 --seed S`
- parse stderr: text = lines after the `total time:` line (strip one timestamped log prefix
  line); steps/timings from that line. Non-zero exit or missing marker line = engine error.

### M1 - llama-diffusion-server (new tool in the fork; the real target)
~200 lines: cpp-httplib (vendored in llama.cpp) wrapping diffusion_generate; single worker
thread (GPU is serial; httplib queues naturally).
- `GET /health` -> `{"status":"ok","model":"...","mask_piece":"<|mask|>","n_ubatch":512}`
- `POST /generate` body:
  `{"prompt": str, "infill": bool, "steps": int, "conf_threshold": float, "eps": float,
   "block_length": int, "algorithm": int, "temp": float, "top_k": int, "seed": int,
   "max_length": int?}`
  -> `{"text": str, "steps_done": int, "ms_total": float, "ms_per_step": float}`
  (template applied server-side for non-infill prompts, mirroring the CLI).
- Engine fix CORRECTED (see docs/dllm-engine-improvements.md sec 2): the per-call sampler
  attach re-reserve was MEASURED at ~11 ms (not ~1 s as earlier assumed), and a fresh chain
  per request sidesteps the one-way backend-init assert entirely. diffusion_generate's
  existing per-call setup/teardown is daemon-compatible as is - the server only hoists
  model + context creation (the 2-4 s part).
- ModelServer (Elixir): Req client; call timeout = steps x 350 ms + 10 s margin; health-check
  on init and after errors; on engine crash MuonTrap restarts it, ModelServer waits for
  /health (model reload ~4 s) and retries the in-flight request once.

## 5. Runner VM - verification ladder

`priv/runner.exs`, spawned per verification (M0/M1) or kept as a pre-warmed pool (M2).
Protocol: one JSON object on stdin -> one JSON object on stdout, then exit.

Request: `{"code": str, "tests": str|null, "timeout_ms": int,
           "checks": ["parse","format","credence","compile","test"]}`

Ladder (stops at first failing stage; each stage timed):
1. parse: `Code.string_to_quoted(code, columns: true)`; on error report kind=parse with
   line/column/end_line/end_column span + message + token.
2. format: `Code.format_string!` (rescue TokenMissingError/SyntaxError/Mismatched...);
   returns formatted code (canonicalization, also a cheap second parse gate).
3. credence: `Credence.fix(code)` (path dep on ~/projects/credence) -> fixed code +
   remaining issues + applied_rules trace. Runs here because its semantic phase compiles.
4. compile: `Code.with_diagnostics(fn -> Code.compile_string(fixed) end)` -> diagnostics
   (kind=compile, position {l,c}, span, message, severity; warnings-as-errors policy flag).
5. test: `ExUnit.start(autorun: false, capture_log: true)`; compile fixed code + tests;
   custom formatter GenServer accumulates `{:test_finished, %ExUnit.Test{}}` -> per-test
   {name, state, failure message, file:line}; `ExUnit.run()` map merged in.

Response: `{"ok": bool, "code": str(post-credence), "stage": "parse|...|test"|null,
  "errors": [{"kind": str, "line": int, "column": int?, "end_line": int?, "end_column": int?,
              "message": str, "test": str?}],
  "credence": {"applied_rules": [...], "issues": [...]},
  "timings_ms": {...}}`

Safety: harness enforces timeout_ms by killing the OS process (Port + kill, or
MuonTrap.cmd); the runner additionally self-arms `:timer.kill_after`. Module-name collisions
are a non-issue (fresh VM per run). Cost: ~0.7 s/run (M2 pool removes it).

## 6. Masker - error -> mask plan

Input: code (post-credence) + VerifyReport. Output:
`%MaskPlan{canvas: String.t(), holes: [%{lines: Range.t(), reason: atom()}], strategy: atom()}`

Strategy ladder by error kind (escalate one level per failed repair attempt):
- parse with span -> mask `line..end_line` (the verified span makes this precise)
- parse "missing terminator" -> mask from opening line to end of enclosing fragment
- compile undefined var/function -> mask the statement at position's line (+- 0 lines)
- compile other -> mask position's line +- 1
- test failure -> mask the WHOLE body of the function named in the failing call chain
  (locate def ranges via `Code.string_to_quoted(code, token_metadata: true, columns: true)`
  AST walk - do/end ranges come from token metadata)
- escalation: statement -> line+-1 -> enclosing function body -> REDRAFT

Hole sizing (the fixed-length constraint - diffusion fills exactly as many tokens as
markers): estimate tokens ~= ceil(bytes(removed_text)/3) (BPE prior for code); attempt
sequence per hole: 1.0x, then 1.5x, then 0.5x. Marker splicing is pure string surgery:
`pre <> String.duplicate(mask_piece, n) <> post` (mask_piece from /health; never token IDs).
Canvas scope: the enclosing function/module only, target <= 256 tokens (36-120 ms/step).

## 7. Loop policy (Kintsugi.Loop, one process per task)

States: :drafting -> :verifying -> (:done | :masking -> :repairing -> :verifying | :redraft).

- draft(seed): recipe `:fast` (timestep, threshold 0.6, temp 0.2, cap 128 -> ~17 steps,
  ~4.6 s) or `:quality` (block 32, threshold 0.9, temp 0.01 -> ~47 steps, ~12 s). Config.
- extract code fence from draft text (```elixir ... ```), fall back to whole text.
- verify ladder (sec 5). Pass -> :done.
- repair: for each error site (max 2 sites, nearest-first): MaskPlan -> infill request
  (`:repair` recipe: threshold 0.9, steps cap 32) -> verify. Up to 3 hole-size attempts per
  site, then escalate mask strategy, then count as failed site.
- redraft policy: >2 distinct error sites after credence, or any site exhausts its
  escalation ladder -> redraft with new seed. Max 3 drafts per task, then :failed with the
  best candidate + report attached.
- budgets: per-stage timeouts (draft 60 s, repair 20 s, verify 30 s incl. tests), total task
  budget 5 min (all config).

Expected economics per task: 4.6 s draft + ~0.7 s verify + (0.3-1 s repair + 0.7 s verify) x
k repairs -> typically < 10 s to compiler+test-verified code.

## 8. Message flow (one task, M1)

```
1  CLI/caller        -> LoopSup        start_child(TaskSpec{instruction, scaffold?, tests?})
2  Loop              -> ModelServer    {:generate, %{prompt, recipe: :fast, seed: 1}}  (GenServer.call,
                                       timeout 60s; mailbox = GPU queue)
3  ModelServer       -> engine         POST /generate {...}            (Req, JSON)
4  engine            -> ModelServer    200 {"text": draft, "steps_done": 17, ...}
5  Loop              -> RunnerSup      async runner.exs <- {"code", "tests", "checks": all}
6  runner            -> Loop           {"ok": false, "stage": "compile",
                                        "errors": [{"kind":"compile","line":3,...}],
                                        "code": post_credence_code}
7  Loop (Masker)     internal          MaskPlan{canvas, holes:[line 3], 1.0x}
8  Loop              -> ModelServer    {:generate, %{prompt: canvas, infill: true, recipe: :repair}}
9  engine            -> ModelServer    200 {"text": repaired_canvas, ...}
10 Loop              -> RunnerSup      verify again (full ladder incl. tests)
11 runner            -> Loop           {"ok": true, ...}
12 Loop              -> Metrics        {:task_done, %{outcome: :verified, wall_ms, drafts: 1,
                                        repairs: 1, repair_converged: true, stage_timings}}
13 Loop              -> caller         {:ok, %Result{code, report, metrics}}
```
Crash paths: engine dies mid-call -> Req error -> ModelServer health-waits (~4 s reload,
MuonTrap restarted it) -> one retry -> else {:error, :engine_down} bubbles to Loop -> task
retried once from current state. Runner hangs -> killed at timeout_ms -> counted as a
verification error of kind=timeout -> treated like a test failure (escalate/redraft).

## 9. Metrics (the numbers that decide the thesis)

Per task (JSONL + ETS aggregate): outcome, wall_clock_to_verified_ms, n_drafts, n_repairs,
repair_converged? (all sites fixed within <=3 attempts), per-stage timings, recipe, seed,
error kinds seen. Aggregates: compile rate / test pass rate / median wall-clock per recipe -
this is how draft threshold (0.5/0.6/0.7/0.8-block/0.9-block) gets tuned, not tokens/sec.

## 10. Project layout + deps

```
kintsugi/
  mix.exs            deps: req ~0.6, muontrap ~1.7, credence (path: ../credence)
  config/config.exs  engine url/binary+model paths, recipes, budgets, policies
  lib/kintsugi/      application.ex model_server.ex model_client.ex(M0) runner.ex(spawn+protocol)
                     verifier.ex masker.ex loop.ex task_spec.ex metrics.ex
  priv/runner.exs    the verification ladder (sec 5)
  priv/tasks/*.json  benchmark tasks {instruction, scaffold?, tests}
  test/              harness unit tests (masker policies, parsers - pure, no GPU)
```

## 11. Milestones

- M0 (first session): scaffold; ModelClient via CLI; runner.exs ladder (parse/format/
  credence/compile/test); Masker for parse+compile errors; Loop v0; 5 sample tasks (fib,
  reverse, GenServer cache, etc.); acceptance: `mix kintsugi.solve priv/tasks/fib.json`
  returns verified code end-to-end, metrics line written.
- M1 (engine): llama-diffusion-server in the fork (+ attach-once restructure of
  diffusion.cpp); ModelServer + MuonTrap.Daemon; per-call overhead 2-4 s -> ~0.
- M2: pre-warmed runner pool (kills the 0.7 s/verify); test-failure masking via
  token_metadata def-ranges; 20-50 task benchmark set (Tunex data is a ready source);
  threshold sweep -> pick default recipe by compile-rate + wall-clock.
- M3: parallel hole-size candidates (Task.async_stream, GPU-queue pipelined), remote engine
  on the 2x3090 box (URL change only), Credo stage, telemetry dashboard.

## 12. Verification of this plan

- M0 e2e: solve fib task from scratch on the laptop; deliberately break the draft (inject a
  bad token) and watch mask->repair converge; `kill -9` the engine mid-task (M1) and watch
  supervisor recovery; runner timeout with an infinite-loop test.
- Harness unit tests run without GPU (masker/parser logic on canned VerifyReports).
- Benchmark report from M2 set is the go/no-go for the thesis numbers.

## 13. Unresolved questions

- None blocking. To decide during M0: exact code-fence extraction rules for multi-block
  responses; whether warnings-as-errors should gate (start: yes, config flag).

## Implementation log (2026-06-11): kintsugi/ exists and works

The harness now lives IN-TREE at kintsugi/ (deliberately: the engine and harness grow
together; no upstream-merge ambition for this directory). Built and verified end-to-end
against llama-diffusion-server + Dream-7B on the RTX 5070.

### What was built (all zero-dependency: :httpc + Elixir >= 1.18 builtin JSON)
- Kintsugi.Engine - HTTP client: connect/1 (reads /health incl. mask_piece + replicas),
  generate/3, infill/3, tokenize/2. One Engine = one server = possibly a whole rig
  (the server's --diffusion-replicas does the GPU dispatch, so the harness needs no pool
  for a single host - a pool is only needed across HOSTS).
- Kintsugi.Masker - mask_line/4, mask_lines/5 (indentation-preserving, byte-identical
  elsewhere), hole_size/3 char-heuristic fallback.
- Kintsugi.Verifier - compile/1 via Code.string_to_quoted + Code.with_diagnostics
  (line-accurate diagnostics; candidate modules purged after), run/3 executes a check
  string in a SEPARATE OS process with timeout (generated code never runs in the harness
  VM).
- Kintsugi.forge/3 (instruction -> draft -> repair loop), heal/3 (existing code ->
  repair loop), repair/4, verify/2.
- 13 tests: 10 offline + 3 live (tagged :engine, KINTSUGI_ENGINE=... mix test --include
  engine).

### Measured results (Dream-7B Q4_K_M, threshold 0.6, temp 0.2, top-k 40)
- forge("a function double/1...") -> compiling code in 1431 ms, 0 repairs needed.
- heal(broken fibonacci with a syntax error, check "2 = Fib.fib(3)") -> HEALED in
  1502 ms total across 2 repair rounds, functionally verified.

### Discoveries (the reimplementation-grade details)
1. THE infill failure mode is hole sizing, not model quality. Masked diffusion fills
   EXACTLY n positions; an undershot hole TRUNCATES otherwise-perfect code (observed
   verbatim: 18-token hole produced "def fib(n), do: fib(n-1) + fib(n - 2" - correct
   except the missing ")" that did not fit). Naive chars/3 sizing caused 4 consecutive
   failed repairs. Fix: tokenize the replaced text via /tokenize for the TRUE length,
   then sweep {n, n+2, round(1.4n)} and take the first fill that COMPILES (compile is
   ~free vs a 400 ms infill). After the fix the same case heals in 2 rounds.
2. Fills can be MULTI-LINE: mask positions can become newline tokens (the healed fib
   came back as an if/do block spanning 3 lines inside a 1-line hole). Do not assume
   line-shaped fills when re-locating diagnostics.
3. Vary the seed per repair round (we use seed + round): identical canvas + identical
   seed re-produces the identical failed fill.
4. Tokenization detail: a fill that begins with a space-prefixed token after preserved
   indentation renders with one extra space (cosmetic in Elixir).
5. Elixir diagnostics ordering: Code.with_diagnostics logged diagnostics carry the
   precise message+line while the rescued CompileError is just "cannot compile module" -
   surface the logged ones first or the masker aims at nothing.
