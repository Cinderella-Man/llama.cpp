# Kintsugi

Verify-and-repair code generation on diffusion LLMs, wrapping this repo's
`llama-diffusion-server`. Named for the art of repairing pottery with gold: the
compiler finds the cracks, the diffusion model fills them, the fixed seams are
where the value is.

Zero runtime dependencies: `:httpc` + the built-in `JSON` module.

## The loop

    draft (confidence-threshold decode, ~a dozen forward passes)
      -> compile (Code.with_diagnostics - deterministic, line-accurate)
      -> mask the offending span (the model's mask piece, hole sized by /tokenize)
      -> infill ONLY the hole (fixed text stays byte-identical)
      -> recompile -> repeat

## Running it from scratch

### 0. Prerequisites

- Elixir >= 1.18 (built-in `JSON`; this project is developed on 1.19 / OTP 28)
- CMake + a C++ compiler
- For GPU speed (strongly recommended - this is the point of the fork): an NVIDIA
  GPU with >= 6 GB VRAM and the CUDA toolkit. Note the toolkit/hardware pairing:
  CUDA 13.x for Turing and newer, CUDA 12.x if you need to target Pascal mining
  cards (CUDA 13 cannot compile for them - see `../docs/dllms/p106-mining-fleet.md`).
  CPU-only works for trying things out, just slower per step.

### 1. Build llama.cpp (the parent directory of this project)

    cd ..                                   # the llama.cpp checkout
    cmake -B build -DGGML_CUDA=ON           # drop -DGGML_CUDA=ON for CPU-only
    cmake --build build -j --target llama-diffusion-server llama-diffusion-cli

### 2. Get a masked-diffusion model (GGUF)

Kintsugi needs the "masked" family (a model with a mask token): Dream, LLaDA, or
DiffuCoder. Dream-7B-Instruct at Q4_K_M (~4.7 GB, fits a 6 GB card) is the tested
default:

    # any of the community GGUF conversions works, e.g.:
    huggingface-cli download bartowski/Dream-org_Dream-v0-Instruct-7B-GGUF \
        Dream-org_Dream-v0-Instruct-7B-Q4_K_M.gguf --local-dir ~/models/dream7b

(DiffusionGemma also works through the same server but is the "canvas" family -
`forge` works, `heal`/infill does not yet, and it needs ~20 GB host RAM with
`--cpu-moe`.)

Fast-dLLM v2 (block-AR family, e.g. the 1.5B at 986 MB Q4_K_M) is also served -
it out-drafts Dream-7B on code at ~3x the speed (add `--diffusion-block-kv` for
flat ~10 ms steps; GPU sampling is on by default and takes them to ~8.4 ms /
191 tok/s raw), BUT block-causal attention cannot see text after a hole:
infill/`heal` is structurally unsupported, so recovery is whole-draft retry only.
Use `conf_threshold` 0.9 with it (0.6 is Dream's scale and corrupts block-AR
output - thresholds are model-scale-specific).

### 3. Start the engine

    ./build/bin/llama-diffusion-server \
        -m ~/models/dream7b/Dream-org_Dream-v0-Instruct-7B-Q4_K_M.gguf \
        -ub 512 -ngl 99 \
        --diffusion-eps 0.001 --diffusion-steps 128 \
        --temp 0.2 --top-k 40

    # sanity check - should report family "masked" and a mask_piece:
    curl -s http://127.0.0.1:8080/health

Flag notes: `-ngl 99` = all layers on GPU (omit for CPU-only); `-ub 512` = max
canvas/draft length; the `--diffusion-*`/`--temp`/`--top-k` values are the
defaults every request inherits (override per request). GPU sampling is on by
default. On a multi-GPU box add `--diffusion-replicas 0` (one replica per GPU,
parallel requests) and `--host`/`--port` to taste.

### 4. Run kintsugi

    cd kintsugi
    mix test                                    # 10 offline tests, no server needed
    KINTSUGI_ENGINE=http://127.0.0.1:8080 \
        mix test --include engine               # + 3 live integration tests

    # interactive:
    iex -S mix

    iex> {:ok, eng} = Kintsugi.Engine.connect("http://127.0.0.1:8080")

    # instruction -> verified code, one call (measured: 1.4 s on an RTX 5070 laptop)
    iex> {:ok, code} = Kintsugi.generate(eng, "a function double/1 that doubles a number")

    # broken code -> verified code (measured: 1.5 s, 2 repair rounds)
    iex> broken = "defmodule Fib do\n  def fib(0), do: 0\n  def fib(1), do: 1\n  def fib(n), do fib(n - 1) + fib(n - 2)\nend"
    iex> {:ok, fixed, stats} = Kintsugi.heal(eng, broken, %{"check" => "2 = Fib.fib(3)"})

## API

THE function - everything else is plumbing it uses:

- `Kintsugi.generate(engine, instruction, opts)` -> `{:ok, code} | {:error, reason}`.
  The caller asks for code and receives VERIFIED code; drafting, compile checks,
  masked repairs, escalation (wider holes, whole-body remask) and even fresh
  redrafts when a draft is unhealable all happen invisibly behind this call.
- `Kintsugi.generate_with_stats(engine, instruction, opts)` - same, returning
  `{:ok, code, stats}`: drafts, repairs, history, `ms_wall` (wall-clock;
  `ms_total` is engine-side generation time), and throughput - `tokens` is the
  tokenized FINAL answer only (discarded drafts and failed fills spend time but
  never count as tokens, so nothing is double counted) and
  `tokens_per_second = tokens / ms_wall`.

Plumbing (public for direct use, e.g. healing human-written code):

- `Kintsugi.heal(engine, code, opts)` - repair EXISTING code (no drafting).
- `Kintsugi.forge(engine, instruction, opts)` - one draft + its repair loop.
- `Kintsugi.verify(code, check)` - compile + optional functional check.
- `Kintsugi.repair(engine, code, diagnostics, opts)` - one mask+infill round.
- `Kintsugi.generate_hybrid(draft_engine, repair_engine, instruction, opts)` -
  two-engine ladder (e.g. Fast-dLLM drafts + Dream repairs). Measured and
  REJECTED as a default (same pass set as Dream-only at half the throughput,
  see ../docs/dllms/throughput-plans/04_layer_d.md), kept for experiments.

Compilation alone accepts semantically wrong code - pass `"check"` (assertions,
e.g. `"9 = Mod.triple(3)"`) when correctness matters; generate/forge/heal all
honor it.

Useful opts (string keys, passed through to the server): `"seed"`, `"steps"`,
`"conf_threshold"`, `"temp"`, `"top_k"`, `"max_repairs"` (default 4), and
`"check"` - a string of assertions like `"4 = Mod.f(2)"`, executed in an
ISOLATED OS process with a timeout (generated code never runs in the harness VM).
The kill is OS-level (`timeout -k`): a BEAM-side `Task.shutdown` alone orphans
the child, and runaway generated code in orphans OOM-killed this machine twice
before that was understood - do not "simplify" it away.

## Hard-won knowledge (details in ../docs/dllms/dllm-elixir-harness.md)

- Masked diffusion fills EXACTLY the hole - an undershot hole truncates perfectly
  valid code mid-expression. Size holes by tokenizing the replaced text
  (`/tokenize`), not by characters, and sweep {n, n+2, 1.4n}; first compiling
  fill wins (`Kintsugi.repair/4` does all of this).
- Fills can be multi-line: mask positions can become newline tokens.
- Vary the seed per repair round or identical canvases re-fail identically.
- One server endpoint can be a whole multi-GPU rig (`--diffusion-replicas`) -
  the harness only needs multiple endpoints for multiple HOSTS.

## Bench v2 protocol

The bench is the project's measuring instrument - every optimization claim goes
through it. Design + empirical calibration: ../docs/dllms/dllm-elixir-harness-measuring-updates.md.

    mix run bench/cases.exs          # self-test: every check vs its reference solution
    mix run bench/bench.exs URL LABEL PROFILE [DRAFT_URL]
    mix run bench/compare.exs old.jsonl new.jsonl   # gated diff, exit 1 on regression

Profiles (see `@profiles` in bench/bench.exs for the full, current set): `baseline`,
`kvpfx32`, `ec05`, `remask03`, `win64`, `grow`, `big384`, `mh2`, `winroute`,
`fastdllm` (block-AR, threshold 0.9), `e3kv`/`e4bs`/`e5*` (block-AR kv + GPU-sampling
A/B family, see 05_layer_e.md), `d4` (two-engine hybrid; pass the draft
server as DRAFT_URL). The C5 slim/mid profiles were removed from the runnable set -
read 03_layer_c.md before considering re-adding them.

Rules (calibrated, not guessed):
- AC power required (the runner refuses on battery; measured 5x skew).
- Same-seed outcomes are deterministic: identical-config reruns must match exactly.
- A/B configs run against ONE server process (profiles are request params - walls are
  only comparable within a process; cross-restart tails reach +138% on marginal cases).
- Regression = pass lost in p/h/i tiers, or tier median wall +10%.
- Baselines: bench/results/baselines/*.jsonl (committed; per-model).
- ONE bench at a time, never concurrently. The runner protects the machine itself:
  before every case it sweeps stray `kintsugi_run_` processes and ABORTS (exit 75,
  partial results preserved) if available RAM drops under 4 GB.
