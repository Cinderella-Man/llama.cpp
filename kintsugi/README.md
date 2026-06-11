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
  cards (CUDA 13 cannot compile for them - see `../docs/p106-mining-fleet.md`).
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

    # instruction -> compiling code (measured: 1.4 s on an RTX 5070 laptop)
    iex> {:ok, code, stats} = Kintsugi.forge(eng, "a function double/1 that doubles a number")

    # broken code -> verified code (measured: 1.5 s, 2 repair rounds)
    iex> broken = "defmodule Fib do\n  def fib(0), do: 0\n  def fib(1), do: 1\n  def fib(n), do fib(n - 1) + fib(n - 2)\nend"
    iex> {:ok, fixed, stats} = Kintsugi.heal(eng, broken, %{"check" => "2 = Fib.fib(3)"})

## API

- `Kintsugi.forge(engine, instruction, opts)` - draft then repair until it
  verifies. Returns `{:ok, code, stats}` or `{:error, reason, stats}`. On
  success `stats` includes throughput: `tokens` (the tokenized FINAL answer
  only - drafts that got repaired over and discarded hole-size variants spend
  time but are never counted, so nothing is double counted), `ms_wall`
  (wall-clock incl. compile checks; `ms_total` is engine-side generation time)
  and `tokens_per_second = tokens / ms_wall`.
- `Kintsugi.heal(engine, code, opts)` - repair EXISTING code (no drafting).
- `Kintsugi.verify(code, check)` - compile + optional functional check.
- `Kintsugi.repair(engine, code, diagnostics, opts)` - one mask+infill round.

Useful opts (string keys, passed through to the server): `"seed"`, `"steps"`,
`"conf_threshold"`, `"temp"`, `"top_k"`, `"max_repairs"` (default 4), and
`"check"` - a string of assertions like `"4 = Mod.f(2)"`, executed in an
ISOLATED OS process with a timeout (generated code never runs in the harness VM).

## Hard-won knowledge (details in ../docs/dllm-elixir-harness.md)

- Masked diffusion fills EXACTLY the hole - an undershot hole truncates perfectly
  valid code mid-expression. Size holes by tokenizing the replaced text
  (`/tokenize`), not by characters, and sweep {n, n+2, 1.4n}; first compiling
  fill wins (`Kintsugi.repair/4` does all of this).
- Fills can be multi-line: mask positions can become newline tokens.
- Vary the seed per repair round or identical canvases re-fail identically.
- One server endpoint can be a whole multi-GPU rig (`--diffusion-replicas`) -
  the harness only needs multiple endpoints for multiple HOSTS.
