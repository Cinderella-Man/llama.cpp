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

Compilation alone accepts semantically wrong code - pass `"check"` (assertions,
e.g. `"9 = Mod.triple(3)"`) when correctness matters; generate/forge/heal all
honor it.

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
