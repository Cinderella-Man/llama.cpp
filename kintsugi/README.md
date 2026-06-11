# Kintsugi

Verify-and-repair code generation on diffusion LLMs, wrapping this repo's
`llama-diffusion-server`. Named for the art of repairing pottery with gold: the
compiler finds the cracks, the diffusion model fills them, the fixed seams are
where the value is.

Zero dependencies: `:httpc` + the built-in `JSON` module (Elixir >= 1.18).

## The loop

    draft (confidence-threshold decode, ~a dozen forward passes)
      -> compile (Code.with_diagnostics - deterministic, line-accurate)
      -> mask the offending span (the model's mask piece, hole sized by /tokenize)
      -> infill ONLY the hole (fixed text stays byte-identical)
      -> recompile -> repeat

## Use

    # start an engine (one server can already be a whole multi-GPU rig):
    #   llama-diffusion-server -m Dream-7B-Q4_K_M.gguf -ub 512 -ngl 99 \
    #       --diffusion-eps 0.001 --diffusion-steps 128 --temp 0.2 --top-k 40

    {:ok, eng} = Kintsugi.Engine.connect("http://127.0.0.1:8080")

    # instruction -> compiling code (measured: 1.4 s on an RTX 5070 laptop)
    {:ok, code, stats} = Kintsugi.forge(eng, "a function double/1 that doubles a number")

    # broken code -> verified code (measured: 1.5 s, 2 repairs)
    {:ok, fixed, _} = Kintsugi.heal(eng, broken_code, %{"check" => "2 = Fib.fib(3)"})

`forge/3` drafts then repairs; `heal/3` repairs existing code; both accept
`"check" => "4 = Mod.f(2)"` for functional verification (runs in an isolated OS
process, never in the harness VM) and `"max_repairs"`.

## Test

    mix test                                    # offline unit tests
    KINTSUGI_ENGINE=http://127.0.0.1:8080 \
        mix test --include engine               # + live integration

## Hard-won knowledge (also in ../docs/dllm-elixir-harness.md)

- Masked diffusion fills EXACTLY the hole - an undershot hole truncates perfectly
  valid code mid-expression. Size holes by tokenizing the replaced text
  (`/tokenize`), not by characters, and sweep {n, n+2, 1.4n}, first compiling
  fill wins (Kintsugi.repair/4 does this).
- Fills can be multi-line: mask positions can become newline tokens.
- Vary the seed per repair round or identical canvases re-fail identically.
