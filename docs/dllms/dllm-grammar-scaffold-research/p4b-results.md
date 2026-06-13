# P4b - grammar-constrained diffusion decode (the build the user requested)

Built a tractable frontier realization of grammar-constrained decode in the engine and
measured it. Engine diff: `p4b-engine.diff`. Probe: `p4b_grammar_infill.exs`. Samples:
`p4b-samples.jsonl`. Dream-7B, RTX 5070, 2026-06-13.

## What was built

`examples/diffusion/diffusion.cpp` (+ `diffusion.h`, `diffusion-server.cpp`): when a
`grammar` (GBNF) is supplied AND the request is infill (canvas is pure code), each
denoising step (CPU sampling path):
1. finds the committed-prefix frontier = the first masked position (its entire prefix is
   committed);
2. builds a fresh grammar sampler, accepts the committed prefix [0, frontier);
3. masks the frontier position's candidates to grammar-valid tokens before sampling.
A latch disables the grammar for the rest of a run once the committed canvas drifts outside
the grammar (otherwise `llama_sampler_accept` throws "empty grammar stack"). Plumbed via a
`grammar` request param. `grammar_str == nullptr` => byte-identical to before.

This is the tractable realization, NOT the full eth-sri any-order CFG-intersection: only the
left-coherent frontier is constrained; out-of-order commits beyond it are free; no
suffix-aware lookahead (LAVE).

## Two decisive implementation findings

1. **Grammar is not wired into the GPU backend sampler** - only the CPU sampling path.
   Production defaults to `backend_sampling=true` (GPU). Using a grammar therefore forces
   the slow CPU path on top of the grammar's own cost.
2. **Subset-strictness crashes the decode.** First (un-latched) run: 15/30 requests
   returned HTTP 500 `Unexpected empty grammar stack after accepting piece` - the committed
   canvas drifted outside the hand-written subset grammar and `accept` threw. This is the
   P3 false-reject risk manifesting at decode time. The latch turns the crash into a silent
   fallback, but then the grammar barely engages.

## A/B result (latched, fair - both arms CPU sampling, identical canvas/seed)

| arm                | pass | parse-fail | compile | check | latency (sum) |
|--------------------|------|-----------|---------|-------|---------------|
| grammar OFF (CPU)  | 7/30 | 9         | 9       | 5     | 12.8 s        |
| grammar ON         | 4/30 | 10        | 11      | 5     | 267 s (20.9x) |

- Grammar **hurt** pass (7 -> 4), did **not** reduce parse failures (9 -> 10), changed only
  4/30 outcomes, and was **~21x slower**.
- Un-latched variant (for the record): grammar engaged more (parse 9 -> 5, it DID cut
  syntax errors) but tanked pass to 1/30 and errored on 15/30. So when the grammar actually
  bites it cuts syntax but destroys correctness; when made robust (latched) it barely bites.
  Both ends lose.

## Why it loses (consistent with P0/P1/P1c/P3)

- Constraining only the frontier (or forcing odd in-grammar tokens) pushes the model into
  worse completions - same quality damage scaffold caused (P1c), now with 21x the cost.
- A hand-written subset grammar is too narrow: real Dream output drifts out of it
  constantly, so the constraint either crashes or latches off. A full Elixir CFG would
  reduce drift but is very hard to write and partly not CFG-expressible (P3), and even full
  engagement tanks correctness.
- The full-vocab (151k) grammar mask per step + per-step rebuild + loss of GPU sampling =
  ~21x latency, fatal for a throughput-oriented stack.

## Verdict

Grammar-constrained diffusion decode is a **decisive loss** on this stack: pass-negative,
not even reliably syntax-reducing once made robust, and ~21x slower. Combined with P1c
(scaffold -9 pass, -47% tok/s) the syntactic-constraint line is closed for Dream-Elixir.

Scope caveat (unchanged): this is Dream-7B + a small Elixir bench. The published +7%
(Muendler et al.) used DiffuCoder/LLaDA on C++/JSON/multi-region infilling with a full
CFG-intersection algorithm. The one branch NOT tested is reproducing that exact setting
(DiffuCoder-7B is on disk). Our negative is for THIS stack and this (tractable) method.
