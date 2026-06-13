# Correctness validation of diffusion_generate_infill_batch

The engine function decodes N infill canvases in one batched forward (seq_id = canvas
index), mirroring diffusion_generate's infill config per-seq. Validated on Dream-7B
Q4_K_M, 5070, infill config (steps 16, conf_threshold 0.9, early_commit 0.5, temp 0.2,
top_k 40, backend sampling).

## Results

| test | result | meaning |
|---|---|---|
| **N=1 batched vs `/generate` infill** (6 canvases, masks 3-8) | **6/6 byte-identical** | the function's LOGIC is bit-identical to diffusion_generate for a single canvas |
| N=3 batched vs 3× single (12 fills across 4 groupings) | 8/12 identical (4 differ) | multi-seq divergence |
| argmax (top_k=1) N=4 | 3/4 identical | divergence persists without RNG |
| temp0.2 divergences / 4 canvases, 5 seeds | [1,0,1,1,0], avg 0.6 | ~15% of canvases flip one token |

## Diagnosis: logically exact, NOT bit-exact (multi-seq FP noise)

- **N=1 = 6/6 exact proves the per-canvas logic, indexing, and per-seq sampler are
  correct** — a single-seq run of the batched function reproduces diffusion_generate
  byte-for-byte.
- The N>=2 divergence persists even at **top_k=1 (argmax, RNG removed)**, so it is NOT
  a sampler-RNG-ordering artifact — it is **floating-point reduction-order noise**:
  batching multiple seqs changes the GEMM tiling, so logits differ in the last bits and
  near-tie positions flip argmax. Each flip is an *equally valid* token (the divergent
  fills are well-formed, not garbage).
- This is the SAME class of non-determinism already documented in this repo: Probe 0's
  m-tier scored 2/3 on re-run vs the doc's 1/3 at the *same seed* (GPU non-determinism).

## Consequence for the spec's "lossless / bit-exact" claim
Too strong. Multi-seq GPU batching is never bit-identical to single-row decode. The
lever is **near-lossless**: ~15% of canvases differ by one near-tie token, concentrated
on low-confidence positions (successful repairs commit high-confidence tokens, so the
divergence lands mostly on failing repairs that fail either way). Whether this shifts
the bench PASS COUNT is the empirical gate (±2 band). Measured next in the A/B.
