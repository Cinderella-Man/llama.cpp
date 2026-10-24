#pragma once

#include "common.cuh"

// One-token decode attention for compute capability 6.1 (Pascal with dp4a, the 5-SM P106-090): FLASH_ATTN_EXT with an
// F32 query of head size 128, F16 K and V, 8 query heads per KV head, an F16 mask and optional sinks, as Flint decodes.
// The stock TILE kernel gives such a call two launches and reads every key once per tile row; this one reads each key
// and value row once for all 8 query heads of its KV head and merges the key splits in the last block of the same
// launch:
//   - a block has one warp per KV head (4), all on the same key range; the keys are split over blockIdx.x;
//   - lane l owns dimensions 4l..4l+3 of all 8 heads: the query, pre-scaled by scale*log2(e), and the V accumulator
//     live in registers, and K and V are converted once per key;
//   - scores of 4 keys x 8 heads are summed over the lanes with one transpose-reduce, an online softmax in base 2 runs
//     per 32-key tile, and a tile whose mask is -inf for all its keys is skipped;
//   - every block writes its partial output, maximum and sum; the last one to finish (a counter) merges them, adds the
//     sinks and writes the result.
// Results match the stock kernels to rounding (FP32 throughout, a different summation order), not bit for bit.
// GGML_CUDA_FATTN_PASCAL: 0 off, 1 compute capability 6.1 only (default), 2 any NVIDIA GPU (for testing).
// Returns true if it ran the call; false leaves it to the stock kernels.
bool ggml_cuda_flash_attn_ext_pascal(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
