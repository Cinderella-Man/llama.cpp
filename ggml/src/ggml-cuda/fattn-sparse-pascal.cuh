#pragma once

#include "common.cuh"

// Sparse decode attention for GPUs without Turing tensor cores (the P106-090 and other Pascal cards): FLASH_ATTN_EXT
// with the n_kv_max hint, as qwen4exp's 12 QSA layers call it, reads only the cells the mask keeps (the indexer's
// ~2,051) instead of the whole cache. Upstream's gather of those cells exists only in the Turing MMA kernel, so below
// Turing every layer reads every cell: 64 MiB per layer at 32k (F16), against 4 MiB here. The kernel and its design are
// in fattn-sparse-pascal-impl.cuh; this file holds the conditions, the scratch and the switch.
//
// It takes: an F32 query of head size 128 or 256 (K and V the same), 4, 8, 12 or 16 query heads per KV head, F16 or
// Q8_0 K and V (both the same type), an F16 mask, n_kv_max > 0, at most GGML_CUDA_FATTN_SPARSE_PASCAL_NQ query tokens
// (default 8: decode and verify batches; prompts keep the dense kernels), optional sinks, no ALiBi or softcap, no
// kv_rows. Anything else stays with the stock kernels, as does any GPU with Turing tensor cores in mode 1.
//
// GGML_CUDA_FATTN_SPARSE_PASCAL: 0 off (default), 1 on GPUs without Turing tensor cores (compute capability below 7.5, or
// a build without Turing code) once the cache holds at least _MIN_RATIO x n_kv_max cells per query token (each token
// reads its own list, the dense kernels read the cache once for all of them), 2 any NVIDIA GPU at any cache size (for
// testing).
// GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO: cells per n_kv_max and query token from which mode 1 takes a call (default
// 1.5).
// GGML_CUDA_FATTN_SPARSE_PASCAL_NQ: most query tokens per sequence (default 8).
// GGML_CUDA_FATTN_SPARSE_PASCAL_SPLITS: splits per list (default: about 4 blocks per SM, at least 64 cells a split).
// GGML_CUDA_FATTN_SPARSE_PASCAL_NSM: SM count the splits assume (default: the device's; 5 emulates a P106-090).
// GGML_CUDA_FATTN_SPARSE_PASCAL_FAULT (testing the tests): 1 output x 1.01, 2 mask values ignored, 3 every split's last
// list entry dropped.

// Runs the node and returns true if the path takes it; returns false (and does nothing) otherwise.
bool ggml_cuda_flash_attn_ext_sparse_pascal(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
