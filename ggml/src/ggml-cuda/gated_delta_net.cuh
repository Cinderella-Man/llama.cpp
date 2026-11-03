#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

// fused-kernel recurrent-state input: each sequence's initial state read straight from the cache row that the
// skipped get_rows would have gathered (see ggml_cuda_try_gdn_state_gather_fusion)
struct ggml_cuda_gated_delta_net_state_rows {
    const float *   data;       // cache row 0
    const int32_t * ids;        // the cache row of each sequence
    int64_t         row_stride; // between cache rows, in elements
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion);
// with rows it also reads the initial states from the cache instead of src[5]
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache,
                                              const ggml_cuda_gated_delta_net_state_rows * rows = nullptr);
