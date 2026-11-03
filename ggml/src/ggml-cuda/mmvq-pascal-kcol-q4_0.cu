#include "mmvq-pascal-kcol.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "mmvq-pascal-kcol-impl.cuh"

void ggml_cuda_kcol_launch_q4_0(const kcol::params & p, cudaStream_t stream) {
    kcol::launch<kcol::KQ4_0>(p, stream);
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
