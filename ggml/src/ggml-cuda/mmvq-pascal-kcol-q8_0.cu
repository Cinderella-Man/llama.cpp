#include "mmvq-pascal-kcol.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "mmvq-pascal-kcol-impl.cuh"

void ggml_cuda_kcol_launch_q8_0(const kcol::params & p, cudaStream_t stream) {
    kcol::launch<kcol::KQ8_0>(p, stream);
}

int ggml_cuda_kcol_blocks_q8_0(const int ncols, const int R) {
    return kcol::kernel_blocks<kcol::KQ8_0>(ncols, R);
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
