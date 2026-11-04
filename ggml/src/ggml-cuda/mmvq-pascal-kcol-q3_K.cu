#include "mmvq-pascal-kcol.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "mmvq-pascal-kcol-impl.cuh"

void ggml_cuda_kcol_launch_q3_K(const kcol::params & p, cudaStream_t stream) {
    kcol::launch<kcol::KQ3_K>(p, stream);
}

int ggml_cuda_kcol_blocks_q3_K(const int ncols, const int R) {
    return kcol::kernel_blocks<kcol::KQ3_K>(ncols, R);
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
