#include "mmvq-pascal-kcol.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "mmvq-pascal-kcol-impl.cuh"

void ggml_cuda_kcol_launch_iq4_nl(const kcol::params & p, cudaStream_t stream) {
    kcol::launch<kcol::KIQ4_NL>(p, stream);
}

int ggml_cuda_kcol_blocks_iq4_nl(const int ncols, const int R) {
    return kcol::kernel_blocks<kcol::KIQ4_NL>(ncols, R);
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
