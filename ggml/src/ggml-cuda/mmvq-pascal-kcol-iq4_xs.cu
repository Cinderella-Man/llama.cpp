#include "mmvq-pascal-kcol.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "mmvq-pascal-kcol-impl.cuh"

void ggml_cuda_kcol_launch_iq4_xs(const kcol::params & p, cudaStream_t stream) {
    kcol::launch<kcol::KIQ4_XS>(p, stream);
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
