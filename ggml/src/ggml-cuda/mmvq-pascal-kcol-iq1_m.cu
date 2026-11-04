#include "mmvq-pascal-kcol.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "mmvq-pascal-kcol-impl.cuh"

namespace kcol {
// the kernels copy ggml's own device table of the grid into shared memory
template <> struct grid_src<KIQ1_M> {
    static __device__ __forceinline__ const uint32_t * get(const args &) { return (const uint32_t *) iq1s_grid_gpu; }
};
} // namespace kcol

void ggml_cuda_kcol_launch_iq1_m(const kcol::params & p, cudaStream_t stream) {
    kcol::launch<kcol::KIQ1_M>(p, stream);
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
