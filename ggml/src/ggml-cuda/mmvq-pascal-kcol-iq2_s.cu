#include "mmvq-pascal-kcol.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "mmvq-pascal-kcol-impl.cuh"

namespace kcol {
// the kernels copy ggml's own device table of the grid into shared memory
template <> struct grid_src<KIQ2_S> {
    static __device__ __forceinline__ const uint32_t * get(const args &) { return (const uint32_t *) iq2s_grid; }
};
} // namespace kcol

void ggml_cuda_kcol_launch_iq2_s(const kcol::params & p, cudaStream_t stream) {
    kcol::launch<kcol::KIQ2_S>(p, stream);
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
