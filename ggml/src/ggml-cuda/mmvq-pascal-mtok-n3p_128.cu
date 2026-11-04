#include "mmvq-pascal-mtok-np.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

MMVQP_MTOK_NP_DECL(n3p_128) {
    mtok_np::launch<GGML_TYPE_N3P_128>(p, fusion, grid, smem, gate, su, ntm, stream);
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
