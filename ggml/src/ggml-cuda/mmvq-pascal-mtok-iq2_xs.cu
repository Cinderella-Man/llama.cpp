#include "mmvq-pascal-mtok-iq.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

MMVQP_MTOK_IQ_DECL(iq2_xs) {
    mtok_iq::launch<GGML_TYPE_IQ2_XS>(p, fusion, grid, smem, gate, su, ntm, stream);
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
