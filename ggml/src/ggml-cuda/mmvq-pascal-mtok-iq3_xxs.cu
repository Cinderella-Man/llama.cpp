#include "mmvq-pascal-mtok-iq.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

MMVQP_MTOK_IQ_DECL(iq3_xxs) {
    mtok_iq::launch<GGML_TYPE_IQ3_XXS>(p, fusion, grid, smem, gate, su, ntm, stream);
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
