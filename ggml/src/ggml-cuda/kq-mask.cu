#include "kq-mask.cuh"
#include "convert.cuh"

// llama_hparams::is_masked_swa, with swa_type numbered as llama_swa_type
static __device__ __forceinline__ bool kq_mask_swa_masked(const int swa_type, const int n_swa, const int p0, const int p1) {
    switch (swa_type) {
        case 1: // standard
            return p1 - p0 >= n_swa;
        case 2: // chunked
            return p0 < (p1 / n_swa) * n_swa;
        case 3: // symmetric
            return p1 - p0 < -(n_swa / 2) || p1 - p0 > n_swa / 2;
        default:
            return false;
    }
}

// One block row per token row (blockIdx.y = token i of stream blockIdx.z), threads over the KV cells; the token's
// position and sequence bit are read once per thread, each cell's position and sequence bits once.
template <typename T>
static __global__ void kq_mask_kernel(
        const int * __restrict__ cells, const int * __restrict__ toks, T * __restrict__ dst,
        const int n_kv, const int n_tps, const int64_t nb1, const int64_t nb3, const int swa_type, const int n_swa) {
    const int j = blockIdx.x*blockDim.x + threadIdx.x;
    const int i = blockIdx.y;
    const int s = blockIdx.z;
    if (j >= n_kv) {
        return;
    }
    const int * cpos = cells + 2*(int64_t) n_kv*s;
    const int * tpos = toks  + 2*(int64_t) n_tps*s;
    const int p0 = cpos[j];
    const int p1 = tpos[i];
    const bool same_seq = ((unsigned int) cpos[n_kv + j] >> tpos[n_tps + i]) & 1;
    const bool keep = same_seq && p0 <= p1 && !kq_mask_swa_masked(swa_type, n_swa, p0, p1);
    T * row = (T *) ((char *) dst + i*nb1 + s*nb3);
    row[j] = ggml_cuda_cast<T>(keep ? 0.0f : -INFINITY);
}

void ggml_cuda_op_kq_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * cells = dst->src[0];
    const ggml_tensor * toks  = dst->src[1];
    GGML_ASSERT(cells->type == GGML_TYPE_I32 && toks->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(cells) && ggml_is_contiguous(toks));

    const int n_kv     = (int) dst->ne[0];
    const int n_tps    = (int) dst->ne[1];
    const int n_stream = (int) dst->ne[3];
    const int swa_type = ggml_get_op_params_i32(dst, 0);
    const int n_swa    = ggml_get_op_params_i32(dst, 1);
    GGML_ASSERT(n_tps <= 65535 && n_stream <= 65535);

    const dim3 grid((n_kv + CUDA_KQ_MASK_BLOCK_SIZE - 1) / CUDA_KQ_MASK_BLOCK_SIZE, n_tps, n_stream);
    cudaStream_t stream = ctx.stream();
    if (dst->type == GGML_TYPE_F16) {
        kq_mask_kernel<half><<<grid, CUDA_KQ_MASK_BLOCK_SIZE, 0, stream>>>(
            (const int *) cells->data, (const int *) toks->data, (half *) dst->data, n_kv, n_tps, dst->nb[1], dst->nb[3], swa_type, n_swa);
    } else {
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        kq_mask_kernel<float><<<grid, CUDA_KQ_MASK_BLOCK_SIZE, 0, stream>>>(
            (const int *) cells->data, (const int *) toks->data, (float *) dst->data, n_kv, n_tps, dst->nb[1], dst->nb[3], swa_type, n_swa);
    }
}
