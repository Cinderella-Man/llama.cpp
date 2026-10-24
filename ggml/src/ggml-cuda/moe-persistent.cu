#include "moe-persistent.cuh"

#include <cstdlib>

// Layer shapes the kernel is built for (Flint).
#define MOEP_H   2048 // hidden size
#define MOEP_NE  256  // routed experts
#define MOEP_KU  6    // experts used per token
#define MOEP_F   512  // routed expert width
#define MOEP_FS  1024 // shared expert width

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

namespace {

// Counters and flags at the start of the context's buffer (32-bit words), as in bench/persistent_pascal.
constexpr int S_NEXT    = 0;                 // next work item
constexpr int S_RT_DONE = 1;                 // router items done
constexpr int S_TOPK    = 2;                 // top-k published
constexpr int S_SG_DONE = 3;                 // shared gate/up items done
constexpr int S_SACT    = 4;                 // shared activations quantized
constexpr int S_EG_DONE = 5;                 // + e: routed gate/up items of expert slot e done
constexpr int S_EACT    = S_EG_DONE + MOEP_KU; // + e: activations of expert slot e quantized
constexpr int S_EXIT    = S_EACT + MOEP_KU;  // warps that have left
constexpr int S_COMB    = 32;                // + g: contributions to 8-row output group g
constexpr int N_SYNC    = S_COMB + MOEP_H / 8;

// The scratch of one layer after the counters, in bytes from the start of the buffer (16-byte aligned).
struct moep_scratch {
    static constexpr size_t sync    = 0;
    static constexpr size_t logits  = (N_SYNC * sizeof(unsigned) + 15) / 16 * 16;           // float [NE]
    static constexpr size_t ids     = logits  + MOEP_NE * sizeof(float);                    // int32 [NE]
    static constexpr size_t weights = ids     + MOEP_NE * sizeof(int32_t);                  // float [KU], padded
    static constexpr size_t act     = weights + 16 * sizeof(float);                         // float [KU][F]
    static constexpr size_t aq      = act     + MOEP_KU * MOEP_F * sizeof(float);           // block_q8_1 [KU][F/32]
    static constexpr size_t dexp    = aq      + MOEP_KU * MOEP_F / QK8_1 * sizeof(block_q8_1); // float [KU][H]
    static constexpr size_t sact    = dexp    + MOEP_KU * MOEP_H * sizeof(float);           // float [FS]
    static constexpr size_t sq      = sact    + MOEP_FS * sizeof(float);                    // block_q8_1 [FS/32]
    static constexpr size_t sd      = sq      + MOEP_FS / QK8_1 * sizeof(block_q8_1);       // float [H]
    static constexpr size_t check   = sd      + MOEP_H * sizeof(float);                     // float [H]: check mode
    static constexpr size_t size    = check   + MOEP_H * sizeof(float);
};

static int moep_mode() {
    static const int mode = [] {
        const char * env = getenv("GGML_CUDA_PERSISTENT_MOE");
        return env ? atoi(env) : 0;
    }();
    return mode;
}

static int moep_grid_override() {
    static const int grid = [] {
        const char * env = getenv("GGML_CUDA_PERSISTENT_MOE_GRID");
        return env ? atoi(env) : 0;
    }();
    return grid;
}

static bool moep_aligned(const ggml_tensor * t, const size_t al) {
    return (uintptr_t) t->data % al == 0 && t->nb[1] % al == 0 && t->nb[2] % al == 0 && t->nb[3] % al == 0;
}

// The layer's weight types: the (gate/up, down, shared) combinations that have a kernel. The router type (F16, BF16
// or F32) is a runtime branch inside the kernel.
struct moep_combo {
    ggml_type gu;
    ggml_type dn;
    ggml_type sh;
};

// TODO(Q88 phase 1): the kernels, from bench/persistent_pascal/src/persistent.cuh with the Pascal products of
// mmvq-pascal-impl.cuh: K2h-r16's four combinations first, then UD-Q4_K_XL's nine.
static constexpr moep_combo moep_combos[] = {
    { GGML_TYPE_COUNT, GGML_TYPE_COUNT, GGML_TYPE_COUNT }, // none yet
};

static int moep_find_combo(const ggml_type gu, const ggml_type dn, const ggml_type sh) {
    for (int c = 0; c < (int) (sizeof(moep_combos) / sizeof(moep_combos[0])); ++c) {
        if (moep_combos[c].gu == gu && moep_combos[c].dn == dn && moep_combos[c].sh == sh) {
            return c;
        }
    }
    return -1;
}

// Byte-wise comparison of the persistent kernel's output with the per-op path's (check mode).
static __global__ void k_moep_compare(const uint32_t * a, const uint32_t * b, const int n, unsigned * n_differ) {
    __shared__ int any;
    if (threadIdx.x == 0) {
        any = 0;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (a[i] != b[i]) {
            any = 1;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && any) {
        atomicAdd(n_differ, 1u);
    }
}

static void moep_alloc(ggml_backend_cuda_context & ctx) {
    auto & st = ctx.moe_persistent;
    if (st.buf != nullptr) {
        return;
    }
    ggml_cuda_set_device(ctx.device);
    // first use comes before any CUDA graph capture (ggml-cuda evaluates a graph uncaptured during its warmup), as for
    // the norm's Q8_1 buffer; the counters are zeroed in stream order before the first launch
    CUDA_CHECK(cudaMalloc(&st.buf, moep_scratch::size));
    CUDA_CHECK(cudaMemsetAsync(st.buf, 0, moep_scratch::size, ctx.stream()));
    CUDA_CHECK(cudaHostAlloc((void **) &st.err_host, 2 * sizeof(unsigned), cudaHostAllocMapped));
    st.err_host[0] = st.err_host[1] = 0;
    CUDA_CHECK(cudaHostGetDevicePointer((void **) &st.err_dev, st.err_host, 0));
}

} // namespace

bool ggml_cuda_moe_persistent_enabled(const int cc) {
    const int mode = moep_mode();
    return mode != 0 && GGML_CUDA_CC_IS_NVIDIA(cc) && (mode != 1 || cc == GGML_CUDA_CC_DP4A);
}

bool ggml_cuda_moe_persistent_check_mode() {
    static const bool check = [] {
        const char * env = getenv("GGML_CUDA_PERSISTENT_MOE_CHECK");
        return env && atoi(env) != 0;
    }();
    return check;
}

bool ggml_cuda_moe_persistent_supported(const ggml_cuda_moe_layer & L) {
    const ggml_tensor * x = L.x;
    if (x->type != GGML_TYPE_F32 || x->ne[0] != MOEP_H || ggml_nrows(x) != 1 || !ggml_is_contiguous(x) ||
            L.dst->type != GGML_TYPE_F32 || ggml_nelements(L.dst) != MOEP_H || !ggml_is_contiguous(L.dst) ||
            L.norm_w->type != GGML_TYPE_F32 || ggml_nelements(L.norm_w) != MOEP_H || !ggml_is_contiguous(L.norm_w)) {
        return false;
    }
    // the kernel reads x while it writes dst (x is read again by the residual add after the layer, so they never share
    // memory in a real graph)
    const char * x0 = (const char *) x->data;
    const char * d0 = (const char *) L.dst->data;
    if (x0 < d0 + ggml_nbytes(L.dst) && d0 < x0 + ggml_nbytes(x)) {
        return false;
    }
    const ggml_tensor * r = L.router;
    if ((r->type != GGML_TYPE_F16 && r->type != GGML_TYPE_BF16 && r->type != GGML_TYPE_F32) || r->ne[0] != MOEP_H ||
            r->ne[1] != MOEP_NE || !ggml_is_contiguous(r)) {
        return false;
    }
    const auto expert_shape = [](const ggml_tensor * t, const int64_t k, const int64_t rows) {
        return t->ne[0] == k && t->ne[1] == rows && t->ne[2] == MOEP_NE && t->ne[3] == 1 && ggml_is_contiguous(t);
    };
    const auto dense_shape = [](const ggml_tensor * t, const int64_t k, const int64_t rows) {
        return t->ne[0] == k && t->ne[1] == rows && t->ne[2] == 1 && t->ne[3] == 1 && ggml_is_contiguous(t);
    };
    if (!expert_shape(L.gate_exps, MOEP_H, MOEP_F) || !expert_shape(L.up_exps, MOEP_H, MOEP_F) ||
            !expert_shape(L.down_exps, MOEP_F, MOEP_H) || !dense_shape(L.gate_sh, MOEP_H, MOEP_FS) ||
            !dense_shape(L.up_sh, MOEP_H, MOEP_FS) || !dense_shape(L.down_sh, MOEP_FS, MOEP_H) ||
            L.gate_exps->type != L.up_exps->type || L.gate_sh->type != L.up_sh->type || L.gate_sh->type != L.down_sh->type) {
        return false;
    }
    // staged weight loads
    for (const ggml_tensor * t : { L.gate_exps, L.up_exps, L.down_exps, L.gate_sh, L.up_sh, L.down_sh }) {
        if (!moep_aligned(t, 16)) {
            return false;
        }
    }
    return moep_find_combo(L.gate_exps->type, L.down_exps->type, L.gate_sh->type) >= 0;
}

void ggml_cuda_op_moe_persistent(ggml_backend_cuda_context & ctx, const ggml_cuda_moe_layer & L) {
    moep_alloc(ctx);
    auto & st = ctx.moe_persistent;
    const bool check = ggml_cuda_moe_persistent_check_mode();
    float * out = check ? (float *) ((char *) st.buf + moep_scratch::check) : (float *) L.dst->data;
    GGML_UNUSED(out);
    GGML_UNUSED(moep_grid_override);

    // TODO(Q88 phase 1): launch the combination's kernel on ctx.stream() with the resident grid (occupancy API, once
    // per kernel and device) or GGML_CUDA_PERSISTENT_MOE_GRID
    GGML_ABORT("GGML_CUDA_PERSISTENT_MOE: no kernel for this layer");

    st.n_layers++;
    if (check) {
        st.check_dst = L.dst;
    }
}

void ggml_cuda_moe_persistent_check_node(ggml_backend_cuda_context & ctx, const ggml_tensor * node) {
    auto & st = ctx.moe_persistent;
    if (node != st.check_dst) {
        return;
    }
    k_moep_compare<<<1, 256, 0, ctx.stream()>>>(
        (const uint32_t *) ((const char *) st.buf + moep_scratch::check), (const uint32_t *) node->data, MOEP_H, st.err_dev + 1);
    CUDA_CHECK(cudaGetLastError());
    st.check_dst = nullptr;
    st.n_checked++;
}

void ggml_cuda_moe_persistent_begin_graph(ggml_backend_cuda_context & ctx) {
    auto & st = ctx.moe_persistent;
    if (st.err_host != nullptr && *(volatile unsigned *) &st.err_host[0] != 0) {
        GGML_ABORT("GGML_CUDA_PERSISTENT_MOE: a persistent MoE kernel timed out on %s (a bug); run with "
                   "GGML_CUDA_PERSISTENT_MOE=0 and report it", ctx.name.c_str());
    }
    st.n_layers  = 0;
    st.n_checked = 0;
    st.check_dst = nullptr;
}

void ggml_cuda_moe_persistent_end_graph(ggml_backend_cuda_context & ctx) {
    auto & st = ctx.moe_persistent;
    if (!ggml_cuda_moe_persistent_check_mode() || st.n_layers == 0) {
        return;
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    const unsigned n_differ = *(volatile unsigned *) &st.err_host[1];
    st.err_host[1] = 0;
    GGML_LOG_INFO("%s: %s: persistent MoE kernel ran %d layers, %d compared with the per-op path, %u differed\n",
                  __func__, ctx.name.c_str(), st.n_layers, st.n_checked, n_differ);
}

void ggml_cuda_moe_persistent_free(ggml_backend_cuda_context & ctx) {
    auto & st = ctx.moe_persistent;
    if (st.buf != nullptr) {
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaFree(st.buf));
        st.buf = nullptr;
    }
    if (st.err_host != nullptr) {
        CUDA_CHECK(cudaFreeHost(st.err_host));
        st.err_host = nullptr;
        st.err_dev  = nullptr;
    }
}

#else

bool ggml_cuda_moe_persistent_enabled(const int cc) {
    GGML_UNUSED(cc);
    return false;
}

bool ggml_cuda_moe_persistent_check_mode() {
    return false;
}

bool ggml_cuda_moe_persistent_supported(const ggml_cuda_moe_layer & layer) {
    GGML_UNUSED(layer);
    return false;
}

void ggml_cuda_op_moe_persistent(ggml_backend_cuda_context & ctx, const ggml_cuda_moe_layer & layer) {
    GGML_UNUSED_VARS(ctx, layer);
    GGML_ABORT("fatal error");
}

void ggml_cuda_moe_persistent_check_node(ggml_backend_cuda_context & ctx, const ggml_tensor * node) {
    GGML_UNUSED_VARS(ctx, node);
}

void ggml_cuda_moe_persistent_begin_graph(ggml_backend_cuda_context & ctx) {
    GGML_UNUSED(ctx);
}

void ggml_cuda_moe_persistent_end_graph(ggml_backend_cuda_context & ctx) {
    GGML_UNUSED(ctx);
}

void ggml_cuda_moe_persistent_free(ggml_backend_cuda_context & ctx) {
    GGML_UNUSED(ctx);
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
