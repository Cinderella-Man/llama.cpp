#include "mmvq-pascal-kcol.cuh"
#include "mmvq-pascal.cuh"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <string>
#include <vector>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "mmvq-pascal-kcol-impl.cuh"

namespace {

static int kcol_env_int(const char * name, const int def) {
    const char * env = getenv(name);
    return env ? atoi(env) : def;
}

static int kcol_mode() {
    static const int mode = kcol_env_int("GGML_CUDA_MMVQ_PASCAL_KCOL", 1);
    return mode;
}

struct kcol_overrides {
    int r;
    int itt;
    int smem;
    int nsm;
};

static const kcol_overrides & kcol_shape_env() {
    static const kcol_overrides o = {
        kcol_env_int("GGML_CUDA_MMVQ_PASCAL_KCOL_R", 0),
        kcol_env_int("GGML_CUDA_MMVQ_PASCAL_KCOL_ITT", 0),
        kcol_env_int("GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM", 0),
        kcol_env_int("GGML_CUDA_MMVQ_PASCAL_KCOL_NSM", 0),
    };
    return o;
}

// GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=q4_K,q6_K: weight types (ggml type names) left to the other kernels
static bool kcol_skip_type(const ggml_type type) {
    static const std::vector<ggml_type> skip = [] {
        std::vector<ggml_type> types;
        const char * env = getenv("GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP");
        const std::string list = env ? env : "";
        size_t a = 0;
        while (a < list.size()) {
            size_t b = list.find(',', a);
            if (b == std::string::npos) {
                b = list.size();
            }
            const std::string name = list.substr(a, b - a);
            for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
                const char * tn = ggml_type_name((ggml_type) t);
                if (tn && name == tn) {
                    types.push_back((ggml_type) t);
                }
            }
            a = b + 1;
        }
        return types;
    }();
    return std::find(skip.begin(), skip.end(), type) != skip.end();
}

static int kcol_type(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q8_0:   return kcol::KQ8_0;
        case GGML_TYPE_Q2_0:   return kcol::KQ2_0;
        case GGML_TYPE_Q3_K:   return kcol::KQ3_K;
        case GGML_TYPE_Q4_K:   return kcol::KQ4_K;
        case GGML_TYPE_Q5_K:   return kcol::KQ5_K;
        case GGML_TYPE_Q6_K:   return kcol::KQ6_K;
        case GGML_TYPE_IQ4_XS: return kcol::KIQ4_XS;
        default:               return -1;
    }
}

} // namespace

bool ggml_cuda_mmvq_pascal_kcol(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                ggml_tensor * dst) {
    const int mode = kcol_mode();
    if (mode == 0 || !ggml_cuda_mmvq_pascal_active(ggml_cuda_info().devices[ctx.device].cc)) {
        return false;
    }
    const int t = kcol_type(src0->type);
    if (t < 0 || kcol_skip_type(src0->type)) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    // plain 2-D products: rows of src0 nb[1] bytes apart, ncols FP32 columns of src1 and dst
    const int64_t K     = src0->ne[0];
    const int64_t nrows = src0->ne[1];
    const int64_t ncols = src1->ne[1];
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[2] != 1 || dst->ne[3] != 1 ||
            src1->ne[0] != K || dst->ne[0] != nrows || dst->ne[1] != ncols || ncols < 1 || ncols > KCOL_MAX_COLS) {
        return false;
    }
    if (src0->nb[0] != ggml_type_size(src0->type) || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) ||
            src1->nb[1] % sizeof(float) != 0 || dst->nb[1] % sizeof(float) != 0) {
        return false;
    }
    if (src0->nb[1] > INT_MAX || nrows > INT_MAX || K > INT_MAX || src1->nb[1] / sizeof(float) > INT_MAX ||
            dst->nb[1] / sizeof(float) > INT_MAX) {
        return false;
    }
    if (ncols == 1 && mode == 1) {
        // one column: only where the one-token kernels are slow or missing
        const bool q8_short = t == kcol::KQ8_0 && K % 64 == 0 && (K / 64) % 4 != 0;
        if (t != kcol::KQ3_K && !q8_short) {
            return false;
        }
    }

    const kcol_overrides & o = kcol_shape_env();
    const int nsm = o.nsm > 0 ? o.nsm : ggml_cuda_info().devices[ctx.device].nsm;
    kcol::shape s;
    if (!kcol::pick_shape(t, (int) K, (int) nrows, (int) ncols, (int) src0->nb[1], (uintptr_t) src0->data, nsm, o.r, o.itt,
                          o.smem, s)) {
        return false;
    }

    ggml_cuda_pool_alloc<char> rec(ctx.pool(), kcol::records_bytes(t, (int) K, (int) ncols));

    kcol::params p;
    p.vx             = (const char *) src0->data;
    p.y              = (const float *) src1->data;
    p.rec            = (uint32_t *) rec.get();
    p.dst            = (float *) dst->data;
    p.K              = (int) K;
    p.nrows          = (int) nrows;
    p.row_bytes      = (int) src0->nb[1];
    p.ncols          = (int) ncols;
    p.stride_col_y   = (int) (src1->nb[1] / sizeof(float));
    p.stride_col_dst = (int) (dst->nb[1] / sizeof(float));
    p.s              = s;

    cudaStream_t stream = ctx.stream();
    switch (t) {
        case kcol::KQ8_0:   ggml_cuda_kcol_launch_q8_0(p, stream);   break;
        case kcol::KQ2_0:   ggml_cuda_kcol_launch_q2_0(p, stream);   break;
        case kcol::KQ3_K:   ggml_cuda_kcol_launch_q3_K(p, stream);   break;
        case kcol::KQ4_K:   ggml_cuda_kcol_launch_q4_K(p, stream);   break;
        case kcol::KQ5_K:   ggml_cuda_kcol_launch_q5_K(p, stream);   break;
        case kcol::KQ6_K:   ggml_cuda_kcol_launch_q6_K(p, stream);   break;
        case kcol::KIQ4_XS: ggml_cuda_kcol_launch_iq4_xs(p, stream); break;
        default:            GGML_ABORT("fatal error");
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#else

bool ggml_cuda_mmvq_pascal_kcol(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                ggml_tensor * dst) {
    GGML_UNUSED_VARS(ctx, src0, src1, dst);
    return false;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
