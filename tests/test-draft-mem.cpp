// The draft-memory switches that run without a GPU: the head shortlist's bookkeeping (Q592), its logits against the
// full head on the CPU backend for F32 and quantized heads, the shortlist's fallback policy and setting, and the
// direct-I/O staging setting (Q580). The shared tensors (Q564) need two real models; they are checked on the files.

#include "../src/llama-head-shortlist.h"
#include "../src/llama-mmap.h"
#include "../common/draft-shortlist.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static int n_checks = 0;
static int n_fail   = 0;

#define CHECK(cond) do { n_checks++; if (!(cond)) { n_fail++; fprintf(stderr, "%s:%d: failed: %s\n", __FILE__, __LINE__, #cond); } } while (0)

static void test_list() {
    llama_head_shortlist sl;
    CHECK(!sl.init(100, 0,   10));    // k_low 0
    CHECK(!sl.init(100, 100, 10));    // nothing left out
    CHECK(!sl.init(100, 10,  70000)); // past the CUDA expert path's grid

    CHECK(sl.init(100, 10, 5));
    CHECK(sl.contains(0) && sl.contains(9) && !sl.contains(10) && !sl.contains(-1) && !sl.contains(100));

    // 3 is below k_low, 50 comes twice, -4 and 100 are out of range, and there is room for five: 50 99 12 13 14
    const int32_t ids[] = { 3, 50, 50, 99, -4, 100, 12, 13, 14, 15, 16 };
    CHECK(sl.add(ids, 11) == 5);
    CHECK(sl.n_extra() == 5 && sl.size() == 15);
    CHECK(sl.contains(50) && sl.contains(99) && sl.contains(14) && !sl.contains(15));
    CHECK(sl.extra[0] == 50 && sl.extra[1] == 99 && sl.extra[4] == 14);
    CHECK(sl.add(ids, 11) == 0);

    // the room is clamped to the ids left out
    llama_head_shortlist s2;
    CHECK(s2.init(20, 15, 100) && s2.m_cap == 5);
}

// the shortlist's logits equal the full head's for the ids in the list and are -INFINITY for the others, also after
// ids are added once the device copy exists
static void test_logits(ggml_type type, uint32_t n_extra_init, uint32_t n_extra_late) {
    const int64_t  n_embd  = 256;
    const int64_t  n_vocab = 3000;
    const uint32_t k_low   = 700;

    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(be);

    ggml_init_params ipw = { 4*ggml_tensor_overhead(), nullptr, true };
    ggml_context * cw = ggml_init(ipw);
    ggml_tensor * head = ggml_new_tensor_2d(cw, type,          n_embd, n_vocab);
    ggml_tensor * h    = ggml_new_tensor_2d(cw, GGML_TYPE_F32, n_embd, 1);
    ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors(cw, be);

    std::mt19937 rng(1234 + (int) type);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> wf(n_embd*n_vocab);
    for (auto & v : wf) {
        v = nd(rng);
    }
    if (type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(head, wf.data(), 0, ggml_nbytes(head));
    } else {
        std::vector<uint8_t> q(ggml_nbytes(head));
        ggml_quantize_chunk(type, wf.data(), q.data(), 0, n_vocab, n_embd, nullptr);
        ggml_backend_tensor_set(head, q.data(), 0, q.size());
    }
    std::vector<float> hf(n_embd);
    for (auto & v : hf) {
        v = nd(rng);
    }
    ggml_backend_tensor_set(h, hf.data(), 0, ggml_nbytes(h));

    // the top ids (like a vocabulary's special tokens), then random ones, some of them below k_low or repeated
    std::vector<int32_t> ids;
    for (int32_t id = n_vocab - 1; id >= 0 && ids.size() < n_extra_init / 4; --id) {
        ids.push_back(id);
    }
    std::uniform_int_distribution<int32_t> ud(0, n_vocab - 1);
    while (ids.size() < n_extra_init + n_extra_late) {
        ids.push_back(ud(rng));
    }

    llama_head_shortlist sl;
    CHECK(sl.init(n_vocab, k_low, 1024));
    sl.add(ids.data(), n_extra_init);
    CHECK(sl.alloc(buft));
    sl.add(ids.data() + n_extra_init, n_extra_late);
    sl.sync();
    CHECK(sl.n_dev == sl.extra.size());

    ggml_init_params ipg = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * cg = ggml_init(ipg);
    ggml_tensor * full = ggml_mul_mat(cg, head, h);
    ggml_tensor * shrt = llama_head_shortlist_logits(cg, sl, head, h);
    CHECK(shrt->ne[0] == n_vocab && shrt->ne[1] == 1 && shrt->type == GGML_TYPE_F32);

    ggml_cgraph * gf = ggml_new_graph(cg);
    ggml_build_forward_expand(gf, full);
    ggml_build_forward_expand(gf, shrt);

    ggml_gallocr_t ga = ggml_gallocr_new(buft);
    CHECK(ggml_gallocr_alloc_graph(ga, gf));
    CHECK(ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS);

    std::vector<float> lf(n_vocab);
    std::vector<float> ls(n_vocab);
    ggml_backend_tensor_get(full, lf.data(), 0, n_vocab*sizeof(float));
    ggml_backend_tensor_get(shrt, ls.data(), 0, n_vocab*sizeof(float));

    int n_in = 0;
    int n_bad_in = 0;
    int n_bad_out = 0;
    float max_diff = 0.0f;
    for (int64_t id = 0; id < n_vocab; ++id) {
        if (sl.contains((int32_t) id)) {
            n_in++;
            const float d = std::fabs(ls[id] - lf[id]);
            max_diff = std::max(max_diff, d);
            n_bad_in += !(d <= 1e-4f*std::max(1.0f, std::fabs(lf[id])));
        } else {
            n_bad_out += !(std::isinf(ls[id]) && ls[id] < 0.0f);
        }
    }
    CHECK(n_in == (int) sl.size());
    CHECK(n_bad_in == 0);
    CHECK(n_bad_out == 0);

    // the argmax over the list is the full head's when the full argmax is in the list
    int64_t am_f = 0;
    int64_t am_s = 0;
    for (int64_t id = 1; id < n_vocab; ++id) {
        am_f = lf[id] > lf[am_f] ? id : am_f;
        am_s = ls[id] > ls[am_s] ? id : am_s;
    }
    if (sl.contains((int32_t) am_f)) {
        CHECK(am_s == am_f);
    }

    printf("logits %-5s: %4u ids in the list (%u below %u, %u more), max |diff| %.3g, %d bad inside, %d bad outside\n",
            ggml_type_name(type), sl.size(), k_low, k_low, sl.n_extra(), max_diff, n_bad_in, n_bad_out);

    ggml_gallocr_free(ga);
    ggml_free(cg);
    sl.buf.reset();
    sl.ctx.reset();
    ggml_backend_buffer_free(bw);
    ggml_free(cw);
    ggml_backend_free(be);
}

static void test_policy() {
    common_draft_shortlist_policy p;
    p.rounds = 2;
    CHECK(p.next(true)  == true);
    CHECK(p.next(false) == false); // the target's token was outside: full head now and for one more draft
    CHECK(p.next(true)  == false);
    CHECK(p.next(true)  == true);
    CHECK(p.next(false) == false);
    CHECK(p.next(false) == false); // a second miss restarts the count
    CHECK(p.next(true)  == false);
    CHECK(p.next(true)  == true);
    CHECK(p.n_short == 3 && p.n_full == 5 && p.n_miss == 3);

    common_draft_shortlist_policy never;
    never.rounds = 0;
    CHECK(never.next(false) == true && never.next(true) == true && never.n_miss == 1 && never.n_full == 0);

    auto c = common_draft_shortlist_config::from_string("32768");
    CHECK(c.k_low == 32768 && c.m_cap == 8192 && c.rounds == 2);
    c = common_draft_shortlist_config::from_string("16384,4096,0");
    CHECK(c.k_low == 16384 && c.m_cap == 4096 && c.rounds == 0);
    c = common_draft_shortlist_config::from_string("");
    CHECK(c.k_low == 0);
    c = common_draft_shortlist_config::from_string(nullptr);
    CHECK(c.k_low == 0);
}

int main() {
    // read once, on first use
    setenv("LLAMA_DIO_STAGING", "2x16", 1);
    CHECK(llama_direct_io_buffer_count() == 2);
    CHECK(llama_direct_io_buffer_size()  == 16u*1024*1024);

    test_list();
    test_policy();

    test_logits(GGML_TYPE_F32,  0,   0);
    test_logits(GGML_TYPE_F32,  200, 50);
    test_logits(GGML_TYPE_Q8_0, 200, 50);
    test_logits(GGML_TYPE_Q5_K, 200, 50);
    test_logits(GGML_TYPE_Q6_K, 200, 50);
    test_logits(GGML_TYPE_Q3_K, 0,   300);

    printf("%d checks, %d failed\n", n_checks, n_fail);
    return n_fail == 0 ? 0 : 1;
}
