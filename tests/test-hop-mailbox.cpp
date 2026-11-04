// test-hop-mailbox: the card hop through a mailbox in mapped host memory (GGML_CUDA_HOP_MAILBOX) copies bytes exactly,
// in order, at every size, through every ring setting, without a host wait per hop.
//
// For each setting (the GGML_CUDA_HOP_* variables set before the backends are made, every other one unset) the test
// makes a backend on each of two or three CUDA cards and issues rounds of hops without synchronizing: per round a
// pattern written into a view of a pool tensor on card A (ggml_backend_tensor_set_async), the copy to card B
// (ggml_backend_tensor_copy_async, the scheduler's path between two cards), the copy back from B to A or on to card C,
// and the result read back on the last card (ggml_backend_tensor_get_async). As the scheduler does before each input
// copy, the destination waits on an event recorded after its last read of the pool (the guard a reader on the copy
// stream waits on). Every few rounds a long matrix product on one of the cards holds its stream, so the other side of a
// hop waits: writers on a full ring for acknowledgements, readers for flags. The rounds use the hop shapes of
// Flash-Next's residual (10,880 to 163,840 bytes), a prompt micro-batch's (5.6 MB), small and odd sizes, offsets in
// the pool of 0 and 256 bytes and a misaligned one (the mailbox leaves the last two to the other paths).
//
// Checks: every round's bytes arrive exactly; with the mailbox on, the number of hops each context logs as made by
// mailbox when it is freed equals the number of copies the mailbox takes (so a setting cannot pass through the other
// paths alone); no kernel timed out.
// Usage: test-hop-mailbox [--rounds N] [--setting name]   (exit status 0 when every check passes; 0 with a note when
// there are fewer than two CUDA cards)

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

static int n_checks = 0;
static int n_fail   = 0;

static void check(bool ok, const std::string & what) {
    n_checks++;
    if (!ok) {
        n_fail++;
        fprintf(stderr, "FAIL: %s\n", what.c_str());
    }
}

// the log, kept for the mailbox's counts
static std::string g_log;

static void log_cb(ggml_log_level level, const char * text, void * /*user_data*/) {
    g_log += text;
    if (level == GGML_LOG_LEVEL_ERROR || strstr(text, "hop") != nullptr) {
        fputs(text, stderr);
    }
}

static const char * k_vars[] = {
    "GGML_CUDA_HOP_MAILBOX", "GGML_CUDA_HOP_MAILBOX_CHUNKS", "GGML_CUDA_HOP_MAILBOX_SLOTS", "GGML_CUDA_HOP_MAILBOX_SLOT_KB",
    "GGML_CUDA_HOP_MAILBOX_MAX_KB", "GGML_CUDA_HOP_MAILBOX_STREAM", "GGML_CUDA_HOP_MAILBOX_THREADS",
    "GGML_CUDA_HOP_MAILBOX_POLL_US", "GGML_CUDA_HOP_MAILBOX_TIMEOUT_MS", "GGML_CUDA_HOP_STAGED", "GGML_CUDA_HOP_CHUNKS",
    "GGML_CUDA_HOP_SLOTS", "GGML_CUDA_HOP_SLOT_KB", "GGML_CUDA_HOP_STREAM",
};

struct setting {
    std::string name;
    std::vector<std::pair<std::string, std::string>> env;
    bool   mailbox;   // copies may take the mailbox
    size_t max_bytes; // the mailbox's limit (0: none)
};

static void set_env(const setting & s) {
    for (const char * v : k_vars) {
        unsetenv(v);
    }
    for (const auto & kv : s.env) {
        setenv(kv.first.c_str(), kv.second.c_str(), 1);
    }
}

static uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

struct round_t {
    size_t bytes;
    size_t offset;
    int    route;  // 0: A -> B; 1: A -> B -> A; 2: A -> B -> C (three cards)
    int    hold;   // -1: none; else the card whose stream a matrix product holds before the round
};

// the sum of the counts "by mailbox: N hops" in the log since pos
static int64_t mailbox_hops_logged(size_t pos) {
    int64_t sum = 0;
    const std::string key = "by mailbox: ";
    for (size_t p = g_log.find(key, pos); p != std::string::npos; p = g_log.find(key, p + 1)) {
        long long n = 0;
        char word[16] = {};
        if (sscanf(g_log.c_str() + p + key.size(), "%lld %15s", &n, word) == 2 && strcmp(word, "hops,") == 0) {
            sum += n;
        }
    }
    return sum;
}

static bool mailbox_takes(const setting & s, size_t bytes, size_t offset) {
    return s.mailbox && bytes % 16 == 0 && offset % 16 == 0 && (s.max_bytes == 0 || bytes <= s.max_bytes);
}

static void run_setting(const setting & s, const std::vector<ggml_backend_dev_t> & devs, int n_rounds) {
    set_env(s);
    const int n_dev = (int) std::min<size_t>(devs.size(), 3);

    std::vector<ggml_backend_t> be(n_dev);
    for (int d = 0; d < n_dev; ++d) {
        be[d] = ggml_backend_dev_init(devs[d], nullptr);
        GGML_ASSERT(be[d] != nullptr);
    }

    const std::vector<size_t> sizes   = { 10880, 40960, 16, 43520, 163840, 256, 20480, 4096, 1u << 20, 5570560, 10, 4100 };
    const std::vector<size_t> offsets = { 0, 256, 0, 4 };
    std::vector<round_t> rounds;
    size_t max_extent = 0;
    for (int r = 0; r < n_rounds; ++r) {
        round_t rd;
        rd.bytes  = sizes[r % sizes.size()];
        rd.offset = offsets[(r / sizes.size()) % offsets.size()];
        rd.route  = r % 3 == 2 ? (n_dev >= 3 ? 2 : 1) : r % 3;
        rd.hold   = r % 7 == 3 ? 1 : r % 7 == 5 ? 0 : -1;
        rounds.push_back(rd);
        max_extent = std::max(max_extent, rd.bytes + rd.offset);
    }

    // per card a pool the rounds' views live in, and a matrix product that holds its stream for a while
    const int64_t nm = 1024;
    const size_t n_tensors = 4 + (size_t) n_rounds * 4 + 64;
    std::vector<ggml_context *> ctx(n_dev);
    std::vector<ggml_tensor *> pool(n_dev), ma(n_dev), mb(n_dev);
    std::vector<ggml_cgraph *> hold_graph(n_dev);
    std::vector<ggml_backend_buffer_t> buf(n_dev);
    for (int d = 0; d < n_dev; ++d) {
        ggml_init_params ip = { ggml_tensor_overhead() * n_tensors + ggml_graph_overhead(), nullptr, true };
        ctx[d]  = ggml_init(ip);
        pool[d] = ggml_new_tensor_1d(ctx[d], GGML_TYPE_I8, (int64_t) max_extent);
        ma[d]   = ggml_new_tensor_2d(ctx[d], GGML_TYPE_F32, nm, nm);
        mb[d]   = ggml_new_tensor_2d(ctx[d], GGML_TYPE_F32, nm, nm);
        ggml_tensor * x = mb[d];
        for (int i = 0; i < 12; ++i) {
            x = ggml_mul_mat(ctx[d], ma[d], x);
        }
        hold_graph[d] = ggml_new_graph(ctx[d]);
        ggml_build_forward_expand(hold_graph[d], x);
    }
    // the views: per round one per card it passes
    std::vector<std::vector<ggml_tensor *>> view(n_dev, std::vector<ggml_tensor *>(n_rounds));
    for (int r = 0; r < n_rounds; ++r) {
        for (int d = 0; d < n_dev; ++d) {
            view[d][r] = ggml_view_1d(ctx[d], pool[d], (int64_t) rounds[r].bytes, rounds[r].offset);
        }
    }
    for (int d = 0; d < n_dev; ++d) {
        buf[d] = ggml_backend_alloc_ctx_tensors(ctx[d], be[d]);
        GGML_ASSERT(buf[d] != nullptr);
        std::vector<float> m(nm * nm);
        for (size_t i = 0; i < m.size(); ++i) {
            m[i] = (float) ((mix(i + 77*d) % 2001) - 1000) * 1e-6f;
        }
        ggml_backend_tensor_set(ma[d], m.data(), 0, ggml_nbytes(ma[d]));
        ggml_backend_tensor_set(mb[d], m.data(), 0, ggml_nbytes(mb[d]));
    }
    std::vector<ggml_backend_event_t> ev(n_dev);
    for (int d = 0; d < n_dev; ++d) {
        ev[d] = ggml_backend_event_new(devs[d]);
        GGML_ASSERT(ev[d] != nullptr);
        ggml_backend_event_record(ev[d], be[d]);
    }

    // the scheduler's input copy: the destination waits on its guard, then the copy
    int64_t n_mailbox_expected = 0;
    auto hop = [&](int from, int to, int r) {
        ggml_backend_event_wait(be[to], ev[to]);
        ggml_backend_tensor_copy_async(be[from], be[to], view[from][r], view[to][r]);
        n_mailbox_expected += mailbox_takes(s, rounds[r].bytes, rounds[r].offset);
    };

    std::vector<std::vector<uint8_t>> pattern(n_rounds), out(n_rounds);
    for (int r = 0; r < n_rounds; ++r) {
        const round_t & rd = rounds[r];
        pattern[r].resize(rd.bytes);
        out[r].assign(rd.bytes, 0);
        for (size_t i = 0; i < rd.bytes; ++i) {
            pattern[r][i] = (uint8_t) mix(((uint64_t) r << 32) ^ i);
        }
        if (rd.hold >= 0 && rd.hold < n_dev) {
            ggml_backend_graph_compute_async(be[rd.hold], hold_graph[rd.hold]);
        }
        ggml_backend_tensor_set_async(be[0], view[0][r], pattern[r].data(), 0, rd.bytes);
        hop(0, 1, r);
        int last = 1;
        if (rd.route == 1) {
            hop(1, 0, r);
            last = 0;
        } else if (rd.route == 2) {
            hop(1, 2, r);
            last = 2;
        }
        ggml_backend_tensor_get_async(be[last], view[last][r], out[r].data(), 0, rd.bytes);
        // the guard for the next copy into this card's pool
        ggml_backend_event_record(ev[last], be[last]);
        if (last != 1) {
            ggml_backend_event_record(ev[1], be[1]);
        }
    }
    for (int d = 0; d < n_dev; ++d) {
        ggml_backend_synchronize(be[d]);
    }

    int bad_rounds = 0;
    for (int r = 0; r < n_rounds; ++r) {
        if (memcmp(out[r].data(), pattern[r].data(), rounds[r].bytes) != 0) {
            if (bad_rounds++ < 5) {
                size_t i = 0;
                while (out[r][i] == pattern[r][i]) {
                    ++i;
                }
                fprintf(stderr, "  %s: round %d (%zu bytes at offset %zu, route %d) differs from byte %zu\n",
                        s.name.c_str(), r, rounds[r].bytes, rounds[r].offset, rounds[r].route, i);
            }
        }
    }
    check(bad_rounds == 0, s.name + ": " + std::to_string(bad_rounds) + " of " + std::to_string(n_rounds) +
            " rounds differ");

    const size_t log_pos = g_log.size();
    for (int d = 0; d < n_dev; ++d) {
        ggml_backend_event_free(ev[d]);
        ggml_backend_buffer_free(buf[d]);
        ggml_free(ctx[d]);
        ggml_backend_free(be[d]);
    }
    const int64_t n_mailbox = mailbox_hops_logged(log_pos);
    check(n_mailbox == n_mailbox_expected, s.name + ": " + std::to_string(n_mailbox) + " hops by mailbox, expected " +
            std::to_string(n_mailbox_expected));
    check(g_log.find("timed out", log_pos) == std::string::npos, s.name + ": a mailbox kernel timed out");
    printf("%-22s %s: %d rounds over %d cards, %" PRId64 " hops by mailbox\n", s.name.c_str(),
           bad_rounds == 0 && n_mailbox == n_mailbox_expected ? "ok  " : "FAIL", n_rounds, n_dev, n_mailbox);
}

int main(int argc, char ** argv) {
    int n_rounds = 96;
    std::string only;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--rounds") && i + 1 < argc) {
            n_rounds = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--setting") && i + 1 < argc) {
            only = argv[++i];
        } else {
            fprintf(stderr, "usage: %s [--rounds N] [--setting name]\n", argv[0]);
            return 2;
        }
    }

    ggml_log_set(log_cb, nullptr);
    ggml_backend_load_all();
    std::vector<ggml_backend_dev_t> devs;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU && strncmp(ggml_backend_dev_name(dev), "CUDA", 4) == 0) {
            devs.push_back(dev);
        }
    }
    if (devs.size() < 2) {
        printf("test-hop-mailbox: %zu CUDA cards, the mailbox needs two: nothing to check\n", devs.size());
        return 0;
    }

    const std::vector<setting> settings = {
        { "peer-copy",          {},                                                                   false, 0 },
        { "mailbox",            { {"GGML_CUDA_HOP_MAILBOX", "2"} },                                   true,  0 },
        { "one-chunk",          { {"GGML_CUDA_HOP_MAILBOX", "2"}, {"GGML_CUDA_HOP_MAILBOX_CHUNKS", "1"} }, true, 0 },
        { "tiny-ring",          { {"GGML_CUDA_HOP_MAILBOX", "2"}, {"GGML_CUDA_HOP_MAILBOX_CHUNKS", "64"},
                                  {"GGML_CUDA_HOP_MAILBOX_SLOTS", "4"}, {"GGML_CUDA_HOP_MAILBOX_SLOT_KB", "1"} }, true, 0 },
        { "tiny-ring-compute",  { {"GGML_CUDA_HOP_MAILBOX", "2"}, {"GGML_CUDA_HOP_MAILBOX_CHUNKS", "64"},
                                  {"GGML_CUDA_HOP_MAILBOX_SLOTS", "4"}, {"GGML_CUDA_HOP_MAILBOX_SLOT_KB", "1"},
                                  {"GGML_CUDA_HOP_MAILBOX_STREAM", "0"} }, true, 0 },
        { "copy-stream",        { {"GGML_CUDA_HOP_MAILBOX", "2"}, {"GGML_CUDA_HOP_MAILBOX_STREAM", "2"} }, true, 0 },
        { "threads64-poll0",    { {"GGML_CUDA_HOP_MAILBOX", "2"}, {"GGML_CUDA_HOP_MAILBOX_THREADS", "64"},
                                  {"GGML_CUDA_HOP_MAILBOX_POLL_US", "0"} }, true, 0 },
        { "threads1024-poll50", { {"GGML_CUDA_HOP_MAILBOX", "2"}, {"GGML_CUDA_HOP_MAILBOX_THREADS", "1024"},
                                  {"GGML_CUDA_HOP_MAILBOX_POLL_US", "50"}, {"GGML_CUDA_HOP_MAILBOX_CHUNKS", "3"},
                                  {"GGML_CUDA_HOP_MAILBOX_SLOTS", "6"} }, true, 0 },
        { "max16k-staged",      { {"GGML_CUDA_HOP_MAILBOX", "2"}, {"GGML_CUDA_HOP_MAILBOX_MAX_KB", "16"},
                                  {"GGML_CUDA_HOP_STAGED", "2"} }, true, 16 * 1024 },
    };
    for (const setting & s : settings) {
        if (only.empty() || only == s.name) {
            run_setting(s, devs, n_rounds);
        }
    }
    printf("test-hop-mailbox: %d checks, %d failed\n", n_checks, n_fail);
    return n_fail == 0 ? 0 : 1;
}
