#include "pipe-store.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

static const char    PIPE_STORE_MAGIC[8] = { 'P', 'I', 'P', 'E', 'S', 'T', 'O', '1' };
static const int32_t PIPE_STORE_END      = -2;    // past the end of the stream
static const int64_t PIPE_STORE_SCAN_MAX = 2048;  // ranges larger than this are counted once and cached

struct common_pipe_store {
    const int32_t * rev = nullptr;   // the token stream, reversed
    const int32_t * sa  = nullptr;   // suffix array of rev
    int64_t         n   = 0;

    void *               map      = nullptr;   // the mapped file (POSIX)
    size_t               map_size = 0;
    std::vector<int32_t> copy;                 // or the file's contents read into memory

    struct top { int32_t tok, cnt, total; };
    std::mutex                        mtx;
    std::unordered_map<uint64_t, top> big;     // (depth, range start) -> top next token of a large range

    int32_t at(int64_t i) const { return i < n ? rev[i] : PIPE_STORE_END; }

    // the token after the occurrence whose reversed suffix starts at r (the one before it in rev), or -1
    int32_t next_of(int64_t r) const { return r > 0 ? rev[r - 1] : -1; }

    ~common_pipe_store() {
#ifndef _WIN32
        if (map) {
            munmap(map, map_size);
        }
#endif
    }
};

common_pipe_store * common_pipe_store_load(const std::string & path) {
    auto * st = new common_pipe_store();
    const size_t hdr = sizeof(PIPE_STORE_MAGIC) + sizeof(int64_t);
    const uint8_t * base = nullptr;
    size_t size = 0;
#ifndef _WIN32
    const int fd = open(path.c_str(), O_RDONLY);
    struct stat sb;
    if (fd >= 0 && fstat(fd, &sb) == 0 && (size_t) sb.st_size >= hdr) {
        void * m = mmap(nullptr, (size_t) sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
        if (m != MAP_FAILED) {
            st->map      = m;
            st->map_size = (size_t) sb.st_size;
            base = (const uint8_t *) m;
            size = st->map_size;
        }
    }
    if (fd >= 0) {
        close(fd);
    }
#else
    FILE * f = fopen(path.c_str(), "rb");
    if (f) {
        fseek(f, 0, SEEK_END);
        const long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (sz >= (long) hdr && sz % 4 == 0) {
            st->copy.resize((size_t) sz / 4);
            if (fread(st->copy.data(), 1, (size_t) sz, f) == (size_t) sz) {
                base = (const uint8_t *) st->copy.data();
                size = (size_t) sz;
            }
        }
        fclose(f);
    }
#endif
    int64_t n = 0;
    if (base) {
        memcpy(&n, base + sizeof(PIPE_STORE_MAGIC), sizeof(n));
    }
    if (!base || memcmp(base, PIPE_STORE_MAGIC, sizeof(PIPE_STORE_MAGIC)) != 0 || n <= 0 ||
            size != hdr + 2*(size_t) n*sizeof(int32_t)) {
        delete st;
        return nullptr;
    }
    st->n   = n;
    st->rev = (const int32_t *) (base + hdr);
    st->sa  = st->rev + n;
    return st;
}

void common_pipe_store_free(common_pipe_store * store) {
    delete store;
}

int64_t common_pipe_store_size(const common_pipe_store * store) {
    return store ? store->n : 0;
}

// the most frequent next token in sa[lo, hi), its count, and the count of all next tokens
static common_pipe_store::top store_top(common_pipe_store & st, int depth, int64_t lo, int64_t hi) {
    const bool     cacheable = hi - lo > PIPE_STORE_SCAN_MAX;
    const uint64_t ck        = ((uint64_t) depth << 40) | (uint64_t) lo;
    if (cacheable) {
        std::lock_guard<std::mutex> lock(st.mtx);
        const auto it = st.big.find(ck);
        if (it != st.big.end()) {
            return it->second;
        }
    }
    std::unordered_map<int32_t, int32_t> cnt;
    cnt.reserve((size_t) std::min<int64_t>(hi - lo, 4096));
    int32_t best = -1, bc = 0, total = 0;
    for (int64_t i = lo; i < hi; i++) {
        const int32_t t = st.next_of(st.sa[i]);
        if (t < 0) {
            continue;
        }
        total++;
        const int32_t c = ++cnt[t];
        if (c > bc || (c == bc && t < best)) {
            best = t;
            bc   = c;
        }
    }
    const common_pipe_store::top r = { best, bc, total };
    if (cacheable) {
        std::lock_guard<std::mutex> lock(st.mtx);
        st.big[ck] = r;
    }
    return r;
}

int common_pipe_store_longest(common_pipe_store * store, const llama_token * ctx, int n, int max_len,
                              llama_token * tok, int32_t * cnt, int32_t * occ) {
    if (!store || n <= 0) {
        return 0;
    }
    common_pipe_store & st = *store;
    const int lmax = std::min(std::min(n, max_len), 64);
    int64_t los[64], his[64];
    int64_t lo = 0, hi = st.n;
    int depth = 0;
    for (int k = 0; k < lmax; k++) {
        const int32_t sym = ctx[n - 1 - k];
        // the suffixes in [lo, hi) share k tokens, so their k-th token is sorted
        const int32_t * a = std::partition_point(st.sa + lo, st.sa + hi,
                                                 [&](int32_t s) { return st.at((int64_t) s + k) < sym; });
        const int32_t * b = std::partition_point(a, st.sa + hi,
                                                 [&](int32_t s) { return st.at((int64_t) s + k) <= sym; });
        if (a >= b) {
            break;
        }
        lo = a - st.sa;
        hi = b - st.sa;
        los[k] = lo;
        his[k] = hi;
        depth = k + 1;
    }
    // a length whose occurrences all end an answer has no token after it, and neither has any longer one: back off
    while (depth > 0) {
        const auto t = store_top(st, depth, los[depth - 1], his[depth - 1]);
        if (t.tok >= 0) {
            *tok = t.tok;
            *cnt = t.cnt;
            *occ = t.total;
            return depth;
        }
        depth--;
    }
    return 0;
}

llama_token common_pipe_store_guess(common_pipe_store * store, const llama_token * ctx, int n) {
    const int    min_len   = 2;
    const int    max_len   = 32;
    const double min_share = 0.3;
    const int    len       = std::min(n, max_len);
    llama_token tok = -1;
    int32_t     cnt = 0;
    int32_t     occ = 0;
    const int L = common_pipe_store_longest(store, ctx + n - len, len, max_len, &tok, &cnt, &occ);
    if (L < min_len || cnt < min_share*occ) {
        return -1;
    }
    return tok;
}
