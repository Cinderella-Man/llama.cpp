#include "keepalive.cuh"
#include "common.cuh"

#include <atomic>

// GGML_CUDA_KEEPALIVE (keepalive-policy.h has the settings): one block of one thread per card, launched on a stream
// of the card's lowest priority, that reads the card's word in mapped host memory every GGML_CUDA_KEEPALIVE_POLL_US
// (spinning on clock64() in between) until the host ends its run or no graph has been computed on the card for the
// timeout. While it runs, the driver counts the card as busy. It measures the SM clock every 100 ms and leaves a report.

static __device__ __forceinline__ unsigned long long ggml_cuda_keepalive_gtime() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

static __global__ void ggml_cuda_keepalive_kernel(const volatile unsigned long long * word, ggml_cuda_keepalive_report * rep,
                                                  const uint32_t gen, const uint32_t beat, const long long poll_cycles,
                                                  const unsigned long long idle_ns) {
    uint32_t hist[GGML_CUDA_KEEPALIVE_BINS];
    for (int i = 0; i < GGML_CUDA_KEEPALIVE_BINS; ++i) {
        hist[i] = 0;
    }
    const unsigned long long t0 = ggml_cuda_keepalive_gtime();
    ggml_cuda_keepalive_watch w = { beat, t0 };
    unsigned long long polls = 0;
    unsigned long long t_s = t0;
    long long          c_s = clock64();
    int r;
    for (;;) {
        // the time first, then the word: a timeout is decided on a word read after its time, so a beat written before
        // that time is always seen
        const unsigned long long now = ggml_cuda_keepalive_gtime();
        const unsigned long long v   = *word;
        polls++;
        r = ggml_cuda_keepalive_poll(v, gen, now, idle_ns, w);
        if (r != GGML_CUDA_KEEPALIVE_RUN) {
            break;
        }
        if (now - t_s >= GGML_CUDA_KEEPALIVE_SAMPLE_NS) {
            const long long          c   = clock64();
            const unsigned long long mhz = (unsigned long long) (c - c_s) * 1000ull / (now - t_s);
            hist[mhz / 50 < GGML_CUDA_KEEPALIVE_BINS ? mhz / 50 : GGML_CUDA_KEEPALIVE_BINS - 1]++;
            t_s = now;
            c_s = c;
        }
        const long long c0 = clock64();
        while (clock64() - c0 < poll_cycles) {
        }
    }
    rep->reason = (uint32_t) r;
    rep->polls  = polls;
    rep->ns     = ggml_cuda_keepalive_gtime() - t0;
    for (int i = 0; i < GGML_CUDA_KEEPALIVE_BINS; ++i) {
        rep->hist[i] = hist[i];
    }
    __threadfence_system();
    rep->gen = gen;
    __threadfence_system();
}

// what a card does for the controller; dev is the physical CUDA device
struct ggml_cuda_keepalive_ops {
    struct slot {
        alignas(64) volatile unsigned long long word;
        ggml_cuda_keepalive_report report;
    };

    slot *       host = nullptr; // one per card, mapped pinned host memory, never freed
    bool         failed = false;
    cudaStream_t streams[GGML_CUDA_MAX_DEVICES] = {};
    int          clock_khz[GGML_CUDA_MAX_DEVICES] = {};

    // the slots, made on the first start (no block runs yet, so the allocation waits for nothing)
    bool ready() {
        if (host == nullptr && !failed) {
            if (cudaHostAlloc((void **) &host, GGML_CUDA_MAX_DEVICES * sizeof(slot), cudaHostAllocPortable | cudaHostAllocMapped) !=
                cudaSuccess) {
                (void) cudaGetLastError();
                host   = nullptr;
                failed = true;
                GGML_LOG_WARN("%s: no mapped host memory for GGML_CUDA_KEEPALIVE: off\n", __func__);
            } else {
                memset((void *) host, 0, GGML_CUDA_MAX_DEVICES * sizeof(slot));
            }
        }
        return host != nullptr;
    }

    void write(int dev, uint64_t w) {
        if (ready()) {
            host[dev].word = w;
            std::atomic_thread_fence(std::memory_order_seq_cst);
        }
    }

    bool launch(int dev, uint32_t gen, uint32_t beat, uint32_t idle_ms, uint32_t poll_us) {
        if (!ready()) {
            return false;
        }
        int prev = 0;
        CUDA_CHECK(cudaGetDevice(&prev));
        if (prev != dev) {
            CUDA_CHECK(cudaSetDevice(dev));
        }
        if (streams[dev] == nullptr) {
            int least = 0, greatest = 0;
            CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
            CUDA_CHECK(cudaStreamCreateWithPriority(&streams[dev], cudaStreamNonBlocking, least));
            CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz[dev], cudaDevAttrClockRate, dev));
        }
        unsigned long long * word = nullptr;
        ggml_cuda_keepalive_report * rep = nullptr;
        CUDA_CHECK(cudaHostGetDevicePointer((void **) &word, (void *) &host[dev].word, 0));
        CUDA_CHECK(cudaHostGetDevicePointer((void **) &rep, (void *) &host[dev].report, 0));
        host[dev].report.gen = 0;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        const long long poll_cycles = (long long) poll_us * clock_khz[dev] / 1000;
        ggml_cuda_keepalive_kernel<<<1, 1, 0, streams[dev]>>>(word, rep, gen, beat, poll_cycles, (unsigned long long) idle_ms * 1000000ull);
        const cudaError_t err = cudaGetLastError();
        if (prev != dev) {
            CUDA_CHECK(cudaSetDevice(prev));
        }
        if (err != cudaSuccess) {
            GGML_LOG_WARN("%s: CUDA%d: the keep-alive block did not launch: %s\n", __func__, dev, cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool done(int dev) {
        const cudaError_t err = cudaStreamQuery(streams[dev]);
        if (err == cudaErrorNotReady) {
            return false;
        }
        CUDA_CHECK(err);
        return true;
    }

    void wait(int dev) {
        CUDA_CHECK(cudaStreamSynchronize(streams[dev]));
    }

    const ggml_cuda_keepalive_report & report(int dev) {
        return host[dev].report;
    }
};

using ggml_cuda_keepalive_ctl_t = ggml_cuda_keepalive_ctl<ggml_cuda_keepalive_ops, GGML_CUDA_MAX_DEVICES>;

// never destroyed: a buffer may be freed while static objects are destroyed at exit
static ggml_cuda_keepalive_ctl_t & ggml_cuda_keepalive() {
    static ggml_cuda_keepalive_ctl_t * ctl = [] {
        const ggml_cuda_keepalive_config cfg = ggml_cuda_keepalive_parse(
            getenv("GGML_CUDA_KEEPALIVE"), getenv("GGML_CUDA_KEEPALIVE_MS"), getenv("GGML_CUDA_KEEPALIVE_POLL_US"));
        if (cfg.mode != GGML_CUDA_KEEPALIVE_OFF) {
            GGML_LOG_INFO("%s: GGML_CUDA_KEEPALIVE=%s: a block per card holds its clock, %u ms after the card's last graph, "
                          "polling every %u us\n", __func__, ggml_cuda_keepalive_mode_name(cfg.mode), cfg.idle_ms, cfg.poll_us);
        }
        return new ggml_cuda_keepalive_ctl_t(cfg, *new ggml_cuda_keepalive_ops());
    }();
    return *ctl;
}

void ggml_cuda_keepalive_graph_begin(int device, bool capture) {
    ggml_cuda_keepalive_ctl_t & ka = ggml_cuda_keepalive();
    if (ka.on()) {
        ka.graph_begin(device, capture);
    }
}

void ggml_cuda_keepalive_graph_end(int device) {
    ggml_cuda_keepalive_ctl_t & ka = ggml_cuda_keepalive();
    if (ka.on()) {
        ka.graph_end(device);
    }
}

void ggml_cuda_keepalive_quiesce() {
    ggml_cuda_keepalive().quiesce();
}

void ggml_cuda_keepalive_log(int device) {
    ggml_cuda_keepalive_ctl_t & ka = ggml_cuda_keepalive();
    if (!ka.on()) {
        return;
    }
    const int dev = device;
    const ggml_cuda_keepalive_stats s = ka.take_stats(dev);
    if (s.starts == 0) {
        return;
    }
    GGML_LOG_INFO("%s: CUDA%d: %llu runs (%llu ended by the host, %llu by the timeout, %llu without a report), %.1f s; "
                  "SM clock while running: p10 %d, median %d, p90 %d MHz, %.0f %% below 1400, %.0f %% at 1800 or more "
                  "(%llu samples of 100 ms)\n", __func__, dev, (unsigned long long) s.starts, (unsigned long long) s.stopped,
                  (unsigned long long) s.timeouts, (unsigned long long) s.lost, s.seconds, s.mhz_quantile(0.1),
                  s.mhz_quantile(0.5), s.mhz_quantile(0.9), 100.0 * s.share(0, 1400), 100.0 * s.share(1800, 100000),
                  (unsigned long long) s.samples());
}

void ggml_backend_cuda_keepalive_kick(void) {
    ggml_cuda_keepalive_ctl_t & ka = ggml_cuda_keepalive();
    if (ka.cfg.mode == GGML_CUDA_KEEPALIVE_REQUEST) {
        ka.kick();
    }
}
