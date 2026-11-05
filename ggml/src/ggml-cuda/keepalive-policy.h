#pragma once

// GGML_CUDA_KEEPALIVE (keepalive.cu): one small block per card, on a low-priority stream of its own, polling a word in
// mapped host memory with a clock64() back-off, so that the driver counts the card as busy and keeps its boost clock.
// A layer split over many cards leaves each card idle most of a token (on nine P106 cards ~8/9 of it), and the driver
// then runs a decoding card at its base clock (1,354 against up to 1,911 MHz on a P106-090).
//
// The parts that need no GPU live here: the settings, the word the host writes and each block polls, a block's decision
// at each poll (the same function runs in the kernel), the report a block leaves when it ends, and the controller that
// starts the blocks after graph computes and stops them before a capture, a free, a host allocation or a device-wide
// synchronize, any of which would otherwise wait for a block that only the host can end. tests/test-cuda-keepalive.cpp
// checks them on a CPU, with a thread standing in for each card's block.
//
//   GGML_CUDA_KEEPALIVE          unset, empty or 0: off (nothing is launched or allocated)
//                                decode (or 1): a card's block starts after a graph is computed on it, and ends
//                                  itself when no graph has been computed on the card for GGML_CUDA_KEEPALIVE_MS
//                                request: decode, and every card's block also starts when llama-server takes a request,
//                                  so that the cards leave their idle state while the prompt is prepared
//                                idle: decode with a timeout long enough to cover the time the server idles between an
//                                  agent's turns (default 120 s), so the cards stay at their clock between requests
//   GGML_CUDA_KEEPALIVE_MS       how long a block keeps running after its card's last graph, 10..3600000 (default
//                                1000; 120000 for idle). The timeout also bounds the wait of any synchronizing call the
//                                controller is not told about.
//   GGML_CUDA_KEEPALIVE_POLL_US  the block's back-off between two reads of the word, 1..100000 us (default 50): the
//                                longest a stop waits, and one PCIe read per back-off
// When a context is freed, each card's runs are logged with the SM clock the block measured while it ran.

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>

#if defined(__CUDACC__)
#define GGML_CUDA_KEEPALIVE_HD __host__ __device__
#else
#define GGML_CUDA_KEEPALIVE_HD
#endif

enum ggml_cuda_keepalive_mode {
    GGML_CUDA_KEEPALIVE_OFF,
    GGML_CUDA_KEEPALIVE_DECODE,
    GGML_CUDA_KEEPALIVE_REQUEST,
    GGML_CUDA_KEEPALIVE_IDLE,
};

struct ggml_cuda_keepalive_config {
    ggml_cuda_keepalive_mode mode    = GGML_CUDA_KEEPALIVE_OFF;
    uint32_t                 idle_ms = 1000;
    uint32_t                 poll_us = 50;
};

// The three variables' values (nullptr: unset). An unknown mode is off; out-of-range numbers are clamped.
static inline ggml_cuda_keepalive_config ggml_cuda_keepalive_parse(const char * mode, const char * ms, const char * poll_us) {
    ggml_cuda_keepalive_config c;
    if (mode == nullptr || mode[0] == '\0' || strcmp(mode, "0") == 0) {
        c.mode = GGML_CUDA_KEEPALIVE_OFF;
    } else if (strcmp(mode, "1") == 0 || strcmp(mode, "decode") == 0) {
        c.mode = GGML_CUDA_KEEPALIVE_DECODE;
    } else if (strcmp(mode, "request") == 0) {
        c.mode = GGML_CUDA_KEEPALIVE_REQUEST;
    } else if (strcmp(mode, "idle") == 0) {
        c.mode = GGML_CUDA_KEEPALIVE_IDLE;
    } else {
        c.mode = GGML_CUDA_KEEPALIVE_OFF;
    }
    c.idle_ms = c.mode == GGML_CUDA_KEEPALIVE_IDLE ? 120000 : 1000;
    if (ms != nullptr && ms[0] != '\0') {
        const long v = strtol(ms, nullptr, 10);
        c.idle_ms = (uint32_t) (v < 10 ? 10 : v > 3600000 ? 3600000 : v);
    }
    if (poll_us != nullptr && poll_us[0] != '\0') {
        const long v = strtol(poll_us, nullptr, 10);
        c.poll_us = (uint32_t) (v < 1 ? 1 : v > 100000 ? 100000 : v);
    }
    return c;
}

static inline const char * ggml_cuda_keepalive_mode_name(ggml_cuda_keepalive_mode m) {
    switch (m) {
        case GGML_CUDA_KEEPALIVE_DECODE:  return "decode";
        case GGML_CUDA_KEEPALIVE_REQUEST: return "request";
        case GGML_CUDA_KEEPALIVE_IDLE:    return "idle";
        default:                          return "off";
    }
}

// The word the host writes and a block polls: the run's generation in the low 32 bits (a block runs while it equals
// its own; the host ends a run by writing 0), and the card's beat in the high 32 bits (one more after every graph).
static inline uint64_t ggml_cuda_keepalive_word(uint32_t gen, uint32_t beat) {
    return ((uint64_t) beat << 32) | gen;
}

enum ggml_cuda_keepalive_state {
    GGML_CUDA_KEEPALIVE_RUN     = 0,
    GGML_CUDA_KEEPALIVE_STOPPED = 1, // the host ended the run
    GGML_CUDA_KEEPALIVE_TIMEOUT = 2, // no beat for the timeout
};

// what a block has seen of the beat
struct ggml_cuda_keepalive_watch {
    uint32_t           beat;
    unsigned long long t_beat; // ns, the block's clock, when the beat last changed (or the run started)
};

// A block's decision at one poll of the word, at time now_ns.
GGML_CUDA_KEEPALIVE_HD static inline int ggml_cuda_keepalive_poll(uint64_t word, uint32_t gen, unsigned long long now_ns,
                                                                  unsigned long long idle_ns, ggml_cuda_keepalive_watch & w) {
    if ((uint32_t) word != gen) {
        return GGML_CUDA_KEEPALIVE_STOPPED;
    }
    const uint32_t beat = (uint32_t) (word >> 32);
    if (beat != w.beat) {
        w.beat   = beat;
        w.t_beat = now_ns;
        return GGML_CUDA_KEEPALIVE_RUN;
    }
    return now_ns - w.t_beat >= idle_ns ? GGML_CUDA_KEEPALIVE_TIMEOUT : GGML_CUDA_KEEPALIVE_RUN;
}

// The SM clock the block measured while it ran (its clock64() against the global timer, every
// GGML_CUDA_KEEPALIVE_SAMPLE_NS), in bins of 50 MHz.
#define GGML_CUDA_KEEPALIVE_SAMPLE_NS 100000000ull
#define GGML_CUDA_KEEPALIVE_BINS      48

// how often a beat also asks whether a running block has ended (a stream query)
#define GGML_CUDA_KEEPALIVE_CHECK_MS  250

// What a block leaves in mapped host memory when it ends; gen is written last.
struct ggml_cuda_keepalive_report {
    uint32_t           reason; // GGML_CUDA_KEEPALIVE_STOPPED or _TIMEOUT
    uint32_t           pad;
    unsigned long long polls;
    unsigned long long ns;     // from the start to the end
    uint32_t           hist[GGML_CUDA_KEEPALIVE_BINS];
    volatile uint32_t  gen;
};

// A card's runs, summed over the reports the host has read.
struct ggml_cuda_keepalive_stats {
    uint64_t starts   = 0;
    uint64_t stopped  = 0; // ended by the host: a capture, a free, a host allocation, a device-wide synchronize
    uint64_t timeouts = 0; // ended themselves: no graph for the timeout
    uint64_t lost     = 0; // ended with no report of this run (a bug)
    double   seconds  = 0;
    uint64_t hist[GGML_CUDA_KEEPALIVE_BINS] = {};

    void add(const ggml_cuda_keepalive_report & r) {
        stopped  += r.reason == GGML_CUDA_KEEPALIVE_STOPPED;
        timeouts += r.reason == GGML_CUDA_KEEPALIVE_TIMEOUT;
        seconds  += r.ns * 1e-9;
        for (int i = 0; i < GGML_CUDA_KEEPALIVE_BINS; ++i) {
            hist[i] += r.hist[i];
        }
    }

    uint64_t samples() const {
        uint64_t n = 0;
        for (uint64_t h : hist) {
            n += h;
        }
        return n;
    }

    // the clock below which a share q of the samples lie, MHz (the bin's middle)
    int mhz_quantile(double q) const {
        const uint64_t n = samples();
        uint64_t c = 0;
        for (int i = 0; i < GGML_CUDA_KEEPALIVE_BINS; ++i) {
            c += hist[i];
            if (n > 0 && c >= q * n) {
                return i * 50 + 25;
            }
        }
        return 0;
    }

    // the share of samples in [lo, hi) MHz, at the bins' resolution
    double share(int lo_mhz, int hi_mhz) const {
        const uint64_t n = samples();
        uint64_t c = 0;
        for (int i = 0; i < GGML_CUDA_KEEPALIVE_BINS; ++i) {
            c += (i * 50 >= lo_mhz && i * 50 < hi_mhz) ? hist[i] : 0;
        }
        return n ? (double) c / n : 0.0;
    }
};

// Starts and stops the cards' blocks. Ops is what a card does (keepalive.cu on CUDA, a thread in the test):
//   void     write(int dev, uint64_t word)          the word into the card's mapped host memory
//   bool     launch(int dev, uint32_t gen, uint32_t beat, uint32_t idle_ms, uint32_t poll_us)
//                                                   a block on the card's keep-alive stream; false if it failed
//   bool     done(int dev)                          the block has ended (does not wait)
//   void     wait(int dev)                          until the block has ended
//   const ggml_cuda_keepalive_report & report(int dev)
// Every method takes the controller's lock, so a free on one thread and a graph on another are safe.
template <typename Ops, int MAX_DEVICES>
struct ggml_cuda_keepalive_ctl {
    using clock = std::chrono::steady_clock;

    ggml_cuda_keepalive_config cfg;
    Ops &                      ops;
    std::mutex                 mutex;

    struct dev_state {
        bool              seen    = false; // a graph has been computed on the card
        bool              running = false; // launched and not yet seen to end
        uint32_t          gen     = 0;
        uint32_t          beat    = 0;
        clock::time_point last_beat;       // when the host last wrote a beat to a running block
        clock::time_point last_check;      // when the host last asked whether the block has ended
        ggml_cuda_keepalive_stats stats;
    };
    dev_state devs[MAX_DEVICES];
    uint32_t  next_gen = 1;

    ggml_cuda_keepalive_ctl(const ggml_cuda_keepalive_config & cfg, Ops & ops) : cfg(cfg), ops(ops) {}

    bool on() const { return cfg.mode != GGML_CUDA_KEEPALIVE_OFF; }

    // before a graph's work is enqueued on the card: a capture must not begin while its block runs
    void graph_begin(int dev, bool capture) {
        if (!on() || !capture) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex);
        stop_locked(dev);
    }

    // after a graph's work is enqueued on the card: a beat for its block, which is started if it is not running
    void graph_end(int dev) {
        if (!on() || dev < 0 || dev >= MAX_DEVICES) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex);
        devs[dev].seen = true;
        beat_locked(dev);
    }

    // llama-server took a request (request mode): the block of every card that has computed a graph (the warm-up's
    // cards: the request's), ahead of the prompt
    void kick() {
        if (cfg.mode != GGML_CUDA_KEEPALIVE_REQUEST) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex);
        for (int dev = 0; dev < MAX_DEVICES; ++dev) {
            if (devs[dev].seen) {
                beat_locked(dev);
            }
        }
    }

    // before a free, a host allocation or a device-wide synchronize: every block ended
    void quiesce() {
        if (!on()) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex);
        // every stop first, then every wait: the stops overlap
        for (int dev = 0; dev < MAX_DEVICES; ++dev) {
            if (devs[dev].running) {
                ops.write(dev, ggml_cuda_keepalive_word(0, devs[dev].beat));
            }
        }
        for (int dev = 0; dev < MAX_DEVICES; ++dev) {
            if (devs[dev].running) {
                finish_locked(dev);
            }
        }
    }

    // the card's stats so far (ending its block first, so that its report counts); the stats are reset
    ggml_cuda_keepalive_stats take_stats(int dev) {
        std::lock_guard<std::mutex> lock(mutex);
        stop_locked(dev);
        ggml_cuda_keepalive_stats s = devs[dev].stats;
        devs[dev].stats = ggml_cuda_keepalive_stats();
        return s;
    }

    bool running(int dev) {
        std::lock_guard<std::mutex> lock(mutex);
        return devs[dev].running;
    }

  private:
    void stop_locked(int dev) {
        if (dev < 0 || dev >= MAX_DEVICES || !devs[dev].running) {
            return;
        }
        ops.write(dev, ggml_cuda_keepalive_word(0, devs[dev].beat));
        finish_locked(dev);
    }

    // wait for the card's block to end and take its report
    void finish_locked(int dev) {
        dev_state & d = devs[dev];
        ops.wait(dev);
        const ggml_cuda_keepalive_report & r = ops.report(dev);
        if (r.gen == d.gen) {
            d.stats.add(r);
        } else {
            d.stats.lost++;
        }
        d.running = false;
    }

    void beat_locked(int dev) {
        if (dev < 0 || dev >= MAX_DEVICES) {
            return;
        }
        dev_state & d = devs[dev];
        const clock::time_point now = clock::now();
        d.beat++;
        if (d.running) {
            // A block whose last beat is less than its timeout ago (by a margin far above a PCIe read) cannot end by
            // itself before it reads this one. Any other may be ending as the beat is written: it is ended and a
            // fresh one started, which costs one stop after each gap of about the timeout. A block that ended while
            // this thread was held up between taking a beat's time and writing it is found by the check every
            // GGML_CUDA_KEEPALIVE_CHECK_MS.
            const auto since = std::chrono::duration_cast<std::chrono::milliseconds>(now - d.last_beat).count();
            if (since + 5 >= (long long) cfg.idle_ms) {
                stop_locked(dev);
            } else if (now - d.last_check >= std::chrono::milliseconds(GGML_CUDA_KEEPALIVE_CHECK_MS)) {
                d.last_check = now;
                if (ops.done(dev)) {
                    finish_locked(dev);
                }
            }
        }
        if (d.running) {
            ops.write(dev, ggml_cuda_keepalive_word(d.gen, d.beat));
        } else {
            const uint32_t gen = next_gen++;
            if (next_gen == 0) {
                next_gen = 1;
            }
            ops.write(dev, ggml_cuda_keepalive_word(gen, d.beat));
            if (ops.launch(dev, gen, d.beat, cfg.idle_ms, cfg.poll_us)) {
                d.gen        = gen;
                d.running    = true;
                d.last_check = now;
                d.stats.starts++;
            }
        }
        d.last_beat = now;
    }
};
