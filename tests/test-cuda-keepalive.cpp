// GGML_CUDA_KEEPALIVE's parts that run without a GPU (ggml/src/ggml-cuda/keepalive-policy.h): the settings, a block's
// decision at each poll, and the controller that starts the cards' blocks after graphs and ends them before captures,
// frees and device-wide synchronizes. A thread stands in for each card's block: the kernel's loop over the same
// decision function, on the host's clock, reading the time and then the word as the kernel does.
// Usage: test-cuda-keepalive   (exit status 0 when every check passes)

#include "../ggml/src/ggml-cuda/keepalive-policy.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

static int n_checks = 0;
static int n_fail   = 0;

#define CHECK(cond) do { n_checks++; if (!(cond)) { n_fail++; fprintf(stderr, "%s:%d: failed: %s\n", __FILE__, __LINE__, #cond); } } while (0)

static unsigned long long now_ns() {
    return (unsigned long long) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void sleep_ms(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

// the cards: a word, a report and a thread each
struct mock_ops {
    static constexpr int N = 4;
    struct card {
        std::mutex                 m;
        uint64_t                   word = 0;
        ggml_cuda_keepalive_report report{};
        std::thread                th;
        std::atomic<bool>          ended{true};
        std::atomic<bool>          force_end{false}; // the test ends the block as its timeout would
        std::atomic<uint32_t>      beat_seen{0};
        int                        launches = 0;
        int                        writes   = 0;
    };
    card cards[N];
    std::atomic<int> bad_launches{0}; // a launch while the card's last block was not waited for

    void write(int dev, uint64_t w) {
        std::lock_guard<std::mutex> lock(cards[dev].m);
        cards[dev].word = w;
        cards[dev].writes++;
    }

    bool launch(int dev, uint32_t gen, uint32_t beat, uint32_t idle_ms, uint32_t poll_us) {
        card & c = cards[dev];
        if (c.th.joinable()) {
            bad_launches++;
            c.th.join();
        }
        c.launches++;
        c.ended      = false;
        c.force_end  = false;
        c.report.gen = 0;
        c.th = std::thread([&c, gen, beat, idle_ms, poll_us] {
            const unsigned long long t0 = now_ns();
            ggml_cuda_keepalive_watch w = { beat, t0 };
            unsigned long long polls = 0;
            int r;
            for (;;) {
                unsigned long long t;
                uint64_t v;
                {
                    std::lock_guard<std::mutex> lock(c.m);
                    t = now_ns();
                    v = c.word;
                }
                polls++;
                r = ggml_cuda_keepalive_poll(v, gen, t, (unsigned long long) idle_ms * 1000000ull, w);
                if (r == GGML_CUDA_KEEPALIVE_RUN && c.force_end) {
                    r = GGML_CUDA_KEEPALIVE_TIMEOUT;
                }
                if (r != GGML_CUDA_KEEPALIVE_RUN) {
                    break;
                }
                c.beat_seen = w.beat;
                std::this_thread::sleep_for(std::chrono::microseconds(poll_us));
            }
            c.report.reason = (uint32_t) r;
            c.report.polls  = polls;
            c.report.ns     = now_ns() - t0;
            for (uint32_t & h : c.report.hist) {
                h = 0;
            }
            c.report.hist[1885 / 50] = 1; // one sample at a P106-090's boost clock
            c.report.gen = gen;
            c.ended      = true;
        });
        return true;
    }

    bool done(int dev) {
        return cards[dev].ended.load();
    }

    void wait(int dev) {
        if (cards[dev].th.joinable()) {
            cards[dev].th.join();
        }
    }

    const ggml_cuda_keepalive_report & report(int dev) {
        return cards[dev].report;
    }

    ~mock_ops() {
        for (card & c : cards) {
            c.force_end = true;
            if (c.th.joinable()) {
                c.th.join();
            }
        }
    }
};

using ctl_t = ggml_cuda_keepalive_ctl<mock_ops, mock_ops::N>;

static ggml_cuda_keepalive_config cfg_of(const char * mode, const char * ms = nullptr, const char * poll = "50") {
    return ggml_cuda_keepalive_parse(mode, ms, poll);
}

static void test_parse() {
    ggml_cuda_keepalive_config c = ggml_cuda_keepalive_parse(nullptr, nullptr, nullptr);
    CHECK(c.mode == GGML_CUDA_KEEPALIVE_OFF && c.idle_ms == 1000 && c.poll_us == 50);
    CHECK(ggml_cuda_keepalive_parse("", nullptr, nullptr).mode == GGML_CUDA_KEEPALIVE_OFF);
    CHECK(ggml_cuda_keepalive_parse("0", nullptr, nullptr).mode == GGML_CUDA_KEEPALIVE_OFF);
    CHECK(ggml_cuda_keepalive_parse("on", nullptr, nullptr).mode == GGML_CUDA_KEEPALIVE_OFF);
    CHECK(ggml_cuda_keepalive_parse("decodes", nullptr, nullptr).mode == GGML_CUDA_KEEPALIVE_OFF);
    c = ggml_cuda_keepalive_parse("1", nullptr, nullptr);
    CHECK(c.mode == GGML_CUDA_KEEPALIVE_DECODE && c.idle_ms == 1000);
    CHECK(ggml_cuda_keepalive_parse("decode", nullptr, nullptr).mode == GGML_CUDA_KEEPALIVE_DECODE);
    c = ggml_cuda_keepalive_parse("request", nullptr, nullptr);
    CHECK(c.mode == GGML_CUDA_KEEPALIVE_REQUEST && c.idle_ms == 1000);
    c = ggml_cuda_keepalive_parse("idle", nullptr, nullptr);
    CHECK(c.mode == GGML_CUDA_KEEPALIVE_IDLE && c.idle_ms == 120000);
    c = ggml_cuda_keepalive_parse("idle", "5000", "200");
    CHECK(c.idle_ms == 5000 && c.poll_us == 200);
    CHECK(ggml_cuda_keepalive_parse("decode", "3", nullptr).idle_ms == 10);
    CHECK(ggml_cuda_keepalive_parse("decode", "99999999", nullptr).idle_ms == 3600000);
    CHECK(ggml_cuda_keepalive_parse("decode", "", nullptr).idle_ms == 1000);
    CHECK(ggml_cuda_keepalive_parse("decode", nullptr, "0").poll_us == 1);
    CHECK(ggml_cuda_keepalive_parse("decode", nullptr, "1000000").poll_us == 100000);
    CHECK(strcmp(ggml_cuda_keepalive_mode_name(GGML_CUDA_KEEPALIVE_REQUEST), "request") == 0);
}

static void test_poll() {
    ggml_cuda_keepalive_watch w = { 5, 1000 };
    CHECK(ggml_cuda_keepalive_poll(ggml_cuda_keepalive_word(7, 5), 7, 1500, 1000, w) == GGML_CUDA_KEEPALIVE_RUN);
    CHECK(ggml_cuda_keepalive_poll(ggml_cuda_keepalive_word(7, 5), 7, 1999, 1000, w) == GGML_CUDA_KEEPALIVE_RUN);
    CHECK(ggml_cuda_keepalive_poll(ggml_cuda_keepalive_word(7, 5), 7, 2000, 1000, w) == GGML_CUDA_KEEPALIVE_TIMEOUT);
    // a new beat resets the timeout, however late
    CHECK(ggml_cuda_keepalive_poll(ggml_cuda_keepalive_word(7, 6), 7, 2500, 1000, w) == GGML_CUDA_KEEPALIVE_RUN);
    CHECK(w.beat == 6 && w.t_beat == 2500);
    CHECK(ggml_cuda_keepalive_poll(ggml_cuda_keepalive_word(7, 6), 7, 3499, 1000, w) == GGML_CUDA_KEEPALIVE_RUN);
    // the host's stop wins over a beat and over a timeout
    CHECK(ggml_cuda_keepalive_poll(ggml_cuda_keepalive_word(0, 7), 7, 2600, 1000, w) == GGML_CUDA_KEEPALIVE_STOPPED);
    CHECK(ggml_cuda_keepalive_poll(ggml_cuda_keepalive_word(0, 6), 7, 9999, 1000, w) == GGML_CUDA_KEEPALIVE_STOPPED);
    // another run's generation stops this one
    CHECK(ggml_cuda_keepalive_poll(ggml_cuda_keepalive_word(8, 6), 7, 2600, 1000, w) == GGML_CUDA_KEEPALIVE_STOPPED);
    // the beat wraps
    ggml_cuda_keepalive_watch v = { 0xffffffffu, 0 };
    CHECK(ggml_cuda_keepalive_poll(ggml_cuda_keepalive_word(1, 0), 1, 5000, 1000, v) == GGML_CUDA_KEEPALIVE_RUN && v.beat == 0);
}

static void test_stats() {
    ggml_cuda_keepalive_stats s;
    ggml_cuda_keepalive_report r{};
    r.reason = GGML_CUDA_KEEPALIVE_STOPPED;
    r.ns     = 2000000000ull;
    r.hist[1354 / 50] = 3; // 1350-1399: base
    r.hist[1885 / 50] = 7; // 1850-1899: boost
    s.add(r);
    r.reason = GGML_CUDA_KEEPALIVE_TIMEOUT;
    r.hist[1354 / 50] = 0;
    r.hist[1885 / 50] = 10;
    s.add(r);
    CHECK(s.stopped == 1 && s.timeouts == 1 && s.samples() == 20);
    CHECK(s.seconds > 3.99 && s.seconds < 4.01);
    CHECK(s.share(0, 1400) == 3.0 / 20 && s.share(1800, 100000) == 17.0 / 20);
    CHECK(s.mhz_quantile(0.1) == 1375 && s.mhz_quantile(0.5) == 1875 && s.mhz_quantile(0.9) == 1875);
}

static void test_off() {
    mock_ops ops;
    ctl_t ka(cfg_of(nullptr), ops);
    CHECK(!ka.on());
    ka.graph_begin(0, true);
    ka.graph_end(0);
    ka.graph_end(1);
    ka.kick();
    ka.quiesce();
    for (const auto & c : ops.cards) {
        CHECK(c.launches == 0 && c.writes == 0);
    }
    CHECK(ka.take_stats(0).starts == 0);
}

static void test_decode() {
    mock_ops ops;
    ctl_t ka(cfg_of("decode", "1000"), ops);
    ka.graph_end(0);
    CHECK(ops.cards[0].launches == 1 && ka.running(0));
    const uint32_t gen0 = ka.devs[0].gen;
    for (int i = 0; i < 20; ++i) {
        ka.graph_end(0);
        sleep_ms(2);
    }
    CHECK(ops.cards[0].launches == 1);
    sleep_ms(20);
    CHECK(ops.cards[0].beat_seen == ka.devs[0].beat); // the block follows the beats
    ka.graph_begin(0, false);
    CHECK(ka.running(0) && !ops.cards[0].ended);
    // a capture ends the block, and it reports that the host ended it
    ka.graph_begin(0, true);
    CHECK(!ka.running(0) && ops.cards[0].ended);
    CHECK(ops.cards[0].report.reason == GGML_CUDA_KEEPALIVE_STOPPED && ops.cards[0].report.gen == gen0);
    // the next graph starts a new run
    ka.graph_end(0);
    CHECK(ops.cards[0].launches == 2 && ka.running(0) && ka.devs[0].gen != gen0);
    ka.graph_end(1);
    CHECK(ops.cards[1].launches == 1 && ops.cards[2].launches == 0);
    // a free ends every block
    ka.quiesce();
    CHECK(!ka.running(0) && !ka.running(1) && ops.cards[0].ended && ops.cards[1].ended);
    ggml_cuda_keepalive_stats s = ka.take_stats(0);
    CHECK(s.starts == 2 && s.stopped == 2 && s.timeouts == 0 && s.lost == 0 && s.samples() == 2);
    CHECK(ka.take_stats(0).starts == 0); // taken stats are reset
    CHECK(ka.take_stats(1).starts == 1);
    CHECK(ops.bad_launches == 0);
}

static void test_timeout() {
    mock_ops ops;
    ctl_t ka(cfg_of("decode", "50"), ops);
    ka.graph_end(0);
    sleep_ms(300);
    CHECK(ops.cards[0].ended); // the block ended itself
    CHECK(ka.running(0));      // the host has not looked yet
    // a beat after a gap of the timeout ends the old run (already over: its report says timeout) and starts one
    ka.graph_end(0);
    CHECK(ops.cards[0].launches == 2 && ka.running(0));
    ka.quiesce();
    ggml_cuda_keepalive_stats s = ka.take_stats(0);
    CHECK(s.starts == 2 && s.timeouts == 1 && s.stopped == 1 && s.lost == 0);
    CHECK(ops.bad_launches == 0);
}

static void test_beats_hold() {
    mock_ops ops;
    ctl_t ka(cfg_of("decode", "1000"), ops);
    // beats every 5 ms for 1.5 s, past the 1 s timeout: one run throughout
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(1500)) {
        ka.graph_end(0);
        sleep_ms(5);
    }
    CHECK(ops.cards[0].launches == 1 && !ops.cards[0].ended);
    ka.quiesce();
    ggml_cuda_keepalive_stats s = ka.take_stats(0);
    CHECK(s.starts == 1 && s.stopped == 1 && s.timeouts == 0);
}

static void test_check() {
    // a block that ended while the host thought it running (as after a held-up host thread) is found by the beats'
    // check within GGML_CUDA_KEEPALIVE_CHECK_MS, and a new one started
    mock_ops ops;
    ctl_t ka(cfg_of("decode", "60000"), ops);
    ka.graph_end(0);
    ops.cards[0].force_end = true;
    while (!ops.cards[0].ended) {
        sleep_ms(1);
    }
    const auto t0 = std::chrono::steady_clock::now();
    while (ops.cards[0].launches < 2 && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5)) {
        ka.graph_end(0);
        sleep_ms(10);
    }
    const auto took = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    CHECK(ops.cards[0].launches == 2 && ka.running(0));
    CHECK(took <= GGML_CUDA_KEEPALIVE_CHECK_MS + 200);
    ka.quiesce();
    ggml_cuda_keepalive_stats s = ka.take_stats(0);
    CHECK(s.starts == 2 && s.timeouts == 1 && s.stopped == 1);
    CHECK(ops.bad_launches == 0);
}

static void test_request() {
    {
        mock_ops ops;
        ctl_t ka(cfg_of("request"), ops);
        ka.kick(); // no card has computed a graph yet: nothing starts
        for (const auto & c : ops.cards) {
            CHECK(c.launches == 0);
        }
        ka.graph_end(0);
        ka.graph_end(2);
        ka.quiesce();
        // a request starts the blocks of the cards that have computed graphs, ahead of the prompt
        ka.kick();
        CHECK(ops.cards[0].launches == 2 && ops.cards[2].launches == 2);
        CHECK(ops.cards[1].launches == 0 && ops.cards[3].launches == 0);
        CHECK(ka.running(0) && ka.running(2));
        ka.kick(); // a second kick while they run: beats only
        CHECK(ops.cards[0].launches == 2 && ops.cards[2].launches == 2);
        ka.quiesce();
    }
    {
        mock_ops ops;
        ctl_t ka(cfg_of("decode"), ops);
        ka.graph_end(0);
        ka.quiesce();
        ka.kick(); // only request mode kicks
        CHECK(ops.cards[0].launches == 1 && !ka.running(0));
    }
}

static void test_idle() {
    mock_ops ops;
    ctl_t ka(cfg_of("idle"), ops);
    CHECK(ka.cfg.idle_ms == 120000);
    ka.graph_end(1);
    sleep_ms(200);
    CHECK(ka.running(1) && !ops.cards[1].ended); // the block outlives the gap between requests
    ka.quiesce();
    CHECK(ops.cards[1].ended && ops.cards[1].report.reason == GGML_CUDA_KEEPALIVE_STOPPED);
}

static void test_threads() {
    // graphs, captures and frees from three threads at once: every run is accounted for, none is lost
    mock_ops ops;
    ctl_t ka(cfg_of("decode", "20", "20"), ops);
    std::atomic<bool> stop{false};
    std::thread graphs([&] {
        for (int i = 0; i < 3000; ++i) {
            ka.graph_end(i % 3);
            if (i % 50 == 0) {
                sleep_ms(30); // past the timeout now and then
            }
        }
        stop = true;
    });
    std::thread captures([&] {
        while (!stop) {
            ka.graph_begin(1, true);
            sleep_ms(1);
        }
    });
    std::thread frees([&] {
        while (!stop) {
            ka.quiesce();
            sleep_ms(3);
        }
    });
    graphs.join();
    captures.join();
    frees.join();
    ka.quiesce();
    uint64_t starts = 0;
    for (int dev = 0; dev < mock_ops::N; ++dev) {
        CHECK(!ka.running(dev));
        const ggml_cuda_keepalive_stats s = ka.take_stats(dev);
        CHECK(s.lost == 0 && s.starts == s.stopped + s.timeouts);
        CHECK(s.starts == (uint64_t) ops.cards[dev].launches);
        starts += s.starts;
    }
    CHECK(starts > 3);
    CHECK(ops.bad_launches == 0);
}

int main() {
    test_parse();
    test_poll();
    test_stats();
    test_off();
    test_decode();
    test_timeout();
    test_beats_hold();
    test_check();
    test_request();
    test_idle();
    test_threads();
    printf("test-cuda-keepalive: %d checks, %d failed\n", n_checks, n_fail);
    return n_fail == 0 ? 0 : 1;
}
