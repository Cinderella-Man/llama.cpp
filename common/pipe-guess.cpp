#include "pipe-guess.h"
#include "pipe-store.h"

#include <algorithm>

uint64_t common_pipe_guesser::key(const llama_token * p, int n) {
    // FNV-1a over the token bytes, mixed with the context length
    uint64_t h = 0xcbf29ce484222325ULL;
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n*sizeof(llama_token); i++) {
        h ^= b[i];
        h *= 0x100000001b3ULL;
    }
    return h ^ (uint64_t) n;
}

void common_pipe_guesser::index(int32_t i) {
    pos_of[hist[i]].push_back(i);
}

void common_pipe_guesser::reset(const llama_token * tokens, size_t n) {
    hist.assign(tokens, tokens + n);
    pos_of.clear();
    for (int32_t i = 0; i < (int32_t) hist.size(); i++) {
        index(i);
    }
    counts.clear();
    count_from = hist.size() > max_count ? hist.size() - max_count : 0;
    n_counted  = count_from;
}

void common_pipe_guesser::push(llama_token tok) {
    hist.push_back(tok);
    index((int32_t) hist.size() - 1);
}

void common_pipe_guesser::count_to(size_t n) {
    for (size_t j = std::max(n_counted, count_from + 1); j < n; j++) {
        for (int ln = 1; ln <= max_ctx && (size_t) ln <= j - count_from; ln++) {
            auto & v = counts[key(&hist[j - ln], ln)];
            bool found = false;
            for (auto & e : v) {
                if (e.first == hist[j]) {
                    e.second++;
                    found = true;
                    break;
                }
            }
            if (!found) {
                v.push_back({ hist[j], 1 });
            }
        }
    }
    n_counted = std::max(n_counted, n);
}

llama_token common_pipe_guesser::copy_guess() const {
    const int n = (int) hist.size();
    if (n < 2) {
        return -1;
    }
    const auto it = pos_of.find(hist[n - 1]);
    if (it == pos_of.end()) {
        return -1;
    }
    const auto & cand = it->second;

    int best_len = 0;
    int best_end = -1;
    int n_seen   = 0;
    for (auto c = cand.rbegin(); c != cand.rend() && n_seen < max_cand; ++c) {
        const int i = *c;
        if (i >= n - 1) {
            continue; // the last token itself
        }
        n_seen++;
        int len = 1;
        while (len < max_copy && i - len >= 0 && hist[i - len] == hist[n - 1 - len]) {
            len++;
        }
        if (len > best_len) {
            best_len = len;
            best_end = i;
            if (len == max_copy) {
                break;
            }
        }
    }
    if (best_len < min_match) {
        return -1;
    }
    return hist[best_end + 1];
}

llama_token common_pipe_guesser::cont_guess(int min_len) const {
    const int n = (int) hist.size();
    for (int ln = std::min(max_ctx, n); ln >= min_len; ln--) {
        const auto it = counts.find(key(&hist[n - ln], ln));
        if (it == counts.end()) {
            continue;
        }
        llama_token best = -1;
        int32_t     bc   = 0;
        for (const auto & e : it->second) {
            if (e.second > bc) {
                best = e.first;
                bc   = e.second;
            }
        }
        return best;
    }
    return -1;
}

llama_token common_pipe_guesser::guess(const llama_token * inflight, size_t n_inflight) {
    if (kind == NONE) {
        return -1;
    }

    // the continuation counts cover committed tokens only
    const size_t n_committed = hist.size();
    if (kind == CONT) {
        count_to(n_committed);
    }

    // the tokens in flight extend the history for the lookups, and leave again
    for (size_t i = 0; i < n_inflight; i++) {
        push(inflight[i]);
    }

    llama_token g = copy_guess();
    if (g < 0 && kind == CONT && (store || !cont_any)) {
        g = cont_guess(2);
        if (g < 0 && store) {
            g = common_pipe_store_guess(store, hist.data(), (int) hist.size());
            n_store += g >= 0;
        }
    }
    if (g < 0 && kind == CONT && cont_any) {
        g = cont_guess();
    }

    for (size_t i = 0; i < n_inflight; i++) {
        pos_of[hist.back()].pop_back();
        hist.pop_back();
    }

    return g;
}
