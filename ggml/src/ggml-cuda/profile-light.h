#pragma once

// The parts of GGML_CUDA_PROFILE's lighter modes (GGML_CUDA_PROFILE_MODE, ggml-cuda.cu) that need no GPU: the mode
// string, the layer a node belongs to, layer mode's segments, sample mode's choice of launch groups, and the sets of
// CUDA-graph executables sample mode keeps per phase. tests/test-cuda-profile-light.cpp checks them on a CPU.

#include <cctype>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

enum ggml_cuda_profile_mode {
    GGML_CUDA_PROFILE_MODE_GROUP,   // an event between every two launch groups, a stream sync after every evaluation
    GGML_CUDA_PROFILE_MODE_LAYER,   // an event where the layer changes, and a pair around the whole evaluation
    GGML_CUDA_PROFILE_MODE_SAMPLE,  // a pair around every N-th launch group (phase rotating per evaluation), and around the evaluation
    GGML_CUDA_PROFILE_MODE_GRAPH,   // a pair around the whole evaluation only
};

#define GGML_CUDA_PROFILE_SAMPLE_MAX 64

// "group" (also nullptr or ""), "layer", "sample=<N>" with 1 <= N <= GGML_CUDA_PROFILE_SAMPLE_MAX, "graph".
// n is N for sample mode and 1 otherwise. False for anything else.
static inline bool ggml_cuda_profile_parse_mode(const char * s, ggml_cuda_profile_mode & mode, int & n) {
    n = 1;
    if (s == nullptr || s[0] == '\0' || strcmp(s, "group") == 0) {
        mode = GGML_CUDA_PROFILE_MODE_GROUP;
        return true;
    }
    if (strcmp(s, "layer") == 0) {
        mode = GGML_CUDA_PROFILE_MODE_LAYER;
        return true;
    }
    if (strcmp(s, "graph") == 0) {
        mode = GGML_CUDA_PROFILE_MODE_GRAPH;
        return true;
    }
    if (strncmp(s, "sample=", 7) == 0) {
        const char * d = s + 7;
        if (d[0] == '\0' || d[0] == '0' || strlen(d) > 2) {
            return false;
        }
        for (const char * c = d; *c; ++c) {
            if (!isdigit((unsigned char) *c)) {
                return false;
            }
        }
        const int v = atoi(d);
        if (v < 1 || v > GGML_CUDA_PROFILE_SAMPLE_MAX) {
            return false;
        }
        mode = GGML_CUDA_PROFILE_MODE_SAMPLE;
        n    = v;
        return true;
    }
    return false;
}

static inline const char * ggml_cuda_profile_mode_name(const ggml_cuda_profile_mode mode) {
    switch (mode) {
        case GGML_CUDA_PROFILE_MODE_GROUP:  return "group";
        case GGML_CUDA_PROFILE_MODE_LAYER:  return "layer";
        case GGML_CUDA_PROFILE_MODE_SAMPLE: return "sample";
        case GGML_CUDA_PROFILE_MODE_GRAPH:  return "graph";
    }
    return "?";
}

#define GGML_CUDA_PROFILE_LAYER_UNKNOWN (-1)  // the name says nothing: the group stays in the current segment
#define GGML_CUDA_PROFILE_LAYER_OUTPUT  (-2)  // llama.cpp's "result_*" tensors (final norm, output layer)
#define GGML_CUDA_PROFILE_LAYER_INPUT   (-3)  // llama.cpp's "inp_*" tensors (embeddings, positions, masks)

// "blk.12.ffn_up_exps.weight" -> 12
static inline int ggml_cuda_profile_layer_of_weight(const char * name) {
    if (name == nullptr || strncmp(name, "blk.", 4) != 0) {
        return GGML_CUDA_PROFILE_LAYER_UNKNOWN;
    }
    int n = 0;
    int i = 4;
    for (; isdigit((unsigned char) name[i]) && i < 4 + 6; ++i) {
        n = 10*n + (name[i] - '0');
    }
    return i > 4 && name[i] == '.' ? n : GGML_CUDA_PROFILE_LAYER_UNKNOWN;
}

// llama.cpp's graph callback names a layer's tensors "<name>-<layer>" ("attn_norm-12"); views and copies append
// " (view)", " (reshaped)", " (copy of ...)" and the like, so only the first word counts.
static inline int ggml_cuda_profile_layer_of_name(const char * name) {
    if (name == nullptr || name[0] == '\0') {
        return GGML_CUDA_PROFILE_LAYER_UNKNOWN;
    }
    if (strncmp(name, "result", 6) == 0) {
        return GGML_CUDA_PROFILE_LAYER_OUTPUT;
    }
    if (strncmp(name, "inp_", 4) == 0) {
        return GGML_CUDA_PROFILE_LAYER_INPUT;
    }
    const size_t len = strcspn(name, " ");
    size_t j = len;
    while (j > 0 && isdigit((unsigned char) name[j - 1])) {
        --j;
    }
    if (j < len && len - j <= 6 && j >= 2 && name[j - 1] == '-') {
        return atoi(name + j);
    }
    return ggml_cuda_profile_layer_of_weight(name);
}

static inline std::string ggml_cuda_profile_layer_label(const int layer) {
    switch (layer) {
        case GGML_CUDA_PROFILE_LAYER_OUTPUT:  return "output";
        case GGML_CUDA_PROFILE_LAYER_INPUT:   return "input";
        case GGML_CUDA_PROFILE_LAYER_UNKNOWN: return "other";
    }
    return "layer " + std::to_string(layer);
}

// Layer mode's segments of one evaluation. begin() comes with the evaluation's first event (event 0); group() is called
// before each launch group with the group's layer and returns true when an event has to be recorded before the group,
// because the segment before it ends there; end() returns true when a last event closes the last segment. Segment j
// of `segs` then runs from event j to event j + 1. A group whose layer is unknown stays in the current segment.
struct ggml_cuda_profile_layer_plan {
    struct segment {
        int layer;
        int first;     // its first launch group
        int n_groups;
    };
    std::vector<segment> segs;
    segment cur = { GGML_CUDA_PROFILE_LAYER_UNKNOWN, 0, 0 };
    int     k   = 0;

    void begin() {
        segs.clear();
        cur = { GGML_CUDA_PROFILE_LAYER_UNKNOWN, 0, 0 };
        k   = 0;
    }

    bool group(const int layer) {
        bool record = false;
        if (layer != GGML_CUDA_PROFILE_LAYER_UNKNOWN && layer != cur.layer) {
            if (cur.n_groups > 0) {
                segs.push_back(cur);
                record = true;
            }
            cur = { layer, k, 0 };
        }
        cur.n_groups++;
        k++;
        return record;
    }

    bool end() {
        if (cur.n_groups == 0) {
            return false;
        }
        segs.push_back(cur);
        cur.n_groups = 0;
        return true;
    }
};

// Sample mode: launch group k of an evaluation in phase `phase` (0..n-1) is timed when k % n == phase, and an
// evaluation's phase is its index among the evaluations of the same graph, modulo n; so n evaluations in a row time
// every group once.
static inline bool ggml_cuda_profile_sampled(const int k, const int n, const int phase) {
    return n > 0 && k % n == phase;
}

// Sample mode under CUDA graphs: each phase's events are nodes of its own captured graphs, so a graph keeps one set of
// chunk executables per phase. `cur` (the graph's own set, the one every other code path uses) always holds the active
// phase's set; the others wait in `parked`. A set captured before the graph's latest warm-up (`capture_gen`, bumped at
// each one) has stale pointers and must be captured again before it is launched.
template <typename Set>
struct ggml_cuda_profile_phase_sets {
    std::vector<Set>      parked;
    std::vector<uint64_t> gen;    // the capture generation of each phase's set
    int                   active = 0;

    void select(Set & cur, const int phase, const int n_phases) {
        if ((int) parked.size() < n_phases) {
            parked.resize(n_phases);
            gen.resize(n_phases, 0);
        }
        if (phase == active) {
            return;
        }
        std::swap(cur, parked[active]);  // parked[active] is empty while it is active
        std::swap(cur, parked[phase]);
        active = phase;
    }

    bool stale(const uint64_t capture_gen) const {
        return !gen.empty() && gen[active] != capture_gen;
    }

    void captured(const uint64_t capture_gen) {
        if (!gen.empty()) {
            gen[active] = capture_gen;
        }
    }
};
