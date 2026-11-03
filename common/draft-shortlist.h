#pragma once

#include <cstdint>
#include <cstdlib>

// When a draft uses its shortlisted head (Q592, llama_head_shortlist_*). A token of the target outside the list shows
// that the text has reached tokens the list lacks (a rare identifier, another script), so the next `rounds` drafts use
// the full head; the token then joins the list as part of the context.
struct common_draft_shortlist_policy {
    int32_t rounds = 2; // full-head drafts after a target token outside the list; 0: never fall back
    int32_t left   = 0;

    uint64_t n_short = 0; // drafts with the shortlist
    uint64_t n_full  = 0; // drafts with the full head
    uint64_t n_miss  = 0; // target tokens outside the list

    // last_in_list: whether the target's latest token was in the list; returns whether this draft uses the shortlist
    bool next(bool last_in_list) {
        if (!last_in_list) {
            n_miss++;
            left = rounds;
        }
        const bool use_short = left == 0;
        if (left > 0) {
            left--;
        }
        (use_short ? n_short : n_full)++;
        return use_short;
    }
};

// LLAMA_DRAFT_SHORTLIST=<k_low>[,<m_cap>[,<rounds>]]: the ids below k_low, up to m_cap more (the special tokens, the
// ids of LLAMA_DRAFT_SHORTLIST_FILE, then the context's tokens) and the fallback rounds; unset or k_low 0: off
struct common_draft_shortlist_config {
    uint32_t k_low  = 0;
    uint32_t m_cap  = 8192;
    int32_t  rounds = 2;

    static common_draft_shortlist_config from_string(const char * s) {
        common_draft_shortlist_config c;
        if (s == nullptr || *s == '\0') {
            return c;
        }
        char * end = nullptr;
        c.k_low = (uint32_t) std::strtoul(s, &end, 10);
        if (end && *end == ',') {
            c.m_cap = (uint32_t) std::strtoul(end + 1, &end, 10);
            if (end && *end == ',') {
                c.rounds = (int32_t) std::strtol(end + 1, &end, 10);
            }
        }
        return c;
    }
};
