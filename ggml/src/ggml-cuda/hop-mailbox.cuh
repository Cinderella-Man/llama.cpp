#pragma once

#include "common.cuh"

// ---- card hops: a mailbox in mapped host memory (GGML_CUDA_HOP_MAILBOX) ---------------------------------------------
//
// A copy between two cards without peer access goes card -> host -> card. The driver's peer copy and the staged hop
// (GGML_CUDA_HOP_STAGED) move it with DMA copies that the host enqueues per chunk, ordered by events between the two
// cards. The mailbox takes the host out of the copy path: per boundary (source card -> this card) a ring of slots in
// pinned host memory mapped into both cards, a sequence-number flag and an acknowledgement word per slot. A hop is one
// kernel on each card:
//   - on the source card's stream, after the work that wrote the tensor: a one-block kernel writes the bytes chunk by
//     chunk into the ring (stores over PCIe into mapped memory), and after each chunk __threadfence_system() and then
//     the chunk's sequence number into its slot's flag. Before it overwrites a slot it waits for the slot's previous
//     chunk to be acknowledged;
//   - on this card's stream: a one-block kernel polls the flag of the next chunk, reads every chunk that has arrived
//     straight from the ring (zero-copy loads that bypass the caches), writes the destination tensor and acknowledges
//     the slots it read.
// The chunks stream: the destination reads chunk c over its own link while the source writes chunk c + 1 over its
// link. No cudaMemcpy, no event and no host wait per hop; the destination's stream is ordered after the copy because
// its kernel runs on it, so the scheduler's expectation of cpy_tensor_async holds. The bytes are copied exactly.
//   GGML_CUDA_HOP_MAILBOX          0 (default): off; 1: copies between cards without peer access; 2: also between cards
//                                  with peer access (for checks on boards that have it). It takes precedence over
//                                  GGML_CUDA_HOP_STAGED for the copies it takes; the others use the staged hop or the
//                                  peer copy as before
//   GGML_CUDA_HOP_MAILBOX_CHUNKS   chunks per hop, 1..256 (default 8); each at least 256 bytes and at most a slot
//   GGML_CUDA_HOP_MAILBOX_SLOTS    slots per boundary, a power of two 2..256 (default 16)
//   GGML_CUDA_HOP_MAILBOX_SLOT_KB  KiB per slot, 1..4096 (default 32): a boundary holds slots x KiB of pinned, mapped
//                                  host memory, allocated on its first hop (512 KiB by default, 4 MiB for nine cards)
//   GGML_CUDA_HOP_MAILBOX_MAX_KB   hops larger than this take the staged hop or the peer copy (default 0: no limit)
//   GGML_CUDA_HOP_MAILBOX_STREAM   where the reading kernel runs: 0 on this card's compute stream; 1 (default) there
//                                  for hops that fit the ring (decode, verify passes) and on a copy stream of this card
//                                  for larger ones (prompt micro-batches), so that the source card is not held by the
//                                  ring while this card still computes the previous micro-batch; 2 the copy stream
//                                  always. On the copy stream the reader first waits on the scheduler's guard for the
//                                  input copy it writes (hop_guard), and this card's compute stream waits on it
//   GGML_CUDA_HOP_MAILBOX_THREADS  threads of each kernel, 64..1024, a multiple of 32 (default 256)
//   GGML_CUDA_HOP_MAILBOX_POLL_US  the longest pause between two polls of a flag, in microseconds, 0..1000 (default 4):
//                                  a waiting kernel polls at once, then backs off exponentially up to it, so that the
//                                  cards waiting for their turn (a token's later cards, most of the time) poll the host
//                                  links rarely
//   GGML_CUDA_HOP_MAILBOX_TIMEOUT_MS  a kernel that waits longer than this traps (default 60000): a mailbox that never
//                                  fills is a bug, and the process stops instead of computing with stale data
// Copies the mailbox does not take (fall back to the staged hop or the peer copy): sizes or addresses that are not
// multiples of 16 bytes, and hops above GGML_CUDA_HOP_MAILBOX_MAX_KB. A line per boundary is logged when its ring is
// made, and its counts when the context is freed. The waiting kernel keeps one block of the card busy while it polls.

struct ggml_cuda_hop_mailbox;

// copies nbytes from src (on ctx_src's card) to dst (on ctx_dst's card) through the boundary's mailbox, made on first
// use and kept in ctx_dst->hop_mailboxes; afterwards ctx_dst's stream is ordered after the copy, as the scheduler
// expects of cpy_tensor_async. Returns false, having enqueued nothing, for a copy the mailbox does not take
bool ggml_cuda_hop_mailbox_copy(ggml_backend_cuda_context * ctx_src, ggml_backend_cuda_context * ctx_dst, char * dst,
        const char * src, size_t nbytes);

// waits for the mailbox's last reads, logs its counts and frees it
void ggml_cuda_hop_mailbox_free(ggml_cuda_hop_mailbox * mb, ggml_backend_cuda_context * ctx_dst);
