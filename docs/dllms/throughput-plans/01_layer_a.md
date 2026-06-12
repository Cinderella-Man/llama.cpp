# Layer A: step-to-step KV caching for masked dLLMs - IMPLEMENTATION GUIDE

Status: verified design, not yet implemented. Everything below was checked first-hand:
reference algorithms read from cloned source (NOT papers alone), every engine interface
read from this repo with file:line, model geometry read from the GGUF header. An
implementer should need to discover NOTHING beyond this document.

References on disk (clone again if gone):
- /tmp/fast-dllm  = github.com/NVlabs/Fast-dLLM (read: v1/llada/generate.py,
  v1/dream/model/generation_utils_block.py)
- /tmp/dllm-cache = github.com/maomaocun/dLLM-cache (V-verify feature caching - Layer A2,
  NOT covered by this guide; future work)

---

## 1. What this is and why it works

Today: every denoising step re-runs the ENTIRE canvas (prompt + generation region)
through the transformer. steps x full-forward. For a 17-step draft on a 242-token
canvas, ~94% of all K/V computation is recomputing vectors that barely changed.

Why caching works (arXiv:2505.22618, Fig. "cache similarity"): the K/V activations of a
given position across ADJACENT denoising steps have cosine similarity > 0.94 - between
steps only a few tokens commit, so almost all K/V vectors are near-identical.

CRITICAL SUBTLETY (discovered reading the reference, contradicts the naive design): in a
fully bidirectional model there is NO exactly-static K/V - not even the prompt's.
Prompt K/V at layer l >= 2 depend on canvas hidden states at layer l-1 (attention mixes
canvas into prompt at every layer). Only layer-1 K/V (computed from raw embeddings) are
truly static. THEREFORE every caching scheme here is APPROXIMATE, made safe by
re-warming the cache periodically (per block). There is no "exact, zero-risk" phase -
quality must be gated by the kintsugi bench at every step.
(DiffusionGemma is different: it was TRAINED with a block-causal prompt - its prompt
attends only to itself - so DG's prompt cache IS exact. Dream/LLaDA were not.)

## 2. The reference algorithms (read from source, with the details papers omit)

### 2a. Vanilla (what we do now) - v1/llada/generate.py:86
Full [prompt|canvas] forward every step. Commit by confidence. NFE = steps.

### 2b. PREFIX cache - v1/llada/generate.py:132 (generate_with_prefix_cache)
Per BLOCK (canvas split into blocks of e.g. 32, processed left to right):
1. ONE full-canvas forward (use_cache=True) -> commit step-0 tokens. Commit policy:
   mask_index[:, current_block_end:] = 0 - commits allowed anywhere BEFORE/IN the
   current block, never after it.
2. TRIM the returned KV cache to [0 .. block_start) - prompt + already-committed blocks.
3. Every remaining step of the block forwards ONLY the SUFFIX x[:, block_start:]
   (block + all remaining masked blocks!), attending [trimmed cache | fresh suffix].
   Commits restricted to the current block (mask_index[:, block_length:] = 0).
4. Block fully committed -> next block (fresh full warm).
Per-step rows shrink from L to (L - block_start): saves the PREFIX recompute only.
The suffix is recomputed every step (no suffix staleness).

### 2c. DUAL cache - v1/llada/generate.py:211 (generate_with_dual_cache)
Same block loop, but the warm forward caches ALL L positions (prefix AND suffix -
hence "dual"), and per-block steps forward ONLY the block (C rows):
- replace_position boolean mask marks [s..e); the model REPLACES the cache rows of the
  block with each step's fresh block K/V (their modeling code does this internally).
- Block queries attend the FULL L-length cache (bidirectional, no mask restriction).
- Suffix K/V stay FROZEN within the block (the extra approximation vs prefix cache).
Per-step rows = C (e.g. 32): both attention AND FFN/lm_head shrink ~L/C-fold.

### 2d. Dream-specific quirk: shift_logits - dream/model/generation_utils_block.py:494
Dream predicts token at position p from the logits row of position p-1 (our engine
handles this via shift_logits/get_row_for_pos, examples/diffusion/diffusion.cpp:444:
row = shift ? max(pos-1, 0) : pos). Under BLOCK decoding the first block position s
needs the logits row of position s-1, which is NOT in a [s..e) batch. The reference
"solves" it by torch.cat([logits[:,:1], logits[:,:-1]]) - i.e. REUSES row s for
position s (an approximation!). WE can be exact: include one extra leading token in
the block batch (rows s-1..e). For the first canvas block, s-1 = last prompt token,
which always exists. LLaDA has shift_logits=false - no extra token needed.

### 2e. Where thresholds plug in
get_transfer_index(logits, ..., threshold) is exactly our conf_threshold commit branch;
both reference variants support it (quota=None when threshold set). Our threshold
machinery (de-tempering, degeneracy guard) carries over inside blocks unchanged.

## 3. Verified engine map (this repo, exact interfaces)

### 3a. Dream graph today - src/models/dream.cpp (read in full; 138 lines)
- hparams.causal_attn = false (:15); graph: build_inp_embd -> build_inp_pos ->
  build_attn_inp_no_cache() (:68) -> per layer: build_qkv + ggml_rope_ext(Q,K with
  inp_pos) -> build_attn(inp_attn, wo, ..., Q, K, V, ..., 1/sqrt(head_dim), il) (:94)
  -> FFN (SILU, LLM_FFN_PAR). inp_out_ids row-selection at last layer (:98).
  NOTE: rope positions come from batch.pos via build_inp_pos - absolute positions in
  block batches just work (the example already sets batch.pos explicitly).
- llada.cpp identical structure (:87 build_attn_inp_no_cache).
- Struct (src/models/models.h:453): llama_model_dream has NO state fields yet - the pkv
  fields go here (copy the DG block, models.h:853-860).

### 3b. Routing - why diffusion models hit encode()
src/llama-model.cpp:2013: create_memory returns nullptr for LLM_ARCH_DREAM/LLADA/
LLADA_MOE/RND1/DIFFUSION_GEMMA -> llama-context.cpp:1722 decode() routes to encode() ->
encode() (:1515) runs the full batch, non-causal, outputs ALL rows. Block batches are
just smaller batches through the same path - NO core routing change needed.

### 3c. THE IN-HOUSE PRECEDENT - DiffusionGemma pkv machinery (all verbatim-verified)
Everything needed already exists for DG; this project = generalize it to Dream/LLaDA
plus block semantics.
- Model state fields - src/models/models.h:853-860:
    enum pkv_phase_t { PKV_UNIFIED = 0, PKV_PREFILL = 1, PKV_DECODE = 2 };
    mutable pkv_phase_t pkv_phase = PKV_UNIFIED;
    mutable int64_t     pkv_P     = 0;   // prompt length of the current block
    mutable int64_t     pkv_cap   = 0;   // allocated capacity (grow-only)
    mutable std::vector<ggml_tensor *> pkv_k;  // per layer [head_dim, n_head_kv, cap]
    mutable std::vector<ggml_tensor *> pkv_v;
    mutable ggml_context * pkv_ctx; mutable ggml_backend_buffer_t pkv_buf;
- Store allocator - diffusion-gemma.cpp dg_ensure_pkv_store (:631): lazy, grow-only,
  F32, per-layer 3D tensors [n_embd_head_k(il), n_head_kv(il), cap], allocated from
  dev_layer(0)'s buffer type (SINGLE-GPU assumption - fine: our multi-replica server
  pins one model per GPU; do NOT use with row/layer-split models).
- Public API - include/llama.h:574:
    LLAMA_API void llama_diffusion_set_phase(struct llama_model*, int phase, int32_t P);
  Implementation (:671 in diffusion-gemma.cpp) dynamic_casts to DG only - the
  generalization point. State is MODEL-global and mutable: one in-flight request per
  model instance (our server serializes per replica - OK; never share one model across
  concurrent generations).
- Graph phase branches - diffusion-gemma.cpp:439-465:
    PREFILL: after rope, ggml_cpy(Kcur -> ggml_view_3d(pkv_k[il], hd, nkv, n_tokens))
             (+ same for V), wrapped in ggml_build_forward_expand(gf, ...); attention
             proceeds normally on the fresh tensors.
    DECODE:  pk/pv = ggml_view_3d(pkv_k[il], hd, nkv, P); Kfull = ggml_concat(ctx0, pk,
             Kcur, 2); attention on (Q_canvas, Kfull, Vfull).
  For Dream insert the same branches between rope and build_attn (dream.cpp:88-94).
- Mask construction - diffusion-gemma.cpp:378-409 (creation) + :112-164 (input class
  llm_graph_input_attn_diffusion_decode):
    is_decode: self_kq_mask = ggml_new_tensor_4d(ctx0, type_mask, n_kv=P+C, C, 1, 1)
    type_mask = cparams.flash_attn ? F16 : F32; ggml_set_input(); set _cnv alias.
    set_input() fills: canvas query q allows ALL keys (prompt + canvas) = 0.0f, else
    -INF. SWA branch exists for DG; Dream has swa_type NONE so self_kq_mask_swa is
    never allocated and the class works AS-IS. can_reuse() returns false (mask refills
    every graph build - llama.cpp rebuilds graphs per decode anyway).
    ACTION: move this class to a shared header (e.g. src/models/diffusion-common.h) so
    dream.cpp/llada.cpp/diffusion-gemma.cpp share one copy.
    For Dream DUAL-cache block mode the mask is all-zeros [L, C_batch] - even simpler
    (subclass or a flag: allow everything).
- Example-side usage precedent - examples/diffusion/diffusion.cpp:849-858 (EB path):
    llama_diffusion_set_phase(model, /*PKV_DECODE=*/2, n_input);
    batch.n_tokens = C; batch.pos[i] = n_input + i;  // ABSOLUTE positions
  and the restore after the run: set_phase(model, 0, 0) - always restore UNIFIED in
  every exit path or later requests run with a stale phase (bug class to watch).

### 3d. Sampler / scheduler interplay (verified behaviors)
- Multi-row backend sampling already handles per-step row-count changes (the EOT-shrink
  cur_length machinery exercises this today). Block batches just have fewer rows.
- Row mapping: get_row_for_pos (diffusion.cpp:444) becomes
    row = (shift ? max(pos-1, 0) : pos) - batch_first_pos
  where batch_first_pos = s-1 for Dream block batches (s = block start), s for LLaDA,
  n_input-1 for Dream suffix batches in prefix-cache mode. Confidence/conf arrays stay
  indexed by ABSOLUTE pos - only the row lookup changes.
- The attach-time warmup probe (LLAMA_TOKEN_NULL at step 0 -> CPU fallback) is
  per-request and phase-independent - unchanged.
- sched_reserve sizes buffers worst-case (full ubatch); smaller per-step graphs are
  fine. The graph TOPOLOGY changes with phase (concat nodes, mask shapes): llama.cpp
  builds a fresh graph every decode, so correctness is free; CUDA graphs will
  re-capture on shape change (Ampere+) - expect some replay-rate loss; measure with
  GGML_CUDA_FORCE_GRAPHS on Pascal later.
- Backend sampler eligibility (cfg_scale<=0 etc., diffusion.cpp:~250) is orthogonal.

### 3e. Model geometry (read from the GGUF header, gguf-py)
Dream-7B: 28 layers, 28 Q heads, 4 KV heads, n_embd 3584 -> head_dim 128,
n_embd_k_gqa = 512. rope_freq_base 1e6. DiffuCoder = same arch (Dream).
Store cost: 2 x 512 floats x 4 B x 28 layers = 112 KB/token F32.
  L=242 (kintsugi draft): 27 MB. L=512: 57 MB. F16 halves it. Irrelevant on 6 GB,
  acceptable even on 3 GB P106-090s.

## 4. Cost model (measured base numbers from this machine, RTX 5070)
Full-canvas step at L~242: ~110 ms; at L~114: ~51 ms; at L~512(+prompt): ~230 ms
(FA auto=on). Fixed overhead per decode ~10-20 ms (launch, sampling ~2 ms, readback).
Dual-cache per-step compute at C=32, L=242 ~ C/L of FFN+head + C*L/L^2 of attention
~ 13-15% of a full step + fixed overhead -> expect ~30-40 ms/step on 5070 shapes,
i.e. ~3x wall on drafts after amortizing one warm per block (~8-16 steps/block).
On compute-bound P106s the FLOPs reduction translates ~1:1 (better than 5070).
Prefix-cache alone saves only the prefix fraction: at block 1 of a 192-canvas with
50-token prompt, rows drop 242->192 (~20%); by the last block rows ~ 82 (~66% saved).
Verdict: prefix cache is the LEARNING phase, dual cache is the PAYOFF phase.

## 5. Implementation plan

### Phase 0 - scaffolding (no behavior change)
1. Move llm_graph_input_attn_diffusion_decode + a tiny pkv-state struct into
   src/models/diffusion-common.h; make DG use the shared copy (pure refactor, run DG
   smoke test).
2. Add the pkv field block (3c above) to llama_model_dream and llama_model_llada in
   models.h. Extract dg_ensure_pkv_store into a shared helper parameterized by
   (hparams, dev_layer(0), fields&) - it only touches those.
3. Generalize llama_diffusion_set_phase: try dynamic_cast to DG, then dream, then
   llada (3 casts; or give llama_model_base a virtual pkv_state() returning nullptr).
   Keep the existing signature; semantics per-arch documented in llama.h comment.
4. Add llama_diffusion_set_block(model, int32_t s, int32_t e) (new llama.h API) for
   Phase 2; no-op for models without pkv. Run: 14 sampler tests + DG + Dream smokes.

### Phase 1 - PREFIX cache for Dream/DiffuCoder (then LLaDA)
Engine (src/models/dream.cpp):
5. Graph: read phase/P off the model (like DG :360-376). PKV_WARM (new phase, P=s):
   square no-cache attention over ALL rows + ggml_cpy of the FIRST P rows of K/V into
   the store (DG's PREFILL is the P==n_tokens special case of this). PKV_DECODE:
   Kfull = concat(store[0..P), Kcur) + the shared rectangular mask ([P+C, C],
   all-allow). UNIFIED: current code path untouched.
   IMPORTANT: in DECODE the store holds prompt + already-committed blocks (P grows per
   block) - same mechanism, P is "cached prefix length", NOT "prompt length".
Example (examples/diffusion/diffusion.cpp, threshold path only at first):
6. params.kv_prefix (int block size, 0=off). Loop structure mirroring 2b per block
   [s, e): (a) warm: set_phase(PKV_WARM, s), full-canvas batch, commit at pos < e;
   (b) steps: set_phase(PKV_DECODE, s), batch rows [s-1..cur_length) (Dream shift;
   [s..cur_length) for LLaDA), commits restricted to pos < e; loop until the block has
   no masks; advance to next block. Tail shrink composes (suffix end = cur_length).
7. Flags: --diffusion-kv-prefix N (CLI arg.cpp, DIFFUSION example), server "kv_prefix".
8. Restore set_phase(UNIFIED) on EVERY exit path (success, degeneracy abort, error).
Validation gates (all must pass before Phase 2):
   - 14/14 sampler tests CPU+GPU.
   - Fixed-seed CLI A/B kv_prefix off/on: outputs may DIFFER (approximation!) - the
     gate is kintsugi bench pass-rate (Dream + DiffuCoder) within noise, plus manual
     review of 3 outputs.
   - DG regression (its kv_cache mode) untouched.
   - VRAM delta ~= store size (+27-57 MB).
   - Record per-step ms by block index (expect shrinking).

### Phase 2 - DUAL cache (block mode, the payoff)
9. Extend the store to cap=L and add PKV_BLOCK: batch rows [s-1..e); graph cpy's the
   batch K/V rows into store rows [s-1..e) (ggml_view_3d at byte offset - views take
   BYTE offsets: offset = (s-1) * pkv_k[il]->nb[2]); attention = Q_block x
   store[0..L) view; mask all-zeros [L, C_batch]. The example sets s,e via
   llama_diffusion_set_block.
10. Warm step = PKV_WARM(P=L) (full forward, store ALL rows, commit <= e).
11. Block loop in diffusion_generate mirrors 2c; reuse Phase-1 commit plumbing.
    EOT-tail shrink: shrink only at block boundaries (recommended start), or clamp
    shrink target >= e.
12. Flags: --diffusion-kv-block N (0=off, 32 default per reference), server "kv_block".
    kv_prefix and kv_block are mutually exclusive (kv_block subsumes).
Validation: same gates + block-size sweep {16, 32, 64} on the kintsugi bench; compare
   per-task pass-rate AND tok/s vs Phase 1 and baseline. If pass-rate drops > noise:
   ship Phase 1 only and investigate warm cadence (re-warm every k steps mid-block).

### Phase 3 - tuning / composition (each gets its own bench number)
13. F16 store (halve memory + bandwidth; quality re-bench).
14. Compose with suffix-window pruning (catalog C1): in PKV_BLOCK attend
    store[0 .. min(L, e+W)) instead of [0..L) - one view bound change.
15. Re-measure batched candidates (catalog D3): with dual cache, per-step batches are
    tiny - multi-canvas batching may now be ~free; if so, revisit SSD (catalog D1).
16. Infill mode: repair canvases are small (~60-150 tokens) - measure whether block
    mode helps at all there (fixed overhead likely dominates; kv off for infill is an
    acceptable outcome).

## 6. Pitfalls discovered in advance (so the implementer does not have to)
- ggml_view_3d offsets are BYTES (use ->nb[2] strides), not element counts.
- The DG mask class fills rows for CANVAS QUERIES ONLY (C rows) - the mask must be
  sized [n_kv, C_batch] from the BATCH row count (Dream's +1 shift row!), not from e-s.
- can_reuse=false on the mask inputs is required (mask content depends on P/C which
  change per step) - copying the DG class preserves this.
- Restore PKV_UNIFIED in every exit path; a stale phase makes the NEXT request decode
  garbage (see the restore in diffusion.cpp after the EB loop).
- Phase state is mutable on a shared model: one generation per model instance at a
  time (our server's per-replica mutex already enforces this).
- Store allocated on dev_layer(0): single-GPU per model only (true for our replicas);
  document/assert against -sm row/layer splits.
- shift_logits: block batches start at s-1; the row for pos p is (p-1)-(s-1) = p-s;
  for p=s that is row 0 = position s-1's logits - exactly correct. An off-by-one here
  produces subtly-shifted garbage that sometimes COMPILES - validate with a fixed-seed
  token-level diff in a deterministic mode first.
- Commit policy at warm steps: reference allows pos <= block_end; mirror it first.
- The degeneracy guard + confidence export index by ABSOLUTE pos - they keep working;
  only the get_row_for_pos lambda changes.
- num_transfer_tokens (non-threshold schedules) are per-block in the reference; our
  BLOCK_BASED schedule path already computes per-block counts (diffusion.cpp:322-330);
  threshold mode ignores them.
- CUDA-graph capture: per-phase shapes differ; steps after each phase switch re-capture
  (ms-level cost). On Pascal+GGML_CUDA_FORCE_GRAPHS the re-capture is relatively more
  expensive - measure before enabling both together.

## 7. Open questions - RESOLVED
1. Warm-step commit policy: mirror reference (commits at pos <= block_end). DECIDED.
2. Store dtype: F32 first (DG precedent, zero new risk), F16 in Phase 3. DECIDED.
3. Model order: Dream first (DiffuCoder free - same arch), LLaDA after Phase-1 gates
   pass (llada.cpp edits are mechanical, no shift row). DECIDED.
4. NEW - discovered during verification: the original "Phase 1 is exact, zero quality
   risk" claim was WRONG (sec 1: bidirectional mixing makes even prompt K/V dynamic
   from layer 2 on). Both phases are approximate; the quality gate is the bench, not
   output-identity. RESOLVED into the design.

## 8. Source-of-truth references
arXiv:2505.22618 (Fast-dLLM) + github.com/NVlabs/Fast-dLLM v1/llada/generate.py
(:86 vanilla, :132 prefix, :211 dual), v1/dream/model/generation_utils_block.py
(:454,:494 shift handling; :467-489 replace_position). In-repo:
src/models/diffusion-gemma.cpp (:112 mask class, :378 mask creation, :439 phase
branches, :631 store, :671 API), src/models/models.h:853 (state fields),
include/llama.h:574 (API), examples/diffusion/diffusion.cpp (:444 row mapping, :849 EB
phase usage), src/llama-model.cpp:2013 (memory routing), src/llama-context.cpp:1722 /
:1515 (encode). Catalog context: docs/dllm-throughput-catalog.md Layer A. Honest
expectations: 2-4x on drafts (not 27x - we already run threshold decoding); code
workloads cache worse than prose (dLLM-Cache: HumanEval 1.36x vs GSM8K 5.1x).
