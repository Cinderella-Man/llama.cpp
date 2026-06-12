# Layer A: step-to-step KV caching for masked dLLMs - IMPLEMENTATION GUIDE

Status: IMPLEMENTED (commit 454b98de4) - prefix mode shipped, dual mode experimental.
Section 13 is the implementation log: every discovery made while building, in detail. Everything below was checked first-hand:
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

### 3e-pre. FlashAttention vs rectangular masks - EMPIRICALLY VERIFIED
By construction: upstream's own no-cache path (src/llama-graph.cpp:2175-2185) creates
the mask at EXACT size [n_tokens, n_tokens] with no GGML_KQ_MASK_PAD padding, F16 when
flash_attn; rectangular [n_kv, n_q] masks are what the AR KV path feeds fattn daily.
By execution (this machine, 2026-06-12): DiffusionGemma's cached decode - the SAME
rectangular [P+C, C] mask class and concat pattern Dream will use - ran the full EB
loop with kv_cache=on under BOTH -fa on (954 ms/step, 14 steps) and -fa off
(885 ms/step, 15 steps), sane output both ways. No padding needed; FA optional.
(Perf note: FA bought nothing on DG - it is MoE-bound; on Dream FA is ~7% - keep auto.)

### 3e0. build_attn no-cache overload - how the mask is consumed (llama-graph.cpp:2200)
    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();
    cur = build_attn_mha(q, k_cur, v_cur, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
Three implementer-critical facts: (1) get_kq_mask() returns the *_cnv* alias - the
subclass MUST set self_kq_mask_cnv = self_kq_mask (DG does; forget it and you pass a
null mask). (2) selection is per-layer via hparams.is_swa(il) - Dream/LLaDA have
swa_type NONE so the non-SWA mask is always used; never allocate the swa variant.
(3) n_kv is implicitly K's row count (K passes straight to build_attn_mha) - INVARIANT:
mask->ne[0] == concat'd K ->ne[2], both derived from the same P/C/L values. A mismatch
is a ggml shape assert at graph build (loud, at least).
Sub-view copy precedent (for PKV_WARM's "store first P rows of Kcur"): DG's debug path
does exactly this - ggml_view_3d over Kcur's first P rows -> ggml_cpy into a store
tensor (diffusion-gemma.cpp:432-435). View-of-rope-output -> cpy is proven.

### 3e2. inp_out_ids under block batches - VERIFIED HARMLESS
build_inp_out_ids (src/llama-graph.cpp:1924) is kept even when ALL tokens are outputs,
deliberately, for constant graph topology (upstream comment cites pipeline parallelism).
encode() marks every row as output, so for block batches out_ids is an identity row
selection - no change needed in dream.cpp:98.

### 3e3. Reading model state inside the graph + phase restore precedents
DG graph ctor reads phase via a C-style cast (diffusion-gemma.cpp:296):
    const auto & dmodel = (const llama_model_diffusion_gemma &) model;
    const auto   phase  = dmodel.pkv_phase;
Dream's graph ctor gets the same `const llama_model & model` - copy the pattern.
Phase-restore precedents to imitate: examples/diffusion/diffusion.cpp:834 (restore
before fallback paths) and :966 (restore after the EB loop "for later turns").

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
:1515 (encode). Catalog context: docs/dllms/dllm-throughput-catalog.md Layer A. Honest
expectations: 2-4x on drafts (not 27x - we already run threshold decoding); code
workloads cache worse than prose (dLLM-Cache: HumanEval 1.36x vs GSM8K 5.1x).


## 9. Code skeletons (near-compilable; names final unless noted)

### 9a. models.h - state fields for llama_model_dream (copy for llada)
    // prompt/prefix KV store for cached diffusion decoding (see docs/dllms/
    // throughput-plans/01_layer_a.md). Mirrors llama_model_diffusion_gemma.
    enum pkv_phase_t { PKV_UNIFIED = 0, PKV_WARM = 1, PKV_DECODE = 2, PKV_BLOCK = 3 };
    mutable pkv_phase_t pkv_phase = PKV_UNIFIED;
    mutable int64_t     pkv_P     = 0;   // cached prefix length (WARM: rows to store)
    mutable int64_t     pkv_L     = 0;   // full canvas length (BLOCK mode)
    mutable int64_t     pkv_s     = 0;   // block start (BLOCK mode)
    mutable int64_t     pkv_cap   = 0;
    mutable std::vector<ggml_tensor *> pkv_k, pkv_v;
    mutable ggml_context * pkv_ctx = nullptr;
    mutable ggml_backend_buffer_t pkv_buf = nullptr;
(Numbering note: DG keeps PKV_PREFILL=1/PKV_DECODE=2; Dream's WARM generalizes PREFILL.
Keep the public ints stable: 0=unified, 1=warm/prefill, 2=decode, 3=block.)

### 9b. set_phase generalization (diffusion-gemma.cpp or a new diffusion-common.cpp)
    void llama_diffusion_set_phase(struct llama_model * model, int phase, int32_t P) {
        if (auto * dg = dynamic_cast<llama_model_diffusion_gemma *>(model)) {
            dg->pkv_phase = ...; dg->pkv_P = P;
            if (phase && P > 0) pkv_ensure_store(dg->fields..., P);
            return;
        }
        if (auto * dr = dynamic_cast<llama_model_dream *>(model)) { ...same...; return; }
        if (auto * ll = dynamic_cast<llama_model_llada *>(model)) { ...same...; return; }
    }
    // NEW API (llama.h, next to :574):
    LLAMA_API void llama_diffusion_set_block(struct llama_model*, int32_t s, int32_t L);
pkv_ensure_store = dg_ensure_pkv_store (:631) parameterized by (hparams, dev, fields) -
it already only uses n_layer, n_embd_head_k(il), n_head_kv(il), dev_layer(0).

### 9c. dream.cpp graph - phase branches (insert between rope :88 and build_attn :94)
    const auto & dmodel = (const llama_model_dream &) model;
    const auto   phase  = dmodel.pkv_phase;     // read ONCE before the layer loop
    const int64_t P     = dmodel.pkv_P;
    // mask setup before the layer loop (replaces unconditional :68):
    llm_graph_input_attn_no_cache * inp_attn = nullptr;
    if (phase == PKV_DECODE || phase == PKV_BLOCK) {
        const int64_t n_kv = (phase == PKV_BLOCK) ? dmodel.pkv_L : P + n_tokens;
        auto uptr = std::make_unique<llm_graph_input_attn_diffusion_decode>(
                        hparams, cparams, n_kv - n_tokens, n_tokens);  // all-allow fill
        const auto type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;
        uptr->self_kq_mask = ggml_new_tensor_4d(ctx0, type_mask, n_kv, n_tokens, 1, 1);
        ggml_set_input(uptr->self_kq_mask);
        uptr->self_kq_mask_cnv = uptr->self_kq_mask;
        inp_attn = (llm_graph_input_attn_no_cache *) res->add_input(std::move(uptr));
    } else {
        inp_attn = build_attn_inp_no_cache();
    }
    // per layer, after rope:
    if (phase == PKV_WARM) {                    // square attn + store first P rows
        ggml_tensor * sk = ggml_view_3d(ctx0, dmodel.pkv_k[il], n_embd_head, n_head_kv,
                                        P, dmodel.pkv_k[il]->nb[1], dmodel.pkv_k[il]->nb[2], 0);
        ggml_tensor * kP = ggml_view_3d(ctx0, Kcur, n_embd_head, n_head_kv, P,
                                        Kcur->nb[1], Kcur->nb[2], 0);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, kP, sk));   // same for V
        // attention: unchanged square path (Q,K,V as today)
    } else if (phase == PKV_DECODE) {           // prefix-cache step
        ggml_tensor * pk = ggml_view_3d(ctx0, dmodel.pkv_k[il], n_embd_head, n_head_kv,
                                        P, ..., 0);
        ggml_tensor * Kfull = ggml_concat(ctx0, pk, Kcur, 2);    // same for V
        cur = build_attn(inp_attn, wo, ..., Qcur, Kfull, Vfull, ..., il);
    } else if (phase == PKV_BLOCK) {            // dual-cache step (Phase 2)
        // write batch rows into store at block offset (BYTE offsets!):
        ggml_tensor * dst = ggml_view_3d(ctx0, dmodel.pkv_k[il], n_embd_head, n_head_kv,
                                        n_tokens, nb1, nb2, (size_t)(dmodel.pkv_s) * nb2);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, dst));  // same for V
        ggml_tensor * Kall = ggml_view_3d(ctx0, dmodel.pkv_k[il], n_embd_head, n_head_kv,
                                        dmodel.pkv_L, nb1, nb2, 0);
        cur = build_attn(inp_attn, wo, ..., Qcur, Kall, Vall, ..., il);
    } else { /* UNIFIED: existing code, untouched */ }
ORDERING SUBTLETY (PKV_BLOCK): the cpy into the store and the attention read of the
store are both in one graph - ggml executes in dependency order, and Kall is a VIEW of
the same buffer the cpy writes. ggml does not track view aliasing as a dependency!
Force ordering: build the cpy with ggml_build_forward_expand BEFORE creating Kall, and
make Kall depend on the cpy via ggml_view of the SAME tensor AFTER... safest pattern:
attend Kfull = ggml_concat(store_prefix_view, Kcur, store_suffix_view) instead of
writing-then-reading in one graph (3-way concat: [0..s) cached, [s..e) fresh from
Kcur, [e..L) cached) and do the store WRITE of block rows too (for the NEXT step).
The 3-way-concat read does not alias the written region - no ordering hazard. DG
avoids this by never reading rows it writes in the same graph; mirror that property.

### 9d. diffusion.cpp - prefix-cache loop (Phase 1, threshold path)
    const int32_t kvb = params.kv_prefix;                 // block size, 0 = off
    for (int32_t s = n_input; s < cur_length; s += kvb) {
        const int32_t e = std::min(s + kvb, cur_length);
        // (a) warm: full canvas, store rows [0..s)
        llama_diffusion_set_phase(model, /*WARM*/1, s);
        build_full_batch(0, cur_length);                  // pos 0..cur_length
        decode + sample + commit(mask: pos < e);          // existing threshold code
        // (b) cached steps until block clean
        llama_diffusion_set_phase(model, /*DECODE*/2, s);
        const int32_t b0 = params.shift_logits ? s - 1 : s;   // extra row for shift
        while (block_has_masks(s, e) && steps_left) {
            build_batch(b0, cur_length);                  // pos b0..cur_length
            decode + sample + commit(mask: pos < e);
            // row lookup: row = (shift ? pos-1 : pos) - b0
        }
    }
    llama_diffusion_set_phase(model, 0, 0);               // EVERY exit path
Integration notes: reuse the existing threshold-commit block verbatim - only the
batch-build and get_row_for_pos change; keep the degeneracy guard (absolute pos);
EOT-tail shrink runs at block boundaries only (re-derive cur_length after each block).
Step budget policy: params.steps stays the GLOBAL cap, counted across warms + block
steps together (one shared counter; abort when exhausted, like today's n_steps_done).
The reference instead asserts divisibility and quotas steps per block - we do NOT
(threshold mode self-terminates per block; the quota path only matters for the
non-threshold schedules, where diffusion.cpp:322-330 already computes per-block
transfer counts). Remainder blocks: e = min(s + kvb, cur_length) - no divisibility
requirement, unlike the reference's assert.

### 9e. Flags / server
common/arg.cpp (DIFFUSION example): --diffusion-kv-prefix N, --diffusion-kv-block N ->
common_params_diffusion { int32_t kv_prefix = 0; int32_t kv_block = 0; } (note: DG
already has eb_kv_cache; keep names distinct). diffusion-server.cpp make_masked_params:
dp.kv_prefix = req.value("kv_prefix", p.diffusion.kv_prefix); same for kv_block;
mutually exclusive - kv_block wins, log a warning if both set.

## 10. Interactions with the rest of the catalog (sequencing advice)
- B1 adaptive threshold (tiny, do FIRST): fewer steps per block raises the warm-to-step
  ratio - re-measure Layer A gains AFTER B1 lands so the bench reflects the real ratio.
- G9 skeleton-seeded drafts: pre-committed spans mean blocks may START fully committed -
  the block loop must skip clean blocks without spending a warm forward on each (check
  block_has_masks BEFORE warming; one shared warm covers consecutive clean blocks since
  nothing needs decoding there - just advance s).
- C1 suffix-window pruning: composes in PKV_BLOCK by capping the attended store view at
  e+W (sec 5 step 14); in PKV_DECODE by capping the suffix batch at min(cur_length, e+W).
- EOT-tail shrink: already integrated (block-boundary shrink, sec 5 step 11).
- D1 SSD / D3 batching: only meaningful AFTER dual cache (they exploit the tiny per-step
  batches); re-measure memory-boundness then.
- Infill (kintsugi repairs): leave kv off initially (sec 5 step 16) - repair canvases
  are small and fixed-overhead-dominated; measure before bothering.
- A2 dLLM-Cache (V-verify feature caching): a different, deeper mechanism (caches
  AttnOut/FFN too, partial recompute). Build A1 first; A2 only if the bench shows
  remaining headroom AND code-workload caveat (HumanEval 1.36x) is acceptable.

## 11. Validation commands (copy-paste)
    # build
    cmake --build build -j --target llama-diffusion-cli llama-diffusion-server \
        test-backend-sampler
    # sampler regression (must stay 14/14 on both)
    LLAMACPP_TEST_MODELFILE=~/models/qwen05b/qwen2.5-0.5b-instruct-q8_0.gguf \
        ./build/bin/test-backend-sampler --device gpu   # and --device cpu
    # deterministic-mode token diff (catches row-mapping off-by-ones; threshold OFF,
    # fixed seed, pure schedule):
    ./build/bin/llama-diffusion-cli -m ~/models/dream7b/Dream-*Q4_K_M.gguf -p "test" \
        -ub 256 --diffusion-eps 0.001 --diffusion-steps 32 --temp 0 --seed 7 -ngl 99 \
        [--diffusion-kv-prefix 32]   # run with and without; diff the outputs; EXPECT
                                     # small drift (approximation) - review, not assert
    # FA interaction check
    ... same command with -fa on / -fa off under kv-prefix: both must produce sane text
    # quality gate (the real one): kintsugi bench on both models
    ./build/bin/llama-diffusion-server -m <dream|diffucoder> -ub 512 -ngl 99 \
        --diffusion-eps 0.001 --diffusion-steps 128 --temp 0.2 --top-k 40 \
        --diffusion-kv-prefix 32 &
    cd kintsugi && mix run bench/bench.exs    # pass-rate within noise vs baseline table
    # DG regression (phase API shared): one canvas generate via server + eb kv_cache on
    # VRAM check
    nvidia-smi --query-compute-apps=used_memory --format=csv,noheader  # +27-57 MB only
    # per-step timing: server log "time per step" by block index (expect shrink)

## 12. Changelog of this guide
- r1: initial plan (block/dual cache design, DG precedent, phases).
- r2: corrected "Phase 1 exact" fallacy; added verified FA/mask, out_ids,
  cast/restore findings; code skeletons incl. the PKV_BLOCK view-aliasing hazard and
  its 3-way-concat resolution; catalog interaction map; literal validation commands.
- r4: IMPLEMENTED. Section 13 added: reserve-graph crash, guard false-positive,
  required EOS early-exit, the EOT-shrink benchmark trap, step-inflation vs
  whole-suffix commits, K/V drift -> EOG quarantine + pressure warms, the
  phase-marker reuse pattern, measured matrix, honest positioning.
- r3: FA + rectangular mask upgraded from by-construction to
  EMPIRICALLY VERIFIED (DG cached decode executed under -fa on AND off); build_attn
  mask-consumption contract documented (the _cnv alias trap, per-layer is_swa
  selection, the mask/K n_kv invariant); sub-view-cpy precedent cited
  (diffusion-gemma.cpp:432-435); explicit global step-budget + remainder-block
  policy; catalog moved to docs/dllms/.


## 13. IMPLEMENTATION LOG (r4, 2026-06-12) - what building it actually taught us

Everything below was discovered by implementing and measuring; each item names the
symptom, the root cause, and the fix as committed (454b98de4).

### 13.1 Scheduler reserve graphs build under YOUR phase (crash #1)
Symptom: GGML_ASSERT ggml.c:1751 (view bounds) at step 0.
Cause: sched_reserve builds dummy graphs with n_tokens=1 AND n_tokens=n_ubatch - and it
runs lazily at the FIRST decode (backend-sampler attach), i.e. AFTER set_phase(WARM).
The warm branch's "first P rows of Kcur" view then slices a 1-row tensor with P=19.
DG never hits this because its reserve happens at context init under UNIFIED.
Fix: every phase must produce a buildable graph for ANY n_tokens - warm clamps its
store write (Pw = min(P, n_tokens)); BLOCK falls back to a DECODE-shaped graph when
s + n_tokens > L. RULE: model-state-driven graphs must be total functions of n_tokens.

### 13.2 The degeneracy guard false-positives under block commits (silent abort)
Symptom: kv runs ended after 1 step, "end tokens 19/19 committed, 275/294 still masked".
Cause: with commits restricted near the current block, the model legitimately commits
the EOT region inside block 1 (it knows the answer is short) - those EOTs are NOT a
canvas-suffix, so the guard counted them as a scattered flood while counting the
undecoded-by-design region beyond kv_e as "still masked".
Fix: guard bounds become [n_input, kv_e) with the tail scanned from kv_e (the world
ends at kv_e in block mode).

### 13.3 EOS early-exit is REQUIRED, not optional (catalog C2 graduated)
A short answer on a long canvas EOT-fills block after block, paying one warm forward
per remaining block. Fix: when the current block is fully committed AND ends in a
committed EOG run (>= 2), fill [kv_e, max_length) with that EOG deterministically and
stop. Zero forwards for the entire tail.

### 13.4 THE BENCHMARK TRAP: our own EOT-tail shrink is a competing canvas reducer
Symptom: kv-prefix 167 ms/step vs baseline 42 ms/step at "the same" canvas.
Investigation: nsys showed GPU busy only ~45 ms of the 167 (host-idle mystery), then
per-step row logging (KVTIME) revealed the truth: the BASELINE was not running
320-row steps at all - the EOT-tail shrink collapses it to ~40 rows after step 0 on
short-answer prompts (320 -> 43 -> 41 -> ...). We had disabled shrink under kv mode,
so the cache fought a strawman of itself. The cached steps were actually ~2x faster
than EQUAL-SIZE uncached steps all along.
Fix (composition, not competition): prefix mode shrinks freely (a suffix-end change
does not touch the cached prefix [0..P); the graph rebuilds once per shape change);
dual mode shrinks only when the block is clean (next step re-warms anyway).
LESSON: before benchmarking a new optimization, list every EXISTING optimization that
shapes the baseline's actual work - ours had quietly changed the game.

### 13.5 Block-restricted commits fight whole-canvas threshold decoding (step inflation)
Symptom: story generation 15 steps (baseline) -> 128-step cap (kv), 4x slower.
Cause: our threshold decoder commits confident tokens ANYWHERE; restricting commits to
the current block (the reference's policy) gates progress on the block's slowest token.
The reference papers never see this because their baselines are 128-1024-step quota
schedules - the 27x headlines are vs THAT, not vs whole-canvas threshold decoding.
Fix for prefix mode: commit over the WHOLE decoded suffix (all suffix rows have logits)
- block structure then only determines what gets cached, not what may commit. Surprise
bonus: left-to-right prefix growth + whole-suffix commits = FEWER steps than baseline
(9-10 vs 15 on the story; structured early commits seed later confidence).
Dual mode cannot do this (only block rows have logits) - hence its experimental status:
fastest on short answers (5 steps on the haiku), step-inflated on long free-form text.

### 13.6 K/V drift is fast and shows up as EOT-flooding (quality regression + fix chain)
Symptom: kv-prefix story truncated to "Once upon a time," (baseline: full sentence).
Cause: the >0.94 cosine similarity is between ADJACENT steps; reusing the step-0
snapshot (warmed on an ALL-MASK canvas) across 9 steps compounds drift - and the
degradation mode is specifically INFLATED EOG CONFIDENCE (low-information context ->
EOT pathology). The reference dodges it by completing blocks (and re-warming) quickly.
Fix chain (all three needed, in order of discovery):
 (a) scheduled re-warm every kv_rewarm steps - helped, insufficient alone;
 (b) EOG QUARANTINE: cached steps may commit any TEXT token but never EOG/pad; only
     warm steps (exact forwards) may end the answer. Output became byte-identical to
     baseline on the test prompts;
 (c) EOG PRESSURE: when a cached step would commit ONLY end tokens (quarantine blocked
     everything incl. the best-fallback), the answer is probably done - force the next
     step to warm NOW instead of idling to the next scheduled re-warm.
With (b)+(c), kv_rewarm relaxed to 12 (warms are mostly pressure-driven).

### 13.7 Graph-reuse correctness landmine + the phase-marker input pattern
llama.cpp reuses the previous graph when shapes match and every input's can_reuse()
agrees. Two hazards: (1) blanket can_reuse=false (the DG mask class default) forfeits
reuse for every cached step - pure overhead; (2) worse, with no custom input at all,
two DIFFERENT phases with the same batch shape would silently reuse each other's
graphs - whose store-write views bake in the OLD P/s (consecutive warms of different
blocks are shape-identical!). Fix: llm_graph_input_diffusion_phase, a zero-cost input
added in EVERY phase whose can_reuse compares the model's live pkv state against the
state the graph was built for. The mask class then allows reuse on matching width.

### 13.8 Measured results (RTX 5070, Dream-7B Q4_K_M, threshold 0.6, seed-pinned)
| case                                   | baseline      | kv-prefix 32     | kv-block 32 |
| haiku (short, ub 320)                  | 335 ms / 8 st | 479 ms / 13 st   | 277 ms / 5 st |
| story (medium, ub 512)                 | 1217 / 15     | 2492 / 28        | 128-step cap |
| Elixir GenServer module (code, ub 512) | 22614 / 96    | **11517 / 58 (1.96x)** | n/a |
| kintsugi bench aggregate               | 16.8 tok/s    | 9.1 (redraft-luck dominated; heal 148 tok/s, forge_short -32% wall) |
Honest positioning: the cache wins where content is LONG and real (code - the mission
workload); short chat answers are owned by EOT-shrink + early-exit; kv-block is
fastest on short outputs but step-inflates long ones. Default recommendation:
--diffusion-kv-prefix 32 for code generation; off for short-form; kv-block
experimental pending a commit-policy redesign (lookahead window).
Note the story/haiku regressions are STEP-COUNT effects (warm cadence + quarantine
trading steps for exactness), not per-step cost - cached steps are ~2x cheaper than
equal-size uncached ones throughout.

### 13.9 Odds and ends
- LLaDA: ported mechanically (same graph edits, no shift row); compiles; UNTESTED (no
  LLaDA GGUF on disk) - run the gates before trusting.
- DG regression: clean (858 ms/step canvas run, kv_cache mode intact).
- 14/14 sampler tests CPU+GPU throughout; baseline path byte-identical (UNIFIED).
- Methodology that found the bugs, in order of usefulness: per-step row+ms logging
  (KVTIME pattern) > nsys kernel sums > GGML_SCHED_DEBUG (no output in release builds)
  > theorizing. Instrument FIRST next time.


## 14. EXHAUSTION LOG (r5, 2026-06-12): parameter sweeps + wild ideas, all measured

Method: fixed 3-prompt matrix (haiku ub320 / story ub512 / Elixir GenServer CODE ub512 -
the mission KPI), single-seed for exploration, 3 seeds {3,103,203} for finals (the bench
taught us single-seed deltas are path-luck: rewarm 4/6/8 single-seed gave 11.6/9.4/15.2 s
- a jagged landscape that only seed-sums resolve). Completeness verified by zero mask
pieces in output (all reported configs produced complete, fence-terminated code).

### 14.1 FINAL multi-seed results, CODE KPI (sum of 3 seeds)
| config                  | total (3 seeds) | speedup | notes |
| baseline (kv off)       | 76.7 s          | 1.00x   | 23.5/28.4/24.7 s |
| kv_prefix 32, rewarm 12 | 37.5 s          | 2.04x   | tight: 12.4-12.6 s |
| kv_prefix 32, rewarm 6  | 29.1 s          | 2.64x   | tight: 9.5-10.0 s |
| kv_block 32             | 24.9 s          | 3.08x   | tight: 8.0-8.7 s - the code champion |
Short-form check (single seed): haiku off 0.39 s, prefix+window 0.55 s, kv_block 2.0 s;
story off 1.28 s, kv_block 2.9 s. NOTHING earns default-on; the recommendation is
content-aware (sec 14.4).

### 14.2 kv-block REDEEMED for code (and why the earlier verdict was incomplete)
Sec 13.5 called dual mode "step-inflated on long free-form text" - true for the STORY
(prose: model wants to commit globally; story off=15 steps, blk32=58). But long CODE is
locally-structured: block-ordered commits cost steps (128 vs 96) yet each block step is
~8x cheaper, netting 3.08x. The step count is not the cost model; rows x steps is.
kv_block 64 and kv_block+rewarm tweaks measured slightly worse than plain kv_block 32.

### 14.3 Wild ideas: tried, measured, verdicts
- COMMIT-MASS re-warm trigger (drift tracks canvas changes, so re-warm after N commits
  instead of N steps - principled!): REJECTED. cm24 = 18.4 s vs cadence-12's 11.7 s on
  code. Mechanism: heavy-commit phases are exactly when generation is going WELL; the
  trigger fires warms at the worst moments. Principled != better.
- SUFFIX LOOKAHEAD WINDOW (--diffusion-kv-window W; DPad-style: drop distant masks from
  the batch entirely; in dual mode = sliding-window lookahead commits, the sec 13.5
  "commit-policy redesign"): MIXED, net REJECTED as default. prefix+w64 11.3 s (vs 11.7)
  with 111 cheap steps - marginal; dual+w64 10.0 s vs plain dual 8.0 s - worse (lookahead
  rows cost more than the steps they save); +commit-mass combos hit the step cap. The
  flag stays (harmless, possibly useful on other shapes) but earns no default.
- F16 STORE: BLOCKED by CUDA - ggml_concat asserts src0 F32 (ggml/src/ggml-cuda/
  concat.cu:149) and decode concatenates the store every step. Crashed at step 1 on
  512-canvas (small canvases sneak through via different graph paths - a trap). The env
  toggle was REMOVED (a crashing knob is worse than no knob); revisit via an F16 concat
  kernel or a copy-into-preallocated-buffer layout (no concat at all - which would also
  kill the 3-way-concat aliasing dance of sec 9c). Cast plumbing left in the graphs.
- REWARM CADENCE: swept {4,6,8,12,24}; 6 wins on multi-seed (2.64x vs 2.04x at 12) and
  is now the DEFAULT (kv_rewarm=6). Fresher caches commit better, fewer steps - down to
  the point where warm cost dominates (4 is worse).

### 14.4 Recommendation table (until kintsugi grows content-length routing)
| workload                          | config |
| short answers / chat (<~150 tok)  | kv OFF (EOT-shrink owns this regime) |
| kintsugi drafts at n_gen <= 192   | kv OFF (bench-verified: prefix costs +38% wall) |
| long code generation (>= ~300 tok)| kv_block 32 (3.1x) or kv_prefix 32 rewarm 6 (2.6x, gentler worst-case) |
| prose/story                       | kv OFF |
All defaults remain OFF; flags: --diffusion-kv-prefix/-kv-block/-kv-rewarm/-kv-window
(+ server request params kv_prefix/kv_block/kv_rewarm/kv_rewarm_commits/kv_window).

### 14.5 Exhaustion status
Layer A core: DONE (prefix + dual shipped, tuned, content-routed recommendations).
Tried-and-rejected: commit-mass warms, sliding-window lookahead, F16-store-via-concat.
Remaining in the wider Layer A family (deliberately deferred, see catalog): A2 V-verify
feature caching (code workloads cache worst per its own paper), A3 design bake-off,
F10 cross-request canvas cache (needs a design doc; biggest open idea - repair canvases
are 95% identical across requests), F16 store via layout change, LLaDA execution test
(no model on disk). Regressions green at exit: 14/14 sampler tests, baseline path
byte-identical (363 ms haiku), DG untouched.
