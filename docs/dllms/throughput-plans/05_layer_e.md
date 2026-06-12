# Layer E: model-side - PLAN / IMPLEMENTATION GUIDE (E2 first)

## Context

Layer E of docs/dllms/dllm-throughput-catalog.md: the path to TINY models. Catalog
investigation order item 2, and the named RE-ENTRY TRIGGER for parked Layer D
(04_layer_d.md question 7: at 1.5B the per-row cost shrinks ~5x while the ~18 ms
fixed floor does not - batching/speculation may revive). E1 (SDTT distillation, a
training run) is scoped only after E2 establishes what a tiny dLLM is worth here.

E2 mission: pull Efficient-Large-Model/Fast_dLLM_v2_1.5B, convert to GGUF, run it
on OUR engine, measure against Dream-7B on the bench. "Cheapest path to a tiny
model TODAY" (catalog). A 1.5B block-dLLM fits a 3 GB P106-090 - the rig story.

## E2 source facts (verified 2026-06-12)

- HF: Efficient-Large-Model/Fast_dLLM_v2_1.5B, Apache 2.0. arXiv:2509.26328
  (Fast-dLLM v2): AR-to-block-diffusion conversion of Qwen2.5-1.5B-Instruct with
  ~1B fine-tuning tokens (500x less than Dream's 580B); paper claims 2.5x faster
  than the AR original at batch 1, quality parity on standard benchmarks.
- 1.54B params (1.31B non-embedding), 28 layers, BF16.
- Inference contract (model card quickstart): custom generate(input_ids,
  max_new_tokens, small_block_size=8, threshold=0.9) - blockwise bidirectional
  attention via complementary masks, TOKEN SHIFT within blocks (AR-style
  next-token parameterization, like Dream's shift_logits), hierarchical
  block/sub-block caching.
- Decode semantics vs our engine: block-CAUSAL across blocks + bidirectional
  within an 8-token block + confidence-threshold parallel commits. Our nearest
  machinery: Dream-style masked decode (threshold path) for within-block,
  DiffusionGemma block-AR precedent for blockwise progression, kv_block dual
  cache for the cached path. Expect engine-gap work; v1 target is CORRECT, not
  fast: the square path doing one 8-block at a time with bidirectional masks
  inside the block and committed context behind it.

## E2 work log (everything, including dead ends)

### 2026-06-12: checkpoint pulled and dissected
- Downloaded to ~/models/fast-dllm-v2-1.5b (3.1 GB safetensors, BF16).
- config.json: model_type Fast_dLLM_Qwen = EXACTLY Qwen2.5-1.5B-Instruct hparams
  (1536 hidden, 28 layers, 12 heads / 2 KV heads GQA, qkv-bias, tied embeddings,
  rope_theta 1e6, vocab 151936) + ONE extra field: bd_size 32. Tensor names are
  byte-for-byte Qwen2 (verified via safetensors header; 338 tensors, no lm_head -
  tied). Chat template = standard Qwen2.5 im_start/im_end.
- Mask token: "|<MASK>|" id 151665 (added_tokens.json; NOT in config - the
  converter must set tokenizer.ggml.mask_token_id itself). EOS 151645 (im_end);
  generation_config also lists 151643 (endoftext).
- modeling.py = the inference ground truth, extracted:
  * Attention at inference: eval_block_diff_mask = block_q >= block_kv at
    block_size 32 over the WHOLE sequence (prompt included) - pure BLOCK-CAUSAL:
    bidirectional within each 32-block, causal across blocks. (The fancy 3-part
    complementary mask is TRAINING-only.)
  * TOKEN SHIFT identical to Dream: logits = cat([logits[:,:1], logits[:,:-1]]) -
    row i-1 predicts position i, first row duplicated; our shift_logits machinery
    (get_row_for_pos = max(pos-1,0)) is exactly this.
  * Decode loop: per 32-block: canvas += 32 masks; 4 SUB-BLOCKS of 8, strictly
    left to right; per step forward the last 32 rows (vs cached committed blocks),
    commit within the CURRENT sub-block only: all probs > threshold (card: 0.9)
    PLUS always the argmax-prob one; when the block is mask-free, one more
    forward writes it to the cache AND the last row's argmax becomes the first
    token of the NEXT block (AR step across blocks). EOS stop once no masks
    remain before it; trailing tokens truncated.
  * Committed blocks never change -> a STANDARD causal KV cache works for them
    at block granularity (the model card's "hierarchical cache"; their
    use_block_cache sub-block variant is an optimization, not v1).
  * Defaults: top_p 0.95, temperature 0 (argmax), threshold from caller.
- Semi-AR consequence: canvas grows 32/block - the model NEVER sees masks beyond
  the current block, so Layer C4's stub-adaptation failure mode does not apply.
  No degeneracy-guard analog needed for v1 (EOS handling is explicit).

### Engine integration plan (v1 = correct, uncached square)
1. Converter (conversion/qwen.py): register Fast_dLLM_QwenForCausalLM, subclass
   Qwen2Model, model_arch = new MODEL_ARCH.FAST_DLLM; add_causal_attention(false)
   (RND1 precedent), add_mask_token_id(151665) read from tokenizer added vocab,
   write bd_size as fast-dllm.block_size kv.
2. Fork arch: LLM_ARCH_FAST_DLLM "fast-dllm": qwen2 tensor table + hparams load;
   create_memory -> nullptr (memoryless, Dream/RND1 precedent);
   llm_arch_is_diffusion includes it (server gate).
3. Graph src/models/fast-dllm.cpp: qwen2 forward with a custom no-cache attn
   input whose set_input fills BLOCK-CAUSAL (same-seq && qblock >= kvblock,
   block 32 from hparams) - subclass pattern from dream.cpp's custom mask input.
   v1 has NO pkv phases (square only; kv via the standard committed-block
   property comes later).
4. Example: new diffusion_generate_block_ar() in diffusion.cpp implementing the
   reference loop verbatim (uncached: re-forward [committed | block] each step);
   routed when the model arch is fast-dllm; CLI/server pass block/sub-block/
   threshold via existing flags (--diffusion-block-length, new sub-block flag).
5. Measure: tok/s + quality sanity vs Dream-7B; bench v2; THEN the Layer D
   re-entry checkpoint (re-run llama-diffusion-batch-probe on the 1.5B).

### 2026-06-12 (same session): IMPLEMENTED END TO END - converted, running, measured

Conversion (first try, no dead ends):
- conversion/qwen.py: FastDLLMQwenModel registered (Qwen2Model subclass,
  MODEL_ARCH.FAST_DLLM, add_causal_attention(false), mask id from added vocab,
  bd_size -> "fast-dllm.block_size" kv) + registry entry in conversion/__init__.py
  + gguf-py constants (enum, name, Dream-cloned tensor table).
- F16 GGUF 3.1 GB; Q4_K_M = 986 MB - FITS A 3 GB P106-090. (Vocab quirk logged:
  token 128247 '</s>' flagged not-control by loader; harmless.)

Fork arch (one pass, compiles clean):
- LLM_ARCH_FAST_DLLM in llama-arch.{h,cpp} (name, is_diffusion), llama-model.cpp
  (factory, create_memory nullptr, NEOX rope), models.h class with bd_size member
  (read via ml.get_key string overload, default 32).
- src/models/fast-dllm.cpp: qwen2 graph + llm_graph_input_attn_block_causal -
  square no-cache mask filled BLOCK-CAUSAL BY POSITION (same-seq && p_q/bd >=
  p_kv/bd). ~190 lines, all from dream.cpp/qwen2 precedent.

Decode loop (examples/diffusion/diffusion.cpp diffusion_generate_block_ar):
reference-faithful, uncached v1; routed by arch in CLI + server ("block_ar" in
/health; infill rejected - see LIMITATION below). Three bugs found and fixed:
1. CLI legacy assert needs --diffusion-block-length 32 alongside (eps^block xor).
2. ARGMAX-ONLY v1 was seed-deaf: all bench redrafts byte-identical AND repairs
   impossible -> p-tier 0/18. Fixed with reference sample_with_top_p (seeded
   nucleus sampling at temperature).
3. THE DE-TEMPER LESSON STRIKES AGAIN (Layer B): gating commits on the
   temp-sharpened (or nucleus-renormalized) prob floods every sub-block position
   past any threshold -> simultaneous shifted commits emit adjacent-duplicate
   garbage ("moduleirmodulemodule"). Commit confidence MUST be the plain softmax
   prob of the sampled token. Same scale law as our Dream threshold machinery.
- Threshold is the MODEL'S scale: 0.6 (Dream-tuned) corrupts ("```moduleir");
  0.9 (model card) is correct. Bench profile "fastdllm" carries it.

LIMITATION (architectural, important): block-causal attention means a masked
span mid-sequence cannot see the code AFTER it - INFILL IS STRUCTURALLY
UNSUPPORTED. kintsugi's repair ladder (the project's core mechanism) does not
apply to block-AR models; recovery is whole-draft retry only. Server rejects
infill requests for block_ar models explicitly.

MEASURED (5070 laptop, Q4_K_M, uncached square v1):
- Smoke: coherent 171-token poem, 23.2 ms/step; code drafts ~90 tok/s engine-side
  (61 tok in 0.68 s); templated Doubler draft: clean and correct in 315-500 ms vs
  Dream-7B ~1100 ms = ~3x faster per draft, on an UNCACHED path.
- Commits/step ~1.85 on code at thr 0.9 (parallelism engages on structured text;
  ~1.0 on prose - the content split again).
- BENCH (profile fastdllm, thr 0.9): 21/48 - p 12/18, h 9/9 (Credence-only),
  m/c/a 0, i 0/6 (architectural). Dream-7B full system: 35/48.
- THE FAIR COMPARISON (draft + Credence only, i-tier excluded): FastDLLM-1.5B
  21/42 BEATS Dream-7B 15/42. The 1.5B OUT-DRAFTS the 7B (Qwen2.5-Instruct
  lineage + 2026 conversion recipe vs 2025 Dream); Dream's system lead is
  entirely its repair ladder. p-tier wall: FastDLLM 87.7 s vs Dream 45.9 s -
  failed cases burn 3 full redrafts (no cheap repairs to lean on).

LAYER D RE-ENTRY CHECKPOINT (probe re-run on the 1.5B): still PARKED. Per-row
cost shrank ~3x but the fixed floor shrank WITH it (tiny forward 7.2 ms vs 27 ms
on the 7B): 103-tok canvas K=2-4 batching = 1.20-1.23x (vs 1.00x), tiny-canvas
K=8 = 2.35x, W-scaling 128 rows = 2.78x of 32. The "floor share grows at 1.5B"
hypothesis is REFUTED. (Caveat: PROBE5/6 drive Dream-style machinery this arch
does not use - treat as row-scaling proxies, not decode-path numbers.)

### Verdict and next moves
- E2 EXISTENCE PROOF: DELIVERED. A 986 MB block-dLLM runs on our engine TODAY,
  out-drafts Dream-7B head-to-head, ~3x faster per draft uncached, and fits the
  P106 fleet. The catalog's "cheapest path to a tiny model" claim is CONFIRMED.
- The strategic shape it exposes: FastDLLM-1.5B DRAFTS + Dream-7B REPAIRS is
  exactly catalog D4 (two-model draft/verify) - arrived at from measurements,
  not theory. kintsugi already serializes engines; a two-server split is
  harness-only work. THE natural next experiment.
- v1 leaves on the table: KV caching for committed blocks (standard causal cache
  fits the semantics - big per-step win at longer contexts), sub-block flag
  plumbed but untuned, P106 validation run, temperature/top_p semantics beyond
  the reference defaults.
- E1 (SDTT distillation) scoping now has its baseline: any distilled student
  must beat 21/42 draft-only at <= 1 GB to earn a training run.

## Unresolved questions
1. D4 hybrid (FastDLLM drafts + Dream repairs): build the two-server harness
   path next session (recommended - it is pure kintsugi work, no engine risk)?
2. KV cache for committed blocks: standard llama cache or the fork's pkv
   machinery (recommended: standard - block-causal IS causal at block
   granularity; the fork machinery is for bidirectional-over-store, not needed)?
3. Bench accounting for block-AR models: keep i-tier as structural zeros in the
   48-case total, or report /42 alongside (recommended: report both, the doc
   already does)?
