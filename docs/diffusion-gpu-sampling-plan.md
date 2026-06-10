# Plan: GPU-accelerated sampling for diffusion LLMs

Status: design / local-fork implementation plan (2026-06-10).
Goal: make `llama-diffusion-cli` (Dream, LLaDA, LLaDA-MoE, RND1, DiffuCoder) sample on the GPU instead of copying full logits to the CPU every diffusion step.

## 1. Verification of the "CPU-bound" claim

The claim "llama.cpp does not support running diffusion LLMs on GPU" is **partially true**:

- The model forward pass DOES run on GPU (`-ngl 99` works; non-causal attention, no KV cache).
- The per-step **sampling loop is entirely CPU**:
  - Every step submits all `max_length` positions with `logits=1` and calls `llama_decode`
    (`examples/diffusion/diffusion.cpp:194-253`).
  - The full logits matrix `max_length x n_vocab` (fp32) is copied device-to-host
    (`src/llama-context.cpp:1420-1426`, `ggml_backend_tensor_get_async`).
    At 512 x 128k vocab that is ~256 MB per step, every step.
  - Per masked position, a `llama_token_data_array` of `n_vocab` candidates is built and the
    CPU sampler chain (top_k -> top_p -> temp -> dist) is applied; confidence
    (selected prob / entropy / margin / random) is computed in plain C++
    (`examples/diffusion/diffusion.cpp:287-388`, `calculate_confidence` at 13-46).
  - Transfer-count schedule + `std::partial_sort` over confidences + unmasking: host code.

Per PR #14644 discussion, transfer itself is ~15 ms/step (PCIe-bound); the dominant cost is the
CPU-side per-position sampling. Both go away with backend sampling.

## 2. Existing infrastructure (what we reuse)

Backend (GPU) sampling was merged in PR #17004 (Jan 2026), extended by #23287 (MTP draft path):

- `llama_sampler_i` has `backend_init` / `backend_apply` / `backend_accept` / `backend_set_input`
  (`include/llama.h:1240-1272`); `llama_sampler_data { logits, probs, sampled, candidates }` are
  ggml tensors.
- Samplers with backend graphs in `src/llama-sampler.cpp`: greedy (`ggml_argmax`), top_k
  (`ggml_top_k` + gather), top_p, temp, temp_ext, min_p, dist (softmax + cumsum + step
  inverse-CDF; host-supplied uniform RNG via `backend_set_input`).
- All required ggml ops have CUDA kernels: ARGMAX (`ggml-cuda.cu:2831`), STEP (2894),
  CUMSUM (3099), SUM_ROWS (3102), TOP_K (3114), ARGSORT (3117), batched GET_ROWS/SET_ROWS.
- Attach: `llama_set_sampler(ctx, seq_id, smpl)` (`include/llama.h:1283`) or at context creation
  via `llama_sampler_seq_config`; `common_init_from_params` already wires this when
  `params.sampling.backend_sampling` is set (`common/common.h:282`, `common/common.cpp:1266-1303`).
- Readback: `llama_get_sampled_token_ith / _probs_ith / _candidates_ith`
  (`src/llama-context.cpp:941-1060`) - these already resolve **arbitrary output rows**
  (`output_resolve_row`), and host buffers are sized per output row.
- When all sequences have backend samplers, `needs_raw_logits()`
  (`src/llama-context.cpp:1627-1642`) skips the big logits D2H copy entirely.

Maintainer signal: am17an (diffusion author) on PR #17004: backend sampling "would be useful
immediately for the diffusion-cli. I'm happy to test this when it's ready." No existing PR does this.

## 3. The three blockers

1. **One row per sequence.** `build_sampling()` (`src/llama-graph.cpp:3030-3116`) keeps only the
   LAST output row per sequence (`seq_to_logit_row` overwrites at 3044-3050) and passes a 1-row
   `ggml_view_1d` to `backend_apply`. Diffusion needs all `max_length` rows of one sequence.
2. **Diffusion goes through `encode()`, not `decode()`.** Diffusion archs return `nullptr` from
   `create_memory()` (`src/llama-model.cpp:2011-2017`), so `decode()` redirects to `encode()`
   (`src/llama-context.cpp:1650`). `encode()` copies logits unconditionally and never runs the
   sampling-output copy block (that exists only in `decode()` at 1947-1958).
3. **Explicit guard.** `decode()` errors out if a sequence has more than one output token while
   samplers are attached (`src/llama-context.cpp:1672-1693`).

## 4. Design

### Part 1 - core: multi-row backend sampling

`src/llama-graph.cpp` - `build_sampling()`:
- Track `{first_row, n_rows}` per sequence; require output rows of a sequence to be contiguous
  (true for diffusion: single seq, `split_simple`, all positions). Non-contiguous -> skip backend
  sampling for that seq (falls back to CPU path via `needs_raw_logits`).
- Pass `ggml_view_2d(logits_t, n_vocab, n_rows, ...)` to `backend_apply`; keep `ggml_view_1d`
  when `n_rows == 1` so existing graphs are bit-identical. Keep the `ggml_pad` dummy-row trick
  (3055-3058) for graph staticness.
- Graph stays static across diffusion steps: all positions always have `logits=1`; selecting the
  masked subset stays host-side. No per-step graph reallocation.

`src/llama-sampler.cpp` - generalize `backend_apply` impls to `n_rows = ggml_nrows(logits)`:
- Add helper: `llama_sampler_backend_gather(ctx, values [n,n_rows], idx [k,n_rows]) -> [k,n_rows]`
  via `ggml_reshape_3d` + batched `ggml_get_rows`; replaces ~6 duplicated 1-row gather idioms.
- greedy, temp: already row-wise, no change.
- top_k (1281): fix the two gathers. top_p (1427): drop `ggml_nrows == 1` assert in the sort
  helper, `ggml_sum` -> `ggml_sum_rows`, batched ones/set_rows inclusivity trick.
- min_p (1618): gather max-logit per row, threshold becomes `[1, n_rows]`, broadcast sub.
- temp_ext (2005): `ggml_sum` -> `ggml_sum_rows`, broadcast div.
- dist (1144): `inp_uniform` becomes `[n_rows]`; `backend_set_input` (1201-1215) fills
  `ggml_nelements` draws; `ggml_sum` -> `ggml_sum_rows`; fix candidates gather.
  Note: RNG stream differs from CPU path at the same seed (n draws/step vs 1) - expected.
- `llama_sampler_backend_support()` (559-622): probe with 2-row tensors so backends lacking
  batched ops are rejected up front -> clean CPU fallback.

`src/llama-context.cpp`:
- `encode()`: gate the logits copy (1420-1426) with `needs_raw_logits()`; add the sampling-output
  copy block mirroring `decode()` 1947-1958.
- Relax the one-output-per-seq guard (1672-1693).
- Copy helpers (1522-1625): `build_seq_to_output_row` returns `{first_row, count}`;
  `copy_tensor_async_ints` copies `n_rows` ints in one async call; `_floats` / `_candidates`
  copy per row (host stride is `n_vocab`, device stride is `k`).
- Readback (`get_sampled_*_ith`): no change needed.

`tests/test-backend-sampler.cpp`:
- Flip `test_backend_max_outputs` (970-1007) from expect-fail to expect-success.
- New multi-output test: 1 seq, M output tokens, chains `top_k -> temp -> dist` and greedy;
  compare each `llama_get_sampled_token_ith(i)` against CPU sampling of `llama_get_logits_ith`
  rows from a sampler-free context.

### Part 2 - diffusion-cli wiring

`examples/diffusion/diffusion-cli.cpp`, `diffusion.h`, `diffusion.cpp`:
- Build the same sampler chain (top_k -> top_p -> temp -> dist, mirroring diffusion.cpp:133-145)
  and attach via `llama_set_sampler(ctx, 0, chain)`. Enable with the existing
  `--backend-sampling` flag (`common_params_sampling.backend_sampling`) - no model-specific or
  diffusion-specific CLI flags (maintainer requirement from PR #17454 review).
- Per step (backend path): fill batch as today; `llama_decode`; for each masked position read
  `tok = llama_get_sampled_token_ith(ctx, row)`, candidate probs + ids via
  `llama_get_sampled_probs_ith` / `llama_get_sampled_candidates_ith`, where
  `row = shift_logits ? max(pos-1, 0) : pos` (replaces `get_logits_for_pos`, Dream shifts logits).
- Confidence from the k candidate probs on host (k = top_k, O(k) per position):
  - CONFIDENCE / ORIGIN: prob of `tok` (scan candidates for the sampled id);
  - ENTROPY: `-sum p log p` over the k probs;
  - MARGIN: top-2 scan (do not assume sorted - `ggml_top_k` output order is unspecified);
  - RANDOM: host RNG as today.
  These match CPU semantics: dist's `probs` is the softmax over post-filter logits, the same
  normalized candidate distribution the CPU chain leaves in `cur_p`.
- Schedule, partial_sort, alg_temp confidence sampling, unmasking: unchanged host code.
- CPU fallback retained when: `cfg_scale > 0` (CFG blends raw logits of two passes on host),
  `add_gumbel_noise` (CPU noise math over full vocab), `llama_set_sampler` fails (unsupported
  backend), or `top_k <= 0` (full-vocab probs readback would recreate the 256 MB copy; warn and
  suggest `--top-k`).

Result: per-step D2H goes from `max_length * n_vocab * 4` bytes (~256 MB) to
`n_rows * (4 + 8k)` bytes (~0.4 MB at k=64), and per-position CPU sampling disappears.

### Part 3 (optional, upstream-discuss-first) - in-graph CFG

CFG currently does two full decodes + host blend (diffusion.cpp:215-245). An in-graph blend is a
new pattern; open a discussion with maintainers before attempting. Not needed for the main win.

## 5. Verification

- Build: `cmake -B build -DGGML_CUDA=ON && cmake --build build -j --target llama-diffusion-cli test-backend-sampler`
- `./build/bin/test-backend-sampler` on CPU and CUDA devices; `ctest` sampler subset for 1-row regressions.
- End-to-end: Dream-7B and LLaDA-8B GGUF, run `llama-diffusion-cli` with and without
  `--backend-sampling`. The loop already logs time/step and sampling-time/step
  (diffusion.cpp:398-401) - record before/after.
- Correctness: greedy / temp=0 outputs must match the CPU path exactly; dist outputs differ
  (different RNG stream) - verify distributional sanity, document.

## 6. Risks

- Device memory (corrected in 8.2): `ggml_top_k` is a native op, so the GRAPH only holds
  `[k, n_rows]`; the full-vocab working set is transient CUDA pool memory inside the kernel
  (argsort fallback tmp = ncols x nrows I32 = ~311 MB at 512 x 151936). Still measure VRAM.
- CUDA TOP_K CUB path loops rows sequentially (one DeviceTopK launch per row; 512 launches/step).
  If it dominates the step time, the batched argsort fallback may actually be faster - measure.
- The 2-row support probe is harmless on CUDA (supports_op checks are per-op shape rules
  independent of row count - verified ggml-cuda.cu:5439+); Vulkan/Metal unverified, but failure
  mode is clean CPU fallback.
- `llama_get_logits` returns stale data when the logits copy is skipped; diffusion code must not
  mix backend and CPU paths within a run.
- Per-row async copies for probs/candidates could become the new bottleneck at large
  `max_length` x large k; the per-step timing log will show it.
- Without CUB (CUDART < 11.7), TOP_K/ARGSORT are capped at ne0 <= 1024 -> backend path falls back
  to CPU cleanly via the support probe + warmup check. This machine: CUDA 13.3, CUB on.

## 7. Upstream path

llama.cpp does not accept predominantly AI-generated PRs (AGENTS.md / CONTRIBUTING.md). For
upstreaming, the contributor must rewrite/own the code, hand-write PR descriptions and commit
messages, and disclose AI assistance. Split as:
- PR1: core multi-row backend sampling + tests (useful beyond diffusion: parallel decoding,
  speculative, server batch outputs).
- PR2: diffusion-cli switches to backend sampling with CPU fallback (benchmarks in description;
  am17an offered to test).
- PR3 (optional): in-graph CFG, after a design discussion.

## 8. Deep-dive findings and decisions (verified 2026-06-10)

Everything below was verified by reading the code directly; it corrects/extends sections 1-6.

### 8.1 Confirmed mechanics

- Diffusion archs return nullptr from `create_memory()` (src/llama-model.cpp:2011-2017), so
  `llama_decode` redirects to `encode()` (src/llama-context.cpp:1650). `encode()` copies the full
  logits unconditionally (1420-1426) and never copies sampling outputs back. The sampling GRAPH is
  built even for encoder graphs (`build_sampling()` called unconditionally,
  src/llama-model.cpp:2233; samplers passed via graph_params, src/llama-context.cpp:2319), and
  `llm_graph_input_sampling::set_input` (src/llama-graph.cpp:849-870) refreshes RNG inputs per
  step - so encode only lacks the host copy-back, nothing else.
- `llama_set_sampler` works post-creation: sets `sched_need_reserve = true`
  (src/llama-context.cpp:1153-1208) and the next decode re-reserves. `token_ids_full_vocab` is
  initialized unconditionally at context creation (src/llama-context.cpp:396-404).
- Host output buffers are already sized per output row (`n_vocab * n_outputs_max` for
  logits/probs/candidates, `n_outputs_max` for sampled; src/llama-context.cpp:2048-2155), and
  `get_sampled_*_ith` resolves arbitrary rows. Multi-row readback fits with no resizing.
  Size note: with sampling enabled at 512 rows x 151k vocab the pinned host buffer is ~1.2 GB
  (logits + sampling.logits + sampling.probs + candidates). Fine with 65 GB RAM; scales with
  n_ubatch - do not run with n_ubatch 2048+ without checking RAM.
- ggml op semantics: `ggml_top_k` is per-row, indices "in no particular order" (ggml.h:2385-2390);
  `ggml_get_rows` is batched (`a [n,ne1,ne2]`, `b I32 [n_rows,ne2]`, ggml.h:1656-1660);
  `ggml_set_rows` takes I64 indices per docs (ggml.h:1672-1687) though top_p currently passes I32
  (works; verify when batching); `ggml_cumsum` is per-row on CPU (ops.cpp:1407) and CUDA
  (cumsum.cu:214, with multi-row heuristics added by PR #17004 itself).
- CUDA `top_k` (top-k.cu:51-97): CUB path loops rows SEQUENTIALLY (one DeviceTopK launch per
  row - 512 launches/step at max_length=512); the argsort fallback is fully batched. Measure; if
  the CUB loop dominates, prefer the argsort path for multi-row or note as follow-up.
- CPU sampler chain in diffusion (diffusion.cpp:133-143): top_k -> top_p -> temp -> dist, where
  temp is only added when temperature > 0. The backend temp sampler at temp <= 0 does ARGMAX
  (llama-sampler.cpp:1827-1843) - so to match CPU semantics, do NOT add temp to the backend chain
  when temperature <= 0 (dist then samples the unscaled filtered distribution, same as CPU).
- MARGIN confidence on CPU relies on cur_p being sorted descending (top_k impl sorts). Backend
  candidates are unordered - host must do a top-2 scan over the k probs, not take [0] and [1].
- CPU `calculate_confidence` operates on the POST-FILTER candidate array (after top_k/top_p), and
  dist normalizes probs over it (llama-sampler.cpp:1060-1091). Backend `data->probs` is
  `ggml_soft_max` over post-filter logits (-INF for filtered -> p=0). Entropy/confidence values
  match (filtered entries contribute 0).
- `llama_sampler_chain` backend behavior: `backend_init` marks per-sampler `is_backend` flags and
  `backend_apply` STOPS at the first unsupported sampler (llama-sampler.cpp:694-763) - a
  partially-supported chain silently produces filtered logits instead of sampled tokens, and
  `llama_set_sampler` IGNORES the backend_init result (llama-context.cpp:1182). Detection: after
  the first warmup decode, `llama_get_sampled_token_ith(ctx, 0) == LLAMA_TOKEN_NULL` means the
  backend path is not producing tokens -> detach sampler (`llama_set_sampler(ctx, 0, nullptr)`)
  and fall back to the CPU loop.
- Graph reserve gap: `graph_reserve()` marks only ONE output token per sequence
  (src/llama-context.cpp:2267-2274) regardless of the n_outputs argument, so the worst-case
  reserve graph contains a 1-row sampling subgraph while real diffusion steps need max_length
  rows. VERIFIED consequence (ggml-backend.cpp:1489-1537): an oversized graph does NOT crash -
  `ggml_backend_sched_alloc_splits` re-reserves (`ggml_gallocr_reserve_n`) and retries, failing
  only if the retry fails. So the miss costs a one-time compute-buffer reallocation (VRAM spike +
  latency) on the first real step, not an error. DECISION (unchanged, now "strongly recommended"
  rather than "critical"): in `graph_reserve`, mark ALL ubatch tokens as output when the model
  has no memory (mctx == nullptr). This matches what `encode()` actually does (output_all = true)
  and does not change reserve sizing for causal models.
- diffusion-cli builds `llama_context_params` manually (diffusion-cli.cpp:137-146) - it does NOT
  go through `common_init_from_params`, so the common backend-sampling auto-attach does not
  apply. Attach explicitly with `llama_set_sampler(ctx, 0, chain)`.
- Pre-existing bug: `--diffusion-cfg-scale` and `--diffusion-alg-temp` are parsed into
  `params.diffusion` (common/arg.cpp:3881+) but never copied into `diff_params`
  (diffusion-cli.cpp:202-211) - CFG and alg_temp are dead code via the CLI today. Fix in passing.
- `-bs` / `--backend-sampling` (common/arg.cpp:1920-1926) is a general sampling arg, available to
  the diffusion example.

### 8.2 Second-pass verification (every remaining assumption read first-hand)

Graph/ggml layer:
- `ggml_build_forward_select` (ggml.c:6982, doc ggml.h:2680-2711): builds ALL branch tensors into
  the graph but computes only the selected one - keeps topology constant regardless of which
  samplers are active. Shape-independent; works unchanged for multi-row.
- `ggml_top_k` is a NATIVE op (ggml.c:5328-5340): dst is `I32 [k, ne1, ne2, ne3]` - per-row by
  construction. Graph-visible memory is only `[k, n_rows]`; the full-vocab intermediate exists
  only inside the CUDA kernel (pool-allocated `ncols*nrows` I32 tmp in the argsort fallback,
  top-k.cu:79; the CUB DeviceTopK path avoids it but loops rows sequentially, top-k.cu:66-72).
- `ggml_set_rows` asserts (ggml.c:3901-3917): index tensor may be I64 OR I32; batched with
  broadcast (`b.ne2 % c.ne1 == 0`). Batched top_p inclusivity-fix shapes that satisfy the
  asserts: dst `[1, n, n_rows]`, src `[1, 1, n_rows]`, idx `[1, n_rows, 1]`.
- `ggml_cumsum`, `ggml_argmax`, `ggml_soft_max`, `ggml_argsort` are all per-row on CPU and CUDA
  (ops.cpp:1407, cumsum.cu:214 - the multi-row heuristics there were added by PR #17004 itself).
- Broadcast rules check out for every planned `[1, n_rows]` vs `[n, n_rows]` sub/div
  (ggml_can_repeat: ne0 1 -> n, ne1 n_rows -> n_rows).

Sampler layer:
- Disabled samplers are `llama_sampler_init_empty` no-ops with FULL backend support
  (llama-sampler.cpp:434-519: backend_init returns true, backend_apply does nothing) - e.g.
  `top_k(k<=0)` (1321-1326) and `top_p(p>=1)` (1513-1518). Chain composition with disabled
  samplers is safe, BUT a chain whose top_k is disabled feeds dist full-vocab logits: softmax +
  cumsum + mask at `[151936, 512]` f32 is ~311 MB per intermediate, several of them -> enforce
  top_k > 0 for the backend diffusion path.
- After the chain, `data->candidates [k, n_rows]` (vocab ids) and `data->probs [k, n_rows]`
  (dist's softmax over the SAME filtered set, llama-sampler.cpp:1157+1198) stay aligned: top_p
  reorders both consistently (1453-1457), dist modifies neither layout. Host confidence reads
  (probs[i], candidates[i]) pairs; per-row element counts arrive via `counts[row] = ne0` set
  during the async copy (llama-context.cpp:1592).
- `llama_sampler_logit_bias_backend_apply` (llama-sampler.cpp:3486-3516) writes the bias with
  FLAT row-0 indexing (`reshape_2d(fill(logits,0), 1, nelements)` + set_rows with `[n_bias]`
  vocab indices) - NOT multi-row safe as-is (would bias only row 0's slice). Diffusion's chain
  does not use it; generalize via broadcast set_rows (idx `[n_bias, 1, 1]` broadcasts across
  batches per the ggml_set_rows asserts) or assert n_rows == 1 in it.
- `temp_ext` multi-row deltas confirmed minimal (llama-sampler.cpp:2005-2066): `ggml_sum` at 2034
  -> `ggml_sum_rows`; `max_entropy = logf(ne[0])` is a scalar constant (fine); final
  `ggml_div(logits, dyn_temp)` broadcasts `[1, n_rows]`.
- `min_p` multi-row deltas confirmed (1618-1655): only the max-logit gather (1628-1632) is
  1-row-specific; threshold/sub/step/log/add all broadcast or elementwise.
- Chain prefix semantics confirmed (llama-sampler.cpp:694-763): `backend_apply`/`set_input` stop
  at the FIRST sampler whose backend_init failed; `llama_set_sampler` ignores the init result
  (llama-context.cpp:1182). The warmup-probe fallback (section 8.1) covers this.
- A backend-initialized chain CANNOT be reused for CPU sampling: `llama_sampler_chain_apply`
  (llama-sampler.cpp:642-662) SKIPS every sampler marked `is_backend` when the chain was
  backend-initialized (hybrid continue-on-CPU design), and `is_init` is one-way (re-init asserts,
  llama-sampler.cpp:699). After a warmup fallback, the CPU path must build a FRESH chain - never
  call llama_sampler_apply on the chain that was attached via llama_set_sampler.
- `llama_sampler_init(iface, ctx)` is public (llama.h:1286) - the Task 7 example-side custom
  confidence sampler is possible without core changes.

Context layer:
- `needs_raw_logits` (llama-context.cpp:1627-1642): returns false only when EVERY output token's
  every seq has a backend sampler - diffusion (1 seq, sampler attached) skips the logits copy.
- `process_ubatch` (1269-1330): `res->set_inputs(&ubatch)` runs on BOTH the graph-reuse and
  rebuild paths -> dist's fresh uniform draws are uploaded every step.
- Graph-reuse conditions with samplers (llama-graph.h:660-690 + llama-graph.cpp:873-884):
  n_outputs equal, samplers map equal by POINTER, and per-token `output[i]`/`seq_id[i][0]` equal.
  All stable across diffusion steps (same shape, same attached chain) -> the graph is built once
  and reused for all steps; only input data changes.
- `output_resolve_row` (810-843): the `_ith` index is the BATCH TOKEN INDEX translated through
  `output_ids`; encode() sets `output_ids[i] = i` (1388-1390), so row == position. The
  shift_logits mapping `row = pos - 1` holds. `output_reorder()` is a no-op after encode
  (output_swaps only populated by decode's reordering).
- `ubatch.output[i] = batch.logits[idxs[i]]` in order (llama-batch.cpp:723), `split_simple`
  preserves order -> a single sequence's output rows are contiguous and position-ordered.
- Host pinned-buffer math (output_reserve, llama-context.cpp:2028-2155): buffer is sized from the
  ACTUAL n_outputs of the batch (`max(n_outputs, n_seq_max)`), not cparams max - at 512 rows x
  151936 vocab with sampling on: logits 311 MB + sampling.logits 311 MB + probs 311 MB +
  candidates 311 MB + sampled 2 KB = ~1.25 GB host (pinned if the device offers host buffers).

CLI/args/test layer:
- Args default to `examples = {LLAMA_EXAMPLE_COMMON}` (arg.h:20) and the parser accepts an arg if
  `in_example(ex) || in_example(LLAMA_EXAMPLE_COMMON)` (arg.cpp:1081) -> `-bs` already reaches
  llama-diffusion-cli. The two-name negatable bool pattern exists (arg.cpp:3609-3617,
  `--spec-draft-backend-sampling` / `--no-...`). DECISION: make `-bs` negatable (add
  `--no-backend-sampling`) and set `params.sampling.backend_sampling = true` in diffusion-cli
  BEFORE common_params_parse (pre-parse default override is the established example pattern).
- CPU-path sampling defaults the backend chain must mirror: top_k 40, top_p 0.95, temp 0.8
  (common.h:216-222); chain composition rule: temp added only when temp > 0
  (diffusion.cpp:140-142), dist always last.
- test-backend-sampler requires a real GGUF: `--model` or `get_model_or_exit` (env
  `LLAMACPP_TEST_MODELFILE`, tests/get-model.cpp); `--device cpu|gpu` supported (test file 20-68);
  ctest label "model" (tests/CMakeLists.txt:248). The Gemma-12B GGUFs on disk would work but are
  slow; download Qwen2.5-0.5B-Instruct Q8_0 (~600 MB) as the test model.
  `test_backend_max_outputs` (970-1007) asserts `llama_decode` FAILS on multi-output (1001-1002);
  flip to expect success after Task 1.
- Mask token plumbing: vocab reads `tokenizer.ggml.mask_token_id` (llama-vocab.cpp:2482) and the
  model-load log already prints `MASK token = <id> '<piece>'` (llama-vocab.cpp:3719).
  `llama_token_to_piece(..., special=true)` renders it (llama.h:1146-1154). diffusion-cli already
  tokenizes the prompt with parse_special=true (diffusion-cli.cpp:159-162), so `<mask-piece>`
  markers in the prompt tokenize to mask ids TODAY with no tokenizer change - infill is purely
  init/printing logic.

Backend support / CUDA specifics (third pass):
- `ggml_backend_dev_supports_op` on CUDA (ggml-cuda.cu:5439-5445): TOP_K and ARGSORT are limited
  to `ne[0] <= 1024` WITHOUT CUB; unrestricted with `GGML_CUDA_USE_CUB`. CUB is auto-defined for
  `CUDART_VERSION >= 11070` (common.cuh:109-111) - no CMake flag needed. This machine has CUDA
  13.3 (nvcc verified) -> full-vocab top_k supported. If CCCL < 3.2 the DeviceTopK path is absent
  but the CUB argsort fallback (batched, top-k.cu:74-88) still handles full vocab. Without CUB at
  all (e.g. old toolkit), the support probe fails on TOP_K (the chain's FIRST sampler) -> whole
  chain falls back to CPU via the warmup probe - graceful.
- ARGMAX and CUMSUM supports_op are unconditional true (ggml-cuda.cu:5333-5337, 5475); SUM_ROWS
  requires contiguous src (5446-5449, ours are); SET_ROWS accepts I64 or I32 indices (5257-5264).
- The public sampled-readback wrappers DO synchronize before reading
  (`llama_get_sampled_token_ith` etc. call `ctx->synchronize()`, llama-context.cpp:3631-3650) -
  the async D2H copies are safe to consume right after llama_decode returns.
- `output_reserve` re-fills `sampling.sampled` with LLAMA_TOKEN_NULL on EVERY call
  (llama-context.cpp:2141-2145), and encode calls output_reserve each step - so the
  NULL-token warmup probe is reliable per step, not just on the first one.
- PITFALL for the reserve fix: `ubatch_reserve` (llama-batch.cpp:391-413) zero-initializes
  `n_seq_id[]` and `seq_id[]` (nullptr) for ALL tokens; the current graph_reserve loop
  (llama-context.cpp:2268-2274) only assigns seq ids to the first n_seqs tokens.
  `build_sampling` reads `ubatch.seq_id[i][0]` for every output token -> when marking all tokens
  as output, ALSO set `n_seq_id[i] = 1` and `seq_id[i] = &seq_ids[i / n_seq_tokens]` for every
  token, or it nullptr-derefs during reserve.
- `diffusion_generate`'s input guard (diffusion.cpp:110) rejects `max_length <= n_input` - an
  infill canvas has n_input == max_length, so the guard must allow equality in infill mode (the
  tail mask fill range is then empty, which is correct).

### 8.3 User decisions

- Local fork first; correctness + speed over upstream-shaped caution. Upstream rewrite later.
- Backend sampling DEFAULT ON in diffusion-cli when supported (warmup probe), automatic CPU
  fallback otherwise; `--no-backend-sampling`-style escape via existing flag handling.
- Inpainting is a hard requirement and must keep working with GPU sampling. Interface: mask
  markers written directly in the prompt text (e.g. `<|mask|>` repeated N times per hole),
  tokenized with parse_special=true so each marker becomes one mask token. The denoising loop
  already handles arbitrary mask positions (mask_positions scan, diffusion.cpp:269-277), so this
  is an input/output-interface change, orthogonal to GPU sampling.
- Test models: Dream-7B Q4_K_M (baseline; bartowski GGUF) then DiffuCoder-7B-cpGRPO Q4_K_M
  (Elixir use case). Machine: RTX 5070 Laptop 8 GB, driver 595, 567 GB free disk, no build dir
  yet.

## 9. Step-by-step implementation tasks

Each task is independently buildable/verifiable. Order matters.

### Task 0 - baseline setup
1. `cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=ON` ;
   `cmake --build build -j --target llama-diffusion-cli test-backend-sampler`
2. Download Dream-7B: `hf download bartowski/Dream-org_Dream-v0-Instruct-7B-GGUF --include "*Q4_K_M*" --local-dir ~/models/dream7b`
   and a small causal model for the sampler tests:
   `hf download Qwen/Qwen2.5-0.5B-Instruct-GGUF --include "*q8_0*" --local-dir ~/models/qwen05b`
3. Baseline run (CPU sampling), record total time, time/step, sampling time/step at
   `--diffusion-steps 128 -ub 512 -ngl 99 --diffusion-eps 0.001 --diffusion-algorithm 4`.
   Note the `MASK token` line from the model-load log (needed for infill markers).
4. `LLAMACPP_TEST_MODELFILE=~/models/qwen05b/<file>.gguf ./build/bin/test-backend-sampler
   --device cpu` and `--device gpu` to confirm green before any change.

### Task 1 - core: multi-row plumbing (src/llama-context.cpp, src/llama-graph.cpp)
1. `graph_reserve` (llama-context.cpp:2243): when `mctx == nullptr`, set `ubatch.output[i] = true`
   AND `n_seq_id[i] = 1`, `seq_id[i] = &seq_ids[i / n_seq_tokens]` for ALL tokens
   (ubatch_reserve leaves seq_id nullptr beyond the first n_seqs tokens - see section 8.2
   pitfall; build_sampling derefs seq_id[i][0] for every output token).
2. Relax the decode guard (llama-context.cpp:1672-1693): drop the hard error; allowed now.
3. `build_seq_to_output_row` (llama-context.cpp:1522): return `{first_row, n_rows}` per seq
   (struct or pair); assert contiguity of each sequence's output rows.
4. `copy_tensor_async_ints` (1540): copy `ggml_nbytes(tensor)` to `sampled.data + first_row`.
   `copy_tensor_async_floats` / `_candidates` (1565/1596): loop tensor rows, one async copy per
   row into `dst.data + (first_row + r) * stride`, set `counts[first_row + r] = ne0`. Shortcut:
   single copy when `ne0 == stride`.
5. `encode()` (llama-context.cpp:1341): gate the logits copy with
   `needs_raw_logits(ubatch, sampling.samplers)` (function works as-is); after extraction add the
   sampling copy block mirroring decode (1947-1958) with `row_offset = 0`.
6. `build_sampling()` (llama-graph.cpp:3030): build `{first_row, n_rows}` map (keep ALL output
   rows, not last-wins); pass `ggml_view_2d(logits_t, ne0, n_rows, nb1, first_row * nb1)` when
   n_rows > 1, keep `ggml_view_1d` when n_rows == 1. Keep ggml_pad dummy-row trick and
   `ggml_build_forward_select` logic unchanged.
   Verify: build, run test-backend-sampler (existing 1-row tests must stay green; multi-output
   test still fails at sampler level - expected until Task 2).

### Task 2 - core: generalize backend samplers (src/llama-sampler.cpp)
1. Add static helper `llama_sampler_backend_gather(ctx, values, idx)`:
   `values [n, n_rows]`, `idx I32 [k, n_rows]` -> `[k, n_rows]` via
   `ggml_reshape_3d(values, 1, n, n_rows)` + `ggml_get_rows` + reshape. Replaces the
   `reshape_2d(x, 1, n) + get_rows + reshape_1d` idiom in top_k/temp/dist/min_p (n_rows==1 stays
   numerically identical).
2. greedy (984): no change (`ggml_argmax` per-row already).
3. temp (1822): replace 1-row gathers in the temp<=0 branch with the helper (we will not use that
   branch from diffusion, but tests may).
4. top_k (1281): use helper for both gathers; final shapes `[k, n_rows]`.
5. top_p (1427): remove the `ggml_nrows(a) == 1` assert by replacing the `ggml_sort` lambda with
   the helper; `ggml_sum` -> `ggml_sum_rows` (-> `[1, n_rows]`); clamp unchanged; the
   ones/set_rows inclusivity trick goes batched with shapes that satisfy the ggml_set_rows
   asserts (ggml.c:3906-3914): dst `[1, n, n_rows]`, src `[1, 1, n_rows]`, idx `[1, n_rows, 1]`
   (I32 accepted alongside I64); `scale_bias` bias is scalar - per-row safe.
6. min_p (1618): helper for the max-logit gather (1628-1632, the only 1-row-specific part);
   threshold `[1, n_rows]` broadcasts through `ggml_sub`.
7. temp_ext (2005): `ggml_sum` (2034) -> `ggml_sum_rows`; `max_entropy = logf(ne[0])` stays a
   scalar; final `ggml_div` broadcasts `[1, n_rows]`.
8. dist (1144): `inp_uniform = ggml_new_tensor_1d(ctx, F32, n_rows)`; in apply, reshape to
   `[1, n_rows]` before `ggml_sub` with cumsum `[n, n_rows]`; `ggml_sum` -> `ggml_sum_rows`;
   candidates remap via helper with idx `[1, n_rows]`; `backend_set_input` (1201): fill
   `ggml_nelements(inp_uniform)` draws, single `ggml_backend_tensor_set`.
9. logit_bias (3486-3516): NOT multi-row safe as-is (flat row-0 set_rows indexing). Either
   generalize with broadcast set_rows (idx `[n_bias, 1, 1]` broadcasts across batches) or add
   `GGML_ASSERT(ggml_nrows(data->logits) == 1)` so misuse fails loudly. Diffusion's chain does
   not include it.
10. `llama_sampler_backend_support` (559): probe with `ggml_new_tensor_2d(ctx, F32, n, 2)` (and
    matching 2-row candidates) so non-batched backends reject cleanly.
    Verify: build; existing tests green (1-row goes through same code with n_rows=1).

### Task 3 - tests (tests/test-backend-sampler.cpp)
1. Flip `test_backend_max_outputs` (970-1007) to expect success; assert each row's sampled token
   is valid.
2. Add `test_backend_multi_output`: one seq, M=8 tokens all logits=1, chains
   {greedy}, {top_k(40), dist}, {top_k(40), top_p(0.9), temp(0.8), dist}; per row compare against
   CPU sampling of `llama_get_logits_ith(i)` from a sampler-free context (greedy: exact match;
   dist: valid candidate membership + distribution sanity as existing dist tests do).
3. Run with `LLAMACPP_TEST_MODELFILE=<qwen0.5b q8_0>` on `--device cpu` and `--device gpu`.

### Task 4 - diffusion example: backend path (examples/diffusion/)
1. diffusion-cli.cpp: fix the pre-existing bug - wire `diff_params.cfg_scale =
   params.diffusion.cfg_scale` and `diff_params.alg_temp = params.diffusion.alg_temp`.
2. Flag plumbing: make `-bs` negatable in common/arg.cpp (add `--no-backend-sampling` via the
   two-name bool pattern, cf. arg.cpp:3609-3617); in diffusion-cli main, set
   `params.sampling.backend_sampling = true` BEFORE `common_params_parse` (pre-parse default
   override). diffusion.h: add `bool backend_sampling` to `diffusion_params`, fed from
   `params.sampling.backend_sampling`. Backend path additionally requires `top_k > 0` (else
   dist's softmax/cumsum run at `[n_vocab, n_rows]`, ~311 MB per intermediate at 512 rows) - if
   top_k <= 0, log a warning suggesting `--top-k` and fall back to CPU.
3. diffusion.cpp `diffusion_generate`:
   - Build a DEDICATED backend chain when the backend path is eligible (no cfg_scale, no
     add_gumbel_noise, top_k > 0); attach via `llama_set_sampler(ctx, 0, sampler)`; on false ->
     CPU path.
   - Warmup probe: run the first decode, check `llama_get_sampled_token_ith(ctx, 0)`; if
     LLAMA_TOKEN_NULL -> `llama_set_sampler(ctx, 0, nullptr)`, log fallback, continue on CPU path
     (logits become available again next decode since needs_raw_logits flips). The CPU path must
     use its OWN fresh chain - a backend-initialized chain skips its offloaded samplers in CPU
     apply (llama-sampler.cpp:642-662, see 8.2).
   - Backend per-step path: decode; for each masked pos compute
     `row = shift_logits ? std::max(pos - 1, 0) : pos`; read sampled token, probs + count,
     candidates + count; host confidence: CONFIDENCE/ORIGIN = prob of sampled id (scan),
     ENTROPY = -sum p log p, MARGIN = top-2 scan, RANDOM = host RNG. Keep schedule /
     partial_sort / alg_temp / unmask code unchanged (it consumes (confidence, sampled) pairs).
   - ORIGIN algorithm: backend path needs only sampled tokens - supported.
   - temp <= 0: do not add the temp sampler to the chain (match CPU semantics).
   - Keep the CPU path fully intact behind `if`.
4. Log mask token string at startup with `llama_token_to_piece(vocab, mask_token_id, ...,
   special=true)` (the visual callback's special=false rendering hides it) - needed for the
   inpainting harness. The model-load log already prints it (llama-vocab.cpp:3719).
   Verify: Dream-7B run with `-bs` (default) vs `--no-backend-sampling`; same flags; compare
   quality qualitatively, timings quantitatively (expect sampling time/step to collapse).

### Task 5 - inpainting (examples/diffusion/)
1. Add `--diffusion-infill` arg (common/arg.cpp diffusion section; common_params_diffusion gets
   `bool infill`).
2. diffusion-cli.cpp: when infill, skip chat-template formatting; tokenize the raw prompt with
   parse_special=true (mask markers -> mask ids); the token array IS the canvas:
   `diff_params.max_length = n_input_tokens` (assert <= n_ubatch); print the WHOLE canvas at the
   end (do not strip the first n_input tokens); visual callback prints from 0.
3. diffusion.cpp: add `bool infill` to params; when set, relax the input guard
   (diffusion.cpp:110) to allow `max_length == n_input`; skip the tail
   `std::fill(... mask_token_id)` (canvas already contains masks); assert at least one mask token
   present; restrict to TIMESTEP_BASED schedule initially (block-based assumes masks start at
   n_input - error out with a clear message). Set the visual callback's n_input to 0 so the whole
   canvas renders.
4. Works identically on CPU and backend paths (mask scan is position-agnostic).
   Verify: `-p 'def fib(n) do <|mask|>...<|mask|> end' --diffusion-infill` regenerates only the
   hole; fixed text unchanged in output.

### Task 6 - benchmark + validation
1. Timings: steps {64, 128, 256} x {-bs on, off} on Dream-7B Q4_K_M, `-ngl 99 -ub 512`; record
   total, time/step, sampling time/step, VRAM (nvidia-smi), host RAM.
2. Download DiffuCoder Q4_K_M (`Mungert/DiffuCoder-7B-cpGRPO-GGUF`); smoke-test an Elixir prompt
   end-to-end + one infill prompt.
3. `ctest -R sampler` (or direct test binaries) for regressions; run `test-backend-ops -o
   CUMSUM -o TOP_K -o ARGSORT -o GET_ROWS -o SET_ROWS` style checks if batched-op doubts arise.
4. Record results in this doc (section 10, to be added).

### Task 7 - optional follow-ups (defer)
- In-graph CFG blend (needs design discussion upstream).
- Coalesce per-row probs/candidates D2H copies (single strided copy or device-side pack).
- CUDA segmented top-k (replace per-row CUB loop) - upstream ggml work.
- Custom example-side confidence sampler producing `[n_rows]` confidences on device (cuts
  readback from k floats/row to 1; needs public llama_sampler_i, already exported).

## 10. Empirical baseline - Task 0 results (run 2026-06-10 on the RTX 5070 Laptop)

Everything below was MEASURED, not estimated:

- Build: `cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=ON` auto-selects `CMAKE_CUDA_ARCHITECTURES=
  120a-real` (Blackwell native); full build with nvcc 13.3 succeeds. Binaries in `build/bin/`.
- test-backend-sampler: ALL 13 tests pass on `--device cpu` AND `--device gpu`
  (model: `~/models/qwen05b/qwen2.5-0.5b-instruct-q8_0.gguf`, env `LLAMACPP_TEST_MODELFILE`).
  This exercises top_k/top_p/dist/min_p/temp/temp_ext/logit_bias backend graphs over the FULL
  151936-token Qwen vocab on this GPU - CUB path confirmed working at the exact width diffusion
  needs. `test_max_outputs` confirmed to currently expect failure (will flip in Task 3).
- Dream-7B baseline (CPU sampling path, the thing we are replacing):
  `llama-diffusion-cli -m ~/models/dream7b/Dream-org_Dream-v0-Instruct-7B-Q4_K_M.gguf
  -p "Write an idiomatic Elixir function ..." -ub 512 --diffusion-eps 0.001
  --diffusion-algorithm 4 --diffusion-steps 128 --temp 0.2 --top-k 40 -ngl 99`
  -> total 78845 ms, 615.98 ms/step, sampling 322.93 ms/step.
  **CPU sampling is 52% of every step** - the upper bound for this work is ~2x end-to-end, plus
  whatever the skipped 311 MB/step logits D2H adds.
- Output quality sanity: correct, compiling Elixir (pattern-matched recursive Fibonacci) at 128
  steps / temp 0.2 - the baseline behaves as the dLLMs.md report describes.
- Dream-7B mask_token_id = 151666 (from the run log). The piece string to use as the infill
  marker gets logged in Task 4 (render with special=true).
- Models on disk: `~/models/dream7b/Dream-org_Dream-v0-Instruct-7B-Q4_K_M.gguf` (4.7 GB),
  `~/models/qwen05b/qwen2.5-0.5b-instruct-q8_0.gguf` (676 MB).

Task 0 is COMPLETE. Implementation starts at Task 1.

## 11. Implementation log (what was ACTUALLY changed, with deviations from plan)

### Tasks 1-3 - core multi-row sampling (DONE, all 14 tests pass on CPU and CUDA)

src/llama-graph.cpp `build_sampling()`:
- `seq_to_logit_row` (last-row-wins map) replaced by `seq_to_logit_rows` mapping seq_id ->
  `{first, count}`; `count == 0` is the sentinel for non-contiguous output rows (sequence then
  treated as inactive -> dummy row 0, i_out = 0, CPU fallback via needs_raw_logits).
- Active samplers get `ggml_view_2d(logits_t, ne0, n_rows, nb1, first*nb1)` when n_rows > 1;
  the 1-row case keeps `ggml_view_1d` so single-output graphs keep their exact shape.

src/llama-context.cpp:
- The whole "backend sampling requires at most one output token per sequence" guard in decode()
  was DELETED (was llama-context.cpp:1671-1693).
- `build_seq_to_output_row` -> `build_seq_to_output_rows` returning `llama_seq_output_rows
  {first, count}` with the same count==0 sentinel; the static helper block (struct + 3 copy
  helpers + needs_raw_logits) MOVED above encode() (C++ needs declarations before use - encode
  precedes decode in the file).
- `copy_tensor_async_ints`: copies the whole sampled tensor (`ggml_nbytes`) to
  `sampled.data + first`; asserts `n_rows == ggml_nelements(tensor)`.
- `copy_tensor_async_floats`/`_candidates`: when tensor `ne0 == stride` (host stride = n_vocab)
  do ONE coalesced async copy; otherwise one async copy per row from byte offset
  `r*ne0*sizeof(T)` into `dst.data + (first+r)*stride`; `counts[first+r] = ne0` for every row.
- encode(): logits copy now gated on `needs_raw_logits(ubatch, sampling.samplers)`; sampling
  copy-back block added after the nextn extraction, identical to decode's but with
  `row_offset = 0` (encode is always a single ubatch).
- `graph_reserve()`: when `mctx == nullptr`, marks the first `min(n_tokens, n_outputs)` tokens
  as output AND assigns `n_seq_id[i] = 1`, `seq_id[i] = &seq_ids[i/n_seq_tokens]` (the
  ubatch_reserve nullptr pitfall from 8.2); the memory case keeps the old one-output-per-seq
  loop.

src/llama-sampler.cpp:
- New static helper `llama_sampler_backend_gather(ctx, values [n,n_rows], idx [k,n_rows])
  -> [k,n_rows]` = reshape values to `[1,n,n_rows]` + batched `ggml_get_rows` + reshape back.
  DEVIATION from plan: for n_rows==1 the result is `[k,1]` (2-D) where the old code produced
  1-D `[k]` - numerically identical, graph topology slightly different (extra reshapes); all
  existing 1-row tests still pass bit-exact.
- top_k: both gathers via helper. temp (<=0 branch): argmax result reshaped `[1,n_rows]` then
  helper. top_p: sort lambda (had `nrows==1` assert) replaced by helper; `ggml_sum` ->
  `ggml_sum_rows`; inclusivity fix batched as dst `[1,n,n_rows]` / src `[1,1,n_rows]` /
  idx `[1,n_rows,1]` per the ggml_set_rows broadcast asserts. min_p: argmax reshaped + helper;
  threshold `[1,n_rows]` broadcasts through ggml_sub. temp_ext: `ggml_sum` -> `ggml_sum_rows`;
  final div broadcasts. dist: `inp_uniform` sized `n_rows` (one draw per row), reshaped
  `[1,n_rows]` for the cumsum sub; `ggml_sum` -> `ggml_sum_rows`; candidates remap via helper;
  `backend_set_input` fills `ggml_nelements(inp_uniform)` draws in one tensor_set.
- logit_bias: generalized rather than guarded - bias row built from a 1-row view
  (`ggml_fill(ggml_view_1d(logits, ne0, 0), 0)` + set_rows as before) and broadcast-added to all
  rows (`ggml_add` `[n,n_rows]` + `[n]`).
- Support probe: logits/candidates probe tensors are now 2-row (`ggml_new_tensor_2d(.., n, 2)`).

tests/test-backend-sampler.cpp:
- `test_backend_max_outputs` flipped: multi-output decode must SUCCEED; greedy chain; every
  row's backend token compared against CPU argmax over `llama_get_logits_ith(i)` from a second,
  sampler-free reference context (same model -> bitwise-identical logits). Exact match required
  and observed on CUDA.
- New `test_backend_multi_output` (registered as "multi_output"): top_k(40) -> dist(seed 42)
  chain, all positions output; per row asserts: sampled token valid and among candidates with
  p > 0; probs/candidates counts == 40; every candidate's reference logit >= the row's 40th
  largest logit (std::nth_element on the reference row).
- Verified: 14/14 tests pass on --device cpu AND --device gpu (RTX 5070, Qwen2.5-0.5B q8_0).

### Task 4 - diffusion backend path (DONE, measured 2.55x end-to-end)

common/arg.cpp:
- `-bs`/`--backend-sampling` converted to the negatable two-name bool form (added
  `--no-backend-sampling`), mirroring `--spec-draft-backend-sampling`.

examples/diffusion/diffusion-cli.cpp:
- `params.sampling.backend_sampling = true` set BEFORE `common_params_parse` (pre-parse default
  override; the flag can still turn it off).
- Pre-existing bug fixed: `diff_params.cfg_scale` and `diff_params.alg_temp` are now wired from
  `params.diffusion` (they were parsed but silently dropped - CFG was dead code via the CLI).
- Mask token piece logged at startup via `llama_token_to_piece(..., special=true)`
  (renders as `<|mask|>` id 151666 for Dream-7B).

examples/diffusion/diffusion.h: `diffusion_params.backend_sampling` added.

examples/diffusion/diffusion.cpp `diffusion_generate()`:
- Eligibility: `backend_sampling && cfg_scale <= 0 && !add_gumbel_noise && top_k > 0`;
  each rejection logs its reason and falls back to the CPU path. A DEDICATED chain
  (top_k -> top_p if <1 -> temp if >0 -> dist, same composition rules as the CPU chain) is
  attached via `llama_set_sampler(ctx, 0, ...)`; on false -> CPU. The CPU chain is built
  regardless and is a DIFFERENT object (a backend-initialized chain skips its offloaded
  samplers in CPU apply - see 8.2).
- Warmup probe: on global step 0 after the first decode, if
  `llama_get_sampled_token_ith(ctx, 0) == LLAMA_TOKEN_NULL` the chain did not fully offload ->
  detach sampler, `use_backend = false`, REDO the decode (raw logits were skipped while the
  sampler was attached), continue on CPU.
- Backend per-step path: `row = shift_logits ? max(pos-1, 0) : pos` (lambda `get_row_for_pos`);
  ORIGIN reads only sampled tokens; confidence algorithms read sampled token + candidate
  probs/ids per masked row and compute confidence on host:
  CONFIDENCE = prob of sampled id (scan candidates), ENTROPY = -sum p log p over the k probs,
  MARGIN = top-2 scan (candidates are unordered!), RANDOM = host RNG. The transfer-count
  schedule, partial_sort, alg_temp dist and unmask code are untouched and shared by both paths.
- Cleanup: detach (`llama_set_sampler(ctx, 0, nullptr)`) before freeing the backend chain.

Measured (Dream-7B Q4_K_M, -ngl 99, -ub 512, 128 steps, alg 4, temp 0.2, top_k 40):
- CPU sampling:     78845 ms total, 615.98 ms/step, sampling 322.93 ms/step
- backend sampling: 30878 ms total, 241.23 ms/step, sampling   1.85 ms/step
- => 2.55x end-to-end; per-step sampling cost down 175x. The extra ~80 ms/step beyond the
  sampling time came from the skipped 311 MB/step logits D2H copy.
- All algorithms (0-4) and the block schedule verified working on the backend path.
- Timing-attribution caveat: at low step counts (e.g. 16) the one-time second `sched_reserve`
  (triggered by attaching the sampler after context creation) plus GPU-tail synchronization can
  land inside the "sampling time" bucket - judge performance at realistic step counts.
  A possible future tweak is passing the chain via `llama_context_params.samplers` at context
  creation to avoid the second reserve.

### Task 5 - inpainting (DONE)

- common/common.h: `common_params_diffusion.infill`; common/arg.cpp: `--diffusion-infill`
  (LLAMA_EXAMPLE_DIFFUSION only).
- diffusion-cli.cpp: infill skips the chat template (canvas used verbatim); requires
  `n_input <= n_ubatch` (clear error otherwise); `max_length = n_input` (canvas IS the whole
  sequence); the visual callback and the final print cover the whole canvas (no prompt strip;
  output vector resized to n_generated to drop the unused n_ubatch tail).
- diffusion.cpp: input guard allows `max_length == n_input` in infill mode; infill requires the
  timestep schedule and at least one mask token in the canvas (LOG_ERR + return, not assert).
- The denoising loop needed NO other changes - the mask-position scan is position-agnostic, and
  both CPU and backend sampling paths handle interior masks (shift_logits row mapping included).
- Verified end-to-end on Dream-7B (backend AND CPU paths): a canvas with a masked Elixir
  function body regenerates ONLY the masked span; all fixed tokens byte-identical in the output.
  Usage: write the model's mask piece (Dream: `<|mask|>`) once per token to regenerate, e.g.
  `llama-diffusion-cli --diffusion-infill -p 'def f(x), do: <|mask|><|mask|><|mask|>' ...`

### Task 6 - benchmarks and validation (DONE)

All matrices: Dream-7B Q4_K_M, -ngl 99, -ub 512, alg 4, temp 0.2, top_k 40, same prompt,
runs interleaved back-to-back on the RTX 5070 Laptop.

CANONICAL matrix - AC power, idle machine (third run; the definitive reference):

| steps     | backend total/step | cpu total/step | backend sampling/step | cpu sampling/step | speedup |
|-----------|--------------------|----------------|-----------------------|-------------------|---------|
| 64        | 248.01 ms          | 258.37 ms      | 2.06 ms               | 24.95 ms          | 1.04x   |
| 128       | 245.85 ms          | 263.10 ms      | 2.15 ms               | 29.64 ms          | 1.07x   |
| 256       | 245.61 ms          | 272.51 ms      | 2.17 ms               | 41.52 ms          | 1.11x   |
| 128 rep 2 | 249.86 ms          | 275.98 ms      | 2.21 ms               | 44.11 ms          | 1.10x   |

Backend run-to-run spread: 245.6-249.9 ms/step (~1%). CPU sampling drifts UP through the
session (24.95 -> 44.11 ms/step) as the package warms - even on AC the CPU stage is the
unstable one. On AC the forward pass (~243 ms) dominates; future speedups live there
(KV-cache reuse / Fast-dLLM), not in sampling.

FIRST matrix (later discovered to be contaminated by a user background workload - kept only
to show the failure mode):

| steps | backend total/step | cpu total/step | backend sampling/step | cpu sampling/step |
|-------|--------------------|----------------|-----------------------|-------------------|
| 64    | 245.75 ms          | 254.81 ms      | 1.83 ms               | 23.74 ms          |
| 128   | 285.19 ms          | 548.90 ms      | 2.00 ms               | 223.90 ms         |
| 256   | 336.84 ms          | 368.88 ms      | 2.49 ms               | 40.98 ms          |

SECOND matrix - unknowingly run ON BATTERY (verified:
/sys/class/power_supply/A*/online = 0, CPU parked at 1.5 GHz, GPU power-limited to ~11 W),
which turned out to be the most interesting data point:

| steps | backend total/step | cpu total/step | backend sampling/step | cpu sampling/step |
|-------|--------------------|----------------|-----------------------|-------------------|
| 64    | 346.27 ms          | 1370.44 ms     | 2.57 ms               | 1024.51 ms        |
| 128   | 433.74 ms          | 2291.06 ms     | 17.64 ms              | 1923.81 ms        |
| 256   | 431.80 ms          | (stopped)      | 20.08 ms              | (stopped)         |

On battery the CPU sampling stage collapses to 1-2 SECONDS per step while backend sampling
stays at 2-20 ms -> **5.3x end-to-end at 128 steps** (433.74 vs 2291.06 ms/step). The lower
the power envelope, the bigger the backend-sampling win.

IMPORTANT honest correction to the morning baseline (section 10): CPU-path sampling time is
WILDLY machine-state dependent on this laptop - measured 23.74 / 40.98 / 223.90 / 322.93 /
1024.51 / 1923.81 ms/step across identical workloads. The dominant factors, in order:
AC vs battery power state, concurrent background load, and CPU thermal throttling (the
322.93 baseline ran right after a 20-minute CUDA build had heated the machine; the first
matrix ran alongside a user background workload). The backend path stays at 1.8-20 ms/step
through ALL of these conditions. Conclusions:
- The sampling stage is effectively ELIMINATED (>10x reduction in its best CPU case, ~100x in
  its worst), and - just as valuable - its variance is eliminated: end-to-end speedup observed
  ranged from 4-11% (AC power, idle machine - the canonical matrix) through 2.55x (thermally
  throttled CPU) to 5.3x (on battery - arguably the most representative state for a laptop
  that runs sustained verify-and-remask loops).
- The remaining per-step cost is the forward pass; its growth across the matrix rows
  (244 -> 283 -> 334 ms) is GPU boost-clock decay under sustained load, not a sampling effect.
- Judge any future optimization with interleaved A/B runs; single runs on this machine are not
  comparable across sessions.

DiffuCoder-7B-cpGRPO q4_k_m (the Elixir target model):
- GGUF: `Mungert/DiffuCoder-7B-cpGRPO-GGUF`, file `DiffuCoder-7B-cpGRPO-q4_k_m.gguf` (4.8 GB).
  NOTE: the repo uses LOWERCASE quant names - `--include "*Q4_K_M*"` matches nothing and
  hf download silently downloads zero files.
- Runs end-to-end on the backend path (`dream` arch): 423.60 ms/step, sampling 4.81 ms/step at
  128 steps / 512 tokens. Mask token is the same `<|mask|>` = 151666 (Qwen2.5 vocab + Dream's
  mask token) - the infill marker is identical for Dream and DiffuCoder.
- Quality note: it produced compiling Elixir but ignored a "without Enum.reverse" constraint -
  the deterministic verify loop remains necessary, as the dLLMs.md report predicted.

Final regression gate: 14/14 backend-sampler tests pass on --device cpu AND --device gpu.

Overall diff: 9 files changed, 649 insertions(+), 255 deletions(-) across
src/llama-graph.cpp, src/llama-context.cpp, src/llama-sampler.cpp, common/{common.h,arg.cpp},
examples/diffusion/{diffusion.h,diffusion.cpp,diffusion-cli.cpp}, tests/test-backend-sampler.cpp.

## 12. Status and follow-ups

ALL TASKS COMPLETE (2026-06-10). The working tree contains the full implementation,
uncommitted. Quick usage:

```
# GPU sampling is the default; --no-backend-sampling forces the CPU path
./build/bin/llama-diffusion-cli -m ~/models/dream7b/Dream-org_Dream-v0-Instruct-7B-Q4_K_M.gguf \
  -p "..." -ub 512 --diffusion-eps 0.001 --diffusion-algorithm 4 --diffusion-steps 128 \
  --temp 0.2 --top-k 40 -ngl 99

# inpainting: write <|mask|> per token to regenerate; everything else stays fixed
./build/bin/llama-diffusion-cli --diffusion-infill -p 'def f(x), do: <|mask|><|mask|><|mask|>' ...
```

Deferred follow-ups (from Task 7 of the original plan):
- Attach the sampler via `llama_context_params.samplers` at context creation to avoid the
  second `sched_reserve` (~1s one-time, only matters at very low step counts).
- In-graph CFG blend (CFG currently falls back to CPU sampling - by design).
- Example-side custom confidence sampler to cut readback from k floats/row to 1 (public
  `llama_sampler_init` makes this possible without core changes).
- CUDA segmented top-k (the CUB path launches one DeviceTopK per row).
- For upstreaming: per AGENTS.md the contributor must rewrite/own the code; split as
  PR1 = core multi-row backend sampling + tests, PR2 = diffusion-cli wiring + infill.
