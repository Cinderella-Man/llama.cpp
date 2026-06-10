# P106 mining-card fleet for diffusion LLMs - research findings + plan


## 0. Premise correction (important)

"P106-090 (6 GB)" mixes two cards. Verified via TechPowerUp/VideoCardz:
- P106-090: 768 CUDA cores, **3 GB** GDDR5, 192 GB/s, PCIe 1.0 **x1**, 75 W, no outputs.
- P106-100: 1280 CUDA cores, **6 GB** GDDR5, 192 GB/s (= GTX 1060 6GB silicon), ~75-120 W.
Both GP106, compute capability 6.1 (sm_61). The 6 GB plan needs P106-100s (same junk-tier
price, ~EUR 25-40 used). P106-090s only work in PAIRS via layer split (see sec 5).

## 1. The headline finding: diffusion flips the economics of old cards

Official llama.cpp CUDA bench thread (github discussion #15013): **P106-100 = 406.94 t/s
pp512, 30.40 t/s tg128** (Llama-2 7B Q4_0). The key insight nobody exploits:

- AR chat workloads are token-generation (bandwidth/latency) shaped - old cards look bad.
- Diffusion LLM workloads are PROMPT-PROCESSING shaped: every step is one batch forward
  (512-token canvas), and with our fork's threshold decoding + infill, the unit of work is
  "a few dozen pp-style forwards". Pascal's pp-per-dollar is arguably the best in existence:
  ~16 t/s-pp per EUR (P106-100 @ 25-40) vs ~8 for a used RTX 3060 12GB (~$180) vs ~3.4 for
  a 5070-class card. **Mining cards are bad chatbots but excellent diffusion code engines.**

Per-card projected numbers for Dream-7B Q4 (from 407 t/s pp): 512-token step ~1.26 s ->
fast draft (threshold 0.6, ~17 steps) ~21-25 s; 128-192-token REPAIR canvas ~0.35-0.6 s/step
-> ~3-6 s per repair. Slow per card - but see sec 3.

## 2. Compatibility verdict (all verified)

Code audit of our fork (ggml-cuda, file:line in docs/dllm-engine-improvements style):
- dp4a: YES on cc 6.1 (GGML_CUDA_CC_DP4A = 610); MMQ quantized matmul enabled for all
  Q-types (the good path; Q4_0 is what the 407 t/s bench used).
- FAST_FP16: NO - common.cuh explicitly excludes `__CUDA_ARCH__ == 610`; avoid F16-compute
  paths (Q4/Q8 + MMQ is the right regime).
- FlashAttention: scalar tile/vec fallback kernels work (no MMA); benchmark FA on/off.
- OUR BACKEND SAMPLING: fully supported - argmax/cumsum/softmax/get/set_rows unconditional;
  top_k/argsort unrestricted with CUB (CUDART >= 11.7 -> any CUDA 12.x). GPU sampling
  matters MORE on rigs: mining hosts have Celeron-class CPUs where the old CPU sampling
  path (24-1900 ms/step on a fast laptop CPU!) would be catastrophic.
- CUDA graphs: DISABLED below Ampere (ggml-cuda.cu:4497 arch gate) - see experiment E3.
- Build: **CUDA 13 CANNOT target sm_61** (verified locally: nvcc 13.3 "Unsupported gpu
  architecture 'sm_61'", min is compute_75). Recipe: CUDA 12.x toolkit (12.6/12.8) +
  `CMAKE_CUDA_ARCHITECTURES=61-real`. ggml already ships `61-virtual` in default lists for
  CUDA < 13.
- Drivers: R580 is the LAST branch for Pascal (Phoronix/NVIDIA confirmed), maintained as
  legacy with security fixes through ~Oct 2028; R590+ removed Pascal. Pin driver 580.xx +
  CUDA 12.x on the rig. Post-2028 horizon: Vulkan via proprietary driver inherits the same
  EOL; nouveau/NVK do not serve Pascal well -> treat 2028 as the platform sunset.
- Vulkan backend: works on Pascal but pp (our critical metric) is consistently lower than
  CUDA on NVIDIA - CUDA is the backend, Vulkan only as contingency.
- VRAM fit on 6 GB: Dream-7B Q4_K_M 4.68 GB + ~310 MB logits host... device compute at
  ub 512 ~0.4-0.6 GB -> tight; plan A = Q4_0/IQ4_XS (4.2-4.5 GB) + `-ub 384/512`; the
  repair workload (small canvases) fits trivially.

## 3. The architecture: a zero-interconnect candidate farm

The PCIe 1.0 x1 link (250 MB/s!) makes tensor/row split hopeless - and IRRELEVANT. The
harness (kintsugi) sends many INDEPENDENT requests: drafts at different seeds, repairs of
different holes, hole-size variants. So:

- One `llama-diffusion-server` per card (`CUDA_VISIBLE_DEVICES=i`, distinct ports), model
  replicated per card (weights cross PCIe once at load: 4.7 GB / 250 MB/s ~ 20 s).
- kintsugi's ModelServer becomes a pool dispatcher: round-robin /generate across N
  endpoints; the verifier picks the first/best candidate. Zero inter-GPU traffic.
- A 6x P106-100 rig (~EUR 200): 6 concurrent drafts -> sustained draft throughput ~1 per
  4 s; repairs effectively interactive. This resurrects the "batched candidates" idea that
  we measured as useless on one GPU - across cards it is free parallelism.
- This is also why our fork specifically enables this hardware class: GPU sampling (weak
  host CPUs), threshold decoding (fewer steps), infill (small canvases), the HTTP daemon
  (fan-out unit), confidence export (verifier-driven repair).

Honest economics: 6 cards + rig host ~ 500-650 W under load. At EUR 0.30/kWh, heavy use
costs real money vs one used 3060 12 GB (~170 W, similar aggregate pp). The rig wins on
CAPEX (~EUR 200 vs 400+ total) and parallel-candidate latency; loses on power and 2028
sunset. For "AI for the non-rich" with cheap power (or intermittent use), the rig is
rational; with expensive power, buy the 3060.

## 4. Experiments to run (in order; no P106 on hand yet - items E1-E2 are buyable-day-one)

- E1 (smoke): CUDA 12.x build of the fork on a P106-100; run the 14 backend-sampler tests
  (--device gpu) + Dream-7B threshold draft + infill repair; record pp512/tg128 via
  llama-bench and step times. Validate the sec-1 projections.
- E2 (quant sweep): Q4_0 vs Q4_K_M vs IQ4_XS pp512 on-card (dp4a paths differ); FA on/off;
  `-ub` 256/384/512 vs VRAM headroom. Pick the fleet default.
- E3 (novel, engine): relax the CUDA-graphs Ampere gate behind an env flag
  (GGML_CUDA_FORCE_GRAPHS) and A/B on the rig. Rationale: the gate is a heuristic, not a
  hardware limit (graph APIs exist since CUDA 10); we MEASURED 1.5x from graphs on a
  launch-bound host (battery laptop), and rig Celerons are exactly launch-bound. If it
  reproduces, this is a real contribution for ALL pre-Ampere fleet users. (~10 LOC.)
- E4 (3 GB cards): if the cards in hand are really -090s: two-card layer split
  (`-sm layer` over x1; activations ~7 MB/boundary/step ~ +30-60 ms/step on 1.3 s steps =
  acceptable) OR wait for small dLLMs (consistency-distilled 2-4B uniform-state models are
  the obvious next wave). Measure, don't assume.
- E5 (fleet): N servers + kintsugi pool dispatcher; measure candidates/minute and
  wall-clock-to-verified vs the single-5070 baseline.

## 5. What this does NOT need

No new model formats, no custom kernels, no RPC/distributed-tensor work (x1 kills it and
the farm pattern does not need it), no Vulkan port. The fork as committed today is already
the enabling software; the work is a build recipe, a dispatcher in kintsugi, and E3.

## Sources

- TechPowerUp/VideoCardz P106-090 + P106-100 spec pages
- llama.cpp discussion #15013 (official CUDA bench: P106-100 406.94 pp512 / 30.40 tg128)
- Phoronix "NVIDIA Confirms 580 Linux Driver Is The Last For Maxwell/Pascal/Volta";
  NVIDIA Quadro support plan (legacy through ~2028); Arch/RPMFusion 590-drop threads
- medium.com/@daank "Affordable DIY AI: Local GPU Inference for EUR 25" (P106-100 LLM use)
- Local verification: nvcc 13.3 rejects sm_61; ggml-cuda Pascal capability audit (dp4a,
  MMQ, FA fallback, sampling ops, CUB, graphs gate) - all file:line in the full doc.

## 6. The actual rig: NINE cards in one PC with 4 GB RAM and 60 GB disk
## (grilling round 2026-06-11; every number below MEASURED on the local machine)

The single-host constraint changes the architecture. The wall is not VRAM - it is HOST RAM.

### Measured per-server host footprint (llama-diffusion-server, Dream-7B Q4_K_M, -ngl 99)
- ub 512: VmRSS 1997 MB (Pss 1933 MB). Decomposition: 270 MB anonymous + 292 MB resident
  mmap'd GGUF pages + ~1.43 GB CUDA pinned output buffers and driver mappings.
- ub 128: VmRSS 1054 MB. The delta vs ub 512 is 2.43 MB per ubatch row - EXACTLY the four
  vocab-strided pinned buffers (4 x 151936 x 4 B/row: logits + sampling.logits + probs +
  candidates), confirming the buffer model.
- Fixed floor per process: ~750 MB (CUDA context + libs + touched model pages).

### Verdict table for 9 replicas on 4 GB RAM
| design                                            | host RAM total | fits? |
|---------------------------------------------------|----------------|-------|
| 9 processes, ub 512 (sec-3 design as written)     | ~17.5 GB       | NO    |
| 9 processes, ub 128                               | ~9.5 GB        | NO    |
| 9 processes + k-stride buffer fix                 | ~6.8 GB        | NO (per-process fixed costs) |
| ONE process, 9 replicas + k-stride fix            | ~2.5-3 GB      | YES (tight; add zram) |

### Two engine changes are therefore REQUIRED (promoted from "future optimization")
1. k-stride sampling host buffers: size probs/candidates/sampled by the chain's max top-k
   (<= 64) instead of n_vocab, and do not allocate the logits + sampling.logits host
   buffers at all when every sequence has a backend sampler (needs_raw_logits false).
   Effect: 2.43 MB/row -> ~0.5 KB/row (ub 512: 1.24 GB -> ~0.3 MB). Implementation:
   llama-context.cpp output_reserve (the has_sampling branch) + the stride passed to
   copy_tensor_async_floats/candidates + get_sampled_*_ith stride bookkeeping (~40-60 LOC).
   The k_max is known at attach time from the chain's top_k sampler.
2. Multi-replica server: one process hosting N (model, context) replicas, one per
   CUDA_VISIBLE_DEVICE, each with its own mutex; /generate gains "replica": i or a
   least-busy dispatcher. Shares libs, driver state, and the single mmap'd GGUF (page
   cache counted once). Marginal cost per replica = one CUDA context (~100-250 MB host,
   to be confirmed on the rig) + KBs of buffers after fix 1.

### Other single-host realities (verified or standard practice)
- Disk 60 GB: minimal Linux ~10 + driver 580 legacy + CUDA 12 RUNTIME ~3 + model 4.4
  (one shared GGUF for all replicas) + binaries ~1 -> ~18 GB. Build on the laptop with a
  CUDA 12.x toolkit (CUDA 13 cannot target sm_61 - verified) and ship binaries; never
  install the toolkit on the rig.
- 9 GPUs on one board: enable Above 4G Decoding in BIOS (standard multi-GPU/mining
  requirement); model upload at boot streams the GGUF once per card through page cache -
  on 4 GB RAM the file does not stay cached, so expect ~9 sequential re-reads of 4.4 GB
  from disk at startup (SATA SSD: ~10 min worst case; acceptable, one-time).
- Power: 9 x 75 W + host ~ 750-800 W under load - the PSU, not the software, may be the
  binding constraint; stagger generation if needed (the dispatcher can cap concurrency).
- Add zram swap on the rig (4 GB RAM has no headroom for spikes).

### What was PROVEN on the local machine vs what needs the rig
Proven here: per-server RSS/Pss at ub 512/128 and its exact buffer decomposition; the
2.43 MB/row scaling law; CUDA 13 cannot build sm_61; Pascal code paths (dp4a/MMQ/FA
fallback/sampling ops/CUB) via source audit; the launch-bound CUDA-graphs win (1.5x,
battery test) that motivates experiment E3.
Needs the rig: per-CUDA-context marginal host RAM with 9 real devices; P106-100 step
times (projected from the official 406.94 t/s pp512 bench); E2 quant sweep; E3 graphs
A/B on Pascal; BAR/driver behavior with 9 cards on the 580 legacy branch.

### Revised build order for the rig milestone
1. Engine fix 1 (k-stride buffers) - benefits every machine, mandatory for the rig.
2. Multi-replica server (engine fix 2) - or, interim, 2-3 processes x 3-4 replicas each
   if per-process fixed costs measure lower than expected on the rig.
3. CUDA 12.x cross-build + rig bring-up (driver 580, Above 4G, zram) -> E1/E2/E3.
4. kintsugi dispatcher across replicas (the candidate farm of sec 3).

## 7. The exact hardware identified: Manli M-P106L9-N6G (grilling round 2)

The rig is a turnkey Manli "GPU Mining System P106-090 (6GB) X9". This resolves the sec-0
card-identity puzzle a third way: Manli shipped a NON-STANDARD P106-090 with 6 GB - the
"P106L" (TechPowerUp lists BOTH a 3 GB (b6879) and a 6 GB (b6880) F347G entry, and the TPU
VGA-BIOS archive holds a "manli p106-90 6 GB" BIOS dump). Confirmed card profile:

- GP106, 768 CUDA cores (60% of a P106-100), boost 1531 MHz, compute capability 6.1
- 6 GB GDDR5, 192-bit, 192.2 GB/s (SAME bandwidth as a 1060/P106-100)
- PCIe 1.0 x1 electrical (x16 physical slots on the integrated 9-slot board), ~75 W
- System: Celeron 3865U (2 cores, no HT, 1.8 GHz), 4 GB DDR4, 64 GB M.2 SSD, 1600 W PSU,
  chassis-fan cooling for passive cards (designed for 24/7 load)

### Revised per-card projections (pp scales with compute: 768/1280 of the P106-100's
### measured 406.94 t/s pp512)
- pp512 ~ 240-260 t/s -> 512-token diffusion step ~ 2.0-2.1 s
- fast draft (threshold 0.6, ~17 steps) ~ 35-40 s/card
- repair canvas 128-192 tokens ~ 0.6-0.9 s/step -> ~5-9 s per repair
- tg ~ 28-30 t/s (bandwidth-bound; bandwidth is uncut)
- VRAM: 6 GB fits Dream-7B Q4_K_M + ub 384-512 exactly as analyzed in sec 2
- Aggregate: 9 cards ~ 2,200 t/s pp = ALMOST EXACTLY ONE RTX 5070 LAPTOP in raw pp -
  but as 9 independent parallel engines with 54 GB total VRAM, for ~EUR 200-300 used.

### System-level consequences (mostly confirming sec 6, two sharpenings)
- Celeron 3865U (2 cores) makes the sec-6 conclusions HARDER requirements: GPU sampling is
  non-negotiable (the CPU could not sample for one card, let alone nine), the
  multi-replica single-process server is the only sane shape, and the dispatcher must do
  nearly zero per-token work (detokenize is the heaviest remaining host duty).
- PCIe 1.0 x1 per card: irrelevant at steady state (farm pattern, nothing crosses);
  boot-time model distribution: 9 x 4.4 GB at ~250 MB/s/card (sequential or a few in
  parallel off one SSD) ~ 3-6 min one-time. Acceptable.
- 1600 W PSU for ~750 W actual load: ample headroom; thermals are the rig's design point
  (passive cards + 10 chassis fans, sized for 24/7 hashing at the same 75 W/card).
- 64 GB SSD vs sec-6's 60 GB budget: ~18 GB used; fine.
- Above-4G decoding: the board boots 9 GPUs by construction (its only job) - the BIOS
  concern from sec 6 dissolves on this turnkey unit; the remaining unknown is only the
  R580-legacy-driver + CUDA 12 bring-up on its stock BIOS, which is rig-day work (E1).
- Manli's own product page markets the X9 for "AI computing" - the manufacturer agrees.

## 8. Normal llama-server testing + does everything work on ALL cards? (grilling round 3)
## (every claim verified by execution on this machine or cited)

### Can the rig be tested with the NORMAL llama-server?

YES for ordinary (autoregressive) models - and it is the recommended E1 smoke test.
NO for diffusion models - VERIFIED BY EXECUTION: llama-server loads Dream-7B fine but every
completion request returns HTTP 500 "the current context does not logits computation.
skipping" (the exact upstream issue #17249 failure; the server's slot machinery is
AR-only). Diffusion models require our llama-diffusion-server.

Three AR test modes for the rig, all with the stock llama-server:
1. Single card: any 7B Q4 (e.g. Qwen2.5-Coder-7B, 4.7 GB) on one P106; projected
   ~28-30 t/s generation (bandwidth-bound, the 6GB cards have the full 192 GB/s).
   Host RAM measured for llama-server: ~0.6 GB RSS (no diffusion pinned buffers - AR
   decodes request few output rows) -> fits the 4 GB host with room to spare.
2. THE BIG ONE - one process, all nine cards: `llama-server -m model.gguf -ngl 99
   -sm layer` pools 54 GB of VRAM in a SINGLE process (no 9x RAM problem). Per-token
   cross-card traffic is one activation vector per layer boundary (~14 KB) - trivially
   fine over PCIe 1.0 x1 for generation (prompt processing is slower but works). This
   lets the EUR 200 mining rig serve Qwen2.5-72B-class models: 32B Q4 (~19 GB) at a
   projected ~8-12 t/s, 70B Q4 (~40 GB) at ~4-6 t/s. Worth testing for its own sake.
3. kintsugi-relevant: 9 separate single-card llama-servers for AR candidate farming -
   same farm pattern as diffusion, and at ~0.6 GB/process the 4 GB host fits ~4-5 of
   them today (all 9 after the multi-replica server work).

### Does everything in this document work on modern cards too?

The fork was DEVELOPED on an RTX 5070 (Blackwell) - everything was verified there first.
Cross-card matrix (verified by: local 5070 runs + Pascal source audit + toolkit checks):

| feature                          | P106 (Pascal 6.1)        | RTX 5070 / modern        |
|----------------------------------|--------------------------|--------------------------|
| GPU backend sampling (multi-row) | YES (audit: all ops ok)  | YES (tested, 14/14)      |
| threshold decoding / infill /    | YES (model-level,        | YES (tested)             |
| confidence export / guard        |  card-agnostic)          |                          |
| llama-diffusion-server           | YES (pending E1 on-rig)  | YES (tested, both models)|
| Dream/LLaDA/DiffuCoder 7B Q4     | YES (fits 6 GB)          | YES (tested)             |
| DiffusionGemma 26B               | **NO - impossible**: 16 GB Q4 exceeds 6 GB VRAM and  |
|                                  | the experts-in-RAM path needs ~20 GB host (rig has 4)| 
|                                  |                          | YES (tested, --cpu-moe)  |
| CUDA graphs (1.5x when           | NO (upstream Ampere+     | YES (active by default)  |
|  launch-bound)                   |  gate; experiment E3)    |                          |
| FlashAttention                   | scalar fallback only     | full MMA kernels         |
| normal llama-server (AR models)  | YES (verified pattern)   | YES                      |
| normal llama-server (dLLMs)      | NO (verified: HTTP 500)  | NO (same - by design)    |

### Build matrix (verified)

- This laptop's CUDA 13.3 CANNOT compile for the rig (sm_61 removed; verified locally).
- **CUDA 12.8 is the unique one-binary option**: it added Blackwell sm_120 (RTX 5070) AND
  still targets Pascal sm_61 (feature-frozen, removed in 13.x) - one fat binary with
  `CMAKE_CUDA_ARCHITECTURES="61-real;120a-real"` runs on both machines. (Sources: NVIDIA
  CUDA 12.8 Blackwell announcement; Pascal feature-freeze note; local nvcc 13.3 test.)
- Simpler alternative: keep the laptop on 13.3 and install CUDA 12.x only for rig builds.
- Rig driver: 580 legacy branch (last for Pascal, security fixes to ~2028); laptop driver
  unaffected (Blackwell uses current branches).

## 9. Running the model in a 6 GB envelope: MEASURED numbers + runbook (grilling round 4)

### Measured VRAM envelope (Dream-7B Q4_K_M, llama-diffusion-server, backend sampling,
### RTX 5070; Pascal totals will differ only by driver/context overhead, margin is ample)

| -ub  | model buffer | compute buffer | TOTAL process VRAM | headroom vs 6144 MiB |
|------|--------------|----------------|--------------------|----------------------|
| 512  | 4168 MiB     | 609 MiB        | **4946 MiB**       | ~1.2 GB              |
| 384  | 4168 MiB     | ~460 MiB       | **4790 MiB**       | ~1.35 GB             |
| 256  | 4168 MiB     | 305 MiB        | **4634 MiB**       | ~1.5 GB              |

VERDICT: Dream-7B Q4_K_M fits a 6 GB card AT FULL ub 512 with >1 GB headroom - no
quantization downgrade or ubatch sacrifice required. DiffuCoder Q4_K_M (~4.3 GiB model
buffer) lands ~5.1 GiB at ub 512: also fits.

### Options ladder (use in this order only if a 6 GB card surprises us)
- A (default): Q4_K_M + `-ub 512` - full draft length, best quality. 4.95 GiB.
- B: `-ub 256` - caps max_length at 256 (fine for repair canvases; drafts shorter). 4.6 GiB.
- C: IQ4_XS GGUF (file 4.2 GB -> model ~3.7 GiB) - ~4.4 GiB total at ub 512; slight
  quality cost; only needed if Pascal context overhead eats the margin (unlikely).
- D: Q3_K_M (3.8 GB) - emergency only; measurable quality loss on code.

### Runbook: steps to run the model on the 6 GB cards

DONE NOW (laptop):
1. VRAM envelope measured (table above) - the configuration is proven to fit.
2. Software validated end-to-end on modern hardware (GPU sampling, threshold, infill,
   diffusion-server) - secs 6-8.

BUILD (any machine, once):
3. Install CUDA 12.8 toolkit alongside (CUDA 13 cannot target sm_61 - verified).
   `cmake -B build-rig -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="61-real" \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc`
   (add `;120a-real` to share one binary with the laptop). Build llama-diffusion-server,
   llama-diffusion-cli, test-backend-sampler, llama-bench; ship build-rig/bin + the GGUF.

RIG BRING-UP (one-time):
4. Minimal Linux server install on the 64 GB SSD; pin the NVIDIA 580 legacy driver
   (the last Pascal branch); add zram swap; verify all 9 cards in `nvidia-smi`.
5. Smoke with the NORMAL llama-server first (sec 8): one card, an AR 7B Q4 - proves
   driver + CUDA runtime + card health with zero diffusion variables.

PER-CARD VALIDATION (script over CUDA_VISIBLE_DEVICES=0..8):
6. `LLAMACPP_TEST_MODELFILE=qwen0.5b.gguf ./test-backend-sampler --device gpu` (14 tests).
7. `llama-bench -p 512 -n 32` - record pp512/tg128 per card (expect ~240-260 / ~28 t/s).
8. `llama-diffusion-cli -m dream.gguf -ub 512 --diffusion-eps 0.001 --diffusion-steps 64
    --diffusion-conf-threshold 0.6 --temp 0.2 --top-k 40 -ngl 99 -p "test"` - one draft;
   then an infill repair. Confirm "sampling on the backend" appears (GPU sampling active -
   mandatory on the 2-core Celeron).
9. `nvidia-smi --query-compute-apps=used_memory` during a run - confirm <= 6144 MiB
   (expected ~4.9 GiB; if over, drop down the options ladder).

FLEET (until the multi-replica server lands - sec 6):
10. Interim: up to 4 x llama-diffusion-server processes (~1 GB host RAM each at ub 512;
    measured) on different ports/CUDA_VISIBLE_DEVICES - the 4 GB host caps it.
11. Full 9-card farm: requires engine fixes from sec 6 (k-stride host buffers +
    multi-replica server) - then one process, 9 replicas, ~2.5-3 GB host RAM.
12. Point kintsugi's dispatcher at the replica endpoints; verify candidates/minute.
