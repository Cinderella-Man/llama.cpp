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
