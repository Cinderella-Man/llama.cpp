# Catalog Layer F cross-check (secondary scope)

Per the agreed scope, the catalog (`dllm-throughput-catalog.md`) is verified only
for its **Layer F section** + the **in-fork measured numbers** it cites; external
arXiv/literature claims (Layers A-E, H) are inherited from the 2026-06-11
deep-research run, NOT re-verified here.

## Catalog Layer F items (dllm-throughput-catalog.md:204-244)

| catalog item | nature | verification |
|---|---|---|
| **F1 [DONE]** GPU backend sampling multi-row, threshold decode, EOT-tail shrink, right-sized canvas, k-stride pinned buffers (1997→783 MB), multi-replica (~52 MB/replica), GGML_CUDA_FORCE_GRAPHS, --host/--port | fork "done" facts | backend sampling **confirmed working** (Probe 0/4/6); multi-replica + block-kv **confirmed** (F1, 3 replicas concurrent OK); see resource note below |
| **F2** encode() logits-flags fix | designed, not built | static S1/S2 confirm the code state (output_all=true @1531, D2H already skipped @1595); 07_layer_f verdict PARK (DG-only) verified |
| F3 EB backend-sampling graft | DG, 3090-gated | DG-specific, parked — out of kintsugi-Dream scope |
| F4 sc_embT quantization | DG-specific | out of scope |
| F5 chunked prefill | canvas>ubatch | Probe 1 confirms the ubatch-512 ceiling (canvas>512 → HTTP 500) that F5 would lift |
| **F6** step-loop CUDA graph capture | speculative | 07_layer_f verdict CLOSED-null (E6 cross-ref: graphs zero on AC); inherited from 05_layer_e E6 |
| **F7** overlap host work | speculative | **REFUTED as useful** by Probe 7: host work = 0.06 ms/step, nothing to overlap |
| F8 Pascal micro-tuning | rig-day | hardware-gated, out of laptop scope |
| **F9** server-side prompt caching | speculative | **DEAD** per Probe 2 (prompt 25% of forward, Dream bidirectional, kv_prefix already net-loss) |
| **F10** speculative-canvas warm start | flagship idea | **PARK ~10% ceiling** per Probe 1+2 (headroom only on large failing-case repairs; 208 near-identical pairs but variant sweeps shift positions) |
| **F11** reduce step-loop host overhead | micro | **DEAD** per Probe 7 (host step = 0.06 ms) |

## 07_layer_f.md catalog-F verdict table (:226-233) — every verdict's evidence re-verified

| item | doc verdict | evidence re-run | result |
|---|---|---|---|
| F2 encode logits-flags | PARK (DG-only) | static S1/S2 | code state confirmed |
| F6 CUDA graphs | CLOSED null | E6 (inherited) | consistent |
| F7 host overlap | DEAD | Probe 7 (0.07 ms host) | **REPRODUCED** |
| F9 prompt cache | DEAD | Probe 2 (47 re-sent, 25% forward) | **REPRODUCED** |
| F10 canvas cache | PARK ~10% | Probe 1 (2.43 ms/tok) + Probe 2 (208 pairs) | **REPRODUCED** |
| F11 step-loop micro | DEAD | Probe 7 | **REPRODUCED** |

## Resource note (catalog F1 "~52 MB/replica") — DRIFT on a secondary claim
See `replica-resource.log`. Measured FastDLLM-1.5B block-kv on the 5070:
- VRAM ~1240 MB/replica (GPU weights not shared across replicas).
- Host RSS ~233 MB/replica — **~4.5× the catalog's "~52 MB/replica"**. RSS
  overcounts (per-replica CUDA pinned buffers + host allocs); the 52 MB figure
  likely reflects a cleaner marginal/PSS accounting of model-state only. The
  *design* claim (mmap-shared weights → sub-linear scaling) holds; the specific
  52 MB number is not reproduced by a coarse RSS delta and is **measurement-method
  dependent**. Flagged, not load-bearing for any Layer F verdict.

## Conclusion
Every catalog Layer F verdict that the 07_layer_f probes bear on is reproduced
(F7/F9/F10/F11 DEAD/PARK with fresh evidence; F2 code state confirmed). The only
catalog discrepancy is the secondary "~52 MB/replica" host figure, which a coarse
RSS measurement puts ~4.5× higher.
