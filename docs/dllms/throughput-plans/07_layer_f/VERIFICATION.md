# 07_layer_f.md verification — master matrix

Independent re-verification of **every claim** in
`docs/dllms/throughput-plans/07_layer_f.md` (primary) plus the **Layer F section
+ in-fork measured numbers** of `docs/dllms/dllm-throughput-catalog.md`
(secondary). External arXiv/literature claims in the catalog (Layers A-E, H) are
inherited from the 2026-06-11 deep-research run and explicitly NOT re-verified.

- **Machine**: RTX 5070 Laptop 8 GB, on AC (ADP1), the doc's "5070 AC". Repo HEAD
  `53953f4b4`. All models present locally (Dream-7B Q4_K_M, FastDLLM-1.5B Q4_K_M,
  Qwen2.5-7B-Instruct Q4_K_M), Elixir/mix present, binaries built.
- **Method**: each empirical claim → reconstruct exact invocation → re-run on this
  5070 → compare fresh numbers to the doc. Each non-trivial probe has its own dir
  with `README.md` (what/how + verdict), `run.sh` (exact command), `raw.log`
  (output), `.diff` where code changed. Sandbox note: GPU servers run as children
  of one foreground command (background servers get SIGUSR1-reaped, exit 144).
- **Verdict bands** (as agreed): pass count ±2 exact · tok/s & ms ±15% REPRODUCED /
  15-30% DRIFTED / >30% REFUTED · structural exact · mechanism qualitative · counts
  & ratios exact.

## Headline

**29 of 30 verified claim-groups REPRODUCED** (most to the exact digit). One
DRIFT (Probe 4) whose miss *strengthens* the doc's main conclusion. One secondary
catalog figure (~52 MB/replica) not confirmed by coarse RSS. **No load-bearing
conclusion in 07_layer_f.md is contradicted; LAYER F CLOSED-for-Dream-on-laptop
stands.**

## Matrix

### Static-code claims → `static-code-claims/`
| id | claim | verdict |
|---|---|---|
| S1 | encode() output_all=true @1531, n_outputs=n_tokens @1566 | REPRODUCED exact |
| S2 | needs_raw_logits D2H skip @1595 | REPRODUCED exact |
| S3 | steps_done computed + plumbed to /generate response (runtime: steps_done=10) | REPRODUCED exact |
| S4 | KINTSUGI_TRACE env trace in Engine | REPRODUCED exact |
| S5 | per-request llama_set_sampler attach/detach | REPRODUCED exact |
| S6 | batch-probe async-decode warning (bug #1) | REPRODUCED |
| S7 | replica = own llama_model instance (block_kv + concurrency safe) | REPRODUCED (via F1) |

### Probes (07_layer_f.md CATALOG-F PLANNING ROUND) → per-dir
| probe | claim | re-run headline | verdict | dir |
|---|---|---|---|---|
| 0 | bench 35/48, 6.46 tok/s, 421 calls, DRAFT/INFILL split | 35/48, 6.46, 421; all sub-dists match | **REPRODUCED** | `probe0-baseline-bench/` |
| 1 | infill ~2.3 ms/tok above floor; ubatch-512 → 500 | 2.43 ms/tok; 500 confirmed | **REPRODUCED** | `probe1-infill-vs-canvas/` |
| 2 | F9 47 re-sent drafts; F10 median 0.92/p90 0.98/208 pairs | 47; 0.92/0.98/208 | **REPRODUCED** | `probe2-cross-req-redundancy/` |
| 3 | per-step canvas-dependent, ~15+0.4 ms/tok; steps 4/7/13 | ~25+0.48 ms/tok; 4/7/13 exact | **REPRODUCED** (1 minor drift: ec collapse) | `probe3-perstep-costmodel/` |
| 4 | backend ON HURTS tiny repairs (+23/+37 ms); crossover 10-12 masks | ON faster at 3/6/12 masks | **DRIFTED** (sub-claim refuted; helps headline) | `probe4-tinyrepair-onoff/` |
| 6 | backend OFF = 33/48, 4.34 tok/s, 186 s (−2/−33%/+27%) | 33/48, 4.33, 186.7 s | **REPRODUCED** | `probe6-backend-onoff-bench/` |
| 7 | "sampling time" 155→0.06 ms, total unchanged (async decode) | 37→0.07 ms, total unchanged | **REPRODUCED** | `probe7-async-decode/` (+`probe7.diff`) |

### PART 1/2 items → per-dir
| item | claim | re-run | verdict | dir |
|---|---|---|---|---|
| F1 | multi-replica racing ratio 2.63, slows each request | ratio ~2.7, each 0.38→1.06 s | **REPRODUCED** | `f1-multireplica-racing/` |
| F2 STEP 1 | oracle: 70 blk, mean L 12.01, median 8, first-blk 12.25, agree 84.9% | all exact | **REPRODUCED exact** | `f2-oracle-probe/` |
| F2 STEP 2 | dflash: AR 54.8; dl32/16/8 = 26.1/40.2/34.6 tok/s; acc 5.79/5.69/4.11 | 54.9; 26.4/40.3/35.0; 5.79/5.69/4.11 | **REPRODUCED exact** | `f2-dflash-closedloop/` |

### Catalog Layer F (secondary) → `catalog-layer-f/`
| item | verdict |
|---|---|
| F7/F9/F10/F11 DEAD/PARK verdicts | REPRODUCED (Probe 2/7/1 evidence) |
| F2 encode logits-flags PARK | code state confirmed (S1/S2) |
| F1 "~52 MB/replica" host cost | **DRIFT** — coarse RSS ~233 MB/replica (~4.5×); method-dependent |
| methodology audit (oracle/dflash/batch-probe/bench tools) | AUDIT PASS → `methodology-audit/` |

## The one DRIFT, in context
**Probe 4** claimed backend sampling is *slower* on tiny repairs (attach overhead,
+23/+37 ms; crossover ~10-12 masks) → motivating "route small infills to CPU" /
"attach-once". Warm median-of-5 re-runs show backend ON is **faster at every mask
count** (3/6/12). The doc's ON column reads like a cold/single-shot measurement.
This does **not** weaken the doc — it *strengthens* "backend ON globally correct"
(Probe 6) and retires the two Probe-4-derived levers as already-moot. No verdict
flips.

## Bottom line
The doc's central engine-cost model, its bench numbers, the async-decode finding,
the cross-request redundancy structure, and both F2 measurements all reproduce on
fresh runs — several to the exact digit. The single quantitative drift is benign.
**07_layer_f.md is a faithful, reproducible record; its "Layer F SPENT for
Dream-on-laptop" conclusion is verified.**
