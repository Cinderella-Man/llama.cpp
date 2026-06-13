# F1 — multi-replica seed racing on one GPU (re-verification)

**Doc claim** (07_layer_f.md:330-354): `--diffusion-replicas 3` on the 5070,
FastDLLM kv+bs (E4 config), stack-task drafts at seeds {3,103,203}:
- sequential 3 drafts: 1.38 s (0.43/0.39/0.56 s each)
- concurrent 3 drafts: 1.21 s wall, but EACH slows to 1.07-1.21 s
- ratio vs one draft 2.63 (< 3 bar, but KILLS the use case — first-success
  racing slows the majority case).
Also (S7 / E-rig caveat): 3 concurrent block_kv requests across 3 replicas run
correctly — each replica is its own `llama_model` instance / own pkv store.

## How verified
`run.sh`: FastDLLM-1.5B Q4_K_M server with `--diffusion-block-kv
--diffusion-replicas 3` (backend sampling default ON), conf_threshold 0.9. The
c_stack task at seeds {3,103,203}, n_gen 192. Single baseline (3 serial), then
3 serial (sequential wall), then 3 in parallel via background curls (concurrent
wall + per-request `time_total`). Replica count read from server log. → `raw.log`.

## Result (re-run 2026-06-13 10:14, 5070 on AC)

| metric | doc | re-run | verdict |
|---|---|---|---|
| replicas ready | 3 | **3** | exact (S7 confirmed) |
| single drafts | 0.43/0.39/0.56 s | 0.38/0.38/0.55 s | REPRODUCED |
| sequential wall | 1.38 s | **1.36 s** | REPRODUCED |
| concurrent wall | 1.21 s | **1.20 s** | REPRODUCED |
| each (concurrent) | 1.07-1.21 s | 1.06-1.18 s | REPRODUCED |
| ratio vs 1 draft | 2.63 | ~2.7 (1.20/0.44) | REPRODUCED (<3) |

- **S7 validated**: 3 replicas, all three concurrent block_kv requests returned
  correct results with no error — independent `llama_model` instances compose
  with block-KV, as the E-rig caveat-resolved note claims.
- The **use-case killer reproduces exactly**: concurrency slows *each* request
  from ~0.38 s to ~1.06 s while the wall only improves 1.36→1.20 s. The 5070 has
  no spare arithmetic at 32-row 1.5B shapes; racing on one GPU is net-pointless.

**VERDICT: REPRODUCED.** Ratio ~2.7 (< 3, passes the formal bar) but every
individual draft slows ~2.8×. The F1-CLOSED-for-single-GPU decision holds; cross-
card racing on the rig (where N cards make the ratio ≈ 1) remains the right
pattern, as harness work.
