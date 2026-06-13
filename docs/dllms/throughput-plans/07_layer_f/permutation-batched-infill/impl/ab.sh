#!/usr/bin/env bash
# 07_layer_f batched-infill A/B: baseline (sequential sweep) vs batchsweep (batched),
# same server build (n_seq_max=4 + /infill_batch). One server serves both profiles.
set -uo pipefail
ROOT=/home/car/projects/llama.cpp
DREAM=/home/car/models/dream7b/Dream-org_Dream-v0-Instruct-7B-Q4_K_M.gguf
OUT=$ROOT/docs/dllms/throughput-plans/07_layer_f/permutation-batched-infill/impl

"$ROOT/build/bin/llama-diffusion-server" -m "$DREAM" -ub 512 -ngl 99 \
  --diffusion-eps 0.001 --diffusion-steps 128 --temp 0.2 --top-k 40 \
  --host 127.0.0.1 --port 8080 > "$OUT/server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null' EXIT
for i in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
echo "server up ($(curl -s http://127.0.0.1:8080/health | python3 -c 'import sys,json;print("n_ctx",json.load(sys.stdin).get("n_ctx"))'))"

cd "$ROOT/kintsugi"
echo "===== BASELINE (sequential sweep) ====="
mix run bench/bench.exs http://127.0.0.1:8080 ab-baseline baseline 2>&1 | grep -E "TOTAL|tier|^p |^m |^c |^h |^a |^i "
echo "===== BATCHSWEEP (batched variant sweep) ====="
mix run bench/bench.exs http://127.0.0.1:8080 ab-batchsweep batchsweep 2>&1 | grep -E "TOTAL|tier|^p |^m |^c |^h |^a |^i "
