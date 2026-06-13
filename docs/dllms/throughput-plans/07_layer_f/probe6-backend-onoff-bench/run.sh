#!/usr/bin/env bash
# Probe 6 re-verification: full bench with backend sampling OFF (--no-backend-sampling).
# Backend-ON arm = Probe 0 (35/48, 6.46 tok/s, 147 s). Doc backend-OFF: 33/48,
# 4.34 tok/s, 186 s (-2 pass, -33% deliverable, +27% wall).
set -uo pipefail
ROOT=/home/car/projects/llama.cpp
DREAM=/home/car/models/dream7b/Dream-org_Dream-v0-Instruct-7B-Q4_K_M.gguf
OUT=$ROOT/docs/dllms/throughput-plans/07_layer_f/probe6-backend-onoff-bench

"$ROOT/build/bin/llama-diffusion-server" -m "$DREAM" \
  -ub 512 -ngl 99 --diffusion-eps 0.001 --diffusion-steps 128 \
  --temp 0.2 --top-k 40 --no-backend-sampling --host 127.0.0.1 --port 8080 \
  > "$OUT/server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null' EXIT
for i in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
echo "server up (--no-backend-sampling); bench at $(date -Iseconds)"
cd "$ROOT/kintsugi"
mix run bench/bench.exs http://127.0.0.1:8080 flayer-cpusample-reverify baseline 2>&1
echo "done $(date -Iseconds)"
