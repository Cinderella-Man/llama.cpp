#!/usr/bin/env bash
# Probe 0 re-verification: full 48-case kintsugi bench on Dream-7B Q4_K_M,
# baseline profile, GPU backend-sampling ON (default), KINTSUGI_TRACE captured.
# Reproduces: 35/48, 6.46 tok/s deliverable, 421 engine calls, ~126 s engine wall.
# Sandbox note: server is a child of THIS foreground command and is killed before
# return (background servers get SIGUSR1-reaped by the sandbox).
set -uo pipefail
ROOT=/home/car/projects/llama.cpp
DREAM=/home/car/models/dream7b/Dream-org_Dream-v0-Instruct-7B-Q4_K_M.gguf
OUT=$ROOT/docs/dllms/throughput-plans/07_layer_f/probe0-baseline-bench
TRACE=$OUT/ktrace-reverify.jsonl

"$ROOT/build/bin/llama-diffusion-server" -m "$DREAM" \
  -ub 512 -ngl 99 --diffusion-eps 0.001 --diffusion-steps 128 \
  --temp 0.2 --top-k 40 --host 127.0.0.1 --port 8080 \
  > "$OUT/server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null' EXIT
for i in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
echo "server up; starting bench at $(date -Iseconds)"

cd "$ROOT/kintsugi"
KINTSUGI_TRACE="$TRACE" mix run bench/bench.exs http://127.0.0.1:8080 flayer-reverify baseline 2>&1
echo "bench done at $(date -Iseconds)"
echo "trace lines: $(wc -l < "$TRACE")"
