#!/usr/bin/env bash
# F1 re-verification: multi-replica seed racing on one 5070.
# Doc: --diffusion-replicas 3, FastDLLM kv+bs (E4 config), stack-task drafts at
# seeds {3,103,203}: sequential 1.38 s; concurrent 1.21 s wall but each 1.07-1.21 s;
# ratio vs one draft 2.63 (< 3 bar, but kills the use case).
# Also validates S7: 3 concurrent block_kv requests across 3 replicas run correctly
# (each replica = own llama_model instance / own pkv store).
set -uo pipefail
ROOT=/home/car/projects/llama.cpp
FDLLM=/home/car/models/fast-dllm-v2-1.5b/fast-dllm-v2-1.5b-Q4_K_M.gguf
OUT=$ROOT/docs/dllms/throughput-plans/07_layer_f/f1-multireplica-racing
USER=$(python3 -c "import json;print([json.loads(l)['user'] for l in open('$ROOT/docs/dllms/throughput-plans/07_layer_f/f2-dflash-closedloop/cases8.jsonl') if json.loads(l)['id']=='c_stack'][0])")

"$ROOT/build/bin/llama-diffusion-server" -m "$FDLLM" -ub 512 -ngl 99 \
  --diffusion-block-kv --diffusion-replicas 3 --diffusion-eps 0.001 \
  --diffusion-steps 128 --temp 0.2 --top-k 40 --host 127.0.0.1 --port 8080 \
  > "$OUT/server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null' EXIT
for i in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
echo "replicas in log:"; grep -c "replica.*ready" "$OUT/server.log"
curl -s http://127.0.0.1:8080/generate -d '{"prompt":"hi","n_gen":32,"steps":4,"conf_threshold":0.9,"seed":1}' >/dev/null  # warmup

req () { # $1 seed -> prints time_total seconds
  curl -s -o /dev/null -w '%{time_total}' http://127.0.0.1:8080/generate -H 'Content-Type: application/json' \
    -d "{\"prompt\":$(python3 -c "import json;print(json.dumps('''$USER'''))"),\"n_gen\":192,\"conf_threshold\":0.9,\"temp\":0.2,\"top_k\":40,\"seed\":$1}"
}

echo "=== single draft (seed 3) x3 for baseline ==="
for s in 3 103 203; do echo "seed $s: $(req $s)s"; done

echo "=== SEQUENTIAL 3 drafts ==="
t0=$(date +%s.%N)
for s in 3 103 203; do echo "  seed $s: $(req $s)s"; done
t1=$(date +%s.%N)
echo "sequential wall: $(python3 -c "print(f'{$t1-$t0:.2f}')")s"

echo "=== CONCURRENT 3 drafts ==="
t0=$(date +%s.%N)
req 3 > "$OUT/c3.txt" & p1=$!
req 103 > "$OUT/c103.txt" & p2=$!
req 203 > "$OUT/c203.txt" & p3=$!
wait $p1 $p2 $p3
t1=$(date +%s.%N)
echo "each: s3=$(cat $OUT/c3.txt)s s103=$(cat $OUT/c103.txt)s s203=$(cat $OUT/c203.txt)s"
echo "concurrent wall: $(python3 -c "print(f'{$t1-$t0:.2f}')")s"
rm -f "$OUT"/c3.txt "$OUT"/c103.txt "$OUT"/c203.txt
echo "done $(date -Iseconds)"
