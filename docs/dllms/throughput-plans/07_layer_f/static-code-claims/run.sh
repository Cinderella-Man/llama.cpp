#!/usr/bin/env bash
# Re-runs every static-code-claim check in this dir. No GPU. From repo root.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
echo "HEAD: $(git rev-parse --short HEAD)"

echo "=== S1 output_all=true @1531, n_outputs=n_tokens @1566 ==="
sed -n '1531p;1566p' src/llama-context.cpp
grep -n "bool output_all" src/llama-batch.h

echo "=== S2 needs_raw_logits D2H skip @1594-1595 ==="
sed -n '1594,1595p' src/llama-context.cpp

echo "=== S3 steps_done plumbing ==="
grep -n "out_steps_done" examples/diffusion/diffusion.h
grep -n 'res\["steps_done"\]' examples/diffusion/diffusion-server.cpp

echo "=== S4 KINTSUGI_TRACE ==="
grep -n "KINTSUGI_TRACE" kintsugi/lib/kintsugi/engine.ex

echo "=== S5 per-request sampler attach/detach ==="
grep -n "llama_set_sampler(ctx, 0," examples/diffusion/diffusion.cpp

echo "=== S6 batch-probe async warning ==="
grep -n "llama_decode is async" examples/diffusion/diffusion-batch-probe.cpp
