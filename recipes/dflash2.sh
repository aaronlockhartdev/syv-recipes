#!/bin/bash
# DFlash2 recipe: block drafter (7 drafts in one pass), TP=2, prefix caching,
# int8 per-token-head KV, vision enabled.
#
# Flags are exactly what the upstream launcher produced for
#   SPEC=dflash2 CTX=long PREFIX_CACHE=1 EXTRA_ARGS="--tensor-parallel-size 2"
# (DFLASH_TOKENS=7 is its default), minus the single-card KV_MEM pin -- under
# TP>1 the launcher drops it and sizes the KV pool from gpu-memory-utilization.
#
# The env vars support the patch stack; the vllm line is the complete
# server configuration.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$DIR")"
MODEL=${MODEL:-$REPO/models/Qwen3.8-27B-W4A16-AutoRound-fast}
DRAFT=${DRAFT:-$REPO/models/Qwen3.8-27B-DFlash2-W4A16}
PORT=${PORT:-8080}
export PATH="$REPO/.venv/bin:$PATH"

[ -f "$MODEL/config.json" ] || { echo "no model at $MODEL -- run: python prepare/build_fast_model.py <dir> (or: docker run ... prepare)" >&2; exit 1; }
[ -f "$DRAFT/config.json" ] || { echo "no drafter at $DRAFT -- run: python prepare/fetch_dflash2.py <dir>" >&2; exit 1; }
if [ ! -x "$REPO/.venv/bin/vllm" ] && ! command -v vllm >/dev/null; then
  echo "no vllm found -- create the uv venv first (README: Bare metal), or run this in the container" >&2; exit 1
fi

# a dead engine leaves its /dev/shm offload region and the next boot dies on it (upstream #33)
if [ "${VLLM_OFFLOAD_KEEP_SHM:-0}" != 1 ]; then
  for f in /dev/shm/vllm_offload_*.mmap; do
    [ -e "$f" ] || continue
    grep -lqs "$f" /proc/[0-9]*/maps 2>/dev/null || { echo "[dflash2] removing stale offload region $f"; rm -f "$f"; }
  done
fi

# split-KV verify attention reading the int8 cache
# (patches/spec-decode-attn.patch + spec-decode-int8-kv.patch); QMAX = the 8-token verify block
export VLLM_SPEC_DECODE_ATTN=1
export VLLM_SPEC_DECODE_ATTN_QMAX=8
# the V2 runner (forced by dflash) doesn't count its ~1.4 GiB of CUDA graphs
# against gpu-memory-utilization, so they're reserved here
# (patches/hybrid-kv-groups-v2-cudagraph.patch)
export VLLM_V2_CUDAGRAPH_MEM_MIB=1400
# vision tower in pinned host RAM by default (upstream default; patches/vision-tower-cpu-offload.patch):
# off the VRAM budget, bit-exact, ~+12% per image forward; =0 keeps it GPU-resident
export VLLM_VISION_CPU_OFFLOAD_GB=${VLLM_VISION_CPU_OFFLOAD_GB:-1}
# torch sampler (flashinfer's needs nvcc to JIT)
export VLLM_USE_FLASHINFER_SAMPLER=0
# keep DeltaNet's transient workspace from fragmenting the allocator (boot OOM)
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# no --language-model-only: the server takes image input.
# Vision: up to 16 images per request, each capped at 2097152 px = 2048
# tokens -- the cap (not the count) sets the encoder's profiled peak in
# the KV pool (at most the 4096-token encoder budget); the count only
# bounds per-request context. xxhash: faster prefix-cache hashes than sha256.
exec vllm serve "$MODEL" \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port $PORT \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.93 \
  --max-model-len auto \
  --max-num-seqs 4 \
  --api-server-count 1 \
  --attention-backend TRITON_ATTN \
  --kv-cache-dtype int8_per_token_head \
  --mamba-ssm-cache-dtype float16 \
  --async-scheduling \
  --max-num-batched-tokens 4096 \
  --enable-prefix-caching \
  --prefix-caching-hash-algo xxhash \
  --mamba-cache-mode align \
  --limit-mm-per-prompt '{"image":{"count":16}}' \
  --mm-processor-kwargs '{"size":{"shortest_edge":65536,"longest_edge":2097152}}' \
  --speculative-config '{"method":"dflash","model":"'"$DRAFT"'","num_speculative_tokens":7}' \
  --compilation-config '{"max_cudagraph_capture_size":32,"custom_ops":["+rms_norm","+silu_and_mul"]}' \
  --reasoning-parser qwen3 \
  --enable-prompt-tokens-details \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
