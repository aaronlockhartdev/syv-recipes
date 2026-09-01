#!/bin/bash
# MTP recipe: Qwen's own MTP head (3 drafts, probabilistic), TP=2, prefix
# caching, int8 per-token-head KV, vision enabled.
#
# Flags are what the upstream launcher produced for
#   SPEC=mtp CTX=long PREFIX_CACHE=1 EXTRA_ARGS="--tensor-parallel-size 2"
# with three deviations:
#   1. int8 per-token-head KV on the Triton backend (upstream: fp8/FlashInfer)
#      -- same per-token width, ~2x the pool of bf16.
#   2. VLLM_SPEC_DECODE_ATTN=1. Upstream enabled the split-KV verify kernel
#      only for bf16-KV and dflash2; patches/spec-decode-int8-kv.patch
#      teaches it the int8 cache. Upstream never measured MTP with it, so
#      validate draft acceptance on your workload.
#   3. cudagraph_mode=PIECEWISE. The default (FULL_AND_PIECEWISE) has a
#      documented MTP corruption: with a prefix-cache hit, one prompt length
#      in 128 (here: length % 128 == 4) returns "" / "#" or fluent wrong
#      text. Upstream forced PIECEWISE for MTP for correctness; at the
#      served lengths it costs nothing measured.
#
# The env vars support the patch stack; the vllm line is the complete
# server configuration.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$DIR")"
MODEL=${MODEL:-$REPO/models/Qwen3.8-27B-W4A16-AutoRound-fast}
PORT=${PORT:-8080}
export PATH="$REPO/.venv/bin:$PATH"

[ -f "$MODEL/config.json" ] || { echo "no model at $MODEL -- run: python prepare/build_fast_model.py <dir> (or: docker run ... prepare)" >&2; exit 1; }
if [ ! -x "$REPO/.venv/bin/vllm" ] && ! command -v vllm >/dev/null; then
  echo "no vllm found -- create the uv venv first (README: Bare metal), or run this in the container" >&2; exit 1
fi

# a dead engine leaves its /dev/shm offload region and the next boot dies on it (upstream #33)
if [ "${VLLM_OFFLOAD_KEEP_SHM:-0}" != 1 ]; then
  for f in /dev/shm/vllm_offload_*.mmap; do
    [ -e "$f" ] || continue
    grep -lqs "$f" /proc/[0-9]*/maps 2>/dev/null || { echo "[mtp] removing stale offload region $f"; rm -f "$f"; }
  done
fi

# split-KV verify attention reading the int8 cache (see header)
export VLLM_SPEC_DECODE_ATTN=1
# vision tower in pinned host RAM by default (upstream default; patches/vision-tower-cpu-offload.patch):
# off the VRAM budget, bit-exact, ~+12% per image forward; =0 keeps it GPU-resident
export VLLM_VISION_CPU_OFFLOAD_GB=${VLLM_VISION_CPU_OFFLOAD_GB:-1}
# torch sampler (flashinfer's needs nvcc to JIT)
export VLLM_USE_FLASHINFER_SAMPLER=0
# keep DeltaNet's transient workspace from fragmenting the allocator (boot OOM)
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# no --language-model-only: the server takes image input.
# Vision: up to 8 images per request, each capped at 2097152 px = 2048
# tokens -- the cap (not the count) sets the encoder's profiled peak in
# the KV pool (at most the 4096-token encoder budget); the count only
# bounds per-request context. xxhash: faster prefix-cache hashes than sha256.
exec vllm serve "$MODEL" \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port $PORT \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.93 \
  --max-model-len auto \
  --max-num-seqs 8 \
  --api-server-count 1 \
  --attention-backend TRITON_ATTN \
  --kv-cache-dtype int8_per_token_head \
  --mamba-ssm-cache-dtype float16 \
  --async-scheduling \
  --max-num-batched-tokens 4096 \
  --enable-prefix-caching \
  --prefix-caching-hash-algo xxhash \
  --mamba-cache-mode align \
  --limit-mm-per-prompt '{"image":{"count":8}}' \
  --mm-processor-kwargs '{"size":{"shortest_edge":65536,"longest_edge":2097152}}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}' \
  --compilation-config '{"max_cudagraph_capture_size":32,"cudagraph_mode":"PIECEWISE","custom_ops":["+rms_norm","+silu_and_mul"]}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
