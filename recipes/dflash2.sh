#!/bin/bash
# DFlash2 recipe: block drafter (7 drafts in one pass), tensor-parallel 2,
# prefix caching, int8 per-token-head KV cache, vision enabled.
#
# The flags are the exact argument set the upstream launcher produced for
#   SPEC=dflash2 CTX=long PREFIX_CACHE=1 DFLASH_TOKENS=7 (default)
#   EXTRA_ARGS="--tensor-parallel-size 2"
# minus the single-card KV_MEM pin (the launcher drops it under TP>1 and
# sizes the KV pool from gpu-memory-utilization instead).
#
# The exported env vars support the patch stack (split-KV verify attention,
# V2-runner graph accounting, vision-tower offload) and the
# flashinfer/torch-allocator interaction; the vllm line below is the
# complete server configuration.

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

# a dead engine leaves its OffloadingConnector region in /dev/shm and the next
# boot dies in shared_offload_region.py; with a restart policy that loops (#33)
if [ "${VLLM_OFFLOAD_KEEP_SHM:-0}" != 1 ]; then
  for f in /dev/shm/vllm_offload_*.mmap; do
    [ -e "$f" ] || continue
    grep -lqs "$f" /proc/[0-9]*/maps 2>/dev/null || { echo "[dflash2] removing stale offload region $f"; rm -f "$f"; }
  done
fi

# split-KV verify attention (patches/spec-decode-attn.patch) reading the int8
# cache (patches/spec-decode-int8-kv.patch); QMAX = the 8-token verify block
export VLLM_SPEC_DECODE_ATTN=1
export VLLM_SPEC_DECODE_ATTN_QMAX=8
# the V2 runner (forced by dflash) does not count its ~1.4 GiB of CUDA graphs
# against gpu-memory-utilization; reserve them explicitly
# (patches/hybrid-kv-groups-v2-cudagraph.patch)
export VLLM_V2_CUDAGRAPH_MEM_MIB=1400
# on by default (the original's VISION_OFFLOAD=1): the vision tower
# (~0.88 GiB, sharded across both GPUs) stays in pinned host RAM and each
# module is copied to the GPU for its own forward -- zero resident VRAM,
# bit-exact output, ~+12% on vision forwards (measured on a PCIe 4.0 x16
# 3090); =0 keeps the tower GPU-resident
# (patches/vision-tower-cpu-offload.patch)
export VLLM_VISION_CPU_OFFLOAD_GB=${VLLM_VISION_CPU_OFFLOAD_GB:-1}
# flashinfer's sampler needs a current nvcc to JIT; the torch sampler is fine
export VLLM_USE_FLASHINFER_SAMPLER=0
# DeltaNet's transient workspace fragments the allocator; expandable segments
# are what keep the boot from OOMing
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# no --language-model-only: the server takes image input; where the tower
# lives is VLLM_VISION_CPU_OFFLOAD_GB above.
# The two multimodal flags are the original repo's vision arguments: at most
# one image per request, and a pixel cap below the processor default because
# vLLM profiles the encoder at the largest image it accepts and that peak
# comes out of the KV pool (2097152 px = 2048 image tokens).
# --prefix-caching-hash-algo xxhash: 128-bit xxHash instead of the default
# sha256 for prefix-cache block hashes (faster; needs the xxhash package).
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
  --limit-mm-per-prompt '{"image":{"count":1}}' \
  --mm-processor-kwargs '{"size":{"shortest_edge":65536,"longest_edge":2097152}}' \
  --speculative-config '{"method":"dflash","model":"'"$DRAFT"'","num_speculative_tokens":7}' \
  --compilation-config '{"max_cudagraph_capture_size":32,"custom_ops":["+rms_norm","+silu_and_mul"]}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
