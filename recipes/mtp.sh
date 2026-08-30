#!/bin/bash
# MTP recipe: Qwen's own MTP head (3 drafts, probabilistic draft sampling),
# tensor-parallel 2, prefix caching, int8 per-token-head KV cache, vision
# enabled.
#
# The flags are the argument set the upstream launcher produced for
#   SPEC=mtp CTX=long PREFIX_CACHE=1 EXTRA_ARGS="--tensor-parallel-size 2"
# with two deliberate deviations:
#   1. the KV cache is int8 per-token-head on the Triton backend instead of
#      fp8 on FlashInfer (same per-token width, ~2x the pool of bf16), and
#   2. VLLM_SPEC_DECODE_ATTN=1 is exported. Upstream only enabled the split-KV
#      verify kernel for bf16-KV and dflash2; here patches/spec-decode-int8-kv.patch
#      teaches it to read the int8 cache, which is what makes the verify step
#      (every decode step of a speculating request) fast enough to matter.
#      Upstream never measured MTP with it -- validate acceptance on your
#      workload before trusting it.
#
# The exported env vars support the patch stack (split-KV verify attention,
# vision-tower offload) and the flashinfer/torch-allocator interaction; the
# vllm line below is the complete server configuration.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$DIR")"
MODEL=${MODEL:-$REPO/models/Qwen3.8-27B-W4A16-AutoRound-fast}
PORT=${PORT:-8080}
export PATH="$REPO/.venv/bin:$PATH"

[ -f "$MODEL/config.json" ] || { echo "no model at $MODEL -- run: python prepare/build_fast_model.py <dir> (or: docker run ... prepare)" >&2; exit 1; }
if [ ! -x "$REPO/.venv/bin/vllm" ] && ! command -v vllm >/dev/null; then
  echo "no vllm found -- create the uv venv first (README: Bare metal), or run this in the container" >&2; exit 1
fi

# a dead engine leaves its OffloadingConnector region in /dev/shm and the next
# boot dies in shared_offload_region.py; with a restart policy that loops (#33)
if [ "${VLLM_OFFLOAD_KEEP_SHM:-0}" != 1 ]; then
  for f in /dev/shm/vllm_offload_*.mmap; do
    [ -e "$f" ] || continue
    grep -lqs "$f" /proc/[0-9]*/maps 2>/dev/null || { echo "[mtp] removing stale offload region $f"; rm -f "$f"; }
  done
fi

# split-KV verify attention reading the int8 cache (see the header)
export VLLM_SPEC_DECODE_ATTN=1
# off by default: the vision tower (~0.88 GiB, sharded across both GPUs)
# stays GPU-resident; VLLM_VISION_CPU_OFFLOAD_GB=1 moves it to pinned host
# RAM instead -- zero resident VRAM, bit-exact output, ~+12% on vision
# forwards (measured on a PCIe 4.0 x16 3090)
# (patches/vision-tower-cpu-offload.patch)
export VLLM_VISION_CPU_OFFLOAD_GB=${VLLM_VISION_CPU_OFFLOAD_GB:-0}
# flashinfer's sampler needs a current nvcc to JIT; the torch sampler is fine
export VLLM_USE_FLASHINFER_SAMPLER=0
# DeltaNet's transient workspace fragments the allocator; expandable segments
# are what keep the boot from OOMing
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# no --language-model-only: the server takes image input; where the tower
# lives is VLLM_VISION_CPU_OFFLOAD_GB above.
# --prefix-caching-hash-algo xxhash: 128-bit xxHash instead of the default
# sha256 for prefix-cache block hashes (faster; needs the xxhash package).
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
  --max-num-batched-tokens 2048 \
  --enable-prefix-caching \
  --prefix-caching-hash-algo xxhash \
  --mamba-cache-mode align \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}' \
  --compilation-config '{"max_cudagraph_capture_size":32,"custom_ops":["+rms_norm","+silu_and_mul"]}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
