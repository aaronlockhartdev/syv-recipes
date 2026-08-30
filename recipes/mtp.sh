#!/bin/bash
# MTP recipe: Qwen's own MTP head (3 drafts, probabilistic draft sampling),
# tensor-parallel 2, prefix caching, int8 per-token-head KV cache.
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

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$DIR")"
MODEL=${MODEL:-$REPO/models/Qwen3.8-27B-W4A16-AutoRound-fast}
PORT=${PORT:-18020}
export PATH="$REPO/venv/bin:$PATH"

[ -f "$MODEL/config.json" ] || { echo "no model at $MODEL -- run: python prepare/build_fast_model.py <dir> (or: docker run ... prepare)" >&2; exit 1; }
if [ ! -x "$REPO/venv/bin/vllm" ] && ! command -v vllm >/dev/null; then
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
# flashinfer's sampler needs a current nvcc to JIT; the torch sampler is fine
export VLLM_USE_FLASHINFER_SAMPLER=0
# DeltaNet's transient workspace fragments the allocator; expandable segments
# are what keep the boot from OOMing (off on WSL2, where the paravirt driver
# rejects the VMM calls)
if grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null || [ -n "${WSL_DISTRO_NAME:-}" ]; then
  export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}
else
  export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
fi

if [ -z "${VLLM_API_KEY:-}" ] && [ -f "$REPO/api_key.txt" ]; then
  export VLLM_API_KEY="$(cat "$REPO/api_key.txt")"
fi

exec vllm serve "$MODEL" \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port $PORT \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.93 \
  --max-model-len 150000 \
  --max-num-seqs 8 \
  --api-server-count 1 \
  --language-model-only \
  --attention-backend TRITON_ATTN \
  --kv-cache-dtype int8_per_token_head \
  --mamba-ssm-cache-dtype float16 \
  --async-scheduling \
  --max-num-batched-tokens 2048 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}' \
  --compilation-config '{"max_cudagraph_capture_size":32,"custom_ops":["+rms_norm","+silu_and_mul"]}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
