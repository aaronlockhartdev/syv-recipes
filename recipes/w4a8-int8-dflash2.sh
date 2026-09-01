#!/bin/bash
# w4a8-int8-dflash2: the dflash2 stack with W4A8 Marlin linears.
#
# w4a16-int8-dflash2's configuration plus VLLM_MARLIN_INPUT_DTYPE=int8,
# limited to the layers INT8_LAYERS selects (patches/marlin-int8-layer-
# select.patch; its default exclude keeps the int8-weight lm_head and the
# MTP module on W4A16, where W8A8 is unsupported). Prefill is compute-
# bound at every concurrency, so the int8 tensor-core GEMMs buy prefill
# speed; decode is memory-bound and unchanged. Upstream measured this on
# the dflash2 stack (seeded, prefix caching; their INT8 lane is bf16 KV):
#
#   prefill tok/s            1k      4k      16k     51k
#   W4A16 (baseline)       1,437   1,494   1,410   1,200
#   mlp (the default)      1,638   1,696   1,587   1,320    (+13-14%)
#   all*                   1,845   1,937   1,791   1,423    (+27-30%)
#
#   *all = mlp|linear_attn|self_attn, expanded below. The layer matcher is
#   a regex over the layer name, so the word "all" itself would match
#   nothing -- this recipe maps it to the explicit list upstream means.
#
# Quality is the documented int8 trade: the default mlp set is the gentler
# variant (+2.2% PPL, IFBench flat); all is GSM8K 95.0% (baseline 96.5)
# and PPL +4.1%, mostly prose, code flat. The GDN-only middle
# (mlp|linear_attn) crashes at first forward on this torch/vLLM combo --
# an inductor codegen bug; use mlp or all.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$DIR")"
MODEL=${MODEL:-$REPO/models/Qwen3.8-27B-W4A16-AutoRound-fast}
DRAFT=${DRAFT:-$REPO/models/Qwen3.8-27B-DFlash2-W4A16}
PORT=${PORT:-8080}
INT8_LAYERS=${INT8_LAYERS:-mlp}
# expand upstream's "all" shorthand to the explicit list (see header)
[ "$INT8_LAYERS" = all ] && INT8_LAYERS='mlp|linear_attn|self_attn'
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
    grep -lqs "$f" /proc/[0-9]*/maps 2>/dev/null || { echo "[w4a8-int8-dflash2] removing stale offload region $f"; rm -f "$f"; }
  done
fi

# split-KV verify attention reading the int8 KV cache
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
# W4A8 activations for the selected layers (see header)
export VLLM_MARLIN_INPUT_DTYPE=int8
export VLLM_MARLIN_INT8_INCLUDE_RE=$INT8_LAYERS
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
