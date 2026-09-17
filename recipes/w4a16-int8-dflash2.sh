#!/bin/bash
# w4a16-int8-dflash2 (the default): DFlash2 drafter (7 drafts in one pass),
# TP=2, prefix caching, int8 per-token-head KV, vision enabled.
#
# Flags are what the upstream launcher produced for
#   SPEC=dflash2 CTX=long PREFIX_CACHE=1 EXTRA_ARGS="--tensor-parallel-size 2"
# (DFLASH_TOKENS=7 is its default), minus the single-card KV_MEM pin -- under
# TP>1 the launcher drops it and sizes the KV pool from gpu-memory-utilization.
# Two deviations from the launcher values:
#   1. --max-num-seqs 8 (upstream's CTX=long ships 4 -- it keeps 4 slots when
#      the block is long, for the graphs); max_cudagraph_capture_size goes
#      32 -> 64 with it, the launcher's own formula at 8 slots (8 x the
#      8-token verify block) -- at 32 an 8-request decode batch would run
#      piecewise, ~8% slower (upstream's CTX=fast note). The
#      VLLM_V2_CUDAGRAPH_MEM_MIB reservation stays 1400: upstream sizes it
#      identically at k<=7 for 4 and 8 slots (1900 only above 7 drafts).
#   2. --max-num-batched-tokens 8192 (upstream ships 2048). Its batch lane
#      measured 2048 beating 8192: bigger chunks inflate the profiled
#      activation peak, which shrinks the KV/state page pool. We accept the
#      smaller pool for half the prefill steps (16k prompt: two steps,
#      not four at 4096).
#
# The env vars support the patch stack; the vllm line is the complete
# server configuration.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$DIR")"

# .env at the repo root: fills any variable below that is unset or empty in
# the real environment (which always wins); values may be quoted, whole-line
# # comments only.
if [ -f "$REPO/.env" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line#"${line%%[![:space:]]*}"}"
    case "$line" in ''|\#*) continue ;; esac
    k="${line%%=*}"; v="${line#*=}"
    [ "$k" != "$line" ] || continue
    k="${k%"${k##*[![:space:]]}"}"
    case "$k" in ''|*[!A-Za-z0-9_]*|[0-9]*) continue ;; esac
    # never let a .env flip a shell-control variable (GLOBIGNORE would
    # silently disable the /dev/shm cleanup glob below, among others)
    case "$k" in IFS|GLOBIGNORE|CDPATH|BASH_ENV|ENV|SHELLOPTS|PS1|LINENO|PWD|OLDPWD|SECONDS|RANDOM|UID|EUID) continue ;; esac
    v="${v#"${v%%[![:space:]]*}"}"
    v="${v%"${v##*[![:space:]]}"}"
    v="${v%$'\r'}"
    v="${v#\"}"; v="${v%\"}"
    if [ "${#v}" -ge 2 ] && [ "${v:0:1}" = "${v: -1}" ]; then
      [ "${v:0:1}" = "'" ] && v="${v:1:${#v}-2}"
    fi
    [ -n "$v" ] || continue
    [ -n "${!k:-}" ] || export "$k=$v"
  done < "$REPO/.env"
fi

VENV=${VENV:-$REPO/.venv}
# Swift variant: a truthy SWIFT (1/true/yes, or /path, as in setup/prepare) serves
# the Swift W4A16 model -- SWIFT_MODEL points at it (default: the build
# destination) and wins over MODEL. Never point MODEL at the Swift dir:
# setup.py builds the fast model into $MODEL.
case "${SWIFT:-}" in
  1|true|yes) MODEL=${SWIFT_MODEL:-$REPO/models/Qwen3.8-27B-Swift-W4A16} ;;
  0|false|no|'') MODEL=${MODEL:-$REPO/models/Qwen3.8-27B-W4A16-AutoRound-fast} ;;
  *) MODEL=${SWIFT_MODEL:-$SWIFT} ;;
esac
DRAFT=${DRAFT:-$REPO/models/Qwen3.8-27B-DFlash2-W4A16}
PORT=${PORT:-8080}
export PATH="$VENV/bin:$PATH"

[ -f "$MODEL/config.json" ] || { echo "no model at $MODEL -- build it first (prepare/build_fast_model.py; the Swift variant: prepare/build_swift_model.py; or: docker run ... prepare)" >&2; exit 1; }
[ -f "$DRAFT/config.json" ] || { echo "no drafter at $DRAFT -- run: python prepare/fetch_dflash2.py <dir>" >&2; exit 1; }
if [ ! -x "$VENV/bin/vllm" ] && ! command -v vllm >/dev/null; then
  echo "no vllm found -- create the uv venv first (README: Bare metal), or run this in the container" >&2; exit 1
fi

# a dead engine leaves its /dev/shm offload region and the next boot dies on it (upstream #33)
if [ "${VLLM_OFFLOAD_KEEP_SHM:-0}" != 1 ]; then
  for f in /dev/shm/vllm_offload_*.mmap; do
    [ -e "$f" ] || continue
    grep -lqs "$f" /proc/[0-9]*/maps 2>/dev/null || { echo "[w4a16-int8-dflash2] removing stale offload region $f"; rm -f "$f"; }
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
# hold the mamba "align" state snapshots a multi-turn prefix hit resumes
# from alive until the request ends (upstream #52 / vllm#45238;
# patches/mamba-align-checkpoint-order.patch). That patch ships default
# off -- we default it on (deviation); retention is bounded (<=3 blocks per
# request per group). Set VLLM_MAMBA_ALIGN_KEEP_CHECKPOINTS=0 to opt out.
export VLLM_MAMBA_ALIGN_KEEP_CHECKPOINTS=${VLLM_MAMBA_ALIGN_KEEP_CHECKPOINTS:-1}
# keep DeltaNet's transient workspace from fragmenting the allocator (boot OOM)
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# no --language-model-only: the server takes image input.
# Vision: image count is unlimited; each image is capped at 2097152 px
# = 2048 tokens, and that cap sets the encoder's profiled peak in the
# KV pool (at most the 4096-token encoder budget). xxhash: faster
# prefix-cache hashes than sha256.
# draft_sample_method is required on 0.28.0: the native DFlash2 inherits the
# upstream speculator base, which allocates the draft-logits buffer only when
# the config asks; without it the rejection test loses its denominator and
# acceptance drops ~16% (upstream #73).
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
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --prefix-caching-hash-algo xxhash \
  --mamba-cache-mode align \
  --mm-processor-kwargs '{"size":{"shortest_edge":65536,"longest_edge":2097152}}' \
  --speculative-config '{"method":"dflash","model":"'"$DRAFT"'","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}' \
  --compilation-config '{"max_cudagraph_capture_size":64,"custom_ops":["+rms_norm","+silu_and_mul"]}' \
  --reasoning-parser qwen3 \
  --enable-prompt-tokens-details \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  ${EXTRA_ARGS:-}
