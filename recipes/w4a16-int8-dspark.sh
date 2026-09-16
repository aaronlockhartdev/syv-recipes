#!/bin/bash
# w4a16-int8-dspark: the community DSpark drafter
# (RadixArk/Qwen3.8-27B-DSpark, bf16, 7 drafts per step) on the exact stack of
# w4a16-int8-dflash2 -- TP=2, prefix caching, int8 per-token-head KV, vision
# enabled. Only the drafter model and the speculative method differ.
#
# DSpark is native in vLLM 0.28.0 (speculative method "dspark"); serving this
# checkpoint needs two things upstream hit and documented (upstream #84 /
# issue #25 items 15-16), and both are handled for you:
#   1. patches/dspark-draft-quant-config.patch -- the loader asks for the
#      draft's quantization config through the target's hf_overrides and
#      refuses a bf16 draft that carries none of its own; the patch returns
#      None in that case.
#   2. the checkpoint's config.json must name the architecture
#      Qwen3DSparkModel, not the published DSparkDraftModel (the 0.28.0
#      registry maps the published name to the DeepSeek V4 class, which dies
#      on a DeepSeek-only field). prepare/fetch_dspark.py installs exactly
#      that copy: the rename in config.json, everything else and the
#      weights untouched (hard-linked).
#   No drafter at $DRAFT?  python prepare/fetch_dspark.py <dir>
#
# Memory: the ~3.7 GB bf16 drafter is charged twice -- as weights (half per
# card under TP=2) and as a larger per-request state than the 1.2 GB W4A16
# head. Upstream measured that a drafter this size does not fit their pinned
# 24 GB budgets at any context, but does fit unpinned at
# --gpu-memory-utilization 0.97 with a 8192-token max len on their 3090.
# This recipe keeps our standard 0.93 / native max len for shape
# consistency; if it fails to boot on 24 GB cards, that is why -- lower
# --max-model-len (8192 fits upstream) or raise --gpu-memory-utilization.
# This file is data: edit it.
#
# Performance: upstream measured DSpark LOSING to the shipped dflash2 head on
# the same requests -- 3.37 vs 3.83 tokens/step and 114 vs 150 tok/s on their
# 3090, 3.19 vs 3.77 and 115 vs 160 on their 4090 (their bf16-KV lane,
# single card; upstream: "documented so nobody re-derives the two errors, not
# as a recommendation"). This shape (TP=2, int8 KV, native max len) is
# unmeasured -- that is the point of this recipe. Compare acceptance
# (tokens/step, the README's Notes on how to read it) and
# perplexity/GSM8K against w4a16-int8-dflash2 before trusting it.
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
MODEL=${MODEL:-$REPO/models/Qwen3.8-27B-W4A16-AutoRound-fast}
DRAFT=${DRAFT:-$REPO/models/Qwen3.8-27B-DSpark}
PORT=${PORT:-8080}
export PATH="$VENV/bin:$PATH"

[ -f "$MODEL/config.json" ] || { echo "no model at $MODEL -- run: python prepare/build_fast_model.py <dir> (or: docker run ... prepare)" >&2; exit 1; }
[ -f "$DRAFT/config.json" ] || { echo "no drafter at $DRAFT -- run: python prepare/fetch_dspark.py <dir> (or: docker run ... prepare)" >&2; exit 1; }
if [ ! -x "$VENV/bin/vllm" ] && ! command -v vllm >/dev/null; then
  echo "no vllm found -- create the uv venv first (README: Bare metal), or run this in the container" >&2; exit 1
fi

# a dead engine leaves its /dev/shm offload region and the next boot dies on it (upstream #33)
if [ "${VLLM_OFFLOAD_KEEP_SHM:-0}" != 1 ]; then
  for f in /dev/shm/vllm_offload_*.mmap; do
    [ -e "$f" ] || continue
    grep -lqs "$f" /proc/[0-9]*/maps 2>/dev/null || { echo "[w4a16-int8-dspark] removing stale offload region $f"; rm -f "$f"; }
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
# draft_sample_method is required on 0.28.0 (upstream #73): without it the
# rejection test loses its denominator and acceptance drops ~16%.
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
  --speculative-config '{"method":"dspark","model":"'"$DRAFT"'","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}' \
  --compilation-config '{"max_cudagraph_capture_size":64,"custom_ops":["+rms_norm","+silu_and_mul"]}' \
  --reasoning-parser qwen3 \
  --enable-prompt-tokens-details \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  ${EXTRA_ARGS:-}
