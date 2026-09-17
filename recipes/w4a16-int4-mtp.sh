#!/bin/bash
# w4a16-int4-mtp: Qwen's own MTP head (3 drafts, probabilistic) on the int4
# per-token-head KV cache, TP=2, prefix caching, vision enabled.
#
# This combination is ours: upstream's int4 lane is dflash2 (PR #42) and
# their MTP lane is fp8/FlashInfer (single card) -- the int4 x MTP cell has
# never been measured upstream, so treat every number as unmeasured until
# you run it here. The int4 side inherits the int4-dflash2 story: halved
# cache bytes (~2x the int8 context capacity) at ~20% decode vs the bf16
# path, depth quality measured once upstream on a single 4090 (96.0% GSM8K,
# 100k needle at 90% depth).
#
# Three items are not optional here (all from the w4a16-int4-dflash2 story;
# upstream: docs/wsl2-4090.md):
#   --prefix-match-unit 808: under int4's halved-page geometry the
#     sliding-window prefix-cache guard returns a permanent clean miss
#     unless the hash unit EQUALS the drafter's sliding-window block size
#     (a finer divisor gets zero reuse from the guard's alignment clause),
#     and the hybrid coordinator's min turns that one group's miss into
#     zero reuse model-wide. The block size moves with the draft count:
#     848 at n=7 (the dflash2 sibling), 808 at n=3 (MTP's 3 drafts; hash
#     unit 1616). int8 never needs the flag (its geometry lands 864/864).
#   VLLM_INT4_MQ_3D=1: the 3D multi-query verify dispatch for the int4
#     cache (patches/spec-decode-int4-kv-mq3d.patch). MTP verifies 4
#     queries per step; without it they take the 2D path, which upstream
#     measured at ~8 tok/s at depth against ~25-29 with the 3D kernel.
#   The int4 boot-blocker trio: patches/int4-kv-per-token-head.patch,
#     patches/spec-decode-int4-kv-mq3d.patch and the spec-decode-scratch-*
#     pair (3D scratch sizing + budgeting). The split-KV verify kernel
#     (VLLM_SPEC_DECODE_ATTN) reads bf16/int8 caches only, so it is
#     deliberately unset here.
#
# Inherited from the MTP family (the w4a16-int8-mtp story):
#   cudagraph_mode=PIECEWISE. The default (FULL_AND_PIECEWISE) has a
#   documented MTP corruption: with a prefix-cache hit, one prompt length
#   in 128 returns "" / "#" or fluent wrong text. Upstream forced
#   PIECEWISE for MTP for correctness; at the served lengths it costs
#   nothing measured.
#   draft_sample_method=probabilistic: required on 0.28.0 (upstream #73);
#   without it the rejection test loses its denominator and acceptance
#   drops ~16%.
#
# Deviations from the int8-mtp sibling, item for item: the KV dtype, the
# --prefix-match-unit 808, VLLM_INT4_MQ_3D=1 for the 3D verify dispatch,
# and no VLLM_SPEC_DECODE_ATTN. --max-num-batched-tokens 8192,
# --max-num-seqs 8 and the 32 capture size are shared with the sibling
# (the 8192 is the same pool-for-steps trade the int4-dflash2 recipe
# documents); the 8192 peak trims this recipe's pool a little, so expect
# somewhat less context capacity than the single-card int4 figures.
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
PORT=${PORT:-8080}
export PATH="$VENV/bin:$PATH"

[ -f "$MODEL/config.json" ] || { echo "no model at $MODEL -- build it first (prepare/build_fast_model.py; the Swift variant: prepare/build_swift_model.py; or: docker run ... prepare)" >&2; exit 1; }
if [ ! -x "$VENV/bin/vllm" ] && ! command -v vllm >/dev/null; then
  echo "no vllm found -- create the uv venv first (README: Bare metal), or run this in the container" >&2; exit 1
fi

# a dead engine leaves its /dev/shm offload region and the next boot dies on it (upstream #33)
if [ "${VLLM_OFFLOAD_KEEP_SHM:-0}" != 1 ]; then
  for f in /dev/shm/vllm_offload_*.mmap; do
    [ -e "$f" ] || continue
    grep -lqs "$f" /proc/[0-9]*/maps 2>/dev/null || { echo "[w4a16-int4-mtp] removing stale offload region $f"; rm -f "$f"; }
  done
fi

# the 3D multi-query verify dispatch for the int4 cache (see header)
export VLLM_INT4_MQ_3D=1
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
exec vllm serve "$MODEL" \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port $PORT \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.93 \
  --max-model-len auto \
  --max-num-seqs 8 \
  --api-server-count 1 \
  --attention-backend TRITON_ATTN \
  --kv-cache-dtype int4_per_token_head \
  --mamba-ssm-cache-dtype float16 \
  --async-scheduling \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --prefix-caching-hash-algo xxhash \
  --prefix-match-unit 808 \
  --mamba-cache-mode align \
  --mm-processor-kwargs '{"size":{"shortest_edge":65536,"longest_edge":2097152}}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}' \
  --compilation-config '{"max_cudagraph_capture_size":32,"cudagraph_mode":"PIECEWISE","custom_ops":["+rms_norm","+silu_and_mul"]}' \
  --reasoning-parser qwen3 \
  --enable-prompt-tokens-details \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  ${EXTRA_ARGS:-}
