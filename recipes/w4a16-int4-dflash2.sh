#!/bin/bash
# w4a16-int4-dflash2: the dflash2 stack with the int4 per-token-head KV cache.
#
# The context-capacity play: int4 KV halves the bytes of the int8 cache, so
# the same VRAM pool holds ~2x the context (upstream's single-card profile
# of exactly this shape served a 314,915-token pool at 256k max len, vs
# 57,669 for the bf16 config -- PR #42). Costs ~20% decode vs the bf16
# FlashAttention path.
#
# Two items to trust-but-verify on your workload (this box, TP=2):
# (1) quality at depth was measured once upstream, on a single RTX 4090:
# 96.0% GSM8K (200 questions, greedy) and a 100k-token needle retrieved at
# 90% depth -- inside the band the other configs read (95.0-96.5%), but it
# has not been measured on this TP=2 shape; (2) VLLM_INT4_MQ_3D (below) is
# the multi-query 3D dispatch: upstream found two defects in it -- silent
# 2D fallback on 9-16-token batches (#46 gate campaign) and out-of-budget
# scratch (#57 review) -- that the two spec-decode-scratch-* patches fix,
# so the 2D fallback exists mainly as a debug switch --
# sanity-check outputs against w4a16-bf16-dflash2 on the same prompts.
#
# --prefix-match-unit 848 is not optional here: under int4's halved-page
# geometry the drafter's sliding-window block is 848 tokens against a 1696
# hash unit, and without the flag the prefix cache can never match this KV
# layout (upstream: wsl2-4090.md). Needs three patches: patches/int4-kv-
# per-token-head.patch (boot blockers for int4 KV with the drafter),
# patches/spec-decode-int4-kv-mq3d.patch (VLLM_INT4_MQ_3D) and the
# spec-decode-scratch-* pair (3D scratch sizing + budgeting). The split-KV
# verify kernel reads bf16/int8 caches only, so VLLM_SPEC_DECODE_ATTN is
# deliberately unset here.

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
DRAFT=${DRAFT:-$REPO/models/Qwen3.8-27B-DFlash2-W4A16}
PORT=${PORT:-8080}
export PATH="$VENV/bin:$PATH"

[ -f "$MODEL/config.json" ] || { echo "no model at $MODEL -- run: python prepare/build_fast_model.py <dir> (or: docker run ... prepare)" >&2; exit 1; }
[ -f "$DRAFT/config.json" ] || { echo "no drafter at $DRAFT -- run: python prepare/fetch_dflash2.py <dir>" >&2; exit 1; }
if [ ! -x "$VENV/bin/vllm" ] && ! command -v vllm >/dev/null; then
  echo "no vllm found -- create the uv venv first (README: Bare metal), or run this in the container" >&2; exit 1
fi

# a dead engine leaves its /dev/shm offload region and the next boot dies on it (upstream #33)
if [ "${VLLM_OFFLOAD_KEEP_SHM:-0}" != 1 ]; then
  for f in /dev/shm/vllm_offload_*.mmap; do
    [ -e "$f" ] || continue
    grep -lqs "$f" /proc/[0-9]*/maps 2>/dev/null || { echo "[w4a16-int4-dflash2] removing stale offload region $f"; rm -f "$f"; }
  done
fi

# the V2 runner (forced by dflash) doesn't count its ~1.4 GiB of CUDA graphs
# against gpu-memory-utilization, so they're reserved here
# (patches/hybrid-kv-groups-v2-cudagraph.patch)
export VLLM_V2_CUDAGRAPH_MEM_MIB=1400
# vision tower in pinned host RAM by default (upstream default; patches/vision-tower-cpu-offload.patch):
# off the VRAM budget, bit-exact, ~+12% per image forward; =0 keeps it GPU-resident
export VLLM_VISION_CPU_OFFLOAD_GB=${VLLM_VISION_CPU_OFFLOAD_GB:-1}
# dflash2's verify is an 8-row multi-query batch; without this the int4
# attention does a ~20-CTA 2D serial walk of the whole KV every step
# (upstream measured 3.6 tok/s at 72.6k depth, vs 29.0 with speculation
# off), with the 3D dispatch it uses the full grid
# (patches/spec-decode-int4-kv-mq3d.patch)
export VLLM_INT4_MQ_3D=1
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

# tuned Marlin (patches/marlin-tune-table.patch, built by
# prepare/build_marlin_tune.py): auto-enable when the extension is
# importable; VLLM_MARLIN_TUNE=0 keeps it off. Upstream measured
# +2-20% per GEMM but only ~+0.4% end-to-end at a 250 W cap
if [ -z "${VLLM_MARLIN_TUNE:-}" ] && "${VENV}/bin/python" -c 'import marlin_best, marlin_tune_ext' >/dev/null 2>&1; then
  export VLLM_MARLIN_TUNE=1
fi

# no --language-model-only: the server takes image input.
# Vision: up to 16 images per request, each capped at 2097152 px = 2048
# tokens -- the cap (not the count) sets the encoder's profiled peak in
# the KV pool (at most the 4096-token encoder budget); the count only
# bounds per-request context. xxhash: faster prefix-cache hashes than sha256.
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
  --max-num-seqs 4 \
  --api-server-count 1 \
  --attention-backend TRITON_ATTN \
  --kv-cache-dtype int4_per_token_head \
  --mamba-ssm-cache-dtype float16 \
  --async-scheduling \
  --max-num-batched-tokens 4096 \
  --enable-prefix-caching \
  --prefix-caching-hash-algo xxhash \
  --prefix-match-unit 848 \
  --mamba-cache-mode align \
  --limit-mm-per-prompt '{"image":{"count":16}}' \
  --mm-processor-kwargs '{"size":{"shortest_edge":65536,"longest_edge":2097152}}' \
  --speculative-config '{"method":"dflash","model":"'"$DRAFT"'","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}' \
  --compilation-config '{"max_cudagraph_capture_size":32,"custom_ops":["+rms_norm","+silu_and_mul"]}' \
  --reasoning-parser qwen3 \
  --enable-prompt-tokens-details \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
