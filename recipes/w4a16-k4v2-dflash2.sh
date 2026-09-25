#!/bin/bash
# w4a16-k4v2-dflash2: the dflash2 stack with the KVarN 4-bit-key / 2-bit-value
# KV cache (kvarn_k4v2_g128) -- the long-context play for the model's full
# 262k.
#
# KVarN (Huawei CSL, Apache-2.0) is a native vLLM attention backend ported
# onto the 0.29.0 this repo runs: Hadamard rotation + iterative variance
# normalization + 4/2-bit RTN per 128-token tile, ~840 B/token/layer
# (fp8: 2048 B) -- roughly half the int4 per-token-head cache, so the same
# VRAM holds the full 262,144-token context the model declares. Upstream
# measurements (single 3090; not ours): 262k fits (a 420k-token pool at 4
# slots vs ~200k at fp8), needle-in-a-haystack correct at 4k...240k,
# perplexity +0.16%, prefill within +/-5% of fp8, and a decode tax that
# runs from ~6% on short prompts to 2.13x at 112k context (upstream
# docs/long-context.md; their MTP single-user lane -- this recipe's
# dflash2 drafter reads and writes the same cache).
#
# It needs three patches (in build order): patches/kvarn-0.29.0.patch
# (the vLLM wiring: cache-dtype literals, KVQuantMode.KVARN, the backend
# registry + CUDA priority list, the KV-cache-spec branch, the tail-pool
# max_num_seqs cap), patches/kvarn-files-0.29.0.patch (the backend
# modules themselves) and patches/kvarn-v2-runner-0.29.0.patch (the V2
# runner, sliding-cache and DFlash2 correctness fixes). All are in the
# set prepare/patch_vllm.py applies (and the Docker build applies the
# same way); there is no separate install step.
#
# Two flags are not optional here:
#   --block-size 128: the KVarN tile IS the block (its variance
#     normalization runs per 128-token tile). The hybrid page unification
#     then pads the attention page to the 2048-token Gated-DeltaNet page;
#     the backend reads it in 128-token kernel tiles.
#   --prefix-match-unit 128: the prefix hash unit must equal the tile, so
#     cache hits land on tile boundaries -- a non-multiple of 128 corrupts
#     the pool (upstream single-user launcher). The k4v2 analogue of the
#     int4 recipe's 848.
#
# --prefix-cache-retention-interval 13056: at 7 drafts the attention block
# is 2176 tokens, and vLLM's default dense retention means two long
# conversations advanced in turn evict each other's mamba snapshots (0%
# prefix reuse, a full re-prefill every turn); one in six (6 x 2176) kept
# 93-99.5% on 0.28; on 0.29 the first reuse after a cold turn lands on the
# last retained snapshot (80.0% at 32.6k) and the turns after it run
# 99.3-99.4% (upstream #174, single 3090 -- their measurement, not ours).
# The split-KV verify attention (VLLM_SPEC_DECODE_ATTN) reads bf16/int8
# caches only, so it is off here: the KVarN backend brings its own
# dequant verify path (upstream: KVARN_FUSED_VERIFY, on by default).
#
# Deviations from the upstream single-card launcher (CTX=huge SPEC=dflash2):
#   1. TP=2 with gpu-memory-utilization sizing: all the recipes here run
#      2x3090; the launcher's single-card 5.26 GiB KV_MEM pin is dropped
#      at TP>1 (its own rule), as in the other recipes.
#   2. --max-model-len 245760, not auto: the full 262144 does not boot on
#      a single 24 GB card (one max-len request needs ~5.05 GiB of the
#      4.90 GiB pool at k=7, upstream PR #13), and our TP=2 headroom for
#      it is unmeasured. 245760 is the launcher's k<=7 default; the pool
#      at TP=2 has room for the full 262144 -- raise it via EXTRA_ARGS
#      once you have measured it on this box.
#   3. --max-num-seqs 2 (max_cudagraph_capture_size 16 = 2 x the 8-token
#      verify block), not the 8/64 the sibling dflash2 recipes here run:
#      the KVarN fp16 tail pool (the first 128 tokens per request stay
#      unquantized) plus the per-request state pages are budgeted against
#      the KV pool, and upstream found MAX_SEQS>12 fatal on a 24 GB card
#      at this context (gotcha 38/39), defaulting the single-user
#      dflash2+huge profile to 2 slots.
#   4. --max-num-batched-tokens 2048, not the 8192 the sibling recipes
#      deviate to: under KVarN the activation peak a bigger chunk
#      profiled shrinks exactly the pool this recipe sells, and every
#      quoted number for this lane was measured at 2048.
#
# Upstream gotcha 51 / #64: KVarN + MTP + prefix caching corrupts
# prompt_logprobs (perplexity ~23% high, NaN 400s); the same combo with
# this recipe's dflash2 drafter is clean. Quality measurement on a MTP
# server under kvarn_k4v2 needs PREFIX_CACHE=0 (EXTRA_ARGS).
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
    grep -lqs "$f" /proc/[0-9]*/maps 2>/dev/null || { echo "[w4a16-k4v2-dflash2] removing stale offload region $f"; rm -f "$f"; }
  done
fi

# the KVarN backend dequantizes the packed tiles inside its own verify
# path; the split-KV verify kernel (patches/spec-decode-*.patch) reads
# bf16/int8 caches only, so it stays off here
export VLLM_SPEC_DECODE_ATTN=0
# the fp16 tail pool (first 128 tokens per request, never quantized) is
# budgeted as a share of the post-weight VRAM envelope: 0.15 is the
# upstream single-user huge default (their batch lane uses 0.25)
export KVARN_POOL_MEM_FRAC=${KVARN_POOL_MEM_FRAC:-0.15}
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
# at TP=2 custom all-reduce exports its graph buffers over CUDA IPC and an expandable (VMM) segment has none to export, so default the allocator plain (upstream #163/#176; set the variable to override)
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}

# no --language-model-only: the server takes image input.
# Vision: image count is unlimited; each image is capped at 2097152 px
# = 2048 tokens, and that cap sets the encoder's profiled peak in the
# KV pool (at most the 4096-token encoder budget). xxhash: faster
# prefix-cache hashes than sha256. No --attention-backend: the kvarn_k4v2
# cache dtype selects the KVarN backend itself (the kvarn-0.29.0 patch
# registers it in the CUDA priority list, like TurboQuant).
# draft_sample_method is required on 0.29.0: the native DFlash2 inherits the
# upstream speculator base, which allocates the draft-logits buffer only when
# the config asks; without it the rejection test loses its denominator and
# acceptance drops ~16% (upstream #73).
exec vllm serve "$MODEL" \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port $PORT \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.93 \
  --max-model-len 245760 \
  --max-num-seqs 2 \
  --api-server-count 1 \
  --kv-cache-dtype kvarn_k4v2_g128 \
  --block-size 128 \
  --mamba-ssm-cache-dtype float16 \
  --async-scheduling \
  --max-num-batched-tokens 2048 \
  --enable-prefix-caching \
  --prefix-caching-hash-algo xxhash \
  --prefix-match-unit 128 \
  --prefix-cache-retention-interval 13056 \
  --mamba-cache-mode align \
  --mm-processor-kwargs '{"size":{"shortest_edge":65536,"longest_edge":2097152}}' \
  --speculative-config '{"method":"dflash","model":"'"$DRAFT"'","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}' \
  --compilation-config '{"max_cudagraph_capture_size":16,"custom_ops":["+rms_norm","+silu_and_mul"]}' \
  --reasoning-parser qwen3 \
  --enable-prompt-tokens-details \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  ${EXTRA_ARGS:-}
