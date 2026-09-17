# syv-recipes — Qwen3.8-27B serving for a two-GPU box

A fork of [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090),
specialized to one machine shape: **two 24 GB GPUs (tensor-parallel 2)**,
serving Qwen3.8-27B with vLLM 0.28.0 and its custom patch stack. Everything
not needed for serving is removed; each recipe calls `vllm serve` with
explicit flags — no CTX/SPEC-style configuration to construct.

| recipe | KV cache | speculation | reach for it when |
|---|---|---|---|
| [w4a16-int8-dflash2.sh](recipes/w4a16-int8-dflash2.sh) (default) | `int8_per_token_head`, native max len | DFlash2 drafter, 7 drafts in one pass | the all-round pick: fastest decode, double the context pool of bf16 |
| [w4a16-int8-mtp.sh](recipes/w4a16-int8-mtp.sh) | `int8_per_token_head`, native max len | Qwen's own MTP head, 3 drafts, probabilistic | no separate drafter model; the head ships inside the checkpoint |
| [w4a16-bf16-dflash2.sh](recipes/w4a16-bf16-dflash2.sh) | unquantized bf16, FlashAttention | DFlash2, 7 | the quality baseline -- no quantized-KV approximation anywhere |
| [w4a16-int4-dflash2.sh](recipes/w4a16-int4-dflash2.sh) | `int4_per_token_head` | DFlash2, 7 | context capacity: double the context of int8 in the same VRAM |
| [w4a16-k4v2-dflash2.sh](recipes/w4a16-k4v2-dflash2.sh) | `kvarn_k4v2_g128` (KVarN 4/2-bit tiles) | DFlash2, 7 | the full 262k context: ~2x the int4 pool, at a context-length-dependent decode tax |
| [w4a16-k4v2-mtp.sh](recipes/w4a16-k4v2-mtp.sh) | `kvarn_k4v2_g128` (KVarN 4/2-bit tiles) | Qwen's own MTP head, 3 drafts, probabilistic | the full 262k context with no separate drafter model; 8 seats for 4-8 concurrent |
| [w4a8-int8-dflash2.sh](recipes/w4a8-int8-dflash2.sh) | `int8_per_token_head` + W4A8 linears | DFlash2, 7 | faster prefill at the documented quality cost |
| [w4a16-int8-dspark.sh](recipes/w4a16-int8-dspark.sh) | `int8_per_token_head` | DSpark community drafter (bf16), 7 | upstream measured it slower than the dflash2 head on their boxes; unmeasured on this shape |

The numbers in the notes below are upstream's, measured on their reference
box; we have not benchmarked these recipes -- run them on your hardware
before quoting a figure. Most predate the 0.28.0 rebase: upstream has not
re-measured its 0.28.0 matrix yet (see Notes).

All recipes take image input (no `--language-model-only`), with no
image-count limit per request; each image is capped at 2097152 px =
2048 tokens, and that cap sets the encoder's profiled peak, which
comes out of the KV pool. The ~0.9 GB vision
tower is offloaded to pinned host RAM by default
(`VLLM_VISION_CPU_OFFLOAD_GB=1`; `=0` keeps it GPU-resident). They also
use `--enable-prefix-caching` with `--prefix-caching-hash-algo xxhash`
(the `xxhash` package is in requirements.txt),
`--gpu-memory-utilization 0.93` (under TP>1 the pool is sized from
utilization), `--mamba-ssm-cache-dtype float16` (halves the GDN state
cost), `--max-num-seqs 8` (w4a16-k4v2-dflash2: 2) and
`--max-num-batched-tokens 8192` (w4a16-k4v2-dflash2, w4a16-k4v2-mtp: 2048), the qwen3 reasoning parser,
qwen3_xml tool parsing, and `--enable-prompt-tokens-details`, so every
response's `usage.prompt_tokens_details.cached_tokens` shows how much of
the prompt the prefix cache served. Port 8080 (`PORT=…`) and a `.venv` at
the repo root (`MODEL`, `DRAFT` overridable the same way). The dflash2-
family recipes run the default FULL_AND_PIECEWISE capture (upstream swept
dflash2 clean across every 128-residue prompt length in FULL mode);
w4a16-int8-mtp.sh forces PIECEWISE -- see its header.

## The full matrix (model-quant x KV x speculation)

The recipe name *is* the cell: `<model-quant>-<kv>-<speculation>.sh`. Both
model rows run the same W4A16-AutoRound checkpoint -- "w4a8" is the
serve-time activation quant (`VLLM_MARLIN_INPUT_DTYPE=int8`), not a
different download.

| model | KV | spec | status | decode | prefill (16k) | quality |
|---|---|---|---|---|---|---|
| w4a16 | int8 | dflash2 | recipe (default) | | | |
| w4a16 | int8 | mtp | recipe (upstream's MTP lane was fp8/FlashInfer; MTP + int8 + split-KV never measured upstream) | | | |
| w4a16 | int8 | dspark | recipe (upstream measured dspark on bf16 KV only -- the int8 cell is ours to fill) | | | |
| w4a16 | bf16 | dflash2 | recipe (the quality baseline) | | | |
| w4a16 | int4 | dflash2 | recipe (upstream measured depth quality once, on a single 4090: 96.0% GSM8K, 100k needle retrieved; untested on TP=2) | | | |
| w4a16 | k4v2 (KVarN) | dflash2 | recipe (upstream measured the MTP lane on one 3090: 262k fits, 2.13x decode at 112k; the dflash2 cell is ours to fill) | | | |
| w4a16 | k4v2 (KVarN) | mtp | recipe (upstream's CTX=huge SPEC=mtp lane, single 3090: 2.13x decode at 112k, +0.16% PPL; the TP=2, 8-seat cell is ours to fill) | | | |
| w4a16 | bf16 | mtp | valid, no recipe (the split-KV kernel's native path is bf16) | | | |
| w4a16 | int4 | mtp | untested (does the 3D dispatch cover MTP verify, or is the 2D walk?) | | | |
| w4a8 | int8 | dflash2 | recipe | | | |
| w4a8 | bf16 | dflash2 | valid, no recipe (upstream's single-user INT8 default lane) | | | |
| w4a8 | int4 | dflash2 | untested | | | |
| w4a8 | int8 | mtp | untested (the layer-select patch keeps `mtp.*` on W4A16) | | | |
| w4a8 | bf16 | mtp | untested | | | |
| w4a8 | int4 | mtp | untested | | | |

Fill the last three columns with your own measurements (upstream's
reference-box numbers are in the notes, deliberately not here). Cells that
never become recipes: fp8 KV wedges 3090-class cards (upstream issue #34)
-- also why our mtp recipe is int8 rather than upstream's fp8 lane; a w4a8
layer set of `mlp|linear_attn` crashes at first forward (inductor bug);
every mtp cell runs PIECEWISE capture (FULL_AND_PIECEWISE corrupts output
on prefix-cache hits).

## Layout

```
Dockerfile          uv + pinned vLLM 0.28.0 + every patch in patches/
requirements.txt    the pinned set (vllm pulls torch 2.13 / flashinfer itself)
setup.py            bare-metal one-shot: venv, deps, patches, model prep
recipes/            the eight serve configurations
prepare/            build_fast_model.py, fetch_dflash2.py, fetch_dspark.py, patch_vllm.py -- model prep + idempotent patching (runs standalone)
patches/            the 34 vLLM patches (below)
docker/             entrypoint.sh, prepare.sh
```

## Docker

```bash
docker build -t syv-recipes .

# optional but recommended: reuse the host's existing HF hub cache -- the
# one-liner resolves to $HF_HUB_CACHE, or $HF_HOME/hub, or the default,
# however the host configures it
HUB="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
docker run -d --name qwen --gpus all --ipc=host -p 8080:8080 \
  -v qwen-models:/app/models -v qwen-cache:/cache \
  -v "$HUB":/cache/.cache/huggingface/hub \
  --restart unless-stopped syv-recipes            # entrypoint default: w4a16-int8-dflash2
```

The entrypoint prepares the models on first start (downloads through the
mounted hub cache — seconds when it is warm; `qwen-cache` holds everything
else), then execs the recipe: `syv-recipes w4a16-int8-mtp`,
`w4a16-bf16-dflash2`, `w4a16-int4-dflash2`, `w4a8-int8-dflash2` or
`w4a16-int8-dspark` -- for the others, `syv-recipes prepare` for prep only,
`PREPARE=0` to skip it (the prep honors `SWIFT=1` for the Swift variant --
see the note). `qwen-models` receives the assembled dirs:
hard-linked off the cache when both volumes share a filesystem, a second
~25 GB copy when they don't. `VLLM_API_KEY=…` turns on key auth; without
it the server binds 0.0.0.0 and is open. Any of these can be passed from a
`.env` with `--env-file`.

The `-v "$HUB":…` mount is a run-time one (docker build cannot see host
directories, and the cache is consumed at run time by the prep anyway);
the container side is pinned in the Dockerfile. Caveats: the container
runs as root, so files it downloads into your host cache are root-owned;
on Docker Desktop the mount crosses virtiofs, so the prep's hard-links
fall back to a full copy.

## Bare metal (uv)

```bash
./setup.py             # venv + pinned deps + the 30 patches + all three models (DSPARK=/path redirects the DSpark dir; =0 skips it; SWIFT=1 adds the ~20 GB Swift variant)
bash recipes/w4a16-int8-dflash2.sh  # or any of the other seven
```

`setup.py` is idempotent — re-run it any time; it is also the recovery
path after a venv wipe. `patch_vllm.py` converges whatever state it
finds: a stamped venv whose vllm version, hash of `patches/`, and hash of the files the patches touch
all still match is a fast "Audited … in place" no-op; a fully patched but
not-yet-stamped tree (patched by an older run) is audited and stamped
without touching a file; a partially patched tree (a crash mid-run) is
completed in build order; a changed patch set, a vllm
version change, or a hand-edited tree is detected and vllm reinstalled
pristine before re-apply (the attestation covers the
hunks: a hand edit that leaves every hunk's before/after content intact
is indistinguishable from a patched tree). Paths are overridable with
env vars of the same names the recipes use (defaults: `.venv`,
`models/Qwen3.8-27B-W4A16-AutoRound-fast`,
`models/Qwen3.8-27B-DFlash2-W4A16`), so a custom layout stays consistent
between setup and serve:

```bash
MODEL=/data/qwen VENV=/data/qwen/.venv ./setup.py
MODEL=/data/qwen bash recipes/w4a16-int8-dflash2.sh
```

The same overrides can live in a `.env` file at the repo root instead:
every recipe and both `setup.py` and `patch_vllm.py` read it, but only for
a variable that is unset or empty in the real environment, which always
wins. Values may be quoted; whole-line `#` comments only. The variables the
scripts and recipes consume are `VENV`, `MODEL`, `DRAFT`, `DSPARK`, `SWIFT`, `PORT` and `EXTRA_ARGS` (e.g.
`VENV=/data/qwen/.venv`); note the `VLLM_*` env vars each recipe hard-exports
are always set by the recipe itself, so a `.env` cannot change them.
`EXTRA_ARGS` is special: every recipe appends it (unquoted) at the end of its
`vllm serve` line, so it can carry any vLLM flags on top of the recipe's own;
for single-valued options its value wins over the recipe's
(e.g. `EXTRA_ARGS="--disable-async-scheduling"`), and it is on your own responsibility.
Both preps fetch the DSpark drafter by default, into the recipe's own
default dir; a path form (`DSPARK=/dir`) puts it in `/dir` instead, in
which case the recipe finds it only if `DRAFT` points at the same place
(`=0` is an explicit no-op).
In the container use the native equivalent, `docker run --env-file .env`
(`.env` is gitignored and out of the build context).

What `setup.py` runs, in order, if you'd rather do it by hand (install uv
first, and GNU `patch` if your distro lacks it:
`curl -LsSf https://astral.sh/uv/install.sh | sh`):

```bash
uv venv .venv --python 3.14
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python prepare/patch_vllm.py
.venv/bin/python prepare/build_fast_model.py models/Qwen3.8-27B-W4A16-AutoRound-fast
.venv/bin/python prepare/fetch_dflash2.py        models/Qwen3.8-27B-DFlash2-W4A16
# optional: the DSpark drafter (for recipes/w4a16-int8-dspark.sh)
.venv/bin/python prepare/fetch_dspark.py          models/Qwen3.8-27B-DSpark
# optional: the ukisai Swift variant, W4A16 (~20 GB; see the Notes bullet)
.venv/bin/python prepare/build_swift_model.py     models/Qwen3.8-27B-Swift-W4A16
```

The recipes put the venv's `bin` on PATH (`VENV`, defaulting to `.venv`) and
default to port 8080; `MODEL`, `DRAFT` and `PORT` may be overridden with
env vars of the same names.

## The kept patches (all apply to vLLM 0.28.0)

DFlash2 itself is native in vLLM 0.28.0 (upstream PR #52816) — the 0.27.1
`dflash2-backport` is retired and no longer applied. The `dflash2-*`
patches below extend the native implementation instead.

- `dflash2-lookup-drafting.patch` — lookup-augmented drafting (`VLLM_DFLASH2_LOOKUP=1`, off by default): propose continuations of earlier occurrences of the current suffix; on 0.28.0 it also carries the W4A16 draft-checkpoint support (packed-qkv dequant, quantized lm_head sharing) that native DFlash2 needs for this model
- `dflash2-ngram-chains.patch` — drafter-free n-gram chains on top of lookup (`VLLM_DFLASH2_CHAIN=1`, off by default, greedy by default): while a request keeps reproducing its own context, whole verify blocks come from history and the drafter's forward and graph replay are skipped (upstream #38, ported from Dmtrii-tesla's fork)
- `dflash2-prewarm.patch` — pre-compiles the `_prepare_dflash_inputs_kernel` Triton variants at boot, so the first large prefill never JIT-compiles mid-request (#48)
- `dflash2-z-adaptive-emitted.patch` — fixes the emitted-token accounting the lookup's adaptive block-length choice reads (upstream's 2026-09 re-cut replaces the old comment that contradicted the fix: `num_sampled` is the count the sampler emitted, and `num_rejected` is defined relative to it)
- `dspark-draft-quant-config.patch` — lets a bf16 DSpark drafter load beside the quantized target: the loader asked for the draft's quantization config through the target's hf_overrides as a callable, which `get_quant_config` refuses — return None when the draft carries none of its own (upstream #84 / issue #25 item 15, validated on 0.28.0; the w4a16-int8-dspark recipe)
- `hybrid-kv-groups-v2-cudagraph.patch` — KV-group sizing for the drafter's sliding-window layers; explicit CUDA-graph memory accounting (`VLLM_V2_CUDAGRAPH_MEM_MIB`)
- `hybrid-sw-block-promote.patch` — lets a quantized KV cache fit in a hybrid target+drafter (block-size promotion instead of page padding)
- `int4-kv-per-token-head.patch` — boot blockers for int4 per-token-head KV with the drafter
- `kvarn-0.28.0.patch` — the KVarN (Huawei CSL) dense KV-cache backend wiring for 0.28.0: the `kvarn_*` cache-dtype literals, `KVQuantMode.KVARN`, the backend registry + CUDA priority list, the packed-tile KV-cache spec, and the fp16 tail-pool `max_num_seqs` cap (upstream kvarn/)
- `kvarn-files-0.28.0.patch` — the KVarN backend modules themselves (quantization config, Triton kernels, `KVarNAttentionBackend`), upstream's port of KVarN's vLLM 0.23 fork onto 0.28.0, create-only (the w4a16-k4v2-dflash2 recipe)
- `kvarn-v2-runner-0.28.0.patch` — the V2-runner, sliding-cache and DFlash2 correctness fixes the KVarN lane needs; applied after the base port, cut against a tree carrying `mamba-align-retire-null-gaps` (upstream kvarn/)
- `marlin-int8-layer-select.patch` -- env-selectable int8-activation layers for the Marlin path (the w4a8-int8-dflash2 recipe)
- `marlin-int8-negative-scales.patch` — correctness fix for negative group scales in W4A8
- `marlin-repack-staged-sm80.patch` — staged Marlin repack (load-time allocation hygiene; the header records #27's corrected history — the old "VMM churn" theory was disproven)
- `marlin-tune-table.patch` — routes `marlin_gemm` through a locally built tunable Marlin extension (`VLLM_MARLIN_TUNE=1`, a no-op without that build): +3-7% on the M≤16 decode/verify GEMMs, +2-20% on W4A8 chunked-prefill GEMMs
- `mamba-align-checkpoint-order.patch` — retention of the mamba state snapshots a conversation's prefix-cache hits resume from; fixes the ~1-in-4-5-turn TTFT spikes (upstream #52 / vllm#45238). Ships default off; all eight recipes default it on (`VLLM_MAMBA_ALIGN_KEEP_CHECKPOINTS=1`)
- `mamba-align-retire-null-gaps.patch` — retires mamba align-mode state blocks across null gaps instead of holding them until the request ends (upstream backport of vllm#55450): a 480k-token prefill leaves 71 retained state blocks per Mamba group where 8-9 is the bound — peak pool pressure exactly where the k4v2 lane runs. The 0.28.0 sync left it pending a 3090 A/B against our `VLLM_MAMBA_ALIGN_KEEP_CHECKPOINTS=1` default (both touch the align-snapshot free path); it is adopted with the KVarN lane, which the v2-runner patch was cut against, and upstream #101 measured it inert on the dflash2 profile
- `mamba-chunked-prefill-align.patch` — correctness fix: GDN/Mamba state loss and NaN during chunked prefill (the state-copy source column, and an uninitialized-memory mask in the flash-linear-attention chunk-o kernel)
- `offload-dflash-eagle-groups.patch` — OffloadingConnector group flagging under dflash
- `qwen3_5-embed-quant.patch` — route the embedding table through the quantized path (the fast model needs it)
- `qwen3_5-mtp-draft-vocab.patch` — vocab-truncated MTP draft head (the fast model's 40k draft head)
- `sampler-small-topk-fast-softmax.patch` — sort-free top-k/top-p, multi-block softmax, truncated drafts
- `spec-decode-attn.patch` — split-KV verify attention (`VLLM_SPEC_DECODE_ATTN`), bf16 path
- `spec-decode-int4-kv-mq3d.patch` — multi-query 3D dispatch for the int4-KV verify (`VLLM_INT4_MQ_3D`; the w4a16-int4-dflash2 recipe)
- `spec-decode-int8-kv.patch` — teaches the split-KV kernel to read the int8 per-token-head cache
- `spec-decode-scratch-token-units.patch` — sizes the int4-3D softmax scratch buffers in query tokens, not sequences (the #46 gate campaign: 9-16-token verify batches had been falling back to 2D silently)
- `spec-decode-scratch-within-budget.patch` — allocates those buffers inside the memory profile, so the reported KV budget is honest (#57 merge review)
- `spec-sampler-prewarm.patch` — compiles the rejection sampler's Triton kernels at boot with a spec-shaped dummy verify, so the first draft-verifying request never JITs mid-request (#48)
- `speed-knobs-envs.patch` — registers the speed knobs as env vars (torch.compile cache key)
- `triton-prefill-attn-int8.patch` — int8-QK Triton prefill attention for the head_dim-256 layers (`VLLM_PREFILL_ATTN=int8`; bf16 KV, single-request prefill chunks only, off by default; 1.27-1.35x on the kernel vs FA2)
- `vision-tower-cpu-offload.patch` — vision tower in pinned host RAM (`VLLM_VISION_CPU_OFFLOAD_GB`)
- `vllm-pr50021-gdn-spec-bounds.patch` — bounds on accepted-token state lookups in the GDN/Mamba spec kernels
- `vllm-pr54282-draft-gumbel-salt.patch` — vLLM #54282: the Gumbel noise that re-samples a rejected draft token was correlated with the noise that produced it, biasing acceptance toward the drafter; salts the draft's stream (1<<30) so the two are independent (upstream backport; the dflash2 hunk ported by hand)
- `xgrammar-spec-terminated.patch` — structured output survives tokens accepted past the grammar's end

Removed from the upstream stack: the whole WSL2 lane (out of scope).
KVarN was dropped with it at the fork and is back in the set with the
w4a16-k4v2-dflash2 recipe (adopted on request; its V2-runner port and the
retire-null-gaps fix its patch set was cut against are in too).
Lookup-augmented drafting and n-gram chains are in the set (both off by
default) — adopted in this sync.

## Notes

- **int8 KV is a trade** (the three int8-KV recipes): double the pool of
  bf16, at the cost of the Triton backend and a per-step unpack;
  single-user spec-decoded quality at depth is unmeasured upstream (batch
  mode: 100k-needle ok, PPL neutral — upstream docs/long-context.md).
  Verify perplexity/GSM8K on your workload before trusting it.
- **w4a16-bf16-dflash2 is the baseline**: upstream measured 96.5% GSM8K on
  this exact shape (dflash2, 4.80 tokens/step). If a quantized recipe's
  output looks off, this is the one to compare against.
- **w4a16-int4-dflash2**: depth quality has been measured once upstream,
  on a single RTX 4090 — 96.0% GSM8K (200 questions, greedy) and a
  100k-token needle retrieved at 90% depth, inside the 95.0-96.5% band the
  other configs read; it has not been measured on this TP=2 shape. The
  3D verify dispatch is vetted: upstream found two defects in the 3D path
  — the silent 2D fallback on 9-16-token batches (its #46 gate campaign)
  and scratch allocated outside the memory budget (the #57 review) — and
  the two `spec-decode-scratch-*` patches fix both. Compare outputs against
  w4a16-bf16-dflash2 before trusting it anyway — it costs ~20% decode vs
  the bf16 path, and its prefix cache only works with the
  `--prefix-match-unit 848` flag the recipe passes (without it the
  drafter's 848-token sliding-window block can never match the 1696-token
  hash unit).
- **w4a16-k4v2-dflash2**: the long-context play. KVarN's 4-bit-key /
  2-bit-value tiles (~840 B/token/layer) halve the int4 cache, so the same
  VRAM holds the model's full 262,144-token context. Upstream measurements
  (single 3090, not ours): a 420k-token pool at 4 slots vs ~200k at fp8,
  needle-in-a-haystack correct at 4k...240k, perplexity +0.16%, prefill
  within +/-5% of fp8; the decode tax runs from ~6% on short prompts to
  2.13x at 112k context (their MTP single-user lane — the dflash2 cell is
  ours to fill on TP=2). Two flags are load-bearing: `--block-size 128`
  (the tile) and `--prefix-match-unit 128` (prefix hits land on tile
  boundaries; a non-multiple corrupts the pool). Upstream gotcha 51 / #64:
  KVarN + MTP + prefix caching corrupts prompt_logprobs (perplexity ~23%
  high, NaN 400s); this recipe's dflash2 drafter is clean on that combo.
  Compare outputs against w4a16-bf16-dflash2 before trusting it.
- **w4a16-k4v2-mtp**: the full 262k context with no separate drafter
  model: the upstream single-user launcher's CTX=huge SPEC=mtp lane
  (DRAFT_TOKENS=3, its huge default) at TP=2 with 8 seats for the 4-8
  concurrent range. Upstream measurements (single 3090, not ours): 262k
  fits, needle-in-a-haystack correct at 4k...240k, perplexity +0.16%,
  prefill within +/-5% of fp8, and the decode tax runs ~6% on short
  prompts to 2.13x at 112k. The launcher's 200k default is the single
  card's pool cap (one 262k request needs ~5.05 GiB of the 4.90 GiB pool,
  upstream PR #13), not the shape: at TP=2 the request's KV shards across
  both cards, so the recipe serves the full 262144. Upstream gotcha 51 /
  #64: KVarN + MTP + prefix caching corrupts prompt_logprobs (perplexity
  ~23% high, NaN 400s); ordinary generation is not implicated. Measure
  quality with prefix caching off for the run (EXTRA_ARGS); serve with it
  on. The TP=2 C4-C8 cell is ours to fill.
- **w4a8-int8-dflash2 is a quality-for-speed trade**: the default MLP-only
  layer set costs +2.2% PPL for +13-14% prefill; `INT8_LAYERS=all` (the
  recipe expands the upstream shorthand to `mlp|linear_attn|self_attn`) is
  +27-30% at GSM8K 95.0 vs 96.5 and +4.1% PPL. Decode is unchanged either
  way (memory-bound).
- **MTP + split-KV verify**: a configuration upstream never measured (see
  the w4a16-int8-mtp header); check draft acceptance on your workload.
- **The 0.28.0 rebase** (this sync): DFlash2 went native, the old
  backport is retired, and the set above all apply to 0.28.0 (three
  patches keep upstream's older "written against 0.27.1" stamp — they
  apply cleanly to the 0.28.0 tree).
  Upstream has not re-measured its 0.28.0 matrix; the upstream numbers
  quoted in these notes are the 0.27.1 stack's. The one measured delta is
  the #73 fix below: 0.28.0 with `draft_sample_method` reads 3.23 tok/step
  / 121.7 tok/s against 0.27.1's 3.19 / 120.5 (reference 3090, CTX=fast =
  bf16 KV, k=15) — at or above the old base.
- **Upstream sync 1834917..bae2023**: adopted the #86 int64 cast in
  `spec-decode-attn.patch` (verbatim) and `triton-prefill-attn-int8.patch`
  (adopted under its kept name: upstream renamed it to prefill-attn-int8, but
  our alphabetical apply order must keep this patch after
  spec-decode-attn; upstream's relies on their patches/series file for that). Not
  adopted: `offload-mtp-serve.patch` (CPU KV tier under
  MTP; our recipes run no connector -- upstream measured 96 -> 102 tok/s
  at CTX=long with it), `offload-wsl2-devptr.patch` (WSL2 out of scope),
  `triton-spec-attn-fp8-kv.patch` (sm89+; our 3090s are sm86),
  `sse-keep-alive` / `engine-completion-log` / `engine-stall-sentinel`
  (observability, non-MTP; candidates for a later sync),
  `int4-mq3d-envs.patch` (our int4 lane reads os.environ directly -- see
  its patch header).
- **`draft_sample_method` is required on 0.28.0** (upstream #73): the
  native speculator base allocates the draft-logits buffer only when the
  speculative config asks for it; without it the rejection test loses its
  denominator and acceptance drops ~16%. The dflash2 recipes set it
  (`"probabilistic"`); if you assemble a speculative config by hand, do
  not omit it. The MTP recipe has always set it.
- **Two opt-in DFlash2 extensions ship with the set**, both off by default
  (the recipes do not set them): `VLLM_DFLASH2_LOOKUP=1` (lookup-augmented
  drafting) and, on top of it, `VLLM_DFLASH2_CHAIN=1` (drafter-free n-gram
  chains; greedy requests by default — at temperature the chain's
  point-mass proposals lose to the drafter's). Upstream measures +7% on
  the "reproduce a 25k-token document" cell with chains (256.9 → 276
  tok/s), flat on prose. Both were measured upstream on a single card;
  lookup is untested on TP=2 — check acceptance and output before relying
  on it (drop either into `.env` to try).
- **DSpark drafter (w4a16-int8-dspark, fetched in every setup)**: vLLM 0.28.0 has a native
  `dspark` speculative method, and the community checkpoint
  (RadixArk/Qwen3.8-27B-DSpark, bf16, 1.86 B params / ~3.7 GB, 7 drafts per
  step) serves on our int8-KV stack once you dodge two traps -- both handled
  for you by `prepare/fetch_dspark.py` and
  `patches/dspark-draft-quant-config.patch`: the loader refuses a draft that
  carries no quantization config of its own, and the published
  `DSparkDraftModel` architecture name maps to the DeepSeek V4 class (the
  fetch installs a copy renamed to `Qwen3DSparkModel` -- weights hard-linked,
  everything else untouched). Upstream measured it LOSING to the shipped
  dflash2 head on the same requests -- 3.37 vs 3.83 tokens/step, 114 vs 150
  tok/s on a 3090; 3.19 vs 3.77, 115 vs 160 on a 4090 (their bf16-KV lane,
  single card; upstream: "documented so nobody re-derives the two errors, not
  as a recommendation") -- and that a drafter this size does not fit a pinned
  24 GB budget at any context; upstream measured it serving unpinned at util
  0.97 with an 8192 max len on their 3090 (and a pinned ~3 GB KV profile on
  their 4090). This recipe keeps our standard
  0.93 / native max len for shape consistency; if it fails to boot on 24 GB
  cards, that is why (the recipe header says what to change). The TP=2 +
  int8 KV cell is ours to measure -- compare acceptance and
  perplexity/GSM8K against w4a16-int8-dflash2 before trusting it.
  `./setup.py` (or the container's w4a16-int8-dspark arm) fetches the
  checkpoint by default (`DSPARK=/path` redirects, `=0` skips it).
- **The ukisai Swift variant (opt-in: `SWIFT=1` in either prep; `SWIFT=/dir`
  redirects)**: ukisai/Swift-Qwen3.8-27b is a reasoning-efficiency fine-tune
  of the base model -- their headline is ~58% fewer thinking tokens at <1%
  accuracy cost on their benches (served bf16 on big boxes).
  `jamesbrunet/Swift-Qwen3.8-27b-W4A16-AutoRound` (ungated, ~20 GB) is its
  W4A16 AutoRound quant, and `prepare/build_swift_model.py` turns it into a
  servable dir with the same operations the fast model gets -- round-to-
  nearest int8 group-128 for `lm_head` (~1.3 GB freed), `embed_tokens`
  (~1.3 GB) and the MTP module (~0.4 GB; the published config would
  otherwise refuse any speculative load) -- plus the froggeric template; one
  CPU pass, ~8 GB RAM. What it is *not* yet: the GPTQ-calibrated int4 heads
  and the 40k MTP draft head need the upstream drafter/ training pipeline
  run against this fine-tune (deferred) -- until then the MTP recipe runs
  the native full-vocab head, and the dflash2/dspark drafters (trained on
  the base model) are unmeasured on it: compare acceptance against the base
  before trusting them. UkisAI's Swift Open License v1.0 (free up to $1M
  revenue, enterprise above) is restrictive, so the dir is fetched and
  built, never committed. Serve:
  `MODEL=models/Qwen3.8-27B-Swift-W4A16 bash recipes/w4a16-int8-mtp.sh`
  (any recipe loads the dir).
- **Multi-turn prefix caching**: `VLLM_MAMBA_ALIGN_KEEP_CHECKPOINTS`
  (on by default in the recipes — the patch ships it off; set it to 0
  to opt out) keeps the mamba state snapshots a conversation's hits
  resume from alive until the request ends. The failure this fixes:
  upstream #47 saw a ~44 s TTFT spike every 4-5 turns at 44k context —
  the mamba group's hit vanishing while the attention group stayed 100%
  cached; the fix is #52 (vllm#45238), whose own repro measured
  2.1 s → 17-31 s on a shrunk pool.
  Diagnose any prefix-cache miss with
  `usage.prompt_tokens_details.cached_tokens` (upstream's gotcha file, entry 39
  after its 2026-09 renumbering audit: a clean 0 after a hit means a full
  recompute).
- **`VLLM_PREFILL_ATTN=int8`** (off by default): int8-QK prefill attention
  for the head_dim-256 layers — 1.27-1.35x on the kernel, +0.4% rising to
  +5.3% end-to-end prefill upstream, but only with bf16 KV, so it applies
  to w4a16-bf16-dflash2 here; on its own it is "within a few percent
  either way" (upstream #62), the gain compounds with the int8-GEMM lane.
- **`VLLM_MARLIN_TUNE=1`** (off by default; we leave it off): routes
  `marlin_gemm` through a separately built tunable Marlin extension;
  a no-op without that build. We are not building it: upstream measured
  +2-20% per GEMM on W4A8 prefill, but ~+0.4% end-to-end at the 250 W
  cap (and the build's source tree is in neither git repo), so stock
  is the right default on this hardware. Uncapped cards would
  additionally gain +3-7% on the M≤16 verify GEMMs.
- **TP=2**: upstream measured +16–35% decode at C1 vs one 3090 (PCIe x8,
  no NVLink); DFlash2 wins at every concurrency on two cards, and the
  15-draft block lost 27% at TP=2 — keep 7.
- **Chat template**: the prep replaces the stock template with
  froggeric's v22.5 (fixed thinking, tool calls, and agentic behavior
  across Qwen 3.5/3.6/3.8); per-request variables — `enable_thinking`,
  `reasoning_effort`, `tool_call_format`, … — go through
  `chat_template_kwargs`.
- First start compiles (torch.compile, CUDA graphs); the caches live in
  `$HOME` (the `/cache` volume in Docker), so it happens once.
- **How to measure acceptance on this stack** (upstream's 0.28.0 acceptance
  bisect left five rules, upstream gotchas 48-52): a benchmark row without
  its compile-cache state is not reproducible -- six GDN kernels autotune at
  first use and the winners pick the trajectory (one image, eight cold boots:
  eight winner sets, five trajectories, 58-189 tool calls on the same
  twelve-turn conversation), and the cache state moves the profiled peak and
  hence the KV pool by 0.92 GiB; a high acceptance rate can mean the
  generation has *collapsed* (one repetition loop read 79.7% acceptance
  against a normal 35-45% -- guard on drafted tokens per round above ~1.2x
  the drafter width, or on distinct-word ratio); per-request acceptance is
  order-dependent (the long block is sticky) -- report **tokens per step =
  1 + accepted/drafted** (the accepted drafts plus the bonus token) and treat
  per-request figures from a sequence as dependent samples, never independent;
  at DFLASH_TOKENS=15 with lookup on, ordinary text runs the trained 7-draft
  block anyway (any production log shows it: per-position acceptance ~0.8-0.96
  over the first seven positions, then a flat ~0.15 tail of eight); and two
  passes with a fixed seed are two replays -- the #73 16% acceptance
  regression was invisible on a 256-token, 30-seed cell and obvious on the
  1024-token cohort, so match the workload to the claim.
- Sampling: Qwen recommends 0.7 / top_p 0.8 for instruct, 1.0 / 0.95 with
  thinking (the default).

## License

Apache-2.0, same as the model; the patch stack and model assets carry the
upstream repo's terms (see its LICENSE and the per-patch headers).
