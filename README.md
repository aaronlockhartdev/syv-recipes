# syv-recipes — Qwen3.8-27B serving for a two-GPU box

A fork of [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090),
specialized to one machine shape: **two 24 GB GPUs (tensor-parallel 2)**,
serving Qwen3.8-27B with vLLM 0.27.1 and its custom patch stack. Everything
not needed for the two recipes below is removed; the recipes call `vllm
serve` with explicit flags — no CTX/SPEC-style configuration to construct.

| recipe | speculation | KV cache | the upstream config it matches |
|---|---|---|---|
| [recipes/dflash2.sh](recipes/dflash2.sh) | DFlash2 block drafter, 7 drafts in one pass | `int8_per_token_head`, prefix caching, native max len, 4 seqs | `SPEC=dflash2 CTX=long PREFIX_CACHE=1`, `--tensor-parallel-size 2` |
| [recipes/mtp.sh](recipes/mtp.sh) | Qwen's MTP head, 3 drafts, probabilistic | `int8_per_token_head`, prefix caching, native max len, 8 seqs | `SPEC=mtp CTX=long PREFIX_CACHE=1`, `--tensor-parallel-size 2`, KV swapped fp8 → int8 |

Both recipes take image input (no `--language-model-only`): up to 16
images per request, each capped at 2097152 px = 2048 tokens. The per-image
cap (not the count) sets the encoder's profiled peak, which comes out of
the KV pool; the count only bounds per-request context. The ~0.9 GB vision
tower is offloaded to pinned host RAM by default (`VLLM_VISION_CPU_OFFLOAD_GB=1`;
`=0` keeps it GPU-resident). They also use `--enable-prefix-caching` with
`--prefix-caching-hash-algo xxhash` (the `xxhash` package is in
requirements.txt), `--gpu-memory-utilization 0.93` (under TP>1 the pool is
sized from utilization), `--mamba-ssm-cache-dtype float16` (halves the GDN
state cost), `--max-num-batched-tokens 4096`, the qwen3 reasoning parser,
and qwen3_coder tool parsing. The mtp recipe additionally forces
`cudagraph_mode=PIECEWISE` — see its header. Port 8080 (`PORT=…`) and a
`.venv` at the repo root (`MODEL`, `DRAFT` overridable the same way).

## Layout

```
Dockerfile          uv + pinned vLLM 0.27.1 + every patch in patches/
requirements.txt    the pinned set (vllm pulls torch 2.13 / flashinfer itself)
setup.sh            bare-metal one-shot: venv, deps, patches, model prep
patch-vllm.sh       applies patches/ to the installed vllm (bare metal)
recipes/            dflash2.sh, mtp.sh — the two ways to serve
prepare/            build_fast_model.py, fetch_dflash2.py — one-time model prep
patches/            the 17 kept vLLM patches (below)
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
  --restart unless-stopped syv-recipes            # entrypoint default: dflash2
```

The entrypoint prepares the models on first start (downloads through the
mounted hub cache — seconds when it is warm; `qwen-cache` holds everything
else), then execs the recipe. `qwen-models` receives the assembled dirs:
hard-linked off the cache when both volumes share a filesystem, a second
~21 GB copy when they don't. `syv-recipes mtp` runs the other one;
`syv-recipes prepare` runs only the prep; `PREPARE=0` skips it.
`VLLM_API_KEY=…` turns on key auth; without it the server binds 0.0.0.0
and is open.

The `-v "$HUB":…` mount is a run-time one (docker build cannot see host
directories, and the cache is consumed at run time by the prep anyway);
the container side is pinned in the Dockerfile. Caveats: the container
runs as root, so files it downloads into your host cache are root-owned;
on Docker Desktop the mount crosses virtiofs, so the prep's hard-links
fall back to a full copy.

## Bare metal (uv)

```bash
bash setup.sh            # venv + pinned deps + the 17 patches + both models
bash recipes/dflash2.sh  # or recipes/mtp.sh
```

`setup.sh` is idempotent and resumable — re-running it is the recovery
path after a venv wipe or a stale-patch abort. Paths are overridable with
env vars of the same names the recipes use (defaults: `.venv`,
`models/Qwen3.8-27B-W4A16-AutoRound-fast`,
`models/Qwen3.8-27B-DFlash2-W4A16`), so a custom layout stays consistent
between setup and serve:

```bash
MODEL=/data/qwen VENV=/data/qwen/.venv bash setup.sh
MODEL=/data/qwen bash recipes/dflash2.sh
```

What `setup.sh` runs, in order, if you'd rather do it by hand (install uv
first, and `patch` if your distro lacks it:
`curl -LsSf https://astral.sh/uv/install.sh | sh`):

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
bash patch-vllm.sh
.venv/bin/python prepare/build_fast_model.py models/Qwen3.8-27B-W4A16-AutoRound-fast
.venv/bin/python prepare/fetch_dflash2.py        models/Qwen3.8-27B-DFlash2-W4A16
```

The recipes put `./.venv/bin` on PATH and default to port 8080; `MODEL`,
`DRAFT` and `PORT` may be overridden with env vars of the same names.

## The kept patches (all written against vLLM 0.27.1)

- `dflash2-backport.patch` — DFlash2 drafter (vLLM PR #52816) on 0.27.1, incl. the V2 model-runner speculator
- `hybrid-kv-groups-v2-cudagraph.patch` — KV-group sizing for the drafter's sliding-window layers; explicit CUDA-graph memory accounting (`VLLM_V2_CUDAGRAPH_MEM_MIB`)
- `hybrid-sw-block-promote.patch` — lets a quantized KV cache fit in a hybrid target+drafter (block-size promotion instead of page padding)
- `int4-kv-per-token-head.patch` — boot blockers for int4 per-token-head KV with the drafter
- `marlin-int8-layer-select.patch` — env-selectable int8-activation layers for the Marlin path
- `marlin-int8-negative-scales.patch` — correctness fix for negative group scales in W4A8
- `marlin-repack-staged-sm80.patch` — staged Marlin repack (sm80 VMM-wedge candidate)
- `offload-dflash-eagle-groups.patch` — OffloadingConnector group flagging under dflash
- `qwen3_5-embed-quant.patch` — route the embedding table through the quantized path (the fast model needs it)
- `qwen3_5-mtp-draft-vocab.patch` — vocab-truncated MTP draft head (the fast model's 40k draft head)
- `sampler-small-topk-fast-softmax.patch` — sort-free top-k/top-p, multi-block softmax, truncated drafts
- `spec-decode-attn.patch` — split-KV verify attention (`VLLM_SPEC_DECODE_ATTN`)
- `spec-decode-int8-kv.patch` — teaches the split-KV kernel to read the int8 per-token-head cache
- `speed-knobs-envs.patch` — registers the speed knobs as env vars (torch.compile cache key)
- `vision-tower-cpu-offload.patch` — vision tower in pinned host RAM (`VLLM_VISION_CPU_OFFLOAD_GB`)
- `vllm-pr50021-gdn-spec-bounds.patch` — bounds on accepted-token state lookups in the GDN/Mamba spec kernels
- `xgrammar-spec-terminated.patch` — structured output survives tokens accepted past the grammar's end

Removed from the upstream stack: KVarN (4/2-bit KV), lookup-augmented
drafting, and n-gram chains — the DFlash2 checkpoint proposes its trained
7 tokens, and the verify block stays 8.

## Notes

- **int8 KV is a trade**: ~2x the pool of bf16, at the cost of the Triton
  backend and a per-step unpack; its quality at depth was never measured
  upstream. Verify perplexity/GSM8K on your workload before trusting it.
- **MTP + split-KV verify**: a configuration upstream never measured (see
  the mtp header); check draft acceptance on your workload.
- **TP=2**: upstream measured +16–35% decode at C1 vs one 3090 (PCIe x8,
  no NVLink); DFlash2 wins at every concurrency on two cards, and the
  15-draft block lost 27% at TP=2 — keep 7.
- **Chat template**: the prep replaces the stock template with Qwen-Sharp
  v22.4.0 (token-efficient thinking and tool calls); per-request variables
  — `enable_thinking`, `reasoning_effort`, `tool_call_format`, … — go
  through `chat_template_kwargs`.
- First start compiles (torch.compile, CUDA graphs); the caches live in
  `$HOME` (the `/cache` volume in Docker), so it happens once.
- Sampling: Qwen recommends 0.7 / top_p 0.8 for instruct, 1.0 / 0.95 with
  thinking (the default).

## License

Apache-2.0, same as the model; the patch stack and model assets carry the
upstream repo's terms (see its LICENSE and the per-patch headers).
