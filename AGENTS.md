# AGENTS.md

## What this repo is

`syv-recipes` is a restructuring and UX improvement of the work done in
`syv-ai/qwen38-27b-rtx3090` (fetched into this repo's `main` branch): the
same vLLM patch stack and models, re-expressed for a 2xRTX-3090 (TP=2,
2x24 GB) box as explicit single-command recipes with uv-based setup and a
one-layer Docker image.

**Direction of authority.** Upstream is the technical source of truth —
the patch set, its measured numbers, correctness gotchas, and commit
history all live there and are *synced into this repo* (into `patches/`,
into comment claims, into this file). This repo owns the deployment layer:
which recipes exist, their default flag values, naming, scripts, and
packaging. A change here that conflicts with an upstream technical fact
needs upstream evidence for it, or an explicit "deviation from upstream"
disclosure in the recipe header (there are three known ones, all disclosed).

## The five recipes

| recipe | stack | one-liner |
|---|---|---|
| `w4a16-int8-dflash2` | TRITON_ATTN, int8 KV, split-KV verify, dflash2 drafter | the fast default |
| `w4a16-int8-mtp` | TRITON_ATTN, int8 KV, MTP head, PIECEWISE graphs | no separate drafter |
| `w4a16-bf16-dflash2` | FLASH_ATTN, bf16 KV, dflash2 | the unquantized quality baseline |
| `w4a16-int4-dflash2` | TRITON_ATTN, int4 KV, dflash2, `--prefix-match-unit 848` | ~2x the context capacity |
| `w4a8-int8-dflash2` | dflash2 + W4A8 Marlin linears (INT8_LAYERS) | faster prefill, documented quality cost |

Naming scheme: `[model_quant]-[kv]-[spec_decode_method]`. Do not break it.
Adding a recipe = a new file following the scheme, an arm in
`docker/entrypoint.sh`, a row in the README table (and a matrix row if the
combination is new to the matrix).

## Invariants

- **`patches/` is the synced set (19).** They apply in alphabetical = build
  order because later patches depend on files created by earlier ones. New
  patches are taken from upstream with provenance noted in the header
  (e.g. "upstream bb739e4"). Never hand-edit vLLM; `prepare/patch_vllm.py`
  is the only thing that touches the installed tree, and it attests the
  result with a 3-field stamp (version + patch-set fingerprint + tree
  digest).
- **MTP must keep `cudagraph_mode: PIECEWISE`** — the default (FULL)
  corrupts one prompt length in 128 (residue `k+1`) under prefix-cache hits.
- **int4 KV must keep `--prefix-match-unit 848`** (drafter sliding-window
  block vs hash unit) or the prefix cache can never match.
- **fp8 KV is excluded** — deterministic Xid-31 on 3090-class (upstream
  issue #34). That is also why our MTP recipe is int8 KV rather than
  upstream's fp8/FlashInfer lane.
- **No WSL2 support** anywhere in this repo (upstream had a whole WSL2
  lane; it is out of scope here). Likewise no env vars for upstream
  features this repo deliberately dropped (KVarN, dflash2-lookup-drafting,
  dflash2-ngram-chains).
- **Vision is on** (no `--language-model-only`) — two 24 GB cards are not
  VRAM-limited; the tower offloads to pinned host RAM by default
  (`VLLM_VISION_CPU_OFFLOAD_GB=1`) since dflash2 + vision OOMs at graph
  capture on 24 GB without it.
- **Empty means unset, everywhere**: bash `${VAR:-default}`, Python
  `os.environ.get(k) or default`, and the `.env` loader all treat an
  empty string as absent. `.env` (repo root, gitignored) only fills
  variables that are unset-or-empty in the real environment, which always
  wins.
- **Recipe scripts are data, not logic**: the whole server configuration is
  inlined in the one `exec vllm serve` line; env vars are only what the
  patch stack needs (and their defaults). No config branching. No comment
  lines inside backslash continuation chains.
- **All scripts are idempotent** and print in uv's style (verb-led lines,
  `+` / `·` / `×`, counts, elapsed time, TTY + NO_COLOR aware).
- **No unmeasured numbers**: the README matrix's measurement columns
  (decode, prefill 16k, quality) are the user's to fill on this hardware.
  Numbers taken from upstream are single-card or a different shape; they
  must be labeled as upstream measurements, never presented as ours.

## Layout

```
Dockerfile  requirements.txt  setup.py  README.md
docker/  entrypoint.sh, prepare.sh
recipes/ the five *.sh
prepare/ build_fast_model.py, fetch_dflash2.py, patch_vllm.py
patches/ the 19 synced patches
```

Defaults: venv `.venv/`, models under `models/`, port 8080, and `Qwen3.8-
27B` is served as `qwen3.8-27b` in every recipe (clients are pinned to that
name).

## Syncing with upstream

`main` tracks the `upstream` remote (`syv-ai/qwen38-27b-rtx3090`,
fast-forward only — keep it clean of local commits). The `recipes` branch
does not merge main; sync is a manual diff:

1. On main: `git fetch upstream && git merge upstream/main`.
2. Read the new upstream commits; the useful artifacts are `patches/*.patch`,
   `single-user/` and `batch/` launchers (the flag/lanes source), `docs/`
   (measured numbers and gotchas), and `prepare/` scripts.
3. Adopt relevant patches into `patches/` and launcher knowledge into the
   recipes; note upstream provenance + measured deltas in comments.
4. Validate: the full set applies in order on a pristine Linux vllm 0.27.1
   and compiles — `prepare/patch_vllm.py` does exactly this and is
   re-run on every setup/serve.

Caveats: on this Mac, the macOS vllm wheel is shorter than the Linux one
and hunks in some patches don't match, so patch validation must happen in
a Linux venv (the Docker build does it). Use `gpatch` on macOS; the
system `patch` is netBSD-derived and incompatible.

## Before calling a change done

- `prepare/patch_vllm.py` clean on the target venv (it self-heals or
  resets; a stamp is only written after the tree compiles)
- `bash -n` on every recipe; `python3 -m py_compile` on every script
- a reviewer pass on non-trivial staged diffs; P0 findings block commit
- no comments introduced inside backslash continuation chains
