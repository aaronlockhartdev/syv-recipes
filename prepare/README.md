# prepare/ — one-time model preparation

The two prep scripts take a **mandatory destination directory** and download
through Hugging Face's built-in cache, so reruns and container rebuilds
never re-download.

```bash
python prepare/build_fast_model.py models/Qwen3.8-27B-W4A16-AutoRound-fast
python prepare/fetch_dflash2.py        models/Qwen3.8-27B-DFlash2-W4A16
```

- **build_fast_model.py** — the modified Qwen3.8-27B "fast" model
  (~19.5 GB): the W4A16 AutoRound checkpoint with embed_tokens requantized
  to int8 in place (shard 6, the one local step), the int4-GPTQ lm_head /
  MTP module / 40k draft head overlaid from
  `syvai/qwen3.8-27b-3090-fast-variant`, and the Qwen-Sharp v22.4.0 chat
  template (`peculiar-ragdoll/Qwen-Sharp-Chat-Templates`). Idempotent,
  resumable, ~8 GB peak RAM.
- **fetch_dflash2.py** — the W4A16 DFlash2 block drafter (~1.2 GB).
- **patch_vllm.py** — applies `patches/` to the venv's vllm; idempotent and
  stamped (`VENV` env, default `./.venv`); runs standalone -- a bare
  `./prepare/patch_vllm.py` re-execs into the venv's python.

In the container `docker/prepare.sh` runs both into `/app/models`
automatically before either recipe starts.
