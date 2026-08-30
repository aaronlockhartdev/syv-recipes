# prepare/ — one-time model preparation

Both scripts take a **mandatory destination directory** and download through
Hugging Face's built-in cache (`$HOME/.cache/huggingface`, a volume in the
container), so reruns and container rebuilds never re-download.

```bash
python prepare/build_fast_model.py models/Qwen3.8-27B-W4A16-AutoRound-fast
python prepare/fetch_dflash2.py        models/Qwen3.8-27B-DFlash2-W4A16
```

- **build_fast_model.py** — the modified Qwen3.8-27B "fast" model
  (~19.5 GB): the W4A16 AutoRound checkpoint, with embed_tokens requantized
  to int8 in place (the one local step, in shard 6) and the prebuilt
  int4-GPTQ lm_head / MTP module / 40k draft head overlaid from
  `syvai/qwen3.8-27b-3090-fast-variant`. Idempotent, resumable, ~8 GB peak RAM.
- **fetch_dflash2.py** — the W4A16 DFlash2 block drafter (~1.2 GB) for the
  dflash2 recipe.

In the container `docker/prepare.sh` runs both into `/app/models` automatically
before either recipe starts (and `... prepare` runs them on demand).
