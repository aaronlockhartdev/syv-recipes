#!/usr/bin/env python3
"""Fetch the bf16 DSpark drafter into a destination directory.

RadixArk/Qwen3.8-27B-DSpark (1.86 B params, ~3.7 GB bf16), the community
drafter vLLM 0.28.0 serves through its native "dspark" speculative method.
Downloaded through Hugging Face's built-in cache, then installed into the
destination dir. A partial cache (an interrupted download) is completed
rather than re-downloaded; a download that still leaves one of the files
the install needs missing is a hard error naming it.

The published checkpoint names its architecture "DSparkDraftModel", which
vLLM 0.28.0's model registry maps to the DeepSeek V4 DSpark class -- the
Qwen3 class is registered as "Qwen3DSparkModel", and loading the published
name dies on a DeepSeek-only field. So the installed config.json is a copy
with only that one field renamed; everything else is untouched, and the
weights are hard-linked (they are bit-identical). The checkpoint's own
dspark.py/dflash.py remote-code modules are copied alongside, so the
directory stays self-contained; vLLM's registry hit means it uses the
native class and never touches them.

Usage:  python prepare/fetch_dspark.py DEST_DIR

Idempotent.
"""
import json
import os
import shutil
import sys
import time

import _ui as ui

REPO = "RadixArk/Qwen3.8-27B-DSpark"
_PUBLISHED_ARCH = "DSparkDraftModel"   # maps to the DeepSeek V4 class in 0.28.0
_QWEN3_ARCH = "Qwen3DSparkModel"       # the class that serves this checkpoint
# what the install consumes -- the snapshot must actually contain all of it
NEED = ("config.json", "dspark.py", "dflash.py", "model.safetensors")
_BAR_MIN = 8 << 20  # copies above this size run as a progress bar


def _copy(src, dstp, size):
    """Copy one file into the destination; the big ones run as a bar."""
    f = os.path.basename(src)
    if size > _BAR_MIN:
        p = ui.Progress(f"Copying {f} ({ui.human(size)})", total=size)
        with open(src, "rb") as a, open(dstp, "wb") as b:
            while True:
                chunk = a.read(1 << 20)
                if not chunk:
                    break
                b.write(chunk)
                p.tick(len(chunk))
        p.finish(True, f"Copied {f} ({ui.human(size)})")
    else:
        shutil.copy(src, dstp)
        ui.ok(f"Copied {f} ({ui.human(size)})")


def _install_config(hub, dst):
    """Install config.json with the architecture renamed. Returns True if
    written (or just rewritten), False if the destination already holds it."""
    src = os.path.realpath(os.path.join(hub, "config.json"))
    dstp = os.path.join(dst, "config.json")
    with open(src) as f:
        cfg = json.load(f)
    src_archs = cfg.get("architectures")
    if src_archs == [_PUBLISHED_ARCH]:
        cfg["architectures"] = [_QWEN3_ARCH]
    text = json.dumps(cfg, indent=2) + "\n"
    if os.path.isfile(dstp):
        with open(dstp) as f:
            if f.read() == text:
                return False
        os.remove(dstp)
    with open(dstp, "w") as f:
        f.write(text)
    if src_archs == [_PUBLISHED_ARCH]:
        ui.ok(f"Wrote config.json (architecture {_PUBLISHED_ARCH} -> {_QWEN3_ARCH})")
    else:
        ui.ok(f"Wrote config.json (published architecture kept: {src_archs})")
    return True


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    t0 = time.monotonic()

    if len(sys.argv) != 2 or not sys.argv[1].strip():
        sys.exit("Usage: python prepare/fetch_dspark.py DEST_DIR")
    dst = os.path.abspath(sys.argv[1])
    os.makedirs(dst, exist_ok=True)

    ui.stage(f"Fetching {REPO}")
    t_r = time.monotonic()
    p = ui.Progress(f"Fetching {REPO}")
    try:
        hub = ui.snapshot(REPO, progress=p, require=list(NEED))
    except Exception as e:
        p.finish(False, f"Fetching {REPO} failed ({e!r})",
                 "check the network and Hugging Face reachability (a proxy, or HF_HUB_OFFLINE=1, blocks it)",
                 "an offline box can only use a warm cache",
                 fatal=True)
    p.finish(True, f"Fetched {REPO} in {ui.dur(time.monotonic() - t_r)}")

    ui.stage(f"Installing {dst}")
    if not _install_config(hub, dst):
        ui.note("config.json already installed with the right architecture")
    for f in ("dspark.py", "dflash.py", "model.safetensors"):
        src = os.path.realpath(os.path.join(hub, f))
        dstp = os.path.join(dst, f)
        if os.path.isfile(dstp) and os.path.getsize(dstp) == os.path.getsize(src):
            continue
        if os.path.lexists(dstp):
            os.remove(dstp)
        size = os.path.getsize(src)
        try:
            os.link(src, dstp)
        except OSError:
            _copy(src, dstp, size)
            continue
        ui.ok(f"Hard-linked {f} ({ui.human(size)})")

    ui.done(f"DSpark drafter ready: {dst} ({ui.dur(time.monotonic() - t0)})")
    ui.note("Serve with: recipes/w4a16-int8-dspark.sh (this dir as DRAFT)")


if __name__ == "__main__":
    main()
