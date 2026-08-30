#!/usr/bin/env python3
"""Fetch the W4A16 DFlash2 block drafter into a destination directory.

The int4-GPTQ requantization of incoai/Qwen3.8-27B-DFlash2 (1.2 GB vs the
3.85 GB bf16 original), prebuilt on the Hub. Downloaded through Hugging
Face's built-in cache, then linked into the destination dir.

Usage:  python prepare/fetch_dflash2.py DEST_DIR

Idempotent.
"""

import os
import shutil
import sys

from huggingface_hub import snapshot_download

REPO = "syvai/Qwen3.8-27B-DFlash2-W4A16"
FILES = ("config.json", "model.safetensors")


def _snapshot(repo):
    # local-first: a warm cache resolves without any network (a blackholed
    # network must not stall a model-ready boot); the first run downloads
    try:
        return snapshot_download(repo, local_files_only=True)
    except Exception:
        return snapshot_download(repo)


def main():
    if len(sys.argv) != 2 or not sys.argv[1].strip():
        sys.exit("usage: python prepare/fetch_dflash2.py DEST_DIR")
    dst = os.path.abspath(sys.argv[1])
    os.makedirs(dst, exist_ok=True)

    hub = _snapshot(REPO)
    for f in FILES:
        src = os.path.realpath(os.path.join(hub, f))
        dstp = os.path.join(dst, f)
        if os.path.isfile(dstp) and os.path.getsize(dstp) == os.path.getsize(src):
            continue
        if os.path.lexists(dstp):
            os.remove(dstp)
        try:
            os.link(src, dstp)  # hard-link where possible; copy across devices
        except OSError:
            shutil.copy(src, dstp)
        print(f"== {f}: {dstp}")
    print("DFlash2 drafter ready:", dst)
    print("serve with: recipes/dflash2.sh (this dir as DRAFT)")


if __name__ == "__main__":
    main()
