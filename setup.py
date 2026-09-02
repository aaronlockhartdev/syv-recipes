#!/usr/bin/env python3
"""One-shot bare-metal setup: venv + pinned deps + the vllm patches +
both model dirs.

Idempotent -- re-run it any time (patch_vllm.py converges whatever state
it finds; the prep scripts are idempotent by design). No positional
args; the three destinations come from env vars of the same names the
recipe scripts use (defaults: ./.venv,
models/Qwen3.8-27B-W4A16-AutoRound-fast,
models/Qwen3.8-27B-DFlash2-W4A16).

Runs under any python3 (the venv does not exist yet); it shells out to
uv and to the venv's own python for the rest.
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent


_SHELL_SPECIAL = frozenset(
    {"IFS", "GLOBIGNORE", "CDPATH", "BASH_ENV", "ENV", "SHELLOPTS", "PS1",
     "LINENO", "PWD", "OLDPWD", "SECONDS", "RANDOM", "UID", "EUID"})


def _dotenv():
    """Fill unset-or-empty env vars from REPO/.env, with the same contract
    the recipes use: the real environment always wins (a set, non-empty value
    is never touched), values may be quoted, whole-line # comments only, and
    a bare KEY= is skipped (empty means unset, like bash's :-)."""
    p = REPO / ".env"
    try:
        if not p.is_file():
            return
        lines = p.read_text().splitlines()
    except (OSError, UnicodeDecodeError):
        return
    for line in lines:
        line = line.lstrip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        if (
            not k
            or k[0].isdigit()
            or not k.replace("_", "").isalnum()
            or not k.isascii()
            or k in _SHELL_SPECIAL
        ):
            continue
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        if v and not os.environ.get(k):
            os.environ[k] = v


_dotenv()
VENV = Path(os.environ.get("VENV") or REPO / ".venv").expanduser()
MODEL = Path(
    os.environ.get("MODEL") or REPO / "models" / "Qwen3.8-27B-W4A16-AutoRound-fast"
).expanduser()
DRAFT = Path(
    os.environ.get("DRAFT") or REPO / "models" / "Qwen3.8-27B-DFlash2-W4A16"
).expanduser()
PY = VENV / "bin" / "python"

TTY = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def c(code, s):
    return f"\033[{code}m{s}\033[0m" if TTY else s


def dim(s):
    print(c("2", s))


def ok(s):
    print(c("32", f"+ {s}"))


def done(s):
    print(c("32", f"\u2713 {s}"))


def fail(label, *hints):
    print(c("31", f"\u00d7 {label}"))
    for h in hints:
        print(c("2", f"  \u2570\u2500> {h}"))
    sys.exit(1)


def dur(t0):
    d = int((time.monotonic() - t0) * 1000)
    return f"{d}ms" if d < 1000 else f"{d / 1000:.1f}s"


def run(label, cmd, *hints):
    """Run a step inheriting stdio (children print their own lines).
    Exit on failure with the step named; return the start time."""
    t0 = time.monotonic()
    r = subprocess.run([str(x) for x in cmd])
    if r.returncode != 0:
        fail(f"{label} failed (exit {r.returncode})", *hints)
    return t0


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    uv = shutil.which("uv")
    if uv is None:
        fail(
            "uv is not on PATH",
            "install it:  curl -LsSf https://astral.sh/uv/install.sh | sh",
            "then put ~/.local/bin on your PATH:  "
            'echo \'export PATH="$HOME/.local/bin:$PATH"\' >> ~/.zshrc',
        )

    if PY.is_file():
        dim(f"\u00b7 venv {VENV} already present -- skipping")
    else:
        VENV.parent.mkdir(parents=True, exist_ok=True)
        t0 = run(
            "creating the venv",
            [uv, "venv", VENV, "--python", "3.12"],
            f"no python 3.12?  install one (e.g.  brew install python@3.12) and re-run",
        )
        if not PY.is_file():
            fail(f"the venv at {VENV} has no python", f"delete it and re-run:  rm -rf {VENV}")
        ok(f"venv {VENV} ({dur(t0)})")

    run(
        "installing the pinned requirements",
        [uv, "pip", "install", "--python", PY, "-r", REPO / "requirements.txt"],
        "on this platform the vllm wheel may not exist -- on bare metal you are "
        "expected to be on Linux with a GPU",
    )

    run(
        "patching vllm",
        [PY, REPO / "prepare" / "patch_vllm.py"],
        "the venv is left as found; re-run to converge, or reset it:  "
        "uv pip install --force-reinstall --no-deps vllm==0.27.1",
    )

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    DRAFT.parent.mkdir(parents=True, exist_ok=True)
    run(
        "building the fast model",
        [PY, REPO / "prepare" / "build_fast_model.py", MODEL],
        "check the HF download above; the HF cache can be mounted into Docker "
        "(see the README) so a later container run does not re-download",
    )
    run(
        "fetching the DFlash2 drafter",
        [PY, REPO / "prepare" / "fetch_dflash2.py", DRAFT],
    )

    done(f"ready -- serve with:  bash {REPO / 'recipes' / 'w4a16-int8-dflash2.sh'}   (or any of recipes/)")


if __name__ == "__main__":
    main()
