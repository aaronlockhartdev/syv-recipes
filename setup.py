#!/usr/bin/env python3
"""One-shot bare-metal setup: venv + pinned deps + the vllm patches +
the three default model dirs (an opt-in Swift variant with SWIFT=1).

Idempotent -- re-run it any time (patch_vllm.py converges whatever state
it finds; the prep scripts are idempotent by design). No positional
args; the destinations come from env vars of the same names the
recipe scripts use (defaults: ./.venv,
models/Qwen3.8-27B-W4A16-AutoRound-fast,
models/Qwen3.8-27B-DFlash2-W4A16,
models/Qwen3.8-27B-DSpark); every setup fetches all three by
default -- DSPARK=/path redirects the DSpark dir, DSPARK=0 skips it.
SWIFT=1 additionally builds the ukisai Swift W4A16 variant (models/
Qwen3.8-27B-Swift-W4A16, ~20 GB); SWIFT=/path redirects it, and an
unset or 0/false/no value skips it.

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
sys.path.insert(0, str(REPO / "prepare"))  # _ui lives with the prep scripts
import _ui as ui


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


def run(label, cmd, *hints, indent=False):
    """Stage a step: bold header, then the child does the work (the
    children are uv-style too). Exit on failure with the step named;
    return the start time. indent re-emits a child's own output (uv)
    six spaces in, as sub-output of this stage."""
    ui.stage(label)
    t0 = time.monotonic()
    if indent:
        rc = ui.run_indented([str(x) for x in cmd])
    else:
        rc = subprocess.run([str(x) for x in cmd]).returncode
    if rc != 0:
        ui.fail(f"{label} failed (exit {rc})", *hints)
    return t0


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    uv = shutil.which("uv")
    if uv is None:
        ui.fail(
            "The uv tool is not on PATH",
            "Install it:  curl -LsSf https://astral.sh/uv/install.sh | sh",
            "Then put ~/.local/bin on your PATH:  "
            'echo \'export PATH="$HOME/.local/bin:$PATH"\' >> ~/.zshrc',
        )

    if PY.is_file():
        ui.note(f"Venv {VENV} already present -- skipping")
    else:
        VENV.parent.mkdir(parents=True, exist_ok=True)
        t0 = run(
            "Creating the venv",
            [uv, "venv", VENV, "--python", "3.14"],
            "No python 3.14?  Install one (e.g.  uv python install 3.14) and re-run",
            indent=True,
        )
        if not PY.is_file():
            ui.fail(f"The venv at {VENV} has no python", f"Delete it and re-run:  rm -rf {VENV}")
        ui.ok(f"Venv {VENV} in {ui.dur(time.monotonic() - t0)}")

    run(
        "Installing the pinned requirements",
        [uv, "pip", "install", "--python", PY, "-r", REPO / "requirements.txt"],
        "On this platform the vllm wheel may not exist -- on bare metal you are "
        "expected to be on Linux with a GPU",
        indent=True,
    )

    run(
        "Patching vllm",
        [PY, REPO / "prepare" / "patch_vllm.py"],
        "The venv is left as found; re-run to converge, or reset it:  "
        "uv pip install --force-reinstall --no-deps vllm==0.28.0",
    )

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    DRAFT.parent.mkdir(parents=True, exist_ok=True)
    run(
        "Building the fast model",
        [PY, REPO / "prepare" / "build_fast_model.py", MODEL],
        "Check the HF download above; the HF cache can be mounted into Docker "
        "(see the README) so a later container run does not re-download",
    )
    run(
        "Fetching the DFlash2 drafter",
        [PY, REPO / "prepare" / "fetch_dflash2.py", DRAFT],
    )

    # fetched by default: DSPARK=/path redirects, DSPARK=0/false/no skips
    dspark = (os.environ.get("DSPARK") or "").strip()
    if dspark not in ("0", "false", "no"):
        dst = (
            REPO / "models" / "Qwen3.8-27B-DSpark"
            if not dspark or dspark in ("1", "true", "yes")
            else Path(dspark)
        ).expanduser()
        dst.parent.mkdir(parents=True, exist_ok=True)
        run(
            "Fetching the DSpark drafter",
            [PY, REPO / "prepare" / "fetch_dspark.py", dst],
        )
    # opt-in: SWIFT=1 builds the Swift W4A16 variant, SWIFT=/path redirects
    swift = (os.environ.get("SWIFT") or "").strip()
    if swift and swift not in ("0", "false", "no"):
        dst = (
            REPO / "models" / "Qwen3.8-27B-Swift-W4A16"
            if swift in ("1", "true", "yes")
            else Path(swift)
        ).expanduser()
        dst.parent.mkdir(parents=True, exist_ok=True)
        run(
            "Building the Swift W4A16 variant",
            [PY, REPO / "prepare" / "build_swift_model.py", dst],
        )
    # some clients (JetBrains AI Assistant) send tool-call arguments as a JSON
    # array instead of an object; harden the templates once for every prepared
    # model (HARDEN_TEMPLATES=0 skips; a template that does not match the
    # known pattern warns and is left alone)
    if (os.environ.get("HARDEN_TEMPLATES") or "1") != "0":
        run(
            "Hardening the chat templates",
            [PY, REPO / "prepare" / "harden_chat_template.py"],
        )
    ui.done(f"Ready -- serve with:  bash {REPO / 'recipes' / 'w4a16-int8-dflash2.sh'}   (or any of recipes/)")


if __name__ == "__main__":
    main()
