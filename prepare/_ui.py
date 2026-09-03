"""Shared uv-style output for the scripts.

The look: verb-led lines (the first letter of every line is capital),
`+` / `·` / `×` markers, counts and elapsed time, and bold stage
headers that group the sub-lines under them. Long-running work shows an
in-place progress line (a uv-style bar for measured work, a spinner
for unmeasured work); the finished or failed status overwrites that
line -- the terminal's erase-the-whole-line escape clears it, so no
trailing characters survive. Foreign command output (uv) is re-emitted
six spaces in, as sub-output of the stage that invoked it.

TTY + NO_COLOR aware, as the scripts' output contract: NO_COLOR kills
the color; the in-place updating needs a TTY, and elsewhere everything
degrades to one line per event (an in-progress line, then its final
line), so a piped or captured run stays readable and greppable.
"""

import os
import subprocess
import sys
import threading
import time

IS_TTY = sys.stdout.isatty()
COLOR = IS_TTY and not os.environ.get("NO_COLOR")

_ERASE_LINE = "\x1b[2K"  # vt100: erase the whole line (no leftover tail)


def c(code, s):
    return f"\033[{code}m{s}\033[0m" if COLOR else s


def dim(s):
    return c("2", s)


def bold(s):
    return c("1", s)


def green(s):
    return c("32", s)


def red(s):
    return c("31", s)


def _cap(s):
    """The house rule: a line that starts with a letter starts capital.
    Markers, digits and symbols lead untouched."""
    if s and s[0].isalpha() and s[0].islower():
        return s[0].upper() + s[1:]
    return s


# --- the line vocabulary ------------------------------------------------


def stage(title):
    """A stage header: bold, at the top indent; its sub-lines follow."""
    print(bold(f"  {_cap(title)}"))


def note(s):
    """A quiet sub-line: already done, already skipped, a detail."""
    print(dim(f"    · {_cap(s)}"))


def ok(s):
    """A sub-line that finished well."""
    print(green(f"    + {_cap(s)}"))


def done(s):
    """Top-level success, one per script run."""
    print(green(f"  \u2713 {_cap(s)}"))


def _hints(hints, pad):
    cont = " " * (len(pad) + 6)
    for i, h in enumerate(hints):
        if i == 0:
            print(dim(f"{pad}  \u2570\u2500> {_cap(h)}"))
        else:
            print(dim(f"{cont}{_cap(h)}"))


def fail(msg, *hints):
    """Top-level failure; exits 1."""
    print(red(f"  \u00d7 {_cap(msg)}"))
    _hints(hints, "  ")
    sys.exit(1)


# --- numbers ------------------------------------------------------------

def dur(secs):
    secs = max(0.0, float(secs))
    if secs < 1:
        return f"{secs * 1000:.0f}ms"
    if secs < 60:
        return f"{secs:.1f}s"
    m, s = divmod(int(secs), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def human(n):
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


# --- foreign commands ----------------------------------------------------

def run_indented(cmd):
    """Run a foreign command (uv, ...) and re-emit its output six
    spaces in, as sub-output of the current stage. The child sees a
    pipe, so TTY-aware tools drop their own layout and print plain
    lines -- which stay readable under the indent. Returns the exit
    code. Do not run while a live Progress bar is active."""
    proc = subprocess.Popen([str(x) for x in cmd], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")

    def _pump():
        for line in proc.stdout:
            sys.stdout.write("      " + line)
            sys.stdout.flush()

    t = threading.Thread(target=_pump, daemon=True)
    t.start()
    rc = proc.wait()
    t.join(timeout=5)
    return rc


# --- the progress line ---------------------------------------------------

_SPIN = "\u280b\u2819\u2839\u2838\u283c\u2834\u2866\u2867\u2807\u280f"
_SHOW_AFTER = 0.25     # seconds before the live line is worth showing
_DRAW_HZ = 10          # how often the live line redraws

_noop = lambda *a, **k: None


def _term_width(default=80):
    try:
        import fcntl
        import termios
        return fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ)[1] or default
    except Exception:
        return default


class Progress:
    """One in-place progress line for a long-running step.

    Call tick(delta) as work completes (bytes, files, ...); finish()
    overwrites the live line with the final `+` (or `×`) line on a TTY,
    and prints one line per event elsewhere. A step that finishes before
    _SHOW_AFTER never shows the live line at all -- no flash of a bar
    for sub-250ms work."""

    def __init__(self, label, total=None):
        self.label = _cap(label)
        self.total = float(total) if total is not None else None
        self.n = 0.0
        self.files_total = None
        self.files_done = 0
        self.t0 = time.monotonic()
        w = _term_width()
        self._bar_w = 16 if w >= 100 else max(8, min(16, w - 48))
        self._shown = False
        self._last = 0.0
        self._finished = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    # updates (thread-safe: HF feeds the download bars from a pool)

    def tick(self, delta=1):
        with self._lock:
            if self._finished:
                return
            self.n += delta
            self._draw()

    def add_total(self, delta):
        with self._lock:
            if self._finished or not delta:
                return
            self.total = (self.total or 0.0) + delta
            self._draw()

    def set_files(self, done, total=None):
        with self._lock:
            if self._finished:
                return
            self.files_done = done
            if total is not None:
                self.files_total = total
            self._draw()

    def draw(self):
        with self._lock:
            self._draw(force=True)

    def finish(self, ok=True, text=None, *hints, fatal=False):
        """Overwrite the live line with the final status line.

        With fatal, exits 1 after printing a failure and its hints."""
        with self._lock:
            self._finished = True
            clear = IS_TTY and self._shown
        self._stop.set()
        if clear:
            sys.stdout.write("\r" + _ERASE_LINE + "\r")
            sys.stdout.flush()
        if ok:
            print(green(f"    + {_cap(text or self.label)}"))
        else:
            print(red(f"    \u00d7 {_cap(text or self.label)}"))
            _hints(hints, "    ")
            if fatal:
                sys.exit(1)

    # rendering

    def _render(self):
        """(plain length, colored line) of the live progress line."""
        elapsed = time.monotonic() - self.t0
        rate = self.n / elapsed if elapsed > 1e-3 else 0.0
        bw = self._bar_w
        if self.total and self.total > 0 and self.n > 0:
            frac = min(1.0, self.n / self.total)
            fill = round(frac * bw)
            bar_p = "\u2588" * fill + "\u2591" * (bw - fill)
            bar_c = green("\u2588" * fill) + dim("\u2591" * (bw - fill))
            pct = f"{frac * 100:3.0f}%"
            meas = f"{human(self.n)}/{human(self.total)}"
            if rate:
                meas += f"  {human(rate)}/s"
        else:
            ch = _SPIN[int(elapsed * 10) % len(_SPIN)]
            bar_p, bar_c = ch, ch
            pct = ""
            meas = human(self.n) if self.n else ""
            if rate:
                meas += f"  {human(rate)}/s"
        extra = ""
        if self.files_total:
            extra = f"{self.files_done}/{self.files_total} files"
        parts_p = [bar_p]
        parts_c = [bar_c + " " * max(0, bw - len(bar_p))]
        for piece in (pct, meas, extra):
            if piece:
                parts_p.append(piece)
                parts_c.append(piece)
        tail = f"({dur(elapsed)})"
        line_p = "    \u00b7 " + self.label + "  "
        line_c = "    \u00b7 " + self.label + "  "
        for p_, c_ in zip(parts_p, parts_c):
            line_p += p_ + "  "
            line_c += c_ + "  "
        return len(line_p) + len(tail), line_c + tail

    def _spin(self):
        """Keep a step visibly moving even without ticks (the spinner
        case), redrawing at most 10 Hz until finish()."""
        self._stop.wait(_SHOW_AFTER)
        while not self._stop.wait(1 / _DRAW_HZ):
            with self._lock:
                self._draw(force=True)

    def _draw(self, force=False):
        if self._finished:
            return
        if not IS_TTY:
            if not self._shown and self.n > 0:
                print(f"    \u00b7 {self.label}")
                self._shown = True
            return
        now = time.monotonic()
        if not self._shown:
            if now - self.t0 < _SHOW_AFTER:
                return
            self._shown = True
        if not force and now - self._last < 1 / _DRAW_HZ:
            return
        self._last = now
        line = self._render()[1]
        sys.stdout.write("\r" + _ERASE_LINE + line + "\r")
        sys.stdout.flush()


# --- Hugging Face snapshot downloads -------------------------------------

def snapshot(repo, progress=None, **kw):
    """snapshot_download with this module's bar instead of its tqdm.

    Tries the local cache first: a warm cache resolves silently and
    instantly. The network pass routes all of huggingface_hub 1.x's
    internal bars into `progress` (a Progress)."""
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo, local_files_only=True, **kw)
    except Exception:
        if progress is None:
            return snapshot_download(repo, **kw)
        return snapshot_download(repo, tqdm_class=_tqdm_bridge(progress), **kw)


def _tqdm_bridge(progress):
    """The tqdm stand-in snapshot_download instantiates for its three
    internal bars: the bytes written into the cache (the one displayed),
    the network bytes (its denominator is estimated, so ignored), and
    the per-file counter (shown as the line's extra).

    Built for huggingface_hub 1.27: the two byte bars come through
    _create_progress_bar, which drops the `name` it is handed for custom
    classes -- so they are told apart by desc -- plus a `total` that is
    assigned by attribute as files are planned, and the files bar straight
    from hf_thread_map (desc plus total, no unit)."""
    lock = threading.Lock()

    class Bar:
        def __init__(self, *, name=None, desc=None, total=None, initial=0,
                     unit=None, unit_scale=False, bar_format=None, **_):
            self._n = 0
            self._total_lock = threading.Lock()
            self._total = None if total is None else total
            self.format_dict = {}
            if (name and name.endswith(".transfer")) or desc == "Downloading bytes":
                self._kind = "transfer"
            elif unit == "B" or desc is None:
                self._kind = "bytes"
            elif total:
                self._kind = "files"
                progress.set_files(0, total)
            else:
                self._kind = "other"
            if self._kind == "bytes":
                if self._total:
                    progress.add_total(self._total)
                if initial:
                    progress.tick(initial)

        def __getattr__(self, attr):
            # whatever else the hub library reaches for: a harmless no-op,
            # so a newer huggingface_hub cannot crash the download
            return _noop

        @property
        def n(self):
            return self._n

        @property
        def total(self):
            with self._total_lock:
                return self._total

        @total.setter
        def total(self, v):
            if v is None:
                return
            # the hub computes v = (bar.total or 0) + file_size and assigns
            # it from worker threads; read and publish atomically so two
            # concurrent plannings cannot drop a delta
            with self._total_lock:
                prev = self._total or 0
                self._total = v
            if self._kind == "bytes" and v > prev:
                progress.add_total(v - prev)

        def update(self, n=1):
            n = int(n or 0)
            if not n:
                return
            with lock:
                self._n += n
                if self._kind == "bytes":
                    progress.tick(n)
                elif self._kind == "files":
                    progress.set_files(self._n)

        def set_postfix_str(self, s, refresh=False):
            pass

        def set_description(self, s):
            pass

        def refresh(self):
            progress.draw()

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    return Bar
