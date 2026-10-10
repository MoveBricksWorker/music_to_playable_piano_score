"""Numba JIT warm-up: turning a multi-minute first call into a one-time cost.

The problem
-----------
``librosa`` 0.11 offloads several helpers to ``numba``.  Numba compiles at *first
call inside each process*, and on Windows that compile takes **tens of seconds to
minutes** of pure CPU.  A cold process therefore appears to hang the first time it
touches ``librosa.stft`` / ``chroma_cqt`` / ``onset_strength`` — no output, one
core pinned, no network activity.

Numba's file cache removes the cost, but only if it is told where to write *before
numba is imported* and if the write actually succeeds.  Two details matter:

* ``NUMBA_CACHE_DIR`` is read at numba import time, so it must be set first.
  :func:`configure` does exactly that and nothing else, which is why the CLI calls
  it before importing any analysis module.
* The default cache location may be unwritable.  :func:`configure` verifies by
  writing a probe file rather than trusting the path.

So the workflow is: :func:`configure` once per process, then either let the first
call pay the compile cost, or run ``e2s warmup`` ahead of time to fill the cache.
:func:`is_warm` reports which situation you are in by consulting numba's own cache
index, and never by running the expensive call itself.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

__all__ = ["configure", "cache_dir", "is_warm", "warm", "WarmReport", "HOT_PATHS"]

# The librosa entry points this project actually calls.  Warming these covers the
# whole pipeline; anything else pays its own compile cost on first use.
HOT_PATHS: tuple[str, ...] = (
    "stft",
    "chroma_stft",
    "chroma_cqt",
    "onset_strength",
    "beat_track",
    "pyin",
)

_PROBE = ".writable"


def _candidates() -> list[Path]:
    """Places we are willing to put a numba cache, best first.

    Order matters.  A per-user cache directory under ``LOCALAPPDATA`` is the
    natural home, but creating it can be denied outright — on a managed or
    sandboxed Windows profile, ``mkdir`` under ``AppData\\Local`` fails with
    ``WinError 5``.  When that happens the cache silently never gets written and
    **every** run pays the full JIT cost again, which is exactly the behaviour that
    makes a cold run look like a hang.  So we probe, and we keep probing until
    something is genuinely writable.
    """
    import tempfile

    roots: list[Path] = []
    explicit = os.environ.get("NUMBA_CACHE_DIR")
    if explicit:
        roots.append(Path(explicit))

    for variable in ("LOCALAPPDATA", "XDG_CACHE_HOME", "APPDATA"):
        base = os.environ.get(variable)
        if base:
            roots.append(Path(base) / "ensemble2score" / "numba-cache")

    roots.append(Path.home() / ".cache" / "ensemble2score" / "numba-cache")
    roots.append(Path(tempfile.gettempdir()) / "ensemble2score-numba-cache")

    # Last resort that is still persistent: a directory next to the working
    # directory.  On a locked-down or sandboxed profile this is often the only
    # writable place that survives between runs, and a cache here is far better
    # than recompiling for minutes on every invocation.
    roots.append(Path.cwd() / ".e2s-cache" / "numba")

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root).lower()
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def _is_writable(target: Path) -> bool:
    """Create ``target`` and prove we can write a file into it."""
    try:
        target.mkdir(parents=True, exist_ok=True)
        probe = target / _PROBE
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


# The directory actually chosen this process, once resolved.
_RESOLVED: Path | None = None


def cache_dir() -> Path:
    """The numba cache directory in use (or the preferred one before configure)."""
    if _RESOLVED is not None:
        return _RESOLVED
    env = os.environ.get("NUMBA_CACHE_DIR")
    if env:
        return Path(env)
    candidates = _candidates()
    return candidates[0] if candidates else Path(".numba-cache")


def configure(*, enabled: bool = True) -> Path | None:
    """Point numba at the first writable cache directory.

    Returns the directory in use, or ``None`` when nothing is writable (numba then
    keeps its default and the user pays the compile cost every run).

    Must be called **before** numba is imported: it reads ``NUMBA_CACHE_DIR`` once,
    at import time.  Calling it later is harmless but has no effect.
    """
    global _RESOLVED
    if _RESOLVED is not None:
        return _RESOLVED
    if not enabled:
        return None

    for candidate in _candidates():
        if _is_writable(candidate):
            os.environ["NUMBA_CACHE_DIR"] = str(candidate)
            _RESOLVED = candidate
            return candidate
    return None


@dataclass
class WarmReport:
    """Outcome of a warm-up run."""

    cache_dir: str
    compiled: list[str]
    already_cached: bool
    seconds: float
    failed: list[tuple[str, str]]

    @property
    def ok(self) -> bool:
        return not self.failed

    def to_dict(self) -> dict[str, object]:
        return {
            "cache_dir": self.cache_dir,
            "compiled": list(self.compiled),
            "already_cached": self.already_cached,
            "seconds": round(self.seconds, 2),
            "failed": [{"path": name, "error": error} for name, error in self.failed],
        }


def _cache_entry_count() -> int:
    """How many compiled objects numba has on disk.

    Numba writes ``*.nbi`` / ``*.nbc`` pairs into the index it manages under
    ``NUMBA_CACHE_DIR``.  Counting files is deliberately dumb and depends on no
    private numba API — the alternative (reading its pickle index) reaches into
    internals that have already changed shape between releases.
    """
    root = cache_dir()
    if not root.is_dir():
        return 0
    return sum(1 for _ in root.rglob("*.nbi"))


def is_warm() -> bool:
    """True when numba already has compiled objects cached on disk.

    Never triggers a compile: it only inspects the cache directory.
    """
    return _cache_entry_count() > 0


def warm(*, verbose: bool = True) -> WarmReport:
    """Exercise the hot librosa paths once so numba writes its cache.

    Safe to call repeatedly: after the first run it is a sub-second no-op.
    """
    import time

    configure()
    started = time.time()
    entries_before = _cache_entry_count()
    compiled: list[str] = []
    failed: list[tuple[str, str]] = []

    try:
        import numpy as np

        import librosa
    except Exception as exc:  # noqa: BLE001 - report, do not raise
        return WarmReport(str(cache_dir()), [], False, time.time() - started, [("import", str(exc))])

    # Two seconds of quiet noise: long enough to exercise padding/framing, short
    # enough that the actual DSP is irrelevant next to the compile time.
    sr = 22050
    y = (np.random.RandomState(0).randn(sr * 2) * 1e-3).astype("float32")

    for name in HOT_PATHS:
        label = f"librosa.{name}"
        try:
            if name == "stft":
                librosa.stft(y, n_fft=2048, hop_length=512)
            elif name == "chroma_stft":
                librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
            elif name == "chroma_cqt":
                librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
            elif name == "onset_strength":
                librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
            elif name == "beat_track":
                librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
            elif name == "pyin":
                librosa.pyin(y, fmin=65.0, fmax=1000.0, sr=sr, frame_length=2048)
            compiled.append(label)
            if verbose:
                print(f"  warmed {label}", flush=True)
        except Exception as exc:  # noqa: BLE001 - keep going, report at the end
            failed.append((label, f"{type(exc).__name__}: {exc}"))

    return WarmReport(
        cache_dir=str(cache_dir()),
        compiled=compiled,
        already_cached=entries_before > 0 and time.time() - started < 2.0,
        seconds=time.time() - started,
        failed=failed,
    )
