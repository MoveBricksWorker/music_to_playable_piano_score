"""Console and path helpers.

Kept dependency-free and free of rich markup so it can be used from tests and
library code without dragging in the CLI stack.
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
from pathlib import Path

__all__ = [
    "ensure_dir",
    "slugify",
    "info",
    "hint",
    "warn",
    "error",
    "rule",
    "set_verbose",
    "is_verbose",
    "capture_noise",
]

_VERBOSE = True

# Some third-party libraries write cosmetic warnings straight to stdout/stderr
# with no API to switch them off.  We filter only these exact phrases.
_NOISE = (
    "messed up beam pair",
    "Overwriting the file",
    "requests/__init__.py",
    "RequestsDependencyWarning",
)


def set_verbose(value: bool) -> None:
    global _VERBOSE
    _VERBOSE = bool(value)


def is_verbose() -> bool:
    return _VERBOSE


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) and return it as a Path."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def slugify(text: str, limit: int = 60) -> str:
    """A filesystem-safe, still-recognisable version of ``text``.

    Keeps unicode letters (song titles are often not ASCII) but strips anything
    that is awkward in a filename.
    """
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", text).strip(" .")
    cleaned = re.sub(r"\s+", "-", cleaned)
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    return (cleaned[:limit].rstrip("-.")) or "score"


def _emit(prefix: str, message: str, stream) -> None:
    if not _VERBOSE:
        return
    print(f"{prefix}{message}", file=stream, flush=True)


def info(message: str) -> None:
    _emit("  ", message, sys.stdout)


def hint(message: str) -> None:
    _emit("  · ", message, sys.stdout)


def warn(message: str) -> None:
    _emit("  ! ", message, sys.stderr)


def error(message: str) -> None:
    _emit("  x ", message, sys.stderr)


def rule(title: str) -> None:
    if not _VERBOSE:
        return
    print(f"\n── {title} " + "─" * max(0, 58 - len(title)), file=sys.stdout, flush=True)


@contextlib.contextmanager
def capture_noise():
    """Swallow the known cosmetic third-party warnings for the duration.

    Used around score export, where music21 prints beam warnings for multi-voice
    measures that it offers no API to silence.
    """
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            yield
    finally:
        for buffer in (out, err):
            for line in buffer.getvalue().splitlines():
                if line.strip() and not any(noise in line for noise in _NOISE):
                    # Anything genuinely unexpected still gets through.
                    print(line, file=sys.stderr)
