"""ensemble2score — audio to multi-instrument ensemble score.

The public surface is deliberately narrow: everything that crosses a stage
boundary is a dataclass from :mod:`ensemble2score.core.models`, and the only
entry points most callers need are :func:`ensemble2score.pipeline.run` and the
``e2s`` command line.

Heavy optional backends (torch, demucs, basic-pitch, numba-accelerated parts of
librosa) are imported lazily, never at package import time.  See
:mod:`ensemble2score.analysis.backends` for why that matters on a cold machine.
"""

from __future__ import annotations

__all__ = ["__version__", "run"]

__version__ = "2.0.0"


def run(*args, **kwargs):
    """Convenience wrapper around :func:`ensemble2score.pipeline.run`.

    Imported lazily so that ``import ensemble2score`` never pulls in numpy,
    librosa or music21.
    """
    from .pipeline import run as _run

    return _run(*args, **kwargs)
