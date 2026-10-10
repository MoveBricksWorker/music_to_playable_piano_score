"""Core, dependency-light primitives: the data model, config and errors.

Nothing in this subpackage may import librosa, torch, music21 or numpy-heavy
modules at import time — the CLI relies on that for a fast ``e2s version``.
"""

from __future__ import annotations

from .errors import (
    BackendError,
    BackendUnavailable,
    ConfigError,
    Ensemble2ScoreError,
    InputError,
    OutputError,
)
from .models import (
    CACHE_SCHEMA,
    BeatTrack,
    Chord,
    DrumHit,
    MeterResult,
    Note,
    Part,
    PartSpec,
    RunResult,
    StageResult,
    StemSet,
    TimeSignature,
    as_array,
    as_list,
)

__all__ = [
    "CACHE_SCHEMA",
    "BackendError",
    "BackendUnavailable",
    "BeatTrack",
    "Chord",
    "ConfigError",
    "DrumHit",
    "Ensemble2ScoreError",
    "InputError",
    "MeterResult",
    "Note",
    "OutputError",
    "Part",
    "PartSpec",
    "RunResult",
    "StageResult",
    "StemSet",
    "TimeSignature",
    "as_array",
    "as_list",
]
