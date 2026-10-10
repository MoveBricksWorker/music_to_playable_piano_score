"""The data model.

Every value that crosses a stage boundary is defined here, and every one of them
round-trips through JSON.  Two rules keep this file honest:

1. **Plain Python containers in the model, numpy at the edges.**  ``beat_times``
   is a ``list[float]``, not an array, because the cache stores JSON and a silent
   numpy round-trip is exactly the kind of thing that used to corrupt old caches.
   Convert with :func:`as_array` where you need vector maths.
2. **No behaviour that needs a heavy import.**  This module is imported by the
   CLI for ``e2s version``, so it must stay dependency-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "as_array",
    "as_list",
    "Note",
    "Part",
    "PartSpec",
    "BeatTrack",
    "TimeSignature",
    "DrumHit",
    "Chord",
    "StemSet",
    "StageResult",
    "RunResult",
]

# The cache payload shape.  Bump this whenever a dataclass below changes in a way
# that would make an old cache entry mean something different.  It is part of the
# cache key (see core.cache), so a bump invalidates every stale entry instead of
# letting it be loaded into a newer model and silently disabling new behaviour.
CACHE_SCHEMA = 3


# --------------------------------------------------------------------------- #
# conversion helpers
# --------------------------------------------------------------------------- #
def as_array(values: Sequence[float]) -> np.ndarray:
    """Sequence of numbers -> ``float64`` ndarray."""
    return np.asarray(values, dtype=float)


def as_list(values: Iterable[float]) -> list[float]:
    """Iterable of numbers -> a JSON-friendly, rounded list of floats.

    Rounded to microseconds: sub-microsecond precision is meaningless for audio
    and it keeps ``run.json`` small and readable.
    """
    return [round(float(v), 6) for v in values]


def _clean(value: Any) -> Any:
    """Recursively make a value JSON-safe (numpy scalars, NaN, infinity)."""
    if isinstance(value, np.ndarray):
        return [_clean(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return _clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "to_dict"):
        return _clean(value.to_dict())
    return value


# --------------------------------------------------------------------------- #
# notation-side spec for one part
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PartSpec:
    """How one stem is turned into a part of the score.

    Frozen on purpose: a spec is looked up by name all over the notation code and
    a mutable shared instance is a bug waiting to happen.
    """

    stem: str
    part_name: str
    instrument: str
    midi_program: int | None = None      # None for percussion
    clef: str = "treble"                 # treble | bass | percussion
    grand_staff: bool = False            # keyboard-like: two staves
    monophonic: bool = False             # keep only the strongest overlapping note
    sounding_pitch: int | None = None    # written pitch + this = sounding; None = no transpose
    drum_channel: bool = False           # export on MIDI channel 10

    def to_dict(self) -> dict[str, Any]:
        return {
            "stem": self.stem,
            "part_name": self.part_name,
            "instrument": self.instrument,
            "midi_program": self.midi_program,
            "clef": self.clef,
            "grand_staff": self.grand_staff,
            "monophonic": self.monophonic,
            "sounding_pitch": self.sounding_pitch,
            "drum_channel": self.drum_channel,
        }


# --------------------------------------------------------------------------- #
# stage: separation
# --------------------------------------------------------------------------- #
@dataclass
class StemSet:
    """Separated audio written to disk, plus how loud each stem turned out."""

    directory: str
    stems: dict[str, str]          # stem name -> absolute path
    backend: str
    sample_rate: int
    level_db: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return list(self.stems)

    def to_dict(self) -> dict[str, Any]:
        return {
            "directory": self.directory,
            "stems": dict(self.stems),
            "backend": self.backend,
            "sample_rate": int(self.sample_rate),
            "level_db": {k: round(float(v), 2) for k, v in self.level_db.items()},
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StemSet":
        return cls(
            directory=d["directory"],
            stems=dict(d["stems"]),
            backend=d["backend"],
            sample_rate=int(d["sample_rate"]),
            level_db={k: float(v) for k, v in d.get("level_db", {}).items()},
            warnings=list(d.get("warnings", [])),
        )


# --------------------------------------------------------------------------- #
# stage: rhythm / meter
# --------------------------------------------------------------------------- #
@dataclass
class BeatTrack:
    """Beat and downbeat positions plus the notated tempo."""

    tempo_bpm: float
    beat_times: list[float]
    downbeat_times: list[float] = field(default_factory=list)
    backend: str = "?"
    tempo_stability: float = 1.0   # 1.0 metronomic, -> 0 strongly variable
    warnings: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.beat_times)

    @property
    def beat_interval(self) -> float:
        beats = as_array(self.beat_times)
        if beats.size < 2:
            return 60.0 / max(self.tempo_bpm, 1e-6)
        return float(np.median(np.diff(beats)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "tempo_bpm": round(float(self.tempo_bpm), 3),
            "beat_times": as_list(self.beat_times),
            "downbeat_times": as_list(self.downbeat_times),
            "backend": self.backend,
            "tempo_stability": round(float(self.tempo_stability), 4),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BeatTrack":
        return cls(
            tempo_bpm=float(d["tempo_bpm"]),
            beat_times=[float(v) for v in d.get("beat_times", [])],
            downbeat_times=[float(v) for v in d.get("downbeat_times", [])],
            backend=d.get("backend", "?"),
            tempo_stability=float(d.get("tempo_stability", 1.0)),
            warnings=list(d.get("warnings", [])),
        )


@dataclass
class TimeSignature:
    """One meter hypothesis, with the evidence that produced it.

    ``confidence`` is a temperature-scaled softmax over the fused cue scores, so
    it is comparable *within* one run and meaningless across runs.
    """

    numerator: int
    denominator: int = 4
    confidence: float = 0.0
    evidence: dict[str, float] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.numerator}/{self.denominator}"

    @property
    def beats_per_bar(self) -> int:
        """How many *detected beats* make up one bar."""
        return self.numerator

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "numerator": int(self.numerator),
            "denominator": int(self.denominator),
            "confidence": round(float(self.confidence), 4),
            "evidence": {k: round(float(v), 4) for k, v in self.evidence.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TimeSignature":
        return cls(
            numerator=int(d["numerator"]),
            denominator=int(d["denominator"]),
            confidence=float(d.get("confidence", 0.0)),
            evidence={k: float(v) for k, v in d.get("evidence", {}).items()},
        )


@dataclass
class MeterResult:
    """Ranked meter hypotheses (best first) and the bar phase they imply."""

    hypotheses: list[TimeSignature]
    phase: int = 0                  # beat index that starts a bar
    bars: int = 0
    backend: str = "heuristic"
    warnings: list[str] = field(default_factory=list)

    @property
    def best(self) -> TimeSignature | None:
        return self.hypotheses[0] if self.hypotheses else None

    @property
    def label(self) -> str:
        return self.best.label if self.best else "?"

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "best": self.label,
            "phase": int(self.phase),
            "bars": int(self.bars),
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MeterResult":
        return cls(
            hypotheses=[TimeSignature.from_dict(h) for h in d.get("hypotheses", [])],
            phase=int(d.get("phase", 0)),
            bars=int(d.get("bars", 0)),
            backend=d.get("backend", "heuristic"),
            warnings=list(d.get("warnings", [])),
        )


# --------------------------------------------------------------------------- #
# stage: transcription
# --------------------------------------------------------------------------- #
@dataclass
class Note:
    """One transcribed note in seconds on the original timeline.

    ``sounding_pitch`` is a MIDI pitch number; octave correction happens before
    a Note is built, so nothing downstream re-decides register.
    """

    pitch: int
    start: float
    end: float
    velocity: int = 80
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.end < self.start:
            self.end = self.start

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "pitch": int(self.pitch),
            "start": round(float(self.start), 6),
            "end": round(float(self.end), 6),
            "velocity": int(self.velocity),
            "confidence": round(float(self.confidence), 4),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Note":
        return cls(
            pitch=int(d["pitch"]),
            start=float(d["start"]),
            end=float(d["end"]),
            velocity=int(d.get("velocity", 80)),
            confidence=float(d.get("confidence", 1.0)),
        )


@dataclass
class Part:
    """One instrument's notes, ready for notation."""

    stem: str
    spec: PartSpec
    notes: list[Note] = field(default_factory=list)
    backend: str = "?"
    warnings: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.spec.part_name

    @property
    def n_notes(self) -> int:
        return len(self.notes)

    @property
    def pitch_range(self) -> tuple[int, int] | None:
        """(lowest, highest) written pitch, or None when the part is empty."""
        if not self.notes:
            return None
        pitches = [n.pitch for n in self.notes]
        return (min(pitches), max(pitches))

    def sorted_notes(self) -> list[Note]:
        return sorted(self.notes, key=lambda n: (n.start, n.pitch))

    def to_dict(self) -> dict[str, Any]:
        rng = self.pitch_range
        return {
            "stem": self.stem,
            "part_name": self.spec.part_name,
            "instrument": self.spec.instrument,
            "backend": self.backend,
            "n_notes": self.n_notes,
            "pitch_range": list(rng) if rng else None,
            "warnings": list(self.warnings),
            "notes": [n.to_dict() for n in self.notes],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Part":
        from ..io.instruments import resolve  # local import keeps models import-light

        return cls(
            stem=d["stem"],
            spec=resolve(d["stem"]),
            notes=[Note.from_dict(n) for n in d.get("notes", [])],
            backend=d.get("backend", "?"),
            warnings=list(d.get("warnings", [])),
        )


# --------------------------------------------------------------------------- #
# stage: percussion and harmony
# --------------------------------------------------------------------------- #
DRUM_VOICES = ("kick", "snare", "hihat")


@dataclass
class DrumHit:
    """A drum onset classified into one of three coarse voices."""

    voice: str
    time: float
    velocity: int = 80
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.voice not in DRUM_VOICES:
            raise ValueError(f"unknown drum voice {self.voice!r}; expected one of {DRUM_VOICES}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "voice": self.voice,
            "time": round(float(self.time), 6),
            "velocity": int(self.velocity),
            "confidence": round(float(self.confidence), 4),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DrumHit":
        return cls(**d)


@dataclass
class Chord:
    """A chord label spanning ``[start, end)`` seconds."""

    start: float
    end: float
    label: str
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(float(self.start), 6),
            "end": round(float(self.end), 6),
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chord":
        return cls(**d)


# --------------------------------------------------------------------------- #
# final result
# --------------------------------------------------------------------------- #
@dataclass
class StageResult:
    """Timing and cache status for one pipeline stage, for the report."""

    stage: str
    seconds: float
    cached: bool = False
    backend: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "seconds": round(float(self.seconds), 3),
            "cached": bool(self.cached),
            "backend": self.backend,
        }


@dataclass
class RunResult:
    """Everything one run produced: the model behind run.json and report.md."""

    input_path: str
    duration: float
    sample_rate: int = 0

    stems: StemSet | None = None
    beats: BeatTrack | None = None
    meter: MeterResult | None = None
    parts: list[Part] = field(default_factory=list)
    drum_hits: list[DrumHit] = field(default_factory=list)
    chords: list[Chord] = field(default_factory=list)

    key: str | None = None
    measures: int = 0

    outputs: dict[str, str] = field(default_factory=dict)
    stages: list[StageResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    @property
    def time_signature(self) -> str:
        return self.meter.label if self.meter else "?"

    @property
    def n_notes(self) -> int:
        return sum(p.n_notes for p in self.parts)

    @property
    def total_seconds(self) -> float:
        return sum(s.seconds for s in self.stages)

    def warn(self, message: str, *, stage: str | None = None) -> None:
        """Record a warning once, prefixed by the stage that produced it."""
        text = f"{stage}: {message}" if stage else message
        if text not in self.warnings:
            self.warnings.append(text)

    def part(self, stem: str) -> Part | None:
        for p in self.parts:
            if p.stem == stem:
                return p
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": self.input_path,
            "duration": round(float(self.duration), 3),
            "sample_rate": int(self.sample_rate),
            "key": self.key,
            "measures": int(self.measures),
            "time_signature": self.time_signature,
            "stems": self.stems.to_dict() if self.stems else None,
            "beats": self.beats.to_dict() if self.beats else None,
            "meter": self.meter.to_dict() if self.meter else None,
            "parts": [p.to_dict() for p in self.parts],
            "n_parts": len(self.parts),
            "n_notes": self.n_notes,
            "n_drum_hits": len(self.drum_hits),
            "drum_hits": [h.to_dict() for h in self.drum_hits],
            "chords": [c.to_dict() for c in self.chords],
            "outputs": dict(self.outputs),
            "stages": [s.to_dict() for s in self.stages],
            "warnings": list(self.warnings),
        }
