"""Time to musical position: quantisation, barlines, pickup, ties.

The one decision that shapes this whole module: **the time-to-position map is
piecewise linear, anchored on every detected beat**, not a single global BPM.  A
global tempo smears rubato and every expressive performance into the wrong
place; anchoring on beats means a track that drifts from 148 to 156 BPM still
quantises correctly.

The second decision: barlines, pickup measures and cross-bar ties are built
explicitly here rather than handed to music21's heuristics.  music21's guesses
are reasonable for engraved music and wrong for a transcription, where we know
exactly where every beat is.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Sequence

import numpy as np

from ..core.models import Note

__all__ = [
    "Grid",
    "Timeline",
    "QuantizedNote",
    "build_timeline",
    "swing_map",
]

# A note is only allowed to extend this far past the last detected beat before
# the trailing region is handled by extrapolation.
_TAIL_BEATS = 8


@dataclass(frozen=True)
class Grid:
    """Beat grid plus the meter, and the conversions that follow from them."""

    beat_times: tuple[float, ...]
    beats_per_bar: int
    grid: int = 4               # divisions per beat (4 = sixteenth notes)
    phase: int = 0              # which beat index starts a bar
    swing: float = 0.0
    denominator: int = 4

    def __post_init__(self) -> None:
        if self.beats_per_bar < 1:
            raise ValueError("beats_per_bar must be >= 1")
        if self.grid < 1:
            raise ValueError("grid must be >= 1")

    @property
    def last_beat(self) -> float:
        return self.beat_times[-1] if self.beat_times else 0.0

    # ------------------------------------------------------------------ #
    def position_of(self, time: float) -> float:
        """Seconds -> fractional beat index, piecewise linear between beats.

        Before the first beat and after the last one the nearest beat interval is
        extrapolated, so a pickup or a trailing note still lands somewhere sane.
        """
        beats = self.beat_times
        if len(beats) == 0:
            return 0.0
        if len(beats) == 1:
            return (time - beats[0]) * 2.0

        index = bisect_right(beats, time)
        if index == 0:
            span = beats[1] - beats[0]
            return (time - beats[0]) / span if span > 0 else 0.0
        if index >= len(beats):
            span = beats[-1] - beats[-2]
            return (len(beats) - 1) + ((time - beats[-1]) / span if span > 0 else 0.0)

        start, end = beats[index - 1], beats[index]
        span = end - start
        return (index - 1) + ((time - start) / span if span > 0 else 0.0)

    def time_of(self, position: float) -> float:
        """Fractional beat index -> seconds.  Inverse of :meth:`position_of`."""
        beats = self.beat_times
        if len(beats) == 0:
            return 0.0
        if len(beats) == 1:
            return beats[0] + position / 2.0

        index = int(np.floor(position))
        fraction = position - index
        if index < 0:
            span = beats[1] - beats[0]
            return beats[0] + position * span
        if index >= len(beats) - 1:
            span = beats[-1] - beats[-2]
            return beats[-1] + (position - (len(beats) - 1)) * span
        return beats[index] + fraction * (beats[index + 1] - beats[index])

    # ------------------------------------------------------------------ #
    def quantize(self, position: float, *, round_half_up: bool = True) -> Fraction:
        """Snap a beat position to the grid, with optional swing.

        Swing delays the off-beat subdivisions by ``swing`` of a grid step, which
        is what makes a shuffle readable instead of notated as straight triplets.
        """
        steps = Fraction(round(position * self.grid), self.grid)
        if self.swing <= 0.0:
            return steps

        step = Fraction(round(steps * self.grid))
        if step % 2 == 1:  # an off-beat subdivision
            return steps + Fraction(round(self.swing * 1000), self.grid * 1000)
        return steps

    def bar_of(self, position: float) -> int:
        """Bar index (0-based) for a fractional beat position."""
        beat_index = int(np.floor(position))
        return max(0, (beat_index - self.phase) // self.beats_per_bar)

    def beat_in_bar(self, position: Fraction) -> Fraction:
        """Position within its bar, measured in beats from the barline."""
        return position - (self.phase + self.bar_of(float(position)) * self.beats_per_bar)

    def absolute_of(self, bar: int, offset_beats: Fraction) -> Fraction:
        """(bar, offset within bar) -> absolute beat position."""
        base = self.phase + bar * self.beats_per_bar
        return Fraction(base) + offset_beats

    def bar_length(self) -> Fraction:
        return Fraction(self.beats_per_bar)

    def bar_count(self, total_positions: float) -> int:
        """How many bars a span covers, given where the music starts."""
        return max(0, int(np.ceil((total_positions - self.phase) / self.beats_per_bar)))


def swing_map(swing: float) -> float:
    """Clamp a user-supplied swing amount into the usable range."""
    return float(min(max(swing, 0.0), 0.66))


# --------------------------------------------------------------------------- #
# notes
# --------------------------------------------------------------------------- #
@dataclass
class QuantizedNote:
    """A note placed on the grid, with the (bar, offset) of both ends.

    ``start``/``end`` are beat positions (``Fraction``), which is what notation
    needs.  ``source_start``/``source_end`` keep the original seconds so the MIDI
    export stays on the real timeline instead of being stretched onto a perfectly
    even grid.
    """

    pitch: int
    velocity: int
    start: Fraction
    end: Fraction
    start_bar: int
    start_offset: Fraction
    end_bar: int
    end_offset: Fraction
    confidence: float = 1.0
    source_start: float = 0.0
    source_end: float = 0.0

    @property
    def duration(self) -> Fraction:
        return self.end - self.start

    @property
    def crosses_bar(self) -> bool:
        return self.end_bar != self.start_bar

    def to_dict(self) -> dict[str, object]:
        return {
            "pitch": int(self.pitch),
            "velocity": int(self.velocity),
            "start_bar": int(self.start_bar),
            "start_offset": float(self.start_offset),
            "duration_beats": float(self.duration),
            "source_start": round(float(self.source_start), 6),
            "source_end": round(float(self.source_end), 6),
            "confidence": round(float(self.confidence), 3),
        }


@dataclass
class Timeline:
    """The whole quantised score: one grid plus every part placed on it."""

    grid: Grid
    parts: dict[str, list[QuantizedNote]] = field(default_factory=dict)
    bar_count: int = 0
    pickup_beats: Fraction = Fraction(0)
    warnings: list[str] = field(default_factory=list)

    @property
    def notes(self) -> list[QuantizedNote]:
        return [note for notes in self.parts.values() for note in notes]

    def to_dict(self) -> dict[str, object]:
        return {
            "beats_per_bar": self.grid.beats_per_bar,
            "grid": self.grid.grid,
            "phase": self.grid.phase,
            "swing": self.grid.swing,
            "bar_count": self.bar_count,
            "pickup_beats": float(self.pickup_beats),
            "parts": {stem: [n.to_dict() for n in notes] for stem, notes in self.parts.items()},
            "warnings": list(self.warnings),
        }


def build_timeline(
    notes_by_part: dict[str, Sequence[Note]],
    *,
    beat_times: Sequence[float],
    beats_per_bar: int,
    phase: int = 0,
    grid: int = 4,
    swing: float = 0.0,
    min_duration_beats: Fraction = Fraction(1, 8),
    music_end: float | None = None,
) -> Timeline:
    """Place every part's notes on a shared grid.

    ``phase`` is the beat index that starts a bar.  A non-zero phase means the
    music begins mid-bar, which is notated as a real pickup measure rather than by
    silently shifting the downbeat.

    ``music_end`` is the last second that contains music.  The bar count comes from
    that, not from the end of the beat grid: extrapolating past the last note
    invents trailing empty bars, and using the raw beat count over-counts by one
    whenever the final beat is not the start of a bar.
    """
    beats = tuple(float(t) for t in beat_times)
    if len(beats) < 2:
        raise ValueError("at least two beats are required to build a timeline")

    block = Grid(
        beat_times=beats,
        beats_per_bar=int(beats_per_bar),
        grid=int(grid),
        phase=int(phase) % int(beats_per_bar),
        swing=swing_map(swing),
    )

    timeline = Timeline(grid=block)
    timeline.pickup_beats = Fraction(block.phase)

    if music_end is None:
        ends = [note.end for notes in notes_by_part.values() for note in notes]
        music_end = max(ends) if ends else beats[-1]

    total_positions = block.position_of(float(music_end))
    timeline.bar_count = block.bar_count(total_positions)

    for stem, notes in notes_by_part.items():
        placed: list[QuantizedNote] = []
        dropped = 0
        for note in notes:
            start = block.quantize(block.position_of(note.start))
            end = block.quantize(block.position_of(note.end))
            if end <= start:
                end = start + Fraction(1, block.grid)
            if end - start < min_duration_beats:
                dropped += 1
                continue
            placed.append(_place(note, start, end, block))
        if dropped:
            timeline.warnings.append(
                f"{stem}: dropped {dropped} note(s) shorter than {float(min_duration_beats):.3f} beats"
            )
        timeline.parts[stem] = sorted(placed, key=lambda n: (n.start, n.pitch))

    return timeline


def _place(note: Note, start: Fraction, end: Fraction, block: Grid) -> QuantizedNote:
    start_bar = block.bar_of(float(start))
    end_bar = block.bar_of(float(end - Fraction(1, 1000)))
    return QuantizedNote(
        pitch=int(note.pitch),
        velocity=int(note.velocity),
        start=start,
        end=end,
        start_bar=start_bar,
        start_offset=block.beat_in_bar(start),
        end_bar=max(end_bar, start_bar),
        end_offset=block.beat_in_bar(end),
        confidence=float(note.confidence),
        source_start=float(note.start),
        source_end=float(note.end),
    )
