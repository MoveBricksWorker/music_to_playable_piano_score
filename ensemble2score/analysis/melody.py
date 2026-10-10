"""Consolidate a frame-based transcription into a singable melody line.

The problem, measured on a real vocal stem
-----------------------------------------
A frame-based transcriber (Basic Pitch) reports one note per stable-pitch region,
so a single sung note becomes several consecutive notes and a note whose pitch
flickers by a semitone for one frame becomes two.  Measured on FictionJunction —
*blessing*'s vocal stem: 667 notes over 192 s, of which

* **29%** of consecutive pairs are the *same pitch* — the signature of frame
  jitter rather than a repeated note;
* 70% of consecutive pairs differ by two semitones or less;
* the gap between one note's end and the next one's start is bimodal: 18% are
  0-50 ms (certainly one note split in two) and 16% are 100-200 ms (probably a
  real re-articulation).

The damage this causes downstream is not cosmetic.  Quantisation rounds each
onset and offset to the grid independently, so a short fragment collapses to zero
duration and is **discarded**: on that stem 146 of 667 notes (22%) vanished, and
the survivors were carved into sixteenth notes.

What this module does about it
------------------------------
Two repairs, in this order, both parameterised so the effect can be measured
rather than assumed:

1. **Join same-pitch neighbours** separated by less than ``join_gap`` seconds, and
   merge same-pitch notes that overlap.  This is what removes the jitter and
   restores a plausible sung duration.
2. **Drop notes shorter than ``min_duration``** afterwards, because joining leaves
   the isolated blips that used to sit between the joined notes.

It deliberately does *not* merge across a pitch change: that would erase the
melody, which is the one thing the transcriber gets right.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..core.models import Note

__all__ = ["MelodyCleanup", "clean_melody", "NEUTRAL", "GENTLE", "STRONG"]


@dataclass(frozen=True)
class MelodyCleanup:
    """Parameters for one clean-up pass.

    ``join_gap`` is the interesting one.  Too small and the fragment survives to be
    discarded by quantisation; too large and two syllables merge into one note. The
    presets exist because that trade-off cannot be settled from the audio alone —
    it has to be looked at (and listened to).
    """

    join_gap: float = 0.08
    min_duration: float = 0.09
    merge_overlaps: bool = True

    def label(self) -> str:
        return f"gap{int(self.join_gap * 1000)}ms_min{int(self.min_duration * 1000)}ms"


# Conservative: only certain jitter is joined.
GENTLE = MelodyCleanup(join_gap=0.05, min_duration=0.08)
# Middle ground; the default.
NEUTRAL = MelodyCleanup(join_gap=0.10, min_duration=0.12)
# Aggressive: assumes nearly every short gap is an artefact.
STRONG = MelodyCleanup(join_gap=0.18, min_duration=0.15)


def clean_melody(
    notes: Sequence[Note],
    cleanup: MelodyCleanup = NEUTRAL,
) -> tuple[list[Note], dict[str, int]]:
    """Join fragments and drop leftovers.

    Returns ``(notes, stats)`` where ``stats`` records what happened, so a caller can
    report it instead of silently changing the melody.
    """
    stats = {"input": len(notes), "joined": 0, "dropped_short": 0, "output": 0}
    if not notes:
        return [], stats

    ordered = sorted(notes, key=lambda n: (n.start, n.end))
    merged: list[Note] = [Note(**vars(ordered[0]))]

    for note in ordered[1:]:
        last = merged[-1]
        same_pitch = last.pitch == note.pitch

        if same_pitch and cleanup.merge_overlaps and note.start < last.end:
            # Still sounding at the same pitch: one note, extended.
            if note.end > last.end:
                last.end = note.end
            last.velocity = max(last.velocity, note.velocity)
            last.confidence = max(last.confidence, note.confidence)
            stats["joined"] += 1
            continue

        if same_pitch and 0.0 <= note.start - last.end <= cleanup.join_gap:
            # A short silence inside one sung note.
            last.end = max(last.end, note.end)
            last.velocity = max(last.velocity, note.velocity)
            last.confidence = min(last.confidence, note.confidence)
            stats["joined"] += 1
            continue

        merged.append(Note(**vars(note)))

    kept = [n for n in merged if n.duration >= cleanup.min_duration]
    stats["dropped_short"] = len(merged) - len(kept)
    stats["output"] = len(kept)
    return kept, stats
