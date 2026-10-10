"""Voice allocation and chord grouping — layout decisions made explicitly.

Why this module exists
----------------------
music21 cannot do this job, and every alternative was tested and rejected:

* ``Part.makeChords`` does not exist in 9.x.
* ``Part.makeVoices`` exists but produced **zero** voices on a stream that plainly
  needed them — a silent no-op, which is worse than an error.
* A ``PartStaff`` nested inside a ``Part`` raises
  ``StreamException: no measures found``.
* ``layout.StaffGroup(..., symbol="brace")`` is accepted and then **not written to
  MusicXML at all** — no ``<part-symbol>``, no ``<staves>``.

So the decision that makes a polyphonic part readable is made here, on plain data.
A transcribed part arrives as a loose bag of overlapping notes; notation needs
monophonic voices each of which may carry chords.

The one rule that matters
-------------------------
A chord is a set of notes that **start together and end together**.  Notes that
share an onset but differ in duration are *not* a chord, and emitting them as one
produces illegal notation (a chord has a single duration).  So notes are allocated
to voices one at a time, longest first, each taking the lowest-numbered voice that
is free for its whole span.  Once every voice is monophonic, notes that share an
onset *within a voice* necessarily share a duration — so grouping by onset inside a
voice is safe, and that is where chords come from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterable, Sequence

__all__ = [
    "PlacedNote",
    "VoicePlan",
    "MAX_VOICES",
    "group_chords",
    "allocate_voices",
    "plan_measure",
]

# MusicXML permits more than four voices per staff, but no reader displays them
# usefully and a transcription artefact would happily produce a dozen.
MAX_VOICES = 4

# Two positions within this many quarter lengths are treated as identical.  The
# quantiser has already snapped to a grid, so this only absorbs Fraction/float noise.
EPSILON = Fraction(1, 1000)


@dataclass
class PlacedNote:
    """One note, positioned in absolute quarter lengths."""

    pitch: int
    start: Fraction
    end: Fraction
    velocity: int = 80
    voice: int = 1

    @property
    def duration(self) -> Fraction:
        return self.end - self.start


@dataclass
class VoicePlan:
    """A measure-worth of notes organised into monophonic voices of chords."""

    voices: list[list[list[PlacedNote]]] = field(default_factory=list)
    dropped: int = 0

    @property
    def voices_used(self) -> int:
        return len(self.voices)

    @property
    def note_count(self) -> int:
        return sum(len(chord) for voice in self.voices for chord in voice)


def group_chords(notes: Sequence[PlacedNote]) -> list[list[PlacedNote]]:
    """Group notes by identical onset **and** duration.

    Notes that merely share a start do not form a chord unless they also share an
    end: a chord in MusicXML carries one duration, so grouping a held note with a
    short one produces something no reader can represent.
    """
    if not notes:
        return []
    ordered = sorted(notes, key=lambda n: (n.start, n.end, n.pitch))
    groups: list[list[PlacedNote]] = [[ordered[0]]]
    for note in ordered[1:]:
        head = groups[-1][0]
        if abs(note.start - head.start) <= EPSILON and abs(note.end - head.end) <= EPSILON:
            groups[-1].append(note)
        else:
            groups.append([note])
    return groups


def allocate_voices(
    notes: Sequence[PlacedNote],
    *,
    max_voices: int = MAX_VOICES,
) -> tuple[list[list[PlacedNote]], int]:
    """Assign notes to voices, treating each chord as one unit.

    The order of the two operations matters and is the whole point of this
    function.  Simultaneous notes that share a duration are grouped **first**, and
    the groups are then allocated to voices, measuring occupancy by the group's
    span.  Allocating note-by-note first would see each note of a struck chord as
    overlapping the previous one, scatter the chord across three voices, and leave
    a block chord engraved as three separate monophonic lines.

    Longer groups are placed first at a shared onset, which puts sustained material
    in the lower voice numbers — how an engraver writes a held bass under a moving
    part.

    Returns ``(voices, dropped)``.  Notes that do not fit within ``max_voices`` are
    counted, not stacked: a voice that cannot represent them mis-states the rhythm
    of everything around it.
    """
    groups = group_chords(notes)
    # A group larger than the voice budget is split: a chord wider than the staff
    # allows is unreadable, and splitting puts the extra notes on the next voice
    # rather than discarding them.
    if max_voices >= 1:
        split: list[list[PlacedNote]] = []
        for group in groups:
            for index in range(0, len(group), max_voices):
                split.append(group[index: index + max_voices])
        groups = split

    # A group's span decides occupancy; sort by onset, then longest first.
    ordered = sorted(
        groups,
        key=lambda group: (group[0].start, -(group[0].end - group[0].start), group[0].pitch),
    )

    voices: list[list[PlacedNote]] = []
    dropped = 0
    for group in ordered:
        span_start = group[0].start
        span_end = max(note.end for note in group)
        if span_end - span_start <= 0:
            dropped += len(group)
            continue

        for index, voice in enumerate(voices):
            # Compare against the voice's occupied span, which is enough because
            # voices are built in time order and each holds whole groups.
            if any(
                other.start < span_end - EPSILON and span_start < other.end - EPSILON
                for other in voice
            ):
                continue
            for note in group:
                note.voice = index + 1
            voice.extend(group)
            break
        else:
            if len(voices) >= max_voices:
                dropped += len(group)
                continue
            for note in group:
                note.voice = len(voices) + 1
            voices.append(list(group))

    return voices, dropped


def plan_measure(
    notes: Iterable[PlacedNote],
    *,
    measure_start: Fraction,
    measure_end: Fraction,
    max_voices: int = MAX_VOICES,
) -> VoicePlan:
    """Organise one measure (or one staff of one measure) into voices of chords.

    Each voice comes back as a list of chords in time order, ready to be written as
    one MusicXML voice with ``<backup>`` separating them.
    """
    within = [
        note
        for note in notes
        if measure_start - EPSILON <= note.start < measure_end
    ]
    if not within:
        return VoicePlan()

    voices, dropped = allocate_voices(within, max_voices=max_voices)
    # Voices arrive grouped already; group again per voice purely to guarantee the
    # order within each voice is by time, which the writer depends on.
    ordered = [group_chords(voice) for voice in voices]
    return VoicePlan(voices=ordered, dropped=dropped)
