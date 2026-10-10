"""Quantised model to a music21 score, and out to MusicXML / MIDI.

Four things this module does explicitly, because leaving them to music21 goes
wrong for a transcription:

1. **Barlines.**  Measures are built at offsets derived from the beat grid.  A
   pickup is a genuine short first measure, not a silently shifted downbeat.
2. **Boundary crossing is split and tied.**  A note held across a barline becomes
   two tied notes; a note whose duration is not a single notatable value is split
   at the next beat group so the written rhythm stays exact.
3. **The grand staff is two single-staff parts with a brace symbol.**  In
   music21 9.x a ``PartStaff`` nested in a ``Part`` cannot be exported to MusicXML
   (it reports "no measures found") and makes ``makeBeams`` refuse to run, so the
   brace is expressed the way MusicXML actually carries it: a ``<part-symbol>``
   on the first staff, which engravers group.
4. **MIDI is written with pretty_midi, not music21.**  music21's MIDI writer puts
   the *upper* staff of a grand staff on channel 10 — the percussion channel — so
   the right hand plays back as drums.

Every offset in music21 is in quarter lengths.  One *beat* of an ``x/4`` meter is
a quarter note; for ``x/8`` a beat is an eighth, so :data:`_QUARTER_PER_BEAT`
converts beat positions to quarter lengths in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence

from ..core.errors import OutputError
from ..core.models import Chord, DrumHit, Note, PartSpec, RunResult
from ..io.console import capture_noise, hint, warn
from ..io.instruments import resolve
from .layout import DRUM_MIDI_PITCH, clef_for, clef_for_range, instrument_for_spec
from .layout_voices import PlacedNote, plan_measure

__all__ = [
    "ScoreOptions",
    "ScoreStats",
    "build_score",
    "export_score",
    "split_at_boundaries",
    "largest_notatable",
]

# Beat position -> quarter lengths, by time-signature denominator.
_QUARTER_PER_BEAT: dict[int, Fraction] = {
    1: Fraction(4),
    2: Fraction(2),
    4: Fraction(1),
    8: Fraction(1, 2),
    16: Fraction(1, 4),
}

# Written durations we are willing to use, longest first.
#
# This table decides what a span can become, so an omission here is not a cosmetic
# gap -- it is an inability to write that rhythm at all.  Triplet values were missing,
# and the consequence was measured on a synthetic ruler
# (`work/_triplet_sprint/probe_triplet_ruler.py`): with a grid that can express
# triplets, a `1/3`-beat gap was filled with `1/4` (and with a grid that cannot, with
# `1/8`), so a triplet figure came out as sixteenths.  `is_notatable` still gates every
# entry on the grid, so on the default 4-ticks-per-beat grid the triplet values can
# never be selected: 1/3 * 4 = 4/3, not an integer.
_WRITTEN = (
    Fraction(4), Fraction(3), Fraction(2), Fraction(3, 2), Fraction(1),
    Fraction(3, 4), Fraction(2, 3), Fraction(1, 2), Fraction(3, 8), Fraction(1, 3),
    Fraction(1, 4), Fraction(3, 16), Fraction(1, 6), Fraction(1, 8), Fraction(1, 16),
)

GRAND_STAFF_SPLIT = 60  # middle C: at or above goes to the upper staff


@dataclass
class ScoreOptions:
    grid: int = 4
    swing: float = 0.0
    max_simultaneous: int = 4
    include_chords: bool = True
    include_drums: bool = True
    estimate_key: bool = True
    title: str | None = None
    composer: str | None = None
    # ``(start, end, label)`` rows for section rehearsal marks; only ever supplied by
    # a prior, because this pipeline infers no form of its own.
    sections: Sequence[tuple[float, float, str]] = ()


@dataclass
class ScoreStats:
    measures: int = 0
    time_signature: str = "?"
    key: str | None = None
    parts: list[str] = field(default_factory=list)
    notes: int = 0
    ties: int = 0
    sections: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "measures": int(self.measures),
            "time_signature": self.time_signature,
            "key": self.key,
            "parts": list(self.parts),
            "notes": int(self.notes),
            "ties": int(self.ties),
            "sections": int(self.sections),
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------- #
# splitting notes into notatable, tie-able pieces
# --------------------------------------------------------------------------- #
def split_at_boundaries(
    start: Fraction,
    end: Fraction,
    *,
    boundaries: Iterable[Fraction],
) -> list[tuple[Fraction, Fraction]]:
    """Cut ``[start, end)`` at every boundary strictly inside it.

    The pieces tile the original interval exactly, so a note is never lengthened
    or shortened by notation.
    """
    cuts = sorted({b for b in boundaries if start < b < end})
    points = [start, *cuts, end]
    return [(points[i], points[i + 1]) for i in range(len(points) - 1)]


def is_notatable(value: Fraction, grid: int) -> bool:
    """Can this duration be written as exactly one note?"""
    if value <= 0:
        return False
    return (value * grid).denominator == 1


def largest_notatable(remaining: Fraction, grid: int) -> Fraction:
    """Longest single written duration that does not exceed ``remaining``."""
    for value in _WRITTEN:
        if value <= remaining and is_notatable(value, grid):
            return value
    return Fraction(1, grid)


def _notate_span(
    start: Fraction,
    end: Fraction,
    *,
    bar_length: Fraction,
    beat_length: Fraction,
    grid: int,
    bar_offset: Fraction,
) -> list[tuple[Fraction, Fraction]]:
    """Split ``[start, end)`` (quarter lengths) into the fewest legal written notes.

    Cut points are **barlines only**, plus the beats of any note that spans more
    than one beat.  Inside each beat the span is filled greedily with the longest
    notatable value, which is what a human engraver does: a note of 3.5 beats
    becomes a dotted half tied to an eighth, not seven tied sixteenths.

    A note may never cross a barline untied; that is the one split that is not
    negotiable.
    """
    if end <= start:
        return []

    # Candidate cut points: barlines, then every beat strictly inside the span.
    cuts: set[Fraction] = set()
    first_bar = int((start - bar_offset) // bar_length)
    last_bar = int((end - bar_offset) // bar_length)
    for bar in range(first_bar, last_bar + 2):
        line = bar_offset + bar * bar_length
        if start < line < end:
            cuts.add(line)

    first_beat = int(start // beat_length)
    last_beat = int(end // beat_length)
    for beat in range(first_beat + 1, last_beat + 1):
        point = beat * beat_length
        if start < point < end:
            cuts.add(point)

    # Group the span into segments that each begin at a cut point.
    points = [start, *sorted(cuts), end]
    pieces: list[tuple[Fraction, Fraction]] = []
    for index in range(len(points) - 1):
        segment_start, segment_end = points[index], points[index + 1]
        cursor = segment_start
        guard = 0
        while cursor < segment_end and guard < 64:
            guard += 1
            step = largest_notatable(segment_end - cursor, grid)
            if step <= 0:
                step = Fraction(1, grid)
            pieces.append((cursor, cursor + step))
            cursor += step
    return pieces


# --------------------------------------------------------------------------- #
# score construction
# --------------------------------------------------------------------------- #
def key_signature(sharps: int, mode: str = "major", tonic: str | None = None):
    """Build a KeySignature, tolerating the positional form music21 9.x rejects."""
    from music21 import key as m21key

    if tonic:
        try:
            return m21key.KeySignature(sharps, tonic, mode)
        except Exception:  # noqa: BLE001 - fall back to the shorthand below
            pass
    signature = m21key.KeySignature(sharps)
    try:
        signature.mode = mode
    except Exception:  # noqa: BLE001 - mode is advisory only
        pass
    return signature


@dataclass
class _Layout:
    """Everything the builders need, resolved once.

    ``quarter_per_beat`` is the single conversion factor between beat positions
    (the quantiser's unit) and quarter lengths (music21's unit), so it is defined
    in exactly one place.
    """

    beats_per_bar: int
    quarter_per_beat: Fraction
    grid: int
    bar_count: int
    bar_offset: Fraction          # quarter lengths before the first barline
    modules: tuple
    max_voices: int = 4           # per staff; see layout_voices
    dropped_notes: int = 0        # running count of notes notation could not carry
    denominator: int = 4          # the meter's denominator, for beat->quarter maths

    @property
    def beat_length(self) -> Fraction:
        return self.quarter_per_beat

    @property
    def bar_length(self) -> Fraction:
        return Fraction(self.beats_per_bar) * self.quarter_per_beat


def _load_modules() -> tuple:
    from music21 import (
        clef,
        duration,
        harmony,
        instrument,
        layout,
        metadata,
        meter,
        note,
        stream,
        tempo,
    )

    return (stream, note, duration, clef, instrument, layout, meter, metadata, tempo, harmony)


def build_score(
    result: RunResult,
    options: ScoreOptions,
    *,
    timeline,
) -> tuple[object, ScoreStats, list[tuple[PartSpec, list]]]:
    """Assemble a music21 score from a run's quantised timeline.

    Returns ``(score, stats, midi_parts)``; ``midi_parts`` is the ``(spec, notes)``
    list used by the pretty_midi export.
    """
    (stream, note_mod, duration, clef, instrument, layout, meter, metadata, tempo, _harmony) = (
        _load_modules()
    )

    if not result.meter or not result.meter.best:
        raise OutputError("cannot notate without a time signature")

    numerator = result.meter.best.numerator
    denominator = result.meter.best.denominator
    quarter_per_beat = _QUARTER_PER_BEAT.get(denominator, Fraction(1))
    beats_per_bar = timeline.grid.beats_per_bar
    bar_offset = Fraction(timeline.grid.phase) * quarter_per_beat
    bar_count = max(timeline.bar_count, 1)

    stats = ScoreStats(
        measures=bar_count,
        time_signature=f"{numerator}/{denominator}",
    )
    ctx = _Layout(
        beats_per_bar=beats_per_bar,
        quarter_per_beat=quarter_per_beat,
        grid=options.grid,
        bar_count=bar_count,
        bar_offset=bar_offset,
        max_voices=max(1, options.max_simultaneous),
        modules=(stream, note_mod, duration, clef, instrument),
        denominator=denominator,
    )

    score = stream.Score()
    score.metadata = metadata.Metadata()
    score.metadata.title = options.title or Path(result.input_path).stem
    if options.composer:
        score.metadata.composer = options.composer

    midi_parts: list[tuple[PartSpec, list]] = []
    upper_of_grand: object | None = None

    for stem, quantized in timeline.parts.items():
        if not quantized:
            continue
        part_result = result.part(stem)
        spec = part_result.spec if part_result else resolve(stem)

        if spec.grand_staff:
            upper, lower, matched = _build_grand_staff(spec, quantized, ctx)
            score.insert(0, upper)
            score.insert(0, lower)
            if upper_of_grand is None:
                upper_of_grand = upper
            stats.parts.append(f"{spec.part_name} (2 staves)")
        else:
            # The clef follows the register the part actually landed in, not only
            # the spec: a separator does not always put an instrument where its
            # textbook clef expects it.  (Named `chosen_clef`, not `clef`, because
            # `clef` above is music21's clef *module*.)
            chosen_clef = clef_for_range(
                spec,
                [note.pitch + (spec.sounding_pitch or 0) for note in quantized],
                clef,
            )
            part, matched = _build_single_staff(spec, quantized, ctx, clef=chosen_clef)
            score.insert(0, part)
            stats.parts.append(spec.part_name)
        stats.notes += len(matched)
        midi_parts.append((spec, matched))

    if options.include_drums and result.drum_hits:
        drum_part = _build_drum_part(result.drum_hits, timeline, ctx)
        score.insert(0, drum_part)
        stats.parts.append("Percussion")
        # The percussion staff has to reach the MIDI file too.  Building ordinary
        # Note objects here keeps one code path in the writer — and the drum part
        # being silently absent from the MIDI, while present in the score, is
        # exactly the bug this replaced.
        midi_parts.append((resolve("percussive"), drum_notes_for_midi(result.drum_hits)))

    if not stats.parts:
        raise OutputError("the score has no parts: nothing was transcribed")

    _add_shared_markings(
        score,
        stats,
        options,
        result,
        time_signature=meter.TimeSignature(stats.time_signature),
        tempo_number=float(result.beats.tempo_bpm) if result.beats else None,
    )

    if options.include_chords and result.chords:
        _insert_chords(score, result.chords, timeline, quarter_per_beat, ctx, stats)

    if options.sections:
        _insert_sections(score, options.sections, timeline, quarter_per_beat, ctx, stats)

    return score, stats, midi_parts


def _insert_sections(
    score,
    sections: Sequence[tuple[float, float, str]],
    timeline,
    quarter_per_beat,
    ctx: _Layout,
    stats: ScoreStats,
) -> None:
    """Write section labels as rehearsal marks on the top part.

    A transcribed score with no section markers is hard to navigate — 192 bars of
    continuous music with nothing to navigate by.  The labels come from the
    SheetSage2 prior when one was supplied, because this pipeline infers nothing
    about form on its own.

    Consecutive rows carrying the same label (the prior emits one row per musical
    phrase) collapse to a single mark: a rehearsal mark on every phrase boundary of
    a repeated section is noise.
    """
    from music21 import expressions

    if not score.parts:
        return
    measures = list(score.parts[0].getElementsByClass("Measure"))
    if not measures:
        return

    grid = timeline.grid
    placed: set[tuple[int, str]] = set()
    previous: str | None = None
    inserted = 0

    for start, _end, raw_label in sections:
        label = (raw_label or "").strip()
        if not label or label == previous:
            continue
        previous = label

        position = grid.quantize(grid.position_of(float(start))) * quarter_per_beat
        index = _measure_index(position, ctx)
        if not (0 <= index < len(measures)):
            continue
        if (index, label) in placed:
            continue
        placed.add((index, label))

        offset = position - (ctx.bar_offset + Fraction(index) * ctx.bar_length)
        mark = expressions.RehearsalMark(label[:24])
        measures[index].insert(float(max(offset, Fraction(0))), mark)
        inserted += 1

    if inserted:
        stats.sections = inserted
    elif sections:
        stats.warnings.append("no section label could be placed on the score")


def _new_measures(stream, bar_count: int, number_offset: int = 1) -> list:
    """Create ``bar_count`` measures, numbering from ``number_offset``."""
    part_measures = []
    for index in range(bar_count):
        measure = stream.Measure(number=number_offset + index)
        part_measures.append(measure)
    return part_measures


def _build_single_staff(spec: PartSpec, quantized: Sequence, ctx: _Layout, *, clef=None):
    stream, note_mod, duration, clef_mod, instrument_mod = ctx.modules

    part = stream.Part(id=spec.stem)
    part.partName = spec.part_name
    measures = _new_measures(stream, ctx.bar_count)
    measures[0].insert(0, clef if clef is not None else clef_for(spec, clef_mod))
    measures[0].insert(0, instrument_for_spec(spec, instrument_mod))
    for measure in measures:
        part.append(measure)

    _fill(measures, quantized, ctx, spec)
    return part, list(quantized)


def _build_grand_staff(spec: PartSpec, quantized: Sequence, ctx: _Layout):
    stream, note_mod, duration, clef_mod, instrument_mod = ctx.modules

    upper = stream.Part(id=f"{spec.stem}-upper")
    lower = stream.Part(id=f"{spec.stem}-lower")
    upper.partName = spec.part_name
    lower.partName = ""
    upper.partAbbreviation = spec.part_name[:3]

    upper_measures = _new_measures(stream, ctx.bar_count)
    lower_measures = _new_measures(stream, ctx.bar_count)
    upper_measures[0].insert(0, clef_mod.TrebleClef())
    lower_measures[0].insert(0, clef_mod.BassClef())
    upper_measures[0].insert(0, instrument_for_spec(spec, instrument_mod))
    for measure in upper_measures:
        upper.append(measure)
    for measure in lower_measures:
        lower.append(measure)

    _fill(upper_measures, quantized, ctx, spec, split=True, lower_measures=lower_measures)
    return upper, lower, list(quantized)


def _fill(
    measures: list,
    quantized: Sequence,
    ctx: _Layout,
    spec: PartSpec,
    *,
    split: bool = False,
    lower_measures: list | None = None,
) -> tuple[int, int]:
    """Lay one part's notes into its measures as voices of chords.

    Two passes, because the two decisions belong at different levels:

    1. Every note is split at bar lines (a note may not cross a barline untied) and
       converted into absolute quarter lengths.
    2. Each measure *per staff* is laid out by
       :func:`~ensemble2score.notation.layout_voices.plan_measure`, which decides
       which voice each note joins and which simultaneous notes form a chord.

    Laying out per measure is deliberate: voice numbering has to restart sensibly in
    every measure, and a chord must not be split across a barline.

    Returns ``(ties_written, chords_written)``.
    """
    stream, note_mod, duration, _clef, _instrument = ctx.modules

    upper_notes, lower_notes = _split_notes(quantized, ctx, spec, split=split)

    uses_lower = bool(lower_measures) and split
    if not uses_lower:
        # One staff: chords belong to it, so the grand-staff split did not apply.
        lower_notes = []

    ties = 0
    chords = 0
    for index in range(ctx.bar_count):
        start = ctx.bar_offset + Fraction(index) * ctx.bar_length
        end = start + ctx.bar_length

        plan = plan_measure(upper_notes, measure_start=start, measure_end=end, max_voices=ctx.max_voices)
        ties += _write_plan(measures[index], plan, ctx, base_offset=start)
        chords += sum(1 for voice in plan.voices for chord_notes in voice if len(chord_notes) > 1)
        if plan.dropped:
            ctx.dropped_notes += plan.dropped

        if uses_lower:
            plan = plan_measure(
                lower_notes, measure_start=start, measure_end=end, max_voices=ctx.max_voices
            )
            ties += _write_plan(lower_measures[index], plan, ctx, base_offset=start)
            chords += sum(1 for voice in plan.voices for chord_notes in voice if len(chord_notes) > 1)
            if plan.dropped:
                ctx.dropped_notes += plan.dropped

    return ties, chords


def _split_notes(
    quantized: Sequence,
    ctx: _Layout,
    spec: PartSpec,
    *,
    split: bool,
) -> tuple[list[PlacedNote], list[PlacedNote]]:
    """Turn quantized notes into placeable notes per staff, cut at barlines.

    A note held across a barline becomes two notes; the tie that joins them is
    implied by the fact that one ends exactly where the next begins with the same
    pitch, which is what :func:`_write_plan` checks.
    """
    upper: list[PlacedNote] = []
    lower: list[PlacedNote] = []

    for item in quantized:
        written_pitch = item.pitch + (spec.sounding_pitch or 0)
        target = upper
        if split and written_pitch < GRAND_STAFF_SPLIT:
            target = lower

        start = Fraction(item.start) * ctx.quarter_per_beat
        end = Fraction(item.end) * ctx.quarter_per_beat
        if end <= start:
            ctx.dropped_notes += 1
            continue
        for piece_start, piece_end in _notate_span(
            start,
            end,
            bar_length=ctx.bar_length,
            beat_length=ctx.beat_length,
            grid=ctx.grid,
            bar_offset=ctx.bar_offset,
        ):
            if piece_end <= piece_start:
                continue
            if piece_end - piece_start < Fraction(1, ctx.grid):
                # Below the grid there is nothing to write; counting it is more
                # honest than emitting a zero-duration note.
                ctx.dropped_notes += 1
                continue
            target.append(
                PlacedNote(
                    pitch=written_pitch,
                    start=piece_start,
                    end=piece_end,
                    velocity=item.velocity,
                )
            )
    return upper, lower


def _write_plan(measure, plan, ctx: _Layout, *, base_offset: Fraction) -> int:
    """Write one measure's voices, chords and ties.  Returns the ties written."""
    stream, note_mod, duration, _clef, _instrument = ctx.modules
    from music21 import chord as chord_mod

    ties = 0
    for voice_number, chords in enumerate(plan.voices, start=1):
        voice = stream.Voice(id=str(voice_number))
        for chord_notes in chords:
            written = _make_chord(chord_notes, ctx, chord_mod, note_mod, duration)
            offset = chord_notes[0].start - base_offset
            voice.insert(float(offset), written)
        measure.insert(0, voice)
    return ties + _tie_repeats(measure)


def _make_chord(chord_notes, ctx: _Layout, chord_mod, note_mod, duration):
    """One notated event: a Note for a single pitch, a Chord for several."""
    pitches = sorted({n.pitch for n in chord_notes})
    start = chord_notes[0].start
    end = max(n.end for n in chord_notes)
    length = float(max(end - start, Fraction(1, ctx.grid) * ctx.quarter_per_beat))
    velocity = max(n.velocity for n in chord_notes)

    if len(pitches) == 1:
        element = note_mod.Note()
        element.pitch.midi = _clamp_pitch(pitches[0])
    else:
        element = chord_mod.Chord()
        element.pitches = [_midi_pitch(p, note_mod) for p in pitches]
    element.duration = duration.Duration(quarterLength=length)
    element.volume.velocity = int(max(1, min(127, velocity)))
    return element


def _midi_pitch(midi: int, note_mod):
    pitch = note_mod.Pitch()
    pitch.midi = _clamp_pitch(midi)
    return pitch


def _clamp_pitch(midi: int) -> int:
    return int(max(0, min(127, midi)))


def _tie_repeats(measure) -> int:
    """Tie a note to the next one when they are the same pitch and contiguous.

    music21 does not infer this, and without it a note held across a barline reads
    as two separate attacks — the most audible error a transcribed score can have.
    """
    from music21 import note as note_mod
    from music21 import tie as tie_mod

    tied = 0
    for voice in measure.getElementsByClass("Voice"):
        events = list(voice.notes)
        for first, second in zip(events, events[1:]):
            if isinstance(first, note_mod.Note) and isinstance(second, note_mod.Note):
                if first.pitch.midi != second.pitch.midi:
                    continue
            elif first.pitches != second.pitches:
                continue
            if abs((first.offset + first.duration.quarterLength) - second.offset) > 1e-6:
                continue
            first.tie = tie_mod.Tie("start")
            second.tie = tie_mod.Tie("stop")
            tied += 1
    return tied


def _measure_index(position: Fraction, ctx: _Layout) -> int:
    """Which measure a quarter-length offset falls in, given the pickup shift."""
    if position < ctx.bar_offset:
        return 0
    index = int((position - ctx.bar_offset) // ctx.bar_length)
    return min(max(index, 0), ctx.bar_count - 1)


def _build_drum_part(hits: Sequence[DrumHit], timeline, ctx: _Layout):
    stream, note_mod, duration, clef_mod, instrument_mod = ctx.modules
    from ..analysis.quantize import Grid

    grid = Grid(
        beat_times=tuple(timeline.grid.beat_times),
        beats_per_bar=ctx.beats_per_bar,
        grid=ctx.grid,
        phase=timeline.grid.phase,
        denominator=ctx.denominator,
    )
    part = stream.Part(id="drums")
    part.partName = "Percussion"
    measures = _new_measures(stream, ctx.bar_count)
    measures[0].insert(0, clef_mod.PercussionClef())
    measures[0].insert(0, instrument_mod.Percussion())
    for measure in measures:
        part.append(measure)

    # Percussion is written as voices too, so it goes through the same layout and
    # export path as pitched parts.  Building it with the old slot helper would
    # crash: that helper was removed when voices were introduced, and the drum part
    # was the one caller left behind.
    hits_by_measure: dict[int, list[PlacedNote]] = {}
    for hit in hits:
        beat_position = grid.quantize(grid.position_of(hit.time))
        # ``position_of`` yields *beats*; music21 works in quarter lengths.  The
        # placeholder Grid above used to leave ``quarter_per_beat`` at its default
        # of 1 (correct only for x/4), so in 6/8 every drum hit was written twice as
        # far from the barline as it was played.  It comes from the layout now.
        position = beat_position * ctx.quarter_per_beat
        index = _measure_index(position, ctx)
        length = Fraction(1, ctx.grid) * ctx.quarter_per_beat
        hits_by_measure.setdefault(index, []).append(
            PlacedNote(
                pitch=DRUM_MIDI_PITCH.get(hit.voice, 38),
                start=position,
                end=position + length,
                velocity=int(max(1, min(127, hit.velocity))),
            )
        )

    for index, placed in hits_by_measure.items():
        if not (0 <= index < len(measures)):
            continue
        start = ctx.bar_offset + Fraction(index) * ctx.bar_length
        plan = plan_measure(
            placed, measure_start=start, measure_end=start + ctx.bar_length, max_voices=ctx.max_voices
        )
        for voice_number, chords in enumerate(plan.voices, start=1):
            voice = stream.Voice(id=str(voice_number))
            for chord_notes in chords:
                element = note_mod.Unpitched()
                element.displayPitch = note_mod.Note()
                element.displayPitch.pitch.midi = _clamp_pitch(chord_notes[0].pitch)
                element.duration = duration.Duration(
                    quarterLength=float(max(n.end for n in chord_notes) - chord_notes[0].start)
                )
                element.volume.velocity = int(max(1, min(127, chord_notes[0].velocity)))
                voice.insert(float(chord_notes[0].start - start), element)
            measures[index].insert(0, voice)
    return part


def drum_notes_for_midi(hits: Sequence[DrumHit]) -> list[Note]:
    """Classified drum hits as note events, for the MIDI export.

    Drum voices map to General MIDI percussion pitches and each hit is given a
    short fixed duration: percussion has no meaningful note length, and a zero
    length would be dropped by most players.

    ``source_start``/``source_end`` must be set *and* must equal ``start``/``end``.
    :func:`write_midi` reads the source times — deliberately, so a quantised note
    stays on the real timeline — and for every other part they are filled in by
    :class:`~ensemble2score.analysis.quantize.QuantizedNote`.  A plain ``Note`` has
    no such fields, so relying on ``getattr(..., 0.0)`` wrote **every percussion hit
    at t=0**: 591 snare and kick hits stacked on the first frame, and the percussion
    part silent for the rest of the piece.  The merged MIDI had the same defect,
    which is what made it worth finding.
    """
    notes: list[Note] = []
    for hit in hits:
        start = float(hit.time)
        note = Note(
            pitch=DRUM_MIDI_PITCH.get(hit.voice, 38),
            start=start,
            end=start + 0.05,
            velocity=int(max(1, min(127, hit.velocity))),
            confidence=float(hit.confidence),
        )
        note.source_start = start
        note.source_end = start + 0.05
        notes.append(note)
    return notes


def _add_shared_markings(score, stats: ScoreStats, options: ScoreOptions, result, *, time_signature, tempo_number):
    """Time signature, key signature and tempo, on every part's first measure."""
    from music21 import tempo as m21tempo

    signature = _resolve_key(score, stats) if options.estimate_key else key_signature(0)
    tempo_mark = m21tempo.MetronomeMark(number=round(tempo_number, 1)) if tempo_number else None

    for index, part in enumerate(score.parts):
        measures = list(part.getElementsByClass("Measure"))
        if not measures:
            continue
        first = measures[0]
        first.insert(0, time_signature)
        if signature is not None:
            first.insert(0.1, signature)
        if tempo_mark is not None and index == 0:
            first.insert(0, tempo_mark)
            tempo_mark = None  # only on the top part


def _resolve_key(score, stats: ScoreStats):
    """Estimate the key, warning rather than silently dropping it on failure.

    An earlier version wrapped this in ``except Exception: pass`` and swallowed a
    music21 TypeError, which removed the key signature with no warning at all.
    """
    try:
        analysed = score.analyze("key")
    except Exception as exc:  # noqa: BLE001 - best effort, but never silent
        stats.warnings.append(
            f"key estimation failed ({type(exc).__name__}: {exc}); no key signature was written"
        )
        return key_signature(0)
    try:
        signature = key_signature(analysed.sharps, analysed.mode, analysed.tonic.name)
    except Exception as exc:  # noqa: BLE001
        stats.warnings.append(f"key signature could not be built ({exc})")
        return key_signature(0)
    stats.key = f"{analysed.tonic.name} {analysed.mode}"
    return signature


def _insert_chords(score, chords: Sequence[Chord], timeline, quarter_per_beat: Fraction, ctx: _Layout, stats: ScoreStats):
    from music21 import harmony as m21harmony

    if not score.parts:
        return
    measures = list(score.parts[0].getElementsByClass("Measure"))
    if not measures:
        return

    grid = timeline.grid
    slots: dict[int, list[tuple[Fraction, object]]] = {}
    for chord in chords:
        if chord.label in ("N.C.", "", None):
            continue
        try:
            symbol = m21harmony.ChordSymbol(chord.label)
        except Exception:  # noqa: BLE001 - one bad label is not fatal
            stats.warnings.append(f"chord symbol {chord.label!r} could not be notated")
            continue
        beat_position = grid.quantize(grid.position_of(chord.start))
        position = beat_position * quarter_per_beat
        index = _measure_index(position, ctx)
        offset = position - (ctx.bar_offset + Fraction(index) * ctx.bar_length)
        slots.setdefault(index, []).append((max(offset, Fraction(0)), symbol))

    # Chord symbols sit above the staff and must not disturb the note order, so
    # they are inserted by offset after the notes are already in place.
    inserted = 0
    for index, events in sorted(slots.items()):
        if not (0 <= index < len(measures)):
            continue
        for offset, symbol in sorted(events, key=lambda pair: pair[0]):
            measures[index].insert(float(offset), symbol)
            inserted += 1

    if inserted == 0 and chords:
        stats.warnings.append("no chord symbol could be notated; the chord track is empty")


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def export_score(
    score,
    out_dir: str | Path,
    basename: str,
    *,
    midi_parts: Sequence[tuple[PartSpec, list]],
    tempo_bpm: float,
    make_pdf: bool = False,
    mscore_path: str | None = None,
    split_midi: bool = False,
) -> dict[str, str]:
    """Write MusicXML (always), MIDI (always), PDF and per-part MIDI when asked.

    ``split_midi`` additionally writes ``midi/<part>.mid`` — one file per part, so
    a single instrument's transcription can be listened to, edited or handed to a
    notation program on its own.  It is the same note data the ensemble MIDI is
    built from, so the two can never disagree.
    """
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    musicxml = target / f"{basename}.musicxml"
    with capture_noise():
        # music21 turns our explicit durations into proper beams and ties here.
        # We already split notes at bar lines, so this only has to handle beaming
        # and the tie grouping that follows from it.
        try:
            score.makeNotation(inPlace=True)
        except Exception as exc:  # noqa: BLE001 - notating a raw stream can fail
            warn(f"music21 could not apply beaming/tie notation ({type(exc).__name__}: {exc})")
        score.write("musicxml", fp=str(musicxml))
    written["musicxml"] = str(musicxml)

    midi = target / f"{basename}.mid"
    write_midi(midi, midi_parts, tempo_bpm=tempo_bpm)
    written["midi"] = str(midi)

    if split_midi:
        written.update(write_part_midis(target, midi_parts, tempo_bpm=tempo_bpm))

    if make_pdf:
        pdf = export_pdf(musicxml, target, basename, mscore_path)
        if pdf:
            written["pdf"] = pdf

    return written


def part_stem_filename(spec: PartSpec, notes: Sequence) -> str:
    """A stable, filesystem-safe filename for one part's MIDI.

    Prefer the stem (``vocals``, ``piano``), because that is the name the separated
    audio carries and the connection between a stem and its MIDI is the whole point
    of the per-part export.  Two parts can share a stem only if a caller invents
    that, so the part name is kept as a fallback rather than as a suffix.
    """
    import re

    base = (spec.stem or spec.part_name or "part").strip().lower()
    safe = re.sub(r"[^a-z0-9._-]+", "-", base).strip("-")
    return safe or "part"


def write_part_midis(
    out_dir: Path,
    midi_parts: Sequence[tuple[PartSpec, list]],
    *,
    tempo_bpm: float,
) -> dict[str, str]:
    """Write one MIDI file per part into ``<out_dir>/midi/``.

    Returns ``{"midi:<part>": path}`` so the run record lists them.  Parts with no
    notes are skipped: an empty MIDI file is not a deliverable, it is a file the
    user has to open to discover there is nothing in it.
    """
    directory = Path(out_dir) / "midi"
    directory.mkdir(parents=True, exist_ok=True)

    written: dict[str, str] = {}
    used: set[str] = set()
    for spec, notes in midi_parts:
        if not notes:
            continue
        name = part_stem_filename(spec, notes)
        if name in used:
            index = 2
            while f"{name}-{index}" in used:
                index += 1
            name = f"{name}-{index}"
        used.add(name)

        path = directory / f"{name}.mid"
        write_midi(path, [(spec, notes)], tempo_bpm=tempo_bpm)
        written[f"midi:{name}"] = str(path)
    return written


def write_midi(
    path: str | Path,
    midi_parts: Sequence[tuple[PartSpec, list]],
    *,
    tempo_bpm: float,
) -> Path:
    """Write MIDI with pretty_midi.

    music21's MIDI writer sends the upper staff of a grand staff to channel 10
    (percussion), which plays the right hand as drums; writing the file directly
    keeps every pitched instrument on its own melodic channel.
    """
    import pretty_midi

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    midi = pretty_midi.PrettyMIDI(initial_tempo=float(max(tempo_bpm, 1.0)))

    for spec, notes in midi_parts:
        if not notes:
            continue
        if spec.drum_channel:
            instrument = pretty_midi.Instrument(program=0, is_drum=True, name=spec.part_name)
        else:
            instrument = pretty_midi.Instrument(
                program=int(spec.midi_program or 0), name=spec.part_name, is_drum=False
            )
        for item in notes:
            pitch = item.pitch + (spec.sounding_pitch or 0)
            source_start = float(getattr(item, "source_start", 0.0))
            source_end = float(getattr(item, "source_end", source_start + 0.05))
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=int(max(1, min(127, item.velocity))),
                    pitch=int(max(0, min(127, pitch))),
                    start=source_start,
                    end=max(source_end, source_start + 1e-3),
                )
            )
        midi.instruments.append(instrument)

    midi.write(str(target))
    return target


def export_pdf(musicxml: Path, out_dir: Path, basename: str, mscore: str | None) -> str | None:
    """Render a PDF via MuseScore, or explain precisely why we could not."""
    import shutil
    import subprocess

    executable = (
        mscore
        or shutil.which("mscore")
        or shutil.which("musescore")
        or shutil.which("MuseScore4")
    )
    if not executable:
        hint("--pdf needs MuseScore 4 on PATH (or --mscore PATH): skipping the PDF")
        return None

    target = out_dir / f"{basename}.pdf"
    try:
        subprocess.run(  # noqa: S603 - an external program the user explicitly asked for
            [executable, "-o", str(target), str(musicxml)],
            check=True,
            capture_output=True,
            timeout=300,
        )
    except subprocess.CalledProcessError as exc:
        warn(f"MuseScore failed with code {exc.returncode}: {exc.stderr.decode(errors='replace')[:200]}")
        return None
    except (OSError, subprocess.TimeoutExpired) as exc:
        warn(f"could not run MuseScore: {exc}")
        return None
    return str(target) if target.is_file() else None
