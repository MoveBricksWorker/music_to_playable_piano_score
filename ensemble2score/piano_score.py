"""Render a two-hand arrangement into a real piano score: MusicXML, MIDI, PDF.

The pitches and their grid positions come from the model; everything mechanical comes
from here, so that the engraved result is reproducible from the reply alone:

* **Durations** are derived from the next event in the same hand (see
  :func:`~ensemble2score.piano_arrange.sounding_lengths`) and then split at barlines
  and tied, using the same rules the ensemble score uses.  The model is never asked
  for rhythm, because a quantised transcription has no reliable rhythm to copy and an
  invented one cannot be checked.
* **Voices** are allocated so that each hand is monophonic-with-chords.  A chord is a
  set of notes that start *and end* together, which is what MusicXML can carry;
  grouping on onset alone produces notation no reader can represent.  That rule is
  not reimplemented here — :mod:`ensemble2score.notation.layout_voices` already
  established it.
* **Barlines are constructed**, not inferred, from the bar grid the pipeline already
  inferred from the audio.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from .core.errors import OutputError
from .piano_arrange import (
    TICKS_PER_BEAT,
    Arrangement,
    ArrangementInput,
    sounding_lengths,
)

__all__ = ["PianoScoreStats", "build_piano_score", "export_piano_score", "hand_parts"]


@dataclass
class PianoScoreStats:
    measures: int = 0
    right_notes: int = 0
    left_notes: int = 0
    written_elements: int = 0
    ties: int = 0
    time_signature: str = "?"
    key: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "measures": int(self.measures),
            "right_notes": int(self.right_notes),
            "left_notes": int(self.left_notes),
            "notes": int(self.right_notes + self.left_notes),
            "written_elements": int(self.written_elements),
            "ties": int(self.ties),
            "time_signature": self.time_signature,
            "key": self.key,
            "warnings": list(self.warnings),
        }


def hand_parts(
    arrangement: Arrangement,
    material: ArrangementInput,
) -> dict[str, list[tuple[Fraction, Fraction, int]]]:
    """``{hand: [(start_ql, end_ql, pitch)]}`` in quarter lengths on the score's timeline.

    Quarter lengths are what music21 wants: one beat of ``x/4`` is a quarter note, so a
    tick is ``1 / ticks_per_beat`` of a quarter.

    Positions are converted to **score offsets**, where beat 1 of bar 1 is exactly
    ``beats_per_bar``.  That makes the pickup fall out of the arithmetic instead of
    needing a special case: an event before the first barline lands at a smaller
    offset, and the engraver turns it into a genuine short first measure rather than
    dropping it or stacking it onto the downbeat.
    """
    grid = material.grid
    per_bar = max(1, grid.beats_per_bar) * grid.ticks_per_beat
    out: dict[str, list[tuple[Fraction, Fraction, int]]] = {}
    for hand, events in (("RH", arrangement.right), ("LH", arrangement.left)):
        lengths = sounding_lengths(events, ticks_per_beat=grid.ticks_per_beat)
        notes: list[tuple[Fraction, Fraction, int]] = []
        for tick, pitches in sorted(events.items()):
            # `to_tick` subtracted the phase on the way in; adding it back here puts the
            # event where it belongs relative to the score's first downbeat.
            offset = tick + grid.phase * grid.ticks_per_beat
            start = _score_offset(offset, grid, per_bar)
            length = Fraction(
                max(1, int(round(lengths[tick] * grid.ticks_per_beat))), grid.ticks_per_beat
            )
            for pitch in pitches:
                notes.append((start, start + length, int(pitch)))
        out[hand] = notes
    return out


def _score_offset(offset: int, grid, per_bar: int) -> Fraction:
    """Grid offset in ticks -> quarter lengths on the score's timeline.

    The score's origin is where the **music** starts, so the first event sits at offset
    0 and the grid's phase is already accounted for by `offset`.  A negative offset is a
    pickup and lands at a negative quarter length, ahead of bar 1 — which is what the
    engraver needs in order to give it a short first measure of its own.
    """
    bar, within = divmod(offset, per_bar)
    return Fraction(bar * grid.beats_per_bar) + Fraction(within, grid.ticks_per_beat)


def _key_signature(key: str | None, stats: PianoScoreStats):
    """A music21 KeySignature for the run's key, or ``None``.

    The accidental count comes from music21's own key table (``Key("E-","major").sharps``
    is ``-3``) rather than from a literal.  Writing ``key_signature(0, ...)`` produces a
    *blank* signature, which engraves E-flat major with no flats and an accidental on
    every E, A and B in the piece — the difference between a score a player can read and
    one they cannot.
    """
    from music21 import key as m21key

    from .notation.score import key_signature

    if not key:
        return None
    # music21 spells a flat with a trailing dash ("E- major"), which is also how
    # run.json carries it, so the tonic is passed through unchanged: rewriting the dash
    # to "b" gives a name music21 does not recognise.
    tonic = key.replace(":", " ").split()[0]
    mode = "minor" if "minor" in key.lower() else "major"
    try:
        built = m21key.Key(tonic, mode)
        return key_signature(built.sharps, mode, tonic)
    except Exception as exc:  # noqa: BLE001 - a missing signature beats no score
        stats.warnings.append(
            f"could not notate the key signature for {key!r} ({type(exc).__name__}: {exc})"
        )
        return None


def _tick_from_score_offset(offset: Fraction, grid, per_bar: int) -> int:
    """Inverse of :func:`_score_offset`: quarter lengths back to a grid tick.

    Needed so a per-bar decision (the key in force) can be asked of the material, which
    indexes everything by tick.  The phase has to come back out, because `to_tick`
    subtracted it on the way in.
    """
    beats = offset / grid.beats_per_bar
    whole = int(beats)                       # completed bars
    within_beats = beats - whole
    within_ticks = within_beats * grid.ticks_per_beat
    return (
        whole * per_bar
        + int(round(float(within_ticks)))
        - grid.phase * grid.ticks_per_beat
    )


def build_piano_score(
    arrangement: Arrangement,
    material: ArrangementInput,
    *,
    title: str | None = None,
    composer: str | None = None,
):
    """Build a two-staff music21 score.  Returns ``(score, stats)``.

    The layout decisions are all here, and each was a bug before it was a rule:

    * **Measures come from the score timeline, not from bar indices.**  `hand_parts`
      places beat 1 of bar 1 at ``beats_per_bar``, so a pickup simply has a smaller
      offset.  Each measure is a half-open span, and a note is filed under the span that
      contains its **onset** — a note landing exactly on a barline starts the new measure.
      Deriving the measure index arithmetically instead put such a note in the bar it
      *ended*, where it was clipped to nothing and dropped, bar after bar.
    * **A note is cut at its own measure's end**, not at multiples of a nominal bar.  A
      pickup measure is short, and a note running past it has to be tied into bar 1;
      leaving the overflow uncut produced measures holding two bars of music and XML
      engravers refuse to open.
    * **Ties come only from cuts.**  Two *separate* events of one pitch that happen to
      touch are two attacks: tying them turns two eighths into a quarter and silently
      rewrites the rhythm.
    * **Trailing silent bars are kept.**  Only measures past the run's own bar count that
      every hand leaves empty are trimmed — dropping trailing silence in general lost a
      quiet ending one bar at a time.
    """
    from music21 import chord, clef, meter, note as m21note, stream, tempo

    from .notation.score import _notate_span

    stats = PianoScoreStats(
        time_signature=f"{material.grid.beats_per_bar}/4",
        key=material.key,
    )
    notes = hand_parts(arrangement, material)
    if not notes.get("RH") and not notes.get("LH"):
        raise OutputError("the arrangement has no notes: nothing to engrave")

    grid = material.grid
    per_bar = max(1, grid.beats_per_bar) * grid.ticks_per_beat
    bar_length = Fraction(grid.beats_per_bar)
    beat_length = Fraction(1)
    ties = 0

    # The run's own bar count is the floor: a song that ends quietly keeps its ending.
    if material.bars:
        bar_count = material.bars
    else:
        last_tick = max(
            [t for t in arrangement.right] + [t for t in arrangement.left] + [0]
        )
        bar_count = (last_tick + grid.phase * grid.ticks_per_beat) // per_bar + 1

    score = stream.Score()
    if title or composer:
        from music21 import metadata

        score.metadata = metadata.Metadata()
        if title:
            score.metadata.title = title
        if composer:
            score.metadata.composer = composer

    # One geometry for the whole score: barlines belong to the piece, not to a hand.
    #
    # The first measure is built here explicitly rather than left to the growth loop.
    # That is what silently shifted every barline by one bar: with no pickup the first
    # span came out as [bar_length, 2*bar_length), so notes in bar 1 had negative local
    # offsets, were clipped to nothing, and vanished — while the measure they belonged in
    # stayed empty.
    pickup = Fraction(grid.phase) if grid.phase else Fraction(0)
    has_pickup = bool(pickup)
    first_length = bar_length - pickup if has_pickup else bar_length
    spans: list[tuple[Fraction, Fraction]] = [(Fraction(0), first_length)]
    if has_pickup:
        spans.append((first_length, first_length + bar_length))

    def next_begin() -> Fraction:
        """Where the next full measure begins, counting bars from bar 1."""
        added = len(spans) - (1 if has_pickup else 0)
        return first_length + added * bar_length

    highest = build_bounds(notes)
    while spans[-1][1] < highest or len(spans) < bar_count:
        begin = next_begin()
        spans.append((begin, begin + bar_length))

    for hand, spec in (
        ("RH", {"name": "Piano (right)", "clef": clef.TrebleClef()}),
        ("LH", {"name": "Piano (left)", "clef": clef.BassClef()}),
    ):
        part = stream.Part(id=hand)
        part.partName = spec["name"]
        part.append(spec["clef"])
        part.append(meter.TimeSignature(f"{grid.beats_per_bar}/4"))
        signature = _key_signature(material.key, stats)
        if signature is not None:
            part.append(signature)
        if material.tempo_bpm:
            part.append(tempo.MetronomeMark(number=round(material.tempo_bpm, 1)))

        # Key changes, as a signature at the start of the bar they take effect in.  The key
        # in force is sampled per bar rather than per note, because a signature is a
        # property of the barline: emitting one mid-bar would be unreadable, and a
        # modulation in this material always lands on a bar.
        by_bar_key: dict[int, str] = {}
        for bar, (measure_start, _measure_end) in enumerate(spans):
            tick = _tick_from_score_offset(measure_start, grid, per_bar)
            key = material.key_at(tick)
            if key:
                by_bar_key[bar] = key

        by_bar: list[list[tuple[Fraction, Fraction, int]]] = [[] for _ in spans]
        for start, end, pitch in notes[hand]:
            by_bar[_measure_of(spans, start)].append((start, end, pitch))

        for bar, (measure_start, measure_end) in enumerate(spans):
            measure = stream.Measure(number=bar + 1)
            if bar == 0 and has_pickup:
                measure.paddingLeft = pickup
            length = measure_end - measure_start

            # A modulation is notated where it happens, so the accidentals that follow are
            # the exceptions rather than the rule.  The signature still has to be valid, so
            # a key music21 cannot spell is skipped rather than allowed to break the score.
            if bar in by_bar_key and bar > 0:
                current = by_bar_key[bar]
                previous = by_bar_key.get(bar - 1, material.key)
                if current != previous:
                    change = _key_signature(current, stats)
                    if change is not None:
                        measure.append(change)

            pieces: list[tuple[Fraction, Fraction, int, bool]] = []
            for onset, end, pitch in by_bar[bar]:
                start = max(onset - measure_start, Fraction(0))
                finish = min(end - measure_start, length)
                if finish <= start:
                    continue
                cut = _notate_span(
                    start,
                    finish,
                    bar_length=length,
                    beat_length=beat_length,
                    grid=grid.ticks_per_beat,
                    bar_offset=Fraction(0),
                )
                was_split = len(cut) > 1 or (end - measure_start) > length
                for piece_start, piece_end in cut:
                    piece_end = min(piece_end, length)
                    if piece_start >= length or piece_end <= piece_start:
                        continue
                    pieces.append((piece_start, piece_end, pitch, was_split))

            grouped: dict[tuple[Fraction, Fraction], list[int]] = {}
            split_keys: set[tuple[Fraction, Fraction]] = set()
            for piece_start, piece_end, pitch, was_split in pieces:
                key = (piece_start, piece_end)
                grouped.setdefault(key, []).append(pitch)
                if was_split:
                    split_keys.add(key)

            voice = stream.Voice()
            for (piece_start, piece_end), group in sorted(grouped.items()):
                unique = sorted(set(group))
                element = chord.Chord(unique) if len(unique) > 1 else m21note.Note(unique[0])
                element.quarterLength = piece_end - piece_start
                if (piece_start, piece_end) in split_keys:
                    element._from_split = True
                voice.insert(piece_start, element)
            if len(voice.notes):
                measure.insert(0, voice)
            else:
                measure.insert(0, m21note.Rest(quarterLength=length))

            ties += _tie_contiguous(measure)
            part.append(measure)

        if hand == "RH":
            stats.right_notes = sum(len(v) for v in arrangement.right.values())
        else:
            stats.left_notes = sum(len(v) for v in arrangement.left.values())
        score.insert(0, part)

    # music21 pads every hand to the same measure count.  Only measures that sit beyond
    # the run's own bar count **and** that every hand leaves empty are trimmed.
    removable = (
        min(_trailing_empty_measures(part) for part in score.parts) if score.parts else 0
    )
    excess = max(0, _measure_count(score) - bar_count)
    drop = min(removable, excess)
    if drop:
        for part in score.parts:
            measures = list(part.getElementsByClass("Measure"))
            for measure in measures[len(measures) - drop:]:
                part.remove(measure)

    # The measure count as the score actually has it.  A pickup is a real measure, so a
    # piece with an anacrusis comes out one measure longer than the run's bar count —
    # which is correct, and is why this is read back from the score instead of assumed.
    stats.measures = _measure_count(score)
    stats.ties = ties
    stats.written_elements = sum(
        len(element.pitches) if element.isChord else 1
        for part in score.parts
        for element in part.recurse().notes
    )
    return score, stats


def build_bounds(notes: dict[str, list[tuple[Fraction, Fraction, int]]]) -> Fraction:
    """The last offset either hand reaches — the music's end on the score timeline."""
    return max(
        (end for hand_notes in notes.values() for _s, end, _p in hand_notes),
        default=Fraction(0),
    )


def _measure_of(spans: list[tuple[Fraction, Fraction]], start: Fraction) -> int:
    """Index of the measure that **contains** ``start``.

    Boundaries are half-open and searched rather than computed: a note landing exactly
    on a barline belongs to the measure that begins there, and an arithmetic shortcut
    gets that case wrong for every bar in the piece.
    """
    for index, (begin, end) in enumerate(spans):
        if begin <= start < end:
            return index
    # Past the end of the last measure (or before the first): clamp rather than crash.
    return 0 if start < spans[0][0] else len(spans) - 1


def _measure_count(score) -> int:
    """Measures in the longest part, which is what the writer pads to."""
    return max(
        (len(list(part.getElementsByClass("Measure"))) for part in score.parts), default=0
    )


def _trailing_empty_measures(part) -> int:
    """How many measures at the end of this part hold nothing but rests."""
    from music21 import note as m21note

    measures = list(part.getElementsByClass("Measure"))
    count = 0
    for measure in reversed(measures):
        played = [
            element
            for element in measure.recurse().notes
            if not isinstance(element, m21note.Rest)
        ]
        if played:
            break
        count += 1
    return count


def _tie_contiguous(measure) -> int:
    """Tie notes of one pitch that meet exactly **where a note was split**.

    The distinction this function has to respect is the difference between one note
    held over a barline and two repeated notes that happen to touch:

    * a note the engraver had to split (it crossed a barline, or outran a beat group)
      is emitted as adjacent pieces that must be tied, or it reads as two attacks —
      the most audible error a transcribed score can have;
    * two *separate* events of the same pitch that happen to meet are two attacks, and
      tying them silently rewrites the rhythm: two eighths become a quarter.

    Only the first is a tie.  Lengths are compared rather than positions, because a
    split piece never fills its whole beat group while an unsplit repeated note does —
    which is exactly what :func:`~ensemble2score.notation.score._notate_span` decided
    when it cut.  Ties within one measure are counted here; the caller sums them.
    """
    from music21 import tie as tie_mod

    tied = 0
    for voice in measure.getElementsByClass("Voice"):
        events = list(voice.notes)
        for first, second in zip(events, events[1:]):
            if first.pitches != second.pitches:
                continue
            if abs((first.offset + first.duration.quarterLength) - second.offset) > 1e-6:
                continue
            if not _is_split_piece(first) and not _is_split_piece(second):
                # Two plain events that meet: a re-articulation, not a held note.
                continue
            first.tie = tie_mod.Tie("start")
            second.tie = tie_mod.Tie("stop")
            tied += 1
    return tied


def _is_split_piece(element) -> bool:
    """Did this element come out of a split rather than straight from an event?

    A split piece is one that had to be cut to fit: either it fills less than the
    distance to the next event, or it begins or ends away from a whole beat.  Both are
    recorded by the splitter, which never produces a piece equal to a plain event.
    """
    return bool(getattr(element, "_from_split", False))


def export_piano_score(
    score,
    out_dir: str | Path,
    basename: str,
    *,
    arrangement: Arrangement | None = None,
    material: ArrangementInput | None = None,
    make_pdf: bool = False,
    mscore: str | None = None,
) -> dict[str, str]:
    """Write ``.musicxml``, ``.mid`` and optionally ``.pdf``; return what was written."""
    from .notation.score import export_pdf

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    musicxml = directory / f"{basename}.musicxml"
    try:
        score.write("musicxml", fp=str(musicxml))
    except Exception as exc:  # noqa: BLE001
        raise OutputError(f"could not write the piano MusicXML: {exc}") from exc
    written["musicxml"] = str(musicxml)

    if arrangement is not None and material is not None:
        midi = directory / f"{basename}.mid"
        _write_midi(arrangement, material, midi)
        written["midi"] = str(midi)

    if make_pdf:
        pdf = export_pdf(musicxml, directory, basename, mscore)
        if pdf:
            written["pdf"] = pdf
    return written


def _write_midi(arrangement: Arrangement, material: ArrangementInput, path: Path) -> Path:
    """A piano MIDI that can be listened to against the original.

    Written from the arrangement rather than from the music21 score, so that what is
    heard is exactly what the model proposed and not a re-reading of the engraving.
    """
    import pretty_midi

    grid = material.grid
    midi = pretty_midi.PrettyMIDI(
        initial_tempo=float(material.tempo_bpm) if material.tempo_bpm else 120.0
    )
    beat_seconds = 60.0 / (material.tempo_bpm or 120.0)
    for hand, program in (("RH", 0), ("LH", 0)):
        events = arrangement.right if hand == "RH" else arrangement.left
        if not events:
            continue
        lengths = sounding_lengths(events, ticks_per_beat=grid.ticks_per_beat)
        instrument = pretty_midi.Instrument(program=program, name=f"Piano {hand}")
        for tick, pitches in sorted(events.items()):
            # A pickup can sit ahead of tick 0, giving a negative time.  Clamp the start
            # **and derive the end from the clamped value**: mixing a clamped start with an
            # unclamped end produced end == start, which pretty_midi rejects outright.
            raw_start = tick * beat_seconds / grid.ticks_per_beat
            start = max(0.0, raw_start)
            length = max(lengths[tick] * beat_seconds, 0.02)
            end = start + length
            for pitch in pitches:
                instrument.notes.append(
                    pretty_midi.Note(
                        velocity=80, pitch=int(pitch), start=start, end=end,
                    )
                )
        midi.instruments.append(instrument)
    midi.write(str(path))
    return path


def load_arrangement_json(path: str | Path) -> tuple[Arrangement, dict]:
    """Read a saved arrangement (as the CLI stores one) back into an object."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    arrangement = Arrangement(track=payload.get("track", ""))
    arrangement.right = {
        int(tick): [int(p) for p in pitches]
        for tick, pitches in (payload.get("right") or {}).items()
    }
    arrangement.left = {
        int(tick): [int(p) for p in pitches]
        for tick, pitches in (payload.get("left") or {}).items()
    }
    arrangement.warnings = list(payload.get("warnings") or [])
    return arrangement, payload
