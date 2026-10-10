"""Build the material an LLM needs to write a playable piano score.

The problem this solves
-----------------------
Two transcribers disagree, and each is right about different things:

* **SheetSage2** produces a reliable *melody* — monophonic, continuous, in the right
  register (measured duty cycle 64-90% against basic-pitch's 31-49% on the same vocal
  stems).  It is not a polyphonic transcriber, so it says little about texture.
* **basic-pitch** produces more complete but messier material: its polyphony on a
  separated stem is largely invention (which is why the pipeline's confidence floor
  removes 60-80% of it), yet it is the only source that reports the actual
  accompaniment voicing.

Merging them is a *symbolic* judgement — which line carries the music, what to keep,
how to spread it across two hands — which is exactly what a language model is good at
and what no threshold can do.  So this module does not decide the arrangement.  It
assembles the evidence, on one shared bar grid, in a form a model can actually read,
and it puts the deterministic parts (positions, grouping, verification) in code.

What it deliberately does *not* ask the model for
-------------------------------------------------
Note **durations**.  The model is asked for the highest note of a top line and the
lowest note of a bottom line; the sounding length between consecutive events is then
derived arithmetically.  A quantised transcription has no reliable durations to begin
with (the handoff measured this as the single largest error source), and asking for
them invites invented rhythm that cannot be checked.  Keeping duration out of the
model's job keeps its output verifiable.

Positions are ``bar:beat.tick`` on the run's own beat grid, so a bar line is where the
pipeline already established it and the model never has to infer metre.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "Voice",
    "ArrangementInput",
    "Arrangement",
    "load_arrangement_input",
    "render_input_text",
    "parse_arrangement",
    "render_arrangement_abc",
    "NOTE_NAMES_SHARP",
    "NOTE_NAMES_FLAT",
]


NOTE_NAMES_SHARP = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
NOTE_NAMES_FLAT = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")

# Position grid: ticks per beat.  Sixteen to the beat is the same resolution the
# notation stage quantises to, so a position here is a position there.
TICKS_PER_BEAT = 4


def pitch_name(pitch: int, *, flats: bool = False) -> str:
    names = NOTE_NAMES_FLAT if flats else NOTE_NAMES_SHARP
    return f"{names[int(pitch) % 12]}{int(pitch) // 12 - 1}"


def _uses_flats(key: str | None) -> bool:
    """Should this key be spelled with flats?

    The test has to run on the **relative major**, not on the tonic as written.  An earlier
    version only checked the tonic against a table of flat *major* keys, so ``F minor`` and
    ``C minor`` — whose relative majors are A-flat and E-flat, both flat keys — were spelled
    with sharps.  That is how ``Db:maj7`` came out as ``C#`` and ``Ab`` as ``G#`` in a score
    with four flats in the signature.
    """
    if not key:
        return False
    tonic = key.replace(":", " ").split()[0].strip().replace("-", "b")
    minor = "min" in key.lower()
    if minor:
        # Up a minor third to the relative major.
        relative = {
            "C": "Eb", "C#": "E", "D": "F", "Eb": "Gb", "E": "G", "F": "Ab",
            "F#": "A", "G": "Bb", "G#": "B", "A": "C", "Bb": "Db", "B": "D",
        }.get(tonic, tonic)
        tonic = relative
    return tonic in {"F", "Bb", "Eb", "Ab", "Db", "Gb", "Cb"}


# --------------------------------------------------------------------------- #
# the run's own timing
# --------------------------------------------------------------------------- #
@dataclass
class BarGrid:
    """Seconds <-> ``bar:beat.tick`` on the beat grid the pipeline already inferred.

    Built from ``run.json`` rather than re-derived, so the model's bar 1 is the score's
    bar 1 and its barlines land where the engraved score's do.

    ``phase`` is the beat index the first barline falls on.  The grid is indexed from
    tick 0, so when the music starts *before* the tracker's first downbeat — which
    happens whenever the tracker places bar 1 a beat or two late, as on 夜明けの歌 and
    blessing — the opening notes land at negative ticks.  That region is a genuine
    pickup: the model is shown those positions as ``0:4.3`` and may write them back, so
    they are kept and notated as an anacrusis rather than discarded.
    """

    beat_times: list[float]
    beats_per_bar: int = 4
    phase: int = 0
    ticks_per_beat: int = TICKS_PER_BEAT

    @property
    def pickup_ticks(self) -> int:
        """How far before tick 0 the music may legitimately start.

        One bar's worth before the first barline, or zero when the grid already begins
        at a barline — never a licence to accept anything at all.
        """
        return self.phase * self.ticks_per_beat if self.phase else self.ticks_per_beat

    def beat_position(self, seconds: float) -> float:
        """Seconds -> fractional beat index, linear between detected beats."""
        beats = self.beat_times
        if not beats:
            return 0.0
        if len(beats) == 1:
            return (seconds - beats[0]) * 2.0
        from bisect import bisect_right

        index = bisect_right(beats, seconds)
        if index == 0:
            span = beats[1] - beats[0]
            return (seconds - beats[0]) / span if span > 0 else 0.0
        if index >= len(beats):
            span = beats[-1] - beats[-2]
            return (len(beats) - 1) + ((seconds - beats[-1]) / span if span > 0 else 0.0)
        start, end = beats[index - 1], beats[index]
        span = end - start
        return (index - 1) + ((seconds - start) / span if span > 0 else 0.0)

    def position(self, seconds: float) -> tuple[int, int, int]:
        """Seconds -> ``(bar, beat, tick)``, all 0-based and tick-quantised."""
        beats = self.beat_position(seconds)
        ticks = int(round(beats * self.ticks_per_beat))
        return self.from_tick(ticks)

    def from_tick(self, ticks: int) -> tuple[int, int, int]:
        offset = ticks + self.phase * self.ticks_per_beat
        per_bar = self.beats_per_bar * self.ticks_per_beat
        bar, within = divmod(offset, per_bar)
        beat, tick = divmod(within, self.ticks_per_beat)
        return bar, beat, tick

    def to_tick(self, bar: int, beat: int, tick: int) -> int:
        per_bar = self.beats_per_bar * self.ticks_per_beat
        return bar * per_bar + beat * self.ticks_per_beat + tick - self.phase * self.ticks_per_beat

    def label(self, bar: int, beat: int, tick: int) -> str:
        """1-based, for humans and for the model: ``3:2.1`` is bar 3, beat 2, tick 1."""
        return f"{bar + 1}:{beat + 1}.{tick}"

    def parse_label(self, text: str) -> int | None:
        """``"3:2.1"`` -> absolute tick.  ``None`` when it is not a position.

        Inverse of :meth:`label`, phase included — so a label this grid produced reads
        back to the tick it came from, and a label a model invented is interpreted in
        the same coordinate system it was shown.
        """
        parsed = _parse_position(text)
        return None if parsed is None else self.to_tick(*parsed)


# --------------------------------------------------------------------------- #
# input side
# --------------------------------------------------------------------------- #
@dataclass
class Voice:
    """One source line: a name, where it came from, and its events on the grid.

    ``extra`` carries the additional pitches struck at the same instant.  It matters for
    accompaniment: one pitch per onset throws away the voicing — whether the piano plays
    a triad, a seventh, or a spread arpeggio — which is exactly the information a texture
    has to be read from.
    """

    name: str
    source: str
    ticks: list[int] = field(default_factory=list)
    pitches: list[int] = field(default_factory=list)
    extra: dict[int, list[int]] = field(default_factory=dict)

    @property
    def events(self) -> list[tuple[int, int]]:
        return list(zip(self.ticks, self.pitches))

    def voicing(self, tick: int) -> list[int]:
        """Every pitch sounding at ``tick``: the primary one plus its companions."""
        primary = None
        for index, value in enumerate(self.ticks):
            if value == tick:
                primary = self.pitches[index]
                break
        if primary is None:
            return []
        return sorted({primary, *self.extra.get(tick, [])})

    @property
    def pitch_range(self) -> tuple[int, int] | None:
        all_pitches = list(self.pitches) + [p for values in self.extra.values() for p in values]
        return (min(all_pitches), max(all_pitches)) if all_pitches else None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "source": self.source,
            "events": [[tick, pitch] for tick, pitch in self.events],
            "notes": len(self.pitches),
            "range": list(self.pitch_range) if self.pitch_range else None,
        }


@dataclass
class ArrangementInput:
    """Everything the model is shown, plus the grid it is shown on."""

    track: str
    grid: BarGrid
    tempo_bpm: float = 0.0
    key: str | None = None
    bars: int = 0
    voices: list[Voice] = field(default_factory=list)
    #: ``(start_seconds, label)`` chord segments SheetSage2 reported, if the run has them.
    chords: list[tuple[float, str]] = field(default_factory=list)
    #: SheetSage2's chord *voicings* as a voice (``CHORDS``), one stack per chord change.
    #: A label says what the chord is; the voicing says how it is spread, which is the part
    #: an arrangement actually needs and which no label can carry.
    chord_voicings: Voice | None = None
    #: Key changes as ``(tick, key)``, the key taking effect at that tick.  Empty means one
    #: key for the whole piece.  A song that modulates and is engraved under a single
    #: signature writes an accidental on every note that the *right* signature would have
    #: carried — on the delivered tracks that is 18-20% of all notes, against 1.6% on the
    #: tracks that do not modulate.
    key_changes: list[tuple[int, str]] = field(default_factory=list)
    #: Which file the melody line came from, so a reader can tell whether the opening bars
    #: will have a tune.  Set by the loader.
    melody_source: str = ""
    warnings: list[str] = field(default_factory=list)

    def voice(self, name: str) -> Voice | None:
        for item in self.voices:
            if item.name == name:
                return item
        return None

    @property
    def texture_voices(self) -> list[Voice]:
        """The per-instrument accompaniment material, in a stable order."""
        return [v for v in self.voices if v.name.startswith("TEXTURE:")]

    @property
    def harmony(self) -> Voice:
        """All accompaniment merged into one line, for readers that want a single column.

        The rule-based arranger uses this; the model is shown the separate instruments,
        because telling a piano texture from a string pad is the whole point.
        """
        merged: dict[int, list[int]] = {}
        for voice in self.texture_voices:
            for tick in voice.ticks:
                merged.setdefault(tick, []).extend(voice.voicing(tick))
        ticks = sorted(merged)
        return Voice(
            name="HARMONY",
            source="merged accompaniment",
            ticks=ticks,
            pitches=[max(merged[t]) for t in ticks],
            extra={t: sorted(set(merged[t]))[:-1] for t in ticks if len(set(merged[t])) > 1},
        )

    def chord_at(self, seconds: float) -> str | None:
        """The chord label in force at ``seconds``, or ``None``."""
        found = None
        for start, label in self.chords:
            if start <= seconds + 1e-6:
                found = label
            else:
                break
        return found

    def key_at(self, tick: int) -> str | None:
        """The key in force at ``tick``, falling back to the piece's single key."""
        found = None
        for start, key in self.key_changes:
            if start <= tick:
                found = key
            else:
                break
        return found or self.key

    @property
    def flats(self) -> bool:
        return _uses_flats(self.key)


def load_arrangement_input(
    directory: str | Path,
    *,
    melody_stem: str = "vocals",
    bass_stem: str = "bass",
    comparator: str = "basic-pitch",
) -> ArrangementInput:
    """Read one delivered run into the material for a piano arrangement.

    Sources, and why each is there:

    * ``MELODY`` — SheetSage2's line from the vocal stem.  This is the backbone: it is
      the one line measured to be continuous, so it decides the tune.
    * ``HARMONY`` — the accompaniment register of the basic-pitch rendering (every
      pitched stem except the melody and bass, keeping only notes *below* the melody
      so it cannot compete with it).  Messier, but it is the only material that
      reports the actual inner voicing.
    * ``BASS`` — the basic-pitch bass line, which anchors the left hand.
    """
    import pretty_midi

    path = Path(directory)
    run = json.loads((path / "run.json").read_text(encoding="utf-8"))
    beats = (run.get("beats") or {}).get("beat_times") or []
    meter = run.get("meter") or {}
    best = (meter.get("hypotheses") or [{}])[0]
    grid = BarGrid(
        beat_times=[float(t) for t in beats],
        beats_per_bar=int(best.get("numerator") or 4),
        phase=int(meter.get("phase") or 0),
    )
    arrival = ArrangementInput(
        track=path.name,
        grid=grid,
        tempo_bpm=float((run.get("beats") or {}).get("tempo_bpm") or 0.0),
        key=run.get("key"),
        bars=int(run.get("measures") or 0),
    )
    result = arrival
    # The chord progression SheetSage2 heard, as a harmonic reference for whoever
    # arranges.  It is advisory and labelled as such wherever it appears.
    for chord in run.get("chords") or []:
        label = chord.get("label")
        if not label or label in ("N.C.", "N"):
            continue
        result.chords.append((float(chord.get("start") or 0.0), label))
    if not beats:
        result.warnings.append(
            "run.json carries no beat grid: positions cannot be expressed in bars"
        )
        return result

    def read(directory_name: str, stem: str) -> list[tuple[float, int]]:
        target = path / directory_name / f"{stem}.mid"
        if not target.is_file():
            return []
        midi = pretty_midi.PrettyMIDI(str(target))
        return sorted(
            (float(note.start), int(note.pitch))
            for instrument in midi.instruments
            for note in instrument.notes
        )

    def to_voice(name: str, source: str, raw: list[tuple[float, int]]) -> Voice:
        return _voice_from_events(name, source, raw, grid)

    melody = to_voice("MELODY", f"sheetsage2:{melody_stem}", read("midi", melody_stem))
    result.voices.append(melody)
    result.voices.append(to_voice("BASS", f"{comparator}:{bass_stem}", read(f"alt/{comparator}", bass_stem)))

    # One voice per accompaniment instrument, **unfiltered**.  Clipping the accompaniment
    # to below the melody looked tidy and was the reason the first arrangements had a
    # wrong accompaniment: measured on the delivered tracks it discards 13-61% of the
    # material, and what it discards is the piano's chord voicings and the string layer —
    # precisely the texture the arrangement is supposed to reproduce.  The model is shown
    # the voicing and told to keep whatever it writes under the tune; the *writing* is
    # constrained, the reading is not.
    for stem in ("piano", "guitar", "other", "strings"):
        if stem == melody_stem or stem == bass_stem:
            continue
        raw = read(f"alt/{comparator}", stem)
        if not raw:
            continue
        result.voices.append(to_voice(f"TEXTURE:{stem}", f"{comparator}:{stem}", raw))

    # The same instruments as SheetSage2 read them.  Not a replacement: basic-pitch keeps
    # more detail (including the timbre's own artefacts) while SheetSage2 has already
    # reduced each stem towards what a musician would write.  Both are shown, so the
    # arranger can tell music from timbre instead of having that decided for it.
    for stem in ("piano", "guitar", "other", "strings"):
        if stem == melody_stem or stem == bass_stem:
            continue
        raw = read("midi", stem)
        if not raw:
            continue
        result.voices.append(to_voice(f"TEXTURE:{stem}:SS2", f"sheetsage2:{stem}", raw))

    for voice in result.voices:
        if not voice.pitches:
            result.warnings.append(
                f"{voice.name}: no notes found ({voice.source}); the arrangement will be "
                "missing this material"
            )

    # Prefer SheetSage2's own melody and chord voicings, which live outside the run
    # directory.  Optional: the loader degrades to the run's material and says so.
    _load_sheetsage2_sources(result, path.name)
    return result


def _load_sheetsage2_sources(result: ArrangementInput, track: str) -> None:
    """Prefer SheetSage2's own outputs for the melody and the chord voicings.

    Why these two come from ``sheetsage2_out/`` rather than from the run directory:

    * **The melody.**  The run transcribes the vocal *stem*, so the tune cannot start
      before the voice does.  SheetSage2's ``melody_merged`` is its vocal and
      instrumental lines combined, and on three of the four tracks the instrumental line
      carries 46-64 notes *before* the voice enters — which is exactly the material a piano
      arrangement needs at the opening.  Measured against each run's own beat grid, every
      note of the merged line lands on the same tick lattice, so this is a change of
      source and not a change of timing.
    * **The chord voicings.**  ``chords.mid`` is not a label list: each chord change is a
      4-5 note stack with a real spread (``Db2+Db3+F3+Ab3+C4``), which is the thing an
      arrangement actually needs.  The labels in ``chord.lab`` accompany it.

    Both are optional.  When ``sheetsage2_out/`` is absent the run's own transcription is
    used, and the packet says so, rather than silently degrading.
    """
    import pretty_midi

    ss2 = Path("sheetsage2_out") / track
    if not ss2.is_dir():
        result.warnings.append(
            "sheetsage2_out/ has no directory for this track: the melody comes from the "
            "run's per-stem transcription, which cannot start before the voice does, and "
            "no chord voicings are available"
        )
        return

    def read_midi(target: Path) -> list[tuple[float, int]]:
        if not target.is_file():
            return []
        midi = pretty_midi.PrettyMIDI(str(target))
        return sorted(
            (float(note.start), int(note.pitch))
            for instrument in midi.instruments
            for note in instrument.notes
        )

    merged = read_midi(ss2 / "melody_merged.mid")
    if merged:
        result.melody_source = "sheetsage2:melody_merged"
        # Replace the per-stem melody, keeping its position in the voice list.
        for index, voice in enumerate(result.voices):
            if voice.name == "MELODY":
                result.voices[index] = _voice_from_events(
                    "MELODY", "sheetsage2:melody_merged", merged, result.grid
                )
                break
    else:
        result.warnings.append(
            "sheetsage2_out/ has no melody_merged.mid: the melody stays on the per-stem "
            "reading, so it cannot begin before the voice does"
        )

    voicings = read_midi(ss2 / "chords.mid")
    if voicings:
        result.chord_voicings = _voice_from_events(
            "CHORDS", "sheetsage2:chords.mid", voicings, result.grid
        )

    labels = ss2 / "chord.lab"
    if labels.is_file():
        parsed: list[tuple[float, str]] = []
        for line in labels.read_text(encoding="utf-8").splitlines():
            columns = line.split("\t")
            if len(columns) < 2:
                continue
            try:
                start = float(columns[0])
            except ValueError:
                continue
            label = columns[-1].strip()
            if label and label not in ("N", "N.C."):
                parsed.append((start, label))
        if parsed:
            # SheetSage2's own labels supersede the run's chroma estimate: the arrangement
            # is told to trust the chords, so it should be given the ones the chord model
            # emitted rather than a template-matching guess at the same thing.
            result.chords = parsed


def _voice_from_events(
    name: str, source: str, raw: list[tuple[float, int]], grid: BarGrid
) -> Voice:
    """Grid-position -> Voice, keeping the whole voicing of anything that collides.

    Collisions are real: a quantised grid puts a whole chord on one tick.  For a line that
    is fine — the top note is the line — but for accompaniment the *chord* is the
    information, so the other pitches are kept alongside rather than discarded.
    """
    grouped: dict[int, list[int]] = {}
    for start, pitch in raw:
        tick = grid.to_tick(*grid.position(start))
        grouped.setdefault(tick, []).append(pitch)
    ticks = sorted(grouped)
    primary: list[int] = []
    extra: dict[int, list[int]] = {}
    for tick in ticks:
        ordered = sorted(set(grouped[tick]))
        primary.append(ordered[-1])
        if len(ordered) > 1:
            extra[tick] = ordered[:-1]
    return Voice(name=name, source=source, ticks=ticks, pitches=primary, extra=extra)


# --------------------------------------------------------------------------- #
# rendering the model's view
# --------------------------------------------------------------------------- #


def render_input_text(
    material: ArrangementInput,
    *,
    max_events: int = 6000,
) -> str:
    """The packet: header, then one line per event, grouped by bar.

    Positions are bar-relative on purpose.  The arranger's job is to compare one note
    against another — does the tune sit above the accompaniment, does the left hand reach
    — and both need positions that stay comparable down a column without knowing the
    tempo.

    Two things are shown that an earlier version did not, and they are the difference
    between reading a texture and inventing one:

    * **one column per accompaniment instrument**, unfiltered, and **twice**: as
      basic-pitch read it (more detail, including the timbre's own artefacts) and as
      SheetSage2 read it (already reduced towards what a musician would write).  Which of
      the two is music and which is timbre is a judgement, and it is handed over rather
      than made here;
    * **the chords SheetSage2 reported**, both as a label per bar and as its actual
      voicing — ``Db2+Db3+F3+Ab3+C4`` is a spread the label ``Db:maj7`` cannot carry.
    """
    grid = material.grid
    flats = material.flats
    # Melody first, then accompaniment, then the bass: reading order matches the roles.
    texture_voices = material.texture_voices
    ordered = (
        [v for v in material.voices if v.name == "MELODY"]
        + texture_voices
        + [v for v in material.voices if v.name == "BASS"]
    )
    lines = [
        f"# {material.track} | {grid.beats_per_bar}/4 | {material.tempo_bpm:.1f} BPM | "
        f"{material.key or 'unknown key'} | {material.bars} bars",
        f"# grid: position is bar:beat.tick, 1-based bar and beat, "
        f"{grid.ticks_per_beat} ticks per beat",
    ]
    for voice in ordered:
        span = voice.pitch_range
        lines.append(
            f"# {voice.name}: {len(voice.pitches)} notes from {voice.source}"
            + (
                f", range {pitch_name(span[0], flats=flats)}..{pitch_name(span[1], flats=flats)}"
                if span
                else ""
            )
        )
    if material.chord_voicings is not None:
        lines.append(
            f"# CHORDS: {len(material.chord_voicings.pitches)} chord changes from "
            "sheetsage2_out/<track>/chords.mid - the authoritative harmony, given both as "
            "a label per bar and as its voicing on the CHORDS column"
        )
    columns = [voice.name for voice in ordered] + (
        ["CHORDS"] if material.chord_voicings is not None else []
    )
    lines.append(
        "# columns: position  " + "  ".join(columns)
        + "   (a '.' means nothing starts there; pitches joined by + sound together)"
    )
    lines.append("")

    lookup: dict[int, dict[str, list[int]]] = {}
    for voice in ordered:
        for tick in voice.ticks:
            lookup.setdefault(tick, {})[voice.name] = voice.voicing(tick)
    if material.chord_voicings is not None:
        for tick in material.chord_voicings.ticks:
            lookup.setdefault(tick, {})["CHORDS"] = material.chord_voicings.voicing(tick)

    def cell(name: str, tick: int) -> str:
        pitches = lookup.get(tick, {}).get(name)
        if not pitches:
            return "."
        return "+".join(pitch_name(p, flats=flats) for p in pitches)

    def chord_for(tick: int) -> str:
        bar, beat, sub = grid.from_tick(tick)
        if sub != 0:
            return ""
        seconds = _tick_seconds(grid, tick)
        return material.chord_at(seconds) or ""

    ticks = sorted(lookup)
    if len(ticks) > max_events:
        material.warnings.append(
            f"only the first {max_events} of {len(ticks)} event positions are rendered"
        )
        ticks = ticks[:max_events]

    current_bar = -1
    for tick in ticks:
        bar, beat, sub = grid.from_tick(tick)
        if bar != current_bar:
            bar_start = grid.to_tick(bar, 0, 0)
            label = material.chord_at(_tick_seconds(grid, bar_start)) or "-"
            lines.append(f"=== bar {bar + 1}  chords: {label} ===")
            current_bar = bar
        cells = [cell(name, tick) for name in columns]
        # A chord label is carried on the beat it changes on, so the row reads as a
        # progression rather than a label repeated on every beat.
        chord = chord_for(tick)
        tail = f"   [{chord}]" if chord else ""
        lines.append(
            f"{grid.label(bar, beat, sub):<9} " + "  ".join(c.ljust(10) for c in cells) + tail
        )
    return "\n".join(lines) + "\n"


def _tick_seconds(grid: BarGrid, tick: int) -> float:
    """Seconds for a grid tick, so chord segments (which are in seconds) can be matched."""
    beats = grid.beat_times
    if not beats:
        return 0.0
    index, fraction = divmod(tick, grid.ticks_per_beat)
    if index < 0:
        return beats[0] + (tick / grid.ticks_per_beat) * (beats[1] - beats[0] if len(beats) > 1 else 0.5)
    if index >= len(beats) - 1:
        step = beats[-1] - beats[-2] if len(beats) > 1 else 0.5
        return beats[-1] + (index - (len(beats) - 1) + fraction / grid.ticks_per_beat) * step
    span = beats[index + 1] - beats[index]
    return beats[index] + (fraction / grid.ticks_per_beat) * span


# --------------------------------------------------------------------------- #
# output side
# --------------------------------------------------------------------------- #
@dataclass
class Arrangement:
    """What the model returned: two hands on the input's grid.

    Each hand is a mapping from grid tick to the pitches struck there, so a hand may
    carry a chord.  Durations are deliberately absent — see
    :func:`sounding_lengths`.
    """

    track: str = ""
    right: dict[int, list[int]] = field(default_factory=dict)
    left: dict[int, list[int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def notes(self) -> int:
        return sum(len(v) for v in self.right.values()) + sum(len(v) for v in self.left.values())

    @property
    def onsets(self) -> int:
        return len(self.right) + len(self.left)

    def hand(self, name: str) -> dict[int, list[int]]:
        return self.right if name.upper().startswith("R") else self.left


def _parse_position(text: str) -> tuple[int, int, int] | None:
    """``"3:2.1"`` -> ``(bar, beat, tick)``, all 0-based."""
    try:
        bar_text, _, rest = text.partition(":")
        if not rest:
            return None
        beat_text, _, tick_text = rest.partition(".")
        return (int(bar_text) - 1, int(beat_text) - 1, int(tick_text or 0))
    except (ValueError, TypeError):
        return None


def parse_arrangement(
    text: str,
    grid: BarGrid,
    *,
    max_per_hand: int = 3,
    max_bars: int | None = None,
) -> Arrangement:
    """Read the model's reply.

    The contract is deliberately narrow — ``RH <position> <pitch>...``, one event per
    line — because a narrow contract is one that can be **checked**.  Everything that
    is not a line of that shape is reported rather than quietly skipped: a line the
    parser drops in silence is how content disappears without anyone noticing, which
    is the mistake this project already made once with its LLM rendering.

    ``max_per_hand`` caps chord width per hand.  More than three notes struck in one
    hand at one instant is not a transcription error but an unplayable one, so the
    excess is dropped and counted rather than engraved.

    ``max_bars`` is the song's length.  An event past the last bar is dropped and
    counted, **not** folded into the final measure: clamping it would pile extra notes
    onto a bar that already has its own rhythm, which is worse than losing a note the
    model placed outside the piece.
    """
    right: dict[int, list[int]] = {}
    left: dict[int, list[int]] = {}
    result = Arrangement(track="")
    pitched_missing = 0
    unparsed: list[str] = []
    too_wide = 0
    before_start = 0
    past_end = 0
    per_bar = max(1, grid.beats_per_bar) * grid.ticks_per_beat

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("==="):
            continue
        columns = line.replace(",", " ").split()
        if not columns:
            continue

        hand = "RH"
        if columns[0].upper() in ("RH", "LH", "R", "L"):
            hand = "RH" if columns[0].upper().startswith("R") else "LH"
            columns = columns[1:]
        if not columns:
            pitched_missing += 1
            continue

        # The position is parsed with the *grid's* own arithmetic rather than by
        # rebuilding a label: `BarGrid.to_tick` subtracts the phase, and feeding it a
        # label built from an unphased bar/beat pair double-counts the pickup.
        parsed = _parse_position(columns[0])
        if parsed is None:
            unparsed.append(line[:60])
            continue
        bar, _beat, _tick = parsed
        tick = grid.to_tick(*parsed)
        if tick < -grid.pickup_ticks:
            before_start += 1
            continue
        if max_bars is not None and bar >= max_bars:
            past_end += 1
            continue

        pitches = [p for p in (_pitch_from_token(token) for token in columns[1:]) if p is not None]
        if not pitches:
            pitched_missing += 1
            continue

        pitches = sorted(set(pitches))
        if len(pitches) > max_per_hand:
            too_wide += 1
            pitches = pitches[-max_per_hand:] if hand == "RH" else pitches[:max_per_hand]

        target = right if hand == "RH" else left
        if tick in target:
            # Two lines for the same instant: union rather than overwrite, so a reply
            # written in two passes does not lose its first half.  Sorted, because a
            # hand's pitches are read back as a chord and the lowest note is the one
            # that decides the voicing.
            target[tick] = sorted(set(target[tick]) | set(pitches))
        else:
            target[tick] = pitches

    result.right = dict(sorted(right.items()))
    result.left = dict(sorted(left.items()))
    if pitched_missing:
        result.warnings.append(f"{pitched_missing} line(s) carried a position but no readable pitch")
    if unparsed:
        preview = "; ".join(unparsed[:3])
        result.warnings.append(
            f"{len(unparsed)} line(s) were not in the expected `RH 3:2.0 C4` form and were "
            f"skipped: {preview}"
        )
    if before_start:
        result.warnings.append(
            f"{before_start} line(s) sat before the start of the music: dropped, because "
            "only the pickup region ahead of bar 1 can be notated"
        )
    if past_end:
        result.warnings.append(
            f"{past_end} line(s) sat past bar {max_bars}: dropped rather than folded into "
            "the last measure, which already has its own rhythm"
        )
    if too_wide:
        result.warnings.append(
            f"{too_wide} onset(s) had more than {max_per_hand} notes in one hand: "
            "reduced, because that many struck together is unplayable rather than merely dense"
        )
    return result


def sounding_lengths(
    events: dict[int, list[int]],
    *,
    grid: BarGrid | None = None,
    ticks_per_beat: int = TICKS_PER_BEAT,
    max_beats: float = 2.0,
) -> dict[int, float]:
    """Derive each event's sounding length in beats from the next one.

    The model is never asked for durations, so they are computed here: an event
    sounds until the next event in the same hand, capped at ``max_beats`` so a long
    gap does not produce a note nobody could hold.  This is the one convention that
    makes a monophonic-with-chords hand notate as legato without any of the rhythm
    being invented by the model.
    """
    ticks = sorted(events)
    lengths: dict[int, float] = {}
    for index, tick in enumerate(ticks):
        if index + 1 < len(ticks):
            span = (ticks[index + 1] - tick) / ticks_per_beat
        else:
            span = 1.0
        lengths[tick] = max(0.125, min(float(span), max_beats))
    return lengths


def _pitch_from_token(token: str) -> int | None:
    """``C4``/``Eb4``/``F#5``/(``60``) -> MIDI number."""
    token = token.strip().strip("()")
    if not token:
        return None
    if token.lstrip("-").isdigit():
        value = int(token)
        return value if 0 <= value <= 127 else None

    letter = token[0].upper()
    if letter not in "ABCDEFG":
        return None
    index = 1
    accidental = 0
    while index < len(token) and token[index] in "#b♯♭-":
        char = token[index]
        accidental += 1 if char in "#♯" else (-1 if char in "b♭-" else 0)
        index += 1
    digits = token[index:]
    if not digits.lstrip("-").isdigit():
        return None
    octave = int(digits)
    natural = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[letter]
    pitch = (octave + 1) * 12 + natural + accidental
    return pitch if 0 <= pitch <= 127 else None


# --------------------------------------------------------------------------- #
# rendering the arrangement for a human / for a reader
# --------------------------------------------------------------------------- #
def render_arrangement_abc(
    arrangement: Arrangement,
    grid: BarGrid,
    *,
    key: str | None = None,
    tempo_bpm: float = 0.0,
    title: str = "",
    flats: bool | None = None,
) -> str:
    """Two-voice ABC, one voice per hand, with real barlines.

    ABC is the readable form: bar lines and accidentals make a wrong octave or a
    misplaced bar obvious at a glance, which is what a reviewer needs.  Each hand is
    a monophonic line, so durations are written as ``z``/note runs and the bar is
    filled by the positions themselves — the notation stage remains authoritative for
    the engraved score.
    """
    if flats is None:
        flats = _uses_flats(key)
    names = NOTE_NAMES_FLAT if flats else NOTE_NAMES_SHARP
    per_bar = max(1, grid.beats_per_bar) * grid.ticks_per_beat
    total = max(
        max(arrangement.right, default=0),
        max(arrangement.left, default=0),
    )
    bars = total // per_bar + 1

    def token(pitch: int) -> str:
        name = names[pitch % 12]
        octave = pitch // 12 - 1
        letter = name[0]
        accidental = {"#": "^", "b": "_"}.get(name[1:2], "")
        if octave >= 5:
            body = letter.lower() + "'" * (octave - 5)
        else:
            body = letter.upper() + "," * (4 - octave)
        return accidental + body

    tonic = (key or "C").replace(":", " ").split()[0].replace("-", "b")
    abc_key = tonic.replace("b", "_").replace("#", "^")
    mode = "m" if key and "minor" in key.lower() else ""
    header = [
        "X:1",
        f"T:{title or arrangement.track}",
        f"M:{grid.beats_per_bar}/4",
        f"L:1/{grid.ticks_per_beat * 4}",
        f"Q:1/4={tempo_bpm:.0f}" if tempo_bpm else "Q:1/4=120",
        f"K:{abc_key}{mode}",
        'V:RH name="Right hand" clef=treble',
        'V:LH name="Left hand" clef=bass',
    ]

    lines = list(header)
    for name, events in (("RH", arrangement.right), ("LH", arrangement.left)):
        # One slot per *beat* here: this rendering is for reading, and the engraved
        # score (which carries the real rhythm) comes from the notation stage.
        lookup = {tick // grid.ticks_per_beat: sorted(pitches) for tick, pitches in events.items()}
        lines.append(f"V:{name}")
        row: list[str] = []
        for bar in range(bars):
            tokens = ["z"] * grid.beats_per_bar
            for beat in range(grid.beats_per_bar):
                pitches = lookup.get(bar * grid.beats_per_bar + beat)
                if not pitches:
                    continue
                if len(pitches) == 1:
                    tokens[beat] = token(pitches[0])
                else:
                    tokens[beat] = "[" + "".join(token(p) for p in pitches) + "]"
            row.append(" ".join(tokens) + " |")
        lines.append("\n".join(row))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# is it playable, and did it keep the music?
# --------------------------------------------------------------------------- #
# A hand at one instant can reach this far.  An octave plus a fifth is a comfortable
# span for most players; beyond it a chord is a rolled chord at best, and at worst a
# mistake the model made while thinking of the notes and not of the hands.
COMFORTABLE_SPAN = 19
# Above this many sounding notes at once the texture stops being playable piano and
# becomes a reduction that has to be thinned by hand.
MAX_SIMULTANEOUS = 5


@dataclass
class PlayabilityReport:
    """Measurable statements about a rendered arrangement."""

    notes: int = 0
    onsets: int = 0
    bars: int = 0
    right_notes: int = 0
    left_notes: int = 0
    written_elements: int = 0
    max_span: int = 0
    wide_chords: int = 0
    max_simultaneous: int = 0
    average_simultaneous: float = 0.0
    crowded_onsets: int = 0
    hand_crossings: int = 0
    left_above_middle_c: int = 0
    notes_per_second: float = 0.0
    melody_kept: float = 0.0
    melody_missing: int = 0
    melody_octave_shifted: int = 0
    invented: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in self.__dict__.items()
        }

    def summary(self) -> str:
        return (
            f"{self.notes} notes in {self.bars} bars ({self.right_notes} RH / "
            f"{self.left_notes} LH), span<={self.max_span}, "
            f"polyphony peak {self.max_simultaneous} (mean {self.average_simultaneous:.2f}), "
            f"melody kept {self.melody_kept:.0%}"
        )


def _sounding_load(
    arrangement: Arrangement, grid: BarGrid
) -> tuple[int, float]:
    """``(peak, time-weighted average)`` notes sounding at once, both hands together."""
    edges: list[tuple[float, int]] = []
    for hand in (arrangement.right, arrangement.left):
        lengths = sounding_lengths(hand, ticks_per_beat=grid.ticks_per_beat)
        for tick, pitches in hand.items():
            start = float(tick)
            end = start + lengths[tick] * grid.ticks_per_beat
            edges.append((start, len(pitches)))
            edges.append((end, -len(pitches)))
    if not edges:
        return 0, 0.0
    # Ends before starts at the same instant: a note that stops exactly where the next
    # begins is one note's worth of sound, not two.
    edges.sort(key=lambda item: (item[0], item[1]))

    current = 0
    peak = 0
    weighted = 0.0
    previous: float | None = None
    for position, delta in edges:
        if previous is not None and current:
            weighted += current * (position - previous)
        current += delta
        peak = max(peak, current)
        previous = position
    span = edges[-1][0] - edges[0][0]
    return peak, (weighted / span if span > 0 else 0.0)


def _crowded_onsets(arrangement: Arrangement, grid: BarGrid) -> int:
    """How many onsets begin while more than :data:`MAX_SIMULTANEOUS` notes sound."""
    edges: list[tuple[float, int]] = []
    for hand in (arrangement.right, arrangement.left):
        lengths = sounding_lengths(hand, ticks_per_beat=grid.ticks_per_beat)
        for tick, pitches in hand.items():
            start = float(tick)
            edges.append((start, len(pitches)))
            edges.append((start + lengths[tick] * grid.ticks_per_beat, -len(pitches)))
    if not edges:
        return 0
    edges.sort(key=lambda item: (item[0], item[1]))

    onsets = sorted(set(arrangement.right) | set(arrangement.left))
    crowded = 0
    current = 0
    index = 0
    for onset in onsets:
        while index < len(edges) and edges[index][0] <= onset:
            current += edges[index][1]
            index += 1
        if current > MAX_SIMULTANEOUS:
            crowded += 1
    return crowded


def evaluate_arrangement(
    arrangement: Arrangement,
    material: ArrangementInput,
    *,
    seconds: float = 0.0,
) -> PlayabilityReport:
    """Score the arrangement for playability and for what it kept.

    Two different questions, and both matter:

    * **Playability** is physical — span, how many notes sound at once, whether the
      hands cross — and is the whole point of the exercise.
    * **Fidelity** is whether the tune survived.  A report that looks clean because
      the melody was dropped has failed the more important half, which is the mistake
      ``score_llm_output.py`` exists to catch.
    """
    report = PlayabilityReport()
    grid = material.grid
    per_bar = max(1, grid.beats_per_bar) * grid.ticks_per_beat

    report.right_notes = sum(len(v) for v in arrangement.right.values())
    report.left_notes = sum(len(v) for v in arrangement.left.values())
    report.notes = report.right_notes + report.left_notes
    report.onsets = arrangement.onsets
    report.bars = material.bars or (
        max(max(arrangement.right, default=0), max(arrangement.left, default=0)) // per_bar + 1
    )

    for hand in (arrangement.right, arrangement.left):
        for pitches in hand.values():
            if not pitches:
                continue
            span = max(pitches) - min(pitches)
            report.max_span = max(report.max_span, span)
            if span > COMFORTABLE_SPAN:
                report.wide_chords += 1

    # Sounding load, measured with a sweep line: with the derived lengths, a note
    # sounds from its onset to the next onset in its hand, so the number of notes
    # sounding at any instant changes only at those edges.  Walking the edges is exact
    # and linear, where sampling at onsets would both miss the interior of a long note
    # and cost a comparison per event.
    report.max_simultaneous, report.average_simultaneous = _sounding_load(
        arrangement, grid
    )
    report.crowded_onsets = _crowded_onsets(arrangement, grid)

    # Hands should stay on their own side of middle C, near enough.
    for tick, pitches in arrangement.right.items():
        if pitches and min(pitches) < 60:
            report.hand_crossings += 1
    for tick, pitches in arrangement.left.items():
        if pitches and max(pitches) > 60:
            report.left_above_middle_c += 1

    if seconds > 0:
        report.notes_per_second = report.notes / seconds

    # Fidelity: the melody is the reason this arrangement exists, so its survival is
    # reported separately from everything else.
    melody = material.voice("MELODY")
    if melody is not None and melody.pitches:
        placed = set(arrangement.right) | set(arrangement.left)
        played = {
            pitch
            for hand in (arrangement.right, arrangement.left)
            for pitches in hand.values()
            for pitch in pitches
        }
        kept = 0
        shifted = 0
        missing = 0
        for tick, pitch in melody.events:
            window = [t for t in placed if abs(t - tick) <= grid.ticks_per_beat // 2]
            if not window:
                missing += 1
                continue
            candidates = {p for t in window for p in (arrangement.right.get(t) or []) + (arrangement.left.get(t) or [])}
            if pitch in candidates:
                kept += 1
            elif any(abs(pitch - other) % 12 == 0 for other in candidates):
                shifted += 1
            else:
                missing += 1
        report.melody_kept = kept / len(melody.pitches)
        report.melody_missing = missing
        report.melody_octave_shifted = shifted
        report.invented = max(0, len(played) - len({p for _t, p in melody.events}))

    if report.wide_chords:
        report.warnings.append(
            f"{report.wide_chords} onset(s) span more than {COMFORTABLE_SPAN} semitones in "
            "one hand: those are rolled chords at best"
        )
    if report.crowded_onsets:
        report.warnings.append(
            f"{report.crowded_onsets} onset(s) have more than {MAX_SIMULTANEOUS} notes "
            "sounding at once"
        )
    if report.hand_crossings:
        report.warnings.append(
            f"{report.hand_crossings} right-hand onset(s) reach below middle C"
        )
    if report.left_above_middle_c:
        report.warnings.append(
            f"{report.left_above_middle_c} left-hand onset(s) reach above middle C"
        )
    if report.melody_missing and report.melody_kept < 0.5:
        # Note the `melody_missing` guard rather than a truthiness test on the ratio:
        # `if report.melody_kept` is False for exactly 0.0, so the *worst* outcome —
        # every melody note gone — was the one case that raised no warning at all.
        report.warnings.append(
            f"only {report.melody_kept:.0%} of the melody's notes are present: the "
            "arrangement may have become tidy by dropping the tune"
        )
    return report
