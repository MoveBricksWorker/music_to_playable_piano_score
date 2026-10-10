"""Turn a transcription into text a language model can read and edit.

Why this exists
---------------
The weakest parts of this pipeline are all *symbolic* judgements, not perceptual
ones: an isolated octave error, an enharmonic spelling that contradicts the key, a
part that is really two hands, a second part that is really the first one copied.
Those are exactly the judgements a language model is good at — given a
representation it can actually read.

What not to give it
-------------------
Not a ``.mid`` file: the bytes carry no readable pitch or time, and a base64 blob
helps nobody.  Not a raw MIDI event stream either: delta times in ticks are awkward
to reason about and verbose.  The representation here is deliberately
**bar-relative and comma-separated**, because that is what the model needs to
compare one note against another:

    # Deliverable 4stem | 4/4 | 90.9 BPM | E- major | 2 parts
    [vocals] 491 notes, bars 1-73
    1:1.0   G4   1.00  v72
    1:2.0   A4   0.50  v80   B4 0.50 v74
    1:3.0   rest 1.00

Time is ``bar:beat.subdivision`` in *beats*, so ``3:2.5`` is unambiguous without
knowing the tempo, and a duration in beats is comparable down a column.  Pitches are
spelled against the piece's key, so an E♭ major score says ``Eb4`` and never ``D#4``
— which is one of the defects the model is being asked to find.

Nothing here is lossy: every note in the MIDI appears exactly once, with its exact
start and duration, so a corrected version can be mapped back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "ScoreText",
    "note_name",
    "load_run",
    "render_score_text",
    "render_abc",
    "write_all",
]

# Flats this many steps round the circle of fifths; below it the key is written with
# sharps.  C is the pivot, F is the first flat key.
_FLAT_KEYS = {"F", "Bb", "Eb", "Ab", "Db", "Gb", "Cb", "F-", "Bb-", "Eb-"}

_SHARP_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_FLAT_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")

# music21 spells keys with a trailing dash for a flat ("E- major"); that is also the
# spelling run.json carries.
_KEY_ALIASES = {
    "E-": "Eb", "B-": "Bb", "A-": "Ab", "D-": "Db", "G-": "Gb", "C-": "Cb",
    "F-": "Fb", "C#": "C#", "F#": "F#",
}


@dataclass
class Part:
    """One transcribed part, in the form the renderers need."""

    stem: str
    program: int
    is_drum: bool
    notes: list[tuple[float, float, int, int]] = field(default_factory=list)
    # (start_seconds, end_seconds, pitch, velocity)


@dataclass
class ScoreText:
    """A whole transcription, ready to render."""

    track: str
    beats_per_bar: int
    tempo_bpm: float
    key: str | None
    parts: list[Part]
    beat_times: list[float]
    phase: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.beat_times[-1] if self.beat_times else 0.0


# --------------------------------------------------------------------------- #
# pitch spelling
# --------------------------------------------------------------------------- #
def uses_flats(key: str | None) -> bool:
    if not key:
        return False
    tonic = key.split()[0] if " " in key else key.split(":")[0]
    return tonic in _FLAT_KEYS or tonic in {"E-", "B-", "A-", "D-", "G-", "C-"}


def note_name(pitch: int, *, flats: bool = False) -> str:
    """MIDI number -> scientific pitch name, spelled for the key."""
    names = _FLAT_NAMES if flats else _SHARP_NAMES
    return f"{names[int(pitch) % 12]}{int(pitch) // 12 - 1}"


def key_for_display(key: str | None) -> str:
    """``"E- major"`` -> ``"Eb major"``; ``"Eb:major"`` -> ``"Eb major"``."""
    if not key:
        return "unknown"
    text = key.replace(":", " ")
    parts = text.split()
    if not parts:
        return "unknown"
    tonic = _KEY_ALIASES.get(parts[0], parts[0])
    return " ".join([tonic, *parts[1:]])


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_run(directory: str | Path) -> ScoreText:
    """Read a delivered run directory: its MIDI parts and its run record."""
    import json

    import pretty_midi

    path = Path(directory)
    run_path = path / "run.json"
    if not run_path.is_file():
        raise FileNotFoundError(f"no run.json in {path}")
    run = json.loads(run_path.read_text(encoding="utf-8"))

    meter = run.get("meter") or {}
    best = (meter.get("hypotheses") or [{}])[0]
    beats = run.get("beats") or {}
    score = ScoreText(
        track=Path(run.get("input", path.name)).stem,
        beats_per_bar=int(best.get("numerator") or 4),
        tempo_bpm=float(beats.get("tempo_bpm") or 0.0),
        key=run.get("key"),
        parts=[],
        beat_times=[float(t) for t in beats.get("beat_times") or []],
        phase=int(meter.get("phase") or 0),
    )

    midi_dir = path / "midi"
    for midi_path in sorted(midi_dir.glob("*.mid")):
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        for instrument in pm.instruments:
            notes = [
                (float(n.start), float(n.end), int(n.pitch), int(n.velocity))
                for n in instrument.notes
            ]
            if notes:
                score.parts.append(
                    Part(
                        stem=midi_path.stem,
                        program=int(instrument.program),
                        is_drum=bool(instrument.is_drum),
                        notes=notes,
                    )
                )
    if not score.beat_times:
        score.warnings.append(
            "run.json has no beat grid: positions fall back to seconds, not bars"
        )
    return score


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #
def _position_map(beat_times: Sequence[float]) -> tuple[float, float]:
    """Return ``(t0, beat_seconds)`` for the simple linear case."""
    if len(beat_times) < 2:
        return (beat_times[0] if beat_times else 0.0), 0.5
    import statistics

    intervals = [b - a for a, b in zip(beat_times, beat_times[1:]) if b > a]
    step = statistics.median(intervals) if intervals else 0.5
    return beat_times[0], step


def seconds_to_position(seconds: float, beat_times: Sequence[float], grid: int) -> str:
    """Seconds -> ``bar:beat.subdivision`` (1-based bar, 1-based beat)."""
    from bisect import bisect_right

    if not beat_times:
        return f"{seconds:.3f}s"
    beats = beat_times
    index = bisect_right(beats, seconds) - 1
    if index < 0:
        beat_position = (seconds - beats[0]) / max(beats[1] - beats[0], 1e-9) if len(beats) > 1 else 0.0
    elif index >= len(beats) - 1:
        step = beats[-1] - beats[-2] if len(beats) > 1 else 0.5
        beat_position = (len(beats) - 1) + (seconds - beats[-1]) / max(step, 1e-9)
    else:
        span = beats[index + 1] - beats[index]
        beat_position = index + (seconds - beats[index]) / max(span, 1e-9)

    total_beats = beat_position
    bar = int(total_beats // 4) + 1  # replaced by the caller's meter below
    return f"{total_beats:.3f}b@{bar}"


def _quantise(value: float, grid: int) -> float:
    return round(value * grid) / grid


def _render_part(part: Part, score: ScoreText, grid: int, flats: bool) -> list[str]:
    """One part, as one line per onset: position, pitches, duration, velocity.

    Grouping is by **onset time**, not by a rounded grid slot.  Bucketing on a coarse
    grid silently destroys content: this piano part has 1438 distinct onset times
    across its 72 bars, and rounding them to 1/16-beat slots left only 244 slots, so
    notes that merely landed near each other were merged and then overwritten by the
    next one with the same key.  Measured on the real part: 1983 notes in, 719 out —
    64% of the music lost before a model ever saw it.  Grouping on the times
    themselves keeps every event, so the note count on the page equals the note count
    in the MIDI.  Simultaneous notes still share a line, which is the whole point.

    Duration is in sixteenth-note **units**, so every value is a small integer, and a
    rest is written as ``rest`` with its length.  Anything not on the grid is reported
    rather than silently moved: the position column is quantised, the durations are
    not re-derived from it.
    """
    from bisect import bisect_right

    beats = score.beat_times or [0.0]
    per_bar = max(1, score.beats_per_bar)

    def beat_position(seconds: float) -> float:
        index = bisect_right(beats, seconds) - 1
        if index < 0:
            step = (beats[1] - beats[0]) if len(beats) > 1 else 0.5
            return (seconds - beats[0]) / max(step, 1e-9)
        if index >= len(beats) - 1:
            step = (beats[-1] - beats[-2]) if len(beats) > 1 else 0.5
            return (len(beats) - 1) + (seconds - beats[-1]) / max(step, 1e-9)
        span = beats[index + 1] - beats[index]
        return index + (seconds - beats[index]) / max(span, 1e-9)

    slots = max(1, per_bar * grid)
    groups: dict[float, list[tuple[int, int, int]]] = {}
    off_grid = 0
    for start, end, pitch, velocity in part.notes:
        raw = beat_position(start) * grid
        if abs(raw - round(raw)) > 0.25:
            off_grid += 1
        length = max(int(round((beat_position(end) - beat_position(start)) * grid)), 1)
        groups.setdefault(round(start, 6), []).append((pitch, length, velocity))

    out = [f"[{part.stem}] {len(part.notes)} notes, program {part.program}"]
    if part.is_drum:
        out.append("# percussion: pitch numbers are General MIDI drum keys")
    if off_grid:
        out.append(
            f"# {off_grid} onset(s) fall more than a 1/{grid} beat off the grid: the "
            f"positions below are quantised labels and the durations are not re-derived "
            f"from them, so nothing has been moved"
        )

    placed = 0
    ticks: list[int] = []
    for start in sorted(groups):
        raw = beat_position(start) * grid
        ticks.append(int(round(raw)))

    previous_end = 0.0
    for index, start in enumerate(sorted(groups)):
        tick = ticks[index]
        bar = tick // slots + 1
        beat = (tick % slots) // grid + 1
        sub = tick % grid
        position = f"{bar}:{beat}.{sub}"

        gap = beat_position(start) - previous_end
        if gap >= grid * 2:
            out.append(f"{'':<10} {'rest':<10} {int(round(gap)):>4}    -")

        entries = groups[start]
        names = " ".join(note_name(p, flats=flats) for p, _l, _v in sorted(entries))
        length = min(l for _p, l, _v in entries)
        velocity = max(v for _p, _l, v in entries)
        out.append(f"{position:<10} {names:<10} {length:>4}  v{velocity}")
        placed += len(entries)
        previous_end = max(
            previous_end, beat_position(start) + max(l for _p, l, _v in entries) / grid
        )

    if placed != len(part.notes):
        # A silent mismatch here is how 64% of a part went missing once already.
        out.append(
            f"# WARNING: {placed} of {len(part.notes)} notes were written; "
            f"{len(part.notes) - placed} could not be placed"
        )
    return out


def parse_score_text(text: str, *, flats: bool = False) -> list[Part]:
    """Read the tabular form back into parts.

    The inverse of :func:`render_score_text`, and the reason the format is worth
    having: a corrected rendering can be turned back into notes rather than only
    looked at.  It is also how the round trip gets checked — counting notes by
    splitting lines on whitespace gets it wrong, because ``rest`` sits in the pitch
    column and ``v84`` ends in a digit only by luck.

    Recognised line shape::

        1:2.0      Eb4  G4        4  v80
        ^position  ^pitches       ^length  ^velocity
    """
    parts: list[Part] = []
    current: Part | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            stem = line[1:].split("]")[0].strip()
            current = Part(stem=stem, program=0, is_drum=False, notes=[])
            parts.append(current)
            continue
        if current is None:
            continue
        columns = line.split()
        if len(columns) < 4 or ":" not in columns[0]:
            continue
        position = columns[0]
        # Read the two trailing columns from the right, so a chord of any width in
        # between is preserved: taking only ``columns[1]`` keeps the lowest note of
        # every chord and silently discards the rest.
        if not columns[-1].startswith("v"):
            continue
        velocity = int(columns[-1][1:]) if columns[-1][1:].isdigit() else 80
        if not columns[-2].lstrip("-").isdigit():
            continue
        length_units = int(columns[-2])
        names = columns[1:-2]
        if not names or names == ["rest"]:
            continue
        bar_text, _, beat_text = position.partition(":")
        beat, _, tick = beat_text.partition(".")
        per_bar = 4
        beat_position = (
            (int(bar_text) - 1) * per_bar + (int(beat) - 1) + (int(tick or 0) / 4)
        )
        for name in names:
            try:
                pitch = _pitch_from_name(name)
            except ValueError:
                continue
            current.notes.append(
                (beat_position, beat_position + length_units / 4, pitch, velocity)
            )
    return parts


def _pitch_from_name(name: str) -> int:
    """``Eb4``/``C#5`` -> MIDI number.  Raises ``ValueError`` on anything else."""
    if not name or name[0].upper() not in _ABC_STEP_INDEX:
        raise ValueError(f"not a pitch name: {name!r}")
    letter = name[0].upper()
    accidental = 0
    index = 1
    while index < len(name) and name[index] in "#b":
        accidental += 1 if name[index] == "#" else -1
        index += 1
    digits = name[index:]
    if not digits.lstrip("-").isdigit():
        raise ValueError(f"not a pitch name: {name!r}")
    octave = int(digits)
    semitone = _ABC_STEP_INDEX.index(letter)
    # Semitone offsets of the natural notes within an octave: C D E F G A B.
    natural = (0, 2, 4, 5, 7, 9, 11)[semitone]
    return (octave + 1) * 12 + natural + accidental


def count_notes(text: str) -> int:
    """Note events in a rendering, or -1 if the file's own total disagrees."""
    return sum(len(part.notes) for part in parse_score_text(text))


def render_score_text(score: ScoreText, *, grid: int = 4, include_drums: bool = False) -> str:
    """Render the compact, editable text form."""
    lines = [
        f"# {score.track} | {score.beats_per_bar}/4 | {score.tempo_bpm:.1f} BPM | "
        f"{key_for_display(score.key)} | {len(score.parts)} part(s)",
        "# columns:  position(bar:beat.tick)  pitch(es)  length(1/16 beats)  velocity",
        "# several pitches on one line are a chord; 'rest' is a gap, not silence",
    ]
    flats = uses_flats(score.key)
    for part in score.parts:
        if part.is_drum and not include_drums:
            continue
        lines.append("")
        lines.extend(_render_part(part, score, grid, flats))
    lines.append("")
    lines.append(f"# {sum(len(p.notes) for p in score.parts)} notes total")
    for warning in score.warnings:
        lines.append(f"# WARNING: {warning}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# ABC
# --------------------------------------------------------------------------- #
_ABC_STEP_INDEX = "CDEFGAB"


def _abc_pitch(pitch: int, *, flats: bool) -> str:
    """ABC token for a MIDI pitch, spelled to the key.

    ABC octave convention, verified against music21's own reader rather than assumed:
    ``C`` is middle C (MIDI 60), ``c`` is an octave above (72), ``c'`` above that (84),
    and ``C,``/``C,,`` go down (48/36).  Getting this wrong is not cosmetic — an
    earlier version wrote MIDI 60 and 72 both as ``C``, so the output could not be
    read back at all and the whole octave was shifted.
    """
    name = note_name(pitch, flats=flats)
    letter = name[0]
    # The accidental, when present, is the second character: ``Bb4`` -> "b", ``C#4``
    # -> "#", ``C4`` -> "".  Testing for a digit instead of for an accidental reads
    # ``Bb4`` as letter "B" with no accidental and no octave.
    accidental = name[1] if len(name) > 1 and name[1] in "#b" else ""
    digits = name[1 + len(accidental):]
    if not digits:
        raise ValueError(f"cannot parse pitch name {name!r} from MIDI {pitch}")
    octave = int(digits)
    index = _ABC_STEP_INDEX.index(letter)

    if octave >= 5:
        # Lowercase from the octave above middle C, one apostrophe per further octave.
        token = "cdefgab"[index] + "'" * (octave - 5)
    else:
        token = "CDEFGAB"[index] + "," * (4 - octave)

    mark = {"#": "^", "b": "_"}.get(accidental[:1], "")
    return f"{mark}{token}"


def render_abc(score: ScoreText, *, include_drums: bool = False, grid: int = 2) -> str:
    """A readable ABC rendering: one voice per part, eighths on the grid.

    ABC is used alongside the tabular form because the two answer different
    questions.  The table is for *editing* — one line per onset, columns to compare.
    ABC is for *reading* — bar lines, beaming and key signature make a wrong
    accidental or a misplaced bar obvious at a glance, which is what a model needs to
    judge whether a phrase makes sense.

    This is a reading rendering, not the engraving path: the MusicXML from the
    notation stage stays authoritative for the printed score.
    """
    from bisect import bisect_right

    beats = score.beat_times or [0.0]
    per_bar = max(1, score.beats_per_bar)
    flats = uses_flats(score.key)

    def beat_position(seconds: float) -> float:
        index = bisect_right(beats, seconds) - 1
        if index < 0:
            step = (beats[1] - beats[0]) if len(beats) > 1 else 0.5
            return (seconds - beats[0]) / max(step, 1e-9)
        if index >= len(beats) - 1:
            step = (beats[-1] - beats[-2]) if len(beats) > 1 else 0.5
            return (len(beats) - 1) + (seconds - beats[-1]) / max(step, 1e-9)
        span = beats[index + 1] - beats[index]
        return index + (seconds - beats[index]) / max(span, 1e-9)

    tonic = key_for_display(score.key).split()[0]
    abc_key = tonic.replace("b", "_").replace("#", "^")
    mode = "m" if "minor" in key_for_display(score.key).lower() else ""
    lines = [
        "X:1",
        f"T:{score.track}",
        f"M:{per_bar}/4",
        f"L:1/{grid * per_bar}",
        f"Q:1/4={score.tempo_bpm:.0f}",
        f"K:{abc_key}{mode or ''}",
    ]
    # A bar must span the meter, not a fixed number of slots.  Writing one token per
    # slot with ``L:1/8`` makes each token an eighth note, so a 4/4 bar needs eight of
    # them; emitting only ``grid`` of them gives a bar of half the length, and every
    # bar in the file then reads as half-full.
    grid = 2  # tokens per beat, i.e. eighth notes
    tokens_per_bar = max(1, per_bar * grid)

    for part in score.parts:
        if part.is_drum and not include_drums:
            continue
        buckets: dict[int, list[int]] = {}
        for start, _end, pitch, _velocity in part.notes:
            buckets.setdefault(int(round(beat_position(start) * grid)), []).append(pitch)
        if not buckets:
            continue
        lines.append(f"V:{part.stem} name=\"{part.stem}\"")
        last = max(buckets)
        bar_index = 0
        pending = ""
        while bar_index * tokens_per_bar <= last:
            bar_start = bar_index * tokens_per_bar
            tokens: list[str] = []
            for tick in range(bar_start, bar_start + tokens_per_bar):
                pitches = buckets.get(tick)
                if not pitches:
                    tokens.append("z")
                    continue
                names = [_abc_pitch(p, flats=flats) for p in sorted(set(pitches))]
                if len(names) > 1:
                    tokens.append("[" + "".join(names) + "]")
                else:
                    tokens.append(names[0])
            bar = " ".join(tokens) + " |"
            # Join whole bars onto one output line, and break *between* bars only.
            # Breaking mid-bar to satisfy a width limit splits a measure across two
            # lines, which reads as two measures — a real reader then counts half the
            # bars the file actually has.
            if pending and len(pending) + 1 + len(bar) > 110:
                lines.append(pending)
                pending = bar
            else:
                pending = f"{pending} {bar}" if pending else bar
            bar_index += 1
        if pending:
            lines.append(pending)
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# convenience
# --------------------------------------------------------------------------- #
def write_all(directory: str | Path, out_dir: str | Path | None = None) -> dict[str, Path]:
    """Write both renderings for one delivered run."""
    path = Path(directory)
    target = Path(out_dir) if out_dir else path / "llm"
    target.mkdir(parents=True, exist_ok=True)
    score = load_run(path)
    written = {
        "notes": target / "score_notes.txt",
        "abc": target / "score.abc",
    }
    written["notes"].write_text(render_score_text(score), encoding="utf-8")
    written["abc"].write_text(render_abc(score), encoding="utf-8")
    return written
