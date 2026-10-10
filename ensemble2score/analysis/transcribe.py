"""Per-stem transcription.

``basic-pitch``
    Polyphonic, instrument-agnostic, Apache-2.0, and its weights ship inside the
    package so there is nothing to download.  This is the default.
``pyin``
    librosa's probabilistic YIN.  Monophonic only; used as the fallback when
    basic-pitch is unavailable, and as an independent second opinion for the
    octave sanity check.

Octave jumps
------------
Single-voice transcription is famous for random octave errors, and they are the
most audible defect in the final score: the melody suddenly leaps an octave and
back.  :func:`fix_octave_jumps` looks for a short segment that sits ~12 semitones
away from a stable surrounding register, with no gradual approach, and pulls it
back.  It only ever touches an isolated excursion, never a sustained register
change, and every correction is reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Protocol, Sequence

import numpy as np

from ..core.errors import BackendError, BackendUnavailable
from ..core.models import Note, PartSpec
from ..io import audio as audio_io
from ..io.console import warn

__all__ = [
    "Transcriber",
    "build_transcriber",
    "BasicPitchTranscriber",
    "PyinTranscriber",
    "fix_octave_jumps",
    "fit_to_instrument_range",
    "merge_short_notes",
    "gate_silence",
    "gate_stem",
    "filter_by_confidence",
    "limit_simultaneous",
    "write_note_midi",
    "PLAUSIBLE_RANGE",
]

FRAME_LENGTH = 2048
HOP_LENGTH = 512

# Physically plausible *fundamental* ranges, as MIDI pitch numbers, for the
# instruments this project produces.  Deliberately generous at the edges: the aim
# is to remove the impossible, not to second-guess a performer.
#
# Why this is necessary: a polyphonic transcriber reports energy peaks, and a
# plucked or bowed string radiates strong harmonics.  On real material this project
# measured an electric-bass part spanning MIDI 24-86, where the top is roughly the
# 16th harmonic of a low E rather than a note any bassist played.  Filtering by
# range is what turns "every energy peak" back into "the note".
PLAUSIBLE_RANGE: dict[str, tuple[int, int]] = {
    "vocals": (36, 88),      # C2..E6: wide, but still human
    "bass": (24, 67),        # E1..G4, the top of a bass guitar's fingerboard
    "keys": (21, 88),        # A0..E7: a real 88-key piano, not "anything midi can name"
    "piano": (21, 88),
    "guitar": (40, 88),      # E2..E6
    "strings": (28, 100),
    "other": (28, 96),       # unknown instrument: instrumental, not full-range midi
}

# Share of a part that may fall outside its instrument's range before we say so.
_OUTLIER_FRACTION = 0.15


class Transcriber(Protocol):
    name: str

    def transcribe(self, audio_path: str, spec: PartSpec) -> tuple[list[Note], list[str]]: ...


# --------------------------------------------------------------------------- #
# post-processing shared by both backends
# --------------------------------------------------------------------------- #
def consolidate_notes(
    notes: Sequence[Note],
    *,
    join_gap: float = 0.08,
    merge_overlaps: bool = True,
) -> tuple[list[Note], list[str]]:
    """Join notes a transcriber split apart, before quantisation sees them.

    Every frame-based transcriber fragments: one sung note becomes several
    consecutive notes, and a note whose pitch wobbles by a semitone for a single
    frame becomes two.  Measured on a real vocal stem, 667 raw notes contained 146
    that quantised to **zero duration** and were then discarded 鈥?22% of the melody
    gone, and the survivors carved into sixteenth notes.

    Two repairs, both conservative:

    * **Overlaps.**  A monophonic part must not have two notes sounding at once.  A
      later note that starts over an earlier one of the *same* pitch is the same
      note; its end extends the earlier note rather than starting a new one.
    * **Short gaps.**  Two notes of the same pitch separated by less than
      ``join_gap`` seconds are one note.  The gap threshold is deliberately small:
      it should absorb transcription stutter, not rebar a phrase.

    Pitch changes are left alone 鈥?merging across a real pitch change would erase
    the melody.  A one-frame blip is handled because the *neighbouring* notes then
    join across it, and the blip itself is dropped as a short note later.
    """
    if not notes:
        return list(notes), []

    ordered = sorted(notes, key=lambda n: (n.start, n.end, n.pitch))
    merged: list[Note] = [Note(**vars(ordered[0]))]
    joined = 0

    for note in ordered[1:]:
        last = merged[-1]
        if merge_overlaps and last.pitch == note.pitch and note.start < last.end:
            # Same pitch sounding again while still held: extend, do not restart.
            if note.end > last.end:
                last.end = note.end
                last.velocity = max(last.velocity, note.velocity)
            joined += 1
            continue
        if last.pitch == note.pitch and 0 <= note.start - last.end <= join_gap:
            last.end = max(last.end, note.end)
            last.velocity = max(last.velocity, note.velocity)
            last.confidence = min(last.confidence, note.confidence)
            joined += 1
            continue
        merged.append(Note(**vars(note)))

    messages: list[str] = []
    if joined:
        messages.append(
            f"joined {joined} fragmented note(s) ({len(notes)} -> {len(merged)}): "
            "transcribers split one sustained note into several consecutive ones"
        )
    return merged, messages


def merge_short_notes(notes: Sequence[Note], *, minimum: float) -> tuple[list[Note], int]:
    """Drop notes shorter than ``minimum`` seconds.  Returns ``(notes, dropped)``."""
    if minimum <= 0:
        return list(notes), 0
    kept = [n for n in notes if n.duration >= minimum]
    return kept, len(notes) - len(kept)


def gate_silence(
    audio: np.ndarray,
    sample_rate: int,
    *,
    floor_margin_db: float = 12.0,
    active_margin_db: float = 30.0,
    min_gap_seconds: float = 0.5,
    frame: int = 2048,
    hop: int = 512,
) -> tuple[np.ndarray, float]:
    """Zero the long silent stretches of a stem before a transcriber sees them.

    A frame-based transcriber normalises its input, so **silence is not safe input**.
    Measured on the six-stem vocal part of one track: the intro sits at -74 dB against
    -10 dB for the rest of the song, a 64 dB difference that is digital silence for
    practical purposes, and demucs had separated it correctly.  basic-pitch still
    returned **117 notes** in that intro, with confidences reaching 0.79.

    Those notes cannot be filtered afterwards, and that is the important part: the
    confidence distributions of the invented notes and the real singing overlap almost
    completely (intro median 0.442, max 0.790; body median 0.560, p10 0.357).  Any
    confidence floor high enough to remove the inventions deletes most of the part --
    at 0.75, 38 of 705 real notes survive.  Confidence is the model's opinion of a
    note, not of the audio, so the test has to be made on the audio.

    Silence is recognised by **two** conditions, because one is not enough:

    * the frame is close to the stem's own noise floor (within ``floor_margin_db``), and
    * the frame is far below the stem's active level (``active_margin_db``).

    A single "N dB below the average" rule fails on real material: applied to four
    tracks it would silence 56% of collapse's bass, 67% of blessing's bass and 48% of
    collapse's piano, because the p90 level of a dynamic part sits near its peaks and
    everything below looks quiet next to it.  Quietness is not silence; the floor
    condition is what tells them apart, and a genuinely soft passage sits well above
    the floor and survives.

    **Only gaps at least ``min_gap_seconds`` long are removed**, and this turned out to
    matter more than how aggressive the thresholds are.  Measured on one vocal stem,
    every threshold tested gated roughly a quarter of the body; what changed was how
    many separate cuts it made -- 34 short ones at the gentlest setting, 8 at the
    strictest.  Every cut is a place where a sustained note can be split in two, and a
    split fragment can fall below ``--min-note-duration`` and vanish, which the user
    heard as missing notes in the melody.  The gate exists to remove stretches where the
    instrument is not playing, not to punch holes in a vocal line, so short gaps are
    closed up again and left to the confidence filter -- which removes whole notes
    instead of cutting them.

    Returns ``(gated_audio, gated_share)``.
    """
    values = np.asarray(audio, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[1] < frame:
        return np.asarray(audio), 0.0

    mono = values.mean(0)
    # Frames are centred on their hop position, so a frame describes exactly the
    # ``frame`` samples around it.  Taking frames from an offset grid instead means a
    # frame near a boundary averages loud and silent audio together and never looks
    # silent, leaving both edges of every gap untouched -- measured as a residual
    # ``1e-5`` left in a stretch that should have been zero.
    half = frame // 2
    length = values.shape[1]
    starts = np.arange(0, max(0, length - frame) + 1, hop)
    if starts.size == 0:
        return np.asarray(audio), 0.0
    index = starts[:, None] + np.arange(frame)[None, :]
    envelope = np.sqrt((mono[index] ** 2).mean(axis=1))
    envelope_db = 20 * np.log10(np.maximum(envelope, 1e-12))

    floor = float(np.percentile(envelope_db, 2))
    active = float(np.percentile(envelope_db, 75))
    silent = (envelope_db < floor + floor_margin_db) & (
        envelope_db < active - active_margin_db
    )
    if not silent.any():
        return np.asarray(audio), 0.0

    # A frame covers up to ``half`` samples either side of its centre and was judged
    # from the samples inside that span, so a silent frame marks exactly that span.
    mark = np.zeros(length, dtype=bool)
    for start in starts[silent]:
        mark[max(0, start - half) : start + frame + half] = True

    # Close any run too short to be the instrument stopping.
    keep = np.ones(length, dtype=bool)
    minimum_run = max(1, int(min_gap_seconds * sample_rate))
    position = 0
    while position < length:
        if not mark[position]:
            position += 1
            continue
        end = position
        while end < length and mark[end]:
            end += 1
        if end - position >= minimum_run:
            keep[position:end] = False
        position = end

    if keep.all():
        return np.asarray(audio), 0.0
    gated = values.copy()
    gated[:, ~keep] = 0.0
    return gated, float(1.0 - keep.mean())


def gate_stem(
    path: str,
    floor_margin_db: float,
    active_margin_db: float,
) -> tuple[str | None, float]:
    """Gate a stem file's near-silent stretches, returning ``(path, gated_share)``.

    Returns the original path unchanged when nothing needed gating, so the common case
    costs no temporary file.  The gated copy is written beside the original with a
    ``_gated`` suffix so it stays inside the run's own output directory and is visible
    when a run is inspected.
    """
    from pathlib import Path

    from ..io import audio as audio_io

    source = Path(path)
    audio, sample_rate = audio_io.load(str(source), mono=False)
    gated, share = gate_silence(
        audio,
        sample_rate,
        floor_margin_db=floor_margin_db,
        active_margin_db=active_margin_db,
    )
    if share <= 0.0:
        return None, 0.0
    target = source.with_name(f"{source.stem}_gated{source.suffix}")
    audio_io.save(str(target), gated, sample_rate)
    return str(target), share


def filter_by_confidence(
    notes: Sequence[Note], *, minimum: float
) -> tuple[list[Note], int]:
    """Drop notes the transcriber was not confident about.

    Separated stems are full of things that are not the instrument: bleed from the
    other parts, reverb tails, breath, and separation artefacts.  A frame-based model
    reports those as notes at low amplitude, so its own confidence is the cheapest
    available test for whether a note is really there.

    Measured with basic-pitch on the six-stem piano and guitar parts, the test is not
    marginal 鈥?it is the difference between a line and an invention:

    ==========  =======  ============  =======  =======  ====================
    stem        notes    median conf   >= 0.5   >= 0.6   notes/onset at >= 0.6
    ==========  =======  ============  =======  =======  ====================
    vocals        796         0.546     59.3%    39.9%          1.00
    guitar       1803         0.464     39.6%    18.2%          1.07
    piano        2352         0.406     21.6%     6.4%          1.03
    ==========  =======  ============  =======  =======  ====================

    Notes per onset converging to 1.00 as the floor rises is the point: what the model
    is confident about is monophonic, and its polyphonic output on these stems is
    mostly hallucination.  A floor keeps the line and drops the invention.

    ``minimum <= 0`` disables the filter, so a caller can ask for the raw output.
    """
    if minimum <= 0:
        return list(notes), 0
    kept = [n for n in notes if n.confidence >= minimum]
    return kept, len(notes) - len(kept)


def limit_simultaneous(notes: Sequence[Note], maximum: int) -> tuple[list[Note], int]:
    """Keep at most ``maximum`` notes sounding at once.

    Notation cannot express more than a handful of simultaneous voices meaningfully,
    and a transcriber's extras are usually artefacts.  The loudest (then lowest)
    notes win, so a bass line survives under a chord.
    """
    if maximum < 1:
        return list(notes), 0

    ordered = sorted(notes, key=lambda n: (n.start, -n.velocity, n.pitch))
    kept: list[Note] = []
    removed = 0
    for note in ordered:
        overlapping = sum(1 for k in kept if k.start < note.end and note.start < k.end)
        if overlapping >= maximum:
            removed += 1
            continue
        kept.append(note)
    return sorted(kept, key=lambda n: (n.start, n.pitch)), removed


def fix_octave_jumps(
    notes: Sequence[Note],
    *,
    max_shift_seconds: float = 1.0,
    register_semitones: int = 3,
    max_notes: int = 4,
    octave_tolerance: float = 2.0,
) -> tuple[list[Note], list[str]]:
    """Pull isolated octave excursions back into the surrounding register.

    An excursion is corrected only when all of these hold:

    * the run sits roughly an octave (within ``octave_tolerance`` semitones of 12)
      away from the median of the notes **before** it;
    * the surrounding registers on both sides agree with each other, so this really
      is a leap out of a stable register rather than a phrase that moves;
    * the run is a similar distance from the median of the notes **after** it, so
      the register it jumped *to* is not itself the stable one;
    * it lasts no longer than ``max_shift_seconds`` and contains at most
      ``max_notes`` notes.

    The third condition is what makes this direction-aware, and it is separate from
    the second on purpose.  Requiring only "the two sides are close to each other"
    transposes legitimate leaps: in a melody that goes 60-62-64-76-74-72, the 76 is
    an octave above the notes before it, the notes after it are within three
    semitones of those before, and the old code pulled it down to 64 鈥?inverting the
    contour at the phrase's high point.  Asking whether the run is *also* an octave
    from what follows distinguishes "leapt up and stayed" from "leapt up and came
    back".

    The note-count cap is what stops **chords** from being transposed.  Notes are
    ordered by onset, so the upper notes of a block chord are "an octave above the
    preceding line" and every one of them qualified as an isolated excursion; on real
    tracks this shifted dozens of notes and flattened exactly the voicings the score
    exists to convey.
    """
    if len(notes) < 3:
        return list(notes), []

    ordered = sorted(notes, key=lambda n: n.start)
    pitches = np.array([n.pitch for n in ordered], dtype=float)
    messages: list[str] = []
    fixed = 0

    def median_of(values: np.ndarray) -> float | None:
        return float(np.median(values)) if values.size else None

    index = 0
    while index < len(ordered):
        # Find the run of consecutive-in-time notes that share this octave shift.
        start = index
        before = pitches[max(0, start - 4): start]
        reference = median_of(before) if start >= 2 else None
        if reference is None:
            index += 1
            continue

        offset = pitches[start] - reference
        if abs(abs(offset) - 12.0) > octave_tolerance:
            index += 1
            continue

        end = start
        while (
            end + 1 < len(ordered)
            and abs((pitches[end + 1] - reference) - offset) <= register_semitones
        ):
            end += 1

        span = ordered[end].end - ordered[start].start
        after = pitches[end + 1: end + 5]
        after_reference = median_of(after)
        stable_surroundings = (
            after_reference is not None
            and abs(float(np.median(before)) - after_reference) <= register_semitones
        )
        # The register the run leapt to must not be the one the music continues in.
        run_pitch = float(np.median(pitches[start: end + 1]))
        isolated = (
            after_reference is not None
            and abs(abs(run_pitch - after_reference) - 12.0) <= octave_tolerance
        )

        run_length = end - start + 1
        if (
            span <= max_shift_seconds
            and stable_surroundings
            and isolated
            and run_length <= max_notes
        ):
            # Drop or raise the run by exactly the octave it is out by, keeping its
            # sign: an "octave error" is a note reported in the wrong octave, so the
            # correction is one octave, not whatever lands nearest the surrounding
            # median.  On a line that sits between two scale degrees, snapping to the
            # median would move the run by a sixth and change the melody.
            shift = -12 if offset > 0 else 12
            for note in ordered[start: end + 1]:
                note.pitch += shift
                note.confidence = max(0.0, note.confidence - 0.05)
            pitches[start: end + 1] += shift
            fixed += run_length
            index = end + 1
        else:
            index = start + 1

    if fixed:
        messages.append(
            f"octave correction: shifted {fixed} note(s) back into the surrounding register"
        )
    return ordered, messages


def fit_to_instrument_range(
    notes: Sequence[Note],
    spec: PartSpec,
    *,
    fix_octaves: bool = True,
    octave_tolerance: int = 3,
) -> tuple[list[Note], list[str]]:
    """Remove notes the instrument physically cannot play.

    A polyphonic model reports energy peaks, and harmonics of a low note are real
    energy peaks at pitches nobody played.  Measured symptom: an electric-bass part
    whose pitches ran to MIDI 86, roughly the 16th harmonic of a low E.  Filtering
    to the instrument's plausible fundamental range removes those, and also catches
    the commonest transcription error of all 鈥?a whole part sitting an octave out.

    Two things this deliberately does **not** do any more:

    * **It does not clamp near-misses onto the range edge.**  The old behaviour
      pulled anything within three semitones of the limit *onto* the limit, so every
      distinct note just above a bass guitar's top G came out as the same G 鈥?which
      is not a wrong note, it is a wrong melody, and it also manufactured unisons
      out of thin air.  Out-of-range notes are dropped, and the count is reported.
    * **It does not shift single notes by an octave.**  A real octave error in a
      monophonic part is caught by :func:`fix_octave_jumps` against the surrounding
      register, which is evidence about *that part*.  Here the only octave repair
      offered is a global one: if shifting the whole part by an octave puts
      materially more of it in range, the part was transcribed an octave out and the
      whole part is shifted.  Anything less than that is a harmonic, not a note.
    """
    if not notes:
        return list(notes), []

    key = spec.stem if spec.stem in PLAUSIBLE_RANGE else spec.instrument.lower()
    low, high = PLAUSIBLE_RANGE.get(key, PLAUSIBLE_RANGE["other"])
    if spec.sounding_pitch:
        # Compare sounding pitch against sounding range.
        low, high = low + spec.sounding_pitch, high + spec.sounding_pitch

    messages: list[str] = []
    working = list(notes)

    if fix_octaves:
        shifted, shift = _octave_shift_for_range(working, low, high)
        if shift:
            working = shifted
            messages.append(
                f"{key}: the whole part sat an octave out of range "
                f"({'+' if shift > 0 else ''}{shift} semitones): shifted it back"
            )

    kept: list[Note] = []
    dropped = 0
    for note in working:
        if low <= note.pitch <= high:
            kept.append(note)
        else:
            dropped += 1

    if dropped:
        messages.append(f"{key} range {low}-{high}: dropped {dropped} note(s) outside it")

    if len(notes) >= 20 and dropped / len(notes) > _OUTLIER_FRACTION:
        messages.append(
            f"{dropped / len(notes):.0%} of this part lies outside the {key} range "
            f"({low}-{high}): it may be mislabelled, or the separation may have bled"
        )
    return sorted(kept, key=lambda n: (n.start, n.pitch)), messages


def _octave_shift_for_range(
    notes: Sequence[Note], low: int, high: int
) -> tuple[list[Note], int]:
    """Shift a whole part by an octave when that puts much more of it in range.

    Returns ``(notes, semitones)``; ``semitones`` is 0 when no shift is justified.
    Requiring a large *gain* in in-range notes, rather than merely any gain, is what
    stops a part that legitimately spans the whole range from being transposed.
    """
    inside = sum(1 for note in notes if low <= note.pitch <= high)
    best_shift, best_inside = 0, inside
    for shift in (12, -12, 24, -24):
        count = sum(1 for note in notes if low <= note.pitch + shift <= high)
        if count > best_inside:
            best_shift, best_inside = shift, count

    # Both a margin and a fraction: a handful of notes moving in is noise, and a
    # two-octave shift is only ever right when the part was almost entirely out.
    if not best_shift or best_inside < inside + max(3, 0.25 * len(notes)):
        return list(notes), 0

    shifted: list[Note] = []
    for note in notes:
        moved = Note(**vars(note))
        moved.pitch = int(note.pitch) + best_shift
        shifted.append(moved)
    return shifted, int(best_shift)


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
class BasicPitchTranscriber:
    """Spotify's Basic Pitch (Apache-2.0).  Polyphonic, weights bundled."""

    name: str = "basic-pitch"

    def __init__(
        self,
        device: str = "auto",
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.3,
        minimum_note_length: float = 0.05,
        **_: object,
    ) -> None:
        self.name = "basic-pitch"
        self.device = device
        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold
        self.minimum_note_length = minimum_note_length

    def transcribe(self, audio_path: str, spec: PartSpec) -> tuple[list[Note], list[str]]:
        try:
            from basic_pitch import ICASSP_2022_MODEL_PATH
            from basic_pitch.inference import predict
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailable(
                f"basic-pitch is not usable ({type(exc).__name__}: {exc}); "
                'install it with: pip install "ensemble2score[transcription]"'
            ) from exc

        _model_output, _midi_data, note_events = predict(
            audio_path,
            ICASSP_2022_MODEL_PATH,
            onset_threshold=self.onset_threshold,
            frame_threshold=self.frame_threshold,
            minimum_note_length=self.minimum_note_length * 1000.0,
        )

        notes: list[Note] = []
        for event in note_events:
            start, end, pitch, amplitude, _bends = event
            notes.append(
                Note(
                    pitch=int(pitch),
                    start=float(start),
                    end=float(end),
                    velocity=int(np.clip(round(float(amplitude) * 127), 1, 127)),
                    confidence=float(amplitude),
                )
            )
        return sorted(notes, key=lambda n: (n.start, n.pitch)), []


def _load_transkun_config(conf_path: Path):
    """Read `transkun/pretrained/2.0.conf` through `moduleconf`.

    `moduleconf` is listed in Transkun's own dependencies but is not actually installed by
    `pip install transkun --no-deps`, which is how it has to be installed here (see the note
    on `ncls` below).  It is therefore required explicitly and its absence is a clear error
    rather than a silent fallback.

    An earlier version of this file carried a hand-written stub that built the config class
    with `Config()` and stopped there.  That looks fine and is wrong: `moduleconf`'s
    `makeModuleConf` instantiates the class **and then `loadDict` copies every value out of
    the JSON into it**.  Skipping the second step leaves every hyper-parameter at its default
    -- `baseSize` 160 instead of 64 -- and `load_state_dict` then fails with a wall of shape
    mismatches.  The real package gets `missing=0, unexpected=0`, so the stub is gone; a
    dependency is cheaper than a reimplementation of it.
    """
    try:
        import moduleconf
    except Exception as exc:  # noqa: BLE001
        raise BackendUnavailable(
            f"moduleconf is required to read Transkun's configuration "
            f"({type(exc).__name__}: {exc}); install it with: pip install moduleconf"
        ) from exc
    return moduleconf.parseFromFile(str(conf_path))


class TranskunTranscriber:
    """Transkun 2.0 (MIT, weights shipped inside the wheel).

    Chosen over basic-pitch for piano specifically.  Measured on this project's material
    (`work/_research_2026/transkun_probe.py`):

        mean simultaneous notes per onset   Transkun 1.24-1.42, max 6
        SheetSage2                          1.00 on every stem (it is a lead-sheet model and
                                            cannot emit a chord at all)
        share of onsets that are single notes, Transkun: ~80%

    and on the arrangement task it was the one downstream change that visibly helped: the
    triplet-and-arpeggio material that was 20% semitone-chopped at 4 tick/beat came down to
    3-9%.

    The package is imported lazily and its failure modes are converted to
    :class:`BackendUnavailable` so `auto` can fall back rather than crashing a run.

    Two details that are easy to get wrong and are handled here:

    * ``model.transcribe`` wants ``(frames, channels)``, so mono audio must be unsqueezed on
      the last axis.  Passing ``(frames,)`` raises deep inside the model.
    * the checkpoint lives inside the installed package, not in a cache directory, so the
      path is derived from ``transkun.__file__``.
    """

    name: str = "transkun"

    def __init__(self, device: str = "auto", **_: object) -> None:
        self.name = "transkun"
        self.device = device
        self._model = None
        self._conf = None
        self._warnings: list[str] = []
        self._fs = 44100

    # -- setup --------------------------------------------------------------- #
    def _load(self):
        if self._model is not None:
            return self._model, self._conf
        try:
            import torch
            import transkun
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailable(
                f"transkun is not usable ({type(exc).__name__}: {exc}); "
                "install it with: pip install transkun"
            ) from exc

        package = Path(transkun.__file__).resolve().parent
        conf_path = package / "pretrained" / "2.0.conf"
        weight_path = package / "pretrained" / "2.0.pt"
        if not weight_path.is_file():
            raise BackendUnavailable(f"transkun checkpoint missing at {weight_path}")

        manager = _load_transkun_config(conf_path)
        conf = manager["Model"].config

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        model = manager["Model"].module.TransKun(conf=conf).to(device)
        checkpoint = torch.load(str(weight_path), map_location=device)
        state = checkpoint.get("best_state_dict") or checkpoint.get("state_dict")
        if state is None:
            raise BackendUnavailable("transkun checkpoint has no recognisable state dict")
        model.load_state_dict(state, strict=False)
        model.eval()
        self._model, self._conf = model, conf
        self._fs = int(conf.fs)
        return model, conf

    # -- transcription ------------------------------------------------------- #
    def transcribe(self, audio_path: str, spec: PartSpec) -> tuple[list[Note], list[str]]:
        import torch

        model, conf = self._load()
        warnings = list(self._warnings)
        self._warnings = []

        try:
            import torchaudio

            wav, fs = torchaudio.load(audio_path)
            audio = wav.mean(dim=0).numpy().astype(np.float32)
        except Exception:  # noqa: BLE001
            # torchaudio's MP3 path needs a backend that is not always present; pydub reads
            # the same files through ffmpeg, which this project already requires.
            from pydub import AudioSegment

            segment = AudioSegment.from_file(audio_path)
            frames = np.array(segment.get_array_of_samples()).reshape(-1, segment.channels)
            audio = (frames.astype(np.float32) / 2 ** 15).mean(axis=1)
            fs = segment.frame_rate

        if fs != self._fs:
            try:
                import soxr

                audio = soxr.resample(audio, fs, self._fs).astype(np.float32)
            except Exception:  # noqa: BLE001
                import librosa

                audio = librosa.resample(audio, orig_sr=fs, target_sr=self._fs)
            warnings.append(f"resampled {fs} Hz -> {self._fs} Hz for Transkun")
            fs = self._fs

        x = torch.from_numpy(audio).to(self.device)
        if x.dim() == 1:
            x = x[:, None]          # (frames,) -> (frames, channels); the model needs this

        with torch.no_grad():
            result = model.transcribe(
                x, stepInSecond=None, segmentSizeInSecond=None, discardSecondHalf=False
            )

        notes: list[Note] = []
        # `model.transcribe` returns a `transkun.Data.NoteList`, whose elements are
        # `transkun.Data.Note` **objects** with `.start/.end/.pitch/.velocity`, not rows.
        # The first version of this method indexed them as `row[0]..row[3]`, which raised
        # TypeError for every note; the per-note try/except then swallowed all of them and
        # the backend cheerfully reported zero notes for a 40 MB piano stem.  Rows are still
        # accepted, because that is what a `NoteList` looks like when iterated by some
        # versions, and silently returning nothing is the worst possible failure mode here.
        rows = getattr(result, "notes", result)
        for row in rows:
            try:
                if hasattr(row, "pitch"):
                    start = float(row.start)
                    end = float(row.end)
                    pitch = int(round(float(row.pitch)))
                    velocity = float(row.velocity)
                else:
                    start = float(row[0])
                    end = float(row[1])
                    pitch = int(round(float(row[2])))
                    velocity = float(row[3])
            except (TypeError, IndexError, ValueError, AttributeError):
                continue
            # Transkun emits control changes in the same list as notes, encoded as a
            # negative pitch (`writeMidi` turns them back into CC#-pitch).  They are not
            # notes and must not be counted as such.
            if pitch <= 0:
                continue
            if end <= start:
                end = start + 0.02
            if not 0 <= pitch <= 127:
                continue
            notes.append(
                Note(
                    pitch=pitch,
                    start=start,
                    end=end,
                    velocity=int(np.clip(round(velocity), 1, 127)),
                    # Transkun's neural semi-CRF does not expose a per-note probability.
                    # `Note.confidence` is a float and the shared post-processing chain
                    # filters on it, so 1.0 is used to mean "no opinion" -- every note
                    # survives.  Reporting a fabricated value would silently drop real
                    # notes, and 0.0 would drop all of them; 1.0 plus the warning below is
                    # the only honest option.  The right fix is a `confidence is None`
                    # convention threaded through the chain, which is a wider change than
                    # this backend.
                    confidence=1.0,
                )
            )
        if not notes:
            warnings.append(
                "Transkun returned no notes for this stem; an empty result is reported "
                "rather than being treated as a silent success"
            )
        warnings.append(
            "Transkun reports no per-note confidence, so the confidence filter has no "
            "effect on this backend's notes"
        )
        return sorted(notes, key=lambda n: (n.start, n.pitch)), warnings


class PyinTranscriber:
    """librosa pyin.  Monophonic; the honest fallback when no model is installed."""
    name: str = "pyin"

    def __init__(self, device: str = "auto", **_: object) -> None:
        self.name = "pyin"
        self.device = device  # accepted for interface parity; pyin is CPU-only
        self.threshold = 0.1
        self.minimum_note_length = 0.08

    def transcribe(self, audio_path: str, spec: PartSpec) -> tuple[list[Note], list[str]]:
        import librosa

        low, high = self._range_for(spec)
        y, sr = audio_io.load(audio_path, mono=True)
        f0, voiced, probability = librosa.pyin(
            y,
            fmin=low,
            fmax=high,
            sr=sr,
            frame_length=FRAME_LENGTH,
            hop_length=HOP_LENGTH,
            fill_na=np.nan,
        )

        frequency = np.asarray(f0, dtype=float)
        pitched = np.isfinite(frequency) & (np.asarray(voiced, dtype=float) > 0)
        confidence = np.nan_to_num(np.asarray(probability, dtype=float))

        notes: list[Note] = []
        index = 0
        while index < frequency.size:
            if not pitched[index]:
                index += 1
                continue
            end = index
            while end + 1 < frequency.size and pitched[end + 1]:
                end += 1

            segment = frequency[index: end + 1]
            for onset_frame, offset_frame, pitch in self._segment_notes(
                segment, index, end, sr
            ):
                start = librosa.frames_to_time(onset_frame, sr=sr, hop_length=HOP_LENGTH)
                stop = librosa.frames_to_time(offset_frame + 1, sr=sr, hop_length=HOP_LENGTH)
                if stop - start < self.minimum_note_length:
                    continue
                strength = float(np.mean(confidence[onset_frame: offset_frame + 1])) or 0.5
                notes.append(
                    Note(
                        pitch=int(round(pitch)),
                        start=float(start),
                        end=float(stop),
                        velocity=int(np.clip(round(strength * 127), 1, 127)),
                        confidence=float(strength),
                    )
                )
            index = end + 1

        return sorted(notes, key=lambda n: (n.start, n.pitch)), [
            "pyin is monophonic: overlapping notes in this stem were reduced to one line"
        ]

    @staticmethod
    def _segment_notes(
        segment: np.ndarray, first_frame: int, last_frame: int, sample_rate: int
    ) -> list[tuple[int, int, float]]:
        """Split a voiced run into stable-pitch notes (median-filtered)."""
        from scipy.signal import medfilt

        pitches = 69.0 + 12.0 * np.log2(segment / 440.0)
        kernel = 5 if pitches.size >= 5 else (pitches.size if pitches.size % 2 else pitches.size - 1)
        smoothed = medfilt(pitches, kernel_size=max(kernel, 1)) if kernel >= 3 else pitches
        rounded = np.round(smoothed)

        results: list[tuple[int, int, float]] = []
        run_start = 0
        for offset in range(1, len(rounded) + 1):
            if offset == len(rounded) or rounded[offset] != rounded[run_start]:
                if offset - run_start >= 2:  # ignore one-frame blips
                    absolute = int(first_frame) + run_start
                    results.append((absolute, int(first_frame) + offset - 1, float(rounded[run_start])))
                run_start = offset
        return results

    @staticmethod
    def _range_for(spec: PartSpec) -> tuple[float, float]:
        """Sensible pitch search bounds per instrument role."""
        if spec.midi_program == 33 or spec.clef == "bass":  # electric bass
            return 30.0, 400.0
        if spec.monophonic and spec.midi_program == 53:  # voice
            return 65.0, 1200.0
        return 55.0, 2000.0


def write_note_midi(
    notes: Sequence[Note],
    spec: PartSpec,
    path: str | Path,
    *,
    tempo_bpm: float = 120.0,
) -> Path | None:
    """Write one stem's notes as a standalone MIDI, or ``None`` when there are none.

    Used for the A/B comparison backend.  This writes **performance-level** times,
    not a quantised grid: the point of the comparison is to hear what the transcriber
    reported, and quantising it here would fold the notation stage's decisions into a
    judgement about the transcriber.  Times are therefore exact seconds and the tempo
    marking is nominal.
    """
    if not notes:
        return None

    import pretty_midi

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    midi = pretty_midi.PrettyMIDI(initial_tempo=float(max(tempo_bpm, 1.0)))
    instrument = pretty_midi.Instrument(
        program=int(spec.midi_program or 0),
        is_drum=bool(spec.drum_channel),
        name=spec.part_name,
    )
    shift = int(spec.sounding_pitch or 0)
    for note in notes:
        start = float(note.start)
        instrument.notes.append(
            pretty_midi.Note(
                velocity=int(max(1, min(127, note.velocity))),
                pitch=int(max(0, min(127, note.pitch + shift))),
                start=max(0.0, start),
                end=max(start + 1e-3, float(note.end)),
            )
        )
    midi.instruments.append(instrument)
    midi.write(str(target))
    return target


def build_transcriber(
    name: str,
    *,
    device: str = "auto",
    onset_threshold: float | None = None,
    frame_threshold: float | None = None,
    minimum_note_length: float | None = None,
    ss2_python: str | None = None,
    ss2_hf_home: str | None = None,
) -> Transcriber:
    """Instantiate a transcriber.

    ``auto`` prefers basic-pitch and falls back to pyin with a warning that the
    result will be monophonic.  ``transkun`` is wired but ``yourmt3`` is not: yourmt3 is
    copyleft, so it cannot be a delivery path.

    ``sheetsage2`` is wired but **never** reachable from ``auto``: it is CC-BY-NC-4.0
    and needs its own interpreter, so it has to be named.  See
    :mod:`~ensemble2score.analysis.sheetsage2_backend`.

    The three thresholds are optional so that a caller which does not care keeps the
    backend's own defaults; the pipeline passes the configured values through.
    """
    kwargs: dict[str, float] = {}
    if onset_threshold is not None:
        kwargs["onset_threshold"] = onset_threshold
    if frame_threshold is not None:
        kwargs["frame_threshold"] = frame_threshold
    if minimum_note_length is not None:
        kwargs["minimum_note_length"] = minimum_note_length

    if name == "pyin":
        return PyinTranscriber(device=device)
    if name == "basic-pitch":
        return BasicPitchTranscriber(device=device, **kwargs)
    if name == "transkun":
        return TranskunTranscriber(device=device)
    if name == "sheetsage2":
        from .sheetsage2_backend import SheetSage2Transcriber, default_hf_home

        return SheetSage2Transcriber(
            python=ss2_python,
            device=device,
            hf_home=ss2_hf_home or default_hf_home(),
        )
    if name == "yourmt3":
        raise BackendError(
            "the 'yourmt3' backend is not wired up, and cannot be: it is copyleft, so it "
            "may not appear in a delivery path. Supported: basic-pitch, pyin, transkun, "
            "sheetsage2."
        )
    if name == "auto":
        try:
            import basic_pitch  # noqa: F401
        except Exception:  # noqa: BLE001
            warn("basic-pitch is not installed: falling back to pyin (monophonic, lower quality)")
            return PyinTranscriber(device=device)
        return BasicPitchTranscriber(device=device, **kwargs)
    raise BackendError(f"unknown transcriber {name!r}")
