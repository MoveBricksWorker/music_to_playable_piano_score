"""Run configuration.

One frozen-by-convention dataclass holds every knob.  It is deliberately plain
data: the pipeline never reads CLI state, and the CLI never reads pipeline state,
so a run can be reproduced exactly from ``run.json``'s ``config`` block.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .errors import ConfigError

__all__ = ["RunConfig", "SEPARATORS", "BEAT_BACKENDS", "TRANSCRIBERS", "PRIORS"]

SEPARATORS = ("auto", "demucs", "hpss", "dir", "none")
# `htdemucs_6s` adds piano and guitar to the four standard stems.  It does **not**
# add strings: no official Demucs model outputs a strings stem, so string parts can
# only ever land inside `other`.
DEMUCS_MODELS = ("htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra")
BEAT_BACKENDS = ("auto", "beat-this", "librosa")
TRANSCRIBERS = ("auto", "basic-pitch", "pyin", "sheetsage2", "transkun", "yourmt3")
PRIORS = ("pop", "none")
DEVICES = ("auto", "cpu", "cuda")


@dataclass
class RunConfig:
    """Everything that controls a single run."""

    # ---- input / output -------------------------------------------------
    input_path: str
    out_dir: str = "out"
    max_seconds: float | None = None
    title: str | None = None
    composer: str | None = None

    # ---- stage 1: separation --------------------------------------------
    separator: str = "auto"
    stems_dir: str | None = None
    # 6-stem by default: `other` is where piano would otherwise be buried, and this
    # project exists to produce one part per instrument.  The cost is time, and the
    # guitar stem is unreliable, but a dedicated piano part is a clear win for an
    # ensemble score.  Pass --demucs-model htdemucs to go back to four stems.
    demucs_model: str = "htdemucs_6s"
    device: str = "auto"
    # A stem quieter than either bound is treated as absent.  Separation models
    # emit near-silent bleed for instruments a song does not contain, and
    # transcribing that bleed fabricates a whole plausible-looking part.
    min_stem_level_db: float = -40.0
    max_stem_drop_db: float = 30.0

    # ---- stage 2: rhythm -------------------------------------------------
    beats: str = "auto"
    tempo: float | None = None           # user override in BPM

    # ---- stage 3: meter --------------------------------------------------
    meter: str | None = None             # user override, e.g. "3/4"
    meter_prior: str = "pop"
    allow_odd_meter: bool = False
    # A SheetSage2 output directory to use as a symbolic prior: its downbeats join
    # the meter evidence, its chords replace template matching, and its key and
    # section labels reach the score.  Never selected automatically — SheetSage2 and
    # its MERT-v2 parent are CC-BY-NC-4.0, so it is opt-in exactly like the other
    # non-permissive components in io/registry.py.
    ss2_prior: str | None = None
    ss2_chords: bool = True              # use the prior's chords instead of chroma

    # ---- stage 4: transcription -----------------------------------------
    transcriber: str = "auto"
    fix_octaves: bool = True             # monophonic octave-jump correction
    min_note_duration: float = 0.05
    # `--transcriber sheetsage2` needs its own interpreter: SheetSage2's released
    # code requires transformers 4.45, which this project's own environment does not
    # carry.  These two point at that environment and at its weight cache; both are
    # part of the transcription cache key, because a different interpreter or a
    # different cache is a different transcriber.
    ss2_python: str | None = None
    ss2_hf_home: str | None = None
    # When set, a second transcriber is run over the same stems and its notes are
    # written beside the score as MIDI only.  The purpose is a direct A/B of two
    # backends on identical input: re-running the whole pipeline would redo
    # separation, beats and notation to answer a question that is purely about
    # transcription.  The primary backend still owns the score.
    alt_transcriber: str | None = None

    # ---- harmony / percussion -------------------------------------------
    chords: bool = True
    drums: bool = True
    chroma_backend: str = "cqt"          # cqt (better) | stft (much faster)

    # ---- transcription ---------------------------------------------------
    # basic-pitch's two decision thresholds were hard-coded in the backend, so the
    # only way to find out whether its sparse piano output was a settings problem or a
    # model limit was to edit the source.  Measured on the piano stem, they are both:
    #   onset 0.5 / frame 0.3 -> 2452 notes, 1.44 notes per onset, 72% of onsets single
    #   onset 0.3 / frame 0.2 -> 6549 notes, 1.63 per onset, 66% single, chords to 8
    #   onset 0.2 / frame 0.1 -> 28345 notes, 2.62 per onset, chords to 21 (noise)
    # The permissive end manufactures notes, so the thresholds stay where they were by
    # default and are exposed for measurement rather than silently changed.
    onset_threshold: float = 0.5
    frame_threshold: float = 0.3
    minimum_note_length: float = 0.05    # seconds; below this a note is not real
    # basic-pitch stores an amplitude per note, and the distribution of those
    # amplitudes says how much of a part is real.  Measured on the six-stem piano and
    # guitar stems, a large majority of the notes sit at low confidence, and the notes
    # it is confident about are almost entirely single:
    #
    #   stem     notes   median conf   >=0.5    >=0.6   notes per onset at >=0.6
    #   vocals     796        0.546     59.3%    39.9%         1.00
    #   guitar    1803        0.464     39.6%    18.2%         1.07
    #   piano     2352        0.406     21.6%     6.4%         1.03
    #
    # Notes per onset falling to 1.00 as the floor rises is the whole story: the model
    # is trustworthy on monophonic lines and inventing the rest, so the polyphony it
    # reports on a piano stem is largely hallucination.  A floor removes the invention
    # and keeps the line.  It is deliberately below the median for every stem, so no
    # part is emptied -- vocals keep 59% of their notes, piano 22%, guitar 40%.
    min_confidence: float = 0.5
    # Gating near-silent stretches out of a stem *before* transcription.  A separated
    # stem is silent where its instrument is not playing, and a frame-based transcriber
    # normalises its input, so silence is not safe input: measured on one vocal stem,
    # an intro 64 dB below the rest of the song produced 117 invented notes.  Those
    # cannot be filtered afterwards, because their confidences overlap the real notes
    # almost completely -- so the test has to be made on the audio.
    #
    # Silence needs two conditions, because one is not enough: a frame counts only when
    # it is both near the stem's own noise floor and far below its active level.  A bare
    # "N dB below average" rule silenced 56% of one bass part and 67% of another.
    gate_silence: bool = True
    # Margins for that gate.  A wider floor margin catches faint bleed sitting just
    # above a stem's noise floor: measured on one vocal stem, a single note survived at
    # 9.3 s where the stem is 49.6 dB below its own body -- the separation was leaking,
    # not singing -- because that frame sat just outside the old 6 dB band.
    silence_floor_margin_db: float = 12.0   # within this of the stem's 2nd percentile
    silence_active_margin_db: float = 30.0  # and this far below its 75th percentile

    # ---- notation --------------------------------------------------------
    key: bool = True
    grid: int = 4                        # subdivisions per beat (4 = sixteenths)
    swing: float = 0.0
    # Measured on the piano stem, basic-pitch returns chords of up to six notes and
    # only eight at a looser threshold.  A ceiling below what the model produces can
    # only ever damage a chord, never help one: at 4 it removed 372 of 2355 notes and
    # pushed the single-note share *up*, from 72.1% to 73.3%, which is the opposite of
    # what a piano part needs.  At 8 it removes 1 note and nothing is lost.
    max_simultaneous: int = 8
    instrument_map: str | None = None
    pdf: bool = False
    mscore: str | None = None
    # One MIDI file per part, under out/midi/.  Separation plus per-stem
    # transcription already produces one instrument at a time, and being able to
    # listen to a single part without the others is how you tell whether a bad
    # score is a bad transcription or a bad mix of good ones.
    stem_midi: bool = True

    # ---- behaviour -------------------------------------------------------
    cache: bool = True
    offline: bool = False                # never touch the network
    verbose: bool = True
    keep_stems: bool = True

    extras: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        if not self.input_path:
            raise ConfigError("no input audio given")
        self.input_path = str(Path(self.input_path))
        self.out_dir = str(Path(self.out_dir))

        if self.separator not in SEPARATORS:
            raise ConfigError(f"--separator must be one of {SEPARATORS}, got {self.separator!r}")
        if self.beats not in BEAT_BACKENDS:
            raise ConfigError(f"--beats must be one of {BEAT_BACKENDS}, got {self.beats!r}")
        if self.transcriber not in TRANSCRIBERS:
            raise ConfigError(f"--transcriber must be one of {TRANSCRIBERS}, got {self.transcriber!r}")
        if self.alt_transcriber is not None and self.alt_transcriber not in TRANSCRIBERS:
            raise ConfigError(
                f"--alt-transcriber must be one of {TRANSCRIBERS}, got {self.alt_transcriber!r}"
            )
        if self.alt_transcriber is not None and self.alt_transcriber == self.transcriber:
            # Otherwise the A/B writes the same notes twice and looks like a result.
            raise ConfigError(
                f"--alt-transcriber is the same as --transcriber ({self.transcriber!r}): "
                "a comparison needs two different backends"
            )
        if self.meter_prior not in PRIORS:
            raise ConfigError(f"--meter-prior must be one of {PRIORS}")
        if self.device not in DEVICES:
            raise ConfigError(f"--device must be one of {DEVICES}")
        if self.chroma_backend not in ("cqt", "stft"):
            raise ConfigError("--chroma-backend must be 'cqt' or 'stft'")
        if self.demucs_model not in DEMUCS_MODELS:
            raise ConfigError(f"--demucs-model must be one of {DEMUCS_MODELS}")

        if self.stems_dir:
            # Supplying stems is an explicit instruction, so it wins over --separator.
            self.separator = "dir"
        if self.separator == "dir" and not self.stems_dir:
            raise ConfigError("--separator dir needs --stems-dir")

        if self.grid < 1 or self.grid > 16:
            raise ConfigError("--grid must be between 1 and 16")
        if self.grid & (self.grid - 1):
            raise ConfigError("--grid must be a power of two (1, 2, 4, 8, 16)")
        if not 0.0 <= self.swing <= 0.66:
            raise ConfigError("--swing must be between 0.0 and 0.66")
        if self.max_simultaneous < 1:
            raise ConfigError("--max-simultaneous must be at least 1")
        if self.min_note_duration < 0:
            raise ConfigError("--min-note-duration cannot be negative")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ConfigError("--min-confidence must be between 0.0 and 1.0")
        if not 0.0 <= self.onset_threshold <= 1.0:
            raise ConfigError("--onset-threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.frame_threshold <= 1.0:
            raise ConfigError("--frame-threshold must be between 0.0 and 1.0")
        if self.minimum_note_length < 0:
            raise ConfigError("--minimum-note-length cannot be negative")
        if self.max_seconds is not None and self.max_seconds <= 0:
            raise ConfigError("--max-seconds must be positive")
        if self.tempo is not None and not 20.0 <= self.tempo <= 400.0:
            raise ConfigError("--tempo must be between 20 and 400 BPM")
        if self.meter is not None:
            self.meter = self._check_meter(self.meter)

    @staticmethod
    def _check_meter(label: str) -> str:
        text = label.replace(" ", "")
        try:
            numerator, denominator = (int(part) for part in text.split("/"))
        except Exception as exc:  # noqa: BLE001 - re-raised with a better message
            raise ConfigError(f"--meter expects e.g. 4/4, got {label!r}") from exc
        if not 1 <= numerator <= 16:
            raise ConfigError(f"meter numerator out of range: {label!r}")
        if denominator not in (1, 2, 4, 8, 16):
            raise ConfigError(f"meter denominator must be a power of two: {label!r}")
        return f"{numerator}/{denominator}"

    # ------------------------------------------------------------------ #
    @property
    def meter_parts(self) -> tuple[int, int] | None:
        if not self.meter:
            return None
        numerator, denominator = self.meter.split("/")
        return int(numerator), int(denominator)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})
