"""Generate a small, copyright-free ensemble with ground truth.

Tests and CI need something to transcribe that is not a real song: no download, no
licence question, and — crucially — **known truth**, so an end-to-end run can
assert on tempo, time signature, chords and per-stem note content instead of just
"it did not crash".

The output is deliberately simple and maximally separable: pure tones, one
instrument per stem, no reverb, and a drum part built from synthesised kick/snare/
hi-hat so the classifier has a fair chance.

Contents, over a four-chord loop:

* ``vocals``   — the melody, one note per bar, high register, monophonic
* ``other``    — a sustained triad pad, mid register, polyphonic
* ``bass``     — root notes, low register
* ``drums``    — kick on 1 and 3, snare on 2 and 4, eighth-note hi-hat

Files written into ``out_dir``: ``mix.wav``, ``stems/*.wav``, ``truth.json``,
``truth.mid``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["SynthSpec", "make_ensemble", "PROGRESSION"]

SAMPLE_RATE = 22050

# A four-chord loop in C major, spelled as (root midi, triad midi pitches).
PROGRESSION: tuple[tuple[int, tuple[int, int, int]], ...] = (
    (48, (60, 64, 67)),  # C
    (43, (59, 62, 67)),  # G
    (45, (57, 60, 64)),  # Am
    (41, (57, 60, 65)),  # F
)

# One melody note per chord, chosen from the chord itself.
MELODY: tuple[int, ...] = (72, 71, 69, 69)  # C5, B4, A4, A4


@dataclass
class SynthSpec:
    """Knobs for the generated fixture."""

    tempo_bpm: float = 100.0
    bars: int = 8
    beats_per_bar: int = 4
    sample_rate: int = SAMPLE_RATE
    seed: int = 7
    melody_gain: float = 0.5
    pad_gain: float = 0.22
    bass_gain: float = 0.45
    drum_gain: float = 0.6

    @property
    def beat_seconds(self) -> float:
        return 60.0 / self.tempo_bpm

    @property
    def bar_seconds(self) -> float:
        return self.beat_seconds * self.beats_per_bar

    @property
    def duration(self) -> float:
        return self.bar_seconds * self.bars


def _sine(frequency: float, seconds: float, sample_rate: int, *, harmonics: int = 1) -> np.ndarray:
    """A tone with optional harmonics, windowed so it does not click."""
    t = np.arange(int(round(seconds * sample_rate))) / sample_rate
    signal = np.zeros_like(t)
    for harmonic in range(1, harmonics + 1):
        signal += np.sin(2 * np.pi * frequency * harmonic * t) / harmonic
    if signal.size:
        # 10 ms raised-cosine fade in/out.
        fade = max(1, int(0.01 * sample_rate))
        fade = min(fade, signal.size // 2) or 1
        envelope = np.ones_like(signal)
        envelope[:fade] = np.linspace(0.0, 1.0, fade)
        envelope[-fade:] = np.linspace(1.0, 0.0, fade)
        signal *= envelope
    return signal.astype(np.float32)


def _midi_to_hz(pitch: int) -> float:
    return 440.0 * 2.0 ** ((pitch - 69) / 12.0)


def _add(target: np.ndarray, signal: np.ndarray, start: float, sample_rate: int, gain: float) -> None:
    """Mix ``signal`` into ``target`` at ``start`` seconds, clipping to the buffer."""
    begin = int(round(start * sample_rate))
    if begin >= target.size:
        return
    end = min(begin + signal.size, target.size)
    target[begin:end] += gain * signal[: end - begin]


@dataclass
class _Truth:
    notes: dict[str, list[list[float]]] = field(default_factory=dict)
    chords: list[tuple[str, float]] = field(default_factory=list)


def _resolve_spec(spec: SynthSpec | None, overrides: dict) -> SynthSpec:
    """Merge an optional explicit spec with keyword overrides, cleanly."""
    if spec is None:
        return SynthSpec(**overrides)
    if not overrides:
        return spec
    known = {f for f in SynthSpec.__dataclass_fields__}
    unknown = set(overrides) - known
    if unknown:
        raise TypeError(f"unknown SynthSpec option(s): {sorted(unknown)}")
    return SynthSpec(**{**spec.__dict__, **overrides})


def make_ensemble(out_dir: str | Path, spec: SynthSpec | None = None, **overrides) -> dict:
    """Render the fixture and write its files.

    Returns a dict of paths plus a ``truth`` sub-dict, so a test can assert
    directly against the generated ground truth.
    """
    spec = _resolve_spec(spec, overrides)

    out = Path(out_dir)
    stems_dir = out / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)

    sr = spec.sample_rate
    total = int(round(spec.duration * sr))
    buffers = {
        "vocals": np.zeros(total, dtype=np.float32),
        "other": np.zeros(total, dtype=np.float32),
        "bass": np.zeros(total, dtype=np.float32),
        "drums": np.zeros(total, dtype=np.float32),
    }
    truth = _Truth()

    for key in buffers:
        truth.notes[key] = []

    noise = np.random.RandomState(spec.seed)

    for bar in range(spec.bars):
        chord_index = bar % len(PROGRESSION)
        root, triad = PROGRESSION[chord_index]
        bar_start = bar * spec.bar_seconds
        root_name = ("C", "G", "A", "F")[chord_index]

        # ---- melody: one note per bar, just under a full bar so notes separate
        melody_pitch = MELODY[chord_index]
        duration = spec.beat_seconds * (spec.beats_per_bar - 0.25)
        _add(buffers["vocals"], _sine(_midi_to_hz(melody_pitch), duration, sr, harmonics=3),
             bar_start, sr, spec.melody_gain)
        truth.notes["vocals"].append([melody_pitch, bar_start, bar_start + duration])

        # ---- pad: the triad, held for the bar
        pad_duration = spec.beat_seconds * spec.beats_per_bar
        for pitch in triad:
            _add(buffers["other"], _sine(_midi_to_hz(pitch), pad_duration, sr, harmonics=2),
                 bar_start, sr, spec.pad_gain)
        truth.notes["other"].extend(
            [[pitch, bar_start, bar_start + pad_duration] for pitch in triad]
        )

        # ---- bass: root, two half-bar notes
        for half in range(2):
            start = bar_start + half * spec.beat_seconds * 2
            length = spec.beat_seconds * 1.75
            _add(buffers["bass"], _sine(_midi_to_hz(root), length, sr, harmonics=2),
                 start, sr, spec.bass_gain)
            truth.notes["bass"].append([root, start, start + length])

        truth.chords.append((root_name, bar_start))

        # ---- drums: kick 1 & 3, snare 2 & 4, hi-hat on eighths
        for beat in range(spec.beats_per_bar):
            start = bar_start + beat * spec.beat_seconds
            if beat % 2 == 0:
                _add(buffers["drums"], _kick(sr), start, sr, spec.drum_gain)
            else:
                _add(buffers["drums"], _snare(sr, noise), start, sr, spec.drum_gain * 0.9)
            for eighth in range(2):
                _add(
                    buffers["drums"],
                    _hihat(sr, noise),
                    start + eighth * spec.beat_seconds / 2,
                    sr,
                    spec.drum_gain * 0.35,
                )

    mix = np.zeros(total, dtype=np.float32)
    paths: dict[str, str] = {"stems_dir": str(stems_dir), "sample_rate": sr}
    for name, buffer in buffers.items():
        peak = float(np.max(np.abs(buffer))) or 1.0
        # Normalise each stem so the detector thresholds behave consistently.
        buffer = (buffer / peak * 0.8).astype(np.float32)
        _write(stems_dir / f"{name}.wav", buffer, sr)
        mix += buffer * 0.7

    peak = float(np.max(np.abs(mix))) or 1.0
    mix = (mix / peak * 0.95).astype(np.float32)
    mix_path = out / "mix.wav"
    _write(mix_path, mix, sr)

    beat_times = [round(i * spec.beat_seconds, 6) for i in range(spec.bars * spec.beats_per_bar + 1)]
    truth_payload = {
        "tempo_bpm": spec.tempo_bpm,
        "beats_per_bar": spec.beats_per_bar,
        "bars": spec.bars,
        "sample_rate": sr,
        "duration": spec.duration,
        "beat_times": beat_times,
        "downbeat_times": [
            round(bar * spec.bar_seconds, 6) for bar in range(spec.bars + 1)
        ],
        "chords": [{"label": label, "start": start} for label, start in truth.chords],
        "notes": {name: notes for name, notes in truth.notes.items()},
    }
    truth_json = out / "truth.json"
    truth_json.write_text(json.dumps(truth_payload, indent=2), encoding="utf-8")

    truth_mid = out / "truth.mid"
    _write_truth_midi(truth_mid, truth, spec)

    paths.update(
        {
            "mix": str(mix_path),
            "truth": str(truth_json),
            "truth_mid": str(truth_mid),
            "stems": {name: str(stems_dir / f"{name}.wav") for name in buffers},
            "truth_data": truth_payload,
        }
    )
    return paths


# --------------------------------------------------------------------------- #
# percussion synthesis
# --------------------------------------------------------------------------- #
def _kick(sample_rate: int, seconds: float = 0.18) -> np.ndarray:
    """Pitch-swept sine: 90 Hz down to 45 Hz, fast decay."""
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    sweep = 90.0 * np.exp(-t * 18.0) + 45.0
    signal = np.sin(2 * np.pi * np.cumsum(sweep) / sample_rate)
    return (signal * np.exp(-t * 22.0)).astype(np.float32)


def _snare(sample_rate: int, noise: np.random.RandomState, seconds: float = 0.16) -> np.ndarray:
    """Noise burst plus a 200 Hz body."""
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    body = np.sin(2 * np.pi * 200.0 * t) * np.exp(-t * 30.0)
    hiss = noise.randn(t.size) * np.exp(-t * 26.0)
    return (0.5 * body + 0.5 * hiss).astype(np.float32)


def _hihat(sample_rate: int, noise: np.random.RandomState, seconds: float = 0.05) -> np.ndarray:
    """Short high-passed noise burst."""
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    hiss = noise.randn(t.size) * np.exp(-t * 110.0)
    # Crude high-pass: first difference emphasises the top end.
    hiss = np.diff(hiss, prepend=hiss[0])
    return (hiss * 0.5).astype(np.float32)


# --------------------------------------------------------------------------- #
# io helpers
# --------------------------------------------------------------------------- #
def _write(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.ascontiguousarray(audio, dtype=np.float32), sample_rate, subtype="PCM_16")


def _write_truth_midi(path: Path, truth: _Truth, spec: SynthSpec) -> None:
    """A reference MIDI with the same notes, for `e2s eval`."""
    import pretty_midi

    midi = pretty_midi.PrettyMIDI(initial_tempo=float(spec.tempo_bpm))
    programs = {"vocals": (53, False), "other": (0, False), "bass": (33, False), "drums": (0, True)}
    drum_pitches = {"kick": 36, "snare": 38, "hihat": 42}

    for name, (program, is_drum) in programs.items():
        instrument = pretty_midi.Instrument(program=program, is_drum=is_drum, name=name)
        for pitch, start, end in truth.notes.get(name, []):
            instrument.notes.append(
                pretty_midi.Note(velocity=80, pitch=int(pitch), start=float(start), end=float(end))
            )
        if name == "drums":
            instrument.notes.clear()  # drums are written from the pattern below
            for bar in range(spec.bars):
                bar_start = bar * spec.bar_seconds
                for beat in range(spec.beats_per_bar):
                    start = bar_start + beat * spec.beat_seconds
                    voice = "kick" if beat % 2 == 0 else "snare"
                    instrument.notes.append(
                        pretty_midi.Note(
                            velocity=90, pitch=drum_pitches[voice], start=start, end=start + 0.05
                        )
                    )
                    for eighth in range(2):
                        hat_start = start + eighth * spec.beat_seconds / 2
                        instrument.notes.append(
                            pretty_midi.Note(
                                velocity=60, pitch=drum_pitches["hihat"],
                                start=hat_start, end=hat_start + 0.05,
                            )
                        )
        midi.instruments.append(instrument)

    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))
