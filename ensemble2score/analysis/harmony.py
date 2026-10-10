"""Chord symbols — a small, self-contained template matcher.

Why not a library: the good open chord models (BTC, Harmony Transformer, CREMA)
each need their own heavy checkpoint and each brings its own licence question,
while chord symbols are a *nice-to-have* on top of the score.  Template matching
over beat-synchronous chroma with dynamic-programming smoothing is about 100
lines, has nothing to download, and is good enough to print useful symbols.

Accuracy on a dense pop mix is modest.  Chords are labelled advisory everywhere
they appear, and the report says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..core.models import Chord
from ..io import audio as audio_io

__all__ = ["ChordRecognizer", "PITCH_NAMES", "QUALITIES", "NO_CHORD"]

PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# interval set, per-tone weight, and the suffix used when printing.  The root and
# third are weighted highest because they are what distinguishes the qualities.
QUALITIES: dict[str, tuple[tuple[int, ...], dict[int, float], str]] = {
    "maj": ((0, 4, 7), {0: 1.0, 4: 1.0, 7: 0.9}, ""),
    "min": ((0, 3, 7), {0: 1.0, 3: 1.0, 7: 0.9}, "m"),
    "7": ((0, 4, 7, 10), {0: 1.0, 4: 0.95, 7: 0.8, 10: 0.95}, "7"),
    "maj7": ((0, 4, 7, 11), {0: 1.0, 4: 0.95, 7: 0.8, 11: 0.9}, "maj7"),
    "m7": ((0, 3, 7, 10), {0: 1.0, 3: 0.95, 7: 0.8, 10: 0.95}, "m7"),
    "dim": ((0, 3, 6), {0: 1.0, 3: 0.9, 6: 0.9}, "dim"),
    "sus4": ((0, 5, 7), {0: 1.0, 5: 0.9, 7: 0.9}, "sus4"),
}

NO_CHORD = "N"
NO_CHORD_LABEL = "N.C."


def _templates() -> tuple[list[str], np.ndarray]:
    labels: list[str] = []
    rows: list[np.ndarray] = []
    for root in range(12):
        for quality, (intervals, weights, _) in QUALITIES.items():
            vector = np.zeros(12, dtype=float)
            for interval in intervals:
                vector[(root + interval) % 12] = weights[interval]
            labels.append(f"{PITCH_NAMES[root]}:{quality}")
            rows.append(vector / (np.linalg.norm(vector) + 1e-9))
    labels.append(NO_CHORD)
    rows.append(np.zeros(12, dtype=float))
    return labels, np.stack(rows)


def format_label(label: str) -> str:
    """``"C:maj7"`` -> ``"Cmaj7"``; the no-chord state -> ``"N.C."``."""
    if label == NO_CHORD:
        return NO_CHORD_LABEL
    root, quality = label.split(":")
    return f"{root}{QUALITIES[quality][2]}"


@dataclass
class ChordRecognizer:
    """Template matching plus Viterbi smoothing over beat-synchronous chroma."""

    hop_length: int = 512
    switch_penalty: float = 0.06
    energy_floor: float = 1e-3
    relative_energy_floor: float = 0.3  # fraction of the median beat energy
    min_beats: float = 1.0
    max_segments: int = 400
    chroma_backend: str = "cqt"
    _templates_cache: tuple[list[str], np.ndarray] | None = field(default=None, repr=False)

    def recognize(self, y: np.ndarray, sample_rate: int, beat_times: Sequence[float]) -> list[Chord]:
        """Label the span of each beat, merging runs and dropping stubs."""
        import librosa

        beats = np.asarray(beat_times, dtype=float)
        if beats.size < 2:
            return []

        mono = y if y.ndim == 1 else np.mean(y, axis=0)
        chroma = self._chroma(mono, sample_rate)
        profiles, energies = self._beat_profiles(chroma, sample_rate, beats)

        norms = np.linalg.norm(profiles, axis=1, keepdims=True)
        normalized = profiles / np.maximum(norms, 1e-9)

        labels, templates = self._get_templates()
        scores = normalized @ templates.T

        # Silence is judged relative to this piece's own level: an absolute floor
        # misses long decay tails and reports phantom chords at the end of a track.
        floor = max(self.energy_floor, self.relative_energy_floor * float(np.median(energies)))
        scores[energies < floor, -1] = 1.0

        path = self._viterbi(scores)
        segments = self._segments(path, labels, scores, beats)
        if len(segments) > self.max_segments:
            segments = self._keep_longest(segments, self.max_segments)

        # Fragments shorter than min_beats are decay tails or transition noise.
        beat_interval = float(np.median(np.diff(beats))) if beats.size > 1 else 0.5
        threshold = self.min_beats * beat_interval
        return [s for s in segments if s.duration >= threshold]

    # ------------------------------------------------------------------ #
    def _chroma(self, mono: np.ndarray, sample_rate: int) -> np.ndarray:
        import librosa

        if self.chroma_backend == "stft":
            return librosa.feature.chroma_stft(y=mono, sr=sample_rate, hop_length=self.hop_length)
        return librosa.feature.chroma_cqt(y=mono, sr=sample_rate, hop_length=self.hop_length)

    def _beat_profiles(
        self, chroma: np.ndarray, sample_rate: int, beats: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Average chroma and energy between consecutive beats."""
        frames = audio_io.frames_for(beats, sample_rate, self.hop_length, chroma.shape[1])
        profiles = np.zeros((beats.size, chroma.shape[0]), dtype=float)
        energies = np.zeros(beats.size, dtype=float)
        for index in range(beats.size):
            start = int(frames[index])
            end = int(frames[index + 1]) if index + 1 < beats.size else chroma.shape[1]
            end = max(end, start + 1)
            window = chroma[:, start:end]
            if window.size == 0:
                continue
            profiles[index] = window.mean(axis=1)
            energies[index] = float(np.mean(window))
        return profiles, energies

    def _get_templates(self) -> tuple[list[str], np.ndarray]:
        if self._templates_cache is None:
            self._templates_cache = _templates()
        return self._templates_cache

    def _viterbi(self, scores: np.ndarray) -> np.ndarray:
        """Most likely chord sequence with a fixed penalty per change."""
        n_beats, n_labels = scores.shape
        cost = np.full((n_beats, n_labels), -np.inf)
        back = np.zeros((n_beats, n_labels), dtype=int)
        cost[0] = scores[0]

        stay = np.eye(n_labels)
        switch = np.where(stay > 0, 0.0, -self.switch_penalty)
        for t in range(1, n_beats):
            candidate = cost[t - 1][None, :] + switch
            back[t] = np.argmax(candidate, axis=1)
            cost[t] = candidate[np.arange(n_labels), back[t]] + scores[t]

        path = np.zeros(n_beats, dtype=int)
        path[-1] = int(np.argmax(cost[-1]))
        for t in range(n_beats - 2, -1, -1):
            path[t] = back[t + 1, path[t + 1]]
        return path

    def _segments(
        self,
        path: np.ndarray,
        labels: Sequence[str],
        scores: np.ndarray,
        beats: np.ndarray,
    ) -> list[Chord]:
        segments: list[Chord] = []
        start = 0
        for index in range(1, len(path) + 1):
            if index == len(path) or path[index] != path[start]:
                label = labels[path[start]]
                if label != NO_CHORD:
                    confidence = float(np.mean([scores[i, path[i]] for i in range(start, index)]))
                    segments.append(
                        Chord(
                            start=float(beats[start]),
                            end=float(beats[index]) if index < len(beats) else float(beats[-1]),
                            label=format_label(label),
                            confidence=confidence,
                        )
                    )
                start = index
        return segments

    @staticmethod
    def _keep_longest(segments: list[Chord], keep: int) -> list[Chord]:
        ordered = sorted(segments, key=lambda s: s.duration, reverse=True)[:keep]
        return sorted(ordered, key=lambda s: s.start)
