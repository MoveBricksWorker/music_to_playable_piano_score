"""Score an estimate against a reference MIDI file.

Built on ``mir_eval``, which is the reference implementation of the standard
note-level metrics, so the numbers here are comparable with published work.

One caveat worth repeating wherever these numbers appear: **compare pitch content
with onset tolerance, and do not judge rhythm by the offset-aware F1.**  A
quantised transcript will never reproduce performance-level note durations, so
the note+offset F1 collapses even when every pitch and onset is right.  The
literature has the same gap — a strong piano model reports 96.7% note F1 and
82.5% with offsets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .core.errors import InputError
from .io.instruments import resolve

__all__ = ["EvalResult", "evaluate", "evaluate_by_instrument", "format_report"]


@dataclass
class EvalResult:
    """Precision / recall / F1 for one comparison."""

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    onset_precision: float = 0.0
    onset_recall: float = 0.0
    onset_f1: float = 0.0
    n_reference: int = 0
    n_estimate: int = 0
    onset_tolerance: float = 0.05
    offset_ratio: float | None = 0.2

    def to_dict(self) -> dict[str, object]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "onset_precision": round(self.onset_precision, 4),
            "onset_recall": round(self.onset_recall, 4),
            "onset_f1": round(self.onset_f1, 4),
            "n_reference": self.n_reference,
            "n_estimate": self.n_estimate,
            "onset_tolerance": self.onset_tolerance,
            "offset_ratio": self.offset_ratio,
        }


def _load_notes(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """MIDI -> ``(intervals, pitches)`` in mir_eval's format."""
    import pretty_midi

    target = Path(path)
    if not target.is_file():
        raise InputError(f"MIDI file not found: {target}")

    midi = pretty_midi.PrettyMIDI(str(target))
    rows: list[tuple[float, float, int]] = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            end = max(float(note.end), float(note.start) + 1e-3)
            rows.append((float(note.start), end, int(note.pitch)))

    if not rows:
        return np.zeros((0, 2)), np.zeros(0, dtype=int)

    rows.sort(key=lambda r: (r[0], r[2]))
    intervals = np.array([[r[0], r[1]] for r in rows], dtype=float)
    pitches = np.array([r[2] for r in rows], dtype=float)
    return intervals, pitches


def evaluate(
    estimate_path: str | Path,
    reference_path: str | Path,
    *,
    onset_tolerance: float = 0.05,
    offset_ratio: float | None = 0.2,
) -> EvalResult:
    """Compare two MIDI files note by note.

    ``offset_ratio=None`` ignores note endings, which is the right choice when the
    estimate is quantised.
    """
    import mir_eval

    ref_intervals, ref_pitches = _load_notes(reference_path)
    est_intervals, est_pitches = _load_notes(estimate_path)

    result = EvalResult(
        n_reference=int(ref_pitches.size),
        n_estimate=int(est_pitches.size),
        onset_tolerance=float(onset_tolerance),
        offset_ratio=offset_ratio,
    )
    if ref_pitches.size == 0 or est_pitches.size == 0:
        return result  # nothing to score; leave the zeros in place

    kwargs = {
        "onset_tolerance": float(onset_tolerance),
        "offset_ratio": offset_ratio,
        "offset_min_tolerance": 0.05,
    }
    precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches, est_intervals, est_pitches, **kwargs
    )
    onset_p, onset_r, onset_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches, est_intervals, est_pitches,
        onset_tolerance=float(onset_tolerance), offset_ratio=None,
    )

    result.precision, result.recall, result.f1 = float(precision), float(recall), float(f1)
    result.onset_precision = float(onset_p)
    result.onset_recall = float(onset_r)
    result.onset_f1 = float(onset_f1)
    return result


def evaluate_by_instrument(
    estimate_path: str | Path,
    reference_path: str | Path,
    *,
    onset_tolerance: float = 0.05,
    offset_ratio: float | None = None,
) -> dict[str, EvalResult]:
    """Score each MIDI program separately, labelled with our stem name for it."""
    import pretty_midi

    reference = pretty_midi.PrettyMIDI(str(reference_path))
    estimate = pretty_midi.PrettyMIDI(str(estimate_path))

    def bucket(midi) -> dict[int, list[tuple[float, float, int]]]:
        grouped: dict[int, list[tuple[float, float, int]]] = {}
        for instrument in midi.instruments:
            key = -1 if instrument.is_drum else int(instrument.program)
            for note in instrument.notes:
                grouped.setdefault(key, []).append(
                    (float(note.start), max(float(note.end), float(note.start) + 1e-3), int(note.pitch))
                )
        return grouped

    ref_groups = bucket(reference)
    est_groups = bucket(estimate)

    results: dict[str, EvalResult] = {}
    for program, ref_notes in ref_groups.items():
        label = _label_for(program)
        est_notes = est_groups.get(program, [])
        if not ref_notes:
            continue
        results[label] = _score_rows(ref_notes, est_notes, onset_tolerance, offset_ratio)
    return results


def _label_for(program: int) -> str:
    if program == -1:
        return "percussion"
    for stem in ("vocals", "other", "bass", "piano", "guitar", "strings", "keys"):
        spec = resolve(stem)
        if spec.midi_program == program:
            return stem
    return f"program-{program}"


def _score_rows(ref_notes, est_notes, onset_tolerance, offset_ratio) -> EvalResult:
    import mir_eval

    result = EvalResult(
        n_reference=len(ref_notes),
        n_estimate=len(est_notes),
        onset_tolerance=float(onset_tolerance),
        offset_ratio=offset_ratio,
    )
    if not ref_notes or not est_notes:
        return result

    ref_intervals = np.array([[r[0], r[1]] for r in ref_notes], dtype=float)
    ref_pitches = np.array([r[2] for r in ref_notes], dtype=float)
    est_intervals = np.array([[e[0], e[1]] for e in est_notes], dtype=float)
    est_pitches = np.array([e[2] for e in est_notes], dtype=float)

    precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches, est_intervals, est_pitches,
        onset_tolerance=float(onset_tolerance), offset_ratio=offset_ratio,
        offset_min_tolerance=0.05,
    )
    onset_p, onset_r, onset_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches, est_intervals, est_pitches,
        onset_tolerance=float(onset_tolerance), offset_ratio=None,
    )
    result.precision, result.recall, result.f1 = float(precision), float(recall), float(f1)
    result.onset_precision = float(onset_p)
    result.onset_recall = float(onset_r)
    result.onset_f1 = float(onset_f1)
    return result


def format_report(
    overall: EvalResult,
    per_instrument: dict[str, EvalResult] | None = None,
    *,
    estimate: str = "",
    reference: str = "",
) -> str:
    """Human-readable scoring summary."""
    lines = []
    if estimate and reference:
        lines.append(f"estimate : {estimate}")
        lines.append(f"reference: {reference}")
        lines.append("")
    lines.append(
        f"notes (onset+offset):  P {overall.precision:.3f}  R {overall.recall:.3f}  "
        f"F1 {overall.f1:.3f}   ({overall.n_estimate} vs {overall.n_reference} notes)"
    )
    lines.append(
        f"notes (onset only):    P {overall.onset_precision:.3f}  R {overall.onset_recall:.3f}  "
        f"F1 {overall.onset_f1:.3f}   (tolerance {overall.onset_tolerance:.2f}s)"
    )
    if per_instrument:
        lines.append("")
        lines.append("per instrument (onset+offset):")
        width = max(len(name) for name in per_instrument)
        for name, result in sorted(per_instrument.items()):
            lines.append(
                f"  {name:<{width}}  P {result.precision:.3f}  R {result.recall:.3f}  "
                f"F1 {result.f1:.3f}  (n={result.n_estimate}/{result.n_reference})"
            )
    return "\n".join(lines)
