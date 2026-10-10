"""Time-signature inference — the original contribution of this project.

Why this module exists
----------------------
As of writing there is still no maintained open-source library that outputs a
time signature *from audio*.  madmom's downbeat tracker wants ``beats_per_bar``
as an input; librosa, madmom, Essentia and BeatNet expose no meter class; the
academic state of the art is a whole-track classifier over a few numerator
classes.  Commercial tools mostly leave the field to the user.

So the approach here is *evidence fusion with a transparent output* rather than a
black box.  Four cues are computed independently and combined into a ranked list
of hypotheses, each carrying the numbers that produced it, so a human can see why
the tool guessed 4/4 and override it with one flag.

The cues
--------
============  ======  =========================================================
cue           weight  what it measures
============  ======  =========================================================
downbeat        1.6   inter-downbeat spacing / beat spacing, near-integer and
                      consistent.  Only available with a real downbeat tracker.
accent          1.0   per-beat drum accent folded into bars of length ``p``:
                      how much the bar-initial beat stands out.
harmonic        0.5   the same fold on a chord-change indicator (changes tend
                      to land on bar starts).
subdivision   decides  triplet-vs-duple onset energy between beats; picks the
              the denom  denominator (simple vs compound), e.g. 6/8 vs 3/4.
============  ======  =========================================================

Two traps, both fixed here and pinned by tests
----------------------------------------------
1. **Bar phase is not the loudest beat.**  In pop the snare on 2 and 4 is often
   louder than the kick on 1, so choosing the bar start by maximum accent puts
   every barline one beat late and shifts the whole score.  Period detection and
   phase selection are therefore separate: period uses all drum accents, phase
   uses a kick-dominated series (optionally mixed with harmonic change), and if
   a downbeat tracker gave us downbeats the phase is taken from those directly.
2. **Harmonic and subdivision cues must come from pitched, full-band audio.**
   Computing chroma on the drum stem produces noise, which quietly poisons the
   strongest cue.  :mod:`ensemble2score.analysis.pipeline` is responsible for
   passing the right signal; the arguments here are named so that misuse is
   visible at the call site.

Known, irreducible limits (documented, not hidden)
--------------------------------------------------
* 3/4 vs 6/8 and 2/4 vs 4/4 are frequently acoustically identical; the
  subdivision cue only nudges the decision and ambiguity is reported as such.
* One time signature per piece.  Meter changes are not modelled.
* The output is always a ranked hypothesis list.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..core.models import MeterResult, TimeSignature

__all__ = [
    "DEFAULT_CANDIDATES",
    "ODD_CANDIDATES",
    "POP_PRIOR",
    "FLAT_PRIOR",
    "CUE_WEIGHTS",
    "MeterCues",
    "beat_synchronous_accent",
    "harmonic_change_per_beat",
    "subdivision_ratio",
    "infer_meter",
    "derive_downbeats",
    "tempo_from_beats",
    "periodicity",
    "downbeat_spacing_beats",
]

DEFAULT_CANDIDATES: tuple[int, ...] = (4, 2, 3, 6)
ODD_CANDIDATES: tuple[int, ...] = (5, 7)

# A small, explicit prior.  Popular music is overwhelmingly 4/4, so a mild bias
# towards 4 is honest engineering rather than a hidden assumption; pass
# prior="none" (FLAT_PRIOR) to switch it off.
POP_PRIOR: Mapping[int, float] = {4: 1.00, 3: 0.95, 6: 0.85, 2: 0.70, 5: 0.75, 7: 0.65, 9: 0.60, 12: 0.60}
FLAT_PRIOR: Mapping[int, float] = {p: 1.0 for p in (2, 3, 4, 5, 6, 7, 9, 12)}

CUE_WEIGHTS: Mapping[str, float] = {"downbeat": 1.6, "accent": 1.0, "harmonic": 0.5}

# How much credit the downbeat cue gives a bar length that is half or double the
# spacing the tracker reported.  Such a spacing is the signature of a doubled or
# halved beat grid, so it must not be worth nothing — but it must also not outrank
# an exact match, or a genuine 3-beat bar would lose to a 6/4 reading of it.
OCTAVE_MATCH_WEIGHT = 0.35

# Subdivision-ratio thresholds.  Above HIGH the feel is compound (6/8, 12/8);
# between LOW and HIGH the evidence is genuinely ambiguous and we say so.
TRIPLET_RATIO_HIGH = 1.25
TRIPLET_RATIO_LOW = 0.90

# Accent bands: low (kick), mid (snare), high (hi-hat, deliberately lighter).
ACCENT_BANDS: tuple[tuple[float, float], ...] = ((40.0, 120.0), (200.0, 2000.0), (3000.0, 10000.0))
ACCENT_BAND_WEIGHTS: tuple[float, ...] = (1.0, 1.0, 0.4)


# --------------------------------------------------------------------------- #
# cues computed from audio
# --------------------------------------------------------------------------- #
def beat_synchronous_accent(
    y: np.ndarray,
    sample_rate: int,
    beat_times: Sequence[float],
    *,
    window: float = 0.05,
    hop_length: int = 512,
    n_fft: int = 2048,
) -> np.ndarray:
    """Per-beat accent strength from a positive spectral-flux envelope.

    Kick and snare bands are weighted equally: the backbeat is the single most
    reliable meter cue in band music, and weighting the low band higher is what
    makes a naive implementation pick the wrong bar phase.

    Returns a min-max normalised array of length ``len(beat_times)``.
    """
    import librosa

    beats = np.asarray(beat_times, dtype=float)
    if beats.size == 0:
        return np.zeros(0, dtype=float)

    mono = y if y.ndim == 1 else np.mean(y, axis=0)
    spectrum = np.abs(librosa.stft(mono, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    frame_times = librosa.frames_to_time(
        np.arange(spectrum.shape[1]), sr=sample_rate, hop_length=hop_length
    )

    envelope = np.zeros(spectrum.shape[1], dtype=float)
    for (low, high), weight in zip(ACCENT_BANDS, ACCENT_BAND_WEIGHTS):
        band = (freqs >= low) & (freqs < high)
        if not band.any():
            continue
        magnitude = np.log1p(spectrum[band].sum(axis=0))
        flux = np.diff(magnitude, prepend=magnitude[0])
        envelope += weight * np.maximum(flux, 0.0)

    accent = np.zeros(beats.size, dtype=float)
    for index, time in enumerate(beats):
        selected = (frame_times >= time - window) & (frame_times <= time + window)
        if selected.any():
            accent[index] = float(envelope[selected].max())

    span = accent.max() - accent.min()
    if span > 1e-12:
        accent = (accent - accent.min()) / span
    return accent


def harmonic_change_per_beat(
    y: np.ndarray,
    sample_rate: int,
    beat_times: Sequence[float],
    *,
    threshold: float = 0.25,
    hop_length: int = 512,
    backend: str = "cqt",
) -> np.ndarray:
    """Per-beat indicator: 1.0 where the chroma changes noticeably.

    ``backend`` selects the chroma transform.  ``"cqt"`` tracks bass harmonics
    better; ``"stft"`` is markedly cheaper and avoids the numba-compiled CQT path
    entirely, which matters on a cold machine (see
    :mod:`ensemble2score.analysis.warmup`).
    """
    import librosa

    beats = np.asarray(beat_times, dtype=float)
    if beats.size < 2:
        return np.zeros(beats.size, dtype=float)

    mono = y if y.ndim == 1 else np.mean(y, axis=0)
    chroma = _chroma(mono, sample_rate, hop_length, backend)

    frame_times = librosa.frames_to_time(
        np.arange(chroma.shape[1]), sr=sample_rate, hop_length=hop_length
    )
    # Beat -> frame index, clipped to the available range.
    edges = np.searchsorted(frame_times, beats[1:])
    edges = np.clip(edges, 0, chroma.shape[1])
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [chroma.shape[1]]])

    profiles = np.zeros((beats.size, chroma.shape[0]), dtype=float)
    for index, (start, end) in enumerate(zip(starts, ends)):
        if end > start:
            profiles[index] = chroma[:, start:end].mean(axis=1)

    norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    profiles = profiles / np.maximum(norms, 1e-9)

    change = np.zeros(profiles.shape[0], dtype=float)
    distance = 1.0 - np.sum(profiles[1:] * profiles[:-1], axis=1)
    change[1:] = (distance > threshold).astype(float)
    return change


def _chroma(mono: np.ndarray, sample_rate: int, hop_length: int, backend: str) -> np.ndarray:
    import librosa

    if backend == "stft":
        return librosa.feature.chroma_stft(y=mono, sr=sample_rate, hop_length=hop_length)
    return librosa.feature.chroma_cqt(y=mono, sr=sample_rate, hop_length=hop_length)


def subdivision_ratio(
    y: np.ndarray,
    sample_rate: int,
    beat_times: Sequence[float],
    *,
    hop_length: int = 512,
) -> float:
    """Triplet-position onset energy divided by duple-position energy.

    ``> TRIPLET_RATIO_HIGH`` suggests a compound feel (6/8, 12/8);
    ``< TRIPLET_RATIO_LOW`` suggests simple subdivision; in between is reported
    as ambiguous instead of being forced one way.
    """
    import librosa

    beats = np.asarray(beat_times, dtype=float)
    if beats.size < 3:
        return 1.0

    mono = y if y.ndim == 1 else np.mean(y, axis=0)
    envelope = librosa.onset.onset_strength(y=mono, sr=sample_rate, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(envelope.size), sr=sample_rate, hop_length=hop_length)
    if envelope.max() > 0:
        envelope = envelope / envelope.max()

    intervals = np.diff(beats)
    median = float(np.median(intervals))
    if median <= 1e-6:
        return 1.0

    def sample(time: float) -> float:
        index = int(np.argmin(np.abs(times - time)))
        return float(envelope[max(index - 1, 0): index + 2].max())

    triplet: list[float] = []
    duple: list[float] = []
    for index, interval in enumerate(intervals):
        if not 0.8 * median <= interval <= 1.25 * median:
            continue  # tempo change or a missed beat: not comparable
        triplet.extend(sample(beats[index] + k * interval / 3.0) for k in (1, 2))
        duple.extend(sample(beats[index] + k * interval / 4.0) for k in (1, 2, 3))

    if not triplet or not duple:
        return 1.0
    return float(np.median(triplet)) / max(float(np.median(duple)), 1e-6)


# --------------------------------------------------------------------------- #
# pure inference (no audio, unit-testable)
# --------------------------------------------------------------------------- #
# A phase is only declared the bar start when its mean accent is this many standard
# errors clear of the runner-up.  Below it the phases are statistically tied and the
# answer is "no accent-based phase", not "whichever float came out largest".
_PHASE_SIGMA = 1.5


def _fold_variance_ratio(values: np.ndarray, period: int) -> float:
    """Fraction of the series' variance explained by folding it into ``period``.

    This is the statistic the whole cue rests on, and it replaced a raw
    inside/outside contrast that was quietly wrong in two different ways:

    * it **rewards sparse phases**.  On a 3/4 waltz, a period-4 fold that happens to
      select every snare beat scored *higher* than a period-3 fold selecting every
      downbeat, because the "inside" mean was consistent and the "outside" mean was
      diluted.  A variance ratio gives such a fold almost nothing: the four phase
      means are nearly identical, so the period explains no variance at all.
    * it is **biased towards periods that divide the true bar**.  A multiple of the
      real bar length folds consistently by construction.  Under a variance ratio a
      multiple still scores well when the bar really is periodic, but it no longer
      beats the true bar, and a non-divisor such as 3 on a 4/4 pattern collapses to
      near zero.
    """
    n = values.size
    groups = [values[phase::period] for phase in range(period) if values[phase::period].size]
    if len(groups) < 2:
        return 0.0

    total = float(((values - values.mean()) ** 2).sum())
    if total <= 1e-12:
        return 0.0

    grand = float(values.mean())
    between = sum(group.size * (float(group.mean()) - grand) ** 2 for group in groups)
    return float(np.clip(between / total, 0.0, 1.0))


def _fold_phase(values: np.ndarray, period: int) -> int | None:
    """The phase whose beats are loudest, or ``None`` when no phase stands out.

    Returning ``None`` for a tie is the point.  The old code returned
    ``argmax`` unconditionally, so on a 3/4 waltz *every* period-4 phase had a
    near-identical mean and one of them was "chosen" anyway — and that arbitrary
    phase then became the bar phase of the whole score.  A tie is real
    information: it means this period has no accent-defined bar start, and the
    caller should say so instead of inventing one.
    """
    n = values.size
    period = int(period)
    if period < 1:
        return None
    groups = [values[phase::period] for phase in range(period)]
    if any(group.size < 2 for group in groups):
        return None

    means = np.array([float(group.mean()) for group in groups], dtype=float)
    stderrs = np.array(
        [float(group.std(ddof=1)) / np.sqrt(group.size) for group in groups], dtype=float
    )
    winner = int(np.argmax(means))
    others = np.delete(np.arange(period), winner)
    if others.size == 0:
        return winner
    margin = means[winner] - float(means[others].max())
    spread = float(np.hypot(stderrs[winner], float(stderrs[others].max())))
    if spread <= 1e-12:
        # Identical phase means: no bar-start evidence whatsoever.
        return winner if margin > 1e-12 else None
    if margin < _PHASE_SIGMA * spread:
        return None
    return winner


def periodicity(series: np.ndarray, period: int) -> tuple[float, int]:
    """How much of a series' structure a ``period`` fold explains, and its phase.

    Returns ``(score, best_phase)`` with ``score`` in ``[0, 1]`` — the fraction of
    the series' variance explained by the fold (a standard eta-squared).  ``score``
    is the *period* detector: it answers "is there a bar of length p at all".

    ``best_phase`` prefers the loudest phase, which is the musically meaningful bar
    start; see :func:`_fold_phase`.  This is the deliberately simple fallback —
    callers that have a kick-weighted series pass it to
    :func:`derive_downbeats` instead, because on a backbeat the loudest phase is
    the snare.
    """
    values = np.asarray(series, dtype=float)
    n = values.size
    if n < 2 * period or period < 1:
        return 0.0, 0
    score = _fold_variance_ratio(values, period)
    phase = _fold_phase(values, period)
    return score, (0 if phase is None else int(phase))


@dataclass
class MeterCues:
    """The audio-derived cues, bundled so they can be reported and cached."""

    accent: np.ndarray | None = None
    harmonic_change: np.ndarray | None = None
    subdivision: float | None = None
    source: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "subdivision": None if self.subdivision is None else round(float(self.subdivision), 4),
            "accent": None if self.accent is None else [round(float(v), 4) for v in self.accent],
            "harmonic_change": (
                None if self.harmonic_change is None else [int(v) for v in self.harmonic_change]
            ),
        }


def infer_meter(
    beat_times: Sequence[float],
    *,
    accent: np.ndarray | None = None,
    downbeat_times: Sequence[float] | None = None,
    second_downbeats: Sequence[float] | None = None,
    harmonic_change: np.ndarray | None = None,
    phase_series: np.ndarray | None = None,
    subdivision: float | None = None,
    candidates: Sequence[int] = DEFAULT_CANDIDATES,
    allow_odd: bool = False,
    prior: Mapping[int, float] | None = None,
    weights: Mapping[str, float] | None = None,
    temperature: float = 0.12,
    max_hypotheses: int = 3,
) -> MeterResult:
    """Fuse every available cue into ranked time-signature hypotheses.

    Parameters
    ----------
    beat_times
        Detected beat positions in seconds, monotonic.
    accent
        Per-beat accent strength, same length as ``beat_times``.  Drives the
        *period* decision.
    downbeat_times
        Downbeats from a real tracker, when there is one.  This is the strongest
        cue: it turns bar length into a near-integer consistency test.
    second_downbeats
        Downbeats from a second, independent tracker.  Each bar length keeps the
        better of the two readings rather than their average, because the two
        sources disagree in useful ways: one may halve the bar (a 2/4 reading of a
        4/4 song is the usual failure) while the other gets it right, and averaging
        would split the difference into a wrong answer that looks confident.
    harmonic_change
        Per-beat chord-change indicator, same length as ``beat_times``.
    phase_series
        Per-beat series that decides *where* a bar starts, normally the
        kick-dominated accent mixed with harmonic change.  Deliberately separate
        from ``accent``: on a backbeat the loudest beat is the snare, so taking the
        phase from the period cue puts every barline one beat late.  When omitted,
        ``harmonic_change`` is used, then ``accent``.
    subdivision
        Triplet/duple energy ratio from :func:`subdivision_ratio`.
    candidates, allow_odd
        Numerator candidates.  ``4`` should normally stay in, and ``2`` is kept
        because 2/4 vs 4/4 is a genuine ambiguity.
    prior, weights, temperature
        Tuning knobs; the defaults are sane for band/pop material.
    """
    beats = np.asarray(beat_times, dtype=float)
    warnings: list[str] = []

    if beats.size < 4:
        warnings.append("fewer than 4 beats: meter inference is unreliable, assuming 4/4")
        return MeterResult(
            hypotheses=[TimeSignature(4, 4, 0.5, {"fallback": 1.0})],
            phase=0,
            bars=0,
            warnings=warnings,
        )

    candidate_list = sorted({int(c) for c in candidates if int(c) >= 2})
    if allow_odd:
        candidate_list = sorted(set(candidate_list) | set(ODD_CANDIDATES))

    prior_map = dict(POP_PRIOR if prior is None else prior)
    cue_weights = {**CUE_WEIGHTS, **(weights or {})}

    accent_series = _aligned(accent, beats.size, "accent", warnings)
    harmonic_series = _aligned(harmonic_change, beats.size, "harmonic_change", warnings)
    phase_ref = _aligned(phase_series, beats.size, "phase_series", warnings)
    if phase_ref is None:
        # Fall back to whatever is available, in order of how well it locates a
        # bar start.  Never to the raw accent if a harmonic indicator exists: on a
        # backbeat the raw accent points at the snare.
        phase_ref = harmonic_series if phase_ref is None else phase_ref

    beat_interval = float(np.median(np.diff(beats))) if beats.size > 1 else 0.5
    downbeat_scores = _downbeat_scores(downbeat_times, beat_interval, warnings)
    # Scored separately so the warning below can tell "no downbeats at all" from
    # "downbeats from one tracker only".
    second_warnings: list[str] = []
    second_downbeat_scores = (
        _downbeat_scores(second_downbeats, beat_interval, second_warnings)
        if second_downbeats is not None
        else {}
    )

    raw: dict[int, dict[str, float]] = {}
    for period in candidate_list:
        evidence: dict[str, float] = {}
        evidence["accent"] = (
            periodicity(accent_series, period)[0] if accent_series is not None else 0.0
        )
        evidence["harmonic"] = (
            periodicity(harmonic_series, period)[0] if harmonic_series is not None else 0.0
        )
        evidence["downbeat"] = downbeat_scores.get(period, 0.0)
        if second_downbeats is not None:
            # A second, independent downbeat tracker.  The two are combined by taking
            # whichever is more confident per bar length rather than by averaging:
            # min() would let one silent tracker veto the other, and a mean would let
            # a tracker that is confidently wrong drag a correct reading down.
            evidence["downbeat"] = max(
                evidence["downbeat"], second_downbeat_scores.get(period, 0.0)
            )
        raw[period] = evidence

    combined = {
        period: sum(cue_weights.get(name, 1.0) * value for name, value in evidence.items())
        * prior_map.get(period, 0.8)
        for period, evidence in raw.items()
    }

    if not combined or max(combined.values()) <= 1e-6:
        warnings.append("no usable meter cue (no drum accents, no downbeats): assuming 4/4")
        combined = {period: prior_map.get(period, 0.8) for period in candidate_list}

    top = max(combined, key=combined.get)
    if raw.get(top, {}).get("accent", 0.0) < 0.05 and downbeat_scores.get(top, 0.0) == 0.0:
        warnings.append(
            "meter evidence is weak (flat accent pattern): the top hypothesis is a "
            "low-confidence default — verify it and override with --meter if needed"
        )

    # The downbeat cue can also be *nearly* one-hot, so a wrong tracker produces a
    # confident wrong answer.  Measured on FictionJunction - blessing, a 4/4 track
    # whose beat grid came out doubled (143.6 BPM where the pulse is 71.8): the
    # tracker spaced downbeats 8 detected beats apart, the cue matched that exactly,
    # and the run reported 3/4 at p=0.43 with every cue looking self-consistent.
    # When the tracker claims a bar longer than any plausible numerator, the *grid*
    # is the suspect, not the music, and the fix is one flag.
    spacing = downbeat_spacing_beats(downbeat_times, beat_interval)
    if spacing > max(candidate_list):
        half = 60.0 / (2.0 * beat_interval)
        warnings.append(
            f"the downbeat tracker puts {spacing} detected beats in a bar, more than any "
            f"plausible time signature: the beat grid is probably doubled, which makes the "
            f"meter look certain and be wrong. Pass --tempo {half:.0f} to rebuild the grid "
            "with every second beat"
        )

    # Say so when the runner-up is close.  2/4 vs 4/4 rests on a prior, not on
    # acoustics, and a silent prior-decided answer is the kind of thing a user
    # should be told about rather than left to discover in the printed score.
    # This uses the full candidate set: the ranked list passed to the caller is
    # truncated, and comparing within it would name the wrong pair.
    ranked_all = sorted(combined, key=combined.get, reverse=True)
    if len(ranked_all) > 1:
        first, second = ranked_all[0], ranked_all[1]
        if combined[first] > 0 and combined[second] >= 0.9 * combined[first]:
            warnings.append(
                f"{first}/4 and {second}/4 are nearly tied on acoustic evidence "
                f"({combined[first]:.2f} vs {combined[second]:.2f}); the ranking here "
                "rests on the meter prior — override with --meter if you know the piece"
            )

    values = np.asarray(list(combined.values()), dtype=float)
    periods = list(combined)
    scaled = np.exp((values - values.max()) / max(temperature, 1e-6))
    probabilities = scaled / scaled.sum()

    order = np.argsort(-probabilities)[: max(1, max_hypotheses)]
    hypotheses: list[TimeSignature] = []
    for rank, index in enumerate(order):
        period = periods[int(index)]
        numerator, denominator = _time_signature(period, subdivision, rank)
        hypotheses.append(
            TimeSignature(
                numerator=numerator,
                denominator=denominator,
                confidence=float(probabilities[int(index)]),
                evidence={
                    **raw[period],
                    "prior": prior_map.get(period, 0.8),
                    "subdivision": float(subdivision) if subdivision is not None else 0.0,
                },
            )
        )

    if (note := _subdivision_note(subdivision)) is not None:
        warnings.append(note)

    winner_period = periods[int(order[0])]
    # Bar phase priority: a real tracker's downbeats beat any audio heuristic, then
    # the caller's phase reference.  It is deliberately *not* taken from the accent
    # cue's own phase: that is the loudest beat, which on a backbeat is the snare,
    # so using it put every barline one beat late.
    if downbeat_times is not None and len(downbeat_times) > 0:
        first = float(np.asarray(downbeat_times, dtype=float)[0])
        phase = int(np.argmin(np.abs(beats - first))) % max(int(winner_period), 1)
    elif phase_ref is not None:
        phase = _phase_from_series(phase_ref, winner_period)
    else:
        phase = 0
    return MeterResult(hypotheses=hypotheses, phase=int(phase), backend="heuristic", warnings=warnings)


def _aligned(series, expected: int, name: str, warnings: list[str]) -> np.ndarray | None:
    """Validate a per-beat series against the beat count, or disable it."""
    if series is None:
        return None
    values = np.asarray(series, dtype=float)
    if values.size != expected:
        warnings.append(f"{name} length does not match the beat count: cue disabled")
        return None
    return values


def _downbeat_scores(
    downbeat_times: Sequence[float] | None,
    beat_interval: float,
    warnings: list[str],
) -> dict[int, float]:
    """Fraction of inter-downbeat intervals consistent with each bar length."""
    if downbeat_times is None:
        return {}
    downbeats = np.asarray(downbeat_times, dtype=float)
    if downbeats.size < 3 or beat_interval <= 1e-9:
        return {}

    ratios = np.diff(downbeats) / beat_interval
    rounded = np.round(ratios).astype(int)
    rounded = rounded[rounded >= 1]
    if rounded.size == 0:
        return {}

    counts = Counter(rounded.tolist())
    observed, hits = counts.most_common(1)[0]
    if hits / rounded.size < 0.6:
        warnings.append(
            "inter-downbeat spacing is inconsistent (tempo drift or missed downbeats): "
            "the downbeat cue is down-weighted"
        )

    # A half or integer multiple of the observed spacing is weak but real evidence,
    # not zero evidence.  The cue used to be strictly one-hot, which made it a
    # confirmation trap: a doubled beat grid (the commonest tempo failure there is)
    # spaces downbeats 8 detected beats apart, which matches "8/4" exactly and every
    # plausible bar length not at all — so the tracker's own error came back as a
    # perfectly confident, wholly wrong answer.  Measured on blessing, a 4/4 track
    # at 71.8 BPM whose grid came out at 143.6.
    #
    # The weight is deliberately low.  It has to be enough to lift the *right*
    # reading above the tracker's own impossible one, and not enough to let a
    # 6-beat spacing outrank a genuine 3-beat one.
    scores = {int(period): float(np.mean(rounded == period)) for period in set(rounded.tolist())}
    for period in (2, 3, 4, 6, 9, 12):
        if period in scores:
            continue
        scores[period] = OCTAVE_MATCH_WEIGHT * float(
            max(np.mean(rounded == 2 * period), np.mean(rounded * 2 == period))
        )
    return scores


def downbeat_spacing_beats(
    downbeat_times: Sequence[float] | None,
    beat_interval: float,
) -> int:
    """Median inter-downbeat spacing in detected beats, or 0 when unusable."""
    if downbeat_times is None or beat_interval <= 1e-9:
        return 0
    downbeats = np.asarray(downbeat_times, dtype=float)
    if downbeats.size < 3:
        return 0
    ratios = np.diff(downbeats) / beat_interval
    ratios = ratios[ratios >= 1]
    if ratios.size == 0:
        return 0
    return int(round(float(np.median(ratios))))


def _time_signature(numerator: int, subdivision: float | None, rank: int) -> tuple[int, int]:
    """Choose a denominator; use a compound one only with positive triplet evidence."""
    compound = subdivision is not None and subdivision >= TRIPLET_RATIO_HIGH
    if numerator in (6, 9, 12) and compound:
        return numerator, 8
    if numerator == 3 and compound and rank == 0:
        # Three beats with strong triplet subdivision: report the compound
        # reading (6/8) explicitly and keep 3/4 available as another hypothesis.
        return 6, 8
    return numerator, 4


def _subdivision_note(subdivision: float | None) -> str | None:
    if subdivision is None:
        return None
    if TRIPLET_RATIO_LOW <= subdivision < TRIPLET_RATIO_HIGH:
        return (
            f"subdivision ratio {subdivision:.2f} is ambiguous between simple and compound "
            "meter; 3/4 vs 6/8 and 2/4 vs 4/4 cannot be decided from audio alone"
        )
    return None


def _phase_from_series(series: np.ndarray, numerator: int) -> int:
    """Bar phase from one per-beat series, preferring a defensible phase.

    ``_fold_phase`` refuses to name a bar start when the phases are statistically
    tied, which is the honest answer; the raw ``periodicity`` phase is the legacy
    fallback so that behaviour on degenerate input does not change.
    """
    decisive = _fold_phase(np.asarray(series, dtype=float), numerator)
    if decisive is not None:
        return int(decisive)
    return int(periodicity(series, numerator)[1])


def derive_downbeats(
    beat_times: Sequence[float],
    numerator: int,
    *,
    phase: int | None = None,
    phase_series: np.ndarray | None = None,
    accent: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Barlines implied by ``numerator``, and the phase chosen.

    ``phase_series`` decides *where* the bar starts.  Pass a kick-dominated series
    (optionally mixed with harmonic change): using raw accent puts every barline
    one beat late whenever the backbeat snare is louder than the downbeat kick,
    which is the common case in pop.  ``accent`` is only a last-resort fallback.
    """
    beats = np.asarray(beat_times, dtype=float)
    if beats.size == 0 or numerator < 1:
        return np.zeros(0, dtype=float), 0

    if phase is None:
        selector = phase_series if phase_series is not None else accent
        if selector is not None and np.asarray(selector).size == beats.size:
            phase = _phase_from_series(np.asarray(selector, dtype=float), int(numerator))
        else:
            phase = 0

    phase = int(phase) % int(numerator)
    return beats[phase:: int(numerator)], phase


def tempo_from_beats(beat_times: Sequence[float]) -> tuple[float, float]:
    """Return ``(bpm, stability)`` with stability in ``[0, 1]``.

    ``stability`` is a robust metronomic-ness measure: 1.0 for perfectly even
    beats, falling off with the median absolute deviation of the intervals.
    """
    beats = np.asarray(beat_times, dtype=float)
    if beats.size < 2:
        return 0.0, 0.0
    intervals = np.diff(beats)
    median = float(np.median(intervals))
    if median <= 1e-9:
        return 0.0, 0.0
    deviation = float(np.median(np.abs(intervals - median)))
    stability = float(np.clip(1.0 - (deviation / median) * 8.0, 0.0, 1.0))
    return 60.0 / median, stability
