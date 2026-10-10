"""Percussion onsets, classified into kick / snare / hi-hat.

This is onset classification, not drum transcription: we do not recover the
sticking, the velocity nuance, or the toms.  Three voices are enough to print a
usable percussion staff and, more importantly, to feed the meter module the
kick-versus-snare distinction it needs for bar phase.

The classifier is deliberately spectral and readable:

* **kick** — almost all energy below ~120 Hz, with a low centroid;
* **hi-hat** — almost all energy above ~6 kHz, short and noisy;
* **snare** — broadband with a strong 150-250 Hz body plus high-frequency noise.

Anything that fits none of these confidently is dropped rather than guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..core.models import DrumHit
from ..io import audio as audio_io

__all__ = [
    "transcribe_drums",
    "accent_from_hits",
    "PHASE_WEIGHTS",
    "classify_onset",
    "spectral_features",
    "onset_strength_bands",
    "pick_onsets",
    "cluster_onsets",
]

# Weighting used only for bar-phase selection: the kick is the most reliable
# indicator of beat 1, so a loud backbeat snare must not outvote it here.  Using
# the raw accent for phase is what shifts every barline one beat late in pop.
PHASE_WEIGHTS: Mapping[str, float] = {"kick": 1.0, "snare": 0.25, "hihat": 0.05}

# Weighting for the meter *periodicity* cue.  Hi-hats matter here, unlike for the
# phase, but they must not dominate: on a track with eighth-note hats there are
# twice as many hats as backbeats, and counting them equally flattens the accent
# series until the bar has no measurable periodicity at all.
METER_WEIGHTS: Mapping[str, float] = {"kick": 1.0, "snare": 1.0, "hihat": 0.25}

# Band edges used for classification.  The kick band is deliberately wide: a real
# kick drum is not a 60 Hz sine, it has a broadband attack and a pitch sweep, so
# demanding that most energy sit below 120 Hz rejects almost every real kick.
KICK_BAND = (0.0, 250.0)
BODY_BAND = (150.0, 500.0)
HAT_BAND = (5000.0, 16000.0)

# Centroid zones (Hz) for the three voices.  Measured on real material: a kick
# lands near 3.8 kHz, a snare near 5.2 kHz and a hi-hat near 7.0 kHz, so the
# boundaries sit between them rather than at textbook values.
CENTROID_KICK_MAX = 4500.0
CENTROID_SNARE_MAX = 6200.0

# A classification is only reported when its score clears this bar; below it the
# onset is genuinely ambiguous and dropping it beats mislabelling it.
MIN_SCORE = 0.30


def spectral_features(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    *,
    total: float,
) -> dict[str, float]:
    """Normalised band energies and the centroid, for one onset spectrum."""
    def band(low: float, high: float) -> float:
        mask = (freqs >= low) & (freqs < high)
        if not mask.any() or total <= 0:
            return 0.0
        return float(spectrum[mask].sum()) / total

    centroid = float(np.sum(freqs * spectrum) / total) if total > 0 else 0.0
    return {
        "low": band(*KICK_BAND),
        "body": band(*BODY_BAND),
        "high": band(*HAT_BAND),
        "centroid": centroid,
    }


def classify_onset(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    *,
    total: float,
) -> tuple[str, float] | None:
    """Classify one onset spectrum into ``(voice, confidence)`` or ``None``.

    The design separates *brightness* from *character*, which is what makes this
    work on real material:

    * the **centroid** decides how bright the hit is, and therefore which pair of
      voices is even plausible.  A kick has a low centroid even though its attack
      is broadband; a hi-hat has a high one.
    * the **band energies** then decide within that pair.  A snare is broadband
      (real body *and* real top end); a hi-hat is almost pure top end.

    Hard ``if low > 0.5`` style thresholds failed in both directions here — they
    rejected every real kick (whose attack is broadband) and then, once loosened,
    promoted every snare to a hi-hat because a snare also has plenty of top end.
    """
    if total <= 0:
        return None

    features = spectral_features(spectrum, freqs, total=total)
    low, body, high = features["low"], features["body"], features["high"]
    centroid = features["centroid"]

    if centroid < CENTROID_KICK_MAX:
        # Dark: either a kick or a snare.  A kick has far more sub-bass energy.
        # Measured kicks sit near low=0.43, so 0.55 keeps the score below 1.0 and
        # leaves a genuine snare its body-based chance.
        kick = min(low / 0.55, 1.0)
        snare = min(body / 0.08, 1.0) * (1.0 - min(low / 0.55, 1.0))
        scores = {"kick": kick, "snare": snare, "hihat": 0.0}
    elif centroid < CENTROID_SNARE_MAX:
        # Mid-bright: a snare (has body) or a dull hi-hat (mostly top end).
        snare = min(body / 0.06, 1.0)
        hihat = min(high / 0.80, 1.0) * (1.0 - min(body / 0.30, 1.0))
        scores = {"kick": 0.0, "snare": snare, "hihat": hihat}
    else:
        # Bright: a hi-hat unless there is a real body underneath it.
        hihat = min(high / 0.60, 1.0)
        snare = min(body / 0.08, 1.0) * (1.0 - min(high / 0.90, 1.0))
        scores = {"kick": 0.0, "snare": snare, "hihat": hihat}

    voice, score = max(scores.items(), key=lambda pair: pair[1])
    if score < MIN_SCORE:
        return None
    return voice, float(min(score, 1.0))


def onset_strength_bands(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    *,
    bands: Sequence[tuple[float, float]] = ((0.0, 250.0), (250.0, 4000.0), (4000.0, 16000.0)),
    weights: Sequence[float] = (2.0, 1.0, 1.0),
) -> np.ndarray:
    """Positive spectral flux per band, summed with per-band weights.

    Written by hand rather than using ``librosa.onset.onset_strength`` because that
    function missed kick attacks entirely on real material: a low-frequency thump
    contributes almost nothing to a broadband flux measure next to a hi-hat, so the
    kick simply does not appear in the envelope.  Weighting the low band higher is
    what makes the kick detectable, and the kick is what defines the bar.
    """
    envelope = np.zeros(spectrum.shape[1], dtype=float)
    for (low, high), weight in zip(bands, weights):
        mask = (freqs >= low) & (freqs < high)
        if not mask.any():
            continue
        magnitude = np.log1p(spectrum[mask].sum(axis=0))
        flux = np.diff(magnitude, prepend=magnitude[0])
        envelope += weight * np.maximum(flux, 0.0)
    return envelope


def pick_onsets(
    envelope: np.ndarray,
    *,
    sensitivity: float = 1.0,
) -> np.ndarray:
    """Adaptive local-maximum peak picking on an onset envelope.

    The threshold is a low fraction of the envelope's own range.  It has to be low:
    a kick drum's low-band flux is roughly five times weaker than a hi-hat's
    broadband flux, so a "reasonable" threshold detects every hi-hat and no kick at
    all — which is exactly the failure this replaced.
    """
    values = np.asarray(envelope, dtype=float)
    if values.size < 3:
        return np.zeros(0, dtype=int)

    floor = float(np.median(values))
    peak = float(values.max())
    if peak <= floor:
        return np.zeros(0, dtype=int)

    threshold = floor + (peak - floor) * 0.08 * sensitivity
    candidates = np.flatnonzero(
        (values[1:-1] >= values[:-2]) & (values[1:-1] > values[2:]) & (values[1:-1] > threshold)
    )
    return candidates + 1


def backtrack_onsets(
    envelope: np.ndarray,
    peaks: np.ndarray,
    *,
    return_peaks: bool = False,
):
    """Walk each peak back to where its rise began.

    A spectral-flux peak sits *after* the attack, by 150-280 ms on measured
    material.  Left uncorrected that lag is systematic, and it is enough to push a
    drum hit onto the wrong beat when quantising — the whole point of detecting it.

    Returns the corrected frames, or ``(corrected, peaks)`` when
    ``return_peaks=True``.  The peaks matter because the corrected frame sits in the
    silence *before* the attack, so reading the envelope there gives zero: timing
    comes from the corrected frame, but loudness must come from the peak.
    """
    if peaks.size == 0:
        return (peaks, peaks) if return_peaks else peaks

    values = np.asarray(envelope, dtype=float)
    corrected = np.empty_like(peaks)
    for index, peak in enumerate(peaks):
        frame = int(peak)
        # Step back while the envelope is still falling towards the peak; the
        # minimum of that run is the last frame before the attack.
        while frame > 0 and values[frame - 1] <= values[frame]:
            frame -= 1
        corrected[index] = frame
    return (corrected, peaks) if return_peaks else corrected


def cluster_onsets(frames: np.ndarray, *, minimum_gap: int = 2) -> np.ndarray:
    """Keep the strongest frame per cluster of nearby onsets.

    One drum hit can produce several adjacent peaks; without this the same hit is
    classified twice.
    """
    if frames.size == 0:
        return frames
    ordered = np.sort(frames)
    groups: list[list[int]] = [[int(ordered[0])]]
    for frame in ordered[1:]:
        if frame - groups[-1][-1] <= minimum_gap:
            groups[-1].append(int(frame))
        else:
            groups.append([int(frame)])
    return np.array([max(group) for group in groups], dtype=int)


def transcribe_drums(
    audio_path: str,
    *,
    sample_rate: int = 22050,
    hop_length: int = 512,
    n_fft: int = 2048,
    sensitivity: float = 1.0,
    max_hits: int = 20000,
) -> tuple[list[DrumHit], list[str]]:
    """Detect onsets on a drum stem and classify each one.

    Returns ``(hits, warnings)``.  Hits are sorted by time.

    Known limitation: when a kick and a hi-hat land on the *same* instant — which
    is what a straightforward rock pattern does on beat 1 — they are detected as one
    onset and classified out of the two.  The kick usually wins, which is the right
    way round musically (losing a hi-hat is cosmetic; losing the kick moves the bar
    line), but it does mean simultaneous hits are under-counted.  Detecting per
    frequency band and merging would fix it and is a natural next step.
    """
    import librosa

    y, sr = audio_io.load(audio_path, mono=True, sample_rate=sample_rate)
    if audio_io.is_silent(y):
        return [], ["the percussion stem is silent: no percussion part was produced"]

    spectrum = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    envelope = onset_strength_bands(spectrum, freqs)
    peaks = pick_onsets(envelope, sensitivity=sensitivity)
    # Backtrack before clustering so that a corrected onset cannot collide with a
    # later one, and so duplicates are judged on where the hits really begin.
    corrected, peak_frames = backtrack_onsets(envelope, peaks, return_peaks=True)
    onset_frames = cluster_onsets(corrected)
    if onset_frames.size == 0:
        return [], ["no percussion onsets were detected"]
    times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    # Loudness comes from the flux *peak* of the onset nearest each corrected
    # frame, never from the corrected frame itself: that one sits in the silence
    # before the attack, where the envelope is zero by construction.  Reading the
    # envelope there is what made every drum hit come out with velocity 1, which in
    # turn flattened the accent series the meter cue depends on.
    strengths = np.empty(onset_frames.size, dtype=float)
    for index, frame in enumerate(onset_frames):
        nearest = int(peak_frames[int(np.argmin(np.abs(peak_frames - frame)))])
        window = envelope[max(nearest - 2, 0): nearest + 1]
        strengths[index] = float(window.max()) if window.size else float(envelope[nearest])

    # Normalise velocity against a *typical strong hit*, not the maximum.  Using the
    # max collapses everything: a kick's weighted flux is far below a hi-hat's, so
    # `round(strength / max * 127)` rounds almost every hit down to 0 and the
    # clip-to-1 floor then makes **every** velocity identical.
    scale = float(np.percentile(strengths, 90)) if strengths.size else 1.0
    scale = max(scale, 1e-9)

    hits: list[DrumHit] = []
    skipped = 0
    for frame, time, strength in zip(onset_frames, times, strengths):
        # Classify from the attack frames: a kick's identifying low-frequency body
        # lives in the 20-40 ms after the hit, and averaging further out pulls in
        # whatever plays next (often a hi-hat on the same or the next eighth).
        window = spectrum[:, frame: min(frame + 3, spectrum.shape[1])]
        if window.size == 0:
            continue
        magnitude = window.mean(axis=1)
        result = classify_onset(magnitude, freqs, total=float(magnitude.sum()))
        if result is None:
            skipped += 1
            continue
        voice, confidence = result
        hits.append(
            DrumHit(
                voice=voice,
                time=float(time),
                velocity=int(np.clip(round(float(strength) / scale * 100.0), 1, 127)),
                confidence=confidence,
            )
        )
        if len(hits) >= max_hits:
            break

    warnings: list[str] = []
    if skipped:
        warnings.append(f"discarded {skipped} percussion onset(s) that matched no known voice")
    hits.sort(key=lambda h: h.time)
    return hits, warnings


def accent_from_hits(
    hits: Sequence[DrumHit],
    beat_times: Sequence[float],
    *,
    weights: Mapping[str, float] | None = None,
    tolerance: float = 0.06,
) -> np.ndarray:
    """Per-beat accent built from *classified* hits.

    Passing ``weights`` lets the caller decide which voice defines the bar: the
    default counts every voice, while :data:`PHASE_WEIGHTS` emphasises the kick
    and is what bar-phase selection should use.
    """
    beats = np.asarray(beat_times, dtype=float)
    accent = np.zeros(beats.size, dtype=float)
    if beats.size == 0 or not hits:
        return accent

    voice_weights = dict(weights) if weights is not None else {}

    # Half the distance to the neighbouring beats, so no hit is counted twice.
    if beats.size > 1:
        intervals = np.diff(beats)
        windows = np.empty(beats.size, dtype=float)
        windows[0] = intervals[0] / 2.0
        windows[-1] = intervals[-1] / 2.0
        if beats.size > 2:
            windows[1:-1] = np.minimum(intervals[:-1], intervals[1:]) / 2.0
    else:
        windows = np.array([tolerance], dtype=float)

    for hit in hits:
        index = int(np.argmin(np.abs(beats - hit.time)))
        if abs(beats[index] - hit.time) > max(windows[index], tolerance):
            continue
        weight = voice_weights.get(hit.voice, 1.0) if voice_weights else 1.0
        accent[index] += weight * (hit.velocity / 127.0)

    span = accent.max() - accent.min()
    if span > 1e-12:
        accent = (accent - accent.min()) / span
    return accent
