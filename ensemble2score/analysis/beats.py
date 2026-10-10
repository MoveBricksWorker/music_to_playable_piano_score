"""Beat and downbeat tracking.

Two backends:

``beat-this``
    A transformer beat/downbeat tracker (MIT).  The only backend here that gives
    real *downbeats*, which makes :mod:`ensemble2score.analysis.meter` materially
    better because it unlocks the strongest cue.
``librosa``
    Onset-strength + dynamic programming.  No downbeats, slightly less accurate,
    but present in the core dependency set.

Tempo octave errors
-------------------
Half and double tempo are the classic failure of every beat tracker, and they are
not symmetric in cost: reporting 75 BPM for a 150 BPM track silently doubles the
number of bars.  :func:`choose_tempo_octave` therefore compares the onset
autocorrelation at the candidate and its double, and prefers the faster reading
unless the slower one is clearly stronger.  Any remaining doubt is surfaced as a
warning telling the user to override with ``--tempo``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..core.errors import BackendError, BackendUnavailable
from ..core.models import BeatTrack
from ..io import audio as audio_io
from ..io.console import hint, warn
from .meter import tempo_from_beats

__all__ = [
    "BeatTracker",
    "build_beat_tracker",
    "LibrosaBeatTracker",
    "BeatThisTracker",
    "tempo_candidates",
    "choose_tempo_octave",
]

ANALYSIS_SR = 22050
HOP_LENGTH = 512
MIN_BPM = 40.0
MAX_BPM = 240.0

# beat-this is trained at this rate and its post-processor assumes frames at
# ``fps = 50``, i.e. exactly this sample rate.  Anything else corrupts every
# timestamp it returns; see ``BeatThisTracker.track``.
BEAT_THIS_SR = 22050

# The slowest tempo we are willing to conclude from a halving.  Autocorrelation
# cannot separate a beat from half a beat on its own, so all we do is refuse to
# halve below a tempo that would be implausible as a pulse.
#
# Note what is deliberately *absent* here: a rule that doubles a slow reading
# because "music is usually faster".  That was tried and it is wrong.  A slow song
# measured at 71.8 BPM scores *lower* autocorrelation at its own lag than at double
# it (0.408 vs 0.442), because the lag at twice the period inherits the energy of
# the fundamental's harmonics — so the acoustic evidence actively favours the wrong
# answer, and a plausibility prior then "confirms" it.  Trusting the measurement and
# letting the user override with `--tempo` is strictly better than guessing.
MIN_PLAUSIBLE_HALF_BPM = 70.0


class BeatTracker(Protocol):
    name: str

    def track(self, audio_path: str) -> BeatTrack: ...


# --------------------------------------------------------------------------- #
# tempo octave resolution
# --------------------------------------------------------------------------- #
def tempo_candidates(bpm: float) -> tuple[float, ...]:
    """The BPM and its half/double cousins, filtered to a sane range."""
    values = {bpm, bpm * 2.0, bpm / 2.0}
    return tuple(sorted(v for v in values if MIN_BPM <= v <= MAX_BPM))


def choose_tempo_octave(
    bpm: float,
    onset_envelope: np.ndarray,
    *,
    sample_rate: int,
    hop_length: int = HOP_LENGTH,
) -> tuple[float, list[str]]:
    """Resolve a half/double tempo ambiguity using onset autocorrelation.

    Returns ``(bpm, warnings)``.

    The comparison is always against ``bpm / 2``, because the failure that actually
    happens is *doubling*: a track whose kick lands on 1 and 3 and whose snare lands
    on 2 and 4 has strong periodicity at every beat, and a peak-picker reads that as
    twice the tempo.  Halving is therefore the correction worth looking for.  When
    the two readings are close the warning says so, because a silent wrong choice is
    worse than a noisy right one.
    """
    warnings: list[str] = []
    if onset_envelope.size < 4:
        return bpm, warnings

    envelope = np.asarray(onset_envelope, dtype=float)
    envelope = envelope - envelope.mean()
    if not np.any(envelope):
        return bpm, warnings

    autocorrelation = np.correlate(envelope, envelope, mode="full")[envelope.size - 1:]
    if autocorrelation.size == 0 or autocorrelation[0] <= 0:
        return bpm, warnings

    def strength(candidate_bpm: float) -> float:
        lag = int(round(60.0 * sample_rate / (candidate_bpm * hop_length)))
        if lag < 1 or lag >= autocorrelation.size:
            return 0.0
        return float(autocorrelation[lag]) / float(autocorrelation[0])

    # --- direction 2: the reading may be double the real pulse ---
    half = bpm / 2.0
    if half < max(MIN_BPM, MIN_PLAUSIBLE_HALF_BPM):
        return bpm, warnings  # halving would land somewhere implausible

    fast_strength = strength(bpm)
    slow_strength = strength(half)

    # Halving is only safe when the faster reading is implausibly fast for a tapped
    # pulse.  Below that threshold there is no reliable automatic test, and it is
    # worth being explicit about why rather than shipping another heuristic:
    #
    #   track     raw BPM   autocorr at raw / at half   correct
    #   blessing   143.6     0.442 / 0.406              half (71.8)
    #   collapse   152.0     0.364 / 0.266              raw  (152)
    #   victim     152.0     0.477 / 0.431              raw  (152)
    #
    # blessing and victim are nearly identical by that ratio (0.92 vs 0.91) yet
    # need opposite answers, and a grid-fit test scored every candidate within 5% of
    # every other.  So the measurement is kept and the ambiguity is *reported*,
    # because a user who knows the song can settle it in one flag and this code
    # cannot.  When the reading is fast enough that its beat is unlikely to be the
    # pulse, halving is the known failure mode and we take it.
    ambiguous = slow_strength >= 0.85 * fast_strength
    if bpm > 180.0 and slow_strength >= fast_strength:
        warnings.append(
            f"{bpm:.1f} BPM is fast enough to be double the pulse "
            f"(autocorrelation {slow_strength:.2f} at {half:.1f} vs {fast_strength:.2f} "
            f"at {bpm:.1f}): using {half:.1f}. Override with --tempo to force the faster reading"
        )
        return half, warnings

    if ambiguous:
        warnings.append(
            f"tempo may be double the pulse: autocorrelation is nearly equal at "
            f"{half:.1f} and {bpm:.1f} BPM ({slow_strength:.2f} vs {fast_strength:.2f}), so this "
            f"cannot be decided from the audio alone. Keeping {bpm:.1f}; if the bar count looks "
            f"halved or doubled, pass --tempo {half:.0f} or --tempo {bpm:.0f}"
        )
    return bpm, warnings


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
@dataclass
class LibrosaBeatTracker:
    """Onset strength + dynamic-programming beats.  No downbeats."""

    name: str = "librosa"
    start_bpm: float = 120.0

    def track(self, audio_path: str) -> BeatTrack:
        import librosa

        y, sr = audio_io.load(audio_path, mono=True, sample_rate=ANALYSIS_SR)
        onset_envelope = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)

        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_envelope,
            sr=sr,
            hop_length=HOP_LENGTH,
            start_bpm=self.start_bpm,
            trim=False,
        )
        bpm = float(np.atleast_1d(tempo)[0])
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)

        warnings: list[str] = []
        if beat_times.size >= 4:
            bpm, tempo_warnings = choose_tempo_octave(
                bpm, onset_envelope, sample_rate=sr, hop_length=HOP_LENGTH
            )
            warnings.extend(tempo_warnings)
            if abs(bpm - float(np.atleast_1d(tempo)[0])) > 1e-6:
                # Re-derive the grid at the corrected tempo so beats, tempo and
                # quantisation agree instead of fighting each other.
                tempo, beat_frames = librosa.beat.beat_track(
                    onset_envelope=onset_envelope,
                    sr=sr,
                    hop_length=HOP_LENGTH,
                    bpm=bpm,
                    trim=False,
                )
                beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)

        if beat_times.size < 4:
            raise BackendError(
                "beat tracking found fewer than 4 beats: the track may be too short, "
                "too quiet, or have no discernible pulse"
            )

        # Say it plainly.  This backend cannot produce downbeats at all, which
        # disables the strongest cue in meter inference, and for a long time that
        # was invisible: every run in `scores6/` reports `downbeat: 0.00` for every
        # hypothesis and nothing said why.
        warnings.append(
            "no downbeat tracker is active (librosa gives beats only), so the "
            "downbeat cue is unavailable to meter inference; install the "
            "'beats' extra so beat-this can supply downbeats"
        )

        measured_bpm, stability = tempo_from_beats(beat_times)
        return BeatTrack(
            tempo_bpm=measured_bpm or bpm,
            beat_times=[float(t) for t in beat_times],
            downbeat_times=[],
            backend=self.name,
            tempo_stability=stability,
            warnings=warnings,
        )


@dataclass
class BeatThisTracker:
    """Beat This! (MIT) — beats *and* downbeats."""

    name: str = "beat-this"
    device: str = "auto"
    checkpoint: str | None = None

    # The model is trained at 22050 Hz and its post-processor works in frames at
    # ``fps=50``, which is that sample rate.  Feeding it anything else silently
    # corrupts every timestamp.
    model_sample_rate: int = BEAT_THIS_SR

    def track(self, audio_path: str) -> BeatTrack:
        try:
            import torch
            from beat_this.inference import Audio2Beats
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailable(
                f"beat-this is not usable ({type(exc).__name__}: {exc}); "
                'install it with: pip install "ensemble2score[beats]"'
            ) from exc

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Resample to the model's own rate instead of handing it a file.
        #
        # ``File2Beats`` decodes at the file's native rate and passes it straight
        # through, and the released post-processor then reads the frame axis as
        # ``fps=50``.  On a 44.1 kHz file that is wrong twice over — the peaks are
        # detected on a grid at half the intended resolution and every returned
        # timestamp is scaled by 22050/44100.  Measured on a real 44.1 kHz track:
        # 102 beats at a reported 375 BPM with downbeats 1.25 beats apart, against
        # 570 beats at 142.9 BPM with downbeats three beats apart once the audio is
        # resampled.  ``load_audio`` is the one call that knows the API, so a future
        # rename stays a one-line change.
        signal, sample_rate = audio_io.load(
            audio_path, mono=True, sample_rate=self.model_sample_rate
        )
        mono = np.asarray(signal, dtype=np.float64).reshape(-1, 1)
        if mono.shape[0] < sample_rate:
            raise BackendError(
                "beat-this needs at least one second of audio; the file is shorter"
            )

        # ``checkpoint_path`` is only passed when the user named one.  beat_this's
        # ``load_model`` treats ``None`` as "build the architecture and load
        # nothing", so forwarding our default of ``None`` handed back an
        # **untrained** model that returned 102 beats at a reported 375 BPM for a
        # track whose real reading is 570 beats at 142.9.  Omitting the argument
        # lets the library use its own default checkpoint name.
        if self.checkpoint:
            tracker = Audio2Beats(checkpoint_path=self.checkpoint, device=device)
        else:
            tracker = Audio2Beats(device=device)
        beats, downbeats = tracker(mono, sample_rate)

        beat_times = np.asarray(beats, dtype=float)
        downbeat_times = np.asarray(downbeats, dtype=float)
        if beat_times.size < 4:
            raise BackendError(
                "beat-this returned fewer than 4 beats: the track may be too short, too "
                "quiet, or have no discernible pulse"
            )

        bpm, stability = tempo_from_beats(beat_times)
        warnings: list[str] = []
        if downbeat_times.size == 0:
            warnings.append("the tracker returned no downbeats: meter inference loses its best cue")

        return BeatTrack(
            tempo_bpm=bpm,
            beat_times=[float(t) for t in beat_times],
            downbeat_times=[float(t) for t in downbeat_times],
            backend=self.name,
            tempo_stability=stability,
            warnings=warnings,
        )


def build_beat_tracker(name: str, *, device: str = "auto", checkpoint: str | None = None) -> BeatTracker:
    """Instantiate a beat tracker.

    ``auto`` uses beat-this when it is importable and its checkpoint is actually
    fetchable, and otherwise falls back to librosa with a clear hint.  The
    checkpoint check exists because the public mirror has been known to answer
    with an HTML error page, which would otherwise surface as a confusing crash
    deep inside the loader.
    """
    if name == "librosa":
        return LibrosaBeatTracker()
    if name == "beat-this":
        return BeatThisTracker(device=device, checkpoint=checkpoint)
    if name == "auto":
        if not _beat_this_available():
            hint("beat-this is not usable here: using librosa beats (no downbeats)")
            return LibrosaBeatTracker()

        from ..net import checkpoint_fetchable

        # Only probe an explicit checkpoint; probing a default URL we cannot verify
        # makes `auto` fail on networks that block range requests.
        if checkpoint:
            ok, reason = checkpoint_fetchable(checkpoint)
            if not ok:
                warn(f"beat-this checkpoint is not usable ({reason}): using librosa beats")
                return LibrosaBeatTracker()
        return BeatThisTracker(device=device, checkpoint=checkpoint)
    raise BackendError(f"unknown beat tracker {name!r}")


def _beat_this_available() -> bool:
    """Is beat-this installed *and* does it still expose the API we call?

    Checking only that the package imports is not enough.  ``beat_this`` 1.1
    replaced ``Audio2Beat`` with ``File2Beats``, and an import-only check (or one
    naming the old class) sent ``auto`` down a path that could never run: every
    run fell back to librosa, ``downbeat_times`` stayed empty, and meter inference
    lost its strongest cue without ever saying so.  ``auto`` must never fail
    because an optional backend changed shape — but it must not *silently* lose a
    stage either, so this checks the class we actually instantiate.
    """
    try:
        from beat_this.inference import Audio2Beats  # noqa: F401
    except Exception:  # noqa: BLE001 - absent, broken, or renamed: all mean "no"
        return False
    return True
