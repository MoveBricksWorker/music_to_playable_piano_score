"""Audio input/output with one explicit channel convention.

**Internal arrays are always ``(channels, samples)``**, float32, in ``[-1, 1]``.
That matches ``librosa.load(mono=False)`` and keeps "which axis is time" from
being a per-call coin flip — the old code mixed both conventions and needed a
comment to explain it.

Decoding goes through ``soundfile`` when it can handle the format (fast, no
resampling) and only falls back to ``librosa``/``audioread`` for compressed
formats such as MP3 or M4A.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.errors import InputError
from ..core.models import as_array

__all__ = ["AudioInfo", "probe", "load", "save", "level_db", "peak_db", "is_silent", "to_mono"]

# Formats soundfile can decode natively.  Anything else goes through librosa.
_NATIVE_SUFFIXES = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".au", ".w64", ".caf"}


@dataclass(frozen=True)
class AudioInfo:
    path: str
    sample_rate: int
    channels: int
    frames: int
    duration: float
    subtype: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "frames": self.frames,
            "duration": round(self.duration, 3),
            "subtype": self.subtype,
        }


def _require_file(path: str | Path) -> Path:
    target = Path(path)
    if not target.is_file():
        raise InputError(f"audio file not found: {target}")
    if target.stat().st_size == 0:
        raise InputError(f"audio file is empty: {target}")
    return target


def probe(path: str | Path) -> AudioInfo:
    """Read header metadata without decoding the audio."""
    import soundfile as sf

    target = _require_file(path)
    try:
        info = sf.info(str(target))
    except Exception as exc:  # noqa: BLE001 - soundfile raises assorted types
        # Compressed formats are not always readable by libsndfile; let librosa
        # have a go before giving up.
        import librosa

        try:
            y, sr = librosa.load(str(target), sr=None, mono=False)
        except Exception as inner:  # noqa: BLE001
            raise InputError(f"cannot read audio {target.name}: {inner}") from exc
        frames = int(y.shape[-1])
        channels = 1 if y.ndim == 1 else int(y.shape[0])
        return AudioInfo(str(target), int(sr), channels, frames, frames / float(sr), "")

    return AudioInfo(
        path=str(target),
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        frames=int(info.frames),
        duration=float(info.frames) / float(info.samplerate or 1),
        subtype=str(info.subtype or ""),
    )


def load(
    path: str | Path,
    *,
    sample_rate: int | None = None,
    mono: bool = True,
    duration: float | None = None,
    offset: float = 0.0,
) -> tuple[np.ndarray, int]:
    """Load audio as float32 ``(channels, samples)`` (or ``(samples,)`` if mono).

    Returns ``(audio, sample_rate)``.
    """
    target = _require_file(path)
    wants_resample = False

    if target.suffix.lower() in _NATIVE_SUFFIXES:
        import soundfile as sf

        try:
            # soundfile can seek, so trim by frame index after decoding; this
            # keeps one code path for offset/duration instead of two.
            data, sr = sf.read(str(target), dtype="float32", always_2d=True)
            data = data.T  # (samples, channels) -> (channels, samples)
            if offset or duration:
                lo = int(offset * sr)
                hi = None if duration is None else lo + int(duration * sr)
                data = data[:, lo:hi]
            if sample_rate and sample_rate != sr:
                wants_resample = True
            y = data
            return _finish(y, sr, sample_rate, mono, wants_resample)
        except Exception:  # noqa: BLE001 - fall through to librosa
            pass

    import librosa

    y, sr = librosa.load(
        str(target),
        sr=sample_rate,
        mono=mono,
        duration=duration,
        offset=offset,
    )
    if mono:
        return np.asarray(y, dtype=np.float32), int(sr)
    y = np.atleast_2d(np.asarray(y, dtype=np.float32))
    return y, int(sr)


def _finish(
    data: np.ndarray, sr: int, sample_rate: int | None, mono: bool, resample: bool
) -> tuple[np.ndarray, int]:
    y = np.atleast_2d(np.asarray(data, dtype=np.float32))
    if resample and sample_rate:
        y, sr = _resample(y, sr, sample_rate)
    if mono:
        return to_mono(y), int(sr)
    return y, int(sr)


def _resample(y: np.ndarray, sr: int, target_sr: int) -> tuple[np.ndarray, int]:
    import librosa

    # librosa resamples along the last axis, which is our time axis.
    return np.asarray(librosa.resample(y, orig_sr=sr, target_sr=target_sr), dtype=np.float32), target_sr


def to_mono(y: np.ndarray) -> np.ndarray:
    """``(channels, samples)`` or ``(samples,)`` -> ``(samples,)`` float32."""
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        # soundfile-shaped input (samples, channels) is ambiguous; we only ever
        # accept our own convention, so require channels to be the small axis.
        if arr.shape[0] > arr.shape[1]:
            arr = arr.T
        return arr.mean(axis=0).astype(np.float32)
    raise ValueError(f"expected 1-D or 2-D audio, got shape {arr.shape}")


def save(path: str | Path, y: np.ndarray, sample_rate: int, *, subtype: str = "PCM_16") -> Path:
    """Write audio, accepting either channel convention.

    Stems are written as 16-bit PCM: they are intermediates for the transcriber,
    not archival masters, and 16-bit halves the size of a 4-stem split.
    """
    import soundfile as sf

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
        arr = arr.T  # (samples, channels) -> (channels, samples)
    data = arr.T if arr.ndim == 2 else arr  # soundfile wants (samples, channels)
    sf.write(str(target), np.ascontiguousarray(data), int(sample_rate), subtype=subtype)
    return target


def level_db(y: np.ndarray) -> float:
    """RMS level in dBFS (``-inf`` for digital silence)."""
    arr = np.asarray(y, dtype=np.float64)
    if arr.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(arr**2)))
    return 20.0 * float(np.log10(max(rms, 1e-12)))


def peak_db(y: np.ndarray) -> float:
    """Peak level in dBFS."""
    arr = np.asarray(y, dtype=np.float64)
    if arr.size == 0:
        return float("-inf")
    return 20.0 * float(np.log10(max(float(np.max(np.abs(arr))), 1e-12)))


def is_silent(y: np.ndarray, threshold_db: float = -60.0) -> bool:
    return peak_db(y) < threshold_db


def frames_for(times: np.ndarray, sr: int, hop_length: int, n_frames: int) -> np.ndarray:
    """Seconds -> frame indices, clipped into ``[0, n_frames]``.

    Computed arithmetically rather than by searching a materialised time array:
    that trick needs one entry per frame of the whole signal and is easy to get
    wrong on long tracks.
    """
    frames = np.round(as_array(times) * sr / hop_length).astype(int)
    return np.clip(frames, 0, n_frames)
