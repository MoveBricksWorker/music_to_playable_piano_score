"""Source separation backends.

Four options, in descending quality:

``demucs``
    The real thing (Hybrid Transformer Demucs).  Downloads weights on first use.
``hpss``
    librosa harmonic/percussive split.  Not instrument-aware: the harmonic part
    holds everything pitched at once, so per-stem transcription is meaningless —
    useful only to prove the pipeline runs without any model.
``dir``
    Use stems you already have, or that a test fixture generated.
``none``
    Skip separation; rhythm-only runs.

A note on silence
-----------------
Separators emit near-silent bleed for instruments that are not in the song.  A
6-stem model asked about a guitar-only track still writes a ``guitar.wav``.  If
that bleed is transcribed, the pipeline invents an entire part that looks
plausible and is fabricated.  Every stem is therefore measured, and one that is
both very quiet in absolute terms and far below the loudest stem is treated as
absent — a missing part is always better than a fake one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from ..core.errors import BackendError, BackendUnavailable, InputError
from ..core.models import StemSet
from ..io import audio as audio_io
from ..io.console import hint, info, warn

__all__ = [
    "Separator",
    "build_separator",
    "is_empty_stem",
    "AudioStemSeparator",
    "HpssSeparator",
    "DirectorySeparator",
    "NullSeparator",
    "DEMUCS_STEMS",
]

# What each Demucs model can actually produce.  Verified against real output
# rather than the documentation: htdemucs_6s yields drums/bass/other/vocals/
# piano/guitar and, importantly, **no strings**.
DEMUCS_STEMS: dict[str, tuple[str, ...]] = {
    "htdemucs": ("drums", "bass", "other", "vocals"),
    "htdemucs_ft": ("drums", "bass", "other", "vocals"),
    "mdx_extra": ("drums", "bass", "other", "vocals"),
    "htdemucs_6s": ("drums", "bass", "other", "vocals", "piano", "guitar"),
}

_TARGET_SAMPLE_RATE = 44100


class Separator(Protocol):
    name: str

    def separate(self, audio_path: str, out_dir: Path) -> StemSet: ...


def is_empty_stem(
    level_db: float,
    loudest_db: float,
    *,
    absolute_floor_db: float = -40.0,
    relative_drop_db: float = 30.0,
) -> bool:
    """Is this stem effectively absent from the song?

    Both tests must agree before a stem is dropped: it has to be quiet in
    absolute terms *and* far below the loudest stem.  A quiet-but-real part in a
    quiet mix therefore survives, while -45 dBFS bleed next to a -10 dBFS drum bus
    does not.
    """
    if not np.isfinite(level_db):
        return True
    return level_db < absolute_floor_db and (loudest_db - level_db) > relative_drop_db


def _measure(stems: dict[str, Path]) -> dict[str, float]:
    levels: dict[str, float] = {}
    for name, path in stems.items():
        try:
            audio, _ = audio_io.load(path, mono=True)
        except Exception:  # noqa: BLE001 - an unreadable stem reads as absent
            levels[name] = float("-inf")
            continue
        levels[name] = audio_io.level_db(audio)
    return levels


def peak_of(path: Path) -> float:
    """Largest absolute sample in a file, or 0.0 when it cannot be read."""
    try:
        audio, _ = audio_io.load(path, mono=True)
    except Exception:  # noqa: BLE001 - an unreadable stem is handled elsewhere
        return 0.0
    return float(np.abs(np.asarray(audio)).max()) if np.size(audio) else 0.0


def _scale_factor(peaks: dict[str, float]) -> float:
    """One factor that brings a whole stem set inside full scale.

    Kept for stems that arrive as files, where per-sample analysis is not available.
    It is deliberately blunt: scaling everything by ``0.999 / peak`` costs a lot of
    level when the peak is a lone outlier (measured: −13.4 dB, because one drum
    sample reached 4.69), which is why freshly computed stems go through
    :func:`limit_artifacts` instead.
    """
    loudest = max(peaks.values(), default=0.0)
    if not np.isfinite(loudest) or loudest <= 1.0:
        return 1.0
    return 0.999 / loudest


def limit_artifacts(audio: np.ndarray, ceiling: float) -> tuple[np.ndarray, int]:
    """Soften samples that exceed what the source material could contain.

    A separation model regularises its input but does not bound its output, and it
    occasionally emits a **lone sample** far above full scale: measured on a real
    track, the four-stem ``htdemucs`` drum stem peaked at 4.686 while only 0.009% of
    its samples exceeded 1.0, and those samples were isolated single-frame spikes
    (0.29 → 4.41 → 0.61).  Real percussion cannot do that — the excursion within one
    sample would be an impossible slew rate — and nothing in a stem can be louder
    than the loudest sample of the mix it came from, which here was 0.874.  They do
    cluster near real drum onsets (1.5x the mix's RMS there), so they are artefacts
    *of* the transients rather than random noise.

    Writing such a stem as 16-bit PCM hard-clips them into flat-topped steps: 7851
    clipped samples on one track, audible as a crackling edge on percussion.
    Scaling the whole stem down to fit would cost 13 dB for the sake of 0.009% of
    samples, so the excess is compressed instead with a knee that is *continuous in
    value and slope* at ``ceiling`` and asymptotes strictly below full scale — no
    step, and therefore no click of its own.

    Returns ``(audio, n_limited)``.
    """
    values = np.asarray(audio, dtype=np.float64)
    if not np.isfinite(ceiling) or not 0.0 < ceiling < 1.0:
        return np.asarray(audio), 0

    magnitude = np.abs(values)
    mask = magnitude > ceiling
    changed = int(mask.sum())
    if not changed:
        return np.asarray(audio), 0

    # A knee of this width keeps the strongest spike inside full scale.  The
    # compression is ``r / (1 + r)``, which is 0 at the ceiling, tends to 1 as the
    # excess grows, and — the part that matters — has slope 1 there, so value *and*
    # slope are continuous at the knee and the limiter introduces no step of its own.
    # (An exponential form was tried first and rejected: it decays so fast that a
    # sample of 100 came out at 100.)
    #
    # Because the curve only *approaches* the ceiling, its output is clamped there:
    # with 16-bit PCM an asymptote within 3e-5 of the ceiling still rounds to 32767,
    # which reads back as full scale and is indistinguishable from a clipped sample.
    width = 1.0 - ceiling
    excess = magnitude[mask] - ceiling
    ratio = excess / width
    compressed = ratio / (1.0 + ratio)
    out = values.copy()
    out[mask] = np.sign(values[mask]) * np.minimum(ceiling + width * compressed, ceiling)
    return out, changed


def rescale_stems(written: dict[str, Path]) -> float:
    """Apply :func:`_scale_factor` to stem files that are already on disk.

    This is for stems that arrive as files (``--stems-dir``).  It cannot help a
    separator that wrote them itself: by the time a clipped file is read back its
    peaks are exactly 1.0, indistinguishable from a file that was always in range,
    so the scaling has to happen before the write.  Use :meth:`_write_scaled` for
    that.
    """
    peaks = {name: peak_of(path) for name, path in written.items()}
    factor = _scale_factor(peaks)
    if factor == 1.0:
        return 1.0
    for name, path in written.items():
        audio, sr = audio_io.load(path, mono=False)
        audio_io.save(path, np.asarray(audio) * factor, sr)
    return factor


# A 16-bit sample at full scale is exactly 1.0; anything within a single
# quantisation step of it came from a clamped peak rather than a loud one.
_FULL_SCALE_EPS = 3.1e-5

# The limiter's ceiling is held this far below full scale.  Its transfer curve
# approaches 1.0 asymptotically, so without headroom the loudest samples round to
# 32767 on write — which reads back as exactly 1.0000, the same value a clipped
# sample has.  0.5% is inaudible and leaves the file genuinely un-clipped.
_SAFE_CEILING = 0.995


def condition_input(audio: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Bring an over-full-scale mix into the range the separator was trained on.

    A separation model is trained on material that peaks around 1.0, and its own
    pre-processing normalises by the signal's standard deviation.  Loud, clipped
    masters break both assumptions at once — and loudly.  Measured across the four
    tracks in this workspace (44.1 kHz, decoded peak / RMS / samples at full scale):

    ==========  ======  =======  ==========
    track       peak    RMS      at full scale
    ==========  ======  =======  ==========
    blessing    0.991   0.137    0
    Collapse    1.338   0.419    118465
    yoake       1.070   0.248    1554
    Victim      1.093   0.267    1694
    ==========  ======  =======  ==========

    The correlation is exact: the only track whose separator output was clean is the
    only track that was not clipped.  On Collapse the model returned peaks of **24**
    (drums) and **11** (other) wherever the input was clipped, and normal levels
    everywhere it was not — the flattened plateaus of a clipped waveform, divided by
    a standard deviation that is too large for them, are an input the model has never
    seen.

    So the mix is conditioned rather than the output being patched: the plateaus are
    rounded with the same knee used for output artefacts, then the *original* gain is
    restored so that the stems still sum back to the mix the user supplied.

    Returns ``(audio, scale, n_conditioned)``.
    """
    values = np.asarray(audio, dtype=np.float64)
    peak = float(np.abs(values).max()) if values.size else 0.0
    if not np.isfinite(peak) or peak <= 1.0:
        return np.asarray(audio), 1.0, 0

    flat = np.abs(values) >= 1.0 - _FULL_SCALE_EPS
    conditioned, changed = limit_artifacts(values, 0.999)
    scale = peak  # restore the input's own peak after conditioning
    return conditioned * scale, scale, int(changed)


@dataclass
class AudioStemSeparator:
    """Base for separators that write ``<out_dir>/<stem>.wav``."""

    name: str = "audio"
    sample_rate: int = _TARGET_SAMPLE_RATE
    # Trim every stem to this many seconds.  This belongs here rather than only on
    # the mix: `--max-seconds` used to trim just the audio used for beat tracking,
    # so transcription still read full-length stems and the flag had no effect on
    # the score at all — it produced eight bars of a twenty-second piece.
    max_seconds: float | None = None

    def _copy(self, source: Path, target: Path) -> Path:
        """Copy a stem into the output, honouring :attr:`max_seconds`."""
        audio, sr = audio_io.load(source, mono=False, duration=self.max_seconds)
        return audio_io.save(target, audio, sr)

    def _write_scaled(
        self,
        tensors: dict[str, np.ndarray],
        out_dir: Path,
        sample_rate: int,
        *,
        source_peak: float | None = None,
    ) -> tuple[dict[str, Path], float, int]:
        """Write a freshly computed stem set without clipping it.

        The treatment happens here, on the in-memory arrays, because a clipped file
        reads back with peaks of exactly 1.0 and is then indistinguishable from one
        that was always in range.

        Two regimes, both driven by the *source* peak rather than by the stems' own:

        * when a mix peak is known, samples above it are lone model artefacts (see
          :func:`limit_artifacts`) and are softened in place — the rest of the stem
          keeps every bit of its level;
        * otherwise (HPSS, which is a decomposition of the input and cannot exceed
          it) nothing needs doing at all.

        Returns ``(written, scale, limited_samples)``.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        limited = 0
        ceiling = _SAFE_CEILING
        if source_peak is not None and np.isfinite(source_peak) and source_peak > 0:
            # Nothing in a stem can be louder than the loudest sample of the mix it
            # came from, so that is the honest ceiling — but it is held below full
            # scale, because the limiter's transfer curve approaches 1.0 asymptotically
            # and 16-bit PCM rounds anything that close straight to 32767, which is
            # indistinguishable from clipping in the written file.
            ceiling = float(min(_SAFE_CEILING, max(source_peak, 1e-6)))

        for name, data in tensors.items():
            values = np.asarray(data, dtype=np.float64)
            if source_peak is not None:
                values, changed = limit_artifacts(values, ceiling)
                limited += changed
            audio_io.save(out_dir / f"{name}.wav", values, sample_rate)
            written[name] = out_dir / f"{name}.wav"
        return written, 1.0, limited

    def _finish(self, out_dir: Path, written: dict[str, Path]) -> StemSet:
        levels = _measure(written)
        return StemSet(
            directory=str(out_dir),
            stems={name: str(path.resolve()) for name, path in written.items()},
            backend=self.name,
            sample_rate=self.sample_rate,
            level_db=levels,
        )

    def _warn_if_limited(self, stem_set: StemSet, limited: int, ceiling: float) -> None:
        if limited:
            stem_set.warnings.append(
                f"{limited} sample(s) in the separator's output exceeded the loudest sample "
                f"of the input ({ceiling:.3f}) and were softened to fit: they are lone "
                "single-sample spikes of the kind a separation model emits, and writing them "
                "as 16-bit PCM would hard-clip them into audible crackle. Everything below "
                "the ceiling is untouched"
            )

    def _warn_if_scaled(self, stem_set: StemSet, factor: float) -> None:
        if factor < 1.0:
            stem_set.warnings.append(
                f"the separator's stems exceeded full scale; every stem was scaled by "
                f"{factor:.3f} ({20 * np.log10(factor):+.2f} dB) as it was written, because "
                "16-bit PCM would otherwise hard-clip them. One factor for the whole set "
                "keeps the balance between the instruments"
            )


@dataclass
class DemucsSeparator(AudioStemSeparator):
    """Hybrid Transformer Demucs via its Python API."""

    model: str = "htdemucs_6s"
    device: str = "auto"
    name: str = "demucs"

    def _load_model(self):
        """Resolve the Demucs model by name, falling back to a local checkpoint.

        The order here matters and used to be reversed.  Starting from local files
        means starting from files named after a URL hash, and the only thing the old
        lookup could check was the declared ``sources`` — which six checkpoints on a
        typical machine share.  ``--demucs-model htdemucs`` therefore loaded a
        fine-tuned member instead of ``htdemucs``, and the stems it produced peaked at
        4.1 with a −2.6 dB level error (see :mod:`ensemble2score.data.local_models`).
        Demucs already knows which name means which weights, so it is asked first;
        the local file is kept only for machines that cannot reach the network at all.
        """
        from ..data.local_models import CACHE_HINT, find_local_model
        from ..net import hf_unreachable_guard

        network_error: Exception | None = None
        try:
            from demucs.pretrained import get_model
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailable(
                f"demucs is not usable ({type(exc).__name__}: {exc}); "
                'install it with: pip install "ensemble2score[separation]"'
            ) from exc

        with hf_unreachable_guard():
            try:
                return get_model(self.model)
            except Exception as exc:  # noqa: BLE001 - fall through to the local cache
                network_error = exc

        expected = DEMUCS_STEMS.get(self.model)
        if expected:
            # Name passed through so the lookup can insist on the right signature
            # rather than accepting any checkpoint with the same source list.
            local = find_local_model(expected, name=self.model)
            if local is not None:
                hint(
                    f"could not fetch {self.model!r} from the network "
                    f"({type(network_error).__name__}): using a local checkpoint instead"
                )
                return local

        raise BackendUnavailable(
            f"could not load Demucs model {self.model!r} from the network or from a "
            f"local checkpoint ({type(network_error).__name__}: {network_error}).\n{CACHE_HINT}"
        )

    def separate(self, audio_path: str, out_dir: Path) -> StemSet:
        try:
            import torch
            from demucs.apply import apply_model
        except Exception as exc:  # noqa: BLE001 - torch and demucs raise many types
            raise BackendUnavailable(
                f"demucs is not usable ({type(exc).__name__}: {exc}); "
                'install it with: pip install "ensemble2score[separation]"'
            ) from exc

        out_dir.mkdir(parents=True, exist_ok=True)
        device = self._resolve_device(torch)

        model = self._load_model()
        model.to(device)
        model.eval()

        y, sr = audio_io.load(audio_path, mono=False, sample_rate=model.samplerate)
        raw_peak = float(np.abs(np.asarray(y, dtype=np.float64)).max())

        # Condition the *input* before the model sees it.  A clipped master makes the
        # model return peaks of 24 instead of 1: see :func:`condition_input`.
        conditioned, restore, conditioned_samples = condition_input(y)
        audio = torch.from_numpy(np.asarray(conditioned, dtype=np.float32))
        if audio.shape[0] == 1:
            audio = audio.repeat(model.audio_channels, 1)
        elif audio.shape[0] < model.audio_channels:
            audio = audio.repeat(model.audio_channels, 1)[: model.audio_channels]

        # Demucs expects (batch, channels, samples) and normalises internally.
        reference = audio.mean(0)
        mean = reference.mean()
        std = reference.std()
        normalized = (audio - mean) / (std + 1e-8)

        with torch.no_grad():
            sources = apply_model(
                model,
                normalized[None],
                device=device,
                split=True,
                overlap=0.25,
                progress=False,
            )[0]
        sources = sources * std + mean
        # Undo the input conditioning so the stems describe the mix the user actually
        # supplied, not the conditioned copy of it.
        if restore != 1.0:
            sources = sources / restore

        tensors = {
            str(name): sources[index].cpu().numpy()
            for index, name in enumerate(model.sources)
        }
        # The ceiling comes from the mix the model was given, not from the stems:
        # nothing in a stem can be louder than its own source.
        written, factor, limited = self._write_scaled(
            tensors, out_dir, int(model.samplerate), source_peak=raw_peak
        )

        warning = self._quality_warning(model.sources)
        stem_set = self._finish(out_dir, written)
        stem_set.sample_rate = int(model.samplerate)
        if conditioned_samples:
            stem_set.warnings.append(
                f"the input peaks at {raw_peak:.3f} with {conditioned_samples} sample(s) at "
                "full scale, which is a clipped master: the model was given a conditioned copy "
                "(peaks rounded, original gain restored) because a clipped input makes it "
                "return peaks far above full scale. If the stems sound rough, the source is "
                "clipped and there is no clean separation to be had from it"
            )
        self._warn_if_limited(stem_set, limited, min(0.999, raw_peak))
        self._warn_if_scaled(stem_set, factor)
        if warning:
            stem_set.warnings.append(warning)
        return stem_set

    def _resolve_device(self, torch) -> str:
        if self.device == "cpu":
            return "cpu"
        cuda = bool(torch.cuda.is_available())
        if self.device == "cuda" and not cuda:
            warn("CUDA requested but not available: falling back to the CPU")
            return "cpu"
        return "cuda" if cuda else "cpu"

    @staticmethod
    def _quality_warning(sources: list[str]) -> str:
        if "guitar" in sources or "piano" in sources:
            return (
                f"the 6-stem model reports {', '.join(sources)}; its piano and guitar "
                "stems are known to carry heavy bleed from other instruments"
            )
        return ""


@dataclass
class HpssSeparator(AudioStemSeparator):
    """Harmonic/percussive split.  No models, no downloads, no instrument awareness."""

    name: str = "hpss"

    def separate(self, audio_path: str, out_dir: Path) -> StemSet:
        import librosa

        out_dir.mkdir(parents=True, exist_ok=True)
        y, sr = audio_io.load(audio_path, mono=True, sample_rate=self.sample_rate)
        harmonic, percussive = librosa.effects.hpss(y)

        # HPSS is a decomposition, not a regression: the two parts sum to the input,
        # so neither can exceed what the input did.  It goes through the same writer
        # so that every separator obeys one contract, but no limiter is needed.
        written, factor, limited = self._write_scaled(
            {"other": harmonic, "drums": percussive}, out_dir, sr
        )

        stem_set = self._finish(out_dir, written)
        self._warn_if_limited(stem_set, limited, 0.999)
        self._warn_if_scaled(stem_set, factor)
        stem_set.warnings.append(
            "hpss separates harmonic from percussive content, not instruments: the "
            "'other' stem holds every pitched instrument at once, so per-part note "
            "counts are not meaningful"
        )
        return stem_set


@dataclass
class DirectorySeparator(AudioStemSeparator):
    """Use stems that already exist on disk."""

    directory: str = ""
    name: str = "dir"

    def separate(self, audio_path: str, out_dir: Path) -> StemSet:
        source = Path(self.directory)
        if not source.is_dir():
            raise InputError(f"stems directory not found: {source}")

        found: dict[str, Path] = {}
        for path in sorted(source.iterdir()):
            if path.suffix.lower() in {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3"}:
                found[path.stem] = path
        if not found:
            raise InputError(f"no audio stems found in {source}")

        info(f"using {len(found)} stems from {source}")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Copy into the output directory only when the stems live elsewhere, so
        # that `stems/` in the output is always a complete, self-contained set.
        written: dict[str, Path] = {}
        for name, path in found.items():
            target = out_dir / f"{name}.wav"
            if path.resolve() == target.resolve() and self.max_seconds is None:
                written[name] = path
                continue
            written[name] = self._copy(path, target)

        stem_set = self._finish(out_dir, written)
        if written:
            first = next(iter(written.values()))
            stem_set.sample_rate = audio_io.probe(first).sample_rate
        return stem_set


@dataclass
class NullSeparator(AudioStemSeparator):
    """No separation at all; for rhythm-only runs."""

    name: str = "none"

    def separate(self, audio_path: str, out_dir: Path) -> StemSet:
        return StemSet(
            directory=str(out_dir),
            stems={},
            backend=self.name,
            sample_rate=0,
            warnings=["no separation requested: only rhythm analysis is possible"],
        )


def build_separator(name: str, **kwargs) -> Separator:
    """Instantiate a separator by name.

    ``auto`` prefers Demucs and degrades to HPSS with a warning, because a run
    that produces a poor score beats a run that refuses to start.  Explicitly
    requesting a backend that is missing *is* an error.
    """
    max_seconds = kwargs.get("max_seconds")
    if name == "dir":
        return DirectorySeparator(
            directory=str(kwargs.get("stems_dir") or ""), max_seconds=max_seconds
        )
    if name == "none":
        return NullSeparator()
    if name == "hpss":
        return HpssSeparator(max_seconds=max_seconds)
    if name == "demucs":
        return DemucsSeparator(
            model=str(kwargs.get("model", "htdemucs_6s")),
            device=str(kwargs.get("device", "auto")),
            max_seconds=max_seconds,
        )
    if name == "auto":
        try:
            import demucs  # noqa: F401
        except Exception:  # noqa: BLE001
            warn("demucs is not installed: falling back to hpss (much lower quality)")
            return HpssSeparator(max_seconds=max_seconds)
        return DemucsSeparator(
            model=str(kwargs.get("model", "htdemucs_6s")),
            device=str(kwargs.get("device", "auto")),
            max_seconds=max_seconds,
        )
    raise BackendError(f"unknown separator {name!r}")
