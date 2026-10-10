"""The pipeline: run every stage, cache intermediate results, collect warnings.

Ordering matters and is not arbitrary:

1. **Separation** first, because every later stage wants stems.
2. **Beats** next, because meter, quantisation and the MIDI grid all hang off the
   beat times.
3. **Meter cues need care.**  Accent comes from the drum stem, which is where
   rhythmic emphasis lives, but the harmonic and subdivision cues must come from
   pitched full-band audio.  Computing chroma on a drum stem yields noise, and in
   an earlier version that was enough to turn a 4/4 track into 3/4.
4. **Notation last**, once notes, drums and chords exist.

Each stage is cached under a key built from the content hash of its input, so
changing one option recomputes only what that option affects.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import __version__
from .analysis import drums as drum_stage
from .analysis import meter as meter_stage
from .analysis import transcribe as transcribe_stage
from .analysis.harmony import ChordRecognizer
from .analysis.quantize import build_timeline
from .analysis.separate import build_separator, is_empty_stem
from .core.config import RunConfig
from .core.errors import Ensemble2ScoreError, InputError
from .core.models import (
    DRUM_VOICES,
    BeatTrack,
    Chord,
    DrumHit,
    MeterResult,
    Note,
    Part,
    RunResult,
    StageResult,
    StemSet,
    TimeSignature,
)
from .io import audio as audio_io
from .io.cache import Cache, content_hash
from .io.console import hint, info, rule, slugify, warn
from .io.instruments import load_overrides, order_parts, resolve
from .net import hf_unreachable_guard

__all__ = ["Pipeline", "run", "StageOutcome"]

ANALYSIS_SR = 22050
PERCUSSIVE_ONLY = {"drums", "percussive"}
RHYTHM_SOURCES = ("drums", "percussive")

# Share of one part's notes that must land on an *identical* pitch in another part,
# at the same instant, before the two are called a split stem rather than doubling.
# Measured on real runs: a separator split shows 19-26% unison, ordinary octave
# doubling shows 9-13%, so the line sits between them.
_UNISON_ALARM = 0.18


@dataclass
class StageOutcome:
    result: RunResult
    output_dir: Path


@dataclass
class _Cues:
    """Audio-derived meter cues plus the signal they came from."""

    accent: np.ndarray
    harmonic_change: np.ndarray
    subdivision: float
    phase_series: np.ndarray
    source: str = ""


class Pipeline:
    """Runs one :class:`RunConfig` end to end."""

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.out_dir = Path(config.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cache = Cache(self.out_dir / "cache", enabled=config.cache)
        self.result = RunResult(input_path=config.input_path, duration=0.0)
        self.overrides = load_overrides(config.instrument_map)
        # Filled in by _load_prior() once the run starts; None means "no prior".
        self.prior = None
        # Resolved once: every cache key and every backend construction uses these,
        # so a key can never describe a different backend than the one that runs.
        self.backends = self._resolve_backend_names()
        self.device = self._resolve_device()

    # ------------------------------------------------------------------ #
    def run(self) -> StageOutcome:
        cfg = self.config
        source = Path(cfg.input_path)
        if not source.is_file():
            raise InputError(f"input audio not found: {source}")

        rule("input")
        probe = audio_io.probe(source)
        self.result.duration = probe.duration
        self.result.sample_rate = probe.sample_rate
        info(
            f"{source.name}: {probe.duration:.1f}s, {probe.sample_rate} Hz, "
            f"{probe.channels} channel(s)"
        )

        work_audio = str(source)
        if cfg.max_seconds and probe.duration > cfg.max_seconds:
            work_audio = str(self._trim(source, cfg.max_seconds))
            hint(f"trimmed to the first {cfg.max_seconds:.0f}s")

        stems = self._separate(work_audio)
        self.result.stems = stems

        # The symbolic prior, when one was named.  Loaded before the rhythm stages
        # because its downbeats are meter evidence and its chords are a chord source.
        self.prior = self._load_prior()

        if not stems.stems:
            message = "no stems were produced: only rhythm analysis was possible"
            warn(message)
            self.result.warn(message, stage="separation")
            self._write()
            return StageOutcome(self.result, self.out_dir)

        beats = self._beats(work_audio)
        self.result.beats = beats
        if cfg.tempo is not None:
            _apply_tempo_override(beats, float(cfg.tempo))

        cues = self._cues(stems, work_audio, beats)
        self.result.meter = self._meter(beats, cues)

        self._transcribe(stems)
        self.result.drum_hits = self._percussion(stems)
        self.result.chords = self._chords(stems, work_audio, beats)

        self._notate(beats)
        self._write()
        return StageOutcome(self.result, self.out_dir)

    # ------------------------------------------------------------------ #
    # stages
    # ------------------------------------------------------------------ #
    def _trim(self, source: Path, seconds: float) -> Path:
        target = self.out_dir / "cache" / f"trimmed_{int(seconds)}s.wav"
        if target.exists():
            return target
        y, sr = audio_io.load(source, mono=False, duration=seconds)
        return audio_io.save(target, y, sr)

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def _load_prior(self):
        """Load the SheetSage2 prior, or ``None`` when none was asked for.

        A prior that cannot be read produces a warning and no prior rather than a
        failure: it is an improvement to the run, so losing it must not be worse
        than never having asked for it.
        """
        cfg = self.config
        if not cfg.ss2_prior:
            return None
        from .analysis.sheetsage2 import load_prior, looks_like_sheetsage2_output

        if not looks_like_sheetsage2_output(cfg.ss2_prior):
            message = (
                f"--ss2-prior {cfg.ss2_prior} does not look like a SheetSage2 output "
                "directory (no beat.lab / downbeat.lab / events.json): ignoring it"
            )
            warn(message)
            self.result.warn(message, stage="prior")
            return None

        prior = load_prior(cfg.ss2_prior)
        info(
            f"SheetSage2 prior: {len(prior.beat_times)} beats, "
            f"{len(prior.downbeat_times)} downbeats, key {prior.key or '?'}, "
            f"{len(prior.chords)} chord segments, {len(prior.structure)} section(s)"
        )
        for message in prior.warnings:
            warn(f"prior: {message}")
            self.result.warn(message, stage="prior")
        return prior if prior.ok else None

    def _resolve_backend_names(self) -> dict[str, str]:
        """Which concrete backend ``auto`` will pick, without importing anything heavy.

        The cache key must name the *backend*, not the request.  Keyed on the
        literal ``"auto"``, a run that fell back to librosa was reused verbatim
        after the optional backend was installed and became usable — the flag that
        was supposed to select it had no effect on an existing output directory.
        """
        cfg = self.config
        separator = cfg.separator
        if separator == "auto":
            from importlib.util import find_spec

            separator = "demucs" if find_spec("demucs") else "hpss"

        beats = cfg.beats
        if beats == "auto":
            from .analysis.beats import _beat_this_available

            beats = "beat-this" if _beat_this_available() else "librosa"

        transcriber = cfg.transcriber
        if transcriber == "auto":
            from importlib.util import find_spec

            transcriber = "basic-pitch" if find_spec("basic_pitch") else "pyin"

        return {"separator": separator, "beats": beats, "transcriber": transcriber}

    def _resolve_device(self) -> str:
        if self.config.device != "auto":
            return self.config.device
        from importlib.util import find_spec

        if not find_spec("torch"):
            return "cpu"
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _measure(self, stage: str, started: float, *, cached: bool, backend: str = "") -> None:
        self.result.stages.append(
            StageResult(stage=stage, seconds=time.time() - started, cached=cached, backend=backend)
        )

    # ---- stage 1: separation ------------------------------------------ #
    def _separate(self, work_audio: str) -> StemSet:
        cfg = self.config
        key = self.cache.key(
            "separate",
            content_hash(work_audio),
            self.backends["separator"],
            cfg.stems_dir,
            cfg.demucs_model,
            # max_seconds changes the stems themselves, so it has to be part of the
            # key: without it a full run and a trimmed run share a cache entry.
            cfg.max_seconds,
        )
        if self.cache.has(key):
            info("separation: reusing the cached result")
            stems = StemSet.from_dict(self.cache.read(key))
            self._measure("separate", time.time(), cached=True, backend=stems.backend)
            self._report_levels(stems)
            return stems

        rule("stage 1 · source separation")
        started = time.time()
        with hf_unreachable_guard():
            separator = build_separator(
                self.backends["separator"],
                stems_dir=cfg.stems_dir,
                model=cfg.demucs_model,
                device=self.device,
                max_seconds=cfg.max_seconds,
            )
            stems = separator.separate(work_audio, self.out_dir / "stems")

        for name, path in stems.stems.items():
            info(f"stem {name}: {Path(path).name}")
        self._report_levels(stems)
        for message in stems.warnings:
            warn(message)
            self.result.warn(message, stage="separation")

        self.cache.write(key, stems.to_dict())
        self._measure("separate", started, cached=False, backend=stems.backend)
        return stems

    def _report_levels(self, stems: StemSet) -> None:
        if not stems.level_db:
            return
        loudest = max(stems.level_db.values())
        described = []
        for name, level in sorted(stems.level_db.items(), key=lambda kv: -kv[1]):
            empty = is_empty_stem(
                level,
                loudest,
                absolute_floor_db=self.config.min_stem_level_db,
                relative_drop_db=self.config.max_stem_drop_db,
            )
            described.append(f"{name} {level:.1f} dB" + (" (empty)" if empty else ""))
        info("stem levels: " + ", ".join(described))

    # ---- stage 2: beats ----------------------------------------------- #
    def _beats(self, work_audio: str) -> BeatTrack:
        from .analysis.beats import build_beat_tracker

        cfg = self.config
        key = self.cache.key("beats", content_hash(work_audio), self.backends["beats"])
        if self.cache.has(key):
            info("beats: reusing the cached result")
            track = BeatTrack.from_dict(self.cache.read(key))
            self._measure("beats", time.time(), cached=True, backend=track.backend)
            return track

        rule("stage 2 · tempo, beats, downbeats")
        started = time.time()
        tracker = build_beat_tracker(self.backends["beats"], device=self.device)
        track = tracker.track(work_audio)
        info(
            f"tempo {track.tempo_bpm:.1f} BPM ({track.backend}), {len(track.beat_times)} beats, "
            f"{len(track.downbeat_times)} downbeats, stability {track.tempo_stability:.2f}"
        )
        for message in track.warnings:
            warn(message)
            self.result.warn(message, stage="beats")

        self.cache.write(key, track.to_dict())
        self._measure("beats", started, cached=False, backend=track.backend)
        return track

    # ---- stage 3: meter ----------------------------------------------- #
    def _cues(self, stems: StemSet, work_audio: str, beats: BeatTrack) -> _Cues:
        cfg = self.config
        beat_times = np.asarray(beats.beat_times, dtype=float)

        rule("stage 3 · time signature (meter) inference")
        started = time.time()

        rhythm_audio, rhythm_sr, rhythm_name = self._rhythm_audio(stems, work_audio)
        accent = meter_stage.beat_synchronous_accent(rhythm_audio, rhythm_sr, beat_times)

        # The prior can supply the meter outright.  When it does, the harmonic-change and
        # subdivision cues have no consumer at all -- their only reader is `infer_meter`,
        # which is not called on that path -- and they cost 1.68 s for 90 s of audio, so they
        # are skipped.  Zeros rather than None keep `_Cues` a plain data holder.
        reading = self.prior is not None and bool(self.prior.meter_spans)
        harmonic_change = np.zeros_like(beat_times)
        subdivision = 0.0
        if not reading:
            # These two need pitched, full-band content; the accent cue does not.
            mix, mix_sr = audio_io.load(work_audio, mono=True, sample_rate=ANALYSIS_SR)
            harmonic_change = meter_stage.harmonic_change_per_beat(
                mix, mix_sr, beat_times, backend=cfg.chroma_backend
            )
            subdivision = meter_stage.subdivision_ratio(mix, mix_sr, beat_times)

        # Two different series, for two different jobs:
        #  * the meter *periodicity* cue uses a backbeat-weighted one, because counting every
        #    eighth-note hi-hat equally flattens the accent pattern until the bar has no
        #    measurable period left;
        #  * the bar *phase* uses a kick-dominated one, because a loud backbeat snare would
        #    otherwise put every barline one beat late.
        #
        # The phase series is needed on both paths: `derive_downbeats` places barlines with it
        # even when the meter itself was read rather than inferred.
        #
        # Until the phase series was actually threaded through to the inference, the second
        # series was built here and then never used: the phase came from the periodicity cue's
        # own argmax, i.e. from raw accent, which is precisely the snare-on-2 trap this comment
        # describes.
        phase_series = accent
        for name in RHYTHM_SOURCES:
            path = stems.stems.get(name)
            if not path:
                continue
            hits, _ = drum_stage.transcribe_drums(path, sample_rate=rhythm_sr)
            if hits:
                if not reading:
                    accent = drum_stage.accent_from_hits(
                        hits, beat_times, weights=drum_stage.METER_WEIGHTS
                    )
                # The harmonic term is a *tie-breaker*, not an equal partner: on a dense mix
                # the change indicator fires on many beats, and weighting it heavily is what
                # dragged the phase around on real tracks.
                phase_series = 0.7 * drum_stage.accent_from_hits(
                    hits, beat_times, weights=drum_stage.PHASE_WEIGHTS
                ) + 0.3 * np.asarray(harmonic_change, dtype=float)
            break

        backend = cfg.chroma_backend if not reading else f"{cfg.chroma_backend}(phase only)"
        self._measure("meter_cues", started, cached=False, backend=backend)
        return _Cues(
            accent=accent,
            harmonic_change=harmonic_change,
            subdivision=subdivision,
            phase_series=phase_series,
            source=rhythm_name if not reading else f"{rhythm_name} (phase only)",
        )

    def _rhythm_audio(self, stems: StemSet, work_audio: str) -> tuple[np.ndarray, int, str]:
        """The drum stem when it is usable, otherwise the full mix."""
        for name in RHYTHM_SOURCES:
            path = stems.stems.get(name)
            if not path:
                continue
            y, sr = audio_io.load(path, mono=True, sample_rate=ANALYSIS_SR)
            if not audio_io.is_silent(y, threshold_db=-70.0):
                hint(f"meter accent cues taken from the '{name}' stem")
                return y, sr, name
            hint(f"the '{name}' stem is silent: accent cues come from the full mix")
        y, sr = audio_io.load(work_audio, mono=True, sample_rate=ANALYSIS_SR)
        return y, sr, "mix"

    def _locate_meter(
        self,
        spans: list[tuple[float, int, int, int]],
        beat_times: np.ndarray,
    ) -> tuple[int, int, int, list[str]]:
        """Turn SheetSage2's per-beat time signature into a numerator, denominator, phase.

        Returns ``(numerator, denominator, phase, notes)`` where ``notes`` are problems
        worth reporting, not exceptions -- a prior that cannot be reconciled is a reason to
        say so, not to abort a run over.

        Each span is ``(start_time, numerator, denominator, beat_in_bar)``; the last value is
        SheetSage2's own statement of which beat of the bar the span starts on, so the phase
        is read rather than derived.  There is no acoustic fallback: a prior without meter
        labels returns early saying so.
        """
        notes: list[str] = []
        if not spans:
            return 4, 4, 0, ["the prior has no time signature; falling back to 4/4"]

        counts: dict[tuple[int, int], int] = {}
        for index, span in enumerate(spans):
            start, numerator, denominator = span[0], span[1], span[2]
            end = spans[index + 1][0] if index + 1 < len(spans) else float("inf")
            counts[(numerator, denominator)] = counts.get((numerator, denominator), 0) + int(
                np.count_nonzero((beat_times >= start) & (beat_times < end))
            )
        if not counts:
            counts = {(spans[0][1], spans[0][2]): 1}

        # A first span no longer than one bar of its own signature is a pickup (anacrusis) --
        # `yoake` opens with 3 beats of 3/8 before 4/4 takes over, and 3 beats IS one bar of
        # 3/8, so the test is "at most one bar" rather than "less than one bar".  That is
        # ordinary notation, so it is neither a problem to report nor a candidate for "the
        # meter of the piece": excluding it matters, because a one-bar pickup can otherwise
        # out-count a short opening in the real signature.
        pickup: tuple[tuple[int, int], int] | None = None
        ranking = counts
        if len(spans) > 1:
            first = (spans[0][1], spans[0][2])
            first_beats = int(
                np.count_nonzero((beat_times >= spans[0][0]) & (beat_times < spans[1][0]))
            )
            if first_beats <= spans[0][1]:
                pickup = (first, first_beats)
                ranking = {k: v for k, v in counts.items() if k != first} or counts

        main = max(ranking, key=lambda key: ranking[key])
        numerator, denominator = main
        if pickup is not None:
            notes.append(
                f"beat.lab opens with {pickup[1]} beat(s) of {pickup[0][0]}/{pickup[0][1]} "
                f"before {numerator}/{denominator} takes over, which reads as a pickup"
            )

        for reading, count in counts.items():
            if reading == main or (pickup is not None and reading == pickup[0]):
                continue
            if count < reading[0]:
                # Shorter than one bar, so it cannot be followed without putting the whole
                # track on the wrong bar length to capture a bar that does not exist.
                notes.append(
                    f"beat.lab moves to {reading[0]}/{reading[1]} for {count} beat(s), shorter "
                    f"than one bar, so {numerator}/{denominator} is used throughout"
                )
            else:
                # At least a full bar.  This is a genuine meter change that is being ignored,
                # which is a different thing from a passing excursion and has to be said out
                # loud: `collapse` sustains 5/8 for 5 beats, one whole bar, and the honest
                # report is that there was a bar of 5/8 rather than that nothing happened.
                notes.append(
                    f"beat.lab holds {reading[0]}/{reading[1]} for {count} beat(s), at least "
                    f"a full bar, but {numerator}/{denominator} covers more of the track; the "
                    "score will not show that change"
                )

        # --- bar phase ------------------------------------------------------------- #
        #
        # `beat.lab` states which beat of the bar each beat is, so the phase is a reading.
        # An earlier version of this method instead aligned the prior's `downbeat.lab`
        # against `beat_times` and took the offset that matched best.  That is wrong in
        # principle, and measurably wrong here: the prior's beats come from SheetSage2 and
        # `beat_times` comes from a different tracker, so the two grids do not share beat
        # positions at all.  A real run reported "bar phase 0 ... (49/150 agree)" -- a third
        # of a match, which is the signature of comparing two unrelated grids rather than of
        # a bad phase.
        #
        # So the labelled phase is the answer, and `downbeat.lab` is used only as a check on
        # it.  If the check disagrees, the disagreement is reported.
        labelled = spans[0][3] if len(spans[0]) > 3 else 1
        # `beat_in_bar` counts from 1 within the span's own signature, not in beats: a 4/4
        # bar is labelled 1,2,3,4 (not eighths), so the beat-of-bar is the label itself.
        phase = (labelled - 1) % numerator
        if beat_times.size:
            early = int(np.count_nonzero(beat_times < spans[0][0]))
            phase = (early + phase) % numerator

        downbeats = np.asarray(self.prior.downbeat_times, dtype=float) if self.prior else None
        if downbeats is not None and downbeats.size:
            grid = beat_times[phase::numerator]
            agree = 0
            for when in downbeats:
                nearest = int(np.argmin(np.abs(grid - when)))
                if abs(float(grid[nearest]) - float(when)) <= 0.05:
                    agree += 1
            share = agree / float(downbeats.size)
            if share >= 0.5:
                info(
                    f"bar phase {phase} from the beat.lab labels "
                    f"({agree}/{downbeats.size} of the prior's downbeats land on it)"
                )
            else:
                # Not an error: the prior's downbeats live on the prior's own beat grid, so a
                # low agreement is expected when the two trackers disagree.  It is worth
                # saying, because it means the check could not confirm the reading.
                notes.append(
                    f"the beat.lab labels put the bar start on beat {phase}, but only "
                    f"{agree}/{downbeats.size} of the prior's downbeats land there; the two "
                    "beat grids differ, so the labels are used and this check is inconclusive"
                )
        return numerator, denominator, phase, notes

    def _meter(self, beats: BeatTrack, cues: _Cues) -> MeterResult:
        cfg = self.config
        started = time.time()
        beat_times = np.asarray(beats.beat_times, dtype=float)

        if cfg.meter_parts is not None:
            numerator, denominator = cfg.meter_parts
            _, phase = meter_stage.derive_downbeats(
                beat_times, numerator, phase_series=cues.phase_series
            )
            measured = MeterResult(
                hypotheses=[TimeSignature(numerator, denominator, 1.0, {"user_override": 1.0})],
                phase=phase,
                backend="user",
            )
            info(f"time signature overridden by the user: {numerator}/{denominator}")
        elif self.prior is not None and self.prior.meter_spans:
            # SheetSage2's own per-beat reading, taken as-is.
            #
            # This replaces what used to be the only path here: a four-cue weighted fusion
            # (downbeat spacing 1.6, drum accent 1.0, harmonic change 0.5, subdivision to
            # fix the denominator) that inferred a time signature from audio.  On 夜明けの
            # 歌 that inference and the prior disagreed -- the prior read 2/4 where the
            # tracker read 4/4 -- and this code resolved it by keeping whichever bar length
            # was more confident per bar.  Reading `beat.lab` removes the disagreement
            # instead of arbitrating it: the value is a reading, not a guess, and it is
            # available once per beat so a mid-song change survives.
            #
            # Measure first, decide second: the reading is checked for internal consistency
            # against the beat grid, and anything inconsistent is reported rather than
            # quietly trusted.
            spans = self.prior.meter_spans
            numerator, denominator, phase, span_notes = self._locate_meter(spans, beat_times)
            measured = MeterResult(
                hypotheses=[
                    TimeSignature(
                        numerator,
                        denominator,
                        1.0,
                        {"sheetsage2_beat_lab": 1.0},
                    )
                ],
                phase=phase,
                backend="sheetsage2-beat-lab",
            )
            if len(spans) > 1:
                info(
                    f"time signature from the SheetSage2 prior: {numerator}/{denominator} "
                    f"({len(spans)} reading(s) across the track)"
                )
            else:
                info(f"time signature from the SheetSage2 prior: {numerator}/{denominator}")
            for note in span_notes:
                warn(note)
                self.result.warn(note, stage="meter")
        else:
            prior = None if cfg.meter_prior == "pop" else meter_stage.FLAT_PRIOR
            # A SheetSage2 prior is a second downbeat source.  It and beat-this
            # disagree in a useful way — on 夜明けの歌 the prior reads 2/4 where the
            # tracker reads 4/4 — so each bar length keeps whichever source is more
            # confident instead of the two being averaged into a wrong consensus.
            second = self.prior.downbeat_times if self.prior is not None else None
            measured = meter_stage.infer_meter(
                beat_times,
                accent=cues.accent,
                downbeat_times=beats.downbeat_times or None,
                second_downbeats=second or None,
                harmonic_change=cues.harmonic_change,
                phase_series=cues.phase_series,
                subdivision=cues.subdivision,
                allow_odd=cfg.allow_odd_meter,
                prior=prior,
            )
            if second:
                info(
                    f"meter evidence includes the SheetSage2 prior "
                    f"({len(second)} downbeats, its own reading {self.prior.meter})"
                )
            # ``infer_meter`` resolved the phase itself, from the same priority
            # order (real downbeats, then the kick-dominated phase series) that
            # ``_phase`` used to recompute here.  Recomputing was not just
            # redundant: the two paths disagreed, and this one won, so the reported
            # phase could contradict the hypothesis the evidence had selected.

        if not beats.downbeat_times:
            numerator = measured.best.numerator if measured.best else 4
            downbeats, _ = meter_stage.derive_downbeats(
                beat_times, numerator, phase=measured.phase, phase_series=cues.phase_series
            )
            beats.downbeat_times = [float(t) for t in downbeats]

        info(f"subdivision ratio (triplet/duple): {cues.subdivision:.2f}")
        for rank, hypothesis in enumerate(measured.hypotheses, start=1):
            marker = "→" if rank == 1 else " "
            evidence = ", ".join(f"{k}={v:.2f}" for k, v in hypothesis.evidence.items())
            info(f"{marker} {hypothesis.label}  p={hypothesis.confidence:.2f}   [{evidence}]")
        for message in measured.warnings:
            warn(message)
            self.result.warn(message, stage="meter")
        if measured.best and measured.best.confidence < 0.5 and measured.backend != "user":
            message = (
                f"low confidence ({measured.best.confidence:.2f}) for {measured.best.label}: "
                "check the score, and override with --meter if needed"
            )
            warn(message)
            self.result.warn(message, stage="meter")

        self._measure("meter", started, cached=False, backend=measured.backend)
        return measured

    @staticmethod
    def _phase(beats: BeatTrack, measured: MeterResult, cues: _Cues) -> int:
        """Bar phase for a user-supplied meter: downbeats first, then the cues.

        Only used when the meter numerator was overridden, because the phase has
        to be recomputed for the new bar length.  The inferred path takes its phase
        from :func:`~ensemble2score.analysis.meter.infer_meter` directly.
        """
        beat_times = np.asarray(beats.beat_times, dtype=float)
        numerator = measured.best.numerator if measured.best else 4
        if beats.downbeat_times:
            first = float(beats.downbeat_times[0])
            return int(np.argmin(np.abs(beat_times - first))) % max(numerator, 1)
        _, phase = meter_stage.derive_downbeats(
            beat_times, numerator, phase_series=cues.phase_series
        )
        return phase

    # ---- stage 4: transcription --------------------------------------- #
    def _transcribe(self, stems: StemSet) -> None:
        cfg = self.config
        rule("stage 4 · per-stem transcription")

        candidates = [n for n in stems.names if n not in PERCUSSIVE_ONLY]
        if not candidates:
            message = "only percussion stems are present: nothing pitched to transcribe"
            warn(message)
            self.result.warn(message, stage="transcription")
            return

        loudest = max(stems.level_db.values()) if stems.level_db else 0.0
        kept: list[str] = []
        for name in candidates:
            level = stems.level_db.get(name)
            if level is None or not is_empty_stem(
                level,
                loudest,
                absolute_floor_db=cfg.min_stem_level_db,
                relative_drop_db=cfg.max_stem_drop_db,
            ):
                kept.append(name)
                continue
            message = (
                f"stem '{name}' is effectively silent ({level:.1f} dBFS): not transcribed, so no "
                "notes are invented from bleed. Lower --min-stem-level-db to force it."
            )
            warn(message)
            self.result.warn(message, stage="transcription")

        if not kept:
            message = "every pitched stem was empty: nothing to transcribe"
            warn(message)
            self.result.warn(message, stage="transcription")
            return

        transcriber = None
        ordered = order_parts(kept)

        # A backend that pays a large one-off cost per call (SheetSage2 loads a model
        # that dwarfs the transcription itself) is given every stem at once.
        # Backends without `transcribe_many` are unaffected.
        batch: dict[str, tuple[list[Note], list[str]]] = {}
        try:
            transcriber = transcribe_stage.build_transcriber(
                self.backends["transcriber"],
                device=self.device,
                onset_threshold=cfg.onset_threshold,
                frame_threshold=cfg.frame_threshold,
                minimum_note_length=cfg.minimum_note_length,
                ss2_python=cfg.ss2_python,
                ss2_hf_home=cfg.ss2_hf_home,
            )
        except Ensemble2ScoreError as exc:
            message = f"transcriber {self.backends['transcriber']!r} is unavailable: {exc}"
            warn(message)
            self.result.warn(message, stage="transcription")
            return

        if hasattr(transcriber, "transcribe_many") and len(ordered) > 1:
            batch = transcriber.transcribe_many(
                [(stem, stems.stems[stem], resolve(stem, self.overrides)) for stem in ordered]
            )
            info(f"{self.backends['transcriber']}: all stems transcribed in one pass")

        for stem in ordered:
            part, transcriber = self._transcribe_one(
                transcriber, stem, stems, precomputed=batch.get(stem)
            )
            if part is not None:
                self.result.parts.append(part)

        self._check_parts_are_distinct()

        if cfg.alt_transcriber:
            self._alternate(stems, ordered)

    def _check_parts_are_distinct(self, *, tolerance: float = 0.06) -> None:
        """Warn when two parts notate the same music.

        Two very different things look identical to a naive overlap count, and the
        message has to separate them:

        * **A split stem.**  A separation model that was never trained on an
          instrument still emits a stem for it, by dividing whatever it *did* learn.
          ``htdemucs_6s`` is the example: measured on a real track, 21% of the
          "guitar" part sat on an "other" note at the *same pitch* and the same
          instant, and a further 10% an octave away.  The guitar was not a guitar —
          it was the accompaniment, halved.
        * **Doubling.**  An accompaniment that tracks the melody an octave apart is
          ordinary arranging, not a defect.

        The discriminator is unison versus octave: a split yields mostly *identical*
        pitches, doubling yields octaves.  So only a unison share above
        :data:`_UNISON_ALARM` raises anything, and even then the wording names both
        readings instead of telling the user to delete a part that may be real.
        """
        import numpy as np

        for index, first in enumerate(self.result.parts):
            for second in self.result.parts[index + 1 :]:
                if not first.notes or not second.notes:
                    continue
                starts = np.array([n.start for n in second.notes])
                pitches = np.array([n.pitch for n in second.notes])
                unison = octave = 0
                for note in first.notes:
                    near = np.abs(starts - note.start) <= tolerance
                    if not near.any():
                        continue
                    distances = pitches[near] - note.pitch
                    if (distances == 0).any():
                        unison += 1
                    elif (np.abs(np.abs(distances) - 12) <= 1).any():
                        octave += 1

                unison_share = unison / len(first.notes)
                octave_share = octave / len(first.notes)
                if unison_share + octave_share < 0.3:
                    continue

                if unison_share >= _UNISON_ALARM and unison_share > 2 * octave_share:
                    message = (
                        f"parts '{first.stem}' and '{second.stem}' may be the same music: "
                        f"{unison_share:.0%} of '{first.stem}' has a '{second.stem}' note at "
                        f"the same pitch and instant. A separator that was never trained on "
                        f"one of these instruments splits whatever it did learn across two "
                        f"stems — 'htdemucs_6s' does this with piano and guitar. If these two "
                        f"are not really two instruments, re-run with --demucs-model htdemucs "
                        f"(all pitched accompaniment in 'other') rather than reading the same "
                        f"material twice"
                    )
                else:
                    message = (
                        f"parts '{first.stem}' and '{second.stem}' move together "
                        f"({octave_share:.0%} of '{first.stem}' doubles '{second.stem}' an "
                        f"octave away at the same instant). That is normal arranging if one "
                        f"is a melody and the other its accompaniment — worth a glance before "
                        f"the score is engraved"
                    )
                warn(message)
                self.result.warn(message, stage="transcription")

    def _transcribe_one(
        self,
        transcriber,
        stem: str,
        stems: StemSet,
        *,
        precomputed: tuple[list[Note], list[str]] | None = None,
    ):
        """Transcribe one stem.  Returns ``(part_or_None, transcriber)``.

        The backend instance is threaded back out so that model weights are loaded
        once per run rather than once per stem.  ``precomputed`` carries notes a
        batch pass already produced, so gating, caching and post-processing still
        happen exactly once, in one place, whichever way the notes arrived.
        """
        cfg = self.config
        spec = resolve(stem, self.overrides)
        path = stems.stems[stem]
        # Level gating has to happen before the transcriber, and it has to be part of
        # the cache key: a cached transcription produced without the gate is exactly
        # the bug being fixed, and reusing it would hide the fix.
        gated_path: str | None = None
        if cfg.gate_silence:
            gated_path, gated_share = transcribe_stage.gate_stem(
                path, cfg.silence_floor_margin_db, cfg.silence_active_margin_db
            )
            if gated_share > 0.01:
                info(
                    f"{stem}: {gated_share:.0%} of the stem is near-silent and was gated "
                    "before transcription"
                )
            else:
                info(f"{stem}: no near-silent stretch worth gating ({gated_share:.1%})")
        # Every option that changes the *post-processed notes* belongs in the key.
        # The key used to list only the transcriber and the minimum duration, so
        # `--fix-octaves off`, a different `--max-simultaneous` or an edited
        # `--instrument-map` all silently reused a cached transcription that had
        # been produced under the old settings.
        key = self.cache.key(
            "transcribe",
            content_hash(path),
            self.backends["transcriber"],
            cfg.min_note_duration,
            cfg.fix_octaves,
            cfg.max_simultaneous,
            cfg.gate_silence,
            cfg.silence_floor_margin_db,
            cfg.silence_active_margin_db,
            cfg.onset_threshold,
            cfg.frame_threshold,
            cfg.minimum_note_length,
            cfg.min_confidence,
            # A different interpreter or weight cache is a different transcriber, so
            # both belong in the key: without them, installing SheetSage2 into a new
            # venv would silently reuse notes produced by the old one.
            cfg.ss2_python,
            cfg.ss2_hf_home,
            spec.to_dict(),
        )
        started = time.time()
        if transcriber is None and precomputed is None:
            transcriber = transcribe_stage.build_transcriber(
                self.backends["transcriber"],
                device=self.device,
                onset_threshold=cfg.onset_threshold,
                frame_threshold=cfg.frame_threshold,
                minimum_note_length=cfg.minimum_note_length,
                ss2_python=cfg.ss2_python,
                ss2_hf_home=cfg.ss2_hf_home,
            )

        if self.cache.has(key):
            info(f"{stem}: reusing the cached transcription")
            payload = self.cache.read(key)
            part = Part(
                stem=stem,
                spec=spec,
                notes=[Note.from_dict(n) for n in payload.get("notes", [])],
                backend=payload.get("backend", "?"),
                warnings=list(payload.get("warnings", [])),
            )
            self._measure(f"transcribe:{stem}", started, cached=True, backend=part.backend)
            return part, transcriber

        try:
            if precomputed is not None:
                notes, messages = precomputed
            else:
                notes, messages = transcriber.transcribe(gated_path or path, spec)
        except Ensemble2ScoreError as exc:
            warn(f"{stem}: {exc}")
            self.result.warn(str(exc), stage=f"transcription({stem})")
            return None, transcriber
        except Exception as exc:  # noqa: BLE001 - model runtimes raise anything
            warn(f"{stem}: transcription failed ({type(exc).__name__}: {exc})")
            self.result.warn(
                f"transcription failed ({type(exc).__name__}: {exc})",
                stage=f"transcription({stem})",
            )
            return None, transcriber

        notes, messages = self._post_process(notes, messages, spec)

        if not notes:
            message = "no notes were transcribed from this stem"
            warn(f"{stem}: {message}")
            self.result.warn(message, stage=f"transcription({stem})")

        part = Part(stem=stem, spec=spec, notes=notes, backend=transcriber.name, warnings=messages)
        info(
            f"{stem}: {part.n_notes} notes via {part.backend}"
            + (f", range {part.pitch_range}" if part.pitch_range else "")
        )
        for message in messages:
            warn(f"{stem}: {message}")
            self.result.warn(message, stage=f"transcription({stem})")

        self.cache.write(
            key,
            {
                "stem": stem,
                "backend": part.backend,
                "warnings": messages,
                "notes": [n.to_dict() for n in notes],
            },
        )
        self._measure(f"transcribe:{stem}", started, cached=False, backend=part.backend)
        return part, transcriber

    def _post_process(
        self, notes: list[Note], messages: list[str], spec
    ) -> tuple[list[Note], list[str]]:
        """Turn raw transcriber output into notes worth notating.

        Extracted so that the A/B backend is filtered *identically* to the primary
        one.  If the comparison ran its own copy of this chain, a difference between
        the two parts could come from the filters rather than from the backends,
        which would make the comparison worthless.

        Order matters, and it is the order the note events arrive in that decides it:
        consolidation has to come *before* the short-note filter, because a
        frame-based transcriber chops one sung note into several pieces, each below
        ``--min-note-duration``, and filtering first throws the whole melody away
        instead of joining the pieces back together.
        """
        cfg = self.config

        notes, consolidate_messages = transcribe_stage.consolidate_notes(notes)
        messages.extend(consolidate_messages)

        # Confidence filtering comes after consolidation and before everything else:
        # joining fragments first means a note is judged on its whole extent rather
        # than on its weakest frame, and dropping inventions early keeps them from
        # consuming the simultaneous-note budget or the instrument's range check.
        notes, low_confidence = transcribe_stage.filter_by_confidence(
            notes, minimum=cfg.min_confidence
        )
        if low_confidence:
            messages.append(
                f"dropped {low_confidence} note(s) below confidence "
                f"{cfg.min_confidence:.2f}: the transcriber was guessing at those, and "
                "on a separated stem a guess is usually separation noise or reverb"
            )

        notes, dropped = transcribe_stage.merge_short_notes(notes, minimum=cfg.min_note_duration)
        if dropped:
            messages.append(f"dropped {dropped} note(s) shorter than {cfg.min_note_duration:.3f}s")

        # Range filtering comes before the voice limits: harmonics are energy peaks
        # at pitches the instrument cannot produce, and removing them first means
        # the simultaneous-note budget is spent on real notes.
        notes, range_messages = transcribe_stage.fit_to_instrument_range(
            notes, spec, fix_octaves=cfg.fix_octaves
        )
        messages.extend(range_messages)

        if spec.monophonic and cfg.fix_octaves:
            notes, octave_messages = transcribe_stage.fix_octave_jumps(notes)
            messages.extend(octave_messages)

        ceiling = 1 if spec.monophonic else cfg.max_simultaneous
        notes, trimmed = transcribe_stage.limit_simultaneous(notes, ceiling)
        if trimmed:
            messages.append(f"limited to {ceiling} simultaneous note(s): removed {trimmed}")
        return notes, messages

    # ---- stage 4b: the A/B backend ------------------------------------- #
    def _alternate(self, stems: StemSet, order: list[str]) -> None:
        """Run a second transcriber over the same stems, for comparison only.

        The notes are written as per-part MIDI under ``out/alt/`` and never reach the
        score.  Both backends see the *same* gated stem and pass through the same
        :meth:`_post_process`, so a difference between the two outputs is a
        difference between the transcriber, which is the only question this answers.
        Doing it here rather than as a second pipeline run means separation, beats,
        meter and notation are not paid for twice.
        """
        cfg = self.config
        name = cfg.alt_transcriber
        assert name is not None
        rule(f"stage 4b · comparison transcription via {name}")

        try:
            transcriber = transcribe_stage.build_transcriber(
                name,
                device=self.device,
                onset_threshold=cfg.onset_threshold,
                frame_threshold=cfg.frame_threshold,
                minimum_note_length=cfg.minimum_note_length,
                ss2_python=cfg.ss2_python,
                ss2_hf_home=cfg.ss2_hf_home,
            )
        except Ensemble2ScoreError as exc:
            message = f"comparison backend {name!r} is unavailable: {exc}"
            warn(message)
            self.result.warn(message, stage="alternate")
            return

        alt_dir = self.out_dir / "alt" / name
        alt_dir.mkdir(parents=True, exist_ok=True)
        summary: dict[str, int] = {}

        for stem in order:
            path = stems.stems.get(stem)
            if path is None:
                continue
            spec = resolve(stem, self.overrides)
            key = self.cache.key(
                "alternate",
                name,
                content_hash(path),
                cfg.gate_silence,
                cfg.silence_floor_margin_db,
                cfg.silence_active_margin_db,
                cfg.min_note_duration,
                cfg.fix_octaves,
                cfg.max_simultaneous,
                cfg.min_confidence,
                cfg.ss2_python,
                cfg.ss2_hf_home,
                spec.to_dict(),
            )
            if self.cache.has(key):
                cached = self.cache.read(key)
                target = transcribe_stage.write_note_midi(
                    [Note.from_dict(item) for item in cached.get("notes", [])],
                    spec,
                    alt_dir / f"{stem}.mid",
                )
                if target is not None:
                    info(f"{stem}: {len(cached.get('notes', []))} note(s) via {name} (cached)")
                    self.result.outputs[f"alt:{name}:{stem}"] = str(target)
                continue

            gated_path = None
            if cfg.gate_silence:
                # Reused rather than recomputed: the primary pass already wrote it,
                # and gating is a property of the audio, not of the backend.
                gated_path, _share = transcribe_stage.gate_stem(
                    path, cfg.silence_floor_margin_db, cfg.silence_active_margin_db
                )
            try:
                notes, messages = transcriber.transcribe(gated_path or path, spec)
            except Exception as exc:  # noqa: BLE001 - one backend must not kill the run
                message = f"{name} failed on stem {stem!r} ({type(exc).__name__}: {exc})"
                warn(message)
                self.result.warn(message, stage=f"alternate({stem})")
                continue

            notes, messages = self._post_process(notes, messages, spec)
            summary[stem] = len(notes)
            info(f"{stem}: {len(notes)} note(s) via {name} (comparison only)")
            self.cache.write(key, {"notes": [note.to_dict() for note in notes]})
            target = transcribe_stage.write_note_midi(notes, spec, alt_dir / f"{stem}.mid")
            if target is not None:
                self.result.outputs[f"alt:{name}:{stem}"] = str(target)

        if summary:
            self.result.warn(
                f"comparison backend {name}: "
                + ", ".join(f"{stem} {count}" for stem, count in sorted(summary.items()))
                + " note(s); written under out/alt/ and NOT used for the score",
                stage="alternate",
            )

    # ---- stage 5: percussion ------------------------------------------ #
    def _percussion(self, stems: StemSet) -> list[DrumHit]:
        cfg = self.config
        if not cfg.drums:
            return []
        name = next((n for n in RHYTHM_SOURCES if n in stems.stems), None)
        if name is None:
            return []

        level = stems.level_db.get(name, 0.0)
        loudest = max(stems.level_db.values()) if stems.level_db else 0.0
        if is_empty_stem(
            level,
            loudest,
            absolute_floor_db=cfg.min_stem_level_db,
            relative_drop_db=cfg.max_stem_drop_db,
        ):
            message = f"the percussion stem is effectively silent ({level:.1f} dBFS): skipping it"
            warn(message)
            self.result.warn(message, stage="drums")
            return []

        rule("stage 5 · percussion part")
        path = stems.stems[name]
        key = self.cache.key("drums", content_hash(path))
        started = time.time()
        if self.cache.has(key):
            info("percussion: reusing the cached result")
            hits = [DrumHit.from_dict(h) for h in self.cache.read(key)]
            self._measure("drums", started, cached=True, backend="onset-classifier")
            return hits

        hits, messages = drum_stage.transcribe_drums(path)
        counts = {voice: sum(1 for h in hits if h.voice == voice) for voice in DRUM_VOICES}
        info("percussion onsets: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
        if len(hits) > 4000:
            message = f"{len(hits)} percussion onsets will make the drum staff dense"
            warn(message)
            self.result.warn(message, stage="drums")
        for message in messages:
            self.result.warn(message, stage="drums")

        self.cache.write(key, [h.to_dict() for h in hits])
        self._measure("drums", started, cached=False, backend="onset-classifier")
        return hits

    # ---- stage 6: chords ---------------------------------------------- #
    def _chords(self, stems: StemSet, work_audio: str, beats: BeatTrack) -> list[Chord]:
        cfg = self.config
        if not cfg.chords:
            return []
        rule("stage 6 · chord symbols")

        # A SheetSage2 prior's chords are a transcription of the harmony by a model
        # trained to produce them, while this stage is template matching over chroma
        # — the least reliable thing in the chain, by its own documentation.  So the
        # prior wins when one was supplied, and the chroma path stays as the default
        # for runs without one.
        if self.prior is not None and self.prior.chords and cfg.ss2_chords:
            chords = list(self.prior.chords)
            preview = " ".join(c.label for c in chords[:12])
            info(
                f"{len(chords)} chord segments from the SheetSage2 prior: {preview}"
                f"{' …' if len(chords) > 12 else ''}"
            )
            hint("chords come from the SheetSage2 prior, not from chroma template matching")
            self._measure("chords", time.time(), cached=False, backend="sheetsage2-prior")
            return chords

        source = next(
            (
                stems.stems[name]
                for name in ("other", "piano", "keys", "harmonic")
                if name in stems.stems
            ),
            work_audio,
        )
        key = self.cache.key("chords", content_hash(source), cfg.chroma_backend)
        started = time.time()
        if self.cache.has(key):
            info("chords: reusing the cached result")
            chords = [Chord.from_dict(c) for c in self.cache.read(key)]
            self._measure("chords", started, cached=True, backend=cfg.chroma_backend)
            return chords

        y, sr = audio_io.load(source, mono=True, sample_rate=ANALYSIS_SR)
        recognizer = ChordRecognizer(chroma_backend=cfg.chroma_backend)
        chords = recognizer.recognize(y, sr, np.asarray(beats.beat_times, dtype=float))
        if chords:
            preview = " ".join(c.label for c in chords[:12])
            info(f"{len(chords)} chord segments: {preview}{' …' if len(chords) > 12 else ''}")
        else:
            message = "no chord segments were recognised"
            warn(message)
            self.result.warn(message, stage="chords")

        self.cache.write(key, [c.to_dict() for c in chords])
        self._measure("chords", started, cached=False, backend=cfg.chroma_backend)
        return chords

    # ---- stage 7: notation -------------------------------------------- #
    def _notate(self, beats: BeatTrack) -> None:
        from .notation.score import ScoreOptions, build_score, export_score

        cfg = self.config
        rule("stage 7 · notation")
        started = time.time()

        best = self.result.meter.best if self.result.meter else None
        timeline = build_timeline(
            {part.stem: part.notes for part in self.result.parts},
            beat_times=beats.beat_times,
            beats_per_bar=best.numerator if best else 4,
            phase=self.result.meter.phase if self.result.meter else 0,
            grid=cfg.grid,
            swing=cfg.swing,
        )
        for message in timeline.warnings:
            self.result.warn(message, stage="notation")

        if not timeline.parts:
            message = "no notes survived quantisation: no score was written"
            warn(message)
            self.result.warn(message, stage="notation")
            self._measure("notate", started, cached=False, backend="music21")
            return

        options = ScoreOptions(
            grid=cfg.grid,
            swing=cfg.swing,
            max_simultaneous=cfg.max_simultaneous,
            include_chords=cfg.chords,
            include_drums=cfg.drums,
            estimate_key=cfg.key,
            title=cfg.title,
            composer=cfg.composer,
            sections=tuple(self.prior.structure) if self.prior is not None else (),
        )
        score, stats, midi_parts = build_score(self.result, options, timeline=timeline)
        self.result.measures = stats.measures
        self.result.key = stats.key
        for message in stats.warnings:
            self.result.warn(message, stage="notation")

        written = export_score(
            score,
            self.out_dir,
            self._basename(),
            midi_parts=midi_parts,
            tempo_bpm=beats.tempo_bpm,
            make_pdf=cfg.pdf,
            mscore_path=cfg.mscore,
            split_midi=cfg.stem_midi,
        )
        self.result.outputs.update(written)
        info(
            f"{stats.time_signature}, {stats.measures} measures, {len(stats.parts)} part(s), "
            f"{stats.notes} written note(s)"
            + (f", key {stats.key}" if stats.key else "")
            + (f", {stats.ties} tie(s)" if stats.ties else "")
        )
        for name, path in written.items():
            info(f"wrote {name}: {path}")
        self._measure("notate", started, cached=False, backend="music21")

    # ------------------------------------------------------------------ #
    def _basename(self) -> str:
        return self.config.title or slugify(Path(self.config.input_path).stem, 60)

    def _payload(self) -> dict:
        payload = self.result.to_dict()
        payload["generator"] = f"ensemble2score {__version__}"
        payload["config"] = self.config.to_dict()
        return payload

    def _write(self) -> None:
        from .report import write_report
        from .utils import save_json

        save_json(self.out_dir / "run.json", self._payload())
        report = write_report(self.result, self.out_dir, generator=f"ensemble2score {__version__}")
        self.result.outputs["report"] = str(report)
        save_json(self.out_dir / "run.json", self._payload())
        info(f"wrote report: {report}")


def _apply_tempo_override(beats: BeatTrack, requested: float) -> None:
    """Apply a user tempo, keeping the beat grid consistent with it.

    Setting only the BPM number produces a self-contradictory score: the notation
    claims 71.8 BPM while the bar lines are still spaced by the detected 143.6 BPM
    beat, so a 3/4 reading yields 211 bars where 80 is right.

    So when the request is a clean half of the detection, the grid is *rebuilt* by
    merging adjacent detected beats — which is exactly the reading "the real pulse is
    every other click".  A request that is not such a ratio is treated as a notation
    preference only: the beats really did land where they landed, and re-spacing them
    would move the music, so just the tempo marking changes.
    """
    detected = float(beats.tempo_bpm)
    if detected <= 0:
        beats.tempo_bpm = requested
        return

    ratio = detected / requested
    if abs(ratio - 2.0) < 0.04 and len(beats.beat_times) >= 4:
        merged = beats.beat_times[::2]
        kept = len(merged)
        hint(
            f"tempo overridden to {requested:.1f} BPM (detected {detected:.1f}): "
            f"merging detected beats in pairs, {len(beats.beat_times)} -> {kept} beats"
        )
        beats.beat_times = merged
        beats.downbeat_times = beats.downbeat_times[::2]
        beats.tempo_bpm = requested
        beats.tempo_stability = 1.0
        return

    hint(f"tempo overridden by the user: {requested:.1f} BPM (was {detected:.1f})")
    beats.tempo_bpm = requested


def run(config: RunConfig) -> StageOutcome:
    """Run the pipeline for ``config``.  The one function most callers need."""
    from .io.console import set_verbose

    set_verbose(config.verbose)
    return Pipeline(config).run()
