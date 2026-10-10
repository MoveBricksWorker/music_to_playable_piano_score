"""The ``e2s`` command line.

Design rules for this module:

* **Nothing heavy is imported at module scope.**  ``e2s --help`` and
  ``e2s version`` must answer instantly, so librosa, torch, music21 and demucs are
  all imported inside the command that needs them.
* **The numba cache is configured before any analysis import.**  Numba reads
  ``NUMBA_CACHE_DIR`` when it is first imported, so :func:`main` calls
  :func:`ensemble2score.analysis.warmup.configure` first thing.
* **A user-facing failure is one line, not a traceback.**  Expected errors are
  caught and printed; ``--debug`` re-raises for bug reports.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer

from . import __version__
from .analysis.warmup import configure as configure_numba
from .core.config import (
    BEAT_BACKENDS,
    DEMUCS_MODELS,
    DEVICES,
    PRIORS,
    SEPARATORS,
    TRANSCRIBERS,
    RunConfig,
)
from .core.errors import Ensemble2ScoreError

__all__ = ["app", "main"]

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Turn a mixed song into a multi-instrument ensemble score (MusicXML / MIDI).",
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _fail(message: str, code: int = 2) -> None:
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code)


def _guard(fn, *args, **kwargs):
    """Run ``fn``, turning expected errors into one clean line."""
    debug = kwargs.pop("_debug", False)
    try:
        return fn(*args, **kwargs)
    except Ensemble2ScoreError as exc:
        if debug:
            raise
        _fail(str(exc))
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        typer.secho("interrupted", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(130) from None
    except Exception as exc:  # noqa: BLE001 - last line of defence for the CLI
        if debug:
            raise
        _fail(f"{type(exc).__name__}: {exc}\n(run with --debug for the full traceback)", code=1)


def _echo_kv(label: str, value: str, *, ok: bool | None = None) -> None:
    marker = "" if ok is None else ("  ok " if ok else "  -- ")
    colour = typer.colors.GREEN if ok else (typer.colors.YELLOW if ok is False else None)
    text = f"{marker}{label:<26} {value}"
    typer.secho(text, fg=colour)


@app.callback()
def _root(
    version_flag: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Print the version and exit.",
        is_eager=True,
    ),
) -> None:
    """Configure the numba cache before anything can import librosa."""
    if version_flag:
        typer.echo(f"ensemble2score {__version__}")
        raise typer.Exit(0)
    configure_numba()


# --------------------------------------------------------------------------- #
# informational commands
# --------------------------------------------------------------------------- #
@app.command()
def version() -> None:
    """Print the version and exit."""
    typer.echo(f"ensemble2score {__version__}")


@app.command()
def warmup(
    force: bool = typer.Option(False, "--force", help="Rebuild the cache even if it looks warm."),
) -> None:
    """Pre-compile the numba kernels librosa uses (run once after installing).

    Numba compiles on first call *inside each process*.  On Windows that takes
    minutes of CPU, with no output, so a cold run looks like a hang.  This command
    pays that cost once and stores the result, after which every run starts fast.
    """
    from .analysis import warmup as warm_module

    if warm_module.is_warm() and not force:
        typer.secho(
            f"already warm: compiled objects found in {warm_module.cache_dir()}", fg=typer.colors.GREEN
        )
        raise typer.Exit(0)

    typer.echo(f"compiling numba kernels into {warm_module.cache_dir()} …")
    typer.echo("this takes a few minutes once; later runs reuse the cache")
    report = warm_module.warm()
    if report.failed:
        for name, error in report.failed:
            typer.secho(f"  failed {name}: {error}", fg=typer.colors.YELLOW)
    typer.secho(
        f"warmed {len(report.compiled)} path(s) in {report.seconds:.1f}s", fg=typer.colors.GREEN
    )
    if not report.ok:
        raise typer.Exit(1)


@app.command()
def doctor() -> None:
    """Report which backends are installed, and the licence of each."""
    from .io.registry import REGISTRY

    typer.secho(f"ensemble2score {__version__}", bold=True)
    typer.echo(f"python           {sys.version.split()[0]} ({sys.executable})")

    from .analysis import warmup as warm_module

    _echo_kv("numba cache", str(warm_module.cache_dir()), ok=warm_module.is_warm())

    typer.echo("")
    typer.secho("backends", bold=True)
    for module, label, hint in (
        ("numpy", "core: numpy", "required"),
        ("librosa", "core: librosa", "required"),
        ("soundfile", "core: soundfile", "required"),
        ("music21", "core: music21", "required"),
        ("pretty_midi", "core: pretty_midi", "required"),
        ("demucs", "separation: demucs", "pip install -e '.[separation]'"),
        ("torch", "separation: torch", "pip install -e '.[separation]'"),
        ("basic_pitch", "transcription: basic-pitch", "pip install -e '.[transcription]'"),
        ("beat_this", "beats: beat-this", "pip install -e '.[beats]'"),
        ("mir_eval", "eval: mir_eval", "pip install mir_eval"),
    ):
        _echo_kv(label, _module_status(module, hint))

    _echo_kv("MuseScore (for --pdf)", _find_mscore() or "not found on PATH")

    typer.echo("")
    typer.secho("licences", bold=True)
    for component in sorted(REGISTRY, key=lambda c: (c.kind, c.name.lower())):
        flag = "" if component.auto_selectable else "  (never auto-selected)"
        typer.echo(f"  {component.name:<32} {component.licence:<24} {component.licence_class}{flag}")
    typer.echo("")
    typer.echo("Model weights are licensed separately from code; this project ships none.")
    typer.echo("Use with caution: " + ", ".join(c.name for c in REGISTRY if c.requires_warning))


def _module_status(name: str, hint: str) -> str:
    import importlib.util

    if importlib.util.find_spec(name) is None:
        return f"not installed  ({hint})"
    try:
        module = __import__(name)
    except Exception as exc:  # noqa: BLE001 - a broken install is worth reporting
        return f"import failed: {type(exc).__name__}: {exc}"
    version = getattr(module, "__version__", "")
    return f"{version or 'installed'}"


def _find_mscore() -> str | None:
    """Locate the MuseScore binary.

    PATH is checked first, but on Windows the installer does not add itself to it,
    so the well-known install locations are probed too.  Without this, ``--pdf``
    silently reports "not found" on exactly the platform where MuseScore is most
    likely to be installed by hand.
    """
    import os
    import shutil

    for name in ("mscore", "musescore", "MuseScore4"):
        found = shutil.which(name)
        if found:
            return found

    candidates: list[str] = []
    for root in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"),
                 os.environ.get("LOCALAPPDATA", "") + r"\Programs" if os.environ.get("LOCALAPPDATA") else None):
        if not root:
            continue
        candidates.extend(
            os.path.join(root, folder, "bin", exe)
            for folder in ("MuseScore 4", "MuseScore 3")
            for exe in ("MuseScore4.exe", "MuseScore3.exe", "mscore.exe")
        )
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


# --------------------------------------------------------------------------- #
# the pipeline
# --------------------------------------------------------------------------- #
@app.command()
def run(
    input_audio: Path = typer.Argument(..., exists=True, dir_okay=False, help="Input audio file."),
    out: Path = typer.Option(Path("out"), "-o", "--out", help="Output directory."),
    separator: str = typer.Option("auto", "--separator", help=f"One of {', '.join(SEPARATORS)}."),
    stems_dir: Optional[Path] = typer.Option(None, "--stems-dir", help="Use stems you already have."),
    demucs_model: str = typer.Option(
        "htdemucs_6s",
        "--demucs-model",
        help=f"One of {', '.join(DEMUCS_MODELS)}. '6s' adds piano and guitar; no model outputs strings.",
    ),
    device: str = typer.Option("auto", "--device", help=f"One of {', '.join(DEVICES)}."),
    min_stem_level_db: float = typer.Option(-40.0, "--min-stem-level-db", help="Treat quieter stems as absent."),
    max_stem_drop_db: float = typer.Option(30.0, "--max-stem-drop-db", help="Drop stems this far below the loudest."),
    beats: str = typer.Option("auto", "--beats", help=f"One of {', '.join(BEAT_BACKENDS)}."),
    tempo: Optional[float] = typer.Option(None, "--tempo", help="Override the detected BPM."),
    meter: Optional[str] = typer.Option(None, "--meter", help="Override the time signature, e.g. 3/4."),
    meter_prior: str = typer.Option("pop", "--meter-prior", help=f"One of {', '.join(PRIORS)}."),
    allow_odd_meter: bool = typer.Option(False, "--allow-odd-meter", help="Also consider 5/4 and 7/4."),
    ss2_prior: Optional[Path] = typer.Option(
        None,
        "--ss2-prior",
        help="A SheetSage2 output directory to use as a prior (downbeats, key, chords, sections).",
    ),
    ss2_chords: bool = typer.Option(
        True,
        "--ss2-chords/--no-ss2-chords",
        help="With --ss2-prior, take chord symbols from the prior instead of chroma matching.",
    ),
    transcriber: str = typer.Option("auto", "--transcriber", help=f"One of {', '.join(TRANSCRIBERS)}."),
    alt_transcriber: Optional[str] = typer.Option(
        None,
        "--alt-transcriber",
        help=(
            "Also run this second transcriber over the same stems and write its notes "
            "as MIDI under out/alt/ (A/B aid; the score still uses --transcriber)."
        ),
    ),
    ss2_python: Optional[Path] = typer.Option(
        None,
        "--ss2-python",
        help="Interpreter of the environment holding SheetSage2 (for --transcriber sheetsage2).",
    ),
    ss2_hf_home: Optional[Path] = typer.Option(
        None,
        "--ss2-hf-home",
        help="HF cache directory holding the SheetSage2/MERT weights.",
    ),
    fix_octaves: bool = typer.Option(True, "--fix-octaves/--no-fix-octaves", help="Correct isolated octave jumps in monophonic parts."),
    min_note_duration: float = typer.Option(0.05, "--min-note-duration", help="Drop notes shorter than this (seconds)."),
    chords: bool = typer.Option(True, "--chords/--no-chords", help="Write chord symbols."),
    drums: bool = typer.Option(True, "--drums/--no-drums", help="Write a percussion staff."),
    chroma_backend: str = typer.Option("cqt", "--chroma-backend", help="cqt (better) or stft (faster)."),
    key: bool = typer.Option(True, "--key/--no-key", help="Estimate a key signature."),
    grid: int = typer.Option(4, "--grid", help="Quantisation subdivisions per beat (power of two)."),
    swing: float = typer.Option(0.0, "--swing", help="0.0-0.66 swing on off-beat subdivisions."),
    max_simultaneous: int = typer.Option(
        8,
        "--max-simultaneous",
        help=(
            "Max simultaneous notes in a polyphonic part. basic-pitch returns up to "
            "six; a lower ceiling only damages chords."
        ),
    ),
    onset_threshold: float = typer.Option(
        0.5, "--onset-threshold", help="basic-pitch onset threshold (lower finds more notes)."
    ),
    frame_threshold: float = typer.Option(
        0.3, "--frame-threshold", help="basic-pitch frame threshold (lower sustains more notes)."
    ),
    minimum_note_length: float = typer.Option(
        0.05, "--minimum-note-length", help="Seconds; notes shorter than this are artefacts."
    ),
    min_confidence: float = typer.Option(
        0.5,
        "--min-confidence",
        help=(
            "Drop notes the transcriber was unsure about. On a separated stem a "
            "low-confidence note is usually separation noise or reverb. 0 disables."
        ),
    ),
    gate_silence: bool = typer.Option(
        True,
        "--gate-silence/--no-gate-silence",
        help=(
            "Zero near-silent stretches of a stem before transcription. Without it a "
            "transcriber invents notes in silence, and they cannot be filtered after."
        ),
    ),
    instrument_map: Optional[Path] = typer.Option(None, "--instrument-map", help="YAML stem to instrument overrides."),
    pdf: bool = typer.Option(False, "--pdf", help="Also render a PDF (needs MuseScore)."),
    mscore: Optional[Path] = typer.Option(None, "--mscore", help="Path to the MuseScore binary."),
    stem_midi: bool = typer.Option(
        True,
        "--stem-midi/--no-stem-midi",
        help="Also write one MIDI file per part under out/midi/.",
    ),
    title: Optional[str] = typer.Option(None, "--title", help="Score title (also the output basename)."),
    composer: Optional[str] = typer.Option(None, "--composer", help="Composer credit for the score."),
    max_seconds: Optional[float] = typer.Option(None, "--max-seconds", help="Only process the first N seconds."),
    cache: bool = typer.Option(True, "--cache/--no-cache", help="Reuse cached stage results."),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Only print warnings and errors."),
    debug: bool = typer.Option(False, "--debug", help="Show tracebacks instead of one-line errors."),
) -> None:
    """Run the whole pipeline: audio in, ensemble score out."""
    from .pipeline import run as run_pipeline

    config = _guard(
        RunConfig,
        input_path=str(input_audio),
        out_dir=str(out),
        separator=separator,
        stems_dir=str(stems_dir) if stems_dir else None,
        demucs_model=demucs_model,
        device=device,
        min_stem_level_db=min_stem_level_db,
        max_stem_drop_db=max_stem_drop_db,
        beats=beats,
        tempo=tempo,
        meter=meter,
        meter_prior=meter_prior,
        allow_odd_meter=allow_odd_meter,
        ss2_prior=str(ss2_prior) if ss2_prior else None,
        ss2_chords=ss2_chords,
        transcriber=transcriber,
        alt_transcriber=alt_transcriber,
        ss2_python=str(ss2_python) if ss2_python else None,
        ss2_hf_home=str(ss2_hf_home) if ss2_hf_home else None,
        fix_octaves=fix_octaves,
        min_note_duration=min_note_duration,
        chords=chords,
        drums=drums,
        chroma_backend=chroma_backend,
        key=key,
        grid=grid,
        swing=swing,
        max_simultaneous=max_simultaneous,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length,
        min_confidence=min_confidence,
        gate_silence=gate_silence,
        instrument_map=str(instrument_map) if instrument_map else None,
        pdf=pdf,
        mscore=str(mscore) if mscore else None,
        stem_midi=stem_midi,
        title=title,
        composer=composer,
        max_seconds=max_seconds,
        cache=cache,
        verbose=not quiet,
        _debug=debug,
    )
    outcome = _guard(run_pipeline, config, _debug=debug)

    typer.echo("")
    typer.secho(f"done: {outcome.result.time_signature}", bold=True)
    for name, path in outcome.result.outputs.items():
        typer.echo(f"  {name}: {path}")
    if outcome.result.warnings:
        typer.secho(f"  {len(outcome.result.warnings)} warning(s): see report.md", fg=typer.colors.YELLOW)


@app.command()
def rhythm(
    input_audio: Path = typer.Argument(..., exists=True, dir_okay=False),
    out: Optional[Path] = typer.Option(None, "-o", "--out", help="Write the result as JSON here."),
    beats: str = typer.Option("auto", "--beats"),
    meter: Optional[str] = typer.Option(None, "--meter"),
    chroma_backend: str = typer.Option("cqt", "--chroma-backend"),
    max_seconds: Optional[float] = typer.Option(None, "--max-seconds"),
    quiet: bool = typer.Option(False, "-q", "--quiet"),
    debug: bool = typer.Option(False, "--debug"),
) -> None:
    """Analyse tempo, beats and time signature only.  Fast; no separation."""
    from .analysis.beats import build_beat_tracker
    from .analysis import meter as meter_stage
    from .io import audio as audio_io
    from .io.console import set_verbose
    import numpy as np

    set_verbose(not quiet)
    config = _guard(
        RunConfig,
        input_path=str(input_audio),
        out_dir=str(out or "out"),
        separator="none",
        beats=beats,
        meter=meter,
        chroma_backend=chroma_backend,
        max_seconds=max_seconds,
        verbose=not quiet,
        _debug=debug,
    )

    def _analyse() -> dict:
        probe = audio_io.probe(input_audio)
        tracker = build_beat_tracker(beats)
        track = tracker.track(str(input_audio))
        beat_times = np.asarray(track.beat_times, dtype=float)
        mix, sr = audio_io.load(str(input_audio), mono=True, sample_rate=22050)
        acoustic = meter_stage.beat_synchronous_accent(mix, sr, beat_times)
        harmonic = meter_stage.harmonic_change_per_beat(mix, sr, beat_times, backend=chroma_backend)
        subdivision = meter_stage.subdivision_ratio(mix, sr, beat_times)

        if config.meter_parts is not None:
            numerator, denominator = config.meter_parts
            measured = meter_stage.MeterResult(
                hypotheses=[meter_stage.TimeSignature(numerator, denominator, 1.0, {})],
                backend="user",
            )
        else:
            measured = meter_stage.infer_meter(
                beat_times,
                accent=acoustic,
                downbeat_times=track.downbeat_times or None,
                harmonic_change=harmonic,
                subdivision=subdivision,
                allow_odd=config.allow_odd_meter,
            )
        downbeats, phase = meter_stage.derive_downbeats(
            beat_times,
            measured.best.numerator if measured.best else 4,
            phase_series=acoustic,
        )
        measured.phase = phase
        return {
            "input": str(input_audio),
            "duration": round(probe.duration, 3),
            "sample_rate": probe.sample_rate,
            "tempo_bpm": round(track.tempo_bpm, 3),
            "tempo_stability": round(track.tempo_stability, 4),
            "beats": len(track.beat_times),
            "beat_backend": track.backend,
            "meter": measured.to_dict(),
            "subdivision_ratio": round(subdivision, 4),
            "downbeats": len(downbeats),
            "warnings": list(track.warnings) + list(measured.warnings),
        }

    payload = _guard(_analyse, _debug=debug)
    if out:
        Path(out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        typer.secho(f"wrote {out}", fg=typer.colors.GREEN)
    else:
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


@app.command()
def stems(
    input_audio: Path = typer.Argument(..., exists=True, dir_okay=False),
    out: Path = typer.Option(Path("stems"), "-o", "--out"),
    separator: str = typer.Option("auto", "--separator"),
    demucs_model: str = typer.Option(
        "htdemucs_6s",
        "--demucs-model",
        help=f"One of {', '.join(DEMUCS_MODELS)}. '6s' adds piano and guitar; no model outputs strings.",
    ),
    device: str = typer.Option("auto", "--device"),
    quiet: bool = typer.Option(False, "-q", "--quiet"),
    debug: bool = typer.Option(False, "--debug"),
) -> None:
    """Separate the input into instrument stems and stop."""
    from .analysis.separate import build_separator, is_empty_stem
    from .io.console import set_verbose
    from .net import hf_unreachable_guard

    set_verbose(not quiet)

    def _separate():
        with hf_unreachable_guard():
            backend = build_separator(separator, model=demucs_model, device=device)
            return backend.separate(str(input_audio), out)

    stem_set = _guard(_separate, _debug=debug)
    typer.echo("")
    loudest = max(stem_set.level_db.values()) if stem_set.level_db else 0.0
    for name, path in stem_set.stems.items():
        level = stem_set.level_db.get(name, float("-inf"))
        empty = is_empty_stem(level, loudest)
        suffix = "   (effectively silent)" if empty else ""
        typer.echo(f"  {name:<10} {level:7.1f} dBFS  {Path(path).name}{suffix}")


@app.command(name="eval")
def eval_command(
    estimate: Path = typer.Option(..., "-e", "--estimate", exists=True, dir_okay=False),
    reference: Path = typer.Option(..., "-r", "--reference", exists=True, dir_okay=False),
    onset_tolerance: float = typer.Option(0.05, "--onset-tolerance", help="Seconds of onset slack."),
    offset_ratio: Optional[float] = typer.Option(
        0.2, "--offset-ratio", help="Note-offset slack as a ratio of duration; 0 disables."
    ),
    by_instrument: bool = typer.Option(False, "--by-instrument", help="Also score each program separately."),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
    debug: bool = typer.Option(False, "--debug"),
) -> None:
    """Score an estimate MIDI against a reference MIDI."""
    from .eval import evaluate, evaluate_by_instrument, format_report

    ratio = None if offset_ratio is not None and offset_ratio <= 0 else offset_ratio

    def _score():
        overall = evaluate(
            estimate, reference, onset_tolerance=onset_tolerance, offset_ratio=ratio
        )
        per_part = (
            evaluate_by_instrument(
                estimate, reference, onset_tolerance=onset_tolerance, offset_ratio=ratio
            )
            if by_instrument
            else None
        )
        return overall, per_part

    overall, per_part = _guard(_score, _debug=debug)
    if as_json:
        payload = overall.to_dict()
        if per_part:
            payload["by_instrument"] = {k: v.to_dict() for k, v in per_part.items()}
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo(format_report(overall, per_part, estimate=str(estimate), reference=str(reference)))
    if ratio is not None:
        typer.echo("")
        typer.echo(
            "note+offset F1 compares against performance-level durations, so a quantised\n"
            "score scores low even when every pitch and onset is right. Judge pitch with\n"
            "the onset line, or pass --offset-ratio 0."
        )


@app.command()
def synth(
    out: Path = typer.Option(Path("tests/data"), "-o", "--out"),
    tempo: float = typer.Option(100.0, "--tempo"),
    bars: int = typer.Option(8, "--bars"),
    meter: str = typer.Option("4/4", "--meter"),
    seed: int = typer.Option(7, "--seed"),
) -> None:
    """Generate a copyright-free synthetic ensemble with known ground truth."""
    from .synth import SynthSpec, make_ensemble

    if "/" not in meter:
        _fail("--meter expects e.g. 4/4")
    numerator, denominator = (int(part) for part in meter.split("/"))
    if denominator != 4:
        _fail("the generator only supports x/4 meters")

    spec = SynthSpec(tempo_bpm=tempo, bars=bars, beats_per_bar=numerator, seed=seed)
    paths = _guard(make_ensemble, out, spec)
    typer.secho(f"wrote a {bars}-bar {meter} fixture at {tempo:g} BPM", fg=typer.colors.GREEN)
    typer.echo(f"  mix:   {paths['mix']}")
    typer.echo(f"  stems: {paths['stems_dir']}")
    typer.echo(f"  truth: {paths['truth']}")


@app.command(name="neural")
def neural(
    input_audio: Path = typer.Argument(..., exists=True, dir_okay=False, help="Audio to transcribe."),
    out: Path = typer.Option(Path("out_neural"), "-o", "--out", help="Output directory."),
    repo: Path = typer.Option(
        Path("_sheetsage2"), "--repo", help="Checkout of SheetSage2Kern_model."
    ),
    checkpoint: Optional[str] = typer.Option(
        None, "--checkpoint", help="Local .ckpt, or 'org/repo/file.ckpt' to download."
    ),
    device: str = typer.Option("auto", "--device", help="auto / cpu / cuda."),
    chunk: float = typer.Option(
        22.0, "--chunk", help="Seconds per window. The model was trained on ~23 s clips."
    ),
    pdf: bool = typer.Option(False, "--pdf", help="Also render a PDF (needs MuseScore)."),
    quiet: bool = typer.Option(False, "-q", "--quiet"),
    debug: bool = typer.Option(False, "--debug"),
) -> None:
    """Transcribe with SheetSage2: audio in, a lead sheet out.

    This is the *neural* path and it is deliberately separate from ``e2s run``.  It
    produces a **lead sheet** — one melody line plus chord symbols, in the style its
    training data encodes — not a multi-instrument ensemble score.  Its rhythms are
    far better than the frame-based transcriber's on monophonic material, so the two
    are complementary rather than alternatives: use this for a melody or a bass line,
    and ``e2s run`` when you need several parts at once.

    Checkpoints are downloaded from Hugging Face on first use and cached.
    """
    import sys as _sys

    from .io.console import set_verbose
    from .notation.kern import kern_to_musicxml

    set_verbose(not quiet)
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts) not in _sys.path:
        _sys.path.insert(0, str(scripts))

    def _transcribe():
        from sheetsage2_infer import SheetSage2

        model = SheetSage2(repo, checkpoint=checkpoint, device=device)
        return model.transcribe_chunked(input_audio, seconds_per_chunk=chunk)

    output = _guard(_transcribe, _debug=debug)

    out.mkdir(parents=True, exist_ok=True)
    basename = input_audio.stem
    kern_path = out / f"{basename}.kern"
    kern_path.write_text(output.kern, encoding="utf-8")
    typer.echo(f"  kern: {kern_path}")

    def _convert():
        return kern_to_musicxml(output.kern, out / f"{basename}.musicxml", title=basename)

    parsed = _guard(_convert, _debug=debug)
    typer.echo(f"  musicxml: {out / f'{basename}.musicxml'}")
    typer.echo(
        f"  {parsed.time_signature or '?'}, {parsed.measures} measures, "
        f"{parsed.notes} notes, {parsed.chord_symbols} chord symbols"
        + (f", key {parsed.key}" if parsed.key else "")
    )
    for message in parsed.warnings[:5]:
        typer.secho(f"  ! {message}", fg=typer.colors.YELLOW)
    if output.truncated:
        typer.secho(
            "  ! a window hit the model's token limit; try a smaller --chunk",
            fg=typer.colors.YELLOW,
        )

    if pdf:
        from .notation.score import export_pdf

        written = export_pdf(out / f"{basename}.musicxml", out, basename, None)
        if written:
            typer.echo(f"  pdf: {written}")


@app.command(name="open")
def open_score(
    target: Path = typer.Argument(..., help="A run directory, or a .musicxml / .mid / .pdf file."),
    binary: Optional[str] = typer.Option(
        None, "--with", help="Program to open it with (defaults to MuseScore for scores)."
    ),
    audio: bool = typer.Option(False, "--audio", help="Open the MIDI instead, for listening."),
) -> None:
    """Open a score in an editor, viewer or player.

    A run directory usually holds a PDF, a MusicXML and a MIDI.  By default this
    opens the PDF, which is the fastest way to judge whether the transcription is
    right; ``--audio`` opens the MIDI instead so you can listen against the
    original.  MusicXML is handed to MuseScore explicitly, because Windows ships no
    association for the extension — double-clicking it does nothing.
    """
    import subprocess
    import sys

    resolved = _resolve_open_target(target, audio=audio)
    if resolved is None:
        _fail(f"nothing to open in {target}")

    path, kind = resolved
    program = binary or (_find_mscore() if kind == "musicxml" else None)

    try:
        if program:
            subprocess.Popen([program, str(path)])  # noqa: S603 - explicit user request
            typer.secho(f"opening {path.name} with {Path(program).name}", fg=typer.colors.GREEN)
        else:
            # Let the OS decide: this is what makes PDFs work on Windows.
            if sys.platform == "win32":
                import os

                os.startfile(str(path))  # noqa: S606 - deliberate shell association
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])  # noqa: S603, S607
            else:
                subprocess.Popen(["xdg-open", str(path)])  # noqa: S603, S607
            typer.secho(f"opening {path.name}", fg=typer.colors.GREEN)
    except OSError as exc:
        _fail(f"could not open {path.name}: {exc}")

    if kind == "musicxml" and program is None:
        typer.secho(
            "  note: MuseScore was not found, so the OS default is used. Windows has no "
            "MusicXML association by default; install MuseScore 4 or pass --with <program>.",
            fg=typer.colors.YELLOW,
        )


# Preference order for a run directory: something to look at, then to listen to,
# then to read.
_LOOK_ORDER: tuple[tuple[str, str], ...] = (
    ("*.pdf", "pdf"),
    ("*.musicxml", "musicxml"),
    ("report.md", "report"),
)
_LISTEN_ORDER: tuple[tuple[str, str], ...] = (
    ("*.mid", "midi"),
    ("*.musicxml", "musicxml"),
)


def _resolve_open_target(target: Path, *, audio: bool = False) -> tuple[Path, str] | None:
    """Work out which file in a run (or a bare path) to open."""
    if target.is_file():
        suffix = target.suffix.lower()
        kind = {
            ".pdf": "pdf",
            ".musicxml": "musicxml",
            ".mxl": "musicxml",
            ".mid": "midi",
            ".midi": "midi",
            ".md": "report",
        }.get(suffix, "file")
        return target, kind

    if not target.is_dir():
        return None

    for pattern, kind in (_LISTEN_ORDER if audio else _LOOK_ORDER):
        matches = sorted(target.glob(pattern))
        if matches:
            return matches[0], kind
    return None


def main() -> None:
    """Console entry point."""
    configure_numba()
    app()


if __name__ == "__main__":  # pragma: no cover - module execution
    main()
