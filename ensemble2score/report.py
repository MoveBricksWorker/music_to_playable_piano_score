"""The human-readable run report.

``report.md`` is the part of the output a person actually reads, so it is written
to answer three questions in order: what did the tool decide, why did it decide
that, and which parts should I not trust?
"""

from __future__ import annotations

from pathlib import Path

from .core.models import RunResult

__all__ = ["write_report", "render_report"]


def render_report(result: RunResult, *, generator: str = "ensemble2score") -> str:
    lines: list[str] = []
    add = lines.append

    title = Path(result.input_path).stem
    add(f"# {generator} report — {title}")
    add("")
    add(f"- input: `{result.input_path}`")
    add(f"- duration: {result.duration:.1f}s" + (f" @ {result.sample_rate} Hz" if result.sample_rate else ""))
    if result.beats:
        add(
            f"- tempo: **{result.beats.tempo_bpm:.1f} BPM** "
            f"(stability {result.beats.tempo_stability:.2f}, backend `{result.beats.backend}`)"
        )
        add(f"- beats: {len(result.beats.beat_times)}, downbeats: {len(result.beats.downbeat_times)}")
    add(f"- time signature: **{result.time_signature}**")
    add(f"- measures: {result.measures}")
    if result.key:
        add(f"- key estimate: {result.key}")
    add("")

    _rhythm(result, add)
    _separation(result, add)
    _parts(result, add)
    _chords(result, add)
    _timings(result, add)
    _warnings(result, add)
    _files(result, add)

    return "\n".join(lines).rstrip() + "\n"


def _rhythm(result: RunResult, add) -> None:
    if not result.meter:
        return
    add("## Rhythm")
    add("")
    add(
        "Cues are fused into ranked hypotheses; the top one is used for notation. "
        "`confidence` is a softmax over the fused scores, so it compares hypotheses "
        "within this run only."
    )
    add("")
    add("| rank | time signature | confidence | " + " | ".join(
        ["accent", "downbeat", "harmonic", "subdivision"]
    ) + " |")
    add("|---|---|---|" + "---|" * 4)
    for rank, hypothesis in enumerate(result.meter.hypotheses, start=1):
        evidence = hypothesis.evidence
        add(
            f"| {rank} | {hypothesis.label} | {hypothesis.confidence:.2f} | "
            f"{evidence.get('accent', 0.0):.2f} | {evidence.get('downbeat', 0.0):.2f} | "
            f"{evidence.get('harmonic', 0.0):.2f} | {evidence.get('subdivision', 0.0):.2f} |"
        )
    add("")
    add(f"- bar phase (the beat index that starts a bar): {result.meter.phase}")
    if result.meter.backend == "user":
        add("- the time signature was supplied by the user, not inferred")
    add("")
    add(
        "> Bar phase is the single most damaging failure mode: if it is wrong, every "
        "barline is shifted and the whole score is one beat out. If the first barline "
        "looks wrong, re-run with `--meter N/D`."
    )
    add("")


def _separation(result: RunResult, add) -> None:
    if not result.stems:
        return
    add("## Separation")
    add("")
    add(f"- backend: `{result.stems.backend}`")
    add("")
    if result.stems.level_db:
        add("| stem | level (dBFS) | file |")
        add("|---|---|---|")
        for name, level in sorted(result.stems.level_db.items(), key=lambda kv: -kv[1]):
            path = Path(result.stems.stems.get(name, "")).name
            add(f"| {name} | {level:.1f} | `{path}` |")
        add("")
    add(
        "Stems well below the loudest one are bleed from instruments the song does not "
        "contain. Transcribing that bleed would invent a part, so such stems are skipped."
    )
    add("")


def _parts(result: RunResult, add) -> None:
    add("## Transcribed parts")
    add("")
    if not result.parts and not result.drum_hits:
        add("_No parts were transcribed._")
        add("")
        return

    if result.parts:
        add("| part | stem | instrument | backend | notes | pitch range |")
        add("|---|---|---|---|---|---|")
        for part in result.parts:
            rng = part.pitch_range
            add(
                f"| {part.name} | {part.stem} | {part.spec.instrument} | `{part.backend}` | "
                f"{part.n_notes} | {f'{rng[0]}–{rng[1]}' if rng else '—'} |"
            )
        add("")

    if result.drum_hits:
        counts: dict[str, int] = {}
        for hit in result.drum_hits:
            counts[hit.voice] = counts.get(hit.voice, 0) + 1
        summary = ", ".join(f"{voice} {counts.get(voice, 0)}" for voice in ("kick", "snare", "hihat"))
        add(f"Percussion onsets (classified, not transcribed): {summary}")
        add("")

    unreliable = [p for p in result.parts if p.n_notes == 0]
    if unreliable:
        add("Parts with no notes: " + ", ".join(p.stem for p in unreliable))
        add("")


def _chords(result: RunResult, add) -> None:
    if not result.chords:
        return
    add("## Chords (advisory)")
    add("")
    add(
        "Template-matched from beat-synchronous chroma. Expect mistakes on dense mixes; "
        "treat these as a sketch."
    )
    add("")
    preview = " ".join(chord.label for chord in result.chords[:24])
    suffix = " …" if len(result.chords) > 24 else ""
    add(f"`{preview}`{suffix}")
    add("")


def _timings(result: RunResult, add) -> None:
    if not result.stages:
        return
    add("## Stage timings")
    add("")
    add("| stage | seconds | source | backend |")
    add("|---|---|---|---|")
    for stage in result.stages:
        origin = "cache" if stage.cached else "computed"
        add(f"| {stage.stage} | {stage.seconds:.2f} | {origin} | `{stage.backend}` |")
    add("")
    add(f"Total: {result.total_seconds:.1f}s")
    add("")


def _warnings(result: RunResult, add) -> None:
    add("## Warnings and things to check")
    add("")
    if result.warnings:
        for message in result.warnings:
            add(f"- {message}")
    else:
        add("- none")
    add("")
    add("### Suggested checks")
    add("")
    add("1. Open the MusicXML and confirm the time signature and tempo.")
    add(
        "2. Check the first barline. If it is a beat out, re-run with `--meter N/D`; "
        "bar phase comes from accent analysis and can land one beat late."
    )
    add("3. Listen to the MIDI against the original, stem by stem. The least reliable "
        "part is whichever backend is listed with the fewest notes.")
    add("4. Anything with low meter confidence, or a `hpss` separation, is a draft.")
    add("")


def _files(result: RunResult, add) -> None:
    if not result.outputs:
        return
    add("## Files")
    add("")
    for name, path in result.outputs.items():
        add(f"- {name}: `{path}`")
    add("")


def write_report(result: RunResult, out_dir: str | Path, *, generator: str = "ensemble2score 2.0.0") -> Path:
    """Render and write ``report.md``; returns its path."""
    target = Path(out_dir) / "report.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_report(result, generator=generator), encoding="utf-8")
    return target
