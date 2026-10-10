"""Licence registry for every third-party component this project can call.

This table is the single source of truth: ``THIRD_PARTY_LICENSES.md`` and
``e2s doctor`` are both generated from it, and CI checks that the generated file
has not drifted.

The policy it encodes:

* the core and every backend that ``auto`` may pick are MIT / BSD / Apache / ISC;
* copyleft or unclear-licence components are registered, are **never** chosen by
  ``auto``, and warn when selected explicitly;
* **model weights are a separate question from code**.  We ship none and do not
  download from sources whose licensing is unclear.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Component", "REGISTRY", "by_kind", "find", "audit_lines", "PERMISSIVE"]

PERMISSIVE = "permissive"
COPYLEFT = "copyleft"
UNCLEAR = "unclear"


@dataclass(frozen=True)
class Component:
    name: str
    kind: str            # runtime | separation | beats | transcription | harmony | notation
    licence: str
    licence_class: str   # permissive | copyleft | unclear
    url: str
    note: str = ""
    auto_selectable: bool = True
    weights: str = "none shipped"

    @property
    def requires_warning(self) -> bool:
        return self.licence_class != PERMISSIVE or not self.auto_selectable


REGISTRY: tuple[Component, ...] = (
    # ---- core runtime ----------------------------------------------------
    Component("numpy", "runtime", "BSD-3-Clause", PERMISSIVE, "https://numpy.org"),
    Component("scipy", "runtime", "BSD-3-Clause", PERMISSIVE, "https://scipy.org"),
    Component("librosa", "runtime", "ISC", PERMISSIVE, "https://librosa.org"),
    Component("soundfile", "runtime", "BSD-3-Clause", PERMISSIVE, "https://python-soundfile.readthedocs.io"),
    Component("PyYAML", "runtime", "MIT", PERMISSIVE, "https://pyyaml.org"),
    Component("typer", "runtime", "MIT", PERMISSIVE, "https://typer.tiangolo.com"),
    Component("rich", "runtime", "MIT", PERMISSIVE, "https://rich.readthedocs.io"),
    Component("numba", "runtime", "BSD-2-Clause", PERMISSIVE, "https://numba.pydata.org",
              note="optional librosa accelerator; first call compiles LLVM and is slow"),
    # ---- separation ------------------------------------------------------
    Component("demucs", "separation", "MIT", PERMISSIVE, "https://github.com/facebookresearch/demucs",
              weights="htdemucs family downloaded on first use from dl.fbaipublicfiles.com"),
    Component("librosa (hpss)", "separation", "ISC", PERMISSIVE, "https://librosa.org",
              note="harmonic/percussive split; not instrument-aware, quality is much lower"),
    # ---- beats -----------------------------------------------------------
    Component("beat-this", "beats", "MIT", PERMISSIVE, "https://github.com/CPJKU/beat_this",
              weights="checkpoint fetched on first use; the public mirror has been unreliable"),
    Component("librosa (beat)", "beats", "ISC", PERMISSIVE, "https://librosa.org"),
    # ---- transcription ---------------------------------------------------
    Component("basic-pitch", "transcription", "Apache-2.0", PERMISSIVE,
              "https://github.com/spotify/basic-pitch",
              weights="bundled with the package (ICASSP 2022 model)"),
    Component("librosa (pyin)", "transcription", "ISC", PERMISSIVE, "https://librosa.org",
              note="monophonic only; used as a fallback and for the octave sanity check"),
    # ---- notation --------------------------------------------------------
    Component("music21", "notation", "BSD-3-Clause", PERMISSIVE, "https://web.mit.edu/music21/"),
    Component("pretty_midi", "notation", "MIT", PERMISSIVE, "https://github.com/craffel/pretty-midi"),
    Component("MuseScore", "notation", "GPL-3.0", COPYLEFT, "https://musescore.org",
              note="only invoked for --pdf, as an external program; not linked or shipped",
              auto_selectable=False),
    # ---- opt-in only -----------------------------------------------------
    Component("yourmt3", "transcription", "GPL-3.0", COPYLEFT, "https://github.com/mimbres/YourMT3",
              note="copyleft; never chosen by auto", auto_selectable=False,
              weights="Amt 2024 checkpoint from Hugging Face"),
    Component("madmom", "beats", "BSD-3-Clause (code)", UNCLEAR,
              "https://github.com/CPJKU/madmom",
              note="code is BSD but the published models are CC BY-NC-SA (non-commercial)",
              auto_selectable=False),
    Component("piano_transcription_inference", "transcription", "no LICENSE upstream", UNCLEAR,
              "https://github.com/qiuqiangkong/piano_transcription_inference",
              note="no licence file in the upstream repository", auto_selectable=False),
    Component("transkun", "transcription", "MIT", PERMISSIVE,
              "https://github.com/Yujia-Yan/Transkun",
              note="wiring is experimental and untested in this environment",
              auto_selectable=False,
              weights="not fetched; requires a separate checkpoint download"),
    Component("SheetSage2", "harmony", "CC-BY-NC-4.0", UNCLEAR,
              "https://huggingface.co/m-a-p/SheetSage2",
              note="lead-sheet transcription (key, chords, beats, sections) used as a "
                   "prior via --ss2-prior; non-commercial, so never chosen by auto",
              auto_selectable=False,
              weights="model.safetensors (233 MB) plus its MERT-v2-FullSong parent "
                     "(2.5 GB), both fetched on first use from the Hugging Face Hub"),
    Component("SheetSage2 (per-stem transcription)", "transcription", "CC-BY-NC-4.0",
              UNCLEAR, "https://huggingface.co/m-a-p/SheetSage2",
              note="--transcriber sheetsage2: transcribes each separated stem; needs its "
                   "own interpreter because the released code requires transformers 4.45 "
                   "(see analysis/sheetsage2_backend.py); non-commercial",
              auto_selectable=False,
              weights="the same SheetSage2 + MERT-v2-FullSong weights as the prior path"),
)


def by_kind(kind: str) -> list[Component]:
    return [c for c in REGISTRY if c.kind == kind]


def find(name: str) -> Component | None:
    lowered = name.lower()
    for component in REGISTRY:
        if component.name.lower() == lowered:
            return component
    for component in REGISTRY:
        if lowered in component.name.lower():
            return component
    return None


def audit_lines() -> list[str]:
    """Render the markdown table used by THIRD_PARTY_LICENSES.md and CI."""
    lines = [
        "| component | role | licence | class | auto-selectable | weights | link |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in sorted(REGISTRY, key=lambda c: (c.kind, c.name.lower())):
        auto = "yes" if c.auto_selectable else "**no**"
        note = f" — {c.note}" if c.note else ""
        lines.append(
            f"| {c.name}{note} | {c.kind} | {c.licence} | {c.licence_class} | {auto} | "
            f"{c.weights} | <{c.url}> |"
        )
    return lines
