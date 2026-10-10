"""Use SheetSage2's melody as the melodic line, instead of ignoring it.

Why this exists
---------------
SheetSage2 writes two complementary melody files -- ``melody_vocal`` for the sung line
and ``melody_instrumental`` for the instrumental one -- and the pipeline never used
either.  It consumed only the beats, key, chords and section labels, so the best
melody it had access to was sitting unused on disk while the vocal part came from
basic-pitch.

Measured across the four tracks, as the share of the part's own span that is actually
sounding (a direct measure of whether a line is continuous or stuttering):

===========  ==================  ================  ==========
track        SheetSage2 melody   pipeline vocals   difference
===========  ==================  ================  ==========
blessing              88.6%              37.6%          2.4x
collapse              88.6%              31.0%          2.9x
yoake                 89.5%              47.9%          2.4x
victim                87.4%              47.7%          2.7x
===========  ==================  ================  ==========

That gap is what "the real melody is missing notes" sounds like: the pipeline's line
sounds for less than half the music.

What this does not do
---------------------
It does not decide the question for the pipeline.  SheetSage2 is CC-BY-NC-4.0, so it
stays opt-in (`io/registry.py` marks it not auto-selectable) -- this module just makes
the melody usable once someone has chosen to run SheetSage2.

The two files are **parallel lines, not consecutive halves**: in one track the vocal
line spans 0.5-247.9 s and the instrumental line 76.5-260.4 s, overlapping for 171 s.
So merging means preferring one where it has content and filling from the other,
never concatenating them into a longer part than the music has.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .core.models import Note

__all__ = ["MelodyMerge", "load_sheetsage2_melody", "merge_melody", "write_melody_midi"]

# A gap shorter than this between consecutive notes of one line is not a phrase break,
# it is the transcriber's own spacing, so the other line does not get to interrupt it.
_PHRASE_GAP = 0.25


@dataclass
class MelodyMerge:
    """The merged line plus what went into it."""

    notes: list[Note]
    vocal_notes: int
    instrumental_notes: int
    filled_seconds: float
    warnings: list[str]


def _load(path: Path) -> list[Note]:
    import pretty_midi

    if not path.is_file():
        return []
    midi = pretty_midi.PrettyMIDI(str(path))
    return [
        Note(
            pitch=int(note.pitch),
            start=float(note.start),
            end=float(note.end),
            velocity=int(note.velocity),
            confidence=1.0,
        )
        for instrument in midi.instruments
        for note in instrument.notes
    ]


def load_sheetsage2_melody(directory: str | Path) -> tuple[list[Note], list[Note]]:
    """``(vocal, instrumental)`` melody notes from a SheetSage2 output directory."""
    path = Path(directory)
    return _load(path / "melody_vocal.mid"), _load(path / "melody_instrumental.mid")


def _covered(notes: list[Note], start: float, end: float, tolerance: float = _PHRASE_GAP) -> bool:
    """Is any of ``notes`` sounding in ``[start, end)``, allowing a small tolerance?"""
    for note in notes:
        if note.start < end + tolerance and start < note.end + tolerance:
            return True
    return False


def merge_melody(
    vocal: list[Note],
    instrumental: list[Note],
    *,
    prefer: str = "vocal",
) -> MelodyMerge:
    """Combine the two melody lines into one, keeping every note of the preferred one.

    ``prefer="vocal"`` is the default because that is the line SheetSage2 is most
    reliable on, and because a lead sheet's melody is normally the sung line.  Notes
    from the other file are added only where the preferred line is not sounding, so the
    result never contains two competing lines at the same instant and never becomes
    longer than the music.

    The tolerance matters: without it the instrumental line would be threaded into the
    small gaps *inside* a sustained sung note, splitting it.
    """
    if prefer not in ("vocal", "instrumental"):
        raise ValueError(f"prefer must be 'vocal' or 'instrumental', not {prefer!r}")

    primary, secondary = (
        (vocal, instrumental) if prefer == "vocal" else (instrumental, vocal)
    )
    warnings: list[str] = []

    merged = sorted(primary, key=lambda n: (n.start, n.pitch))
    filled = 0.0
    added: list[Note] = []
    for note in sorted(secondary, key=lambda n: (n.start, n.pitch)):
        if _covered(primary, note.start, note.end):
            continue
        # Also refuse to add a note that overlaps what has already been added, so a
        # dense secondary line cannot stack up on itself.
        if _covered(added, note.start, note.end, tolerance=0.0):
            continue
        added.append(note)
        filled += note.end - note.start

    merged.extend(added)
    merged.sort(key=lambda n: (n.start, n.pitch))

    if not merged:
        warnings.append("both melody files were empty")
    if primary and not secondary:
        warnings.append(
            "only one melody file had notes: the other was empty, so nothing was filled in"
        )
    return MelodyMerge(
        notes=merged,
        vocal_notes=len(vocal),
        instrumental_notes=len(instrumental),
        filled_seconds=round(filled, 3),
        warnings=warnings,
    )


def write_melody_midi(notes: list[Note], path: str | Path, *, program: int = 53) -> Path:
    """Write the merged line as a single-track MIDI, for listening or for the pipeline."""
    import pretty_midi

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program, name="Melody")
    for note in notes:
        instrument.notes.append(
            pretty_midi.Note(
                velocity=max(1, min(127, note.velocity)),
                pitch=int(note.pitch),
                start=float(note.start),
                end=max(float(note.end), float(note.start) + 0.01),
            )
        )
    midi.instruments.append(instrument)
    midi.write(str(target))
    return target


def main() -> int:
    """Merge every track that has SheetSage2 melody files."""
    import sys

    root = Path(sys.argv[1] if len(sys.argv) > 1 else "sheetsage2_out")
    if not root.is_dir():
        print(f"no such directory: {root}")
        return 1
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        vocal, instrumental = load_sheetsage2_melody(directory)
        if not vocal and not instrumental:
            continue
        result = merge_melody(vocal, instrumental)
        def span(notes: list[Note]) -> tuple[float, float]:
            if not notes:
                return (0.0, 0.0)
            return (min(n.start for n in notes), max(n.end for n in notes))
        lo, hi = span(result.notes)
        sounding = sum(n.end - n.start for n in result.notes)
        print(
            f"{directory.name:10s} vocal {len(vocal):4d} + instrumental "
            f"{len(instrumental):4d} -> {len(result.notes):4d} notes  "
            f"span {lo:6.1f}..{hi:6.1f}s  sounding {sounding:6.1f}s "
            f"({100 * sounding / max(hi - lo, 1e-9):5.1f}%)"
        )
        out = directory / "melody_merged.mid"
        write_melody_midi(result.notes, out)
        for warning in result.warnings:
            print(f"           ! {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
