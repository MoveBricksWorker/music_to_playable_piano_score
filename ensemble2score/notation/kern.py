"""``**kern`` (Humdrum) to MusicXML, via music21.

Why not write a parser
----------------------
music21 already reads Humdrum natively, and it does it correctly on the output of
SheetSage2: measure numbers, ``4/4``, the ``E- major`` key signature and the note
values all survive.  Re-implementing that grammar would be a large surface for
bugs with no benefit.

What music21 does *not* do is turn the ``**cdata`` spine into chord symbols on its
own in a way we can rely on, so chord labels are recovered from the raw kern text
and attached to the measure they belong to.  That is the only part written here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..core.errors import OutputError

__all__ = ["KernScore", "parse_kern", "kern_to_musicxml", "kern_measures", "KERN_SPINE"]


KERN_SPINE = "**kern"


@dataclass
class KernScore:
    """A parsed ``**kern`` score plus the facts we can state about it."""

    score: object
    measures: int = 0
    notes: int = 0
    chord_symbols: int = 0
    time_signature: str | None = None
    key: str | None = None
    warnings: list[str] = field(default_factory=list)


def kern_measures(text: str) -> list[list[str]]:
    """Split kern text into measures, each a list of data rows.

    A measure starts at a barline row (``=``, ``=12``, ``=-``) and runs to the next
    one.  Header rows (``*``) and comments (``!``) are kept with the measure that
    follows them, because a time-signature change has to be applied to the right bar.
    """
    measures: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("=") and not stripped.startswith("=="):
            if current:
                measures.append(current)
            current = [line]
            continue
        if stripped.startswith("=="):
            if current:
                measures.append(current)
            current = []
            continue
        current.append(line)
    if current:
        measures.append(current)
    return measures


def _chord_labels(text: str) -> list[str]:
    """Chord label per measure, taken from the ``**cdata`` spine.

    The tokeniser emits a chord label only when the harmony changes, using ``.`` to
    mean "same as before", so the labels are carried forward across measures and
    the first non-``.`` value in each measure is what starts there.
    """
    labels: list[str] = []
    for rows in kern_measures(text):
        found = None
        for row in rows:
            fields = row.split("\t")
            if len(fields) < 2:
                continue
            candidate = fields[-1].strip()
            if candidate in ("", ".", "=="):
                continue
            if candidate.startswith("*") or candidate.startswith("="):
                continue
            found = candidate
            break
        labels.append(found or "")
    return labels


def parse_kern(text: str, *, source: str = "") -> KernScore:
    """Parse ``**kern`` text into a music21 score.

    Raises :class:`OutputError` when music21 cannot make sense of the text — a
    caller that wrote the kern file itself should be told, not handed ``None``.
    """
    from music21 import converter, meter, tempo
    from music21 import key as m21key

    if KERN_SPINE not in text:
        raise OutputError("kern text has no **kern spine; nothing to parse")

    try:
        score = converter.parse(text, format="humdrum")
    except Exception as exc:  # noqa: BLE001 - music21 raises a wide range here
        raise OutputError(f"music21 could not parse the kern score: {exc}") from exc

    result = KernScore(score=score)

    # Attach chord symbols by measure, using the raw text (music21 does not expose
    # the **cdata spine as ChordSymbols reliably).
    labels = _chord_labels(text)
    for part in score.parts:
        measures = list(part.getElementsByClass("Measure"))
        result.measures = max(result.measures, len(measures))
        for index, measure in enumerate(measures):
            if index < len(labels) and labels[index]:
                try:
                    measure.insert(0.0, _chord_symbol(labels[index], m21key))
                    result.chord_symbols += 1
                except Exception as exc:  # noqa: BLE001 - one odd label is not fatal
                    result.warnings.append(f"chord {labels[index]!r} not notatable: {exc}")
        break  # the cdata spine belongs to the top part

    for part in score.parts:
        result.notes += len(list(part.recurse().notes))
        first = list(part.getElementsByClass("Measure"))
        for measure in first:
            for signature in measure.getElementsByClass(meter.TimeSignature):
                result.time_signature = signature.ratioString
                break
            found_key = list(measure.getElementsByClass(m21key.KeySignature))
            if found_key:
                result.key = str(found_key[0])
            tempo_marks = list(measure.getElementsByClass(tempo.MetronomeMark))
            if result.time_signature and result.key:
                break
        if result.time_signature:
            break

    return result


def _normalise_root(root: str) -> str:
    """Spell a chord root the way music21 expects.

    Two flat conventions are in play and both have to be accepted.  Humdrum writes
    ``E-``; SheetSage2's ``**cdata`` spine writes ``Eb``.  music21 understands only
    the former, and rejects ``Eb`` outright because it parses the ``b`` as a chord
    abbreviation — so a chord written the other way is silently lost unless it is
    converted here.
    """
    if len(root) >= 2 and root[1] in ("b", "-"):
        return root[0].upper() + "-" + root[2:]
    return root


def _chord_symbol(label: str, m21key):
    """Build a music21 ChordSymbol from a kern chord label.

    Three spellings have to be reconciled, which is the whole difficulty:

    * kern writes flats as ``-`` (``E-``), sharps as ``#``.  music21 uses exactly
      the same convention, so flats are passed through and must **not** be rewritten
      to ``b``: ``harmony.ChordSymbol("Eb")`` raises, because ``b`` is parsed as a
      chord abbreviation.  That mistake silently dropped every flat chord.
    * Harte syntax, which kern uses here, writes the bass after ``/`` and separates
      the quality with ``:`` (``E-:maj``, ``C:min7/G``).
    * music21 wants its own quality abbreviations (``m`` rather than ``min``).

    A label it cannot read is reported by the caller rather than silently dropped.
    """
    from music21 import harmony

    cleaned = label.strip()
    if not cleaned or cleaned in ("N.C.", "N"):
        return None

    # Strip a parenthesised extension list: music21 validates abbreviations strictly
    # and these add nothing to a chord symbol's readability.
    cleaned = cleaned.split("(")[0]

    root, _, rest = cleaned.partition(":")
    if not root or not (root[0].isalpha() and root[0].upper() in "ABCDEFG"):
        return None
    root = _normalise_root(root)

    quality = rest.split("/")[0] if rest else ""
    bass = rest.split("/")[1] if "/" in rest else None

    music21_quality = {
        "maj": "",
        "min": "m",
        "dim": "dim",
        "aug": "aug",
        "hdim7": "m7b5",
        "min7": "m7",
        "maj7": "maj7",
        "dom7": "7",
    }.get(quality, quality)

    text = f"{root}{music21_quality}"
    if bass and bass not in ("none", ""):
        text += f"/{bass}"
    return harmony.ChordSymbol(text)


def kern_to_musicxml(
    kern_text: str,
    out_path: str | Path,
    *,
    title: str | None = None,
) -> KernScore:
    """Convert ``**kern`` text to a MusicXML file, returning what was written."""
    result = parse_kern(kern_text)
    if title:
        from music21 import metadata

        if result.score.metadata is None:
            result.score.metadata = metadata.Metadata()
        result.score.metadata.title = title

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        result.score.write("musicxml", fp=str(target))
    except Exception as exc:  # noqa: BLE001
        raise OutputError(f"could not write MusicXML: {exc}") from exc
    return result
