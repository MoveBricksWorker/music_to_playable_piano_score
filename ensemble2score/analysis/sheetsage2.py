"""SheetSage2 as a *prior* for this pipeline, not as a replacement for it.

Why this module exists
----------------------
SheetSage2 (``m-a-p/SheetSage2``) transcribes audio to a **lead sheet**: melody,
chords, beats, key and structure.  This project produces a **multi-part ensemble
score**, which is a different artefact — explicit barlines, pickup measures,
cross-bar ties, one part per separated stem, a grand staff.  So the useful thing is
not to swap one for the other but to feed SheetSage2's symbolic reading into the
stages where this pipeline is weakest, which are the ones with no direct acoustic
evidence: the bar phase, the key, and above all the chords (template matching over
chroma is the least reliable part of the chain).

What it contributes, measured
-----------------------------
On 夜明けの歌 both systems were run independently and agree where it matters:

* key — SheetSage2 says ``Eb:major``; this pipeline's own estimate is E♭ major.
* tempo — 89.6 BPM against 90.9 BPM.
* bar length — **they disagree, informatively**.  SheetSage2 emits ``M:2/4`` with
  exactly 2 beats between downbeats, but its chord sequence closes every *four*
  chords (``Cm7 → Abmaj7 → Bb7 → Eb``), i.e. every four beats.  Its beat grid and
  downbeat *phase* are right; its bar is half a bar.  Fusing two trackers is
  therefore better than trusting either, which is what ``second_downbeats`` does.

Licence
-------
SheetSage2 and its MERT-v2 parent are **CC-BY-NC-4.0** — non-commercial.  Following
this project's existing policy for copyleft and unclear components (see
``io/registry.py``: YourMT3, madmom), it is **never chosen by ``auto``** and is used
only when a path is passed explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ..core.models import Chord

__all__ = [
    "SheetSage2Prior",
    "load_prior",
    "looks_like_sheetsage2_output",
    "format_chord_label",
    "beat_times_from_prior",
]

# SheetSage2 writes lead-sheet labels in a Harte-like ``root:quality`` form.
# music21 parses the abbreviations this project already uses ("Cm7", "Ebmaj7"), so
# the labels are converted rather than passed through and hoped for.
_QUALITY_SUFFIX: dict[str, str] = {
    "maj": "",
    "min": "m",
    "maj7": "maj7",
    "min7": "m7",
    "7": "7",
    "dim": "dim",
    "hdim7": "m7b5",
    "aug": "aug",
    "sus4": "sus4",
    "sus2": "sus2",
    "min6": "m6",
    "maj6": "6",
    "min9": "m9",
    "maj9": "maj9",
    "9": "9",
    "11": "11",
    "13": "13",
    "min11": "m11",
    "min13": "m13",
}

# A prior is only worth loading if it has the things that carry information.
_MIN_BEATS = 4


@dataclass
class SheetSage2Prior:
    """The symbolic reading SheetSage2 produced for one track.

    Times are seconds on the original timeline, exactly like every other stage in
    this project, so a prior can be dropped in without rescaling anything.
    """

    directory: Path
    beat_times: list[float] = field(default_factory=list)
    downbeat_times: list[float] = field(default_factory=list)
    key: str | None = None                 # e.g. "Eb:major"
    chords: list[Chord] = field(default_factory=list)
    structure: list[tuple[float, float, str]] = field(default_factory=list)
    meter: tuple[int, int] | None = None   # SheetSage2's own reading, often halved
    # ``(start_time, numerator, denominator, beat_in_bar)`` straight out of ``beat.lab``,
    # one entry per change.  The fourth value is the beat of the bar the first beat of that
    # span sits on, which is what makes the bar phase a reading rather than a derivation.
    # Kept alongside the single ``meter`` above because that field is the first value from
    # ``events.json`` and throws away everything after it: it cannot express `collapse`
    # moving to 5/8, and it cannot say when anything changed.
    meter_spans: list[tuple[float, int, int, int]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def meter_at(self, when: float) -> tuple[int, int] | None:
        """The time signature SheetSage2 reported at ``when`` (seconds).

        Falls back to the single ``meter`` value when no spans were read, so a prior
        built from ``events.json`` alone still answers.
        """
        best: tuple[int, int] | None = None
        for span in self.meter_spans:
            if span[0] <= when:
                best = (span[1], span[2])
            else:
                break
        if best is not None:
            return best
        if self.meter_spans:
            # before the first span: use the earliest reading rather than nothing
            return (self.meter_spans[0][1], self.meter_spans[0][2])
        return self.meter

    @property
    def ok(self) -> bool:
        return len(self.beat_times) >= _MIN_BEATS

    @property
    def key_signature_name(self) -> str | None:
        """``"Eb:major"`` -> ``"E- major"``, which is music21's spelling."""
        if not self.key or ":" not in self.key:
            return None
        tonic, mode = self.key.split(":", 1)
        if not tonic:
            return None
        spelled = tonic[0].upper() + tonic[1:]
        if len(spelled) > 1 and spelled[1] in "b#":
            spelled = f"{spelled[0]}{'-' if spelled[1] == 'b' else '#'}{spelled[2:]}"
        return f"{spelled} {mode}"

    def to_dict(self) -> dict[str, object]:
        return {
            "directory": str(self.directory),
            "beat_times": len(self.beat_times),
            "downbeat_times": len(self.downbeat_times),
            "key": self.key,
            "n_chords": len(self.chords),
            "structure": list(self.structure),
            "meter": list(self.meter) if self.meter else None,
            "meter_spans": [list(span) for span in self.meter_spans],
            "warnings": list(self.warnings),
        }


def looks_like_sheetsage2_output(directory: str | Path) -> bool:
    """Does this directory hold a SheetSage2 run?"""
    path = Path(directory)
    if not path.is_dir():
        return False
    return any(
        (path / name).is_file()
        for name in ("beat.lab", "downbeat.lab", "events.json", "chord.lab")
    )


def _read_lab(path: Path) -> list[tuple[float, float, str]]:
    """Read a ``start<TAB>end<TAB>label`` file, skipping anything malformed."""
    rows: list[tuple[float, float, str]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        try:
            start = float(parts[0])
            end = float(parts[1])
        except ValueError:
            continue
        label = parts[2].strip() if len(parts) > 2 else ""
        if end < start:
            start, end = end, start
        rows.append((start, end, label))
    return rows


def _read_times(path: Path) -> list[float]:
    """Read a single-column time file (``downbeat.lab``), which has tabs appended."""
    times: list[float] = []
    if not path.is_file():
        return times
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        field = line.split("\t")[0].strip()
        if not field:
            continue
        try:
            times.append(float(field))
        except ValueError:
            continue
    return sorted(times)


def _read_beat_times(path: Path) -> list[float]:
    """``beat.lab`` is ``time<TAB>beat_in_bar<TAB>beats_per_bar<TAB>…``."""
    times: list[float] = []
    if not path.is_file():
        return times
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        field = line.split("\t")[0].strip()
        if not field:
            continue
        try:
            times.append(float(field))
        except ValueError:
            continue
    return sorted(times)


def _read_meter_spans(
    path: Path,
) -> tuple[list[tuple[float, int, int, int]], list[str]]:
    """Read the per-beat time signature out of ``beat.lab``.

    ``beat.lab`` is ``time<TAB>beat_in_bar<TAB>numerator<TAB>denominator``, so the file
    this pipeline already reads for beat times also carries SheetSage2's own time
    signature **once per beat**, plus which beat of the bar each row is.  Both are worth
    having: the signature is a reading rather than an inference, and ``beat_in_bar`` gives
    the bar phase directly instead of requiring it to be derived.

    Measured on the four songs in this project:

        victim    2/4 for all 299 beats, beat_in_bar cycling [1, 2]
        blessing  3/4 for all 636 beats, beat_in_bar cycling [1, 2, 3]
        yoake     4/4 for 293 beats, 3/8 for the first 3
        collapse  4/4 for 342 beats, 5/8 for 5 beats around 25.5-27.4 s

    Returns ``(spans, warnings)`` where each span is ``(start_time, numerator,
    denominator, beat_in_bar)`` covering beats from ``start_time`` until the next span.  A
    single span means the signature never changed.

    ``beat_in_bar`` is the **most common** value in the span, not the first.  The first is
    not reliable: every one of these files opens with a malformed pair -- victim starts
    ``[1, 1, 2, 1, ...]`` with its first two beats 0.13 s apart, where the second row should
    have been beat 2.  The modal value is the steady state and is unaffected by that.

    Two things this deliberately does not do.  It does not reject a signature because it is
    unusual: `collapse` really does move to 5/8, and second-guessing that is how a correct
    reading gets discarded.  It does not repair the ragged opening either -- it is recorded
    as-is and a warning is raised, so a consumer can see the artefact rather than receiving
    a silently cleaned value.
    """
    warnings: list[str] = []
    if not path.is_file():
        return [], warnings
    per_beat: list[tuple[float, int, int, int]] = []
    malformed = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            malformed += 1
            continue
        try:
            when = float(parts[0])
            beat_in_bar = int(parts[1])
            numerator = int(parts[2])
            denominator = int(parts[3])
        except ValueError:
            malformed += 1
            continue
        if numerator <= 0 or denominator <= 0:
            malformed += 1
            continue
        per_beat.append((when, beat_in_bar, numerator, denominator))
    if malformed:
        warnings.append(f"beat.lab: {malformed} line(s) without a usable meter")
    if not per_beat:
        return [], warnings

    per_beat.sort(key=lambda row: row[0])
    grouped: list[list[tuple[float, int, int, int]]] = []
    for row in per_beat:
        if grouped and grouped[-1][0][2:] == row[2:]:
            grouped[-1].append(row)
        else:
            grouped.append([row])

    spans: list[tuple[float, int, int, int]] = []
    for group in grouped:
        counts: dict[int, int] = {}
        for _when, beat_in_bar, _n, _d in group:
            counts[beat_in_bar] = counts.get(beat_in_bar, 0) + 1
        modal = max(counts, key=lambda key: counts[key])
        spans.append((group[0][0], group[0][2], group[0][3], modal))

    if len(spans) > 1:
        first_end = spans[1][0]
        beats_in_first = sum(1 for t, _b, _n, _d in per_beat if t < first_end)
        if beats_in_first <= spans[0][1]:
            warnings.append(
                f"beat.lab opens with {beats_in_first} beat(s) of "
                f"{spans[0][1]}/{spans[0][2]} before changing at {first_end:.2f}s, "
                "which reads as a pickup"
            )
    return spans, warnings


def format_chord_label(label: str) -> str | None:
    """``"C:min7"`` -> ``"Cm7"``; ``"N"`` -> ``None`` (no chord)."""
    text = (label or "").strip()
    if not text or text in {"N", "N.C.", "X"}:
        return None
    if ":" not in text:
        # Already an abbreviation, or something music21 may understand.
        return text
    root, _, quality = text.partition(":")
    quality = quality.split("/")[0].strip()
    if not root or not quality:
        return None
    suffix = _QUALITY_SUFFIX.get(quality)
    if suffix is None:
        # Unknown quality: keep the Harte form, which is still more informative
        # than dropping the segment.
        return f"{root}{quality}"
    return f"{root}{suffix}"


def _read_meter_and_key(directory: Path) -> tuple[tuple[int, int] | None, str | None, list[str]]:
    """Pull the meter and key out of ``events.json``.

    The meter is only ever used for reporting: SheetSage2 halves the bar on
    material where this pipeline reads it correctly, so its value is evidence to
    fuse, never an instruction to follow.
    """
    import json

    warnings: list[str] = []
    path = directory / "events.json"
    if not path.is_file():
        return None, None, warnings
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        warnings.append(f"SheetSage2 events.json could not be read ({exc})")
        return None, None, warnings

    events = payload.get("events") if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        return None, None, warnings

    meter: tuple[int, int] | None = None
    key: str | None = None
    for event in events:
        values = event.get("values") if isinstance(event, dict) else None
        if not isinstance(values, dict):
            continue
        rhythm = values.get("rhythm")
        if meter is None and isinstance(rhythm, dict) and rhythm.get("meter"):
            pair = tuple(rhythm["meter"])
            if len(pair) == 2:
                meter = (int(pair[0]), int(pair[1]))
        if key is None and values.get("key"):
            key = str(values["key"])
        if meter and key:
            break
    return meter, key, warnings


def load_prior(directory: str | Path) -> SheetSage2Prior:
    """Load a SheetSage2 output directory into a :class:`SheetSage2Prior`.

    A missing or unreadable file degrades that one cue and records why, rather than
    failing the run: a prior is an improvement, and losing it must not be worse than
    never having had it.
    """
    path = Path(directory)
    prior = SheetSage2Prior(directory=path)
    if not path.is_dir():
        prior.warnings.append(f"SheetSage2 prior directory not found: {path}")
        return prior

    prior.beat_times = _read_beat_times(path / "beat.lab")
    prior.downbeat_times = _read_times(path / "downbeat.lab")

    if not prior.beat_times:
        # Fall back to the beat grid implied by events.json, which carries the same
        # information in a heavier form.
        events_path = path / "events.json"
        if events_path.is_file():
            import json

            try:
                payload = json.loads(events_path.read_text(encoding="utf-8"))
                events = payload.get("events") if isinstance(payload, dict) else payload
                times = sorted(
                    float(e["time"])
                    for e in events or []
                    if isinstance(e, dict) and "time" in e and "timestamp" in (e.get("tokens_by_field") or {})
                )
                prior.beat_times = times
                if times:
                    prior.warnings.append(
                        f"beat.lab was empty; {len(times)} beat times recovered from events.json"
                    )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                prior.warnings.append(f"beat times unavailable from events.json ({exc})")

    prior.meter, prior.key, extra = _read_meter_and_key(path)
    prior.warnings.extend(extra)

    # The per-beat reading from `beat.lab`.  This is the one that can change mid-song and
    # the one the pipeline prefers over its own inference, so it is read independently of
    # `events.json`: if the two ever disagree, the per-beat file is the finer record.
    prior.meter_spans, meter_warnings = _read_meter_spans(path / "beat.lab")
    prior.warnings.extend(meter_warnings)
    if not prior.meter_spans:
        # back-fill so a prior without beat.lab columns still answers `meter_at`
        if prior.meter is not None and prior.beat_times:
            prior.meter_spans = [(prior.beat_times[0], prior.meter[0], prior.meter[1], 1)]
    else:
        labelled = prior.meter_spans[0]
        if prior.meter is not None and (labelled[1], labelled[2]) != prior.meter:
            prior.warnings.append(
                f"events.json reports {prior.meter[0]}/{prior.meter[1]} but beat.lab "
                f"reports {labelled[1]}/{labelled[2]}; beat.lab wins as the per-beat record"
            )
        prior.meter = (labelled[1], labelled[2])

    for start, end, label in _read_lab(path / "chord.lab"):
        formatted = format_chord_label(label)
        if formatted is None or end <= start:
            continue
        prior.chords.append(Chord(start=start, end=end, label=formatted, confidence=1.0))

    prior.structure = [
        (start, end, label) for start, end, label in _read_lab(path / "structure.lab") if label
    ]

    if not prior.ok:
        prior.warnings.append(
            f"only {len(prior.beat_times)} beat times in {path}: the prior will be ignored"
        )
    if prior.beat_times and not prior.downbeat_times:
        prior.warnings.append("the prior has beats but no downbeats: the meter cue is unavailable")
    return prior


def beat_times_from_prior(prior: SheetSage2Prior) -> Sequence[float]:
    """The prior's beat grid, for callers that only want to compare grids."""
    return list(prior.beat_times)
