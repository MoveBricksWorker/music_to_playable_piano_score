"""Stem -> instrument mapping.

Separation backends hand back stem *names*, and notation needs to know what each
one is: which part of the score it belongs to, which MIDI program, which clef,
whether it is polyphonic enough to deserve a grand staff, and whether it is a
transposing instrument.

The map is plain data and can be overridden with a YAML file; unknown fields are
rejected rather than silently ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..core.errors import ConfigError
from ..core.models import PartSpec

__all__ = [
    "DEFAULT_MAP",
    "resolve",
    "load_overrides",
    "order_parts",
    "SCORE_ORDER",
]

# Canonical score order, melody first and percussion last.  Anything not listed
# sorts after the known names, alphabetically, so custom stems stay predictable.
SCORE_ORDER = (
    "vocals",
    "lead",
    "melody",
    "piano",
    "keys",
    "keyboard",
    "guitar",
    "strings",
    "other",
    "bass",
    "drums",
    "percussive",
)

_DEFAULTS: dict[str, PartSpec] = {
    "vocals": PartSpec("vocals", "Voice", "Vocalist", 53, "treble", monophonic=True),
    "lead": PartSpec("lead", "Lead", "Lead", 81, "treble", monophonic=True),
    "melody": PartSpec("melody", "Melody", "Lead", 81, "treble", monophonic=True),
    "piano": PartSpec("piano", "Piano", "Piano", 0, "treble", grand_staff=True),
    "keys": PartSpec("keys", "Keyboard", "Piano", 0, "treble", grand_staff=True),
    "keyboard": PartSpec("keyboard", "Keyboard", "Piano", 0, "treble", grand_staff=True),
    # Guitar is a single treble staff, not a grand staff.  Guitar music is written
    # on one staff, and pretending otherwise produced a two-staff part whose lower
    # staff held almost nothing: measured on a real run, the upper staff of the
    # guitar's grand staff was empty on the first page while a handful of notes sat
    # alone on the lower one.  (Writing it an octave up via `sounding_pitch` would
    # be the fully correct notation, but that also changes MIDI playback, so the
    # register is left alone here and only the staff count is fixed.)
    "guitar": PartSpec("guitar", "Guitar", "Guitar", 24, "treble"),
    "strings": PartSpec("strings", "Strings", "StringEnsemble", 48, "treble", grand_staff=True),
    "other": PartSpec("other", "Keyboard", "Piano", 0, "treble", grand_staff=True),
    "bass": PartSpec("bass", "Bass", "ElectricBass", 33, "bass", monophonic=True),
    "drums": PartSpec("drums", "Drums", "Percussion", None, "percussion", drum_channel=True),
    "percussive": PartSpec(
        "percussive", "Percussion", "Percussion", None, "percussion", drum_channel=True
    ),
}

DEFAULT_MAP: dict[str, PartSpec] = dict(_DEFAULTS)

# Stem name fragments -> canonical spec, checked in order.  Lets "vocals_deverb"
# or "other_2" resolve sensibly instead of falling back to a generic part.
_FRAGMENTS: tuple[tuple[str, str], ...] = (
    ("vocal", "vocals"),
    ("voice", "vocals"),
    ("drum", "drums"),
    ("perc", "percussive"),
    ("bass", "bass"),
    ("piano", "piano"),
    ("key", "keys"),
    ("guitar", "guitar"),
    ("string", "strings"),
    ("violin", "strings"),
    ("cello", "strings"),
    ("lead", "lead"),
)

_SPEC_FIELDS = frozenset(PartSpec.__dataclass_fields__) - {"stem"}


def _generic(stem: str) -> PartSpec:
    """A sensible part for a stem we have never heard of."""
    pretty = stem.replace("_", " ").replace("-", " ").strip().title() or "Part"
    return PartSpec(stem=stem, part_name=pretty, instrument="Piano", midi_program=0, clef="treble")


def _restamp(spec: PartSpec, stem: str) -> PartSpec:
    """Copy ``spec`` under a different stem name."""
    fields = spec.to_dict()
    fields["stem"] = stem
    return PartSpec(**fields)


def resolve(stem: str, overrides: dict[str, PartSpec] | None = None) -> PartSpec:
    """Look up the spec for ``stem``, honouring overrides then fuzzy fragments."""
    if overrides and stem in overrides:
        return _restamp(overrides[stem], stem)

    known = DEFAULT_MAP.get(stem)
    if known is not None:
        return known

    lowered = stem.lower()
    for fragment, canonical in _FRAGMENTS:
        if fragment in lowered:
            return _restamp(DEFAULT_MAP[canonical], stem)

    return _generic(stem)


def load_overrides(path: str | Path | None) -> dict[str, PartSpec]:
    """Read a YAML ``{stem: {field: value}}`` map into :class:`PartSpec` objects.

    Raises :class:`ConfigError` on anything malformed — a typo in a field name
    silently doing nothing is worse than a failed run.
    """
    if not path:
        return {}
    target = Path(path)
    if not target.is_file():
        raise ConfigError(f"instrument map not found: {target}")
    try:
        raw: Any = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"instrument map is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError("instrument map must be a mapping of stem -> settings")

    overrides: dict[str, PartSpec] = {}
    for stem, settings in raw.items():
        if not isinstance(settings, dict):
            raise ConfigError(f"instrument map entry {stem!r} must be a mapping, got {type(settings).__name__}")
        unknown = set(settings) - _SPEC_FIELDS
        if unknown:
            raise ConfigError(
                f"instrument map entry {stem!r} has unknown field(s): {sorted(unknown)}; "
                f"allowed: {sorted(_SPEC_FIELDS)}"
            )
        base = resolve(str(stem))
        merged = {**base.to_dict(), **settings, "stem": str(stem)}
        try:
            overrides[str(stem)] = PartSpec(**merged)
        except TypeError as exc:
            raise ConfigError(f"instrument map entry {stem!r} is invalid: {exc}") from exc
    return overrides


def order_parts(stems: list[str]) -> list[str]:
    """Sort stem names into score order (melody first, percussion last)."""

    def rank(stem: str) -> tuple[int, str]:
        for index, name in enumerate(SCORE_ORDER):
            if stem == name:
                return (index, "")
        for index, name in enumerate(SCORE_ORDER):
            if name in stem.lower():
                return (index, stem)
        return (len(SCORE_ORDER), stem)

    return sorted(stems, key=rank)
