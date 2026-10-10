"""Score layout helpers: clefs, instruments and drum pitch mapping.

Split out of :mod:`ensemble2score.notation.score` so the note-building code reads
as music rather than as table lookups.
"""

from __future__ import annotations

from ..core.models import PartSpec

__all__ = [
    "DRUM_MIDI_PITCH",
    "clef_for",
    "clef_for_range",
    "instrument_for_spec",
    "part_label",
]

# General MIDI percussion pitches we use, one per classified voice.
DRUM_MIDI_PITCH: dict[str, int] = {"kick": 36, "snare": 38, "hihat": 42}

# Written pitch -> the pitch it sounds at, for transposing instruments.
# The model stores a *sounding* pitch everywhere, so this is only used when a
# part is engraved for an instrument that reads a transposed part.
_CLEF_NAMES = {"treble": "TrebleClef", "bass": "BassClef", "percussion": "PercussionClef", "alto": "AltoClef"}

# Which side of the treble staff's bottom line (E4) a part has to sit on before a
# bass clef reads better.  A part centred below this is written with a fistful of
# ledger lines in treble and cleanly in bass.
_BASS_CLEF_BELOW = 57  # A3


def clef_for(spec: PartSpec, clef_module):
    """The music21 clef object for a part spec."""
    name = _CLEF_NAMES.get(spec.clef, "TrebleClef")
    return getattr(clef_module, name)()


def clef_for_range(spec: PartSpec, written_pitches, clef_module):
    """Clef for this part, letting its actual register override the declared one.

    The spec says what an instrument *usually* reads; the transcription says where
    this particular part ended up, and a separator does not always put an instrument
    in its textbook register.  Measured on a real run: the guitar part came out
    centred on MIDI 48, so engraving it in the declared treble clef produced a staff
    of ledger lines below the staff — technically correct, unreadable in practice.

    Only treble and bass are reconsidered.  A percussion part keeps its percussion
    clef, and an instrument that reads a transposed part keeps the clef chosen for
    it, because changing octave *placement* is the transposition's job, not the
    clef's.
    """
    if spec.clef not in ("treble", "bass") or spec.drum_channel:
        return clef_for(spec, clef_module)

    pitches = [p for p in written_pitches if p is not None]
    if not pitches:
        return clef_for(spec, clef_module)

    import statistics

    median = statistics.median(pitches)
    name = "bass" if median < _BASS_CLEF_BELOW else "treble"
    return getattr(clef_module, _CLEF_NAMES[name])()


def instrument_for_spec(spec: PartSpec, instrument_module):
    """A music21 instrument with the right MIDI program, by name where possible."""
    midi_program = spec.midi_program
    if spec.drum_channel or midi_program is None:
        return instrument_module.Percussion()

    for name in dir(instrument_module):
        candidate = getattr(instrument_module, name)
        if not isinstance(candidate, type):
            continue
        try:
            instance = candidate()
        except Exception:  # noqa: BLE001 - some instruments need arguments
            continue
        if getattr(instance, "midiProgram", None) == midi_program:
            return instance

    # Nothing in music21 declares this program: fall back to the generic synth
    # and keep the program number so the MIDI export stays correct.
    generic = instrument_module.Instrument()
    generic.midiProgram = midi_program
    return generic


def part_label(spec: PartSpec) -> str:
    """Short label used in reports and for part abbreviations."""
    return spec.part_name or spec.stem
