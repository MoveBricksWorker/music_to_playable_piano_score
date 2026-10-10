"""Deciding what a stem's SheetSage2 transcription should contain.

Kept in the package rather than in the worker script for two reasons: it is a
musical decision that deserves tests, and both the worker (which runs under the
pinned ``transformers`` interpreter) and the pipeline need the same answer.

The problem this solves
-----------------------
SheetSage2 is a lead-sheet model.  Given **any** audio it emits two kinds of output:

* a **melody** voice, monophonic by construction and written in the treble whatever
  it was given, and
* a **block-chord** voice, which is a stack of sustained chords.

Measured on real separated stems, neither is right for every instrument:

=========================  ==========  ===============  ==================
stem                       voice       duty cycle       in the register
=========================  ==========  ===============  ==================
vocals / piano / guitar    melody      ~0.90            yes
vocals / piano / guitar    chords      ~4.8             no (below the line)
bass                       melody      ~0.80            13% — it is in the treble
bass                       chords      ~4.8             99.9% inside 24-67
=========================  ==========  ===============  ==================

So the rule is: take the melody when it survives in the instrument's register *as a
line*, and otherwise fall back to the chords.  Taking the union — the obvious first
attempt — is wrong in a way that is easy to miss: on a vocal stem the chords sit
below the vocal register while the melody sits inside it, so a union that is then
range-filtered keeps mostly the chords and the sung line disappears.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

__all__ = ["duty_cycle", "select_voice", "LINE_MAX_DUTY"]

# Above this, a voice is a stack of overlapping notes rather than a melodic line.
# The measured gap is wide (melody ~0.9, chords ~4.8), so the exact cut is not
# delicate; it only has to separate those two populations.
LINE_MAX_DUTY = 1.5


def duty_cycle(notes: Sequence[Mapping[str, float]]) -> float:
    """Sounding time as a share of the span the notes occupy.

    Returns 0.0 for nothing, a value near 1.0 for a line that is always sounding,
    and above 1.0 as soon as notes start overlapping each other.
    """
    if not notes:
        return 0.0
    span = max(n["end"] for n in notes) - min(n["start"] for n in notes)
    if span <= 0:
        return 0.0
    return sum(n["end"] - n["start"] for n in notes) / span


def select_voice(
    voices: Mapping[str, list[dict]],
    register: tuple[int, int] | None = None,
) -> tuple[list[dict], str, float]:
    """Pick one voice from a SheetSage2 result.

    ``voices`` maps a voice name to its notes (``{"melody": [...], "chords": [...]}``).
    ``register`` is the MIDI span the instrument may occupy; when given, each voice
    is first restricted to it, so this can never return notes the pipeline's own
    range filter would reject.

    Returns ``(notes, chosen_name, duty)``.  ``chosen_name`` is ``"none"`` and the
    list empty when nothing usable survived.
    """
    candidates: list[tuple[str, list[dict]]] = []
    for name, notes in voices.items():
        if not notes:
            continue
        kept = list(notes)
        if register is not None:
            low, high = register
            kept = [n for n in kept if low <= n["pitch"] <= high]
        if kept:
            candidates.append((name, sorted(kept, key=lambda n: (n["start"], n["pitch"]))))

    if not candidates:
        return [], "none", 0.0

    lines = [(name, notes) for name, notes in candidates if duty_cycle(notes) <= LINE_MAX_DUTY]
    if lines:
        # Between two lines, the one carrying more notes carries more of the music.
        name, notes = max(lines, key=lambda item: len(item[1]))
        return notes, name, round(duty_cycle(notes), 3)

    # Nothing here is a line.  Hand over the least stacked voice rather than nothing,
    # and let the run report what it was given.
    name, notes = min(candidates, key=lambda item: duty_cycle(item[1]))
    return notes, name, round(duty_cycle(notes), 3)
