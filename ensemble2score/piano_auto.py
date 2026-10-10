"""A deterministic arranger: the same job the LLM does, without a model.

Why this exists
---------------
Two reasons, and the second is the important one.

1. **The tool has to run without a model.** `piano_arrange.py packet` produces work for
   an LLM, which is fine until there is no LLM to hand it to. A rule-based arranger makes
   `deliverables/<track>` → piano score a single command, offline, in seconds.
2. **It is the control group.** "The LLM merge is better than a naive merge" is a claim,
   and a claim needs a baseline. This module *is* the naive merge — melody on top,
   accompaniment below, mechanical rules — so the measured difference between it and the
   model's arrangements says what the model is actually contributing, instead of leaving
   it as an impression.

It is deliberately unmusical in the way a rule-based arranger must be: it knows about
registers, spans, onsets and hand assignment, and nothing at all about phrasing. Where
the rules cannot decide, it does the safe thing (keeps the melody, thins the texture)
rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .piano_arrange import Arrangement, ArrangementInput, sounding_lengths

__all__ = ["AutoRules", "auto_arrange"]


@dataclass(frozen=True)
class AutoRules:
    """The arrangement rules, all of them explicit and measurable.

    Defaults are the ones the model-written arrangements achieved, so the comparison is
    like for like.
    """

    #: A hand may not strike a wider interval than this.  19 semitones is an octave plus
    #: a fifth, which is what the measured LLM arrangements used.
    max_span: int = 19
    #: Notes allowed to sound at once across both hands.
    max_simultaneous: int = 4
    #: Middle C: the right hand stays at or above it, the left at or below it.
    split: int = 60
    #: Highest note the left hand may use, so it never crowds the split.
    left_ceiling: int = 55
    #: Add at most this many harmony notes under a melody note.
    harmony_under_melody: int = 2
    #: Fill left-hand gaps at most this often, in ticks (4 = once per beat).
    left_fill_every: int = 4
    #: Move a transcribed pitch by octaves until it fits, rather than dropping it.
    fold_octaves: bool = True
    #: Least ticks between melody onsets before a harmony note is added underneath, so a
    #: fast melodic line is not thickened into unreadability.
    min_gap_for_harmony: int = 2


def _fold_into_range(
    pitch: int, low: int, high: int, *, enabled: bool
) -> int | None:
    """Move ``pitch`` by whole octaves until it fits, or give up.

    Whole octaves only.  Moving a transcribed pitch by a few semitones changes the tune,
    which is the one edit an arranger is not allowed to make silently.
    """
    if low <= pitch <= high:
        return pitch
    if not enabled:
        return None
    for shift in (12, -12, 24, -24, 36, -36):
        candidate = pitch + shift
        if low <= candidate <= high:
            return candidate
    return None


def auto_arrange(
    material: ArrangementInput,
    *,
    rules: AutoRules | None = None,
) -> Arrangement:
    """Merge the material into a two-hand arrangement by rule.

    The melody goes to the right hand untouched in pitch wherever it fits the register,
    and the accompaniment fills in underneath.  Nothing is invented: every pitch comes
    from one of the input voices.
    """
    rules = rules or AutoRules()
    grid = material.grid
    per_bar = max(1, grid.beats_per_bar) * grid.ticks_per_beat
    limit = material.bars * per_bar if material.bars else None

    melody = material.voice("MELODY")
    # The arranger wants one accompaniment line, so it uses the merged view; the model is
    # shown the instruments separately, because telling a piano texture from a string pad
    # is a judgement this module deliberately does not make.
    harmony = material.harmony
    bass = material.voice("BASS")

    def packed(voice, *, keep_highest: bool) -> dict[int, int]:
        """One event per tick, resolving collisions the way the hand needs."""
        if voice is None:
            return {}
        out: dict[int, int] = {}
        for tick, pitch in voice.events:
            if limit is not None and not (0 <= tick < limit):
                continue
            if tick in out:
                out[tick] = (
                    max(out[tick], pitch) if keep_highest else min(out[tick], pitch)
                )
            else:
                out[tick] = pitch
        return out

    melody_ticks = packed(melody, keep_highest=True)
    harmony_ticks = packed(harmony, keep_highest=True)
    bass_ticks = packed(bass, keep_highest=False)

    arrangement = Arrangement(track=material.track)

    # ---- right hand: the melody, top voice ---------------------------------- #
    right: dict[int, list[int]] = {}
    for tick in sorted(melody_ticks):
        pitch = _fold_into_range(
            melody_ticks[tick], rules.split, 108, enabled=rules.fold_octaves
        )
        if pitch is None:
            continue
        right[tick] = [pitch]

    # Fill under the melody where the line moves slowly enough to take it.
    melody_order = sorted(right)
    for index, tick in enumerate(melody_order):
        top = right[tick][0]
        gap = (
            melody_order[index + 1] - tick
            if index + 1 < len(melody_order)
            else per_bar
        )
        if gap < rules.min_gap_for_harmony:
            continue
        wanted = [
            harmony_ticks[other]
            for other in harmony_ticks
            if tick <= other < tick + gap
        ]
        if not wanted:
            continue
        # Only notes that fit under the top voice and inside the hand's reach.
        candidates = sorted(
            {
                pitch
                for pitch in wanted
                if rules.split <= pitch < top and top - pitch <= rules.max_span
                and pitch not in right[tick]
            },
            reverse=True,
        )
        for pitch in candidates[: rules.harmony_under_melody]:
            right[tick].append(pitch)
        right[tick] = sorted(set(right[tick]))

    # ---- left hand: the bass, with the accompaniment filling gaps ----------- #
    left: dict[int, list[int]] = {}
    for tick in sorted(bass_ticks):
        pitch = _fold_into_range(
            bass_ticks[tick], 21, rules.left_ceiling, enabled=rules.fold_octaves
        )
        if pitch is not None:
            left[tick] = [pitch]

    # Fill left-hand gaps from the accompaniment, **folding it into the register** rather
    # than discarding the notes that sit above the split.  Dropping them instead left the
    # left hand with 79 notes on a track whose accompaniment has 563, which sounds like a
    # melody with a bass note occasionally attached.
    for tick in sorted(harmony_ticks):
        if tick in left:
            continue
        if tick % rules.left_fill_every:
            continue
        pitch = _fold_into_range(
            harmony_ticks[tick], 21, rules.left_ceiling, enabled=rules.fold_octaves
        )
        if pitch is None:
            continue
        left[tick] = [pitch]

    # A second note under a left-hand root, kept inside the hand's reach and kept clear of
    # the root: two notes a semitone or a tone apart in the bass is mud, not harmony.
    for tick in sorted(left):
        if tick % (rules.left_fill_every * 2):
            continue
        root = left[tick][0]
        window = range(tick, tick + rules.left_fill_every * 2)
        for other in sorted(harmony_ticks):
            if other not in window:
                continue
            pitch = _fold_into_range(
                harmony_ticks[other], 21, rules.left_ceiling, enabled=rules.fold_octaves
            )
            if pitch is None or pitch == root:
                continue
            interval = abs(pitch - root)
            if not 3 <= interval <= rules.max_span:
                continue
            left[tick].append(pitch)
            break
        left[tick] = sorted(set(left[tick]))

    arrangement.right = dict(sorted(right.items()))
    arrangement.left = dict(sorted(left.items()))

    # Enforce the polyphony budget from the **later** of the two hands' onsets: a note
    # sounding through a later onset is still sounding, so trimming at the shared instant
    # is the only way "at most N notes at once" can actually hold.
    _enforce_polyphony(arrangement, rules, grid)
    arrangement.warnings.append(
        "arranged by rule, not by a model: registers, spans and hand assignment are "
        "enforced, phrasing is not considered"
    )
    return arrangement


def _enforce_polyphony(arrangement: Arrangement, rules: AutoRules, grid) -> None:
    """Thin the texture until no more than ``max_simultaneous`` notes sound together.

    A sweep over the onsets: a note sounds from its onset to the next onset in its hand,
    so the sounding set changes only at those points.  At each one the budget is spent on
    the lines the arrangement exists to carry — the melody first, the bass second — and
    accompaniment notes yield.  Notes are removed whole, never shortened, so the rhythm of
    whatever remains is untouched.

    This has to consider notes carried over from earlier onsets, not just the notes
    starting here: a note sounding *through* a later onset is exactly what makes a texture
    unplayable, and counting only simultaneous attacks would miss it.
    """
    per_hand = {"RH": arrangement.right, "LH": arrangement.left}
    lengths = {
        hand: {
            tick: max(1, int(round(span * grid.ticks_per_beat)))
            for tick, span in sounding_lengths(
                events, ticks_per_beat=grid.ticks_per_beat
            ).items()
        }
        for hand, events in per_hand.items()
    }

    onsets = sorted(set(arrangement.right) | set(arrangement.left))
    for onset in onsets:
        # Which events are still sounding at this instant, and how many notes each carries.
        active: list[tuple[str, int, int]] = []   # (hand, tick, notes)
        for hand, events in per_hand.items():
            for tick, pitches in events.items():
                if tick <= onset < tick + lengths[hand].get(tick, 1):
                    active.append((hand, tick, len(pitches)))
        total = sum(count for _h, _t, count in active)
        if total <= rules.max_simultaneous:
            continue

        def priority(item: tuple[str, int, int]) -> tuple[int, int]:
            """Shed order: extra notes first, then left-hand fill, then right-hand fill.

            The melody is the top note of the right hand and the bass the lowest note of
            the left, so a hand with one note at an event is never a candidate while any
            hand still carries a spare note.
            """
            hand, tick, count = item
            spare = count - 1
            # Never shed a one-note left-hand event while a right-hand chord has spares,
            # and vice versa: prefer whichever hand is currently the thicker one.
            return (-spare, 0 if hand == "LH" else 1)

        for hand, tick, count in sorted(active, key=priority):
            if total <= rules.max_simultaneous:
                break
            if count <= 1:
                continue
            events = per_hand[hand]
            keep = 1
            # Keep the top note of the right hand (the melody) and the lowest of the left
            # (the bass); those are the lines being carried.
            events[tick] = (
                events[tick][-keep:] if hand == "RH" else events[tick][:keep]
            )
            total -= count - keep
