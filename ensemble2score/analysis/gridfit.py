"""Deciding, locally, whether a rhythm is duple or triple -- then snapping absolutely.

The problem this solves
-----------------------
`BarGrid.position()` rounds each onset to the nearest tick, independently.  On a
4-ticks-per-beat grid a triplet eighth (1/3 beat) is not representable at all, so it
becomes a sixteenth.  Measured on the project's own transcription of *victim* (485
onsets, first 60 s): **0 onsets** landed on a triplet position, and the exported score
contained no triplet duration of any kind.

Raising the resolution to 12 ticks per beat makes 1/3 representable, but snapping each
onset independently then fails in the opposite direction, because a straight sixteenth
(0.25 beat = 3 ticks) and a triplet sixteenth (1/6 beat = 2 ticks) are one tick apart:

    straight sixteenth    0.25 beat -> 3.0 ticks
    triplet sixteenth     1/6  beat -> 2.0 ticks
    triplet eighth        1/3  beat -> 4.0 ticks

Measured consequence of the naive change on the same 485 onsets: the most common
resulting duration became **1/6, 319 times** -- straight sixteenths were dragged onto the
triplet lattice.  So neither resolution alone works: one cannot represent triplets, the
other invents them.

The design, and the three earlier ones it replaces
--------------------------------------------------
Three scoring schemes were built and measured against a synthetic ruler before this one.
All three are worth recording, because each looked reasonable and each failed in a
different way:

1. **Whole-phrase absolute distance to a lattice.**  Fails on a metrically ambiguous bar
   (`eighth, eighth, triplet-eighth, triplet-eighth`): the triplet onsets sit nearer the
   duple positions, so the duple lattice wins and the triplets are rounded away.

2. **Whole-phrase interval error.**  Fails the other way: a 0.5-beat interval is legal on
   *both* lattices, so an ambiguous phrase is decided by whichever ambiguous interval
   happens to dominate.

3. **Sequential snapping with a per-gap lattice choice.**  This one *drifting*: with a gap
   of 1.4 ticks and a smallest step of 3, `round(gap / step)` is 0, so the onset did not
   move -- but the next onset was measured from the *snapped* previous one, so every
   onset's sub-step error was carried forward instead of being absorbed.  Measured drift
   on the real transcription: **+1 tick per onset, 105 ticks by mid-phrase.**

What this module does
---------------------
Two decisions, kept separate on purpose:

* **Which lattice** -- decided from the intervals over a sliding window of recent onsets.
  A window is robust (one ambiguous interval cannot decide it) and local (the choice can
  change at a real metre change).  Hysteresis stops a near-tie from flipping it.
* **Where each onset goes** -- snapped *absolutely* to the chosen lattice, never relative
  to the previous snapped onset.  Absolute snapping cannot accumulate error, which is
  exactly what design 3 got wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "DUAL_LATTICE",
    "DUPLE_STEP",
    "TRIPLE_STEP",
    "TripletDecision",
    "lattice_error",
    "quantize_beats",
]

# The finest resolution on which a duple and a triple lattice are both exact.
DUAL_LATTICE = 12
# Lattice periods, in ticks of DUAL_LATTICE.  3 = 1/4 beat, 4 = 1/3 beat.
DUPLE_STEP = 3
TRIPLE_STEP = 4

# Which steps each lattice is allowed to use.  A lattice is not a single number: music
# writes 1/4, 1/8 and 1/16, and it writes triplet 1/4, 1/8 and 1/16 too.  All members of
# a family are multiples of the family's base, so the test is whether the base divides
# the interval.
#
# The triple family is written as 2, 4, 6, 8, 12 rather than 4, 8, 12 because a triplet
# SIXTEENTH is 2 ticks, and without 2 in the family no grid here could express it at all:
# the identity self-check fed exact triplet sixteenths and the quantiser moved every one
# of them onto the duple lattice, which is the very failure the user reported for fast
# figures.  The families still do not overlap on the steps that distinguish them (3 and
# its multiples vs 2 and its multiples), so straight sixteenths at 3 ticks cost 0 on the
# duple family and 1 on the triple family, and stay duple.
DUPLE_FAMILY = (3, 6, 12)          # 1/4, 1/8, 1/16
TRIPLE_FAMILY = (2, 4, 6, 8, 12)   # triplet 1/16, triplet 1/8, triplet 1/4

# How many recent intervals vote on the lattice.  Short on purpose: a three-note triplet
# group inside a run of eighths has to be able to win the vote, and a long window is
# dominated by the surrounding eighths.  Measured on a synthetic case of exactly that
# shape, a window of 8 kept the triplets on the duple lattice; 4 does not.
WINDOW = 4
# How much better the other lattice must fit before the choice switches, in ticks.
# A genuine triplet figure misses the duple lattice by ~0.75 tick per gap and vice
# versa, so 0.5 admits real metre changes while rejecting near-ties.
HYSTERESIS_TICKS = 0.5
# A gap longer than this is a break: the window is emptied rather than voting across it.
CELL_BREAK_BEATS = 2.0


@dataclass
class TripletDecision:
    """What the quantiser decided, so the choice can be reported rather than assumed."""

    total_onsets: int = 0
    duple_onsets: int = 0
    triple_onsets: int = 0
    switches: int = 0
    moved: int = 0
    max_shift_ticks: float = 0.0
    # (onset index, new step, duple error, triple error) at each change of lattice
    changes: list[tuple[int, int, float, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_onsets": int(self.total_onsets),
            "duple_onsets": int(self.duple_onsets),
            "triple_onsets": int(self.triple_onsets),
            "lattice_switches": int(self.switches),
            "moved": int(self.moved),
            "max_shift_ticks": round(self.max_shift_ticks, 4),
        }

    def summary(self) -> str:
        return (
            f"{self.total_onsets} onset(s): {self.duple_onsets} snapped duple, "
            f"{self.triple_onsets} triple, {self.switches} switch(es); "
            f"{self.moved} moved (max {self.max_shift_ticks:.2f} ticks)"
        )


def _nearest_in_family(value: float, family: tuple[int, ...]) -> int:
    """Nearest value in ``family`` (not the nearest multiple of one step)."""
    return min(family, key=lambda s: abs(value - s))


def lattice_error(intervals: list[float], family: tuple[int, ...]) -> float:
    """Mean error when each interval is expressed as one value of ``family``, in ticks."""
    if not intervals:
        return 0.0
    total = 0.0
    for gap in intervals:
        total += abs(gap - _nearest_in_family(gap, family))
    return total / len(intervals)


def quantize_beats(
    onsets_beats: list[float],
    *,
    ticks_per_beat: int = DUAL_LATTICE,
    window: int = WINDOW,
    collect: TripletDecision | None = None,
) -> list[int]:
    """Onset times in beats -> tick positions, snapping to a locally chosen lattice.

    ``onsets_beats`` must be sorted ascending.  The result is parallel to the input, in
    ticks of ``ticks_per_beat`` per beat, ready for ``BarGrid.from_tick``-style use.
    """
    if not onsets_beats:
        return []

    raw = [b * DUAL_LATTICE for b in onsets_beats]

    # --- pass 1: intervals, and the lattice to use at each onset -------------- #
    #
    # The lattice is resolved for *every* onset before anything is snapped, so an onset
    # is never placed on the default lattice just because the window has not filled up
    # yet.  Getting this wrong cost a whole tick on the first two onsets of a pure
    # triplet run, which is what the identity self-check caught.
    steps: list[int] = [DUPLE_STEP]
    recent: list[float] = []
    step = DUPLE_STEP
    for i in range(1, len(raw)):
        gap = raw[i] - raw[i - 1]
        if gap > CELL_BREAK_BEATS * DUAL_LATTICE:
            recent.clear()
        elif gap > 0:
            recent.append(gap)
            if len(recent) > window:
                recent.pop(0)

        if recent:
            d_err = lattice_error(recent, DUPLE_FAMILY)
            t_err = lattice_error(recent, TRIPLE_FAMILY)
            if t_err < d_err - HYSTERESIS_TICKS:
                new_step = TRIPLE_STEP
            elif d_err < t_err - HYSTERESIS_TICKS:
                new_step = DUPLE_STEP
            else:
                new_step = step
            if new_step != step and collect is not None:
                collect.switches += 1
                collect.changes.append((i, new_step, d_err, t_err))
            step = new_step
        steps.append(step)

    # --- pass 2: snap absolutely, in the lattice decided for each onset ------- #
    #
    # Snapping uses the family's *base* step (3 or 4), i.e. the coarsest regular grid of
    # that family, not the finest member that happens to fit the gap.  Using 2 for a
    # triplet sixteenth would place it a sixteenth of a beat from its neighbours and
    # destroy the very grouping being notated.
    out: list[int] = []
    for i, pos in enumerate(raw):
        chosen = steps[i]
        base = chosen  # DUPLE_STEP or TRIPLE_STEP, which are the family bases
        snapped = int(round(pos / base)) * base
        # Integer division, not round(): `ticks_per_beat` can be 4 while the lattice is
        # 12, and round-tripping 4/12 through float can land a whole tick out.
        value = snapped * ticks_per_beat
        tick = value // DUAL_LATTICE if value % DUAL_LATTICE == 0 \
            else int(round(value / DUAL_LATTICE))
        shift = abs(snapped - pos)
        if collect is not None:
            collect.total_onsets += 1
            if shift > 1e-9:
                collect.moved += 1
            collect.max_shift_ticks = max(collect.max_shift_ticks, shift)
            if chosen == TRIPLE_STEP:
                collect.triple_onsets += 1
            else:
                collect.duple_onsets += 1
        out.append(int(tick))

    return out
