"""织体词表、判定规则、以及从和弦生成音符的引擎。

分工：**代码定"怎么把织体写成音符"，LLM 定"这一段用哪种织体"**。
音高永远来自和弦（可靠来源），所以"音高读错"这个失效模式被结构性消除。

判定规则从声学特征推出织体，规则本身在**已知织体**的合成输入上验证
（见 `verify_texture_rules.py`）。特征含义与区分度实测：

    起音密度 onset_rate   13 倍    每小节多少下
    持续度   sustain       7 倍    长音铺底 vs 断奏
    起音锐度 sharpness     7 倍    击弦 vs 铺底
    低频占比 low_share    11 倍    低音是否主导
    质心     centroid      3 倍    明亮/暗淡

为什么不用"哪个分轨是主奏"：yoake 的 piano 轨 −21.0 dBFS、guitar 轨 −20.2 dBFS，
两者几乎一样响，"主奏"在它们之间反复摇摆，而听到的织体是两者之和。所以判定直接用混合轨。
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# 词表：每种织体给出"一小节几拍几个音"的形状
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Texture:
    name: str
    #: 一小节内每个起音的拍位（以拍为单位，0 起）
    offsets: tuple[float, ...]
    #: 该织体在一个小节内的音符走向：从和弦取第几个音（0 = 最低）
    #: ``"cycle"`` 表示按索引循环取和弦音，``"top"`` 取最高，``"stack"`` 全取
    voicing: str
    #: 一句话说明，给 LLM 和日志看
    description: str
    #: 低音是否单独发一个根音（低音铺底）
    bass: bool = False


#: 网格是每拍 4 个 tick，所以 0.25 拍 = 十六分音符。三连音落不进 1/16 网格，
#: 用 0 / 1/3 / 2/3 拍的近似（0, 1.33, 2.67 tick）—— 舍入到最近的 tick。
VOCABULARY: tuple[Texture, ...] = (
    Texture(
        "pad",
        offsets=(0.0,),
        voicing="stack",
        description="整小节一个长和弦，铺底",
    ),
    Texture(
        "block",
        offsets=(0.0, 1.0, 2.0, 3.0),
        voicing="stack",
        description="每拍一个柱式和弦",
    ),
    Texture(
        "block_half",
        offsets=(0.0, 2.0),
        voicing="stack",
        description="每半小节一个柱式和弦（慢速铺底）",
    ),
    Texture(
        "broken_eighths",
        offsets=(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5),
        voicing="cycle",
        description="八分音符分解和弦（一小节 8 个音）",
    ),
    Texture(
        "broken_sixteenths",
        offsets=tuple(i * 0.25 for i in range(16)),
        voicing="cycle",
        description="十六分音符分解和弦（一小节 16 个音，密集音型）",
    ),
    Texture(
        "triplet",
        offsets=tuple(round(i / 3.0, 3) for i in range(12)),
        voicing="cycle",
        description="三连音分解（每拍三个音，一小节 12 个音；1/16 网格上的近似）",
    ),
    Texture(
        "arp_updown",
        offsets=(0.0, 0.5, 1.0, 1.5, 2.0, 1.5, 1.0, 0.5),
        voicing="cycle",
        description="上下行琶音",
    ),
    Texture(
        "bass_chord",
        offsets=(0.0, 1.0, 2.0, 3.0),
        voicing="stack",
        description="低音加和弦：每拍一个低音，和弦在上方",
        bass=True,
    ),
    Texture(
        "ostinato",
        offsets=(0.0, 0.75, 1.5, 2.25, 3.0),
        voicing="cycle",
        description="切分的固定音型",
    ),
    Texture(
        "tremolo",
        offsets=tuple(i * 0.25 for i in range(16)),
        voicing="alternate",
        description="两音快速交替（颤音）",
    ),
)

BY_NAME = {texture.name: texture for texture in VOCABULARY}
NAMES = tuple(texture.name for texture in VOCABULARY)


# --------------------------------------------------------------------------- #
# 判定规则：声学特征 -> 织体
# --------------------------------------------------------------------------- #
def classify(features: dict[str, float], *, beats_per_bar: int = 4) -> str:
    """Pick a texture from measured features.

    The rules are ordered so the most discriminating feature decides first, and each
    threshold comes from the measured separation between known textures rather than from
    taste.  Density is per second, so it is first converted to attacks per bar — a 3/4 bar
    at 143 BPM is 1.26 s and a 2/4 bar at 77 BPM is 1.56 s, and the same *figure* must map
    to the same texture on both.
    """
    bar_seconds = features.get("bar_seconds") or 1.0
    per_bar = features.get("onset_rate", 0.0) * bar_seconds
    sustain = features.get("sustain", 0.0)
    sharpness = features.get("sharpness", 0.0)
    low = features.get("low_share", 0.0)
    low_std = features.get("low_std", 0.0)

    # Nothing is happening: hold a chord.
    if per_bar < 1.5:
        return "pad"
    # A sustained bed with few attacks.
    if sustain > 0.72 and per_bar < 3.0:
        return "block_half"
    if sustain > 0.72 and per_bar < 5.0:
        return "block"
    # A steady stream of attacks.
    if per_bar >= 12.0:
        return "broken_sixteenths" if sharpness < 8.0 else "tremolo"
    if per_bar >= 9.0:
        return "triplet"
    if per_bar >= 6.5:
        return "broken_eighths"
    # Low-frequency dominated with a steady pulse: bass plus chord.
    if low > 0.45 and per_bar >= 3.0:
        return "bass_chord"
    if low_std > 0.12:
        return "ostinato"
    return "arp_updown"


# --------------------------------------------------------------------------- #
# 生成：织体 + 和弦 -> 音符
# --------------------------------------------------------------------------- #
@dataclass
class TextureNote:
    beat: float          # 以拍为单位的位置（0 起）
    pitches: list[int]
    hand: str            # "RH" / "LH"


#: Intervals heard as a simultaneity against the melody.  Within an octave a semitone,
#: whole tone, tritone or major seventh fuses into audible wrongness; beyond an octave it
#: does not (a bass two octaves down is not a clash), which is why distance is tested too.
HARSH = {1, 2, 6, 11}


def _clash(pitch: int, melody: int) -> bool:
    distance = abs(pitch - melody)
    return distance <= 12 and distance in HARSH


def avoid_melody(pitches: list[int], melody: int | None, *, chord: list[int]) -> list[int]:
    """Move any pitch that clashes with the melody to the nearest chord tone.

    This is the step that keeps a *generated* accompaniment consonant, which the
    transcriber-based attempts could never guarantee: the pitches already come from the
    chord, so a clash is a placement mistake rather than a wrong note, and it can be fixed
    by moving the note within the same harmony instead of deleting it.
    """
    if melody is None or not chord:
        return pitches
    out: list[int] = []
    for pitch in pitches:
        if not _clash(pitch, melody):
            out.append(pitch)
            continue
        candidates = sorted(
            {c + 12 * shift for c in chord for shift in (-2, -1, 0, 1, 2)},
            key=lambda c: (abs(c - pitch), _clash(c, melody)),
        )
        chosen = next((c for c in candidates if not _clash(c, melody)), pitch)
        out.append(chosen)
    return sorted(set(out))


def chord_tones(pitches: list[int]) -> list[int]:
    """Distinct pitch classes of a chord voicing, lowest first, octave duplicates removed.

    SheetSage2's voicings carry octave doublings (``Db2+Db3+F3+Ab3+C4``) which is why the
    skeleton's chords were unplayable at 23-semitone spans.  Keeping one pitch per class
    preserves the harmony and halves the span.
    """
    seen: dict[int, int] = {}
    for pitch in pitches:
        key = pitch % 12
        if key not in seen or pitch > seen[key]:
            seen[key] = pitch
    return sorted(seen.values())


def build_figure_tones(
    voicing: list[int],
    *,
    low: int = 55,
    high: int = 79,
) -> list[int]:
    """Spread a chord's pitch classes upward across ``low..high`` to give a figure notes.

    Why this is needed rather than using the voicing as given: SheetSage2's voicings are
    *harmony* statements — ``Db2+Db3+F3+Ab3+C4`` is a chord, and once its octave duplicates
    are removed it has four pitch classes, three of them low.  A broken-chord figure built
    from those four leaves two notes above middle C, so every "arpeggio" degenerates into
    ``C4 F4 C4 F4`` — two notes alternating, which is a tremolo, not an arpeggio, and is
    mechanically what made earlier attempts sound like a machine.

    Stacking the pitch classes by thirds upward produces the same harmony with enough notes
    for a figure to have a shape.  The chord is unchanged; only its spread is.
    """
    classes = sorted({pitch % 12 for pitch in voicing})
    if not classes:
        return []
    # Order the classes as a stack of thirds upward from the chord's own root, so the spread
    # reads as the chord rather than as a chromatic cluster.
    root_class = min(voicing) % 12
    ordered = sorted(classes, key=lambda c: (c - root_class) % 12)
    # Start at the first octave whose root reaches ``low``.  Computing from the chord's own
    # bottom instead put every pitch below ``low``, so the range test discarded all of them
    # and the figure came back empty.
    starts = [root_class + 12 * octave for octave in range(0, 10) if root_class + 12 * octave >= low]
    if not starts:
        return []
    out: list[int] = []
    for base in starts:
        for cls in ordered:
            pitch = base + ((cls - root_class) % 12)
            if low <= pitch <= high:
                out.append(pitch)
    return sorted(set(out))


def realise(
    texture: Texture,
    voicing: list[int],
    *,
    beats_per_bar: int,
    split: int = 60,
    bass_low: int = 36,
    bass_high: int = 55,
    chord_low: int = 55,
    chord_high: int = 79,
) -> list[TextureNote]:
    """Texture + chord voicing -> notes for one bar.

    **Register is placed, not inherited.**  An earlier version walked the chord from its
    lowest note upward and gave each note to whichever hand its pitch implied.  Because
    SheetSage2's voicings concentrate their notes low, that put 722 left-hand notes into a
    single octave (MIDI 48-59), so the left hand repeated five pitches for a whole piece.

    So the bass and the figure are placed separately: the **bass** takes the chord's root in
    ``bass_low..bass_high``, and the **figure** walks :func:`build_figure_tones`, spread over
    ``chord_low..chord_high`` and split between the hands at middle C.
    """
    tones = chord_tones(voicing)
    if not tones:
        return []
    root = tones[0]

    def place(pitch: int, low: int, high: int) -> int:
        while pitch < low:
            pitch += 12
        while pitch > high:
            pitch -= 12
        return pitch

    figure = build_figure_tones(voicing, low=chord_low, high=chord_high)
    if not figure:
        figure = [place(root, chord_low, chord_high)]
    walk = figure + figure[-2:0:-1] if len(figure) > 2 else figure
    bass_pitch = place(root, bass_low, bass_high)
    out: list[TextureNote] = []

    def emit(beat: float, pitch: int) -> None:
        out.append(TextureNote(beat, [pitch], "LH" if pitch < split else "RH"))

    def emit_stack(beat: float, pitches: list[int]) -> None:
        lower = sorted(p for p in pitches if p < split)
        higher = sorted(p for p in pitches if p >= split)
        # A right-hand stack of six notes is unplayable; keep the top three.
        if len(higher) > 3:
            higher = higher[-3:]
        if len(lower) > 3:
            lower = lower[:3]
        if lower:
            out.append(TextureNote(beat, lower, "LH"))
        if higher:
            out.append(TextureNote(beat, higher, "RH"))

    if texture.name != "pad":
        out.append(TextureNote(0.0, [bass_pitch], "LH"))

    for index, offset in enumerate(texture.offsets):
        if offset >= beats_per_bar:
            continue
        if texture.voicing == "stack":
            emit_stack(offset, figure)
        elif texture.voicing == "alternate":
            emit(offset, walk[index % len(walk)])
        else:  # "cycle"
            emit(offset, walk[index % len(walk)])
    return out


def handed(
    notes: list[TextureNote],
    *,
    melody_floor: int | None,
    split: int = 60,
) -> list[TextureNote]:
    """Give the lower half of a figure to the left hand and the upper half to the right.

    The split is placed relative to **where the melody actually is**, not at a fixed middle C.
    The melody of these songs sits at MIDI 72-89, so a fixed split leaves every arpeggio note
    above it and the right hand carries over a thousand notes while the left hand has nothing
    to do but hold a bass note — which is what the first version of this engine produced.

    ``melody_floor`` is the lowest melody pitch sounding in the bar.  Everything at or above
    it must reach the right hand (it has to be under the tune); the figure is divided just
    below that.  With no melody in the bar, the fixed ``split`` applies.
    """
    if not notes:
        return notes
    figure_pitches = sorted({p for note in notes for p in note.pitches if note.hand == "RH"})
    if melody_floor is not None and figure_pitches:
        below = [p for p in figure_pitches if p < melody_floor]
        boundary = (
            max(below) + 1 if below else min(figure_pitches)
        )
    else:
        boundary = split
    boundary = max(boundary, 40)

    out: list[TextureNote] = []
    for note in notes:
        lower = [p for p in note.pitches if p < boundary]
        upper = [p for p in note.pitches if p >= boundary]
        if lower:
            out.append(TextureNote(note.beat, sorted(lower), "LH"))
        if upper:
            out.append(TextureNote(note.beat, sorted(upper), "RH"))
    return out
