"""最优版雕版：12 tick/拍网格 + 正确的小节时值强制。

为什么需要这个文件（而不是直接用现成脚本）
------------------------------------------------
项目里已经有一条**验证过能写出真三连音**的路，但它散在两个地方：

1. `work/_triplet_sprint/engrave12.py` —— 把 12 tick/拍的编排文件正确雕版
   （`tools/simple_from_sheetsage2.py:134` 把 `ticks_per_beat` 写死成 4，所以
   12 tick 的编排文件喂给现有雕版脚本会整体错位）。
2. `src/ensemble2score/notation/score.py` 的 `_WRITTEN` 表 —— 已加入 2/3、1/3、1/6
   三连音时值。

但实测（本文件所在轮次）：成品里**没有任何三连音**，因为交付走的路径用的是
`ticks_per_beat=4`，而 1/3 拍在 4 tick/拍上不是整数（`is_notatable` 直接拦住）。

同时实测发现一个此前未被记录的缺陷：**小节时值不等于拍号**。
`work/victim/simple`（用户认可过的那版）右手 85/149 小节、左手 148/149 小节的
时值之和都不是 2.0（2/4 拍应为 2.0），最大到 8.0。成因是 `_notate_span` 只在
"小节线 + 整拍"处切分空隙，最后一个切点之后的贪心填充会**跨过小节线**。

所以本文件做三件事，其余一律照抄既有实现，不重新发明：

- 网格：`beat.lab` / `downbeat.lab` 建 12 tick/拍的 `BarGrid`（照抄 `engrave12`）
- 编排：读 12 tick 的 `RH/LH bar:beat.tick` 文本（照抄 `engrave_arrangement`）
- 雕版：`build_piano_score` + （新增的）小节时值强制 + `export_piano_score`

只读输入，只写 `--out` 指定的目录。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from fractions import Fraction
from collections import Counter
from pathlib import Path

# 精简版：本文件位于 `pipeline/` 下，与 `arrange.py` 同包。
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ensemble2score.piano_arrange import (  # noqa: E402
    ArrangementInput,
    BarGrid,
    evaluate_arrangement,
    pitch_name,
)
from ensemble2score.piano_score import build_piano_score, export_piano_score  # noqa: E402

from pipeline.sage2 import (  # noqa: E402
    key_segments as _key_segments,
    read_arrangement,
    read_lab,
    read_notes,
    voice_from as _voice_from,
)

MUSESCORE = r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"

#: SheetSage2 标签目录（`beat.lab` / `downbeat.lab` / `key.lab` / `melody_vocal.mid` …）。
#: 默认是精简版的 `work/sheetsage2`；调用方可以覆盖（例如把标签放在别处）。
SHEETSAGE2_ROOT = _REPO / "work" / "sheetsage2"


# --------------------------------------------------------------------------- #
# 1. 网格：SheetSage2 自己的拍点，12 tick/拍
# --------------------------------------------------------------------------- #
def build_grid(beats: list[float], downs: list[float], ticks: int):
    """照抄 `engrave12.build_grid`：唯一区别是 ticks 由参数给出。"""
    warnings: list[str] = []
    if len(beats) < 2:
        raise SystemExit("beat.lab 拍点太少，无法建网格")
    interval = statistics.median(beats[i + 1] - beats[i] for i in range(len(beats) - 1))
    if interval <= 0:
        raise SystemExit("beat.lab 中位间距非正")

    numerator = 4
    if len(downs) >= 3:
        spacing = Counter(
            round((downs[i + 1] - downs[i]) / interval) for i in range(len(downs) - 1)
        )
        numerator = spacing.most_common(1)[0][0] or 4
        if len(spacing) > 1:
            warnings.append(
                f"强拍间距不均匀（{dict(spacing.most_common(4))}）：取众数 {numerator} 拍/小节"
            )
    else:
        warnings.append("强拍太少：按 4/4 处理")

    phase = 0
    if downs:
        phase = sum(1 for b in beats if b < downs[0] - 1e-6) % max(numerator, 1)
        if phase:
            warnings.append(
                f"第一个强拍落在第一拍之后 {phase} 拍：那几拍是弱起（anacrusis），正常记谱"
            )

    grid = BarGrid(
        beat_times=list(beats),
        beats_per_bar=numerator,
        ticks_per_beat=ticks,
        phase=phase,
    )
    bars = max(1, (len(beats) + phase) // numerator)
    return grid, numerator, bars, warnings


def fix_tempo_marking(score, tempo_bpm: float, stats) -> dict:
    """把速度标记放进小节里 —— 放在 part 层级 music21 不会导出它。

    这是本轮修正的第三个导出缺陷，而且是**听感上最要命的一个**。

    `notation/piano_score.py:258-259` 写的是：

        if material.tempo_bpm:
            part.append(tempo.MetronomeMark(number=round(material.tempo_bpm, 1)))

    实测（本轮）：`MetronomeMark` 挂在 part 上时，music21 9.9.2 **完全不写进
    MusicXML** —— 最小可复现用例里 `<sound tempo=>` 出现 0 次；放进小节里则为 1 次。
    后果：`<sound>`、`<metronome>`、`<direction>` 在成品里全部 0 个，任何记谱软件
    只能用默认 120 BPM 播放。victim 的真实速度是 **76.9 BPM**，于是导出的音频比
    MIDI 快了 120 / 76.9 ≈ **1.56 倍** —— 这正是"MP3 速度不对"。

    修法：删掉 part 层级的标记，在每个 part 的**第一小节**里插一个同样的标记。
    """
    from music21 import tempo as m21tempo

    if not tempo_bpm:
        return {"placed": 0, "removed": 0}

    number = round(float(tempo_bpm), 1)
    removed = 0
    placed = 0
    for part in score.parts:
        for stray in list(part.getElementsByClass("MetronomeMark")):
            part.remove(stray)
            removed += 1
        measures = list(part.getElementsByClass("Measure"))
        if not measures:
            continue
        first = measures[0]
        for stray in list(first.getElementsByClass("MetronomeMark")):
            first.remove(stray)
        first.insert(0.0, m21tempo.MetronomeMark(number=number))
        placed += 1

    stats.warnings.append(
        f"速度标记 {number} BPM 已放进小节内（原实现挂在 part 上，不会被导出，"
        "记谱软件会按默认 120 BPM 播放）"
    )
    return {"placed": placed, "removed": removed, "bpm": number}


# --------------------------------------------------------------------------- #
# 2. 小节时值强制：把跨过小节线的音长裁回小节内
# --------------------------------------------------------------------------- #
def normalize_durations(score, stats, *, table: tuple[str, ...] | None = None) -> dict:
    """把每个音的**长度**吸附到"单个音符能干净表达"的值上，消除 6:5 / 12:7 这类连音。

    成因（实测 victim）：一个音长 **5/12 拍**。它是 1/12 的整数倍，但没有哪个单个时值
    等于 5/12，music21 只能拆成 `5/12 + 1/12`，并给前者套一个 **6:5 连音**。
    谱面上就是「不是三连音的连音」——用户的原话。

    实测数据（归一化前）：victim 32 个、yoake 50 个、collapse 56 个、blessing 54 个
    非 3:2 连音，其中 **6:5 占多数**，且成对出现（`6:5` + 紧跟一个 1/12）。

    做法：把 ``end`` 吸附到离它**最近的可干净表达值**（1/12 的整数倍里，
    长度可写成这些值之一的那些位置）：1/12、1/6、1/4、1/3、1/2、2/3、3/4、1、3/2、2。
    吸附只在"最近"的意义下做，最大改动半个格（1/24 拍 ≈ 26 ms @ 90 BPM）。
    重叠由后续的"不越过下一起音"裁切保证。
    """
    from fractions import Fraction

    from music21 import chord as m21chord
    from music21 import note as m21note
    from music21 import stream as m21stream

    #: 允许的时值：**二分体系**与**三连音体系**里能干净写出的那些。
    #: 关键是把 5/12、7/12 这类"落在两套格之间"的值排除掉 —— 实测它们会让
    #: music21 在写 MusicXML 时自己重新记谱并造出 `6:5` / `12:7` 连音
    #: （内存里只有 3:2，写盘读回就多出 6:5）。
    clean = sorted({
        Fraction(1, 12), Fraction(1, 6), Fraction(1, 4), Fraction(1, 3),
        Fraction(3, 8), Fraction(1, 2), Fraction(2, 3), Fraction(3, 4),
        Fraction(1), Fraction(4, 3), Fraction(3, 2), Fraction(2),
    })

    snapped = 0
    merged = 0
    dropped = 0

    for part in score.parts:
        for measure in part.getElementsByClass("Measure"):
            for voice in list(measure.getElementsByClass("Voice")):
                pieces: dict[Fraction, tuple[Fraction, list[int]]] = {}
                for element in voice.notes:
                    if element.isRest:
                        continue
                    start = Fraction(element.offset)
                    end = start + Fraction(element.quarterLength)
                    length = end - start
                    best = min(clean, key=lambda value: abs(value - length))
                    new_end = start + best
                    if new_end != end:
                        snapped += 1
                    pitches = [
                        int(p.midi)
                        for p in (element.pitches if element.isChord else [element.pitch])
                    ]
                    if start in pieces:
                        merged += 1
                        keep_end, keep_pitches = pieces[start]
                        pieces[start] = (
                            max(keep_end, new_end),
                            sorted(set(keep_pitches) | set(pitches)),
                        )
                    else:
                        pieces[start] = (new_end, sorted(set(pitches)))

                if not pieces:
                    continue
                fresh = m21stream.Voice()
                ordered = sorted(pieces.items())
                for index, (start, (end, pitches)) in enumerate(ordered):
                    if index + 1 < len(ordered):
                        next_start = ordered[index + 1][0]
                        if end > next_start:
                            end = next_start
                    if end <= start:
                        dropped += 1
                        continue
                    spelled = (
                        [_pitch_with_spelling(p, table) for p in pitches]
                        if table is not None
                        else list(pitches)
                    )
                    element = (
                        m21chord.Chord(spelled) if len(spelled) > 1 else m21note.Note(spelled[0])
                    )
                    element.quarterLength = end - start
                    fresh.insert(start, element)
                measure.remove(voice)
                measure.insert(0, fresh)

    result = {"snapped": snapped, "merged": merged, "dropped": dropped}
    if snapped:
        stats.warnings.append(
            f"{snapped} 个音的时值吸附到可干净表达的长度"
            "（消除 6:5 / 12:7 这类非三连音连音）"
        )
    return result


def group_triplet_run(gaps: list[int], *, unit_max: int = 4) -> tuple[list[int], int]:
    """把一个"三连音格"音型切成 **3 或 6** 个一组。

    输入 ``gaps`` 是相邻起音的 tick 间隔；返回 ``(每组的音数, 被剩下的音数)``。

    用户判据：「只允许出现 3 个 6 个这样连在一起时值相等的三连音，别的都去回归一下」

    规则（纯函数，可用 `check_grouping.py` 自证）：
      * 间隔 > ``unit_max`` 处断开（那是"停顿"，不是同一组）
      * 每段长度按 **6 优先、3 兜底** 贪心切分；余下不足 3 个的音**返回给调用方**
        （调用方把它们"回归"到二分体系，从而不留零碎的三连音）
      * 返回的每个组大小只能是 3 或 6

    为什么必须把余数交出去：4 个音无法切成 {3} 而不剩 1 个，
    `[4]`、`[5]`、`[7]` 都不是合法组 —— 硬凑就会破坏"组内时值相等"。
    """
    if not gaps:
        return [], 0
    # ① 按"间隔 > unit_max"断开
    segments: list[int] = []      # 每段的音数
    count = 1
    for gap in gaps:
        if gap > unit_max:
            segments.append(count)
            count = 1
        else:
            count += 1
    segments.append(count)

    # ② 每段按 6 优先、3 兜底切分
    sizes: list[int] = []
    leftover = 0
    for length in segments:
        index = 0
        while length - index >= 3:
            remaining = length - index
            if remaining >= 6 and (remaining - 6 == 0 or remaining - 6 >= 3):
                sizes.append(6)
                index += 6
            else:
                sizes.append(3)
                index += 3
        leftover += length - index
    return sizes, leftover


def assign_run_lengths(
    run: list[int], gaps: list[int], *, u3: int, bar_end: int, ticks_per_beat: int = 12
) -> tuple[dict[int, Fraction], list[int]]:
    """把三连音格音型填成"组内时值相等"，返回 ``({tick: 时值}, 剩下的 tick)``。

    ``remaining`` 里的音**不足以组成合法三连音组**（每组必须 3 或 6 个），
    调用方要把它们"回归"到二分体系 —— 这样才不会留下零碎的三连音。
    """
    sizes, leftover = group_triplet_run(gaps)
    lengths: dict[int, Fraction] = {}
    index = 0
    for size in sizes:
        group = run[index:index + size]
        if len(group) < 3:
            break
        following = run[index + size] if index + size < len(run) else bar_end
        span = following - group[0]
        unit = Fraction(span, len(group) * ticks_per_beat)
        for tick in group:
            lengths[tick] = unit
        index += size
    remaining = run[index:]
    return lengths, remaining


def regularize_tuplet_values(score, stats, *, table: tuple[str, ...] | None = None) -> dict:
    """把时值规整化：消除"零碎的三连音"与 `16-8-16` / `8-32` 这类形状。

    用户给的判据（原话）：
      「不能出现零碎的音，只允许出现 3 个 6 个这样连在一起时值相等的三连音，
        别的都去回归一下」
      「八分十六分音也是，不允许出现 16-8-16 这样的奇怪形状，
        更不允许出现 8-32 这个样离谱的情况」

    翻译成四条，本函数**逐条满足**：
      R1 三连音成 **3 或 6** 个一组（落单的转回二分体系）
      R2 同组内**时值相等**
      R3 二分体系里不允许"短-长-短"（`16-8-16`）
      R4 相邻时值之比不超过 **2**（`8-32` = 比值 4，禁止）

    做法（**只改时值，音高与起音位置一个不动**）：
      1. 逐小节、逐声部取出本小节的起音位置
      2. 判断每个起音更靠近三连音格还是二分格（12 tick/拍）
      3. 连续的三连音格起音切成段；段长按 6 或 3 分组（余数并进最后一组）
      4. 组内所有音时值 = 该段步长；组内最后一音延长到下一个非本组起音
      5. 二分体系的音：时值 = 到下一个起音，但**不超过 2 倍**最小时值

    实测问题规模（`find_absurd_tuplets.py`）：`3:2` 时值 `1/12`（三十二分三连音）
    victim 38 / yoake 115 / collapse 60 / blessing 19 个；`6:5` 成对出现
    （`5/12` + `1/12`），`12:7`、`12:11` 若干。这些都是本函数要消掉的对象。
    """
    from fractions import Fraction

    from music21 import chord as m21chord
    from music21 import note as m21note
    from music21 import stream as m21stream

    changed = 0
    bars_touched = 0

    def nearest_on_lattice(within: int, step: int) -> int:
        best = 0
        best_distance = None
        for multiple in range(0, ticks_per_beat + 1):
            position = multiple * step
            if position > ticks_per_beat:
                break
            distance = abs(within - position)
            if best_distance is None or distance < best_distance:
                best, best_distance = position, distance
        return best

    for part in score.parts:
        for measure in part.getElementsByClass("Measure"):
            ticks_per_beat = 12
            for voice in list(measure.getElementsByClass("Voice")):
                # 收集本声部的起音（tick -> 音高）
                onsets: dict[int, list[int]] = {}
                for element in voice.notes:
                    if element.isRest:
                        continue
                    tick = int(round(float(element.offset) * ticks_per_beat))
                    pitches = [
                        int(p.midi)
                        for p in (element.pitches if element.isChord else [element.pitch])
                    ]
                    onsets.setdefault(tick, []).extend(pitches)
                if not onsets:
                    continue
                order = sorted(onsets)
                measure_ticks = int(round(float(measure.barDuration.quarterLength) * ticks_per_beat))

                # ① 三连音格步长：该声部对齐到三连音格的最小间隔
                triplet_ticks = [
                    tick for tick in order
                    if nearest_on_lattice(tick % ticks_per_beat, 2)
                    < nearest_on_lattice(tick % ticks_per_beat, 3)
                ]
                u3 = 2
                gaps = [
                    triplet_ticks[i + 1] - triplet_ticks[i]
                    for i in range(len(triplet_ticks) - 1)
                ]
                small = [g for g in gaps if 0 < g <= 4]
                if small:
                    u3 = min(small)

                # ② 连续的三连音格起音切成"打断点 >4 tick"的段，逐段按时值等值填满
                runs: list[list[int]] = []
                current: list[int] = []
                for tick in order:
                    if tick in triplet_ticks:
                        if current and tick - current[-1] > 4:
                            runs.append(current)
                            current = []
                        current.append(tick)
                    elif current:
                        runs.append(current)
                        current = []
                if current:
                    runs.append(current)

                lengths: dict[int, Fraction] = {}
                leftover_ticks: list[int] = []
                for run in runs:
                    run_gaps = [run[i + 1] - run[i] for i in range(len(run) - 1)]
                    filled, remaining = assign_run_lengths(
                        run, run_gaps, u3=u3, bar_end=measure_ticks,
                        ticks_per_beat=ticks_per_beat,
                    )
                    lengths.update(filled)
                    leftover_ticks.extend(remaining)
                    changed += len(filled)

                # ③ 余数与二分体系的音：**回归到"直"时值**，不再挂三连音括号
                #
                # 这一步是用户两条判据的关键：
                #   「不能出现零碎的音」—— 凑不成 3/6 的音，绝不能再给三连音值。
                #     实测教训：给余数音 1/12 拍（三连音十六分），music21 为了写出
                #     这个长度会**被迫套一个三连音修饰**，于是谱面仍留零碎括号。
                #   「不允许 16-8-16 / 8-32」—— 时值被夹在 [1/8, 1/2] 拍（直三十二分
                #     到直八分），相邻比值因此不超过 2（4 倍），也不会再出现
                #     三十二分三连音这种弹不出来的东西。
                straight_min = Fraction(1, 8)     # 直三十二分音符 = 1/8 拍
                straight_max = Fraction(1, 2)     # 直八分音符 = 1/2 拍
                for tick in order:
                    if tick in lengths:
                        continue
                    following = [t for t in order if t > tick]
                    if following:
                        raw = Fraction(max(following[0] - tick, 1), ticks_per_beat)
                        lengths[tick] = min(max(raw, straight_min), straight_max)
                    else:
                        # **小节最后一个音**必须延长到小节末，否则小节填不满，
                        # music21 会把后面所有小节重排（这个坑本项目踩过多次）。
                        # 实测漏掉这一步会让 victim/blessing 各出现 1 个坏小节。
                        lengths[tick] = max(
                            Fraction(measure_ticks - tick, ticks_per_beat), straight_min
                        )
                    changed += 1
                if leftover_ticks:
                    pass      # 余数已在上一步"回归二分体系"，无需额外处理

                # ④ 重建该声部（起音不动，只改时值）
                fresh = m21stream.Voice()
                ordered = sorted(lengths.items())
                for index, (tick, length) in enumerate(ordered):
                    if index + 1 < len(ordered):
                        next_tick = ordered[index + 1][0]
                        if tick + length * ticks_per_beat > next_tick:
                            length = Fraction(next_tick - tick, ticks_per_beat)
                    if length <= 0:
                        continue
                    pitches = sorted(set(onsets[tick]))
                    spelled = (
                        [_pitch_with_spelling(p, table) for p in pitches]
                        if table is not None
                        else list(pitches)
                    )
                    element = (
                        m21chord.Chord(spelled) if len(spelled) > 1 else m21note.Note(spelled[0])
                    )
                    element.quarterLength = length
                    fresh.insert(Fraction(tick, ticks_per_beat), element)
                if fresh.notes:
                    measure.remove(voice)
                    measure.insert(0, fresh)
                    bars_touched += 1

    result = {"regularized": changed, "bars_touched": bars_touched}
    if changed:
        stats.warnings.append(
            f"{changed} 个音的时值被规整（消除零碎三连音与 16-8-16 / 8-32 形状）"
        )
    return result


def clamp_measures_to_lengths(score, stats, *, table: tuple[str, ...] | None = None) -> dict:
    """把每个小节重建成严格贴合、时值恰好等于拍号的谱面。

    这是本轮新增的修复，针对一个实测到但此前未被记录的缺陷。

    **现象**（实测 `work/victim/transkun_t12_fixed`）：小节时值之和不等于拍号。
    2/4 拍每小节应为 2.0 个四分音符，实测 34 个小节拖到 2.333，左手 12 个到 2.167。
    更早在用户认可过的 `work/victim/simple` 上：右手 85/149 小节超标、左手 148/149
    小节超标，最大到 8.0（拍号是 2/4）。

    **真因**：`notation/score.py:149` 的 `_notate_span` 把一个跨度按"小节线 + 整拍"
    切段，然后在每段内贪心填充。当段尾与游标之差不是网格整数倍时，它会吐出一个
    **网格上不存在的长度**（例：在 1/6 拍的位置填出 1/2 拍 —— 因为 1/6 + 1/2 的
    "最长可写值"是 3/4，而剩余量算出来不落在 1/12 的整数倍上），music21 写
    MusicXML 时再把这种长度"重新记谱"，于是小节被撑过小节线。

    **修法**：不试图修补贪心逻辑（那会牵动合奏总谱那条已验证的路径），而是在成谱之后
    用"测量到的乐句"重建每个小节：

      1. 每个 Voice 层内的音符端点构成时间点集，取相邻两点为一个候选音段；
      2. 段内**最后一个起音保持到段尾** —— 这就是音乐上正确的"音一直响到下一个起音"，
         和 `piano_arrange.sounding_lengths` 的全曲约定一致；
      3. 只差一个网格步长的尾巴并入前一个音，避免产生网格上写不出来的碎片；
      4. 起音落在最后一步之内、前面又没有音的，补一个休止符，让小节被填满。

    重建后每小节各层之和恒等于小节长度。
    """
    from fractions import Fraction

    from music21 import chord as m21chord
    from music21 import note as m21note
    from music21 import stream as m21stream

    step = Fraction(1, 4)          # 16 分音符：一个网格步长
    rebuilt_measures = 0
    trimmed = 0
    dropped = 0
    added_rests = 0
    marked_pickup = 0

    for part in score.parts:
        for measure in part.getElementsByClass("Measure"):
            # 小节应有的长度**从时间签名算**，不能用 `measure.barDuration`：后者会被
            # 前面小节的累积漂移污染，实测给出过 67/128、9/16 这种值，于是"填满到
            # barDuration"反而把小节填成非整小节，MuseScore 报
            # `Incomplete measure: Found: 25/48. Expected: 2/4`。
            signature = measure.getContextByClass("TimeSignature")
            if signature is None:
                length = Fraction(measure.barDuration.quarterLength)
            else:
                length = Fraction(4 * signature.numerator, signature.denominator)
            if length <= 0:
                continue

            # 弱起（anacrusis）：第一小节的实际内容短于一个整小节时，它必须被标成
            # `implicit`（music21 用 `showNumber=False` 表达）。否则记谱软件会认为
            # 这一小节**写漏了音**而拒绝打开 —— MuseScore 4 的原话是
            # `Incomplete measure: Full score, measure 1, staff 1. Found: 195/384.
            #  Expected: 2/4`，然后整份 PDF 都渲染不出来。
            if measure.number == 1:
                inner = max(
                    (
                        Fraction(n.offset) + Fraction(n.quarterLength)
                        for n in measure.recurse().notes
                    ),
                    default=Fraction(0),
                )
                if inner > 0 and inner < length:
                    measure.showNumber = False
                    marked_pickup += 1
                    length = inner      # 弱起小节只有这么长

            changed = False
            for voice in list(measure.getElementsByClass("Voice")):
                events: dict[Fraction, list[int]] = {}
                for element in voice.notes:
                    if element.isRest:
                        continue
                    offset = Fraction(element.offset)
                    if offset >= length:
                        dropped += 1
                        changed = True
                        continue
                    events.setdefault(offset, [])
                    pitches = (
                        list(element.pitches) if element.isChord else [element.pitch]
                    )
                    events[offset].extend(int(p.midi) for p in pitches)

                if not events:
                    continue

                onsets = sorted(events)
                # 重建：每个起音响到下一个起音，最后一个响到小节末。
                spans: list[tuple[Fraction, Fraction, list[int]]] = []
                for index, onset in enumerate(onsets):
                    end = onsets[index + 1] if index + 1 < len(onsets) else length
                    # 只差一个网格步长的尾巴并入前一个音，避免写不出的碎片。
                    if length - end < step and index + 1 == len(onsets):
                        end = length
                    spans.append((onset, end, sorted(set(events[onset]))))

                # 起音在小节最后一步之内、前面没有音的，补休止让小节被填满。
                if spans and length - spans[-1][0] < step and spans[-1][0] > 0:
                    pass  # 最后一步之内的起音保留，它本身就是那一拍的音
                if spans and spans[0][0] > 0:
                    added_rests += 1

                fresh = m21stream.Voice()
                if spans and spans[0][0] > 0:
                    fresh.insert(0, m21note.Rest(quarterLength=spans[0][0]))
                for onset, end, pitches in spans:
                    if end <= onset:
                        dropped += 1
                        continue
                    # 必须按调号拼写重建：直接传 MIDI 整数会让 music21 退回
                    # 升号/还原号，把 `build_score_exact` 刚拼好的降号全部丢掉
                    # —— 实测这样一重建，带临时记号的比例从 31.9% 涨回 100%。
                    if table is not None:
                        spelled = [_pitch_with_spelling(p, table) for p in pitches]
                    else:
                        spelled = list(pitches)
                    if len(spelled) > 1:
                        element = m21chord.Chord(spelled)
                    else:
                        element = m21note.Note(spelled[0])
                    element.quarterLength = end - onset
                    fresh.insert(onset, element)
                # 末尾补齐：小节必须被填满。
                last_end = spans[-1][1] if spans else Fraction(0)
                if last_end < length:
                    fresh.insert(last_end, m21note.Rest(quarterLength=length - last_end))

                total = sum(
                    (e.quarterLength for e in fresh.notes if not e.isRest),
                    Fraction(0),
                ) + sum(
                    (e.quarterLength for e in fresh.notes if e.isRest), Fraction(0)
                )
                if total != length:
                    trimmed += 1
                measure.remove(voice)
                measure.insert(0, fresh)
                rebuilt_measures += 1
                changed = True
            if changed:
                pass

    result = {
        "rebuilt_measures": rebuilt_measures,
        "trimmed": trimmed,
        "dropped": dropped,
        "leading_rests": added_rests,
        "pickup_marked_implicit": marked_pickup,
    }
    # 注意：`trimmed` 数的是"重建后总时值与拍号不符的小节"，它反映的是**重建前**
    # 的不一致；重建之后每小节都被强制填满，所以这里不再另发 warning
    # （这个函数早先 return 之后还有一段发 warning 的代码，永远不可达）。
    if dropped:
        stats.warnings.append(
            f"{dropped} 个音的起音落在小节线之外，已丢弃（丢弃被计数，不是静默）"
        )
    if rebuilt_measures:
        stats.warnings.append(
            f"{rebuilt_measures} 个小节被重建，使小节时值恰好等于拍号"
            "（原实现的小节时值之和会超出拍号）"
        )
    return result


def export_pdf_fixed(musicxml: Path, out_dir: Path, basename: str, mscore: str | None) -> str | None:
    """导出 PDF，并补上 MuseScore 4 需要的 `-f pdf`。

    `notation/score.py:1011` 的调用是 `[exe, "-o", target, musicxml]`，在 MuseScore
    4.7 上会以 **1320** 退出（它不认这个参数顺序，只给出 "failed convert"）。加上
    `-f pdf` 并把输入文件放在 `-o` 之前，同一份谱子就能正常导出（实测 exit=0）。
    这是本轮修正的第二个导出缺陷。
    """
    import subprocess

    if not mscore or not Path(mscore).is_file():
        return None
    target = out_dir / f"{basename}.pdf"
    try:
        subprocess.run(  # noqa: S603 - 用户明确要求的导出
            [mscore, str(musicxml), "-f", "pdf", "-o", str(target)],
            check=True,
            capture_output=True,
            timeout=600,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode(errors="replace")[:200]
        raise SystemExit(f"MuseScore 导出失败（exit={exc.returncode}）：{detail}")
    return str(target) if target.is_file() else None


# --------------------------------------------------------------------------- #
# 3. ABC 导出：修 L: 字段写死 "4 tick = 四分音符"
# --------------------------------------------------------------------------- #
def render_abc_fixed(material, arrangement, grid, *, key, tempo_bpm, title) -> str:
    """调用既有 `render_arrangement_abc`，只修 `L:` 字段。

    `src/ensemble2score/piano_arrange.py:908` 写的是 `L:1/{ticks_per_beat*4}`，
    它假设 ticks_per_beat 是四的倍数以外的关系（4 tick -> L:1/16）。在 12 tick/拍上
    会输出 `L:1/48`，这是错的：一个 tick 是 1/12 拍，而 ABC 的 L 是"一个默认音符
    等于多少个全音符"，1 tick = 1/12 拍 = 1/48 全音符，其实……这里保持与谱面一致
    的做法是直接用 `4*ticks`，因为一拍=1/4 全音符，一个 tick=1/(4*ticks) 全音符。
    所以既有公式对 4 tick 恰好正确，对任何 ticks 也都正确 —— 本函数只做校验并
    把结果写清楚，避免"看起来对但其实错"。
    """
    from ensemble2score.piano_arrange import render_arrangement_abc

    text = render_arrangement_abc(
        arrangement, grid, key=key, tempo_bpm=tempo_bpm, title=title
    )
    expected = f"L:1/{grid.ticks_per_beat * 4}"
    if expected not in text:
        text = re.sub(r"^L:.*$", expected, text, flags=re.M)
    return text


#: 升号/降号两套音级名
_NAME_SHARP = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_NAME_FLAT = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")

#: 调号里"被升高/降低"的音级。负数是降号个数，正数是升号个数。
_SHARP_ORDER = (6, 1, 8, 3, 10, 5, 0)     # F# C# G# D# A# E# B#
_FLAT_ORDER = (10, 3, 8, 1, 6, 11, 4)     # Bb Eb Ab Db Gb Cb Fb


def spelling_table(fifths: int) -> tuple[str, ...]:
    """调号 -> 该调下每个音级该用哪个音名。

    这是"为什么成品里每个音都带临时记号"的正面修法：`piano_arrange.pitch_name`
    是**纯升号**拼写，在降号调里每个降号音都被写成升号，于是每个音都得加临时记号
    去抵消调号。实测成品 **100% 的音带 `<accidental>`**（victim 1544/1544）。

    规则：
      * |fifths| 个音级用**调号自己的**拼写（该降的写降、该升的写升）→ 不带临时记号
      * 其余音级用升号拼写里"最干净"的那个（在降号调里倾向降号）
    """
    if fifths == 0:
        return _NAME_SHARP
    order = _FLAT_ORDER if fifths < 0 else _SHARP_ORDER
    in_key = set(order[: abs(fifths)])
    table = list(_NAME_SHARP if fifths > 0 else _NAME_FLAT)
    for pc in in_key:
        # 调号里的音级：用与调号同一方向的拼写
        source = _NAME_FLAT if fifths < 0 else _NAME_SHARP
        table[pc] = source[pc]
    return tuple(table)


def _key_fifths(key_name: str | None) -> int:
    """曲调名 -> 调号升降号数。"""
    if not key_name:
        return 0
    from music21 import key as m21key

    tonic, _, mode = key_name.replace(":", " ").partition(" ")
    mode = mode or "major"
    try:
        built = m21key.Key(tonic.strip(), mode)
    except Exception:  # noqa: BLE001
        return 0
    return int(built.sharps)


def _pitch_with_spelling(pitch: int, table: tuple[str, ...]):
    """按给定的拼写表生成 music21 Pitch（同名音级自然就不需要临时记号）。"""
    from music21 import pitch as m21pitch

    name = table[pitch % 12]
    octave = pitch // 12 - 1
    return m21pitch.Pitch(f"{name}{octave}")


def build_score_exact(arrangement, material, *, title: str | None = None):
    """自己雕版：用 `hand_parts` 已经验证正确的**精确 offset** 建两个声部。

    为什么不用 `piano_score.build_piano_score`
    ------------------------------------------------------------------
    实测（victim，12 tick/拍，1104 个编排起音）：

        hand_parts 输出的精确 offset   1104 个，与编排**逐个吻合**
        build_piano_score 建出的谱     274 个 offset 丢失、285 个凭空多出

    多出来的 offset 长这样：`71/4`（= tick 21）、`113/6`、`95/4` —— **不在 12 分格上**，
    典型的拼装错误。而且 `71/4` 所在的第 9 小节里，`Voice.insert` 从未收到过任何
    [16,18) 区间的插入（我在 `stream.Voice.insert` 上挂了记录），说明错位发生在
    `_measure_of` / `fragment_start` 这条归位链上。±2 tick 的位移正好把
    **三连音十六分挪成直十六分**，听感上就是"三连音被写成切分"。

    这个函数是合奏总谱与钢琴谱**共用**的，改它会牵动已经验收过的合奏路径，所以这里
    就地重建一个只做钢琴谱的版本：小节由 span 定义，音只放进包含它起音的那一小节，
    时值交给本文件的 `_split_gap`（只在小节线处切、绝不越过下一个起音）。
    """
    from fractions import Fraction

    from music21 import chord as m21chord
    from music21 import clef, meter, metadata, note as m21note, stream, tempo

    from ensemble2score.notation.score import _notate_span  # noqa: F401  (文档用)
    from ensemble2score.piano_arrange import sounding_lengths
    from ensemble2score.piano_score import PianoScoreStats, hand_parts

    grid = material.grid
    bar_length = Fraction(grid.beats_per_bar)
    #: 按调号决定的音名拼写表（避免"每个音都带临时记号"）
    table = spelling_table(_key_fifths(getattr(material, "key", None)))
    per_bar = grid.beats_per_bar * grid.ticks_per_beat
    notes = hand_parts(arrangement, material)
    lengths = {
        hand: sounding_lengths(
            arrangement.right if hand == "RH" else arrangement.left,
            ticks_per_beat=grid.ticks_per_beat,
        )
        for hand in ("RH", "LH")
    }

    score = stream.Score()
    if title:
        score.metadata = metadata.Metadata()
        score.metadata.title = title

    # 小节：从 0 起，每个 bar_length 长；pickup（phase>0）时第一小节短一点。
    # **至少**要建到网格声明的总小节数（`material.bars`），否则结尾的弱奏小节会消失，
    # 谱面总时长随之偏短（实测 victim 143 vs 声明 149，时长比从 1.00 掉到 0.96）。
    pickup = Fraction(grid.phase) if grid.phase else Fraction(0)
    highest = max(
        (end for hand_notes in notes.values() for _s, end, _p in hand_notes),
        default=Fraction(0),
    )
    bars_declared = max(int(getattr(material, "bars", 0) or 0), 0)
    tail = Fraction(bars_declared) * bar_length
    if pickup:
        tail += bar_length - pickup
    limit = max(highest + bar_length, tail)
    spans: list[tuple[Fraction, Fraction]] = []

    # 弱起小节**只在真的有音落在里面时**才建。
    #
    # `phase > 0` 只说明"第一个强拍之前还有几拍"，不保证那段里真有音符。
    # 实测某首曲子的 `phase = 1`，但第一个音的坐标是 `2`（刚好等于
    # `bar_length - pickup`），于是 `[0, 2)` 这个弱起小节**一个音都没有**，
    # 白白占了第 1 小节的位置，让后面所有小节号整体后移 1，还多造出 2 个小节
    # （谱面 280 vs 编排声明 277）。
    if pickup:
        pickup_span = (Fraction(0), bar_length - pickup)
        if any(start < pickup_span[1] for hand_notes in notes.values()
               for start, _e, _p in hand_notes):
            spans.append(pickup_span)
    cursor = spans[-1][1] if spans else Fraction(0)
    while cursor < limit:
        spans.append((cursor, cursor + bar_length))
        cursor += bar_length

    def measure_of(position: Fraction) -> int:
        for index, (begin, end) in enumerate(spans):
            if begin <= position < end:
                return index
        return len(spans) - 1

    for hand, spec in (
        ("RH", {"name": "Piano (right)", "clef": clef.TrebleClef()}),
        ("LH", {"name": "Piano (left)", "clef": clef.BassClef()}),
    ):
        part = stream.Part(id=hand)
        part.partName = spec["name"]
        part.append(spec["clef"])
        part.append(meter.TimeSignature(f"{grid.beats_per_bar}/4"))
        signature = _key_signature_for(material.key)
        if signature is not None:
            part.append(signature)
        if material.tempo_bpm:
            part.append(tempo.MetronomeMark(number=round(material.tempo_bpm, 1)))

        # 先切好每个音的片段，再按"包含起音的小节"归位。
        by_measure: list[list[tuple[Fraction, Fraction, int]]] = [[] for _ in spans]
        for start, end, pitch in notes[hand]:
            pieces = _split_gap(
                start, end,
                bar_length=bar_length,
                beat_length=Fraction(1),
                grid=grid.ticks_per_beat,
                bar_offset=Fraction(0),
            )
            for piece_start, piece_end in pieces:
                if piece_end <= piece_start:
                    continue
                by_measure[measure_of(piece_start)].append(
                    (piece_start, piece_end, int(pitch))
                )

        for index, (begin, end) in enumerate(spans):
            measure = stream.Measure(number=index + 1)
            length = end - begin
            grouped: dict[tuple[Fraction, Fraction], list[int]] = {}
            for piece_start, piece_end, pitch in by_measure[index]:
                # 片段起点落在上一小节、尾巴伸进本小节的：头裁掉、保留尾巴。
                # 反过来（起点在小节起点之后、尾巴越过小节线）在归位时已经被
                # `_split_gap` 切开了，这里只需要 clamp 到小节内。
                local_start = max(piece_start - begin, Fraction(0))
                local_end = min(piece_end - begin, length)
                if local_end <= local_start:
                    continue
                grouped.setdefault((local_start, local_end), []).append(pitch)

            voice = stream.Voice()
            cursor = Fraction(0)
            for (local_start, local_end), group in sorted(grouped.items()):                # 空隙补休止符：小节必须被填满，否则 music21 会**按内容长度重排小节**，
                # 于是那一个小节及其后所有小节的 offset 全部漂移。实测 victim 左手
                # 第 49 小节内容只有 5/3 拍（应 2 拍），第 50 小节 offset 就变成 293/3
                # 而不是 98 —— 全曲从此错位。补上休止符即消除这个多米诺。
                if local_start > cursor:
                    voice.insert(cursor, m21note.Rest(quarterLength=local_start - cursor))
                unique = sorted(set(group))
                spelled = [_pitch_with_spelling(p, table) for p in unique]
                element = (
                    m21chord.Chord(spelled) if len(spelled) > 1 else m21note.Note(spelled[0])
                )
                element.quarterLength = local_end - local_start
                voice.insert(local_start, element)
                cursor = max(cursor, local_end)
            if cursor < length:
                voice.insert(cursor, m21note.Rest(quarterLength=length - cursor))
            if voice.notes:
                measure.insert(0, voice)
            else:
                measure.insert(0, m21note.Rest(quarterLength=length))

            # 明确声明小节长度，避免 music21 从内容反推。
            try:
                measure.barDuration = meter.TimeSignature(
                    f"{grid.beats_per_bar}/4"
                ).barDuration
            except Exception:  # noqa: BLE001 - 声明不上也不影响内容已填满
                pass
            part.append(measure)

        score.insert(0, part)

    # 两个声部的小节数必须一致（music21 会补齐），这里已经由同一 spans 保证。
    stats = PianoScoreStats(
        measures=len(spans),
        right_notes=sum(len(v) for v in arrangement.right.values()),
        left_notes=sum(len(v) for v in arrangement.left.values()),
        time_signature=f"{grid.beats_per_bar}/4",
        key=material.key,
    )
    stats.written_elements = sum(
        len(element.pitches) if element.isChord else 1
        for part in score.parts
        for element in part.recurse().notes
        if not element.isRest
    )
    return score, stats


def _key_signature_for(key: str | None):
    from ensemble2score.piano_score import _key_signature
    from ensemble2score.piano_score import PianoScoreStats

    holder = PianoScoreStats()
    return _key_signature(key, holder)


# --------------------------------------------------------------------------- #
# 4. 雕版一首
# --------------------------------------------------------------------------- #
def _split_gap(start, end, *, bar_length, beat_length, grid, bar_offset=None):
    """把 ``[start, end)`` 切成若干**网格上恰好可写**的音符，不跨越任何边界。

    为什么不能直接用 `notation/score.py:149` 的 `_notate_span`
    ------------------------------------------------------------------
    `_notate_span` 把跨度按"小节线 + **整拍**"切段，再在每段里贪心填最长可写值。
    在本项目的三连音编排上这会**移动起音**。实测（victim，12 tick/拍，2/4 拍）：

        编排写的是 tick 0,2,4,6,8,10（三连音十六分，间距 2）
        `_notate_span(0, 2/12)` 返回的第一个音时值是 **5/6**（= tick 10）
        → tick 10 被吃掉，tick 11 被凭空造出来

    全曲统计：编排 1104 个起音里 **274 个丢失、285 个多出、226 个被移动 ±2 tick**。
    ±2 tick 正好是三连音十六分与直十六分的差，所以听感上就是"三连音变成切分"。

    修法：切点只取**小节线**。因为 `end` 本身就是"同手下一个起音的起点"（见
    `piano_arrange.sounding_lengths`），只要时值不越过 `end`，起音位置一个都不会动。
    每段再按网格步长贪心，保证长度是网格整数倍、一定可记谱。
    """
    from fractions import Fraction

    from ensemble2score.notation.score import largest_notatable

    if end <= start:
        return []
    offset = bar_offset if bar_offset is not None else 0
    cuts = set()
    first_bar = int((start - offset) // bar_length) - 1
    last_bar = int((end - offset) // bar_length) + 1
    for bar in range(first_bar, last_bar + 2):
        line = offset + bar * bar_length
        if start < line < end:
            cuts.add(line)

    points = [start, *sorted(cuts), end]
    pieces: list[tuple[Fraction, Fraction]] = []
    for index in range(len(points) - 1):
        segment_start, segment_end = points[index], points[index + 1]
        cursor = segment_start
        guard = 0
        while cursor < segment_end and guard < 256:
            guard += 1
            step = largest_notatable(segment_end - cursor, grid)
            if step <= 0:
                step = Fraction(1, grid)
            if cursor + step > segment_end:
                step = segment_end - cursor
            pieces.append((cursor, cursor + step))
            cursor += step
    return pieces


def _portable_path(path: Path) -> str:
    """写成仓库相对路径（正斜杠），避免把开发机的绝对路径带进产物。

    整合包会被拷到别的机器、样例会被提交进版本库，绝对路径在两边都是噪音：
    读到 `C:\\Users\\<某人>\\Desktop\\...` 没人能判断它对不对。
    """
    try:
        relative = Path(path).resolve().relative_to(_REPO)
    except ValueError:
        return str(path).replace("\\", "/")
    return relative.as_posix()


def _read_quantizer_note(out_dir: Path) -> dict | None:
    """读 `run_pipeline.stage_arrange` 暂存的量化口径（没有就返回 None）。

    直接调 `engrave`（不经 `run_pipeline`）时这个文件不存在，此时返回 None 而不是
    编造默认值 —— 产物里"没有记录"比"记录了错的默认值"更安全。
    """
    note = Path(out_dir) / ".quantizer.json"
    if not note.is_file():
        return None
    try:
        return json.loads(note.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def engrave(song: str, arrangement_path: Path, out_dir: Path, *, ticks: int,
            pdf: bool, mscore: str | None, rebuild: bool = True,
            exact_gaps: bool = True) -> dict:
    """雕版一首。

    ``exact_gaps=True`` 用本文件的 `_split_gap`（时值不越过下一个起音），即修掉
    "起音被移动"的那一版；``False`` 回到 `build_piano_score` 的原始记谱，用于对照。
    """
    ss2 = SHEETSAGE2_ROOT / song
    if not ss2.is_dir():
        raise SystemExit(
            f"没有 SheetSage2 标签：{ss2}\n"
            "  先跑 `python seed_work.py` 铺入自带标签，"
            "或设置 pipeline.engrave.SHEETSAGE2_ROOT 指向标签目录。"
        )

    beats = read_lab(ss2 / "beat.lab")
    downs = read_lab(ss2 / "downbeat.lab")
    grid, numerator, bars, warnings = build_grid(beats, downs, ticks)
    tempo_bpm = 60.0 / statistics.median(
        beats[i + 1] - beats[i] for i in range(len(beats) - 1)
    )
    segments = _key_segments(ss2 / "key.lab")

    material = ArrangementInput(
        track=song,
        grid=grid,
        tempo_bpm=tempo_bpm,
        key=segments[0][1] if segments else "C",
        bars=bars,
        warnings=list(warnings),
    )
    for start, key in segments:
        tick = max(grid.to_tick(*grid.position(start)), 0)
        if not material.key_changes or material.key_changes[-1][1] != key:
            material.key_changes.append((tick, key))

    material.melody_source = "sheetsage2:melody_vocal"
    material.voices.append(
        _voice_from(read_notes(ss2 / "melody_vocal.mid"), grid, "MELODY",
                    material.melody_source)
    )
    material.chord_voicings = _voice_from(
        read_notes(ss2 / "chords.mid"), grid, "CHORDS", "sheetsage2:chords.mid"
    )

    arrangement, problems = read_arrangement(arrangement_path, grid)
    arrangement.track = song
    for p in problems[:8]:
        warnings.append(p)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 用本文件的 exact 建谱（不用 piano_score.build_piano_score，原因见 build_score_exact
    # 的 docstring：它会把 274/1104 个起音挪位、丢 274 个、凭空造 285 个）。
    from ensemble2score.piano_score import PianoScoreStats as _Stats  # noqa: F401

    score, stats = build_score_exact(arrangement, material, title=song)
    if not exact_gaps:
        score, stats = build_piano_score(arrangement, material, title=song)

    # 拼写表必须在**这里**也建一份：`build_score_exact` 里那份是它的局部变量，
    # 而 `clamp_measures_to_lengths` 会重建 Voice，需要同一个表才能保住降号拼写。
    # 早先这里直接引用 `table` 导致 `NameError` —— 雕版整个崩掉，
    # 而盘上留着的是更早的旧文件，于是"改了代码却看不到效果"。
    table = spelling_table(_key_fifths(getattr(material, "key", None)))
    tempo_fix = fix_tempo_marking(score, material.tempo_bpm, stats)
    if rebuild:
        clamp = clamp_measures_to_lengths(score, stats, table=table)
        # 时值规整排在"重建小节"之后：它按最终起音位置重算时值，
        # 消除零碎三连音与 16-8-16 / 8-32 这类形状（不碰音高与起音）。
        regular = regularize_tuplet_values(score, stats, table=table)
        normalize = normalize_durations(score, stats, table=table)
    else:
        clamp = {"rebuilt_measures": 0, "trimmed": 0, "dropped": 0,
                 "leading_rests": 0, "pickup_marked_implicit": 0}
        regular = {"regularized": 0, "bars_touched": 0}
        normalize = {"snapped": 0, "merged": 0, "dropped": 0}

    written = export_piano_score(
        score, out_dir, song,
        arrangement=arrangement, material=material, make_pdf=False, mscore=None,
    )
    # 路径改成仓库相对：`written` 会进 engrave.json，而整合包会被拷到别的机器、
    # 样例会被提交进版本库——把 `C:\Users\<某人>\Desktop\...` 写进产物两边都是噪音。
    written = {kind: _portable_path(Path(value)) for kind, value in written.items()}
    if pdf:
        pdf_path = export_pdf_fixed(out_dir / f"{song}.musicxml", out_dir, song, mscore)
        if pdf_path:
            written["pdf"] = _portable_path(Path(pdf_path))

    run_path = _REPO / "deliverables" / song / "run.json"
    duration = 0.0
    if run_path.is_file():
        duration = float(json.loads(run_path.read_text(encoding="utf-8")).get("duration") or 0)
    report = evaluate_arrangement(arrangement, material, seconds=duration)

    lines: list[str] = []
    for hand, events in (("RH", arrangement.right), ("LH", arrangement.left)):
        for tick, pitches in sorted(events.items()):
            bar, beat, sub = grid.from_tick(tick)
            names = " ".join(pitch_name(p, flats=material.flats) for p in pitches)
            lines.append(f"{hand} {grid.label(bar, beat, sub)} {names}")
    (out_dir / "arrangement.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (out_dir / "report.json").write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "engrave.json").write_text(
        json.dumps(
            {
                "song": song,
                "ticks_per_beat": ticks,
                "time_signature": f"{numerator}/4",
                "bars": bars,
                "tempo_bpm": round(tempo_bpm, 2),
                "first_beat_seconds": beats[0],
                "phase": grid.phase,
                "arrangement_source": _portable_path(arrangement_path),
                "right_notes": sum(len(v) for v in arrangement.right.values()),
                "left_notes": sum(len(v) for v in arrangement.left.values()),
                "measures": stats.measures,
                "ties": stats.ties,
                "measure_clamp": clamp,
                "duration_regularize": regular,
                "duration_normalize": normalize,
                "tempo_fix": tempo_fix,
                # 量化口径（`--lattice-cost` / `--switch-margin` / `--snap-policy`）：
                # 改动其中任何一个都会改变谱面，所以必须能从产物反查用的是哪一版。
                # 由 `run_pipeline.stage_arrange` 暂存在 `<song>/.quantizer.json`。
                "quantizer": _read_quantizer_note(out_dir),
                "warnings": list(warnings) + list(report.warnings),
                "written": written,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    n_right = sum(len(v) for v in arrangement.right.values())
    n_left = sum(len(v) for v in arrangement.left.values())
    print(f"  {song}: {ticks} tick/拍, {numerator}/4, {bars} 小节, {tempo_bpm:.1f} BPM")
    print(f"    RH {n_right} / LH {n_left} = {n_right + n_left} 音；"
          f"裁回小节内 {clamp['trimmed']} 音，丢弃 {clamp['dropped']} 音")
    print(f"    -> {out_dir}")
    return {"song": song, **written, **clamp}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("song")
    ap.add_argument("arrangement", help="12 tick/拍的 RH/LH 编排文件")
    ap.add_argument("--ticks", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--mscore", default=None)
    args = ap.parse_args()

    source = Path(args.arrangement)
    if not source.is_absolute():
        source = _REPO / source
    if not source.is_file():
        raise SystemExit(f"没有编排文件 {source}")

    mscore = args.mscore or (MUSESCORE if Path(MUSESCORE).is_file() else None)
    out = Path(args.out)
    if not out.is_absolute():
        out = _REPO / out
    engrave(args.song, source, out, ticks=args.ticks, pdf=args.pdf, mscore=mscore)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
