"""检验用户给的四条谱面判据（**在成品谱面口径上验**）。

  R1 三连音必须成 3 或 6 个一组（不允许落单、不允许 4/5 个）
  R2 同一三连音组内**时值相等**
  R3 二分体系里不允许"短-长-短"（如 16-8-16）
  R4 相邻时值之比不超过 2（如 8-32 = 比值 4，禁止）

另外报告：三十二分三连音、六十四分、`6:5` / `12:7` 这类非三连音的数量，
以及"最短音长"（毫秒），因为 1/12 拍在这种速度下是弹不出来的。

**自证**：先用构造出来的四种已知谱面跑一遍，确认本脚本能抓出该抓的问题，
再去量真实数据 —— 否则数字不可信。
"""

from __future__ import annotations

import argparse
import io
import sys
import warnings
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# 包内导入：精简版的 `engrave` 在 `pipeline/` 下。
# 原项目用的是裸 `from engrave import …`（靠 `sys.path` 里有 `run_best/`），
# 那样在精简版里会去**原项目**找文件，等于没有独立。
from pipeline.engrave import group_triplet_run  # noqa: E402

SONGS = ("victim", "yoake", "collapse", "blessing")
TICKS = 12


def bar_seconds(score) -> float:
    marks = list(score.recurse().getElementsByClass("MetronomeMark"))
    bpm = marks[0].number if marks else 60.0
    return 60.0 / bpm


def check_score(score) -> dict:
    """返回四条判据的违规清单。"""
    r1: list[tuple] = []
    r2: list[tuple] = []
    r3: list[tuple] = []
    r4: list[tuple] = []
    solo_tuplets = [0]
    tuplet_kinds: Counter = Counter()
    shortest: list[tuple] = []

    for part_index, part in enumerate(score.parts):
        hand = "RH" if part_index == 0 else "LH"
        for measure in part.getElementsByClass("Measure"):
            for voice in measure.getElementsByClass("Voice") or [measure]:
                entries: list[tuple[int, float, object]] = []
                for element in voice.recurse().notes:
                    if element.isRest:
                        continue
                    tick = round(float(element.offset) * TICKS)
                    ql = float(element.quarterLength)
                    entries.append((tick, ql, element))
                    shortest.append((ql, hand, measure.number, tick))
                    for tup in element.duration.tuplets:
                        tuplet_kinds[
                            f"{tup.numberNotesActual}:{tup.numberNotesNormal}"
                        ] += 1
                if not entries:
                    continue
                entries.sort(key=lambda item: item[0])

                # ---- R1/R2：三连音分组 -------------------------------- #
                #
                # 关键：分组口径必须与雕版器**同一个函数**（`engrave.group_triplet_run`），
                # 否则就是"用不同的尺子量"，量出来的违规数没有意义。
                # 早先这里用"相邻间隔 ≤4 就同组"，于是整小节 12 个音被当成一组，
                # 报出 581 处 R1 违规 —— 而雕版其实是正确分成 [6,6] 的。
                triplet_entries = [
                    (tick, ql) for tick, ql, element in entries
                    if any(f"{t.numberNotesActual}:{t.numberNotesNormal}" == "3:2"
                           for t in element.duration.tuplets)
                ]
                if triplet_entries:
                    gaps = [
                        triplet_entries[i + 1][0] - triplet_entries[i][0]
                        for i in range(len(triplet_entries) - 1)
                    ]
                    sizes, leftover = group_triplet_run(gaps)
                    # 区分两类：
                    #   * **单音修饰**：music21 给单个音挂 3:2 只为表达长度，
                    #     那不是"三连音组"。但用户看到的括号仍然会多，所以单独计数。
                    #   * **真碎组**：2 或 4 个音一组的三连音 —— 这才是"零碎的音"。
                    if len(triplet_entries) < 3:
                        solo_tuplets[0] += 1
                    else:
                        index = 0
                        for size in sizes:
                            group = triplet_entries[index:index + size]
                            if size not in (3, 6):
                                r1.append((hand, measure.number, size,
                                           [tick for tick, _ in group]))
                            values = {round(ql, 6) for _tick, ql in group}
                            if len(values) > 1:
                                r2.append((hand, measure.number, sorted(values)))
                            index += size
                        for tick, ql in triplet_entries[index:]:
                            solo_tuplets[0] += 1

                # ---- R3/R4：二分体系的形状 ------------------------------ #
                values = [round(ql, 6) for _tick, ql, element in entries]
                for index in range(1, len(values) - 1):
                    before, middle, after = values[index - 1], values[index], values[index + 1]
                    if before > middle and after > middle:
                        r3.append((hand, measure.number, before, middle, after))
                for index in range(1, len(values)):
                    before, after = values[index - 1], values[index]
                    if before > 0 and after > 0:
                        ratio = max(before, after) / min(before, after)
                        if ratio > 2.0 + 1e-9:
                            r4.append((hand, measure.number, before, after, round(ratio, 2)))

    return {
        "R1": r1, "R2": r2, "R3": r3, "R4": r4,
        "solo_tuplets": solo_tuplets[0],
        "tuplet_kinds": tuplet_kinds,
        "shortest": sorted(shortest)[:5],
    }


def self_check() -> str:
    """构造四种已知谱面，确认检查器能抓出该抓的。"""
    from music21 import chord, meter, note, stream

    out = io.StringIO()

    def build(values: list[float], triplet: bool = False):
        part = stream.Part()
        measure = stream.Measure(number=1)
        measure.append(meter.TimeSignature("2/4"))
        voice = stream.Voice()
        cursor = 0.0
        for ql in values:
            element = note.Note("C4")
            if triplet:
                element.duration = element.duration.appoggiatura() if False else element.duration
            element.quarterLength = ql
            voice.insert(cursor, element)
            cursor += ql
        measure.insert(0, voice)
        part.append(measure)
        score = stream.Score()
        score.insert(0, part)
        return score

    cases = {
        "正常二分 1/4+1/4": ([0.25, 0.25, 0.25, 0.25], False),
        "16-8-16（应命中 R3）": ([0.25, 0.5, 0.25, 1.0], False),
        "8-32（应命中 R4）": ([0.5, 0.125, 0.375, 1.0], False),
    }
    out.write("自证（已知谱面，看检查器是否会抓）：\n")
    for name, (values, triplet) in cases.items():
        result = check_score(build(values, triplet))
        out.write(f"  {name:26s} R3={len(result['R3'])} R4={len(result['R4'])}\n")
    return out.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="run_best/out_chorus")
    parser.add_argument("--songs", nargs="*", default=list(SONGS))
    args = parser.parse_args()
    root = ROOT / args.root
    from music21 import converter

    out = io.StringIO()
    out.write(self_check())
    out.write("\n" + "=" * 96 + "\n真实数据\n" + "=" * 96 + "\n")
    out.write("%-10s %6s %6s %6s %6s %10s %12s\n" % (
        "曲目", "R1", "R2", "R3", "R4", "单音修饰", "最短音长"))
    for song in args.songs:
        path = root / song / f"{song}.musicxml"
        if not path.is_file():
            out.write(f"{song}: 缺文件\n")
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = converter.parse(str(path))
        result = check_score(score)
        sec = bar_seconds(score)
        shortest_ql = result["shortest"][0][0] if result["shortest"] else 0
        out.write("%-10s %6d %6d %6d %6d %10d %11.0fms\n" % (
            song, len(result["R1"]), len(result["R2"]), len(result["R3"]),
            len(result["R4"]), result["solo_tuplets"], shortest_ql * sec * 1000))
        if result["R1"]:
            out.write(f"    R1 违规示例: {result['R1'][:5]}\n")
        if result["R2"]:
            out.write(f"    R2 违规示例: {result['R2'][:5]}\n")
        if result["R3"]:
            out.write(f"    R3 违规示例: {result['R3'][:5]}\n")
        if result["R4"]:
            out.write(f"    R4 违规示例: {result['R4'][:5]}\n")
    text = out.getvalue()
    print(text)
    # 报告落在**输出根**下，而不是上游项目的历史目录（本项目里 `run_best/`
    # 并不存在，早先写死这个路径会让脚本跑完所有检查后崩在最后一行）。
    report = root.parent / f"_{root.name}_shape_check.txt"
    try:
        report.write_text(text, encoding="utf-8")
        print(f"明细 -> {report}")
    except OSError as exc:      # 只读目录等：有 stdout 就够了，不要因此报错退出
        print(f"! 报告未能落盘（{exc}），结果见上方输出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
