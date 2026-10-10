"""结构体检：调号、小节是否填满、音数是否与编排一致、连音线碎片有多少。

这是审查这套管线时用的探针，**只读**，不修改任何产物。它查的是那些
"验收脚本不会报、但打开谱子就能看出来"的问题：

  1. 调号在不在（`<fifths>`）—— 曾经因为 `key.lab` 解析读错列而全线丢失；
  2. 小节内容长度是否等于拍号 —— 不等于的话 music21 / MuseScore 会重排小节；
  3. 谱面音数比编排音数多多少 —— 差额是"跨小节线的音被切成连音线碎片"，不是丢音；
  4. 编排里的起音有没有在谱面里丢掉 —— 这是最要命的一类回归。

用法：
    python tools/check_structure.py                 # 检查 out/
    python tools/check_structure.py --root work/_review_fixed
"""
from __future__ import annotations

import argparse
import warnings
from collections import Counter
from fractions import Fraction
from pathlib import Path

from music21 import converter, pitch as m21pitch

ROOT = Path(__file__).resolve().parents[1]
TICKS = 12
DEFAULT_SONGS = ("victim", "yoake", "collapse", "blessing")


def read_arrangement(path: Path) -> dict[tuple[str, int, int], set[int]]:
    """`arrangement.txt` → {(手, 小节, tick): {音高}}。"""
    out: dict[tuple[str, int, int], set[int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 3 or parts[0] not in ("RH", "LH"):
            continue
        try:
            bar_text, rest = parts[1].split(":")
            beat_text, sub_text = rest.split(".")
            pitches = {int(round(m21pitch.Pitch(p).midi)) for p in parts[2:]}
        except (ValueError, IndexError):
            continue
        out.setdefault(
            (parts[0], int(bar_text), (int(beat_text) - 1) * TICKS + int(sub_text)), set()
        ).update(pitches)
    return out


def read_score(path: Path) -> tuple[dict[tuple[str, int, int], set[int]], dict]:
    """成品 MusicXML → 同样的坐标表示，外加几项体检指标。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = converter.parse(str(path))

    events: dict[tuple[str, int, int], set[int]] = {}
    for part in score.parts:
        hand = "RH" if part.partName.endswith("(right)") else "LH"
        for measure in part.getElementsByClass("Measure"):
            for element in measure.recurse().notes:
                if element.isRest:
                    continue
                tick = int(round(float(element.offset) * TICKS))
                pitches = (
                    [int(p.midi) for p in element.pitches]
                    if element.isChord
                    else [int(element.pitch.midi)]
                )
                events.setdefault((hand, measure.number, tick), set()).update(pitches)

    signatures = [
        k.sharps for k in score.recurse().getElementsByClass("KeySignature")
    ]

    # 小节"填满"判据：该小节里**最后一个音的结束点**是否到达小节线。
    #
    # 三个坑：
    #   * 不能用"各声部时值之和"——多声部小节在同一时间轴上并行，相加是错的；
    #   * 也不能用"每个音 offset+quarterLength 之和"——同样重复计算；
    #     取**最大值**对单声部与多声部都成立；
    #   * **必须把休止符算进去**：`measure.recurse().notes` 在 music21 里会把
    #     `Rest` 排除掉（只留 Note/Chord），于是"整小节休止"的小节会被误判成空小节。
    #     实测这个坑一次就误报了 104~211 个"没填满"的小节。
    measures_bad = 0
    measures_total = 0
    pickup_short = 0
    empty = 0
    for part in score.parts:
        for measure in part.getElementsByClass("Measure"):
            measures_total += 1
            reached = 0.0
            filled = measure.recurse().getElementsByClass(("Note", "Chord", "Rest"))
            for element in filled:
                reached = max(
                    reached, float(element.offset) + float(element.quarterLength)
                )
            full = float(measure.barDuration.quarterLength)
            if not filled:
                empty += 1
            elif full > 0 and reached + 1e-6 < full:
                measures_bad += 1
                # 弱起小节（pickup）本来就短于一个整小节，不算问题
                if measure.number == 1 and not measure.showNumber:
                    pickup_short += 1
    measures_bad -= pickup_short

    ties = sum(
        1
        for n in score.recurse().notes
        if n.tie is not None and n.tie.type in ("start", "continue")
    )
    info = {
        "fifths": signatures[0] if signatures else None,
        # 弱起小节（pickup）本来就短于一个整小节，music21 解析时 `showNumber=False`，
        # 上面已从 measures_bad 里排除。
        "measures_bad": measures_bad,
        "measures_total": measures_total,
        "pickup_short": pickup_short,
        "empty_measures": empty,
        "ties": ties,
        "score_notes": sum(
            len(n.pitches) if n.isChord else 1
            for n in score.recurse().notes
            if not n.isRest
        ),
    }
    return events, info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="out", help="成品目录（默认 out）")
    parser.add_argument("--songs", nargs="*", default=list(DEFAULT_SONGS))
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = ROOT / root

    problems = 0
    print(f"结构体检  root={root}")
    print(f"{'曲目':<10}{'调号':>6}{'坏小节':>8}{'连音线':>8}"
          f"{'谱面音数':>9}{'编排起音':>9}{'丢失起音':>9}")
    for song in args.songs:
        xml = root / song / f"{song}.musicxml"
        arrangement = root / song / "arrangement.txt"
        if not xml.is_file() or not arrangement.is_file():
            print(f"{song:<10} 缺文件（{xml.name} / {arrangement.name}）")
            problems += 1
            continue

        arranged = read_arrangement(arrangement)
        score, info = read_score(xml)
        lost = sorted(set(arranged) - set(score))
        extra = sorted(set(score) - set(arranged))
        fifths = "—" if info["fifths"] is None else f"{info['fifths']:+d}"
        if info["fifths"] is None:
            problems += 1
        if info["measures_bad"]:
            problems += 1
        if lost:
            problems += 1

        print(f"{song:<10}{fifths:>6}{info['measures_bad']:>8}{info['ties']:>8}"
              f"{info['score_notes']:>9}{sum(len(v) for v in arranged.values()):>9}"
              f"{len(lost):>9}")
        if info["fifths"] is None:
            print("    ! 谱面里没有调号（<fifths> 缺失）——读 key.lab 时取错列会这样")
        if info["measures_bad"]:
            print(f"    ! {info['measures_bad']}/{info['measures_total']} 个小节没有填到小节线")
        if info["empty_measures"]:
            print(f"    ! {info['empty_measures']} 个小节里既没有音也没有休止符")
        if info["pickup_short"]:
            print(f"    · {info['pickup_short']} 个弱起小节短于整小节（正常记谱）")
        if lost:
            print(f"    ! 编排里有 {len(lost)} 个起音不在谱面上，前 5 个：{lost[:5]}")
        if extra:
            note = f"    · 谱面另有 {len(extra)} 个起音，来自跨小节线的连音线碎片（正常）"
            jumped = [
                (hand, bar, tick)
                for (hand, bar, tick) in extra
                if tick % TICKS == 0
            ]
            if jumped:
                note += f"；其中 {len(jumped)} 个落在小节起点"
            print(note)

    print()
    print("全部正常。" if not problems else f"发现 {problems} 项需要关注。")
    # 有问题的项以非零退出码回传，方便脚本/CI 直接判断（无问题返回 0）。
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
