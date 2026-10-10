"""成品谱面的可弹性验收（唯一可信口径）。

为什么以谱面为准：编排文件里没有"被小节线切开的音的后半段"，而那是真实存在的起音。
实测同一批谱子：编排口径数出 148 处超标，谱面口径数出 **211** 处。差的就是那些切分音。

判据（与 `tools/refine_simple.py:46-47` 一致）：
    单手跨度 ≤ 19 半音
    同时音   ≤ 5 个

另外报三项听感相关指标：左手音区分布、音符/秒、以及左右手音区是否分离。

只读，不写文件（把结果打到 stdout 与可选 JSON）。
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HAND_SPAN = 19
MAX_SIMULTANEOUS = 5


def analyse(song: str, root: Path) -> dict:
    from music21 import converter

    path = root / song / f"{song}.musicxml"
    if not path.is_file():
        return {"song": song, "error": f"没有 {path}"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = converter.parse(str(path))

    wide = 0
    crowded = 0
    span_max = 0
    registers = {"RH": Counter(), "LH": Counter()}
    note_count = 0
    onsets = 0
    worst: list[dict] = []
    durations = Counter()

    for part_index, part in enumerate(score.parts):
        hand = "RH" if part_index == 0 else "LH"
        for measure in part.getElementsByClass("Measure"):
            groups: dict[Fraction, list[int]] = {}
            for element in measure.recurse().notes:
                if element.isRest:
                    continue
                pitches = (
                    [int(p.midi) for p in element.pitches]
                    if element.isChord
                    else [int(element.pitch.midi)]
                )
                groups.setdefault(Fraction(element.offset), []).extend(pitches)
                note_count += len(pitches)
                durations[round(float(element.quarterLength), 4)] += 1
            for offset, pitches in groups.items():
                unique = sorted(set(pitches))
                onsets += 1
                span = unique[-1] - unique[0]
                span_max = max(span_max, span)
                if span > HAND_SPAN:
                    wide += 1
                    if len(worst) < 8:
                        worst.append({"song": song, "hand": hand, "measure": measure.number,
                                      "offset": float(offset), "pitches": unique, "span": span})
                if len(unique) > MAX_SIMULTANEOUS:
                    crowded += 1
                for pitch in unique:
                    registers[hand][pitch // 12] += 1

    rh_low = min((octave for octave, count in registers["RH"].items() if count), default=None)
    lh_low = min((octave for octave, count in registers["LH"].items() if count), default=None)
    return {
        "song": song,
        "wide_chords": wide,
        "crowded_onsets": crowded,
        "span_max": span_max,
        "notes": note_count,
        "onsets": onsets,
        "distinct_durations": len(durations),
        "registers_RH": {str(k): v for k, v in sorted(registers["RH"].items())},
        "registers_LH": {str(k): v for k, v in sorted(registers["LH"].items())},
        "lh_share": round(
            sum(registers["LH"].values())
            / max(sum(registers["RH"].values()) + sum(registers["LH"].values()), 1),
            4,
        ),
        "lowest_octave_RH": rh_low,
        "lowest_octave_LH": lh_low,
        "worst_examples": worst,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="run_best/out")
    parser.add_argument("--label", default="")
    parser.add_argument("--json", default=None)
    parser.add_argument("--songs", nargs="*", default=["victim", "yoake", "collapse", "blessing"])
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = ROOT / root

    rows = [analyse(song, root) for song in args.songs]
    print("=" * 100)
    print(f"可弹性验收（谱面口径）  root={root.relative_to(ROOT)}  {args.label}")
    print("=" * 100)
    print(f"{'曲目':<10}{'宽和弦':>8}{'拥挤起音':>10}{'最大跨度':>10}{'音数':>8}"
          f"{'起音':>7}{'左手占比':>10}{'左手最低八度':>13}")
    for row in rows:
        if "error" in row:
            print(f"{row['song']:<10}  {row['error']}")
            continue
        print(f"{row['song']:<10}{row['wide_chords']:>8}{row['crowded_onsets']:>10}"
              f"{row['span_max']:>10}{row['notes']:>8}{row['onsets']:>7}"
              f"{row['lh_share']:>9.1%}{str(row['lowest_octave_LH']):>13}")
    total_wide = sum(r.get("wide_chords", 0) for r in rows)
    total_crowd = sum(r.get("crowded_onsets", 0) for r in rows)
    print()
    print(f"合计：宽和弦 {total_wide} 处，拥挤起音 {total_crowd} 处 "
          f"（判据：跨度 ≤{HAND_SPAN} 半音、同时音 ≤{MAX_SIMULTANEOUS}）")
    if total_wide == 0 and total_crowd == 0:
        print(">>> 全部达标")
    else:
        print(">>> 仍未达标，最严重的几处：")
        for row in rows:
            for item in row.get("worst_examples", [])[:3]:
                print(f"    {item['song']} {item['hand']} 第{item['measure']}小节 "
                      f"offset={item['offset']:.3f} 音={item['pitches']} 跨度={item['span']}")

    if args.json:
        target = Path(args.json)
        if not target.is_absolute():
            target = ROOT / target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"明细 -> {target.relative_to(ROOT)}")
    return 0 if (total_wide == 0 and total_crowd == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
