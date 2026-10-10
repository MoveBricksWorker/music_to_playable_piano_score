"""逐小节、逐起音把成品谱面与编排文件对齐（定位"丢音 / 多音"的探针）。

`tools/check_structure.py` 给的是汇总数字，这个脚本给的是**逐例样本**——
当汇总说"编排里有 25 个起音不在谱面上"时，用这个看它们长什么样。

用法：
    python tools/diff_arrangement_vs_score.py
    python tools/diff_arrangement_vs_score.py --song victim --root work/_review_fixed
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from music21 import converter, pitch as m21pitch

ROOT = Path(__file__).resolve().parents[1]
TICKS = 12
DEFAULT_SONGS = ("victim", "yoake", "collapse", "blessing")


def arrangement_events(path: Path) -> dict[tuple[str, int, int], set[int]]:
    out: dict[tuple[str, int, int], set[int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 3 or parts[0] not in ("RH", "LH"):
            continue
        bar_text, rest = parts[1].split(":")
        beat_text, sub_text = rest.split(".")
        tick = (int(beat_text) - 1) * TICKS + int(sub_text)
        out.setdefault((parts[0], int(bar_text), tick), set()).update(
            int(round(m21pitch.Pitch(p).midi)) for p in parts[2:]
        )
    return out


def score_events(path: Path) -> dict[tuple[str, int, int], set[int]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = converter.parse(str(path))
    out: dict[tuple[str, int, int], set[int]] = {}
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
                out.setdefault((hand, measure.number, tick), set()).update(pitches)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="out")
    parser.add_argument("--song", default=None, help="只查一首")
    parser.add_argument("--show", type=int, default=5, help="每类打印多少例")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = ROOT / root
    songs = [args.song] if args.song else list(DEFAULT_SONGS)

    for song in songs:
        arrangement = root / song / "arrangement.txt"
        xml = root / song / f"{song}.musicxml"
        if not arrangement.is_file() or not xml.is_file():
            print(f"{song}: 缺文件，跳过")
            continue

        arranged = arrangement_events(arrangement)
        score = score_events(xml)
        only_arranged = sorted(set(arranged) - set(score))
        only_score = sorted(set(score) - set(arranged))
        pitches_lost = sum(len(arranged[k] - score.get(k, set())) for k in arranged)
        pitches_added = sum(len(score[k] - arranged.get(k, set())) for k in score)

        print(f"=== {song} ===")
        print(f"  编排起音 {len(arranged)}  谱面起音 {len(score)}")
        print(f"  只在编排里 {len(only_arranged)}   只在谱面里 {len(only_score)}")
        print(f"  音高不匹配：编排多 {pitches_lost} 个，谱面多 {pitches_added} 个")
        for label, items in (("编排独有", only_arranged), ("谱面独有", only_score)):
            if items:
                print(f"  {label}示例：{items[:args.show]}")
        if only_arranged:
            print("  ! 编排独有 = 起音在雕版中丢失，这是要修的一类回归")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
