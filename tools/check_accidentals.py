"""临时记号体检：写出去的记号里有多少是**调号已经覆盖**的。

`engrave._pitch_with_spelling` 用 `Pitch("B-4")` 这种写法，会**显式带上 accidental 属性**，
music21 导出时照写不误——于是调号里已经降过的音还要再标一次降号。
这个探针量化这件事，用来验证"改成只设 step/octave"以后能省掉多少记号。

用法：
    python tools/check_accidentals.py
    python tools/check_accidentals.py --root work/_review_fixed
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from music21 import converter, key as m21key

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SONGS = ("victim", "yoake", "collapse", "blessing")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="out")
    parser.add_argument("--songs", nargs="*", default=list(DEFAULT_SONGS))
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = ROOT / root

    print(f"{'曲目':<10}{'写出的记号':>11}{'调号已覆盖':>11}{'真需要':>8}{'总音数':>9}")
    for song in args.songs:
        path = root / song / f"{song}.musicxml"
        if not path.is_file():
            print(f"{song:<10} 缺文件")
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = converter.parse(str(path))

        signatures = list(score.recurse().getElementsByClass("KeySignature"))
        if not signatures:
            print(f"{song:<10} 没有调号，无法判断（先修 key.lab 解析）")
            continue
        altered = {p.name for p in m21key.KeySignature(signatures[0].sharps).alteredPitches}

        covered = needed = total = 0
        for element in score.recurse().notes:
            if element.isRest:
                continue
            for p in element.pitches:
                total += 1
                if p.accidental is None:
                    continue
                if p.name in altered:
                    covered += 1
                else:
                    needed += 1

        print(f"{song:<10}{covered + needed:>11}{covered:>11}{needed:>8}{total:>9}")

    print()
    print("「调号已覆盖」那一列就是多余的临时记号：调号里已经升降过的音级不必再标。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
