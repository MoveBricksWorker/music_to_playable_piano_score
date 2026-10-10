"""回归闸门：对 `out/` 里的样例断言那几条**已经被验证过**的硬指标没有退化。

这不是单元测试，是**契约测试**——它绑的是"这套管线当前做到什么程度"，
而不是"某个函数的实现细节"。任何一条失败都意味着 README 里那张表要改，
或者代码引入了回归。

    python tests/regression.py                  # 用 out/
    python tests/regression.py --self-check     # 只自证检查器本身（不需要 out/）
    python tests/regression.py --root out --verbose

`--self-check` 用 [pipeline.check_shapes](../pipeline/check_shapes.py) 的自证：
构造四种已知谱面，确认检查器会抓到该抓的（16-8-16 命中 R3、8-32 命中 R4）。
在**新克隆的仓库**上（没有 `out/`）这是唯一能跑的检查，CI 也会跑它。

## 断言的三条（都是当前真的达标的）

1. **可弹性**：单手跨度 ≤19 半音、同时音 ≤5 —— 由 `pipeline.verify_playable` 计算；
2. **R1**：三连音只以 3 或 6 个一组出现（落单的已回归二分体系）；
3. **不丢音**：`arrangement.txt` 里的每一个起音都能在成品 MusicXML 里找到，
   且音高集合完全一致。

## 故意**不**断言的三条

* R2 / R3 / R4：目前**未达标**（见 README「已知缺陷」）。把它们写成断言会让 CI 常红。
  等修好了再往这里加。
* 临时记号数量：目前偏多，也还没修。
* 中段转调：只拼写不发调号，未修。

依赖：`music21` + `pretty_midi`（**不需要 torch**），所以 CI 很轻。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.check_shapes import check_score  # noqa: E402
from pipeline.verify_playable import analyse  # noqa: E402
from tools.check_structure import read_arrangement, read_score  # noqa: E402

DEFAULT_SONGS = ("victim", "yoake", "collapse", "blessing")


class Report:
    def __init__(self, verbose: bool) -> None:
        self.checks = 0
        self.failures: list[str] = []
        self.verbose = verbose

    def check(self, condition: bool, song: str, what: str, detail: str = "") -> None:
        self.checks += 1
        if condition:
            if self.verbose:
                print(f"    ok   {song:<9} {what}")
            return
        self.failures.append(f"{song}: {what} {detail}".rstrip())
        print(f"    FAIL {song:<9} {what} {detail}")

    def summary(self) -> int:
        print()
        print("=" * 78)
        if self.failures:
            print(f"{len(self.failures)} / {self.checks} 条断言失败：")
            for line in self.failures:
                print(f"  - {line}")
            return 1
        print(f"全部通过（{self.checks} 条断言）")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="out")
    parser.add_argument("--songs", nargs="*", default=list(DEFAULT_SONGS))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--self-check", action="store_true",
        help="只跑检查器的自证，不需要 out/ 里的产物",
    )
    args = parser.parse_args()

    from pipeline.check_shapes import self_check

    print("检查器自证（构造已知谱面，确认能抓到该抓的）")
    print(self_check(), end="")
    if args.self_check:
        return 0

    root = Path(args.root)
    if not root.is_absolute():
        root = ROOT / root

    report = Report(args.verbose)
    print(f"回归闸门  root={root}")
    print(f"{'曲目':<10}{'宽和弦':>7}{'拥挤':>6}{'跨度':>6}{'R1':>5}{'丢音':>6}")
    for song in args.songs:
        xml = root / song / f"{song}.musicxml"
        arrangement = root / song / "arrangement.txt"
        if not xml.is_file() or not arrangement.is_file():
            report.check(False, song, "产物存在", f"（缺 {xml.name} 或 {arrangement.name}）")
            continue

        # ---- 1. 可弹性 ---------------------------------------------------- #
        playable = analyse(song, root)
        if "error" in playable:
            report.check(False, song, "可弹性验收可执行", playable["error"])
            continue
        report.check(
            playable["wide_chords"] == 0, song, "宽和弦 = 0",
            f"实测 {playable['wide_chords']}",
        )
        report.check(
            playable["crowded_onsets"] == 0, song, "拥挤起音 = 0",
            f"实测 {playable['crowded_onsets']}",
        )
        report.check(
            playable["span_max"] <= 19, song, "最大跨度 ≤ 19",
            f"实测 {playable['span_max']}",
        )

        # ---- 2. R1：三连音成 3/6 组 --------------------------------------- #
        from music21 import converter
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = converter.parse(str(xml))
        shapes = check_score(score)
        report.check(
            not shapes["R1"], song, "R1 无落单三连音", f"实测 {len(shapes['R1'])} 处"
        )

        # ---- 3. 不丢音 ---------------------------------------------------- #
        arranged = read_arrangement(arrangement)
        on_score, info = read_score(xml)
        lost = sorted(set(arranged) - set(on_score))
        pitch_mismatch = sum(len(arranged[k] - on_score.get(k, set())) for k in arranged)
        report.check(not lost, song, "编排起音全部落到谱面", f"丢失 {len(lost)} 个")
        report.check(
            pitch_mismatch == 0, song, "起音的音高集合一致",
            f"有 {pitch_mismatch} 个音高对不上",
        )
        report.check(
            info["fifths"] is not None, song, "谱面有调号",
            "（<fifths> 缺失：检查 key.lab 的列解析）",
        )

        print(f"{song:<10}{playable['wide_chords']:>7}{playable['crowded_onsets']:>6}"
              f"{playable['span_max']:>6}{len(shapes['R1']):>5}{len(lost):>6}")

    return report.summary()


if __name__ == "__main__":
    raise SystemExit(main())
