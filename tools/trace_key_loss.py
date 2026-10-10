"""调号追踪：确认 `key.lab` 解析出来的调名对不对，以及它是在哪一步丢的。

背景：`key.lab` 每行是**三段** `起始秒  结束秒  调名`：

    0.01	149.74	F:minor

早先 `pipeline/sage2.key_segments` 取的是第 2 段（结束秒字符串 `"149.74"`），
于是 `material.key` 成了数字串 → 调号解析失败被静默吞掉 → **成品里没有调号**，
拼写表也退回全升号表，每个音都得带临时记号。

这个探针做两件事：
  1. 打印每个 `key.lab` 解析出的 (起始秒, 调名)，以及对应的五度数、拼写表；
  2. 构造一份最小谱面，验证"调号挂在小节外"和"挂在小节里"都能被导出
     （两者 music21 都支持，所以调号丢失一定是**上游传值**的问题，不是导出问题）。

用法：
    python tools/trace_key_loss.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from music21 import clef, meter, note, stream  # noqa: E402

from pipeline.engrave import _key_fifths, _key_signature_for, spelling_table  # noqa: E402
from pipeline.sage2 import key_segments  # noqa: E402

SONGS = ("victim", "yoake", "collapse", "blessing")
TMP = ROOT / "work" / "_trace_key_tmp.musicxml"


def dump(score, label: str) -> bool:
    """把小谱面写出去，返回 XML 里有没有调号。"""
    score.write("musicxml", fp=str(TMP))
    text = TMP.read_text(encoding="utf-8")
    return "<fifths>" in text


def minimal(key_signature, *, inside_measure: bool):
    part = stream.Part(id="X")
    part.partName = "Piano (right)"
    measure = stream.Measure(number=1)
    voice = stream.Voice()
    voice.insert(0, note.Note("C5", quarterLength=1))
    voice.insert(1, note.Note("F5", quarterLength=1))
    measure.insert(0, voice)
    if inside_measure:
        measure.insert(0, clef.TrebleClef())
        measure.insert(0, meter.TimeSignature("2/4"))
        measure.insert(0, key_signature)
    else:
        part.append(clef.TrebleClef())
        part.append(meter.TimeSignature("2/4"))
        part.append(key_signature)
    part.append(measure)
    score = stream.Score()
    score.insert(0, part)
    return score


def main() -> int:
    print("=== 一、key.lab 解析结果 ===")
    for song in SONGS:
        path = ROOT / "work" / "sheetsage2" / song / "key.lab"
        if not path.is_file():
            print(f"  {song:<10} 没有 {path.relative_to(ROOT)}")
            continue
        segments = key_segments(path)
        names = [name for _start, name in segments]
        print(f"  {song:<10} {segments}")
        for name in names:
            fifths = _key_fifths(name)
            table = spelling_table(fifths)
            print(f"      {name:<10} fifths={fifths:+d}  拼写表={table}")
            print(f"      调号对象={_key_signature_for(name)!r}")

    print()
    print("=== 二、调号能不能被导出（两种放法）===")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in ("F:minor", "Eb:major", "E:minor"):
            signature = _key_signature_for(name)
            outside = dump(minimal(signature, inside_measure=False), "part")
            inside = dump(minimal(signature, inside_measure=True), "measure")
            print(f"  {name:<10} 挂 part 上={outside}  放进第 1 小节={inside}")

    TMP.unlink(missing_ok=True)
    print()
    print("两种放法 music21 都能导出 → 调号丢失只可能来自上游传值。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
