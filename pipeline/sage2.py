"""SheetSage2 标签与编排文本的读取工具（**无内部依赖**）。

单独成模块的原因：`arrange` 需要 `engrave.build_grid`，而 `engrave` 需要这些读取函数，
若把读取函数放在 `arrange` 里就会形成循环导入
（`arrange` → `engrave` → `arrange`）。这里只依赖 pretty_midi 与 music21，
放在最底层，两边都能安全导入。
"""

from __future__ import annotations

from pathlib import Path

import pretty_midi


def read_lab(path: Path) -> list[float]:
    """读 SheetSage2 的 `.lab`（每行以时间点开始，单位秒）。"""
    return [
        float(line.split()[0])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_notes(path: Path) -> list[tuple[float, int]]:
    """读 MIDI，返回 ``[(起音秒, 音高)]``（按起音排序）。"""
    midi = pretty_midi.PrettyMIDI(str(path))
    out: list[tuple[float, int]] = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            out.append((float(note.start), int(note.pitch)))
    out.sort()
    return out


def key_segments(path: Path) -> list[tuple[float, str]]:
    """读 `key.lab`，返回 ``[(起始秒, 调名)]``。

    `key.lab` 每行是 **三段**：``起始秒  结束秒  调名``
    （如 ``0.01\\t149.74\\tF:minor``）。调名是**第 3 段**，不是第 2 段 ——
    早先写成 `parts[1]`，拿到的是结束秒（``"149.74"``），于是
    ``material.key`` 成了数字串、调号解析失败、拼写表退回全升号，
    成品里既没有 `<fifths>` 又让每个音都带临时记号。
    """
    segments: list[tuple[float, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        name = parts[2] if len(parts) >= 3 else parts[1]
        segments.append((float(parts[0]), name))
    return segments


def voice_from(notes: list[tuple[float, int]], grid, name: str, source: str):
    """起音与音高 → 网格上的 :class:`Voice`，**保持和弦堆叠不被拆开**。"""
    from ensemble2score.piano_arrange import _voice_from_events

    return _voice_from_events(name, source, notes, grid)


def parse_pitch(text: str) -> int:
    """音名或 MIDI 号 → MIDI 号（编排文件两种都允许）。"""
    try:
        return int(text)
    except ValueError:
        pass
    from music21 import pitch as m21pitch

    return int(round(m21pitch.Pitch(text).midi))


def read_arrangement(path: Path, grid):
    """``RH/LH`` 行 → :class:`Arrangement`，接受音名或 MIDI 号。

    位置走网格自己的 ``parse_label``，这样编排与素材**不可能落在两套 tick 网格上**
    —— 早先这个文件自己解析 ``bar:beat.tick``，而素材用了另一种分隔符，就出过这个问题。
    """
    from ensemble2score.piano_arrange import Arrangement

    arrangement = Arrangement(track=path.stem)
    problems: list[str] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if parts[0] not in ("RH", "LH"):
            problems.append(f"line {number}: not RH/LH: {line[:40]!r}")
            continue
        tick = grid.parse_label(parts[1])
        if tick is None:
            problems.append(f"line {number}: bad position {parts[1]!r}")
            continue
        try:
            pitches = [parse_pitch(p) for p in parts[2:]]
        except ValueError:
            problems.append(f"line {number}: bad pitch in {line[:40]!r}")
            continue
        if not pitches:
            problems.append(f"line {number}: no pitches")
            continue
        target = arrangement.right if parts[0] == "RH" else arrangement.left
        target.setdefault(tick, []).extend(pitches)
    return arrangement, problems
