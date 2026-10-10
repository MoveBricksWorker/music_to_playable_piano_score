"""编排：Transkun 转谱 + SheetSage2 节拍 → RH/LH 编排文件。

从 `run_best/run_all.py` 的 `build_arrangement` 抽取而来，**逻辑与判据一字未改**，
只把路径改成精简版的 `data/` 布局。

分工沿用项目既有约定：
  * 旋律（`melody_vocal.mid`）必须在最高声部
  * 中音 C（60）分手：≥60 右手，<60 左手
  * 删掉与旋律同刻同音高的钢琴音（人声串音）

**量化用 12 tick/拍的双格量化**，这是这一版的核心：4 tick/拍上 1/3 拍 = 4/3 tick
不是整数，三连音会在吸附那一步就被压掉（三个音变一个音）。
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from ensemble2score.piano_arrange import pitch_name, Voice, _voice_from_events
import pretty_midi

from . import engrave
from . import dual_lattice

#: 中音 C：旋律在上，钢琴转谱按此分左右手
SPLIT = 60
TICKS = 12


# --------------------------------------------------------------------------- #
# SheetSage2 标签读取
# --------------------------------------------------------------------------- #
# 读取函数下沉到 `pipeline.sage2`（无内部依赖），避免 arrange ↔ engrave 循环导入。
# 这里以原名转出，外部调用方不需要改动。
from pipeline.sage2 import (  # noqa: E402
    key_segments as _key_segments,
    parse_pitch,
    read_arrangement,
    read_lab,
    read_notes,
    voice_from as _voice_from,
)


# --------------------------------------------------------------------------- #
# Transkun 转谱读取
# --------------------------------------------------------------------------- #
def read_transkun_midi(path: Path) -> list[tuple[float, int, int]]:
    """读 Transkun 的复音 MIDI，返回 ``[(起音秒, 音高, 力度)]``。"""
    midi = pretty_midi.PrettyMIDI(str(path))
    notes: list[tuple[float, int, int]] = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            notes.append((float(note.start), int(note.pitch), int(note.velocity)))
    notes.sort()
    return notes


def dedupe_piano(piano: dict[int, set[int]], melody: dict[int, set[int]]) -> int:
    """删掉与旋律**同刻同音高**的钢琴音（那是人声串到钢琴轨上）。

    只用**相同音高**判定，不做邻音近似 —— 近似删音会误伤真正的齐奏。
    """
    dropped = 0
    for tick, pitches in list(piano.items()):
        if tick not in melody:
            continue
        overlap = pitches & melody[tick]
        if not overlap:
            continue
        dropped += len(overlap)
        piano[tick] = pitches - overlap
        if not piano[tick]:
            del piano[tick]
    return dropped


# --------------------------------------------------------------------------- #
# 编排
# --------------------------------------------------------------------------- #
def build_arrangement(
    song: str,
    transkun_midi: Path,
    sheetsage2_dir: Path,
    out_path: Path,
    *,
    ticks: int = TICKS,
    cost_normalization: str = "density",
    switch_margin: float = 0.01,
    snap_policy: str = "family",
) -> dict:
    """把 Transkun 的音符编排成两只手的谱面骨架。

    ``cost_normalization`` / ``switch_margin`` / ``snap_policy`` 透传给
    :func:`pipeline.dual_lattice.quantize`：默认按格点密度归一化代价、按选定的族
    吸附；传 ``cost_normalization="raw"`` / ``snap_policy="nearest"``
    可切到对照口径（见那边的 docstring）。
    """
    beats = read_lab(sheetsage2_dir / "beat.lab")
    downs = read_lab(sheetsage2_dir / "downbeat.lab")
    grid, numerator, bars, warnings = engrave.build_grid(beats, downs, ticks)
    tempo_bpm = 60.0 / statistics.median(
        beats[i + 1] - beats[i] for i in range(len(beats) - 1)
    )

    melody = read_notes(sheetsage2_dir / "melody_vocal.mid")
    tk_notes = read_transkun_midi(transkun_midi)

    # ---- 双格量化 ---------------------------------------------------------- #
    # 网格 tick 与量化器的"公共刻度"同一刻度：ticks_per_beat = 12 = 一拍。
    def quantize_notes(notes, label: str):
        positions: list[float] = []
        pitches: list[list[int]] = []
        for start, pitch in notes:
            beat_position = grid.beat_position(start)
            if beat_position < 0:
                continue
            positions.append(beat_position * ticks)
            pitches.append([pitch])
        if not positions:
            return {}, None
        result = dual_lattice.quantize(
            positions, pitches,
            cost_normalization=cost_normalization,
            switch_margin=switch_margin,
            snap_policy=snap_policy,
        )
        print(
            f"    {label}: {len(positions)} 起音 → {len(result.onsets)} "
            f"（合并 {result.merged}，格子={result.lattice}，"
            f"三连音窗口 {sum(1 for w in result.per_window if w == 'triple')}"
            f"/{len(result.per_window)}，最大位移 {result.max_shift_ticks} tick）",
            flush=True,
        )
        events: dict[int, set[int]] = {}
        for tick, pit in result.onsets:
            events.setdefault(tick, set()).update(pit)
        return events, result

    piano, piano_stats = quantize_notes(
        [(s, p) for s, p, _v in tk_notes], "钢琴转谱（各轨相加）"
    )
    melody_events, melody_stats = quantize_notes(melody, "旋律（melody_vocal）")

    right: dict[int, set[int]] = {}
    left: dict[int, set[int]] = {}
    melody_ticks: dict[int, int] = {}
    for tick, pitches in (melody_events or {}).items():
        for pitch in pitches:
            right.setdefault(tick, set()).add(pitch)
        melody_ticks[tick] = max(pitches)

    dropped_dupes = dedupe_piano(piano, right)
    if dropped_dupes:
        warnings.append(
            f"{dropped_dupes} 个 Transkun 音与旋律同刻同音高（钢琴轨的人声串音），已删除"
        )

    for tick, pitches in piano.items():
        for pitch in pitches:
            (left if pitch < SPLIT else right).setdefault(tick, set()).add(pitch)

    # 旋律必须是最高声部：盖在旋律之上的钢琴音删掉。
    # 判据是 **旋律自己的最高音**，不是该刻所有音的最高音 —— 后者会把伴奏音误当旋律。
    covered = 0
    for tick, top in melody_ticks.items():
        pitches = right.get(tick)
        if not pitches:
            continue
        keep = {p for p in pitches if p <= top}
        if keep != pitches:
            covered += len(pitches) - len(keep)
            right[tick] = keep
    if covered:
        warnings.append(f"{covered} 个音盖在旋律之上，已删除（旋律必须最高）")

    lines: list[str] = []
    for hand, events in (("RH", right), ("LH", left)):
        for tick in sorted(events):
            pitches = sorted(p for p in events[tick] if p is not None)
            if not pitches:
                continue
            bar, beat, sub = grid.from_tick(tick)
            names = " ".join(pitch_name(p) for p in pitches)
            lines.append(f"{hand} {grid.label(bar, beat, sub)} {names}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 记下"哪些**音高**来自旋律"，用最终编排坐标。
    #
    # 关键是只记旋律自己的音高，不是该 tick 上右手的整组音。早先写成整组，
    # 于是同刻伴奏音也被标成"旋律"，下游可弹性修正会把整组当不可动的保护音，
    # 实测让 victim 有 25 处跨度 20–27 的宽和弦无法折叠。
    melody_records = [
        {"tick": int(tick), "pitches": sorted(int(p) for p in melody_events[tick])}
        for tick in sorted(melody_events or {})
        if tick in melody_ticks
    ]
    (out_path.parent / f"{song}.melody.json").write_text(
        json.dumps(
            {
                "song": song,
                "note": "这些 tick 上的这些音高来自旋律，与 arrangement.txt 同一坐标系",
                "ticks_per_beat": ticks,
                "beats_per_bar": numerator,
                "melody": melody_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "song": song,
        "ticks_per_beat": ticks,
        "time_signature": f"{numerator}/4",
        "bars": bars,
        "tempo_bpm": round(tempo_bpm, 2),
        "rh_notes": sum(len(v) for v in right.values()),
        "lh_notes": sum(len(v) for v in left.values()),
        "dropped_dupes": dropped_dupes,
        "covered_by_melody": covered,
        "warnings": warnings,
        "quantizer": {
            "kind": "dual_lattice",
            # 口径本身要落盘：换了代价口径或吸附策略会改变谱面，事后必须能看出来。
            "cost_normalization": cost_normalization,
            "switch_margin": switch_margin,
            "snap_policy": snap_policy,
            "piano": None
            if piano_stats is None
            else {
                "lattice": piano_stats.lattice,
                "merged": piano_stats.merged,
                "duple_share": round(piano_stats.duple_share, 3),
                "max_shift_ticks": piano_stats.max_shift_ticks,
            },
            "melody": None
            if melody_stats is None
            else {
                "lattice": melody_stats.lattice,
                "merged": melody_stats.merged,
                "duple_share": round(melody_stats.duple_share, 3),
                "max_shift_ticks": melody_stats.max_shift_ticks,
            },
        },
    }
