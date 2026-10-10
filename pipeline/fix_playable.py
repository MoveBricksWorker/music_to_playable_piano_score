"""可弹性约束：让每一处起音都满足「单手跨度 ≤19 半音、同时音 ≤5」。

在**编排文件**上做（那是唯一能表达"哪个音在哪只手、哪个 tick"的中间产物），
然后交给 `run_best/engrave.py` 雕版，最后用 `run_best/verify_playable.py`
在**成品谱面**上验收（谱面才是用户看到的东西，也是唯一可信的口径）。

三条规则，都能单独关掉，便于对照实验：

  R1 去八度重复    同一音级只留最高的那个（项目文档记过这招能把跨度减半）
  R2 跨度折叠      跨度仍 >19 时，把最高音降八度；降完与该音级已有的低八度音撞成
                   同度就丢弃那个音（**音级不变 = 和声不变**）
  R3 音数超限      仍 >5 个音时，按「保最低音 + 保最高音 + 保旋律音」保留，其余从
                   内侧往外删

**刻意不做**的事（做了会破坏别的已验收指标）：

  * 不删和弦外音（那是转谱素材，删了就不"还原"了；折叠保留全部音级）
  * 不动旋律所在的右手最高音（旋律必须最高是已验收的规则）
  * 不跨手搬音（那会改变分工，属于编配决策，不是约束修正）

只读编排文件；结果写到一个新目录，不覆盖任何现有产物。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HAND_SPAN = 19
MAX_SIMULTANEOUS = 5
TICKS_PER_BEAT = 12
BEATS_PER_BAR = 2
PITCH_CLASS = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
    "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}


def parse_token(token: str) -> int | None:
    """音名 -> MIDI。**复用 `ensemble2score.piano_arrange._pitch_from_token`**，
    不自己实现：读和写必须用同一套换算，否则写出去的符号回读会变成另一个音。"""
    sys.path.insert(0, str(ROOT / "src"))
    from ensemble2score.piano_arrange import _pitch_from_token  # noqa: E402

    return _pitch_from_token(token)


def pitch_name(pitch: int) -> str:
    """MIDI -> 音名。**复用 `ensemble2score.piano_arrange.pitch_name`**。

    早先这里自己写成了 `pitch // 12 - 1`，对 MIDI 60 得到 `C-1`（应为 `C4`），
    而 `parse_token` 用的是 `(octave + 1) * 12` —— 读写不对称，
    于是修正后的编排里出现 `C-12`、`E-12` 这种不存在的音，雕版后跨度反而变大
    （实测 victim 最大跨度从 40 变成 43、blessing 出现 42）。
    """
    sys.path.insert(0, str(ROOT / "src"))
    from ensemble2score.piano_arrange import pitch_name as _pitch_name  # noqa: E402

    return _pitch_name(pitch)


def read_arrangement(
    path: Path, beats_per_bar: int = BEATS_PER_BAR
) -> tuple[dict[str, dict[int, list[int]]], list[str]]:
    """编排文件 -> {'RH': {tick: [pitches]}, 'LH': {...}}。

    ``beats_per_bar`` 必须与**写这个文件的程序**用的一致，否则 tick 会整体错位：
    `run_best/out/*/arrangement.txt` 里的标签是 4 拍/小节（`bars = 每 48 tick 一小节`），
    而旋律 sidecar 是另一个单位。实测按 2 拍/小节读 4 拍/小节的文件，旋律命中率
    只有 9.4%（yoake），修成同一单位后为 100%。
    """
    events: dict[str, dict[int, list[int]]] = {"RH": {}, "LH": {}}
    order: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 3 or parts[0] not in events:
            continue
        bar, rest = parts[1].split(":")
        beat, sub = rest.split(".")
        tick = (
            (int(bar) - 1) * beats_per_bar * TICKS_PER_BEAT
            + (int(beat) - 1) * TICKS_PER_BEAT
            + int(sub)
        )
        pitches = [p for p in (parse_token(t) for t in parts[2:]) if p is not None]
        if pitches:
            events[parts[0]][tick] = sorted(set(pitches))
            order.append(parts[0])
    return events, order


def drop_octave_duplicates(
    pitches: list[int], *, reserved: set[int] | None = None
) -> tuple[list[int], int]:
    """R1：同一音级只留最高的那个。保留音（旋律）永不被删。

    早先按"音级"去重，若某个音级上旋律音是低八度、而另有一个更高的同音级音，
    旋律音就会被删掉。所以保留音要先单独保住，再对剩下的做去重。
    """
    keep = {p for p in (reserved or set()) if p in pitches}
    highest: dict[int, int] = {}
    for pitch in pitches:
        if pitch in keep:
            continue
        key = pitch % 12
        if key not in highest or pitch > highest[key]:
            highest[key] = pitch
    # 保留音所在的音级：保留音与更高的同音级音可以共存（不允许删旋律）
    for pitch in keep:
        key = pitch % 12
        if key in highest and highest[key] < pitch:
            highest[key] = pitch
    out = sorted(set(highest.values()) | keep)
    return out, len(pitches) - len(out)


def fold_to_span(
    pitches: list[int], melody_top: int | None, *, reserved: set[int] | None = None
) -> tuple[list[int], int, int]:
    """R2：把音降八度，直到跨度 ≤19；折不动的就删中间音。音级不变 = 和声不变。

    折叠：每次挑**当前最高、且不是旋律音**的音降一个八度，只在这次下移**真的缩小
    了跨度**时接受。早先版本的终止条件写成 `if lowered <= work[0] - 12: break`，
    那是错的（对某个音级 `pitch - 12` 永远不小于 `work[0] - 12`），保护从不触发，
    循环一路折到守卫上限、产出**负 MIDI 值**（实测 `[60,64,69,81]` → `[-132,-128,81]`，
    写成音名是 `C-12 E-12`），雕版后跨度反而更大。

    删音：两个音时折叠只是**互换位置、跨度不变**（`[60,85]` 折成 `[60,73]`，跨度仍 25→13？
    不，13 是达标的；真正无解的是折叠后仍 >19）。所以折叠之后若仍超标，就删掉
    **最高的非旋律音**——保最低音（低音骨架）与旋律。

    返回 (音高, 折叠次数, 删除数)。
    """
    work = sorted(set(pitches))
    keep = set(reserved or set())
    moves = 0
    dropped = 0
    guard = 0
    while len(work) > 1 and work[-1] - work[0] > HAND_SPAN and guard < 64:
        guard += 1
        before = work[-1] - work[0]
        # 先试**最高音**下移一个八度
        folded = False
        for index, raise_octave in ((len(work) - 1, False), (0, True)):
            candidate = work[index]
            if candidate in keep:
                continue
            lowered = candidate - 12 if not raise_octave else candidate + 12
            if not (0 <= lowered <= 127):
                continue
            trial = sorted(set(work[:index] + work[index + 1:] + [lowered]))
            if trial and trial[-1] - trial[0] < before:
                work = trial
                moves += 1
                folded = True
                break
        if folded:
            continue
        # 折叠无法改善：删掉最高的**非保留**音（保留音一个都不许删）
        deletable = [i for i, p in enumerate(work) if p not in keep]
        if not deletable:
            break
        target = deletable[-1]
        if work[target] == work[0] and len(work) > 1:
            others = [i for i in deletable if i != 0]
            if not others:
                break
            target = others[-1]
        work.pop(target)
        dropped += 1
    return work, moves, dropped


def cap_polyphony(
    pitches: list[int], melody_top: int | None, *, reserved: set[int] | None = None
) -> tuple[list[int], int]:
    """R3：音数超限时从内侧删，保最低音、最高音、以及**全部旋律音**。"""
    work = sorted(set(pitches))
    if len(work) <= MAX_SIMULTANEOUS:
        return work, 0
    must_keep = {work[0], work[-1]}
    if melody_top is not None and melody_top in work:
        must_keep.add(melody_top)
    must_keep |= {p for p in (reserved or set()) if p in work}
    if len(must_keep) >= MAX_SIMULTANEOUS:
        # 保留音已超出上限：优先保旋律，再保两端
        ordered = [p for p in work if p in (reserved or set())][:MAX_SIMULTANEOUS]
        return sorted(set(ordered) or set(work[:MAX_SIMULTANEOUS])), len(work) - len(ordered)
    middle = [p for p in work if p not in must_keep]
    # 从最靠近中间的往外删
    middle.sort(key=lambda p: abs(p - (work[0] + work[-1]) / 2))
    keep = sorted(must_keep | set(middle[: max(0, MAX_SIMULTANEOUS - len(must_keep))]))
    while len(keep) > MAX_SIMULTANEOUS:
        removable = [p for p in keep if p not in must_keep]
        if not removable:
            break
        keep.remove(removable[len(removable) // 2])
    return keep, len(work) - len(keep)


def melody_sidecar(song: str, source_root: str = "run_best/out") -> dict:
    """读旋律 sidecar。

    **必须和编排文件同一个目录**：清洗会把起音挪位并同步挪动 sidecar，
    若仍读原始目录那份，保护的是旧 tick 上的旋律 —— 实测那样 collapse 的
    旋律保留率从 0.954 掉到 0.900、blessing 从 0.966 掉到 0.832。
    """
    sidecar = ROOT / source_root / "_arrangements" / f"{song}.melody.json"
    if not sidecar.is_file():
        print(f"  ! 没有旋律 sidecar（{sidecar.relative_to(ROOT)}）：本次不保护旋律。"
              f"先跑 `python run_best/run_all.py` 生成它。")
        return {}
    return json.loads(sidecar.read_text(encoding="utf-8"))


def melody_tops(song: str, source_root: str = "run_best/out") -> dict[int, int]:
    """每个 tick 上旋律的音高（兼容旧 sidecar 的单值格式）。"""
    payload = melody_sidecar(song, source_root)
    if not payload:
        return {}
    if "melody" in payload:
        out: dict[int, int] = {}
        for item in payload["melody"]:
            tick = int(item["tick"])
            if "pitches" in item:
                for pitch in item["pitches"]:
                    out[tick] = max(out.get(tick, int(pitch)), int(pitch))
            else:
                out[tick] = max(out.get(tick, int(item["pitch"])), int(item["pitch"]))
        return out
    return {int(tick): int(pitch) for tick, pitch in payload.get("melody_ticks", {}).items()}


def melody_protected(song: str, source_root: str = "run_best/out") -> dict[int, set[int]]:
    """每个 tick 上**全部需要保护的旋律音高**（不是只保护最高音）。

    早先只保护"旋律最高音"，而该值常常不在编排里（实测 blessing
    `旋律音=83` 但编排里只有 `[60]`），于是保护落空、真正的音在删内声部时被删，
    旋律保留率从 0.966 掉到 0.832。改为保护整组。
    """
    payload = melody_sidecar(song, source_root)
    if not payload:
        return {}
    out: dict[int, set[int]] = {}
    if "melody" in payload:
        for item in payload["melody"]:
            tick = int(item["tick"])
            pitches = (
                [int(p) for p in item["pitches"]]
                if "pitches" in item
                else [int(item["pitch"])]
            )
            out.setdefault(tick, set()).update(pitches)
        return out
    for tick, pitch in payload.get("melody_ticks", {}).items():
        out.setdefault(int(tick), set()).add(int(pitch))
    return out


def process(song: str, *, rules: tuple[bool, bool, bool], source_root: str = "run_best/out") -> dict:
    """``source_root`` 决定读哪一份编排。默认读原始那份；清洗后的产物要先经过
    本函数做可弹性修正，所以把 ``source_root`` 指到清洗目录。

    顺序很重要：**清洗会制造新的跨度违规**（R-A 挪起音、R-B 挪音高），
    实测清洗后宽和弦从 0 变成 18 处。所以必须"先清洗、再修可弹性"，
    否则清洗的产物不能交付。
    """
    source = ROOT / source_root / song / "arrangement.txt"
    # 拍/小节数必须与写这个文件的程序一致（因曲而异：victim 2、yoake 4），
    # 从 sidecar 读，不要写死。
    #
    # **必须带 `source_root`**：早先这里写成 `melody_sidecar(song)`，于是永远读
    # 默认目录 `run_best/out` 的 sidecar，拿到**别的版本**的 beats_per_bar，
    # 整份编排的 tick 坐标会被按错误的小节长度解释 —— 表现为"凭空多出跨度 25 的
    # 宽和弦、且 `fix_playable` 明明有规则却一处都不改"。
    payload = melody_sidecar(song, source_root)
    beats_per_bar = int(payload.get("beats_per_bar", BEATS_PER_BAR)) if payload else BEATS_PER_BAR
    events, _order = read_arrangement(source, beats_per_bar=beats_per_bar)
    tops = melody_tops(song, source_root)
    protected = melody_protected(song, source_root)
    stats = Counter()
    changes: list[dict] = []
    for hand in ("RH", "LH"):
        for tick, pitches in sorted(events[hand].items()):
            original = list(pitches)
            work = sorted(set(pitches))
            reserved = protected.get(tick, set())
            if rules[0]:
                work, removed = drop_octave_duplicates(work, reserved=reserved)
                stats["R1 去掉八度重复音"] += removed
            if rules[1]:
                work, moves, cuts = fold_to_span(work, tops.get(tick), reserved=reserved)
                stats["R2 降八度次数"] += moves
                stats["R2 因跨度删音"] += cuts
            if rules[2]:
                work, dropped = cap_polyphony(work, tops.get(tick), reserved=reserved)
                stats["R3 删内声部音"] += dropped
            work = sorted(set(work))
            if work != original:
                changes.append({
                    "song": song, "hand": hand, "tick": tick,
                    "before": original, "after": work,
                    "span_before": original[-1] - original[0],
                    "span_after": work[-1] - work[0],
                })
                stats["被修改的起音数"] += 1
            events[hand][tick] = work
    return {"song": song, "events": events, "stats": dict(stats),
            "changes": changes, "beats_per_bar": beats_per_bar}


def write_arrangement(
    events: dict[str, dict[int, list[int]]], path: Path, beats_per_bar: int = BEATS_PER_BAR
) -> None:
    lines: list[str] = []
    for hand in ("RH", "LH"):
        for tick in sorted(events[hand]):
            pitches = events[hand][tick]
            if not pitches:
                continue
            bar, within = divmod(tick, beats_per_bar * TICKS_PER_BEAT)
            beat, sub = divmod(within, TICKS_PER_BEAT)
            names = " ".join(pitch_name(p) for p in pitches)
            lines.append(f"{hand} {bar + 1}:{beat + 1}.{sub} {names}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="run_best/out_playable/_arrangements")
    parser.add_argument("--no-r1", action="store_true")
    parser.add_argument("--no-r2", action="store_true")
    parser.add_argument("--no-r3", action="store_true")
    parser.add_argument("--songs", nargs="*", default=["victim", "yoake", "collapse", "blessing"])
    args = parser.parse_args()

    rules = (not args.no_r1, not args.no_r2, not args.no_r3)
    out_root = ROOT / args.out
    report: list[dict] = []
    for song in args.songs:
        result = process(song, rules=rules)
        target = out_root / f"{song}.txt"
        write_arrangement(result["events"], target, beats_per_bar=result["beats_per_bar"])
        report.append({
            "song": song,
            "stats": result["stats"],
            "changes": len(result["changes"]),
            "example_changes": result["changes"][:6],
        })
        print(f"{song}: 修改 {result['stats'].get('被修改的起音数', 0)} 个起音  "
              f"{result['stats']}")
        print(f"   -> {target.relative_to(ROOT)}")
    (out_root.parent / "fix_report.json").write_text(
        json.dumps({"rules": {"R1": rules[0], "R2": rules[1], "R3": rules[2]},
                    "songs": report}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
