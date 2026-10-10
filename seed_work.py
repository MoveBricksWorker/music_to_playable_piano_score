"""把 `data/` 里自带的输入铺进 `work/`，使管线可以从任意阶段开始。

精简版自带四首曲子的**全部中间产物**（Demucs 分轨、sum 转谱 MIDI、
SheetSage2 标签），所以不必重跑模型就能验证下游链路。本脚本负责这一步。

    python seed_work.py            # 四首全铺
    python seed_work.py victim     # 只铺一首
    python seed_work.py --check    # 只检查 data/ 是否齐全，不写文件

已有文件默认**不覆盖**（加 `--force` 才覆盖）。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

SONGS = ("victim", "yoake", "collapse", "blessing")

#: (data 下的相对路径, work 下的相对路径)
LAYOUT: tuple[tuple[str, str], ...] = (
    ("audio/{song}.mp3", "audio/{song}.mp3"),
    ("stems6/{song}/piano.wav", "stems6/{song}/piano.wav"),
    ("stems4/{song}/other.wav", "stems4/{song}/other.wav"),
    ("transkun/{song}.mid", "transkun/{song}.mid"),
    ("sheetsage2/{song}", "sheetsage2/{song}"),
)


def seed(song: str, work: Path, *, force: bool, check_only: bool) -> tuple[int, list[str]]:
    copied = 0
    missing: list[str] = []
    for source_rel, target_rel in LAYOUT:
        source = DATA / source_rel.format(song=song)
        target = work / target_rel.format(song=song)
        if not source.exists():
            missing.append(source_rel.format(song=song))
            continue
        if check_only:
            copied += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            for path in source.rglob("*"):
                if not path.is_file():
                    continue
                destination = target / path.relative_to(source)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() and not force:
                    continue
                shutil.copy2(path, destination)
                copied += 1
        else:
            if target.exists() and not force:
                copied += 1
                continue
            shutil.copy2(source, target)
            copied += 1
    return copied, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("songs", nargs="*", default=None)
    parser.add_argument("--work", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check", action="store_true", help="只检查 data/，不写 work/")
    args = parser.parse_args()

    songs = args.songs or list(SONGS)
    work = Path(args.work).resolve() if args.work else HERE / "work"

    if not DATA.is_dir():
        print(f"找不到 {DATA}")
        return 1
    print(f"输入 {DATA}")
    print(f"输出 {work}{'（仅检查）' if args.check else ''}\n")

    problems = 0
    for song in songs:
        copied, missing = seed(song, work, force=args.force, check_only=args.check)
        flag = "✓" if not missing else "✗"
        print(f"  {flag} {song:10s} {copied:3d} 项" + (f"   缺 {missing}" if missing else ""))
        problems += len(missing)
    print()
    if problems:
        print(f"有 {problems} 项输入缺失")
        return 1
    print("输入齐备。" + ("（未写文件）" if args.check else "可以跑：python run_pipeline.py --from arrange"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
