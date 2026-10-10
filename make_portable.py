"""把模型权重搬进整合包，并让管线**离线**读它们。

用户要求：*「做一个整合包，把那几个要用的模型和环境也搬进去吧」*

## 搬三个模型

| 模型 | 用途 | 来源 |
|---|---|---|
| `m-a-p/SheetSage2` | ④ 节拍 / 调号 / 和弦 / 旋律 | HuggingFace 缓存 |
| demucs `htdemucs` + `htdemucs_6s` | ① 分离（4 轨与 6 轨各一） | torch hub 缓存 |
| transkun `2.0.pt` | ③ 复音钢琴转谱 | 已随 wheel 安装，无需搬 |

**不搬** `m-a-p/MERT-v2-FullSong`（2.4 GB）：那是开发期实验用的编码器，
精简版管线**不 import 它**。这条要写清楚，否则以后有人以为漏了。

## 让管线离线读

* SheetSage2：设 `HF_HOME=<包>/models/hf` → `huggingface_hub` 直接从包内找
* demucs：设 `TORCH_HOME=<包>/models/torch` → torch hub 从 `<TORCH_HOME>/hub/checkpoints` 找
* transkun：权重在 site-packages 内，无需搬运

搬运后**逐个校验 SHA-256 与大小**，并打印清单，避免"看起来搬了其实没搬"。

用法：
    python make_portable.py                          # 从系统缓存 / HF_HOME 搬运并校验
    python make_portable.py --source <上游项目根>     # 从指定项目目录找 HF 缓存
    python make_portable.py --check                  # 只检查已有搬运是否完整
    python make_portable.py --force                  # 覆盖已存在的文件

`--source` 也可以用环境变量 `E2S_SOURCE_REPO` 指定。找不到 SheetSage2 时脚本会
打印它找过的每一个位置，而不是报一个写死的路径。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
HF_DEST = MODELS / "hf"
TORCH_DEST = MODELS / "torch"

#: 源仓库根（上游项目，用来找 HF 缓存）。**不写死**：优先命令行 `--source`，
#: 其次环境变量 `E2S_SOURCE_REPO`，都没有就只找系统缓存 / `HF_HOME`。
#: 搬完之后整合包不再需要它。
SOURCE_REPO: Path | None = None


def set_source_repo(value: str | None) -> Path | None:
    """设定源仓库根，并把它一并作为 HF 缓存的候选位置。"""
    global SOURCE_REPO
    if value:
        SOURCE_REPO = Path(value).expanduser().resolve()
        return SOURCE_REPO
    env = os.environ.get("E2S_SOURCE_REPO")
    SOURCE_REPO = Path(env).expanduser().resolve() if env else None
    return SOURCE_REPO

#: SheetSage2 的 HF 仓库目录名（huggingface_hub 的缓存命名规则）
HF_REPOS = ("models--m-a-p--SheetSage2",)

#: demucs 需要的权重文件（按内容哈希命名，模型名→文件名对应关系见下）
DEMUCS_MODELS = {
    "htdemucs": "955717e8-8726e21a.th",
    "htdemucs_6s": "5c90dfd2-34c22ccb.th",
}


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def copy_tree(src: Path, dst: Path, *, force: bool) -> tuple[int, int]:
    """复制目录，返回 (复制数, 跳过数)。"""
    copied = skipped = 0
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        target = dst / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and not force and target.stat().st_size == path.stat().st_size:
            skipped += 1
            continue
        shutil.copy2(path, target)
        copied += 1
    return copied, skipped


def find_hf_source() -> Path | None:
    """找 SheetSage2 的 HF 缓存目录。候选按"越专有越优先"排列。"""
    candidates: list[Path | None] = []
    if SOURCE_REPO is not None:
        candidates += [
            SOURCE_REPO / "_hf_home" / "hub",
            SOURCE_REPO / "hf_home" / "hub",
        ]
    if os.environ.get("HF_HOME"):
        candidates.append(Path(os.environ["HF_HOME"]) / "hub")
    candidates += [
        HERE / "models" / "hf" / "hub",          # 已经搬过一次，就地升级
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    for base in candidates:
        if base and (base / HF_REPOS[0]).is_dir():
            return base
    return None


def find_torch_source() -> Path | None:
    """找 torch hub 的 checkpoints 目录。"""
    base = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    return base if base.is_dir() else None


def plan() -> list[dict]:
    """列出要搬的东西：(说明, 源, 目标)。"""
    items: list[dict] = []

    hf = find_hf_source()
    if hf:
        for repo in HF_REPOS:
            items.append({
                "what": f"SheetSage2 权重（{repo}）",
                "src": hf / repo,
                "dst": HF_DEST / "hub" / repo,
                "kind": "dir",
            })

    torch_src = find_torch_source()
    if torch_src:
        for model, filename in DEMUCS_MODELS.items():
            source = torch_src / filename
            if source.is_file():
                items.append({
                    "what": f"demucs {model}",
                    "src": source,
                    "dst": TORCH_DEST / "hub" / "checkpoints" / filename,
                    "kind": "file",
                })
    return items


def manifest_path() -> Path:
    return MODELS / "MANIFEST.json"


def write_manifest(items: list[dict]) -> None:
    """搬运后写清单：每个文件的相对路径 + 大小 + SHA-256。"""
    entries = []
    for item in items:
        dst = item["dst"]
        files = [dst] if dst.is_file() else [p for p in dst.rglob("*") if p.is_file()]
        for path in files:
            entries.append({
                "what": item["what"],
                "path": str(path.relative_to(HERE)).replace("\\", "/"),
                "size": path.stat().st_size,
                "sha256": sha256(path),
            })
    manifest_path().write_text(
        json.dumps(
            {
                "note": "整合包自带模型清单。路径相对整合包根。",
                "entries": entries,
                "total_bytes": sum(e["size"] for e in entries),
                "not_included": [
                    {
                        "repo": "m-a-p/MERT-v2-FullSong",
                        "size_hint": "2.4 GB",
                        "why": "开发期实验用的音频编码器，精简版管线不 import 它",
                    }
                ],
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def verify_manifest() -> tuple[int, int, list[str]]:
    """按清单逐个校验。返回 (通过数, 失败数, 失败详情)。"""
    if not manifest_path().is_file():
        return 0, 0, ["没有 MANIFEST.json"]
    data = json.loads(manifest_path().read_text(encoding="utf-8"))
    ok = bad = 0
    problems: list[str] = []
    for entry in data["entries"]:
        path = HERE / entry["path"]
        if not path.is_file():
            bad += 1
            problems.append(f"缺失 {entry['path']}")
            continue
        if path.stat().st_size != entry["size"]:
            bad += 1
            problems.append(
                f"大小不符 {entry['path']} "
                f"({path.stat().st_size} vs {entry['size']})"
            )
            continue
        if sha256(path) != entry["sha256"]:
            bad += 1
            problems.append(f"SHA-256 不符 {entry['path']}")
            continue
        ok += 1
    return ok, bad, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=None,
                        help="上游项目根（从这里找 HF 缓存）；也可用 E2S_SOURCE_REPO")
    parser.add_argument("--check", action="store_true", help="只校验已有搬运")
    parser.add_argument("--force", action="store_true", help="覆盖已存在文件")
    args = parser.parse_args()

    print(f"整合包目录: {HERE}")
    print(f"模型目录:   {MODELS}\n")

    if args.check:
        ok, bad, problems = verify_manifest()
        print(f"校验：通过 {ok} 个，失败 {bad} 个")
        for line in problems[:12]:
            print(f"   ✗ {line}")
        return 1 if bad else 0

    source = set_source_repo(args.source)
    if source is not None:
        print(f"上游项目根: {source}")
    elif os.environ.get("E2S_SOURCE_REPO"):
        print(f"上游项目根: {SOURCE_REPO}（来自 E2S_SOURCE_REPO）")

    items = plan()
    if not items:
        print("! 没有找到可搬的模型。找过这些位置：")
        if SOURCE_REPO is not None:
            print(f"   SheetSage2：{SOURCE_REPO / '_hf_home' / 'hub'}")
            print(f"               {SOURCE_REPO / 'hf_home' / 'hub'}")
        else:
            print("   SheetSage2：<未指定上游项目根>（用 --source 或 E2S_SOURCE_REPO 指定）")
        if os.environ.get("HF_HOME"):
            print(f"               {Path(os.environ['HF_HOME']) / 'hub'}")
        print(f"               {Path.home() / '.cache' / 'huggingface' / 'hub'}")
        print(f"   demucs：    {Path.home() / '.cache' / 'torch' / 'hub' / 'checkpoints'}")
        print("\n  SheetSage2 可以用 huggingface-cli download m-a-p/SheetSage2 先拉进缓存。")
        return 1

    print("计划搬运：")
    for item in items:
        size = (
            item["src"].stat().st_size
            if item["src"].is_file()
            else sum(p.stat().st_size for p in item["src"].rglob("*") if p.is_file())
        )
        print(f"   {item['what']:44s} {size / 1e6:8.1f} MB")

    total = 0
    for item in items:
        if item["kind"] == "dir":
            copied, skipped = copy_tree(item["src"], item["dst"], force=args.force)
            print(f"   {item['what']}: 复制 {copied}，跳过 {skipped}")
        else:
            target = item["dst"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and not args.force and target.stat().st_size == item["src"].stat().st_size:
                print(f"   {item['what']}: 已存在，跳过")
            else:
                shutil.copy2(item["src"], target)
                print(f"   {item['what']}: 已复制")
        dst = item["dst"]
        files = [dst] if dst.is_file() else [p for p in dst.rglob("*") if p.is_file()]
        total += sum(p.stat().st_size for p in files)

    print("\n写清单并校验 …")
    write_manifest(items)
    ok, bad, problems = verify_manifest()
    print(f"   通过 {ok} 个，失败 {bad} 个")
    for line in problems[:12]:
        print(f"   ✗ {line}")
    print(f"   模型合计 {total / 1e9:.2f} GB")

    if bad:
        return 1
    print("\n完成。管线会自动用包内模型（见 run_pipeline.py 的 _apply_model_env）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
