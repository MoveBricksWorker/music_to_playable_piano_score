"""把当前 Python 环境**复制**进整合包（不走 pip 重装）。

用户问：*「环境从原环境中直接复制粘贴不行吗？」* —— 可以，本脚本就是干这个的。
比 pip 重装好在：不用下载数 GB、不需要网络、结果与开发机上的完全一致。

## 为什么复制是可用的

| 项 | 复制后是否有效 | 原因 |
|---|---|---|
| `python.exe` | ✅ | Windows 上它是基础安装的副本；`sys.prefix` 由**可执行文件位置**推导，跟目录走 |
| `site-packages` | ✅ | 包按 `sys.prefix` 定位自己；`.pth` 里放的是相对路径 |
| torch 的 DLL | ✅ | `torch\\lib` 由包自己按运行时路径加载 |
| **`Scripts\\*.exe`** | ❌ | **写死了安装时的绝对路径**（`pip.exe` / `demucs.exe` 等） |
| `pyvenv.cfg` | ⚠️ | 指向 `home = <原基础解释器>`；把基础解释器一起复制即自洽 |

所以**唯一前提**是：调用时不要用那些 `Scripts\\*.exe`，而用
`python.exe -m 模块名`。整合包里的代码全部满足这一点
（`run_pipeline.py` 用 `sys.executable` + `-m`）。

用法：
    python bundle_env.py             # 复制当前解释器到 <包>/python
    python bundle_env.py --check     # 只检查包内解释器是否可用
    python bundle_env.py --trim      # 复制后删掉管线用不到的大包
    python bundle_env.py --from <dir>  # 指定源解释器目录
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEST = HERE / "python"

#: 管线需要能 import 这些才算环境可用（import 名）
REQUIRED = [
    "torch", "torchaudio", "numpy", "scipy", "music21", "pretty_midi",
    "soundfile", "librosa", "soxr", "pydub", "demucs", "transkun",
]

#: 复制后可以删掉的大包（管线不 import）。
#:
#: **保守原则**：只删"确认不是任何依赖的间接依赖"的包。
#: 例如 `matplotlib` / `seaborn` 是 transkun 声明的依赖、也可能被别的库间接 import，
#: 所以**不删** —— 省几十 MB 不值得冒"跑起来少一个模块"的风险。
TRIMMABLE = [
    "tensorflow",            # 1.3 GB，与管线无关
    "~orch",                 # 1.1 GB，是 torch 安装失败的残留目录（名字里带 ~）
    "tensorflow_estimator",
    "keras",
    "bitsandbytes",
    "cv2",
    "pyarrow",
    "pymupdf",
    "plotly",
    "notebook",
    "tensorboard",
    "jupyter",
    "jupyter_client",
    "jupyter_core",
    "jupyter_server",
    "nbformat",
    "ipykernel",
    "ipywidgets",
    "widgetsnbextension",
]


def bundled_python() -> Path:
    return DEST / "python.exe" if sys.platform == "win32" else DEST / "bin" / "python3"


def probe(python: Path) -> tuple[list[str], list[str], str]:
    """返回 (就绪, 缺失, 解释器路径)。"""
    if not python.is_file():
        return [], REQUIRED, ""
    script = (
        "import importlib.util as u, sys;"
        f"names={REQUIRED!r};"
        "print('PY ' + sys.executable);"
        "print('PREFIX ' + sys.prefix);"
        "print('OK ' + ' '.join(n for n in names if u.find_spec(n)));"
        "print('MISSING ' + ' '.join(n for n in names if not u.find_spec(n)))"
    )
    result = subprocess.run([str(python), "-c", script], capture_output=True, text=True)
    ready: list[str] = []
    missing: list[str] = list(REQUIRED)
    reported = ""
    for line in (result.stdout or "").splitlines():
        if line.startswith("PY "):
            reported = line[3:]
        elif line.startswith("OK "):
            ready = line[3:].split()
        elif line.startswith("MISSING "):
            missing = line[8:].split()
    return ready, missing, reported


def source_prefix() -> Path:
    """当前解释器的安装目录（Windows 下是 python.exe 所在目录）。"""
    return Path(sys.base_prefix)


def copy_env(source: Path, force: bool) -> int:
    if DEST.exists() and not force:
        print(f"! {DEST} 已存在。加 --force 覆盖，或先删掉它。")
        return 1
    if DEST.exists():
        print(f"删除已有 {DEST} …")
        shutil.rmtree(DEST)

    print(f"复制 {source} → {DEST}")
    if sys.platform == "win32":
        # robocopy 比逐文件复制快得多，也支持断点续传
        result = subprocess.run(
            ["robocopy", str(source), str(DEST), "/E", "/NFL", "/NDL", "/NJH", "/NP",
             "/R:1", "/W:1", "/MT:16"],
            capture_output=True, text=True,
        )
        # robocopy 的退出码 0-7 都算成功，>=8 才是错误
        if result.returncode >= 8:
            print(f"! robocopy 失败（exit={result.returncode}）")
            print((result.stdout or "")[-1500:])
            return 1
    else:
        shutil.copytree(source, DEST, symlinks=True)

    total = sum(p.stat().st_size for p in DEST.rglob("*") if p.is_file())
    print(f"   复制完成：{total / 1e9:.2f} GB")
    return 0


def trim() -> tuple[int, float]:
    """删掉管线用不到的大包。返回 (删除数, 释放的 GB)。"""
    site = DEST / "lib" / "site-packages"
    if not site.is_dir():
        print(f"! 找不到 {site}")
        return 0, 0.0
    removed = 0
    freed = 0
    for name in TRIMMABLE:
        path = site / name
        if not path.exists():
            continue
        size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) \
            if path.is_dir() else path.stat().st_size
        print(f"   删除 {name:24s} {size / 1e6:8.1f} MB")
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
        removed += 1
        freed += size
    return removed, freed / 1e9


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="只检查包内解释器")
    parser.add_argument("--force", action="store_true", help="覆盖已有 python/")
    parser.add_argument("--trim", action="store_true", help="复制后删掉无关大包")
    parser.add_argument("--from", dest="source", default=None, help="源解释器目录")
    args = parser.parse_args()

    print(f"整合包目录: {HERE}")
    print(f"环境目录:   {DEST}\n")

    if args.check:
        ready, missing, reported = probe(bundled_python())
        if not reported:
            print("包内解释器不可用（或不存在）。跑 `python bundle_env.py` 复制一份。")
            return 1
        print(f"解释器: {reported}")
        print(f"就绪 ({len(ready)}): {' '.join(ready)}")
        print(f"缺失 ({len(missing)}): {' '.join(missing) or '无'}")
        return 1 if missing else 0

    source = Path(args.source).resolve() if args.source else source_prefix()
    if not source.is_dir():
        print(f"! 源解释器目录不存在：{source}")
        return 1
    print(f"源环境: {source}")
    if copy_env(source, args.force) != 0:
        return 1

    if args.trim:
        print("\n瘦身：")
        removed, freed = trim()
        print(f"   删除 {removed} 个包，释放 {freed:.2f} GB")

    print("\n校验：")
    ready, missing, reported = probe(bundled_python())
    print(f"   解释器: {reported or '（不可用）'}")
    print(f"   就绪 ({len(ready)}): {' '.join(ready)}")
    print(f"   缺失 ({len(missing)}): {' '.join(missing) or '无'}")
    if missing:
        print("\n! 环境不完整。可以再跑一次 `python bundle_env.py --force`，"
              "或者用 `python install_env.py` 让 pip 补装。")
        return 1
    print("\n完成。用 `run.bat` 或直接调用包内解释器：")
    print(f'   "{bundled_python()}" run_pipeline.py')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
