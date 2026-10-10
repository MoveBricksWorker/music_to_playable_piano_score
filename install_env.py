"""在整合包里创建可独立运行、可离线用的 Python 环境。

用户要求：*「做一个整合包，把那几个要用的模型和环境也搬进去吧」*

本脚本在**脚本自己所在目录**下建 `.venv`（因此搬到哪台机器都能用），
只装管线真正需要的包 —— 不装 transkun 声明的训练期依赖（见下）。

## 为什么不用 `pip install transkun` 的完整依赖

transkun 的 `install_requires` 里有 `ncls`、`torch-optimizer`、`tensorboard`、
`seaborn`、`matplotlib`、`mir-eval`、`moduleconf`。**推理路径一个都不用**，
而它们的安装又慢又容易失败（`ncls` 要编译 C 扩展）。
`pipeline/transkun_probe.py` 已经用 stub 绕过了 `moduleconf`，
所以这里只装它实际 import 的那些。

## 为什么指定 PyTorch 的官方源

`torch==2.6.0+cu124` 这种带 CUDA 标签的版本只在 PyTorch 自己的 index 上有，
PyPI 上的 `torch` 是另一个构建。Demucs 与 Transkun 都要 GPU，所以走 cu124。

用法：
    python install_env.py                 # 建环境并装依赖
    python install_env.py --check         # 只检查现有环境够不够
    python install_env.py --cpu           # 装 CPU 版 torch（体积小，但很慢）
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / ".venv"

TORCH_INDEX = "https://download.pytorch.org/whl/cu124"

#: 管线实际需要的包。版本与开发环境一致（torch 2.6.0+cu124）。
REQUIREMENTS = [
    "torch==2.6.0",
    "torchaudio==2.6.0",
    "numpy==1.26.4",
    "scipy",
    "music21==9.9.2",
    "pretty_midi==0.2.11",
    "soundfile",
    "librosa==0.11.0",
    "soxr",
    "pydub",
    "demucs==4.1.0",
    "transkun==2.0.1",
    "huggingface_hub",
    "transformers",       # ④ 只有在重新推理 SheetSage2 时才用到；自带标签时不需要
]

#: 装完必须能 import 才算成功（import 名，不一定是包名）
IMPORT_CHECK = [
    "torch", "torchaudio", "numpy", "scipy", "music21", "pretty_midi",
    "soundfile", "librosa", "soxr", "pydub", "demucs", "transkun",
]


def run(cmd: list[str], **kwargs) -> int:
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, **kwargs).returncode


def venv_python() -> Path:
    if sys.platform == "win32":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def check(python: Path) -> tuple[list[str], list[str]]:
    """返回 (已就绪的包, 缺的包)。"""
    script = (
        "import importlib.util as u, sys;"
        f"names={IMPORT_CHECK!r};"
        "print('OK ' + ' '.join(n for n in names if u.find_spec(n)));"
        "print('MISSING ' + ' '.join(n for n in names if not u.find_spec(n)))"
    )
    result = subprocess.run([str(python), "-c", script], capture_output=True, text=True)
    ready: list[str] = []
    missing: list[str] = []
    for line in (result.stdout or "").splitlines():
        if line.startswith("OK "):
            ready = line[3:].split()
        elif line.startswith("MISSING "):
            missing = line[8:].split()
    return ready, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="只检查，不安装")
    parser.add_argument("--cpu", action="store_true", help="装 CPU 版 torch")
    parser.add_argument("--recreate", action="store_true", help="删掉已有 .venv 重建")
    args = parser.parse_args()

    print(f"整合包目录: {HERE}")
    print(f"环境目录:   {VENV}\n")

    if args.check:
        if not venv_python().is_file():
            print("环境不存在。跑 `python install_env.py` 创建。")
            return 1
        ready, missing = check(venv_python())
        print(f"已就绪 ({len(ready)}): {' '.join(ready)}")
        print(f"缺失   ({len(missing)}): {' '.join(missing) or '无'}")
        return 1 if missing else 0

    if VENV.is_dir() and args.recreate:
        import shutil

        print("删除已有 .venv …")
        shutil.rmtree(VENV)

    if not venv_python().is_file():
        print("[1/3] 创建虚拟环境")
        if run([sys.executable, "-m", "venv", str(VENV)]) != 0:
            print("! 创建虚拟环境失败")
            return 1
    else:
        print("[1/3] 虚拟环境已存在，复用")

    python = venv_python()
    print("\n[2/3] 升级 pip")
    run([str(python), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"])

    print("\n[3/3] 安装依赖")
    index = "https://download.pytorch.org/whl/cpu" if args.cpu else TORCH_INDEX
    torch_pkgs = [p for p in REQUIREMENTS if p.startswith(("torch==", "torchaudio=="))]
    other_pkgs = [p for p in REQUIREMENTS if p not in torch_pkgs]

    if run([
        str(python), "-m", "pip", "install", "--index-url", index, *torch_pkgs,
    ]) != 0:
        print("! torch 安装失败")
        return 1
    if run([str(python), "-m", "pip", "install", *other_pkgs]) != 0:
        print("! 其余依赖安装失败")
        return 1

    ready, missing = check(python)
    print(f"\n就绪 ({len(ready)}): {' '.join(ready)}")
    print(f"缺失 ({len(missing)}): {' '.join(missing) or '无'}")
    if missing:
        print("\n提示：`transkun` 声明的训练期依赖（ncls / tensorboard / torch-optimizer 等）"
              "无需安装，\n      pipeline/transkun_probe.py 已绕过它们。")
        return 1
    print("\n环境就绪。下一步：python make_portable.py  然后  python run_pipeline.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
