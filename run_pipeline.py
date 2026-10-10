"""最优版端到端管线：音频 → 可弹的钢琴谱（MusicXML / MIDI / PDF）。

    python run_pipeline.py                     # 四首，从头跑
    python run_pipeline.py victim              # 只跑一首
    python run_pipeline.py victim --from transkun   # 从转谱开始（跳过分离）
    python run_pipeline.py --list-stages

## 完整链路

    ① 分离   Demucs htdemucs_6s → piano.wav ；htdemucs → other.wav
    ② 相加   piano(6轨) + other(4轨) → sum.wav
    ③ 转谱   Transkun 2.0 对 sum.wav → sum_transkun.mid（复音钢琴转谱）
    ④ 节拍   SheetSage2 → beat / downbeat / key / chord / melody_vocal
    ⑤ 编排   12 tick/拍 双格量化 → 中音C分手 → 旋律最高声部 → RH/LH
    ⑥ 修正   可弹性：手内跨度 ≤19 半音、同时音 ≤5
    ⑦ 雕版   时值规整 → 小节填满 → 按调号拼写 → 速度标记
    ⑧ 导出   MusicXML / MIDI / PDF
    ⑨ 验收   可弹性 + 谱面形状 R1–R4 + 小节/时长/旋律回归

## 为什么是"相加"

实测 6 轨的 `piano` 分轨在副歌掉 6 dB 而 4 轨的 `other` 在同处涨 25.8 dB ——
那是 htdemucs_6s 把副歌伴奏路由到了自己的 other 头（Demucs 文档对 piano/guitar
这一对就是这个警告）。相加后副歌起音 147、和弦音占比 81.3%，优于任一分轨单独使用。

## 为什么是 12 tick/拍

4 tick/拍时 1/3 拍 = 4/3 tick 不是整数，三连音会在吸附那一步被压掉
（三个音变一个音）。12 tick/拍下 1/3 拍 = 4 tick、1/6 拍 = 2 tick，都能表示。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

#: 各阶段名称，供 `--from` 使用
STAGES = ("separate", "sum", "transkun", "sheetsage2", "arrange", "fix", "engrave", "render")

SONGS = ("victim", "yoake", "collapse", "blessing")

#: Demucs 模型：6 轨里有专门的 piano 头，4 轨里有更可靠的 other 头
MODEL_6S = "htdemucs_6s"
MODEL_4S = "htdemucs"

MUSESCORE_DEFAULT = r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"


# --------------------------------------------------------------------------- #
# 整合包自带模型：把缓存的根指到包内，使管线**离线**可用
# --------------------------------------------------------------------------- #
def _apply_model_env() -> dict[str, str]:
    """把 `HF_HOME` / `TORCH_HOME` 指到整合包内的 `models/`，并**强制离线**。

    只在包内确实存在对应目录时才改，否则保留系统默认（这样开发机上仍然
    用系统缓存，不会因为整合包没搬模型而失效）。

    * `HF_HOME`    → `models/hf`    （SheetSage2；`huggingface_hub` 从 `<HF_HOME>/hub` 找）
    * `TORCH_HOME` → `models/torch` （demucs；torch hub 从 `<TORCH_HOME>/hub/checkpoints` 找）

    ## 为什么必须设 `HF_HUB_OFFLINE=1`

    demucs 4.1.0 的加载顺序是：**先**从 HuggingFace 拉 `htdemucs.yaml`
    （用 `htdemucs.safetensors`），失败**才**回退到旧仓库用本地 `.th`。
    它没有"只用本地文件"的入口（`repo=None` 不接受路径）。

    于是无网络时它仍会去连，实测**每个模型白等 284 秒**才走回退。
    设 `HF_HUB_OFFLINE=1` 让 huggingface_hub 立刻从缓存失败，实测降到 **1.8 秒**。
    """
    applied: dict[str, str] = {}

    hf = HERE / "models" / "hf" / "hub"
    if hf.is_dir():
        os.environ["HF_HOME"] = str(HERE / "models" / "hf")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        # 万一某条路径没走缓存，也把超时压短，别让用户干等
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "5")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "5")
        applied["HF_HOME"] = os.environ["HF_HOME"]
        applied["HF_HUB_OFFLINE"] = "1"

    torch_home = HERE / "models" / "torch" / "hub" / "checkpoints"
    if torch_home.is_dir():
        os.environ["TORCH_HOME"] = str(HERE / "models" / "torch")
        applied["TORCH_HOME"] = os.environ["TORCH_HOME"]

    return applied


def _check_env() -> bool:
    """确认 `torch` 可用；不可用就给出明确的下一步指令。"""
    try:
        import torch  # noqa: F401
    except ImportError:
        print("!! 当前解释器没有 torch。整合包自带的解释器在：")
        print(f"     {HERE / 'python' / 'python.exe'}")
        print("   三种办法：")
        print(f"     1) 用自带解释器：\"{HERE / 'python' / 'python.exe'}\" run_pipeline.py …")
        print(f"     2) 或直接跑 run.bat（会自动挑解释器）")
        print(f"     3) 复制环境：{sys.executable} bundle_env.py")
        return False
    return True


# --------------------------------------------------------------------------- #
# ③④⑤⑥⑦ 用的编排/雕版（项目内模块，逻辑未改）
# --------------------------------------------------------------------------- #
def _import_pipeline():
    from pipeline import arrange, engrave, fix_playable, verify_playable, check_shapes

    return arrange, engrave, fix_playable, verify_playable, check_shapes


# --------------------------------------------------------------------------- #
# ① 分离
# --------------------------------------------------------------------------- #
def stage_separate(song: str, work: Path, *, force: bool, device: str) -> tuple[Path, Path]:
    """Demucs 分离。产出 `stems6/<曲目>/piano.wav` 与 `stems4/<曲目>/other.wav`。

    **注意 demucs 的输出层级**：`-o <out>` 会得到 `<out>/<模型名>/<曲目>/*.wav`，
    比直觉多一层模型名目录。把它直接当目标目录会找不到文件
    （实测表现为"分离产物不齐：piano=False other=False"，而 demucs 其实成功了）。
    所以这里先输出到临时目录，再把需要的那一轨搬到 `stemsN/<曲目>/` 下。
    """
    audio = work / "audio" / f"{song}.mp3"
    if not audio.is_file():
        raise FileNotFoundError(f"缺少输入音频 {audio}")

    piano = work / "stems6" / song / "piano.wav"
    other = work / "stems4" / song / "other.wav"
    staging = work / "_demucs"

    for model, wanted, stem in (
        (MODEL_6S, piano, "piano.wav"),
        (MODEL_4S, other, "other.wav"),
    ):
        if wanted.is_file() and not force:
            print(f"    {model}: 已有 {wanted.relative_to(work)}，跳过")
            continue

        produced = staging / model / song / stem
        if force or not produced.is_file():
            staging.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable, "-m", "demucs",
                "-n", model, "-d", device,
                "-o", str(staging),
                str(audio),
            ]
            print(f"    {model} …", flush=True)
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0 or not produced.is_file():
                tail = (result.stderr or result.stdout or "").strip().splitlines()[-8:]
                raise RuntimeError(
                    f"demucs {model} 失败（exit={result.returncode}，"
                    f"产出 {produced.is_file()}）：\n" + "\n".join(tail)
                )

        wanted.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(produced, wanted)
        print(f"      -> {wanted.relative_to(work)}")

    if not piano.is_file() or not other.is_file():
        raise RuntimeError(
            f"分离产物不齐：piano={piano.is_file()} other={other.is_file()}"
        )
    return piano, other


# --------------------------------------------------------------------------- #
# ② 相加
# --------------------------------------------------------------------------- #
def stage_sum(song: str, work: Path, *, force: bool) -> Path:
    import numpy as np
    import soundfile as sf

    piano = work / "stems6" / song / "piano.wav"
    other = work / "stems4" / song / "other.wav"
    out = work / "sum" / f"{song}.wav"
    if out.is_file() and not force:
        print(f"    已有 {out.relative_to(work)}，跳过")
        return out

    a, sr_a = sf.read(str(piano), dtype="float32", always_2d=True)
    b, sr_b = sf.read(str(other), dtype="float32", always_2d=True)
    if sr_a != sr_b:
        raise RuntimeError(f"采样率不同：{sr_a} vs {sr_b}")
    n = min(len(a), len(b))
    y = a[:n].astype(np.float64) + b[:n].astype(np.float64)
    peak = float(np.abs(y).max())
    gain = 1.0
    if peak > 0.999:
        gain = 0.999 / peak
        y = y * gain
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), y.astype(np.float32), sr_a, subtype="PCM_16")
    print(f"    {n / sr_a:.1f}s @ {sr_a} Hz，峰值 {peak:.3f}，增益 {gain:.4f}")
    return out


# --------------------------------------------------------------------------- #
# ③ 转谱
# --------------------------------------------------------------------------- #
def stage_transkun(song: str, work: Path, *, force: bool, device: str) -> Path:
    summed = work / "sum" / f"{song}.wav"
    out = work / "transkun" / f"{song}.mid"
    if out.is_file() and not force:
        print(f"    已有 {out.relative_to(work)}，跳过")
        return out
    if not summed.is_file():
        raise FileNotFoundError(f"缺少 {summed}（先跑 sum 阶段）")
    out.parent.mkdir(parents=True, exist_ok=True)
    # 走子进程：transkun_probe 自己会 stub 掉不需要的训练期依赖（moduleconf），
    # 子进程还能把它的日志与主流程隔开，并给出可信的退出码。
    cmd = [sys.executable, str(HERE / "pipeline" / "transkun_probe.py"),
           str(summed), str(out), "--device", device]
    result = subprocess.run(cmd, capture_output=True, text=True)
    for line in (result.stdout or "").splitlines():
        if line.startswith(("[audio]", "[infer]", "[result]")):
            print(f"    {line}")
    if result.returncode != 0:
        tail = (result.stderr or "").strip().splitlines()[-8:]
        raise RuntimeError("transkun 失败：\n" + "\n".join(tail))
    if not out.is_file():
        raise RuntimeError(f"transkun 没有产出 {out}")
    return out


# --------------------------------------------------------------------------- #
# ④ SheetSage2
# --------------------------------------------------------------------------- #
def stage_sheetsage2(song: str, work: Path, *, force: bool, render: bool) -> Path:
    """取 SheetSage2 的标签（beat / downbeat / key / chord / melody_vocal）。

    精简版**自带四首的标签**，所以默认直接吃现成的。要重新推理需要
    `transformers`（项目里是独立的 `.venv-ss2`，与主管线分开，避免版本冲突），
    因此只有在标签缺失或显式 `--force` 时才真的去调模型。
    """
    labels = work / "sheetsage2" / song
    needed = ["beat.lab", "downbeat.lab", "key.lab", "chord.lab", "melody_vocal.mid"]
    have_all = all((labels / name).is_file() for name in needed)
    if have_all and not force:
        print(f"    自带标签齐备（{', '.join(needed)}），跳过推理")
        return labels

    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "需要重新推理 SheetSage2，但当前解释器没有 transformers。\n"
            "  项目里这一步用的是独立环境 .venv-ss2。两种办法：\n"
            "    1) 用自带标签：删除 --force 即可（data/sheetsage2/ 下已备好四首）\n"
            "    2) 装依赖：pip install transformers torch\n"
            f"  原始错误：{exc}"
        ) from exc

    audio = work / "audio" / f"{song}.mp3"
    labels.mkdir(parents=True, exist_ok=True)
    print("    调用 SheetSage2（首次会从 HuggingFace 拉模型）…", flush=True)
    cmd = [sys.executable, str(HERE / "pipeline" / "run_sheetsage2.py"), song]
    env = dict(os.environ)
    env.setdefault("HF_HOME", str(HERE / "hf_home"))
    env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    env["PIPELINE_AUDIO"] = str(audio)
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(HERE))
    print((result.stdout or "").strip()[-2000:])
    if result.returncode != 0:
        tail = (result.stderr or "").strip().splitlines()[-8:]
        raise RuntimeError("SheetSage2 失败：\n" + "\n".join(tail))
    missing = [name for name in needed if not (labels / name).is_file()]
    if missing:
        raise RuntimeError(f"SheetSage2 产出不全，缺 {missing}")
    return labels


# --------------------------------------------------------------------------- #
# ⑤⑥⑦⑧ 编排 → 修正 → 雕版 → 导出
# --------------------------------------------------------------------------- #
def stage_arrange(song: str, work: Path, out_dir: Path, *,
                  lattice_cost: str = "density", switch_margin: float = 0.01,
                  snap_policy: str = "family") -> dict:
    """⑤ 编排：转谱 MIDI + SheetSage2 标签 → RH/LH 编排文件。"""
    arrange, _e, _f, _vp, _cs = _import_pipeline()
    arrangement_dir = out_dir / "_arrangements"
    arrangement = arrangement_dir / f"{song}.txt"

    info = arrange.build_arrangement(
        song,
        work / "transkun" / f"{song}.mid",
        work / "sheetsage2" / song,
        arrangement,
        cost_normalization=lattice_cost,
        switch_margin=switch_margin,
        snap_policy=snap_policy,
    )
    print(f"    {info['time_signature']}，{info['bars']} 小节，"
          f"{info['tempo_bpm']} BPM，RH {info['rh_notes']} / LH {info['lh_notes']}")
    for warning in info["warnings"]:
        print(f"    ! {warning}")

    # 可弹性修正读 `<root>/<song>/arrangement.txt`，所以先在输出目录铺一份
    song_dir = out_dir / song
    song_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(arrangement, song_dir / "arrangement.txt")
    shutil.copy2(arrangement_dir / f"{song}.melody.json", song_dir / f"{song}.melody.json")

    # 量化口径落盘：`--lattice-cost` / `--switch-margin` / `--snap-policy` 任何一个变了
    # 谱面就不同，事后必须能从产物看出用的是哪一版。engrave 阶段会把它写进
    # `engrave.json`（这里先暂存，避免 stage_arrange 与 stage_engrave 之间再传参）。
    (song_dir / ".quantizer.json").write_text(
        json.dumps(info["quantizer"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return info


def stage_fix(song: str, out_dir: Path) -> dict:
    """⑥ 可弹性修正：手内跨度 ≤19 半音、同时音 ≤5。"""
    _a, _e, fix_playable, _vp, _cs = _import_pipeline()
    arrangement = out_dir / "_arrangements" / f"{song}.txt"
    song_dir = out_dir / song
    # process 读的是 <root>/<song>/arrangement.txt；先把它与 _arrangements 对齐，
    # 否则在"只重跑 fix"时读到的是上一次的结果（改动会不幂等）。
    shutil.copy2(arrangement, song_dir / "arrangement.txt")
    fixed = fix_playable.process(song, rules=(True, True, True), source_root=str(out_dir))
    fix_playable.write_arrangement(
        fixed["events"], arrangement, beats_per_bar=fixed["beats_per_bar"]
    )
    shutil.copy2(arrangement, song_dir / "arrangement.txt")
    print(f"    {fixed['stats']}")
    return fixed["stats"]


def stage_engrave(song: str, out_dir: Path, *, mscore: str | None) -> None:
    """⑦ 雕版：时值规整 → 小节填满 → 按调号拼写。"""
    _a, engrave, _f, _vp, _cs = _import_pipeline()
    engrave.engrave(
        song, out_dir / "_arrangements" / f"{song}.txt", out_dir / song,
        ticks=12, pdf=False, mscore=mscore,
    )


def stage_render(song: str, out_dir: Path, *, mscore: str | None) -> None:
    """⑧ 导出 PDF（MuseScore）。"""
    if not mscore:
        print("    未指定 MuseScore，跳过")
        return
    _a, engrave, _f, _vp, _cs = _import_pipeline()
    song_dir = out_dir / song
    result = engrave.export_pdf_fixed(
        song_dir / f"{song}.musicxml", song_dir, song, mscore
    )
    if not (song_dir / f"{song}.pdf").is_file():
        raise RuntimeError(f"PDF 没有生成（export_pdf_fixed 返回 {result!r}）")


# --------------------------------------------------------------------------- #
# ⑨ 验收
# --------------------------------------------------------------------------- #
def stage_verify(out_dir: Path, songs: list[str]) -> int:
    """⑨ 验收：小节数、音符数、旋律保留、谱面时值合规；并提示两个细项验收器。

    ## 口径说明（重要）

    这里读的是 `engrave` 阶段写出的 `report.json`，它是**编排层**的口径
    （`evaluate_arrangement` 在 arrangement 对象上算，且发生在雕版规整**之前**）。
    谱面层的数据只能靠下面两个细项验收器，它们会真的去解析 MusicXML：

        python -m pipeline.verify_playable --root <out>   # 跨度 / 同时音 / 谱面音数
        python -m pipeline.check_shapes   --root <out>    # R1–R4 时值形状

    两者数字不一致是正常的：谱面音数通常比编排音数高 10~14%，差额是**跨小节线的音
    被切成连音线碎片**（起音位置一个都没动，不是丢音也不是多音）。
    """
    print("\n" + "=" * 96)
    print("验收（编排层口径；谱面层请跑下面两个细项验收器）")
    print("=" * 96)

    failures = 0
    print(f"{'曲目':<10} {'小节':>5} {'编排起音':>8} {'编排音数':>8} "
          f"{'旋律保留':>9} {'音/秒':>8}")
    for song in songs:
        report_path = out_dir / song / "report.json"
        if not report_path.is_file():
            print(f"{song:<10} 缺 report.json")
            failures += 1
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        bars = report.get("bars", "?")
        onsets = report.get("onsets", "?")
        arranged = report.get("notes", "?")
        melody = report.get("melody_kept", "?")
        if isinstance(melody, float):
            melody = f"{melody:.3f}"
        per_second = report.get("notes_per_second") or 0.0
        per_second = f"{per_second:.2f}" if per_second else "—"
        print(f"{song:<10} {bars:>5} {onsets:>8} {arranged:>8} "
              f"{melody:>9} {per_second:>8}")
        if report.get("warnings"):
            for warning in report["warnings"]:
                print(f"    ! {warning}")

    print("\n细项验收（真的解析 MusicXML，唯一可信口径）：")
    print(f"    python -m pipeline.verify_playable --root {out_dir.name}")
    print(f"    python -m pipeline.check_shapes   --root {out_dir.name}")
    return failures


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(
        description="音频 → 可弹钢琴谱（MusicXML / MIDI / PDF）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("songs", nargs="*", default=None,
                        help=f"要处理的曲子，默认全部：{', '.join(SONGS)}")
    parser.add_argument("--work", default=None, help="工作目录（默认 ./work）")
    parser.add_argument("--out", default=None, help="成品目录（默认 ./out）")
    parser.add_argument("--from", dest="from_stage", default="separate",
                        choices=STAGES, help="从哪个阶段开始（默认 separate）")
    parser.add_argument("--to", dest="to_stage", default="render",
                        choices=STAGES, help="跑到哪个阶段为止（默认 render）")
    parser.add_argument("--force", action="store_true",
                        help="重跑本阶段（忽略已有产物）")
    parser.add_argument("--device", default="cuda", help="demucs / transkun 用的设备")
    parser.add_argument("--no-render", action="store_true", help="不生成 PDF")
    parser.add_argument("--mscore", default=MUSESCORE_DEFAULT,
                        help="MuseScore 可执行文件路径")
    parser.add_argument(
        "--lattice-cost", default="density", choices=("density", "raw"),
        help="双格量化的代价口径：density=按格点密度归一化（默认，见 dual_lattice 的说明）；"
             "raw=旧的绝对代价口径（会把噪声读成三连音，仅用于对照）",
    )
    parser.add_argument(
        "--switch-margin", type=float, default=0.01,
        help="density 口径下的相对迟滞：新格子要好过当前格子这么多（每起音的相对代价差）"
             "才切换。0 表示不加迟滞",
    )
    parser.add_argument(
        "--snap-policy", default="family", choices=("family", "nearest"),
        help="起音吸附策略：family=一律吸到本窗口选定的族（默认，节奏族干净）；"
             "nearest=在两族格点里挑离真实位置更近的（位移最小，可能混族）",
    )
    parser.add_argument("--list-stages", action="store_true", help="列出阶段名后退出")
    args = parser.parse_args()

    if args.list_stages:
        for index, name in enumerate(STAGES, 1):
            print(f"  {index}. {name}")
        return 0

    songs = args.songs or list(SONGS)
    work = Path(args.work).resolve() if args.work else HERE / "work"
    out_dir = Path(args.out).resolve() if args.out else HERE / "out"
    work.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_env = _apply_model_env()

    start = STAGES.index(args.from_stage)
    stop = STAGES.index(args.to_stage)
    render = not args.no_render and stop >= STAGES.index("render")

    print("=" * 96)
    print(f"精简版管线   曲子={', '.join(songs)}   阶段 {args.from_stage} → {args.to_stage}")
    print(f"工作目录 {work}")
    print(f"成品目录 {out_dir}")
    if model_env:
        for key, value in model_env.items():
            print(f"{key}={value}")
    else:
        print("模型：用系统缓存（整合包内没有 models/，或尚未运行 make_portable.py）")
    # 只有真的要跑模型阶段时才检查 torch（`--from arrange` 不需要）
    model_start = STAGES.index("separate")
    model_stop = STAGES.index("sheetsage2")
    if start <= model_stop and stop >= model_start and not _check_env():
        return 1
    print("=" * 96)

    failed: list[str] = []
    started = time.time()
    for song in songs:
        print(f"\n{'─' * 96}\n{song}\n{'─' * 96}", flush=True)
        try:
            if start <= STAGES.index("separate") <= stop:
                print("  [1/9] 分离（Demucs 6 轨 piano + 4 轨 other）", flush=True)
                stage_separate(song, work, force=args.force, device=args.device)
            if start <= STAGES.index("sum") <= stop:
                print("  [2/9] 相加", flush=True)
                stage_sum(song, work, force=args.force)
            if start <= STAGES.index("transkun") <= stop:
                print("  [3/9] 转谱（Transkun 2.0）", flush=True)
                stage_transkun(song, work, force=args.force, device=args.device)
            if start <= STAGES.index("sheetsage2") <= stop:
                print("  [4/9] 节拍与和弦（SheetSage2）", flush=True)
                stage_sheetsage2(song, work, force=args.force, render=render)
            if start <= STAGES.index("arrange") <= stop:
                print("  [5/9] 编排", flush=True)
                stage_arrange(song, work, out_dir,
                              lattice_cost=args.lattice_cost,
                              switch_margin=args.switch_margin,
                              snap_policy=args.snap_policy)
            if start <= STAGES.index("fix") <= stop:
                print("  [6/9] 可弹性修正（跨度 ≤19 半音、同时音 ≤5）", flush=True)
                stage_fix(song, out_dir)
            if start <= STAGES.index("engrave") <= stop:
                print("  [7/9] 雕版（时值规整 → 小节填满 → 按调号拼写）", flush=True)
                stage_engrave(song, out_dir, mscore=args.mscore)
            if start <= STAGES.index("render") <= stop:
                print("  [8/9] 渲染 PDF", flush=True)
                stage_render(song, out_dir, mscore=args.mscore if render else None)
            print(f"  [9/9] 完成 → {out_dir / song}", flush=True)
        except Exception:
            failed.append(song)
            print(f"  !! {song} 失败\n{traceback.format_exc()}", flush=True)

    elapsed = time.time() - started
    print(f"\n{'=' * 96}")
    if failed:
        print(f"失败：{', '.join(failed)}")
    stage_verify(out_dir, [s for s in songs if s not in failed])
    print(f"总耗时 {elapsed / 60:.1f} 分钟")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
