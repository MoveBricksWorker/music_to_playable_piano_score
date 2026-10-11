"""Run SheetSage2 over one or more tracks and keep its outputs.

Separate from the ensemble2score pipeline on purpose: this is the prior-producing
step, and its outputs (beats, key, chords, melody) are consumed by the pipeline
later.  Running it in its own interpreter also keeps transformers out of the main
one.

用法：

    python pipeline/run_sheetsage2.py <曲名> [<曲名> ...]
    python pipeline/run_sheetsage2.py                  # 跑内置的四首样例

音频按下面的顺序找：

    1. `work/audio/<曲名>.mp3`（或 .wav）—— 管线把输入放这里
    2. `$E2S_AUDIO_DIR/<曲名>.mp3`
    3. `$E2S_AUDIO_DIR/<原名>.mp3`（内置四首的原始文件名，见 DEFAULT_TRACKS）

输出目录：`$E2S_LABEL_DIR`，默认 `work/sheetsage2/<曲名>/`
——**必须与 `run_pipeline.stage_sheetsage2` 期望的位置一致**，
否则管线找不到标签、会报"产出不全"。

**模型许可提醒**：SheetSage2 及其 MERT-v2 父模型是 CC-BY-NC-4.0（非商业）。
本项目只用它产出的标签，不随代码分发权重；重新推理前请确认你的用途。
见 THIRD_PARTY_NOTICES.md。
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent

#: 模型缓存根。整合包把它放在 `<包>/models/hf`；仓库里可能只有系统缓存。
#: **必须在 import transformers 之前设好**：`huggingface_hub` 在导入期就读这些变量。
_BUNDLED_HF = HERE / "models" / "hf"
if _BUNDLED_HF.is_dir():
    os.environ.setdefault("HF_HOME", str(_BUNDLED_HF))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
else:
    os.environ.setdefault("HF_HOME", str(HERE / "hf_home"))

# 无网络时别让每个请求干等：把超时压短，失败要快、错误要清。
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "5")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "5")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

#: 音频搜索目录。默认整合包自带的 `data/audio/`。
AUDIO_DIR = Path(os.environ.get("E2S_AUDIO_DIR") or (HERE / "data" / "audio" / ""))
if not AUDIO_DIR.is_dir():
    # 整合包里输入在 `work/audio/`，仓库里可能只有 `data/audio/`
    fallback = HERE / "work" / "audio"
    if fallback.is_dir():
        AUDIO_DIR = fallback

#: 标签输出根。**默认要与 run_pipeline.stage_sheetsage2 读的位置一致。**
OUT_ROOT = Path(os.environ.get("E2S_LABEL_DIR") or (HERE / "work" / "sheetsage2"))

#: 内置曲目表：曲名 → 原始文件名。只用来给"没给曲名"或"简名找不到"时兜底。
DEFAULT_TRACKS: dict[str, str] = {
    "blessing": "blessing.mp3",
    "collapse": "collapse.mp3",
    "yoake": "yoake.mp3",
    "victim": "victim.mp3",
}

#: 管线真正需要的标签文件（缺一个都算失败）。
REQUIRED_LABELS = (
    "beat.lab", "downbeat.lab", "key.lab", "chord.lab", "melody_vocal.mid",
)


def audio_dirs() -> list[Path]:
    """音频搜索目录，按优先级排列。

    **`work/audio/` 必须排在前面**：管线的输入就放在那里（`run_pipeline` 读的也是
    这个位置）。早先只认 `data/audio/`，于是新曲子在 `work/audio/` 里明明有音频，
    这里却报"找不到"——一个纯粹由搜索顺序造成的假失败。
    """
    dirs: list[Path] = [HERE / "work" / "audio"]
    if os.environ.get("E2S_AUDIO_DIR"):
        dirs.append(Path(os.environ["E2S_AUDIO_DIR"]))
    dirs.append(AUDIO_DIR)
    # 去重且保序
    seen: set[Path] = set()
    unique: list[Path] = []
    for directory in dirs:
        resolved = directory.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(directory)
    return unique


def find_audio(song: str) -> Path | None:
    """找音频：先认 `$PIPELINE_AUDIO`（调用方直接给的全路径），再按目录搜。

    早先这里只认硬编码的四首，别的曲名会被静默 `continue` 掉、脚本还返回 0
    （"成功"），于是 `run_pipeline` 只能报一句含糊的"SheetSage2 产出不全"。
    现在任意曲名都能找到自己的音频。
    """
    override = os.environ.get("PIPELINE_AUDIO")
    if override and Path(override).is_file():
        return Path(override)

    stems = [song]
    if DEFAULT_TRACKS.get(song):
        stems.append(DEFAULT_TRACKS[song])
    for directory in audio_dirs():
        if not directory.is_dir():
            continue
        for stem in stems:
            for suffix in (".mp3", ".wav", ".flac", ".m4a"):
                candidate = directory / f"{stem}{suffix}"
                if candidate.is_file():
                    return candidate
    return None


def main() -> int:
    wanted = [a for a in sys.argv[1:] if not a.startswith("-")]
    render_score = "--no-render" not in sys.argv
    songs = wanted or list(DEFAULT_TRACKS)

    import torch
    from transformers import AutoModel

    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}", flush=True)
    print(f"音频搜索 {[str(d) for d in audio_dirs()]}", flush=True)
    print(f"标签输出 {OUT_ROOT}", flush=True)
    print(f"曲目 {', '.join(songs)}", flush=True)

    # 先把音频都找齐，缺的直接报错退出——不要先花时间加载模型才发现没音频。
    resolved: list[tuple[str, Path]] = []
    missing_audio: list[str] = []
    for song in songs:
        source = find_audio(song)
        if source is None:
            missing_audio.append(song)
            continue
        resolved.append((song, source))
    if missing_audio:
        print(f"!! 找不到音频：{', '.join(missing_audio)}", flush=True)
        print("   找过这些目录（按顺序）：", flush=True)
        for directory in audio_dirs():
            mark = "存在" if directory.is_dir() else "不存在"
            print(f"     [{mark}] {directory}", flush=True)
        print("   文件名需为 '<曲名>.mp3' / '<曲名>.wav' 等。", flush=True)
        return 2
    if not resolved:
        print("!! 没有可处理的曲目", flush=True)
        return 2

    started = time.time()
    model = AutoModel.from_pretrained("m-a-p/SheetSage2", trust_remote_code=True).eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    print(f"model loaded in {time.time() - started:.0f}s", flush=True)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = []
    for song, source in resolved:
        out_dir = OUT_ROOT / song
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 70}\n=== {song}: {source.name}\n{'=' * 70}", flush=True)

        started = time.time()
        options = {"output_dir": str(out_dir)}
        if render_score:
            # 可打印的谱面与钢琴试听；所需素材首次使用时才拉取。
            # 离线环境下这部分可能失败，加 `--no-render` 跳过。
            options["render_score"] = ("pdf",)
            options["render_audio"] = True
        try:
            result = model.transcribe(str(source), **options)
        except Exception:
            print(f"!! {song} failed\n{traceback.format_exc()}", flush=True)
            summary.append({"track": song, "status": "failed"})
            continue
        elapsed = time.time() - started

        # 产出是否齐全：这是调用方唯一会检查的东西，缺了就直接算失败。
        produced = sorted(p.name for p in out_dir.iterdir() if p.is_file())
        absent = [name for name in REQUIRED_LABELS if name not in produced]
        if absent:
            print(f"!! {song} 产出不全，缺 {absent}", flush=True)
            summary.append({"track": song, "status": "incomplete", "missing": absent})
            continue

        events_payload = json.loads((out_dir / "events.json").read_text(encoding="utf-8"))
        events = events_payload["events"] if isinstance(events_payload, dict) else events_payload
        keys = sorted({e["values"]["key"] for e in events if "key" in (e.get("values") or {})})
        chords = sorted({e["values"]["chord"] for e in events if "chord" in (e.get("values") or {})})
        structures = sorted(
            {e["values"]["structure"] for e in events if "structure" in (e.get("values") or {})}
        )

        entry = {
            "track": song,
            "status": "ok",
            "audio": str(source),
            "seconds": round(elapsed, 1),
            "duration_seconds": result.get("duration_seconds"),
            "windows": len(result.get("windows") or []),
            "abc_error": result.get("abc_error"),
            "abc_chars": len(result.get("abc") or ""),
            "keys": keys,
            "distinct_chords": len(chords),
            "structures": structures,
            "n_events": len(events),
            "files": produced,
        }
        summary.append(entry)
        print(
            f"--- {song}: {elapsed / 60:.1f} min, {len(events)} events, "
            f"key={keys}, {len(chords)} distinct chords, structures={structures}",
            flush=True,
        )
        (out_dir / "delivery.json").write_text(
            json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    (OUT_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n" + "=" * 70)
    failed = 0
    for entry in summary:
        if entry.get("status") != "ok":
            failed += 1
            reason = entry.get("missing") or "见上方日志"
            print(f"  {entry['track']:<12} {entry['status'].upper()}  {reason}")
            continue
        print(
            f"  {entry['track']:<12} {entry['seconds'] / 60:>5.1f} min  "
            f"{entry['n_events']:>5} events  key={entry['keys']}  "
            f"{entry['distinct_chords']:>3} chords  {entry['structures']}"
        )

    # 有任何一首失败就返回非零：早先无论发生什么都 return 0，
    # 调用方只能靠"事后数文件"猜出了什么事，错误信息全丢了。
    return 1 if failed or len(summary) != len(resolved) else 0


if __name__ == "__main__":
    raise SystemExit(main())
