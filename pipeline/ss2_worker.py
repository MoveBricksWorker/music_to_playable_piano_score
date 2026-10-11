"""在 **SheetSage2 专用解释器**里跑第 ④ 阶段，产出管线要的标签文件。

## 为什么必须单独一个解释器

SheetSage2 的远程代码写于 `transformers 4.45`，而主管线要求更现代的版本
（5.x 删掉了它依赖的 `BartDecoder(..., embed_tokens=...)` 之类接口）。
**两者不能共用一个环境。** 原始项目的解法就是"独立环境 + shell out"
（见 `scripts/sheetsage2_stem.py` 的模块说明），本文件把那套做法固化下来。

调用方式（由 `run_pipeline.stage_sheetsage2` 负责）：

    <包>/venv-ss2/Scripts/python.exe pipeline/ss2_worker.py <曲名> <音频> <输出目录>

## 为什么还需要父模型

SheetSage2 是 **adapter**：`model.safetensors` 只有 218 MB，真正的编码器是父模型
`m-a-p/MERT-v2-FullSong`（2.4 GB）。`config.json` 里写着

    "base_model_name_or_path": "m-a-p/MERT-v2-FullSong",
    "base_model_revision":     "d8ba1c74…"        ← 缓存必须按这个 revision 存
    "base_model_sha256":       "e6dd2ab1…"        ← 加载后还会逐文件校验

**少任何一样都会加载失败**，而且报的错（"couldn't find it in the cached files"）
不会告诉你缺的是父模型。整合包必须同时带上：

    models/hf/hub/models--m-a-p--MERT-v2-FullSong/snapshots/<base_model_revision>/
        model.safetensors          2.4 GB，SHA-256 必须等于 base_model_sha256
        config.json
        configuration_mert2.py
        modeling_mert2.py
    models/hf/modules/transformers_modules/m-a-p/MERT-v2-FullSong/<revision>/
        configuration_mert2.py + modeling_mert2.py   （trust_remote_code 走这里）

## 产出

直接调 `model.transcribe()`，它本身就会写出管线要的文件，本脚本负责搬运 + 兜底：

    beat.lab  downbeat.lab  key.lab  chord.lab  chords.mid  melody_vocal.mid

`key.lab`/`chord.lab` 是**原样搬**的，格式与四首样例逐字一致。
`melody_vocal.mid` 在纯器乐上是空的，此时回退到 `melody_instrumental.mid`——
但会明确打印用了哪一个，因为"旋律来自哪个声源"直接影响下游的串音删除。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

#: 管线要的标签文件。缺一个都算失败。
REQUIRED = ("beat.lab", "downbeat.lab", "key.lab", "chord.lab",
            "chords.mid", "melody_vocal.mid")

#: 旋律的三个候选，按优先级。
MELODY_CANDIDATES = ("melody_vocal.mid", "melody_instrumental.mid", "melody.mid")


def _configure_hf(hf_home: str | None) -> None:
    """**必须在 import transformers 之前**设置：huggingface_hub 在导入期读这些变量。"""
    if hf_home:
        os.environ["HF_HOME"] = hf_home
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "5")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "5")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("song")
    parser.add_argument("audio")
    parser.add_argument("labels", help="标签输出目录")
    parser.add_argument("--hf-home", default=None, help="模型缓存根（整合包是 models/hf）")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--keep-score", action="store_true",
                        help="保留 SheetSage2 的全部产物（谱面/试听等），默认只留管线要的")
    args = parser.parse_args()

    _configure_hf(args.hf_home)

    import torch
    from transformers import AutoModel

    audio = Path(args.audio).resolve()
    labels = Path(args.labels).resolve()
    if not audio.is_file():
        print(f"!! 找不到音频 {audio}", file=sys.stderr)
        return 2
    labels.mkdir(parents=True, exist_ok=True)

    print(f"  [ss2] 音频 {audio.name}", flush=True)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("  [ss2] 无 CUDA，改用 CPU（会慢很多）", flush=True)
        device = "cpu"

    started = time.time()
    model = AutoModel.from_pretrained("m-a-p/SheetSage2", trust_remote_code=True).eval()
    model = model.to(device)
    print(f"  [ss2] 模型加载 {time.time() - started:.1f}s（{device}）", flush=True)

    started = time.time()
    result = model.transcribe(str(audio), output_dir=str(labels))
    print(f"  [ss2] 推理 {time.time() - started:.1f}s，"
          f"{result.get('num_events', '?')} 个事件，"
          f"{result.get('melody_notes', '?')} 个旋律音", flush=True)

    # ---- 产出校验与兜底 --------------------------------------------------- #
    produced = sorted(p.name for p in labels.iterdir() if p.is_file())

    melody_used = None
    vocal = labels / "melody_vocal.mid"
    # 纯器乐上 `melody_vocal.mid` 只是个空壳（几十字节的头），此时回退到器乐旋律。
    # 阈值取 128 B：一个空 MIDI 大约 60–90 B，真有音符的至少几百字节。
    if not vocal.is_file() or vocal.stat().st_size <= 128:
        for candidate in MELODY_CANDIDATES[1:]:
            source = labels / candidate
            if source.is_file() and source.stat().st_size > 128:
                shutil.copy2(source, vocal)
                melody_used = candidate
                break

    missing = [name for name in REQUIRED if not (labels / name).is_file()]
    if missing:
        print(f"!! [ss2] 产出不全，缺 {missing}", file=sys.stderr)
        print(f"    目录里实际有：{produced}", file=sys.stderr)
        return 1

    # 记下旋律的来源。**下游要靠它决定"要不要删串音"**：
    #   vocal        —— 旋律是独立的人声，与钢琴轨同刻同音高的音确实是串音，该删。
    #   instrumental —— 旋律本来就是这件乐器自己弹的（独奏曲），"串音删除"会把
    #                   真实音符删掉。实测肖邦练习曲上会删掉 272 个音（12%）。
    marker = labels / ".melody_source"
    marker.write_text(melody_used or "melody_vocal.mid", encoding="utf-8")

    print(f"  [ss2] 标签齐备：{', '.join(REQUIRED)}", flush=True)
    if melody_used:
        print(f"  [ss2] 旋律取自 {melody_used}（melody_vocal 为空，说明这首是器乐）", flush=True)
        print("  [ss2] 因此**不做串音删除**——旋律与钢琴是同一条演奏，删了就是删真音符",
              flush=True)
    if not args.keep_score:
        # 只留管线要的：其余产物（events.json / tokens.txt / abc 谱面 …）会让人误以为
        # 那些也是管线输入，而且体积不小。
        for path in labels.iterdir():
            if path.is_file() and path.name not in REQUIRED + (".melody_source",):
                path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
