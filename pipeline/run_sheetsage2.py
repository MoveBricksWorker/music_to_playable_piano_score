"""Run SheetSage2 over every track and keep its outputs.

Separate from the ensemble2score pipeline on purpose: this is the prior-producing
step, and its outputs (beats, key, chords, structure) are consumed by an adapter
later.  Running it in its own environment also keeps transformers 4.x out of the
project's interpreter.

Must be run with an interpreter that has `transformers` + SheetSage2's own code,
from the repository root:

    python pipeline/run_sheetsage2.py victim
    python pipeline/run_sheetsage2.py                    # 用内置曲目表
    E2S_AUDIO_DIR=/path/to/audio python pipeline/run_sheetsage2.py victim

`E2S_AUDIO_DIR` 下的文件名满足 `<曲目>.mp3` 即可，不必是原始文件名。

**模型许可提醒**：SheetSage2 及其 MERT-v2 父模型是 CC-BY-NC-4.0（非商业）。
本项目只用它产出的标签，不随代码分发权重；重新推理前请确认你的用途。
见 THIRD_PARTY_NOTICES.md。
"""
from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(HERE / "hf_home"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

OUT_ROOT = HERE / "sheetsage2_out"

#: 音频所在目录。默认用整合包自带的 `data/audio/`，可用 `E2S_AUDIO_DIR` 覆盖。
AUDIO_DIR = Path(os.environ.get("E2S_AUDIO_DIR") or (HERE / "data" / "audio"))

#: 内置曲目表：整合包自带的四首样例。
TRACKS = [
    ("blessing", "blessing.mp3"),
    ("collapse", "collapse.mp3"),
    ("yoake", "yoake.mp3"),
    ("victim", "victim.mp3"),
]


def main() -> int:
    import sys

    wanted = [a for a in sys.argv[1:] if not a.startswith("-")]
    render_score = "--no-render" not in sys.argv

    import torch
    from transformers import AutoModel

    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}", flush=True)
    print(f"音频目录 {AUDIO_DIR}", flush=True)

    started = time.time()
    model = AutoModel.from_pretrained("m-a-p/SheetSage2", trust_remote_code=True).eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    print(f"model loaded in {time.time() - started:.0f}s", flush=True)

    OUT_ROOT.mkdir(exist_ok=True)
    summary = []
    for key, filename in TRACKS:
        if wanted and key not in wanted:
            continue
        source = AUDIO_DIR / filename
        if not source.is_file():
            # 允许 `<key>.mp3` 这种简名，方便别人用自己的音频
            alt = AUDIO_DIR / f"{key}.mp3"
            source = alt if alt.is_file() else source
        if not source.is_file():
            print(f"!! missing {source}", flush=True)
            continue
        out_dir = OUT_ROOT / key
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 70}\n=== {key}: {filename}\n{'=' * 70}", flush=True)

        started = time.time()
        options = {"output_dir": str(out_dir)}
        if render_score:
            # A printable score and a piano preview straight from the model; the
            # assets needed for this are fetched on first use.
            options["render_score"] = ("pdf",)
            options["render_audio"] = True
        try:
            result = model.transcribe(str(source), **options)
        except Exception:
            print(f"!! {key} failed\n{traceback.format_exc()}", flush=True)
            summary.append({"track": key, "status": "failed"})
            continue
        elapsed = time.time() - started

        events_payload = json.loads((out_dir / "events.json").read_text(encoding="utf-8"))
        events = events_payload["events"] if isinstance(events_payload, dict) else events_payload
        keys = sorted({e["values"]["key"] for e in events if "key" in (e.get("values") or {})})
        chords = sorted({e["values"]["chord"] for e in events if "chord" in (e.get("values") or {})})
        structures = sorted(
            {e["values"]["structure"] for e in events if "structure" in (e.get("values") or {})}
        )
        files = sorted(p.name for p in out_dir.iterdir() if p.is_file())

        entry = {
            "track": key,
            "status": "ok",
            "seconds": round(elapsed, 1),
            "duration_seconds": result.get("duration_seconds"),
            "windows": len(result.get("windows") or []),
            "abc_error": result.get("abc_error"),
            "abc_chars": len(result.get("abc") or ""),
            "keys": keys,
            "distinct_chords": len(chords),
            "structures": structures,
            "n_events": len(events),
            "files": files,
        }
        summary.append(entry)
        print(
            f"--- {key}: {elapsed / 60:.1f} min, {len(events)} events, "
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
    for entry in summary:
        if entry.get("status") != "ok":
            print(f"  {entry['track']:<10} FAILED")
            continue
        print(
            f"  {entry['track']:<10} {entry['seconds'] / 60:>5.1f} min  "
            f"{entry['n_events']:>5} events  key={entry['keys']}  "
            f"{entry['distinct_chords']:>3} chords  {entry['structures']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
