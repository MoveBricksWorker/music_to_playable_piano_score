"""用替代后端生成 SheetSage2 形状的标签，让**新曲子**也能走完第 ④ 阶段。

## 为什么需要它

`data/sheetsage2/` 只预置了四首样例的标签，而 SheetSage2 在本包的解释器上**加载不了**
（它要 transformers 4.45，包内是 5.6.1，见 README 已知缺陷）。于是任何新曲子都会
卡在第 ④ 阶段。这个脚本把那一步换成包里**已经装好、且能离线跑**的组件：

| 管线要的标签 | SheetSage2 提供 | 本脚本的替代 |
|---|---|---|
| `beat.lab` | 拍点模型 | `beat_this`（`ensemble2score` 的神经网络后端） |
| `downbeat.lab` | 强拍 | 同上，由 `BeatTrack.downbeat_times` |
| `key.lab` | 调号头 | chroma + Krumhansl 模板（纯 librosa） |
| `melody_vocal.mid` | 人声旋律头 | `basic_pitch` 对 **Demucs 的 vocals 分轨**做转谱 |
| `chord.lab` / `chords.mid` | 和弦头 | 占位（管线只做存在性检查） |

## 关键设计：旋律必须来自**独立声源**

管线里有一条硬规则——"与旋律同刻同音高的钢琴音当人声串音删掉"。
如果拿**转谱自己的最高音**冒充旋律，那条规则会把大部分钢琴音删掉
（实测某首电子乐 912 个音删掉 645 个），而且被保护的音会让可弹性修正失效。

所以这里坚持用 **Demucs 分离出来的 vocals 分轨**，再用 basic-pitch 转成单音旋律。
这样旋律与钢琴轨在音高上不重合，`dedupe_piano` 只会删掉真正的人声串音。

**没有人声的曲子**（纯器乐）不该走"造一条假旋律"这条路——那会把错误保护进管线。
这种情况脚本默认**拒绝**，并提示用 `--no-melody` 显式选择"不做旋律保护"
（编排层就不锁右手最高音，可弹性修正能正常工作）。

用法：

    python -m pipeline.make_labels 妖怪之山
    python -m pipeline.make_labels <曲名> --no-melody        # 纯器乐
    python -m pipeline.make_labels <曲名> --vocals <path>    # 指定人声分轨
    python -m pipeline.make_labels <曲名> --device cpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: 调式模板（Krumhansl-Schmuckler），用来给 chroma 打分
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_PITCH_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")


def _log(message: str) -> None:
    print(f"    {message}", flush=True)


# --------------------------------------------------------------------------- #
def find_audio(song: str, work: Path) -> Path:
    for suffix in (".mp3", ".wav", ".flac", ".m4a"):
        candidate = work / "audio" / f"{song}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"找不到音频：{work / 'audio' / (song + '.mp3')}")


def find_vocals(song: str, work: Path, explicit: str | None) -> Path | None:
    """找 Demucs 的 vocals 分轨。

    先看 6 轨分离的 `stems6/<曲>/vocals.wav`（需要 `--from separate` 时才存在，
    因为管线的 `stage_separate` 只搬运了 piano 与 other 两轨），
    再看调用方显式给的路径。
    """
    if explicit:
        path = Path(explicit)
        return path if path.is_file() else None
    for relative in (
        work / "stems6" / song / "vocals.wav",
        work / "_demucs" / "htdemucs_6s" / song / "vocals.wav",
        work / "_demucs" / "htdemucs" / song / "vocals.wav",
        work / "stems4" / song / "vocals.wav",
    ):
        if relative.is_file():
            return relative
    return None


def separate_vocals(audio: Path, work: Path, song: str, device: str) -> Path | None:
    """没有 vocals 分轨时，用 Demucs 补跑一次 4 轨分离取人声。

    `htdemucs`（4 轨）的 vocals 头比 6 轨版本可靠，而且这一步与管线的
    `stage_separate` 不冲突（写到 `work/_demucs/` 下）。
    """
    import subprocess

    staging = work / "_demucs"
    target = staging / "htdemucs" / song / "vocals.wav"
    if target.is_file():
        return target
    _log("没有 vocals 分轨，跑一次 Demucs 4 轨取人声 …")
    staging.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, "-m", "demucs", "-n", "htdemucs", "-d", device,
         "-o", str(staging), str(audio)],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not target.is_file():
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-5:]
        _log(f"Demucs 失败（exit={result.returncode}）：" + " | ".join(tail))
        return None
    return target


def _spellable_key(tonic_index: int, mode: str) -> str:
    """挑一个**能记谱**的调名。

    坑：`Gb` 小调在乐理上确实有 **9 个降号**（`music21` 会照实返回 -9），
    但五线谱的调号最多 7 个升降号，9 个降号要求重降号，谱面直接没法看。
    正确的做法是改用等音异名的 `F#` 小调（3 个升号）。

    判据：五度数落在 ±7 以内才可记谱，否则换成 `fifths ± 12` 更小的那个等音。
    """
    from music21 import key as m21key

    name = _PITCH_NAMES[tonic_index % 12]
    for candidate in (name, _ENHARMONIC.get(name, name)):
        try:
            fifths = int(m21key.Key(candidate, mode).sharps)
        except Exception:  # noqa: BLE001
            continue
        if abs(fifths) <= 7:
            return f"{candidate}:{mode}"
        # 超出可记谱范围：试等音异名
        other = _ENHARMONIC.get(candidate)
        if other:
            try:
                if abs(int(m21key.Key(other, mode).sharps)) <= 7:
                    return f"{other}:{mode}"
            except Exception:  # noqa: BLE001
                pass
    return f"{name}:{mode}"


#: 等音异名表：给"调号超过 7 个"的调换一个能记谱的写法。
_ENHARMONIC = {
    "Cb": "B", "B": "Cb", "Gb": "F#", "F#": "Gb",
    "Db": "C#", "C#": "Db", "Ab": "G#", "G#": "Ab",
    "Eb": "D#", "D#": "Eb", "Bb": "A#", "A#": "Bb",
}


def estimate_key(audio: Path) -> tuple[str, float]:
    """chroma + Krumhansl 模板 → "F#:minor" 形式的调名（保证可记谱）。

    返回 (调名, 与次优的得分差)。差值小说明调性感弱，调用方应当提醒。
    """
    import librosa

    y, sr = librosa.load(str(audio), sr=22050, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    profile = chroma.mean(axis=1)
    total = profile.sum() or 1.0
    profile = profile / total
    scored: list[tuple[float, int, str]] = []
    for shift in range(12):
        rotated = np.roll(profile, -shift)
        scored.append((float(np.dot(rotated, _MAJOR)), shift, "major"))
        scored.append((float(np.dot(rotated, _MINOR)), shift, "minor"))
    scored.sort(reverse=True)
    best_score, best_shift, best_mode = scored[0]
    runner_up = scored[1][0]
    # 挑可记谱的调名（等音异名的坑见 _spellable_key）
    return _spellable_key(best_shift, best_mode), best_score - runner_up


def track_beats(audio: Path, device: str) -> tuple[list[float], list[float], float, str]:
    """用 `beat_this` 取拍点与强拍。失败则回退 librosa。"""
    from ensemble2score.analysis.beats import build_beat_tracker

    for name in ("beat-this", "librosa"):
        try:
            tracker = build_beat_tracker(name, device=device)
            result = tracker.track(str(audio))
            beats = [float(t) for t in result.beat_times]
            downs = [float(t) for t in result.downbeat_times]
            if len(beats) >= 4:
                return beats, downs, float(result.tempo_bpm), result.backend
        except Exception as exc:  # noqa: BLE001
            _log(f"{name} 拍点失败（{type(exc).__name__}: {exc}）")
    raise RuntimeError("拍点检测失败：beat-this 与 librosa 都没跑通")


def transcribe_melody(vocals: Path, device: str) -> list[tuple[int, float, float, int]]:
    """对 vocals 分轨做转谱，返回 [(音高, 起, 止, 力度)]，只留最高的那条线。"""
    from ensemble2score.analysis.transcribe import build_transcriber
    from ensemble2score.core.models import PartSpec

    spec = PartSpec(
        stem="vocals", part_name="Vocal", instrument="voice",
        midi_program=0, clef="treble", grand_staff=False,
        monophonic=True, sounding_pitch=True, drum_channel=False,
    )
    for name in ("basic-pitch", "pyin"):
        try:
            transcriber = build_transcriber(name, device=device)
            notes, warnings = transcriber.transcribe(str(vocals), spec)
            for warning in warnings[:3]:
                _log(f"转谱提示：{warning}")
            break
        except Exception as exc:  # noqa: BLE001
            _log(f"{name} 转谱失败（{type(exc).__name__}: {exc}）")
    else:
        raise RuntimeError("人声转谱失败：basic-pitch 与 pyin 都没跑通")

    # 压成单声部：重叠处只留最高的音（旋律线本来就该是一条线）
    ordered = sorted(notes, key=lambda n: (n.start, -n.pitch))
    monophonic: list[tuple[int, float, float, int]] = []
    for note in ordered:
        if monophonic and note.start < monophonic[-1][2] - 1e-6:
            # 与前一个音重叠：保留更高的那个，并裁掉前者的尾巴
            previous = monophonic[-1]
            if note.pitch > previous[0]:
                monophonic[-1] = (previous[0], previous[1], note.start, previous[3])
                monophonic.append((note.pitch, note.start, note.end, note.velocity))
            continue
        monophonic.append((note.pitch, note.start, note.end, note.velocity))
    return [item for item in monophonic if item[2] > item[1]]


def write_melody_midi(notes: list[tuple[int, float, float, int]], path: Path) -> None:
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0, name="Vocal")
    for pitch, start, end, velocity in notes:
        instrument.notes.append(
            pretty_midi.Note(
                velocity=int(velocity) or 100,
                pitch=int(pitch),
                start=float(start),
                end=max(float(end), float(start) + 0.05),
            )
        )
    midi.instruments.append(instrument)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))


def write_empty_midi(path: Path, name: str) -> None:
    midi = pretty_midi.PrettyMIDI()
    midi.instruments.append(pretty_midi.Instrument(program=0, name=name))
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("song")
    parser.add_argument("--work", default=None)
    parser.add_argument("--out", default=None, help="标签输出目录（默认 work/sheetsage2/<曲名>）")
    parser.add_argument("--device", default="cuda", help="beat_this / basic-pitch 用的设备")
    parser.add_argument("--vocals", default=None, help="显式指定人声分轨 wav")
    parser.add_argument("--no-separate", action="store_true",
                        help="没有 vocals 分轨时不要自动跑 Demucs")
    parser.add_argument("--no-melody", action="store_true",
                        help="纯器乐：不生成 melody_vocal.mid（编排层就不锁右手最高音）")
    parser.add_argument("--force", action="store_true", help="覆盖已有标签")
    args = parser.parse_args()

    work = Path(args.work).resolve() if args.work else ROOT / "work"
    labels = Path(args.out).resolve() if args.out else work / "sheetsage2" / args.song
    audio = find_audio(args.song, work)

    if labels.is_dir() and not args.force:
        existing = sorted(p.name for p in labels.iterdir() if p.is_file())
        if existing:
            _log(f"{labels} 里已有 {len(existing)} 个文件，加 --force 覆盖")
            return 1

    labels.mkdir(parents=True, exist_ok=True)
    print(f"曲目 {args.song}")
    _log(f"音频   {audio.relative_to(ROOT)}")
    _log(f"输出   {labels.relative_to(ROOT)}")

    # ---- ① 拍点 / 强拍 ---------------------------------------------------- #
    beats, downs, tempo, backend = track_beats(audio, args.device)
    gaps = np.diff(beats)
    _log(f"拍点   {len(beats)} 个，tempo≈{tempo:.1f} BPM，"
         f"间隔中位 {np.median(gaps):.3f}s（σ={gaps.std():.4f}，后端 {backend}）")
    if len(downs) < 3:
        _log(f"强拍只有 {len(downs)} 个，按每 4 拍补一份")
        downs = beats[::4]

    # ---- ② 调号 ----------------------------------------------------------- #
    key, margin = estimate_key(audio)
    _log(f"调号   {key}（与次优得分差 {margin:.4f}"
         f"{'，调性感弱，仅供参考' if margin < 0.02 else ''}）")

    # ---- ③ 旋律 ----------------------------------------------------------- #
    melody: list[tuple[int, float, float, int]] = []
    if args.no_melody:
        _log("旋律   按你的要求跳过（纯器乐模式，不做旋律保护）")
    else:
        vocals = find_vocals(args.song, work, args.vocals)
        if vocals is None and not args.no_separate:
            vocals = separate_vocals(audio, work, args.song, args.device)
        if vocals is None:
            _log("旋律   找不到 vocals 分轨，跳过（等价于 --no-melody）")
        else:
            melody = transcribe_melody(vocals, args.device)
            pitches = [n[0] for n in melody]
            _log(f"旋律   {len(melody)} 个音，范围 {min(pitches)}-{max(pitches)}"
                 f"（来源 {vocals.name}）")

    # ---- ④ 落盘 ----------------------------------------------------------- #
    (labels / "beat.lab").write_text(
        "\n".join(f"{t:.4f}" for t in beats) + "\n", encoding="utf-8")
    (labels / "downbeat.lab").write_text(
        "\n".join(f"{t:.4f}" for t in downs) + "\n", encoding="utf-8")
    duration = float(sf.info(str(audio)).duration)
    (labels / "key.lab").write_text(
        f"{0.0:.4f}\t{duration:.4f}\t{key}\n", encoding="utf-8")
    (labels / "chord.lab").write_text("", encoding="utf-8")
    write_empty_midi(labels / "chords.mid", "Chords")
    if melody:
        write_melody_midi(melody, labels / "melody_vocal.mid")
    else:
        # 管线要这个文件存在；写成空 MIDI 等价于"没有旋律保护"。
        write_empty_midi(labels / "melody_vocal.mid", "Vocal")

    print()
    _log("产出：")
    for name in ("beat.lab", "downbeat.lab", "key.lab", "chord.lab",
                 "chords.mid", "melody_vocal.mid"):
        path = labels / name
        mark = f"{path.stat().st_size:>8d} B" if path.is_file() else "   缺失"
        _log(f"  {name:18s} {mark}")
    print()
    _log(f"下一步：run.bat {args.song} --from arrange")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
