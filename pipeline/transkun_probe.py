"""Run Transkun (polyphonic piano transcription) on a stem, bypassing its CLI.

Why this exists
---------------
`pip install transkun` pulls in training-only dependencies (``ncls`` interval
trees, ``torch_optimizer``, ``moduleconf``) that the *inference* path does not
need but whose installation is slow and fragile here.  This script stubs the
one that is imported at module scope (``moduleconf``), loads ``pretrained/2.0.pt``
-- which ships inside the wheel -- from the config in ``pretrained/2.0.conf``,
and writes a MIDI file.

The measurement it enables
--------------------------
The project's current transcriber (SheetSage2) is a lead-sheet model whose
melody head is *one categorical distribution per sixteenth note*, so it can
emit at most one note at a time.  Measured on this repository's own outputs:
mean simultaneous notes per onset = 1.00 for every stem, including piano.  A
polyphonic piano transcriber should move that number well above 1 on the same
audio.  If it does, the project's blocking defect is the transcriber, not the
separator; if it does not, the separator is the culprit after all.

Usage:
    python transkun_probe.py <audio> <out.mid> [--device cuda] [--max-seconds N]
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
import types
from pathlib import Path


def install_moduleconf_stub(conf_path: Path):
    """`moduleconf` is imported at module scope but only used to read a JSON conf."""
    try:
        import moduleconf  # noqa: F401
        return False
    except ImportError:
        pass

    m = types.ModuleType("moduleconf")

    def parseFromFile(path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        out = {}
        for section, body in data.items():
            module_name = body["module"]
            config = body.get("config", {})
            mod = importlib.import_module(module_name)
            cls = getattr(mod, body.get("configClassName", "Config"))
            cfg = cls()
            for k, v in config.items():
                setattr(cfg, k, v)
            out[section] = types.SimpleNamespace(module=mod, config=cfg)
        return out

    m.parseFromFile = parseFromFile
    sys.modules["moduleconf"] = m
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("out")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args()

    import transkun

    pkg = Path(transkun.__file__).resolve().parent
    conf_path = pkg / "pretrained" / "2.0.conf"
    weight_path = pkg / "pretrained" / "2.0.pt"
    if not weight_path.exists():
        print(f"[error] checkpoint missing: {weight_path}")
        return 2

    stubbed = install_moduleconf_stub(conf_path)
    print(f"[setup] moduleconf stubbed={stubbed}; checkpoint {weight_path.name} "
          f"({weight_path.stat().st_size/1e6:.1f} MB)")

    import numpy as np
    import torch
    import torchaudio

    from transkun.Data import writeMidi
    import moduleconf

    confManager = moduleconf.parseFromFile(str(conf_path))
    TransKun = confManager["Model"].module.TransKun
    conf = confManager["Model"].config

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"[setup] torch {torch.__version__} device={device} "
          f"fs={conf.fs} seg={conf.segmentSizeInSecond}s hop={conf.segmentHopSizeInSecond}s")

    t0 = time.time()
    checkpoint = torch.load(str(weight_path), map_location=device)
    model = TransKun(conf=conf).to(device)
    state = checkpoint.get("best_state_dict") or checkpoint.get("state_dict")
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[setup] model loaded in {time.time()-t0:.1f}s")

    # Read audio.  WAV via torchaudio avoids needing ffmpeg; other formats fall
    # back to pydub (which does need ffmpeg).
    audio_path = Path(args.audio)
    try:
        wav, fs = torchaudio.load(str(audio_path))
        audio = wav.mean(dim=0).numpy().astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        print(f"[audio] torchaudio failed ({exc}); trying pydub")
        import pydub
        seg = pydub.AudioSegment.from_file(str(audio_path))
        arr = np.array(seg.get_array_of_samples()).reshape(-1, seg.channels)
        audio = (arr.astype(np.float32) / 2 ** 15).mean(axis=1)
        fs = seg.frame_rate

    if args.max_seconds:
        audio = audio[: int(args.max_seconds * fs)]
    print(f"[audio] {len(audio)/fs:.1f}s @ {fs} Hz")

    if fs != conf.fs:
        import soxr
        audio = soxr.resample(audio, fs, conf.fs).astype(np.float32)
        fs = conf.fs
        print(f"[audio] resampled to {fs} Hz")

    x = torch.from_numpy(audio).to(device)
    if x.dim() == 1:
        x = x[:, None]          # model.transcribe expects (frames, channels)
    torch.set_grad_enabled(False)

    t0 = time.time()
    notesEst = model.transcribe(x, stepInSecond=None, segmentSizeInSecond=None,
                                discardSecondHalf=False)
    print(f"[infer] {time.time()-t0:.1f}s")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writeMidi(notesEst).write(str(out))

    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(out))
    notes = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    print(f"[result] {len(notes)} notes -> {out}")
    if notes:
        onsets: dict[int, int] = {}
        for n in notes:
            k = int(round(n.start * 100))
            onsets[k] = onsets.get(k, 0) + 1
        stack = sum(onsets.values()) / len(onsets)
        share1 = sum(1 for v in onsets.values() if v == 1) / len(onsets)
        print(f"[result] mean simultaneous notes per onset = {stack:.2f}")
        print(f"[result] max simultaneous notes = {max(onsets.values())}")
        print(f"[result] share of onsets that are single notes = {share1*100:.1f}%")
        lo = min(n.pitch for n in notes)
        hi = max(n.pitch for n in notes)
        print(f"[result] pitch range = {lo}-{hi}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
