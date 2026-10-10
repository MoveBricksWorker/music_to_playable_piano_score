"""Build the summed accompaniment stem for one track: 4-stem `other` + 6-stem `piano`.

Why this audio and not a model choice
-------------------------------------
`probe_piano_prominence.py` measured that on `victim` the 6-stem `piano` stem loses
6 dB of prominence in the chorus (-20.4 dBFS in the verse, -26.7 in the chorus) while
the 4-stem `other` stem gains 25.8 dB in the same place.  That is htdemucs_6s routing
the chorus accompaniment away from its own piano head -- the failure Demucs'
documentation warns about for exactly the piano/guitar pair.

`compare_by_section.py` then showed the consequence per section, which a whole-song
average hides:

  chorus 1  6-stem piano 88 onsets / 88.2% chord   vs  4-stem other 155 / 76.9%
  chorus 1  summed      147 onsets / 81.3% chord

so the sum keeps the 4-stem note count while scoring better on chord tones than the
4-stem alone, and does not lose anything measurable in the quieter sections.

Both stems are already on disk, so this is a gain change and an addition -- no model
is re-run and no existing file is touched.  The sum is peak-normalised only if it
would otherwise clip.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("track")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    four = REPO / "deliverables_4stem" / args.track / "stems" / "other.wav"
    six = REPO / "deliverables" / args.track / "stems" / "piano.wav"
    for p in (four, six):
        if not p.is_file():
            raise SystemExit(f"missing {p}")

    a, sr_a = sf.read(str(four), dtype="float32", always_2d=True)
    b, sr_b = sf.read(str(six), dtype="float32", always_2d=True)
    if sr_a != sr_b:
        raise SystemExit(f"sample rates differ: {sr_a} vs {sr_b}")

    n = min(len(a), len(b))
    y = a[:n].astype(np.float64) + b[:n].astype(np.float64)

    peak = float(np.abs(y).max())
    scaled = 1.0
    if peak > 0.999:
        scaled = 0.999 / peak
        y = y * scaled

    out = Path(args.out) if args.out else (
        REPO / "work" / "_research_2026" / f"{args.track}_sum4other_piano6.wav"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), y.astype(np.float32), sr_a, subtype="PCM_16")

    print(f"{args.track}: 4stem-other + 6stem-piano -> {out}")
    print(f"  {n/sr_a:.1f}s @ {sr_a} Hz, raw peak {peak:.3f}, "
          f"applied gain {scaled:.4f} ({20*np.log10(scaled) if scaled < 1 else 0:+.2f} dB)")
    print(f"  run transkun on it:  .venv-probe\\Scripts\\python.exe "
          f"work\\_research_2026\\transkun_probe.py \"{out}\" "
          f"work\\_research_2026\\{args.track}_sum_transkun.mid --device cuda")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
