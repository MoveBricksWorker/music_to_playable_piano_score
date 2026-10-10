"""Finding Demucs weights on disk without picking the wrong model.

Why the previous approach was wrong
-----------------------------------
``find_local_model`` used to pick the first checkpoint whose declared ``sources``
matched, iterating files in name order.  Demucs hashes checkpoint filenames, so name
order is arbitrary — and this machine holds **six** four-source checkpoints:

    sources 4   f7e0c4bc, d12395a8, 92cfc3b6, 04573f0d   (htdemucs_ft, four members)
    sources 4   955717e8                                  (htdemucs)
    sources 6   5c90dfd2                                  (htdemucs_6s)

so ``--demucs-model htdemucs`` silently loaded a fine-tuned member instead.  Measured
on 30 s of a real track, that produced drum peaks of 4.148 and a stem sum that
reconstructed the mix at only r=0.930 with a −2.60 dB level error, against r=0.9998
and −0.04 dB for the model demucs actually resolves by name.  Those broken stems are
what got written to disk, and the overshoot is what clipped.

The fix is to stop trying to identify a model by a property it shares with several
others.  Demucs' own ``remote/files.txt`` groups the checkpoints by family:
``955717e8`` is *the* four-source hybrid-transformer model, and the other four-source
files are the fine-tuned members.  That grouping is reproduced here for the checkpoints
this machine actually has, and anything not listed falls back to the strict behaviour
of refusing to choose between look-alikes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

__all__ = [
    "find_local_model",
    "load_local_model",
    "LOCAL_MODEL_DIRS",
    "CACHE_HINT",
    "SIGNATURE_FILES",
]

logger = logging.getLogger(__name__)

CACHE_HINT = (
    "No local Demucs checkpoint matched. Download it once while online, e.g.\n"
    "    python -c \"from demucs.pretrained import get_model; get_model('htdemucs_6s')\"\n"
    "or point --stems-dir at stems you separated yourself"
)

_LOCAL_MODEL_DIRS: tuple[Path, ...] = (
    Path.home() / ".cache" / "torch" / "hub" / "checkpoints",
    Path.home() / ".demucs" / "models",
)
# Kept separate so the probe can be pointed at a temp directory in tests.
LOCAL_MODEL_DIRS = _LOCAL_MODEL_DIRS

# Signature (the part of the filename before the dash) -> model name, taken from
# demucs' own ``demucs/remote/files.txt``, which groups the checkpoints by family:
#
#   # Hybrid Transformer models
#   955717e8-8726e21a.th      <- htdemucs
#   f7e0c4bc-ba3fe64a.th      <- htdemucs_ft (bag members; the bag is a separate entry)
#   d12395a8-e57c48e6.th
#   92cfc3b6-ef3bcb9c.th
#   04573f0d-f3cf25b2.th
#   75fc33f5-1941ce65.th
#   # Experimental 6 sources model
#   5c90dfd2-34c22ccb.th      <- htdemucs_6s
#
# Only the signatures needed to disambiguate are listed; a model with exactly one
# candidate on disk does not need an entry and still resolves.
SIGNATURE_FILES: dict[str, tuple[str, ...]] = {
    "htdemucs": ("955717e8",),
    "htdemucs_6s": ("5c90dfd2",),
    "htdemucs_ft": ("f7e0c4bc", "d12395a8", "92cfc3b6", "04573f0d", "75fc33f5"),
    "mdx_extra": ("0d19c1c6", "5d2d6c55", "7d865c68", "7ecf8ec1", "a1d90b5c"),
}


def _candidate_files(dirs: Iterable[Path] | None = None) -> list[Path]:
    """Checkpoint files under ``dirs``, or under the default locations.

    ``dirs`` is resolved at call time rather than bound as a default argument:
    a default of ``LOCAL_MODEL_DIRS`` is evaluated once at import, so reassigning
    that constant later silently has no effect and the search keeps looking in the
    old place.  That is a nasty failure to debug, so the lookup happens here.
    """
    search = tuple(LOCAL_MODEL_DIRS if dirs is None else dirs)
    found: list[Path] = []
    for directory in search:
        if not directory.is_dir():
            continue
        for pattern in ("*.th", "*.pt", "*.ckpt"):
            found.extend(sorted(directory.glob(pattern)))
    return found


def load_local_model(path: Path):
    """Reconstruct a model from a self-contained checkpoint file."""
    from demucs.repo import load_model

    return load_model(str(path))


def signature_of(path: Path) -> str:
    """The model signature demucs encodes in a checkpoint filename.

    ``955717e8-8726e21a.th`` -> ``955717e8``.  A file without a dash carries no
    signature and returns an empty string.
    """
    stem = path.stem
    return stem.rsplit("-", 1)[0] if "-" in stem else ""


def find_local_model(
    expected_sources: Iterable[str],
    dirs: Iterable[Path] | None = None,
    *,
    name: str | None = None,
):
    """Return a model whose sources match ``expected_sources``, or ``None``.

    When ``name`` is given and its signature is known, only files carrying that
    signature are considered.  Otherwise the search falls back to source counts, but
    only accepts the result when it is **unambiguous**: if several checkpoints share
    the requested source list, none of them is chosen, because silently picking one
    is how a fine-tuned model came to stand in for ``htdemucs``.

    A checkpoint that will not load is skipped rather than fatal: a truncated
    download left behind by an interrupted run is common, and the next candidate may
    be the good one.
    """
    want = frozenset(expected_sources)
    files = _candidate_files(dirs)

    signatures = SIGNATURE_FILES.get(name or "", ())
    if signatures:
        candidates = [path for path in files if signature_of(path) in signatures]
        if not candidates:
            logger.debug("no local checkpoint carries signature %s for %r", signatures, name)
            return None
    else:
        candidates = files

    loaded: list[tuple[Path, object]] = []
    for path in candidates:
        try:
            model = load_local_model(path)
        except Exception as exc:  # noqa: BLE001 - any unpickling failure means "skip"
            logger.debug("skipping %s: %s", path.name, exc)
            continue
        got = frozenset(getattr(model, "sources", ()) or ())
        if got == want:
            loaded.append((path, model))
            if signatures:
                break  # an exact signature match needs no further search

    if not loaded:
        return None
    if len(loaded) > 1:
        logger.warning(
            "%d local checkpoints match sources %s and none carries a known signature "
            "for %r: refusing to guess which one was asked for (%s). Pass an explicit "
            "checkpoint or download the model by name",
            len(loaded), sorted(want), name, ", ".join(p.name for p, _ in loaded),
        )
        return None

    path, model = loaded[0]
    logger.debug("using local checkpoint %s for sources %s", path.name, sorted(want))
    return model


def find_local_model_for(name: str, available_sources: dict[str, tuple[str, ...]]):
    """Look up the source list for ``name`` and find a matching local checkpoint."""
    sources = available_sources.get(name)
    if sources is None:
        return None
    return find_local_model(sources, name=name)
