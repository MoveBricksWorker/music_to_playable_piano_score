"""Stage-level caching.

Design notes
------------
* The cache key includes :data:`ensemble2score.core.models.CACHE_SCHEMA`, the
  package version, and the *content hash* of the input file — not its path.  Move
  a file and the cache still hits; edit it and the cache misses.  The old
  implementation keyed on the path, so a re-encoded file silently reused notes
  transcribed from different audio.
* Values are stored as JSON so a human can open ``out/cache/*.json`` and see
  exactly what a stage produced.
* A corrupt or unreadable entry is treated as a miss, never as an error.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..core.models import CACHE_SCHEMA

__all__ = ["Cache", "content_hash"]

_CHUNK = 1 << 20


def content_hash(path: str | Path, *, size_limit: int | None = None) -> str:
    """Stable hash of a file's contents (and its length).

    ``size_limit`` hashes only the first N bytes, for the rare case of hashing a
    very large file cheaply.  Prefer the default: a partial hash can collide
    between two files that differ only in their tail.
    """
    digest = hashlib.blake2b(digest_size=16)
    target = Path(path)
    remaining = size_limit if size_limit and size_limit > 0 else None
    with target.open("rb") as handle:
        while True:
            want = _CHUNK if remaining is None else min(_CHUNK, remaining)
            if want <= 0:
                break
            chunk = handle.read(want)
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    digest.update(str(target.stat().st_size).encode())
    return digest.hexdigest()


class Cache:
    """JSON-file cache for one output directory."""

    def __init__(self, directory: str | Path, *, enabled: bool = True, namespace: str = "") -> None:
        self.dir = Path(directory)
        self.enabled = enabled
        self.namespace = namespace

    # ------------------------------------------------------------------ #
    def key(self, stage: str, *parts: object) -> str:
        """Deterministic key for a stage and its parameters."""
        from .. import __version__

        payload = [CACHE_SCHEMA, __version__, self.namespace, stage, *[str(p) for p in parts]]
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.blake2b(blob, digest_size=8).hexdigest()

    def path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def has(self, key: str) -> bool:
        if not self.enabled:
            return False
        path = self.path(key)
        if not path.is_file() or path.stat().st_size == 0:
            return False
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False  # unreadable or truncated: treat as a miss
        return True

    def read(self, key: str) -> Any:
        return json.loads(self.path(key).read_text(encoding="utf-8"))

    def write(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        target = self.path(key)
        tmp = target.with_suffix(".json.tmp")
        # Write-then-rename: a killed process can leave a .tmp behind but never a
        # half-written entry that the next run would load as truth.
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        tmp.replace(target)

    def clear(self) -> int:
        """Delete every entry; returns how many were removed."""
        if not self.dir.is_dir():
            return 0
        removed = 0
        for entry in self.dir.glob("*.json"):
            entry.unlink(missing_ok=True)
            removed += 1
        return removed
