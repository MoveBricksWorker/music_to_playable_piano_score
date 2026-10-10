"""Small shared utilities: JSON output and a stage timer.

Deliberately tiny.  Anything bigger belongs in a named module.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["jsonable", "save_json", "load_json", "Timer"]


def jsonable(value: Any) -> Any:
    """Recursively convert a value into plain JSON types.

    NaN and infinity become ``null``: they are not valid JSON and a file that
    cannot be parsed is worse than a missing number.
    """
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return jsonable(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if is_dataclass(value) and not isinstance(value, type):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        return jsonable(value.to_dict())
    return value


def save_json(path: str | Path, payload: Any) -> Path:
    """Write ``payload`` as pretty JSON, atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(jsonable(payload), ensure_ascii=False, indent=2, sort_keys=False)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(target)
    return target


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Timer:
    """Context manager appending ``(name, seconds)`` to a list."""

    def __init__(self, sink: list) -> None:
        self.sink = sink
        self.name = ""
        self.elapsed = 0.0

    def __call__(self, name: str) -> "Timer":
        self.name = name
        return self

    def __enter__(self) -> "Timer":
        import time

        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info) -> None:
        import time

        self.elapsed = time.perf_counter() - self._start
        self.sink.append((self.name, self.elapsed))
