"""I/O and cross-cutting services: files, cache, console, licence registry.

None of these modules import a heavy backend.
"""

from __future__ import annotations

from .cache import Cache, content_hash
from .console import ensure_dir, hint, info, rule, set_verbose, slugify, warn
from .instruments import DEFAULT_MAP, load_overrides, order_parts, resolve
from .registry import REGISTRY, Component, audit_lines, by_kind

__all__ = [
    "Cache",
    "Component",
    "DEFAULT_MAP",
    "REGISTRY",
    "audit_lines",
    "by_kind",
    "content_hash",
    "ensure_dir",
    "hint",
    "info",
    "load_overrides",
    "order_parts",
    "resolve",
    "rule",
    "set_verbose",
    "slugify",
    "warn",
]
