"""Error types.

Every failure that a user can act on gets its own exception so the CLI can turn
it into a single readable line instead of a traceback.
"""

from __future__ import annotations


class Ensemble2ScoreError(Exception):
    """Base class for every error this package raises on purpose."""


class ConfigError(Ensemble2ScoreError):
    """Bad options, bad instrument map, mutually exclusive flags."""


class InputError(Ensemble2ScoreError):
    """The input audio is missing, unreadable or empty."""


class BackendError(Ensemble2ScoreError):
    """A processing backend is unavailable, unlicensed for use, or failed."""


class BackendUnavailable(BackendError):
    """The backend exists but its dependencies or weights are not installed."""


class OutputError(Ensemble2ScoreError):
    """A score could not be written or exported."""


__all__ = [
    "Ensemble2ScoreError",
    "ConfigError",
    "InputError",
    "BackendError",
    "BackendUnavailable",
    "OutputError",
]
