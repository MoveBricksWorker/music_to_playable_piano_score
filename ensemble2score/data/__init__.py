"""Local model assets: checkpoints resolved from disk instead of the network.

Everything here has to work with no connectivity, because the machines this runs on
are behind restricted networks and a model that can only be fetched is a model that
is not available.
"""

from __future__ import annotations

from .local_models import CACHE_HINT, LOCAL_MODEL_DIRS, find_local_model, load_local_model

__all__ = ["CACHE_HINT", "LOCAL_MODEL_DIRS", "find_local_model", "load_local_model"]
