"""Small helpers for executor caches and embedding-scale measurements."""

from __future__ import annotations

from typing import Any


def new_cache() -> Any:
    """Create an empty Transformers dynamic KV cache."""
    from transformers.cache_utils import DynamicCache

    return DynamicCache()
