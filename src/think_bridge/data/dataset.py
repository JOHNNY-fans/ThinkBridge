"""JSON and JSONL loading for Stage 0 and evaluation records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_data_file(path: str | Path) -> list[dict[str, Any]]:
    """Load an object array or object-per-line stream without rewriting rows."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"data file does not exist: {source}")
    text = source.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, dict):
        # Only this unambiguous envelope is a collection. Other dictionaries
        # remain individual records; never discard sibling metadata silently.
        if set(payload) == {"data"} and isinstance(payload["data"], list):
            payload = payload["data"]
        else:
            payload = [payload]
    if not isinstance(payload, list) or any(
        not isinstance(row, dict) for row in payload
    ):
        raise ValueError(
            f"data must be a JSON object array or JSONL object stream: {source}"
        )
    return payload
