"""Append-only audit records for explicit ThinkBridge recovery attempts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Literal

from think_bridge.model.artifact_schema import RESUME_AUDIT_EVENT, artifact_header
from think_bridge.model.contract import require_sha256


def append_checkpoint_resume_event(
    run_dir: Path,
    checkpoint: Path,
    *,
    checkpoint_sha256: str,
    status: Literal["preflight_validated", "accepted"],
) -> None:
    run = Path(run_dir).resolve(strict=True)
    target = Path(checkpoint).resolve(strict=True)
    try:
        target.relative_to(run)
    except ValueError as exc:
        raise ValueError("resume audit checkpoint escaped its run") from exc
    _append(
        run,
        {
            **artifact_header(RESUME_AUDIT_EVENT),
            "status": status,
            "target_kind": "checkpoint",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_path": str(target),
            "checkpoint_sha256": require_sha256(
                checkpoint_sha256, "resume checkpoint_sha256"
            ),
        },
    )


def _append(run_dir: Path, event: dict[str, object]) -> None:
    path = Path(run_dir) / "resume_events.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True, ensure_ascii=False))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
