"""Configuration loading for the maintained ThinkBridge method."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from think_bridge.stage1.configuration import BridgeConfig as TrainingConfig


@dataclass(frozen=True)
class TrainingConfigDocument:
    """Small metadata view used by preflight and CLI diagnostics."""

    config_kind: str
    objective_version: str
    model_family: str
    method: str
    model_name_or_path: str
    tokenizer_name_or_path: str


def _read_mapping(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration root must be an object: {source}")
    return payload


def load_training_config(
    path: str | Path,
    *,
    arm: str | None = None,
    model_name_or_path: str | None = None,
    tokenizer_name_or_path: str | None = None,
    runtime_overrides: Mapping[str, Any] | None = None,
    stage_kind: str | None = None,
) -> TrainingConfig:
    """Load one model-family template plus ordinary runtime overrides."""

    if arm not in {None, "bridge"}:
        raise ValueError("the maintained method does not expose legacy arms")
    payload = _read_mapping(path)
    if payload.get("deepspeed_config"):
        configured = Path(payload["deepspeed_config"])
        if not configured.is_absolute():
            payload["deepspeed_config"] = str(
                Path(path).resolve().parent / configured.name
            )
    return TrainingConfig.from_mapping(
        payload,
        model_name_or_path=model_name_or_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        runtime_overrides=runtime_overrides,
        stage_kind=stage_kind,
    )
