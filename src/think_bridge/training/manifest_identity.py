"""Path-based Stage1 inputs and model/tokenizer compatibility checks."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from think_bridge.model.artifact_schema import (
    STAGE1_TARGET_INDEX,
    require_artifact_header,
)
from think_bridge.model.contract import (
    ALIGNMENT_CAPACITY,
    ANSWER_CAPACITY,
    COT_CONTENT_CAPACITY,
    GEOMETRY_SCHEMA_VERSION,
    OBJECTIVE_VERSION,
    require_manifest_fields,
    validate_boundary_token_identity,
)

_TARGET_INDEX_FIELDS = {
    "artifact_type",
    "schema_version",
    "model_family",
    "artifacts",
    "statistics",
}
_TARGET_INDEX_OPTIONAL_FIELDS = {"compile_contract", "identity", "identity_sha256"}


class TargetIndexIncompleteError(ValueError):
    """A target publication lacks a referenced file or required field."""


class TargetIndexConflictError(ValueError):
    """A current target publication violates its path/schema contract."""


def resolve_target_index_artifact_path(
    raw_path: str | Path,
    *,
    index_path: Path | None,
    **_unused: Any,
) -> Path:
    """Resolve an artifact beside its index without content identity."""

    text = str(raw_path).strip()
    raw = Path(text).expanduser()
    if not text or raw.is_absolute() or raw == Path("."):
        raise ValueError("target artifact locator must be a relative path")
    if ".." in raw.parts:
        raise ValueError("target artifact locator contains lexical traversal")
    if index_path is None:
        return raw
    root = Path(index_path).expanduser().resolve(strict=False).parent
    candidate = (root / raw).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("target artifact escaped its index directory") from exc
    return candidate


def _count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TargetIndexConflictError(
                    f"target artifact has invalid JSON at {path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise TargetIndexConflictError(
                    f"target artifact row must be an object: {path}:{line_number}"
                )
            count += 1
    return count


def validate_target_index_payload(
    index: Mapping[str, Any],
    *,
    index_path: Path | None,
    model_family: str,
    allow_stale: bool = False,
    **_unused: Any,
) -> dict[str, Any] | None:
    """Validate path, JSONL shape, and counts; never compare data digests."""

    if not isinstance(index, Mapping):
        raise TargetIndexConflictError("target index must be a mapping")

    # actual paths, format and row counts instead of rejecting their presence.
    try:
        require_artifact_header(index, STAGE1_TARGET_INDEX, label="target index")
    except ValueError as exc:
        if allow_stale:
            return None
        raise TargetIndexConflictError(str(exc)) from exc
    missing = sorted(_TARGET_INDEX_FIELDS.difference(index))
    unknown = sorted(
        set(index).difference(_TARGET_INDEX_FIELDS | _TARGET_INDEX_OPTIONAL_FIELDS)
    )
    if missing:
        raise TargetIndexIncompleteError(
            f"target index is incomplete; missing={missing}"
        )
    if unknown:
        raise TargetIndexConflictError(f"target index has unknown fields: {unknown}")
    if "compile_contract" in index and not isinstance(
        index["compile_contract"], Mapping
    ):
        raise TargetIndexConflictError(
            "target index compile contract must be a mapping"
        )
    if index.get("model_family") != model_family:
        raise TargetIndexConflictError("target index model family mismatch")
    artifacts = index.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {"train", "validation"}:
        raise TargetIndexConflictError("target index split schema mismatch")
    normalized: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation"):
        raw = artifacts[split]
        if not isinstance(raw, Mapping) or set(raw) != {"path", "count"}:
            raise TargetIndexConflictError(
                f"target index {split} artifact schema mismatch"
            )
        count = raw.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise TargetIndexConflictError(f"target index {split} count is invalid")
        path = resolve_target_index_artifact_path(
            str(raw.get("path", "")), index_path=index_path
        )
        if index_path is not None:
            if not path.is_file():
                raise TargetIndexIncompleteError(
                    f"target index {split} artifact is missing: {path}"
                )
            observed = _count_jsonl_rows(path)
            if observed != count:
                raise TargetIndexConflictError(
                    f"target index {split} row count differs: expected={count} observed={observed}"
                )
        normalized[split] = {"path": str(path), "count": count}
    statistics = index.get("statistics")
    if (
        not isinstance(statistics, Mapping)
        or "train_horizon_coverage" not in statistics
    ):
        raise TargetIndexIncompleteError("target index statistics are incomplete")
    coverage = statistics["train_horizon_coverage"]
    if (
        isinstance(coverage, bool)
        or not isinstance(coverage, (int, float))
        or not math.isfinite(float(coverage))
        or not 0.0 < float(coverage) <= 1.0
    ):
        raise TargetIndexConflictError("target index horizon coverage is invalid")
    return {
        "artifacts": normalized,
        "statistics": dict(statistics),
        "compile_contract": (
            None if "compile_contract" not in index else dict(index["compile_contract"])
        ),
    }


def load_target_index_binding(
    target_index: Path,
    shared_manifest: Mapping[str, Any],
    *,
    model_family: str,
    route2_content_capacity: int,
    runtime_identity: Mapping[str, Any] | None = None,
    **_unused: Any,
) -> dict[str, Any]:
    """Bind path-based targets to the current model/tokenizer runtime."""

    shared = require_manifest_fields(shared_manifest)
    if shared["model_family"] != model_family:
        raise ValueError("shared manifest model family mismatch")
    if (
        isinstance(route2_content_capacity, bool)
        or not isinstance(route2_content_capacity, int)
        or route2_content_capacity <= 0
    ):
        raise ValueError("configured Route2 content capacity is invalid")
    if shared["route2_content_capacity"] != route2_content_capacity:
        raise ValueError("shared manifest Route2 content capacity mismatch")
    target = validate_target_index_payload(
        json.loads(Path(target_index).read_text(encoding="utf-8")),
        index_path=Path(target_index),
        model_family=model_family,
    )
    assert target is not None
    compile_contract = target["compile_contract"]
    if not isinstance(compile_contract, Mapping):
        raise ValueError("target index lacks the current compile contract")
    capacities = compile_contract.get("capacities")
    if (
        not isinstance(capacities, Mapping)
        or capacities.get("route2_content") != route2_content_capacity
    ):
        raise ValueError("target index Route2 content capacity mismatch")
    bound = {
        **shared,
        "objective_version": OBJECTIVE_VERSION,
        "geometry_schema_version": GEOMETRY_SCHEMA_VERSION,
        "alignment_capacity": ALIGNMENT_CAPACITY,
        "cot_content_capacity": COT_CONTENT_CAPACITY,
        "answer_capacity": ANSWER_CAPACITY,
        "route2_content_capacity": route2_content_capacity,
        "target_index_statistics": target["statistics"],
        "target_index_artifacts": target["artifacts"],
    }
    if runtime_identity is None:
        return bound
    if not isinstance(runtime_identity, Mapping):
        raise ValueError("current runtime sidecar identity must be a mapping")
    for field in (
        "tokenizer_sha256",
        "template_sha256",
        "boundary_ids_sha256",
    ):
        if runtime_identity.get(field) != shared.get(field):
            raise ValueError(f"current runtime sidecar {field} differs from manifest")
    boundary_ids = runtime_identity.get("boundary_token_ids")
    if not isinstance(boundary_ids, (list, tuple)):
        raise ValueError("current runtime boundary_token_ids must be a sequence")
    boundary_ids = tuple(int(value) for value in boundary_ids)
    validate_boundary_token_identity(
        boundary_ids,
        boundary_token_count=runtime_identity.get("boundary_token_count"),
        boundary_ids_sha256=runtime_identity.get("boundary_ids_sha256"),
    )
    hidden_size = runtime_identity.get("hidden_size")
    if (
        isinstance(hidden_size, bool)
        or not isinstance(hidden_size, int)
        or hidden_size <= 0
    ):
        raise ValueError("current runtime sidecar hidden_size is invalid")
    if int(shared["z_width"]) != hidden_size:
        raise ValueError("current runtime hidden size differs from manifest")
    bound.update(
        z_width=hidden_size,
        boundary_token_ids=list(boundary_ids),
        boundary_token_count=int(runtime_identity["boundary_token_count"]),
        boundary_ids_sha256=str(runtime_identity["boundary_ids_sha256"]),
    )
    return bound


def build_selected_r_runtime_verification_view(
    parent: Mapping[str, Any], config: Any
) -> dict[str, Any]:
    """Build the deployment-semantic view used to consume an explicit R checkpoint."""

    del config
    return {
        "tokenizer_sha256": parent["tokenizer_sha256"],
        "boundary_token_ids": list(parent["boundary_token_ids"]),
        "boundary_token_count": int(parent["boundary_token_count"]),
        "boundary_ids_sha256": parent["boundary_ids_sha256"],
    }
