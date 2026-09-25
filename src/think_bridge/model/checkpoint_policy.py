"""Checkpoint integrity, validation metric selection, and exact resume identity."""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence

from think_bridge.model.artifact_schema import (
    ACTIVE_OWNER_CONFIG,
    CHECKPOINT_COMPLETE,
    CHECKPOINT_CONFIG_REFERENCE,
    CHECKPOINT_DIRECTORY,
    CHECKPOINT_IDENTITY,
    STARTUP_INVARIANT_IDENTITY,
    STARTUP_INVARIANT_SEAL,
    TOKENIZER_RUNTIME_SIDECAR,
    STAGE_ARTIFACT_LOCATOR,
    STAGE_RESOLVED_CONFIG,
    artifact_header,
    require_artifact_header,
)

from think_bridge.model.contract import (
    D_ANSWER_EOS_RULE,
    D_ANSWER_PROVENANCE_SCHEMA,
    CHECKPOINT_SCHEMA_VERSION,
    COURSE_SCHEMA_VERSION,
    CAUSAL_OBJECTIVE_SCHEMA_VERSION,
    OBJECTIVE_VERSION,
    SPECIFICITY_OBJECTIVE_VERSION,
    ROUTE1_OCCURRENCE_SAMPLER_SCHEMA,
    canonical_json_sha256,
    file_sha256,
    is_bridge_isolated_path,
    require_sha256,
    validate_boundary_token_identity,
    write_atomic_json,
)


@dataclass(frozen=True)
class Route1Selection:
    step: int
    metrics: Mapping[str, Any]


CHECKPOINT_METADATA_NAME = "checkpoint.json"
CHECKPOINT_COMPLETE_NAME = "COMPLETE"
_STARTUP_INVARIANT_SEAL_FIELDS = frozenset(
    {
        "artifact_type",
        "schema_version",
        "identity",
        "identity_sha256",
        "effective_probe_count",
        "dtype",
        "checks",
    }
)
_STARTUP_INVARIANT_IDENTITY_FIELDS = frozenset(
    {
        "artifact_type",
        "schema_version",
        "model_name_or_path",
        "tokenizer_name_or_path",
        "attn_implementation",
        "boundary_text",
        "tokenizer_sha256",
        "template_sha256",
        "boundary_ids_sha256",
        "tokenizer_files_sha256",
    }
)
_STARTUP_INVARIANT_SHA_FIELDS = (
    "tokenizer_sha256",
    "template_sha256",
    "boundary_ids_sha256",
    "tokenizer_files_sha256",
)
_STARTUP_INVARIANT_DTYPES = frozenset(
    {"torch.float32", "torch.float16", "torch.bfloat16"}
)
_CHECKPOINT_REQUIRED_BASE_SIDECARS = frozenset(
    {
        "model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "active_owner_config.json",
        "config_ref.json",
    }
)
_CHECKPOINT_CONFIG_ARTIFACTS_SPLIT = frozenset(
    {"resolved_config", "artifacts", "runtime_sidecars"}
)


def checkpoint_run_contract_kind(run_dir: Path) -> str:
    """Identify the immutable control-plane contract owned by one run."""

    run = Path(run_dir).resolve(strict=True)
    resolved = _read_json_object(
        run / "resolved_config.json", label="resolved run config"
    )
    require_artifact_header(
        resolved, STAGE_RESOLVED_CONFIG, label="resolved run config"
    )
    artifacts = _read_json_object(
        run / "artifacts.json", label="stage artifact locator"
    )
    require_artifact_header(
        artifacts, STAGE_ARTIFACT_LOCATOR, label="stage artifact locator"
    )
    if artifacts.get("stage") != resolved.get("stage") or artifacts.get(
        "scientific_identity_sha256"
    ) != resolved.get("scientific_identity_sha256"):
        raise ValueError("resolved config/artifact locator binding mismatch")
    return "split"


def write_checkpoint_config_reference(
    building: Path,
    *,
    run_dir: Path,
    include_startup_invariant: bool = False,
) -> str:
    """Publish relative, content-bound checkpoint references for one run kind.

    Split checkpoints snapshot their stage-owned resolved config and artifact
    locator inside the checkpoint.  The runtime sidecar remains a run-level
    immutable tokenizer and frozen-executor identity asset.
    """

    root = Path(building).resolve(strict=True)
    run = Path(run_dir).resolve(strict=True)
    kind = checkpoint_run_contract_kind(run)
    resolved_source = run / "resolved_config.json"
    artifacts_source = run / "artifacts.json"
    resolved_local = root / "resolved_config.json"
    artifacts_local = root / "artifacts.json"
    shutil.copy2(resolved_source, resolved_local)
    shutil.copy2(artifacts_source, artifacts_local)
    sources = {
        "resolved_config": resolved_local,
        "artifacts": artifacts_local,
        "runtime_sidecars": run / "runtime_sidecars.json",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "checkpoint immutable run sidecars are missing: " + ", ".join(missing)
        )
    write_atomic_json(
        root / "config_ref.json",
        {
            **artifact_header(CHECKPOINT_CONFIG_REFERENCE),
            "artifacts": {
                name: {
                    "path": os.path.relpath(source, root),
                    "sha256": file_sha256(source),
                }
                for name, source in sources.items()
            },
        },
        replace_mismatch=False,
    )
    return kind


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def startup_invariant_check_sequence(answer_probe_count: int) -> tuple[str, ...]:
    """Return the exact formal checks emitted for one tokenizer-native answer probe."""

    if (
        isinstance(answer_probe_count, bool)
        or not isinstance(answer_probe_count, int)
        or answer_probe_count <= 0
    ):
        raise ValueError("startup invariant answer probe count must be positive")
    checks: list[str] = []
    structural = (
        "valid-token-order",
        "mask-hole",
        "logical-position-ids",
        "causal-support",
        "qstar-coordinate",
        "answer-coordinates",
    )
    for route in ("null", "teacher-cot"):
        checks.extend(f"{route}-{suffix}" for suffix in structural)
        checks.extend(
            (
                f"{route}-compact-qstar-hidden-finite",
                f"{route}-compact-answer-full-vocab-logits-finite",
            )
        )
    checks.extend(f"student-z-{suffix}" for suffix in structural)
    checks.extend(
        (
            "student-z-compact-qstar-and-answer-finite",
            "student-z-input-gradient-finite-nonzero",
            "right-padding-structure-shape-and-finite",
        )
    )
    for route in ("null", "teacher-cot", "student-z"):
        checks.extend(
            f"{route}-cache-and-no-cache-api-shape-finite-step-{step}"
            for step in range(answer_probe_count)
        )
        checks.append(f"{route}-cache-physical-and-logical-cursor-progression")
    checks.append("cache-explicit-logical-position-and-physical-cache-progression")
    return tuple(checks)


def _validated_startup_invariant_identity(
    raw_identity: Any,
) -> dict[str, Any]:

    if (
        not isinstance(raw_identity, dict)
        or not _STARTUP_INVARIANT_IDENTITY_FIELDS.issubset(raw_identity)
        or set(raw_identity).difference(_STARTUP_INVARIANT_IDENTITY_FIELDS)
        or raw_identity.get("artifact_type") != STARTUP_INVARIANT_IDENTITY
        or raw_identity.get("schema_version") != 1
    ):
        raise ValueError("startup invariant identity schema mismatch")
    for field in (
        "model_name_or_path",
        "tokenizer_name_or_path",
        "boundary_text",
    ):
        value = raw_identity.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"startup invariant identity {field} is empty")
    if raw_identity.get("attn_implementation") != "sdpa":
        raise ValueError("startup invariant identity requires SDPA")
    for field in _STARTUP_INVARIANT_SHA_FIELDS:
        require_sha256(raw_identity.get(field), field)
    return {name: raw_identity[name] for name in _STARTUP_INVARIANT_IDENTITY_FIELDS}


def validate_startup_invariant_seal(
    source: Path | Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any] | None = None,
    expected_probe_count: int | None = None,
    expected_answer_probe_count: int | None = None,
    expected_dtype: str | None = None,
) -> dict[str, Any]:
    """Validate one startup proof before reuse or checkpoint trust promotion."""

    if isinstance(source, Mapping):
        payload: Any = dict(source)
    else:
        seal_path = Path(source)
        if seal_path.is_symlink() or not seal_path.is_file():
            raise ValueError("startup invariant seal is not a regular file")
        payload = json.loads(seal_path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or set(payload) != _STARTUP_INVARIANT_SEAL_FIELDS
        or payload.get("artifact_type") != STARTUP_INVARIANT_SEAL
        or payload.get("schema_version") != 1
    ):
        raise ValueError("startup invariant seal schema mismatch")

    raw_identity = payload.get("identity")
    identity = _validated_startup_invariant_identity(raw_identity)
    sealed_identity_sha256 = require_sha256(
        payload.get("identity_sha256"), "identity_sha256"
    )
    if sealed_identity_sha256 != canonical_json_sha256(raw_identity):
        raise ValueError("startup invariant seal identity digest mismatch")
    if expected_identity is not None:
        validated_expected = _validated_startup_invariant_identity(
            dict(expected_identity)
        )
        if identity != validated_expected:
            raise ValueError("startup invariant seal belongs to another identity")

    probe_count = payload.get("effective_probe_count")
    if (
        isinstance(probe_count, bool)
        or not isinstance(probe_count, int)
        or probe_count <= 0
    ):
        raise ValueError("startup invariant effective probe count must be positive")
    if expected_probe_count is not None:
        if (
            isinstance(expected_probe_count, bool)
            or not isinstance(expected_probe_count, int)
            or expected_probe_count <= 0
        ):
            raise ValueError("expected startup invariant probe count must be positive")
        if probe_count != expected_probe_count:
            raise ValueError("startup invariant effective probe count changed")

    dtype = payload.get("dtype")
    if dtype not in _STARTUP_INVARIANT_DTYPES:
        raise ValueError("startup invariant executor dtype is invalid")
    if expected_dtype is not None:
        if expected_dtype not in _STARTUP_INVARIANT_DTYPES:
            raise ValueError("expected startup invariant executor dtype is invalid")
        if dtype != expected_dtype:
            raise ValueError("startup invariant executor dtype changed")

    checks = payload.get("checks")
    if not isinstance(checks, list) or not all(
        isinstance(check, str) for check in checks
    ):
        raise ValueError("startup invariant checks schema mismatch")
    variable_check_count = len(checks) - 29
    if variable_check_count <= 0 or variable_check_count % 3:
        raise ValueError("startup invariant checks are incomplete")
    answer_probe_count = variable_check_count // 3
    expected_checks = startup_invariant_check_sequence(answer_probe_count)
    if tuple(checks) != expected_checks:
        raise ValueError("startup invariant checks do not match the formal sequence")
    if expected_answer_probe_count is not None:
        startup_invariant_check_sequence(expected_answer_probe_count)
        if answer_probe_count != expected_answer_probe_count:
            raise ValueError("startup invariant answer probe count changed")
    return payload


@dataclass(frozen=True)
class SealedCheckpoint:
    """In-process proof that one checkpoint passed its publication boundary.

    The handle is trusted only inside the transaction that either sealed the
    directory or fully validated it.  Persisted paths/digests are not handles:
    a new process must recreate one with :func:`validate_checkpoint_directory`.
    """

    path: Path
    route: str
    step: int
    metadata_sha256: str
    artifact_sha256: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "route": self.route,
            "step": int(self.step),
            "metadata_sha256": self.metadata_sha256,
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SealedCheckpoint":
        if not isinstance(payload, Mapping) or set(payload) != {
            "path",
            "route",
            "step",
            "metadata_sha256",
            "artifact_sha256",
        }:
            raise ValueError("sealed checkpoint handle schema mismatch")
        route = str(payload["route"])
        step = payload["step"]
        _validate_checkpoint_route_step(route, step)
        path = Path(str(payload["path"]))
        if not path.is_absolute() or path.is_symlink() or not path.is_dir():
            raise ValueError("sealed checkpoint handle path is not complete")
        resolved = path.resolve(strict=True)
        parsed_route, parsed_step = parse_checkpoint_path(resolved)
        if parsed_route != route or parsed_step != int(step):
            raise ValueError("sealed checkpoint handle route/step mismatch")
        return cls(
            path=resolved,
            route=route,
            step=int(step),
            metadata_sha256=require_sha256(
                payload["metadata_sha256"], "metadata_sha256"
            ),
            artifact_sha256=require_sha256(
                payload["artifact_sha256"], "artifact_sha256"
            ),
        )


def validate_runtime_sidecars(run_dir: Path) -> dict[str, Any]:
    """Validate immutable tokenizer and frozen-model identity assets."""

    run = Path(run_dir).resolve(strict=True)
    index = run / "runtime_sidecars.json"
    if not index.is_file():
        raise FileNotFoundError("immutable runtime sidecar index is missing")
    payload = json.loads(index.read_text(encoding="utf-8"))
    tokenizer_only_fields = {
        "artifact_type",
        "schema_version",
        "tokenizer_path",
        "tokenizer_files",
        "tokenizer_files_sha256",
        "identity",
        "identity_sha256",
    }
    tokenizer_only = (
        isinstance(payload, dict)
        and set(payload) == tokenizer_only_fields
        and payload.get("artifact_type") == TOKENIZER_RUNTIME_SIDECAR
        and payload.get("schema_version") == 1
    )
    if not tokenizer_only or not isinstance(payload.get("tokenizer_files"), dict):
        raise ValueError("immutable runtime sidecar schema mismatch")
    if tokenizer_only:
        identity = payload.get("identity")
        if not isinstance(identity, Mapping) or payload.get(
            "identity_sha256"
        ) != canonical_json_sha256(identity):
            raise ValueError("immutable tokenizer runtime identity mismatch")
        from think_bridge.model.executor_identity import validate_executor_identity

        validate_executor_identity(identity.get("frozen_executor_identity"))
        for name in (
            "tokenizer_sha256",
            "template_sha256",
            "boundary_ids_sha256",
        ):
            require_sha256(identity.get(name), name)
    relative = Path(str(payload.get("tokenizer_path", "")))
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != "runtime_sidecars/tokenizer"
    ):
        raise ValueError("immutable tokenizer sidecar path escaped its run")
    tokenizer_root = (run / relative).resolve(strict=True)
    try:
        tokenizer_root.relative_to(run)
    except ValueError as exc:
        raise ValueError("immutable tokenizer sidecar path escaped its run") from exc
    if not tokenizer_root.is_dir() or tokenizer_root.is_symlink():
        raise ValueError("immutable tokenizer sidecar root is invalid")
    observed: dict[str, str] = {}
    for path in sorted(tokenizer_root.rglob("*")):
        if path.is_symlink():
            raise ValueError("immutable tokenizer sidecar cannot contain symlinks")
        if path.is_file():
            observed[path.relative_to(tokenizer_root).as_posix()] = file_sha256(path)
    if (
        not observed
        or observed != payload["tokenizer_files"]
        or payload.get("tokenizer_files_sha256") != canonical_json_sha256(observed)
    ):
        raise ValueError("immutable tokenizer sidecar ledger changed")
    return payload


def _validate_checkpoint_route_step(route: str, step: int) -> None:
    if route not in {"route1"}:
        raise ValueError("checkpoint route must be route1")
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError("checkpoint step must be a positive integer")


def checkpoint_path(run_dir: Path, route: str, step: int) -> Path:
    """Return the maintained run-owned Bridge checkpoint directory."""

    _validate_checkpoint_route_step(route, step)
    return Path(run_dir) / f"checkpoint-{step}"


def _checkpoint_step_from_name(path: Path) -> int:
    value = Path(path)
    match = re.fullmatch(r"checkpoint-([1-9][0-9]*)", value.name)
    if match is None:
        raise ValueError(f"malformed ThinkBridge checkpoint path: {value}")
    return int(match.group(1))


def checkpoint_owner_run_directory(path: Path) -> Path:
    """Return the run containing a flat checkpoint directory."""
    value = Path(path)
    identity = value.with_name(value.name.removesuffix(".building"))
    _checkpoint_step_from_name(identity)
    return identity.parent


def checkpoint_building_path(path: Path) -> Path:
    _checkpoint_step_from_name(path)
    return Path(path).with_name(f"{Path(path).name}.building")


def prepare_checkpoint_building_directory(
    final: Path, *, recover_incomplete: bool
) -> Path:
    """Create one transaction directory, optionally discarding an orphan build."""

    final = Path(final)
    _checkpoint_step_from_name(final)
    run_dir = checkpoint_owner_run_directory(final)
    if not is_bridge_isolated_path(run_dir):
        raise ValueError("checkpoint transaction escaped the ThinkBridge namespace")
    building = checkpoint_building_path(final)
    if final.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {final}")
    if building.exists():
        if not recover_incomplete:
            raise FileExistsError(
                f"stale checkpoint transaction requires explicit resume: {building}"
            )
        if not building.is_dir() or building.is_symlink():
            raise ValueError("orphan checkpoint transaction is not a safe directory")
        shutil.rmtree(building)
    building.mkdir(parents=True)
    return building


def parse_checkpoint_path(path: Path) -> tuple[str, int]:
    """Read and validate route and step from the checkpoint metadata."""
    value = Path(path)
    step = _checkpoint_step_from_name(value)
    metadata_path = value / CHECKPOINT_METADATA_NAME
    if not metadata_path.is_file() or metadata_path.is_symlink():
        raise ValueError("checkpoint route requires sealed metadata")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    route = str(metadata.get("route", ""))
    metadata_step = metadata.get("step")
    _validate_checkpoint_route_step(route, metadata_step)
    if int(metadata_step) != step:
        raise ValueError("checkpoint step differs from its path")
    return route, step


def _checkpoint_file_ledger(root: Path) -> dict[str, str]:
    ledger: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("checkpoint directories cannot contain symlinks")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in {CHECKPOINT_METADATA_NAME, CHECKPOINT_COMPLETE_NAME}:
            continue
        ledger[relative] = file_sha256(path)
    return ledger


def _validate_checkpoint_config_reference(
    checkpoint_root: Path, *, run_dir: Path
) -> str:
    reference_path = Path(checkpoint_root) / "config_ref.json"
    if not reference_path.is_file():
        raise ValueError("checkpoint config reference is missing")
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or set(payload) != {"artifact_type", "schema_version", "artifacts"}
        or not isinstance(payload.get("artifacts"), dict)
    ):
        raise ValueError("checkpoint config reference identity mismatch")
    require_artifact_header(
        payload, CHECKPOINT_CONFIG_REFERENCE, label="checkpoint config reference"
    )
    kind = "split"
    allowed_artifact_sets = {_CHECKPOINT_CONFIG_ARTIFACTS_SPLIT}
    run = Path(run_dir).resolve(strict=True)
    expected_artifacts = frozenset(payload["artifacts"])
    if expected_artifacts not in allowed_artifact_sets:
        raise ValueError("checkpoint config reference artifact set mismatch")
    resolved_payload: dict[str, Any] | None = None
    artifact_payload: dict[str, Any] | None = None
    for name in sorted(expected_artifacts):
        row = payload["artifacts"][name]
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise ValueError("checkpoint config reference row is malformed")
        relative = Path(str(row["path"]))
        if relative.is_absolute():
            raise ValueError("checkpoint config reference must be relative")
        target = (Path(checkpoint_root) / relative).resolve(strict=True)
        try:
            target.relative_to(run)
        except ValueError as exc:
            raise ValueError("checkpoint config reference escaped its run") from exc
        if not target.is_file() or file_sha256(target) != row["sha256"]:
            raise ValueError(f"checkpoint immutable config artifact changed: {name}")
        if name == "runtime_sidecars":
            if kind == "split" and target != run / "runtime_sidecars.json":
                raise ValueError("split runtime sidecar reference is not run-owned")
            validate_runtime_sidecars(run)
        elif name == "startup_invariant":
            validate_startup_invariant_seal(target)
        elif name == "run_arguments":
            local_args = Path(checkpoint_root) / "args.json"
            if not local_args.is_file() or file_sha256(local_args) != row["sha256"]:
                raise ValueError(
                    "checkpoint-local args differ from immutable run arguments"
                )
        elif kind == "split" and name == "resolved_config":
            if (
                target
                != Path(checkpoint_root).resolve(strict=True) / "resolved_config.json"
            ):
                raise ValueError(
                    "split checkpoint resolved config is not checkpoint-local"
                )
            resolved_payload = _read_json_object(
                target, label="split checkpoint resolved config"
            )
        elif kind == "split" and name == "artifacts":
            if target != Path(checkpoint_root).resolve(strict=True) / "artifacts.json":
                raise ValueError(
                    "split checkpoint artifact locator is not checkpoint-local"
                )
            artifact_payload = _read_json_object(
                target, label="split checkpoint artifact locator"
            )
    if kind == "split":
        if (
            resolved_payload is None
            or artifact_payload is None
            or resolved_payload.get("artifact_type") != STAGE_RESOLVED_CONFIG
            or resolved_payload.get("schema_version") != 1
            or artifact_payload.get("artifact_type") != STAGE_ARTIFACT_LOCATOR
            or artifact_payload.get("schema_version") != 1
            or artifact_payload.get("stage") != resolved_payload.get("stage")
            or artifact_payload.get("scientific_identity_sha256")
            != resolved_payload.get("scientific_identity_sha256")
        ):
            raise ValueError("split checkpoint stage contract binding mismatch")
    return kind


def seal_checkpoint_directory(
    building: Path,
    final: Path,
    *,
    route: str,
    step: int,
    metadata: Mapping[str, Any],
) -> SealedCheckpoint:
    """Seal a prepared directory and atomically publish it as one checkpoint."""

    _validate_checkpoint_route_step(route, step)
    final = Path(final)
    building = Path(building)
    run_dir = checkpoint_owner_run_directory(final)
    if final != checkpoint_path(run_dir, route, step):
        raise ValueError("checkpoint final path differs from its route/step identity")
    if building != checkpoint_building_path(final):
        raise ValueError("checkpoint building path is not the exact final sibling")
    if final.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {final}")
    if not building.is_dir() or building.is_symlink():
        raise ValueError("checkpoint building transaction is not a directory")
    contract_kind = _validate_checkpoint_config_reference(building, run_dir=run_dir)
    ledger = _checkpoint_file_ledger(building)
    required_sidecars = _CHECKPOINT_REQUIRED_BASE_SIDECARS | (
        {"resolved_config.json", "artifacts.json"}
        if contract_kind == "split"
        else {"args.json"}
    )
    missing = sorted(required_sidecars.difference(ledger))
    runtime_files = sorted(
        name for name in ledger if re.fullmatch(r"runtime/rank-[0-9]{5}\.pt", name)
    )
    if missing or not runtime_files:
        raise ValueError(
            "checkpoint sidecars are incomplete; "
            f"missing={missing}, runtime_files={runtime_files}"
        )
    world_size = metadata.get("world_size")
    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size <= 0
    ):
        raise ValueError("checkpoint metadata world_size must be positive")
    expected_runtime = [f"runtime/rank-{rank:05d}.pt" for rank in range(world_size)]
    if runtime_files != expected_runtime:
        raise ValueError("checkpoint per-rank runtime sidecars are incomplete")
    checkpoint_metadata = {
        **dict(metadata),
        **artifact_header(CHECKPOINT_DIRECTORY),
        "route": route,
        "step": int(step),
        "file_ledger": ledger,
    }
    _validate_checkpoint_metadata_semantics(
        building,
        metadata=checkpoint_metadata,
        route=route,
        step=step,
        ledger=ledger,
    )
    metadata_path = building / CHECKPOINT_METADATA_NAME
    write_atomic_json(metadata_path, checkpoint_metadata, replace_mismatch=False)
    metadata_sha256 = file_sha256(metadata_path)
    artifact_sha256 = canonical_json_sha256(
        {
            **artifact_header(CHECKPOINT_COMPLETE),
            "route": route,
            "step": int(step),
            "metadata_sha256": metadata_sha256,
            "file_ledger": ledger,
        }
    )
    write_atomic_json(
        building / CHECKPOINT_COMPLETE_NAME,
        {
            **artifact_header(CHECKPOINT_COMPLETE),
            "route": route,
            "step": int(step),
            "metadata_sha256": metadata_sha256,
            "artifact_sha256": artifact_sha256,
        },
        replace_mismatch=False,
    )
    building.replace(final)
    return SealedCheckpoint(
        path=final.resolve(strict=True),
        route=route,
        step=int(step),
        metadata_sha256=metadata_sha256,
        artifact_sha256=artifact_sha256,
    )


def checkpoint_seal_from_marker(path: Path) -> SealedCheckpoint:
    """Read the small publication marker without trusting checkpoint payloads.

    This is intentionally *not* an external artifact-validation boundary.  It
    binds path/route/step plus the immutable metadata ledger and is used only
    for in-process pointer/prune bookkeeping.  Resume, evaluation, selection,
    and selected-z consumers must call :func:`validate_checkpoint_directory`.
    """

    checkpoint = Path(path)
    route, step = parse_checkpoint_path(checkpoint)
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("checkpoint is not a complete directory")
    checkpoint = checkpoint.resolve(strict=True)
    resolved_route, resolved_step = parse_checkpoint_path(checkpoint)
    if resolved_route != route or resolved_step != step:
        raise ValueError("checkpoint route/step changed during resolution")
    metadata_path = checkpoint / CHECKPOINT_METADATA_NAME
    complete_path = checkpoint / CHECKPOINT_COMPLETE_NAME
    if (
        metadata_path.is_symlink()
        or complete_path.is_symlink()
        or not metadata_path.is_file()
        or not complete_path.is_file()
    ):
        raise ValueError("checkpoint has no sealed metadata/COMPLETE marker")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if (
        not isinstance(metadata, dict)
        or metadata.get("artifact_type") != CHECKPOINT_DIRECTORY
        or metadata.get("schema_version") != 1
        or metadata.get("route") != route
        or metadata.get("step") != step
        or not isinstance(metadata.get("file_ledger"), dict)
    ):
        raise ValueError("checkpoint metadata identity mismatch")
    ledger = metadata["file_ledger"]
    for relative, digest in ledger.items():
        candidate = Path(str(relative))
        if (
            not isinstance(relative, str)
            or candidate.is_absolute()
            or ".." in candidate.parts
            or relative in {CHECKPOINT_METADATA_NAME, CHECKPOINT_COMPLETE_NAME}
        ):
            raise ValueError("checkpoint metadata ledger path is malformed")
        require_sha256(digest, f"file_ledger[{relative}]")
    expected_complete = {
        **artifact_header(CHECKPOINT_COMPLETE),
        "route": route,
        "step": step,
        "metadata_sha256": file_sha256(metadata_path),
    }
    if (
        not isinstance(complete, dict)
        or set(complete) != {*expected_complete, "artifact_sha256"}
        or any(complete.get(name) != value for name, value in expected_complete.items())
    ):
        raise ValueError("checkpoint COMPLETE identity mismatch")
    artifact_sha256 = canonical_json_sha256(
        {**expected_complete, "file_ledger": ledger}
    )
    if complete.get("artifact_sha256") != artifact_sha256:
        raise ValueError("checkpoint artifact hash mismatch")
    return SealedCheckpoint(
        path=checkpoint,
        route=route,
        step=step,
        metadata_sha256=expected_complete["metadata_sha256"],
        artifact_sha256=artifact_sha256,
    )


def validate_checkpoint_seal(seal: SealedCheckpoint) -> SealedCheckpoint:
    """Cheaply confirm an in-memory handle still matches its seal marker."""

    if not isinstance(seal, SealedCheckpoint):
        raise TypeError("checkpoint seal must be a SealedCheckpoint")
    observed = checkpoint_seal_from_marker(seal.path)
    if observed != seal:
        raise ValueError("sealed checkpoint handle differs from its marker")
    return observed


def validate_checkpoint_directory(path: Path) -> SealedCheckpoint:
    """Fully validate one checkpoint at an external trust boundary."""

    checkpoint = Path(path)
    seal = checkpoint_seal_from_marker(checkpoint)
    checkpoint = seal.path
    contract_kind = _validate_checkpoint_config_reference(
        checkpoint, run_dir=checkpoint_owner_run_directory(checkpoint)
    )
    metadata = json.loads(
        (checkpoint / CHECKPOINT_METADATA_NAME).read_text(encoding="utf-8")
    )
    ledger = _checkpoint_file_ledger(checkpoint)
    if ledger != metadata["file_ledger"]:
        raise ValueError("checkpoint sidecar ledger changed after publication")
    _validate_checkpoint_metadata_semantics(
        checkpoint,
        metadata=metadata,
        route=seal.route,
        step=seal.step,
        ledger=ledger,
    )
    required_sidecars = _CHECKPOINT_REQUIRED_BASE_SIDECARS | (
        {"resolved_config.json", "artifacts.json"}
        if contract_kind == "split"
        else {"args.json"}
    )
    missing = sorted(required_sidecars.difference(ledger))
    world_size = metadata.get("world_size")
    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size <= 0
    ):
        raise ValueError("checkpoint metadata world_size is invalid")
    runtime_files = sorted(
        name for name in ledger if re.fullmatch(r"runtime/rank-[0-9]{5}\.pt", name)
    )
    expected_runtime = [f"runtime/rank-{rank:05d}.pt" for rank in range(world_size)]
    if missing or runtime_files != expected_runtime:
        raise ValueError("checkpoint required/per-rank sidecar set is incomplete")
    return seal


def validate_checkpoint_directory_rank0(
    path: Path, *, distributed: Any, rank: int
) -> tuple[dict[str, Any], SealedCheckpoint]:
    """Fully validate once on rank 0 and broadcast its metadata/sealed handle.

    The broadcast is the distributed trust handoff.  Nonzero ranks must not
    recreate the external boundary by independently hashing a shared directory.
    """

    local_payload: dict[str, Any] | None = None
    if int(rank) == 0:
        try:
            seal = validate_checkpoint_directory(path)
            metadata = json.loads(
                (seal.path / CHECKPOINT_METADATA_NAME).read_text(encoding="utf-8")
            )
            if not isinstance(metadata, dict):
                raise ValueError("checkpoint metadata is not a mapping")
            local_payload = {
                "error": None,
                "metadata": metadata,
                "seal": seal.to_mapping(),
            }
        except Exception as exc:
            local_payload = {
                "error": f"{type(exc).__name__}: {exc}",
                "metadata": None,
                "seal": None,
            }
    if distributed.is_initialized():
        values = [local_payload if int(rank) == 0 else None]
        distributed.broadcast_object_list(values, src=0)
        payload = values[0]
    else:
        if int(rank) != 0:
            raise RuntimeError("non-distributed checkpoint validation requires rank 0")
        payload = local_payload
    if not isinstance(payload, Mapping):
        raise RuntimeError("rank-0 checkpoint validation result is missing")
    if payload.get("error") is not None:
        raise RuntimeError(f"rank-0 checkpoint validation failed: {payload['error']}")
    metadata = payload.get("metadata")
    seal_payload = payload.get("seal")
    if not isinstance(metadata, Mapping) or not isinstance(seal_payload, Mapping):
        raise RuntimeError("rank-0 checkpoint validation payload is malformed")
    seal = SealedCheckpoint.from_mapping(seal_payload)
    requested = Path(path).resolve(strict=True)
    if seal.path != requested:
        raise ValueError(
            "rank-0 checkpoint seal differs from this rank's requested path"
        )
    return dict(metadata), seal


def checkpoint_artifact_sha256(path: Path) -> str:
    """Validate every sidecar in a complete checkpoint directory and hash it."""

    return validate_checkpoint_directory(path).artifact_sha256


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    checkpoint_artifact_sha256(path)
    payload = json.loads(
        (Path(path) / CHECKPOINT_METADATA_NAME).read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise ValueError("checkpoint metadata is not a mapping")
    return payload


def artifact_sha256(path: Path) -> str:
    """Hash an ordinary file or a strictly validated checkpoint directory."""

    value = Path(path)
    if value.is_file():
        return file_sha256(value)
    if value.is_dir():
        return checkpoint_artifact_sha256(value)
    raise FileNotFoundError(f"ThinkBridge artifact is missing: {value}")


def write_checkpoint_pointer(
    pointer: Path, *, run_dir: Path, checkpoint: Path | SealedCheckpoint
) -> None:
    """Atomically write a human-readable path/step pointer, never JSON-in-``.txt``."""

    run = Path(run_dir).resolve(strict=True)
    seal = (
        validate_checkpoint_seal(checkpoint)
        if isinstance(checkpoint, SealedCheckpoint)
        else validate_checkpoint_directory(Path(checkpoint))
    )
    target = seal.path
    route, step = seal.route, seal.step
    try:
        relative = target.relative_to(run)
    except ValueError as exc:
        raise ValueError("checkpoint pointer escaped the run directory") from exc
    expected = checkpoint_path(run, route, step).resolve(strict=True)
    if target != expected:
        raise ValueError("checkpoint pointer target is outside its route namespace")
    value = f"path={relative.as_posix()}\nstep={step}\n"
    pointer = Path(pointer)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    temporary = pointer.with_suffix(pointer.suffix + ".building")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(pointer)


def _checkpoint_pointer_target(pointer: Path, *, run_dir: Path) -> Path:
    lines = Path(pointer).read_text(encoding="utf-8").splitlines()
    if (
        len(lines) != 2
        or not lines[0].startswith("path=")
        or not lines[1].startswith("step=")
    ):
        raise ValueError("checkpoint pointer text schema mismatch")
    relative = Path(lines[0].removeprefix("path="))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("checkpoint pointer path escaped the run directory")
    try:
        step = int(lines[1].removeprefix("step="))
    except ValueError as exc:
        raise ValueError("checkpoint pointer step is not integral") from exc
    target = Path(run_dir) / relative
    _, parsed_step = parse_checkpoint_path(target)
    if parsed_step != step:
        raise ValueError("checkpoint pointer path/step mismatch")
    run = Path(run_dir).resolve(strict=True)
    resolved = target.resolve(strict=True)
    try:
        resolved.relative_to(run)
    except ValueError as exc:
        raise ValueError("checkpoint pointer path escaped the run directory") from exc
    route, _ = parse_checkpoint_path(resolved)
    if resolved != checkpoint_path(run, route, step).resolve(strict=True):
        raise ValueError("checkpoint pointer target is outside its route namespace")
    return target


def read_checkpoint_pointer_target(pointer: Path, *, run_dir: Path) -> Path:
    """Resolve a pointer using only its small marker for internal bookkeeping."""

    target = _checkpoint_pointer_target(pointer, run_dir=run_dir)
    checkpoint_seal_from_marker(target)
    return target


def read_checkpoint_pointer(pointer: Path, *, run_dir: Path) -> Path:
    """Resolve and fully validate a checkpoint pointer at a trust boundary."""

    target = _checkpoint_pointer_target(pointer, run_dir=run_dir)
    validate_checkpoint_directory(target)
    return target


def prune_checkpoint_namespace(
    namespace: Path,
    *,
    route: str,
    protected: Sequence[Path | SealedCheckpoint],
    limit: int = 10,
) -> tuple[Path, ...]:
    """Delete only oldest unprotected complete directories for one route."""
    root = Path(namespace).resolve()
    if (
        route not in {"route1"}
        or isinstance(limit, bool)
        or (not isinstance(limit, int))
        or (limit < 1)
        or (not is_bridge_isolated_path(root))
    ):
        raise ValueError(
            "checkpoint pruning identity is outside sealed ThinkBridge policy"
        )
    route_root = checkpoint_path(root, route, 1).parent
    protected_resolved = {
        validate_checkpoint_seal(path).path
        if isinstance(path, SealedCheckpoint)
        else Path(path).resolve()
        for path in protected
    }
    for path in protected_resolved:
        if path.parent != route_root:
            raise ValueError("protected checkpoint escaped its owner namespace")
    limit = max(limit, len(protected_resolved))
    candidates: list[tuple[int, Path, SealedCheckpoint]] = []
    if route_root.exists() and (not route_root.is_dir()):
        raise ValueError("checkpoint route namespace is not a directory")
    for path in route_root.glob("checkpoint-*"):
        try:
            (candidate_route, step) = parse_checkpoint_path(path)
        except ValueError:
            continue
        if candidate_route != route:
            continue
        seal = checkpoint_seal_from_marker(path)
        candidates.append((step, seal.path, seal))
    removed: list[Path] = []
    for _, path, seal in sorted(candidates, key=lambda row: row[0]):
        if len(candidates) - len(removed) <= limit:
            break
        if path in protected_resolved:
            continue
        _delete_checkpoint_directory(path, namespace=route_root, seal=seal)
        removed.append(path)
    if len(candidates) - len(removed) > limit:
        raise ValueError("too many protected checkpoints to satisfy save_total_limit")
    return tuple(removed)


def _delete_checkpoint_directory(
    path: Path, *, namespace: Path, seal: SealedCheckpoint
) -> None:
    """Delete one marker-validated checkpoint in its exact route root."""

    checkpoint = Path(path).resolve()
    root = Path(namespace).resolve()
    if checkpoint.parent != root or not checkpoint.is_dir() or checkpoint.is_symlink():
        raise ValueError("checkpoint deletion escaped its exact owner namespace")
    if validate_checkpoint_seal(seal).path != checkpoint:
        raise ValueError("checkpoint deletion seal differs from its target")
    shutil.rmtree(checkpoint)


ROUTE1_GATE_METRIC_FIELDS = frozenset(
    {
        "step",
        "seed",
        "judge_path",
        "donor_count",
        "bootstrap_ci_generated",
        "paired_row_sha256",
        "diagnostics_executed",
        "true_z_full_accuracy",
        "a_true",
        "a_direct",
        "a_wrong",
        "raw_g1",
        "robust_g1",
        "robust_g1_ci_low",
        "b_retention",
        "d_retention",
        "true_full_correct",
        "true_full_total",
        "b_denominator",
        "c_denominator",
        "d_denominator",
        "e_denominator",
        "b_accuracy",
        "c_accuracy",
        "d_accuracy",
        "e_accuracy",
        "c_paired_count",
        "wrong_pair_count",
        "control_available_count",
        "control_unavailable_count",
        "wrong_donor_counts",
    }
)
_ROUTE1_FINITE_FIELDS = frozenset(
    {
        "true_z_full_accuracy",
        "a_true",
        "a_direct",
        "a_wrong",
        "raw_g1",
        "robust_g1",
        "robust_g1_ci_low",
        "b_retention",
        "d_retention",
        "b_accuracy",
        "c_accuracy",
        "d_accuracy",
        "e_accuracy",
    }
)
_ROUTE1_INTEGER_FIELDS = frozenset(
    {
        "step",
        "seed",
        "donor_count",
        "true_full_correct",
        "true_full_total",
        "b_denominator",
        "c_denominator",
        "d_denominator",
        "e_denominator",
        "c_paired_count",
        "wrong_pair_count",
    }
)
_ROUTE1_BOOLEAN_FIELDS = frozenset({"bootstrap_ci_generated", "diagnostics_executed"})
_ROUTE1_OPTIONAL_FIELDS = frozenset(
    {
        "diagnostics_executed",
        "control_available_count",
        "control_unavailable_count",
        "wrong_donor_counts",
    }
)
_ROUTE1_STRING_FIELDS = frozenset({"judge_path", "paired_row_sha256"})


def _exact_metric_domain(
    row: Mapping[str, Any],
    allowed: frozenset[str],
    required: frozenset[str],
    label: str,
) -> None:
    missing = sorted(required.difference(row))
    if missing:
        raise ValueError(f"{label} metrics are incomplete; missing={missing}")
    finite_fields = _ROUTE1_FINITE_FIELDS
    integer_fields = _ROUTE1_INTEGER_FIELDS
    boolean_fields = _ROUTE1_BOOLEAN_FIELDS
    string_fields = _ROUTE1_STRING_FIELDS
    nullable_fields = _ROUTE1_FINITE_FIELDS - {"true_z_full_accuracy"}
    invalid_types = sorted(
        {
            name
            for name in integer_fields
            if name in row
            if isinstance(row[name], bool) or not isinstance(row[name], int)
        }
        | {
            name
            for name in finite_fields
            if name in row
            if not (name in nullable_fields and row[name] is None)
            if isinstance(row[name], bool) or not isinstance(row[name], (int, float))
        }
        | {
            name
            for name in boolean_fields
            if name in row and (not isinstance(row[name], bool))
        }
        | {
            name
            for name in string_fields
            if name in row and (not isinstance(row[name], str))
        }
    )
    if invalid_types:
        raise ValueError(
            f"{label} metrics have invalid field types; invalid={invalid_types}"
        )
    nonfinite = sorted(
        (
            name
            for name in finite_fields
            if name in row
            and (not (name in nullable_fields and row[name] is None))
            and (not math.isfinite(float(row[name])))
        )
    )
    if nonfinite:
        raise ValueError(f"{label} metrics must be finite; invalid={nonfinite}")


def validate_route1_gate_metric_schema(row: Mapping[str, Any]) -> None:
    """Validate the exact Route1 metric domain without applying science gates."""

    required = ROUTE1_GATE_METRIC_FIELDS - _ROUTE1_OPTIONAL_FIELDS
    if row.get("diagnostics_executed") is False:
        required -= {"a_wrong", "robust_g1", "robust_g1_ci_low"}
        if (
            row.get("donor_count") != 0
            or row.get("wrong_pair_count") != 0
            or row.get("bootstrap_ci_generated") is not False
        ):
            raise ValueError(
                "Selection-only Route1 report cannot claim control measurements"
            )
        if any(name in row for name in ("a_wrong", "robust_g1", "robust_g1_ci_low")):
            raise ValueError(
                "Skipped Route1 diagnostics must be absent, not fabricated"
            )
    _exact_metric_domain(
        row,
        ROUTE1_GATE_METRIC_FIELDS,
        required,
        "Route1",
    )
    for name in ("control_available_count", "control_unavailable_count"):
        if name in row and (type(row[name]) is not int or row[name] < 0):
            raise ValueError(f"Route1 {name} must be a nonnegative integer")
    counts = row.get("wrong_donor_counts")
    if counts is not None and (
        not isinstance(counts, list)
        or any(type(count) is not int or count < 0 for count in counts)
    ):
        raise ValueError("Route1 wrong_donor_counts must contain nonnegative integers")


def validate_route1_selection_candidates(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    if not rows:
        raise ValueError("Route1 selector received no validation checkpoints")
    candidates: list[Mapping[str, Any]] = []
    seed: int | None = None
    paired_rows: str | None = None
    for row in rows:
        validate_route1_gate_metric_schema(row)
        require_sha256(row["paired_row_sha256"], "paired_row_sha256")
        if row["judge_path"] != "think_bridge.eval.answer_match.judge_answer":
            raise ValueError(
                "Route1 selector judge path is not the shared boxed-answer judge"
            )
        current_seed = int(row["seed"])
        current_pairs = str(row["paired_row_sha256"])
        if seed is None:
            seed, paired_rows = current_seed, current_pairs
        if (current_seed, current_pairs) != (seed, paired_rows):
            raise ValueError("Route1 candidates do not share seed/paired rows")
        candidates.append(
            {name: row[name] for name in ROUTE1_GATE_METRIC_FIELDS if name in row}
        )
    return tuple(candidates)


@dataclass(frozen=True)
class BridgeCheckpointIdentity:
    artifact_type: str
    schema_version: int
    course_schema_version: str
    objective_version: str
    specificity_objective_version: str
    causal_objective_schema_version: str
    geometry_schema_version: str
    attn_implementation: str
    tokenizer_sha256: str
    boundary_token_ids: tuple[int, ...]
    boundary_token_count: int
    boundary_ids_sha256: str
    phase: str
    method: str
    seed: int
    selected_r_sha256: str
    need_z_cohort_definition: str
    d_answer_provenance_schema: str
    d_answer_eos_rule: str
    model_state_schema_sha256: str
    owned_state_sha256: str
    optimizer_state_sha256: str
    scheduler_state_sha256: str
    rng_state_sha256: str
    sampler_state_sha256: str
    cache_identity_sha256: str
    exact_resume_identity_sha256: str
    optimizer_backend: str
    zero_stage: int
    deepspeed_config_sha256: str
    deepspeed_state_semantics_sha256: str
    world_size: int
    route1_local_samples: int
    route1_gradient_accumulation_steps: int
    resume_artifact_sha256: str
    step: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "BridgeCheckpointIdentity":
        required = {field.name for field in fields(cls)}
        missing = sorted(required.difference(raw))
        unknown = sorted(set(raw).difference(required))
        if missing or unknown:
            raise ValueError(
                f"checkpoint identity mismatch; missing={missing}, unknown={unknown}"
            )
        normalized = {key: raw[key] for key in required}
        if not isinstance(normalized["boundary_token_ids"], (list, tuple)):
            raise ValueError("checkpoint boundary_token_ids must be a sequence")
        normalized["boundary_token_ids"] = tuple(normalized["boundary_token_ids"])
        identity = cls(**normalized)
        identity.validate()
        return identity

    def validate(self) -> None:
        geometry = re.fullmatch(
            "recursive-t([1-9][0-9]*)-b([1-9][0-9]*)-k(32|64|128)-answer-only",
            self.geometry_schema_version,
        )
        if (
            self.artifact_type != CHECKPOINT_IDENTITY
            or self.schema_version != CHECKPOINT_SCHEMA_VERSION
            or self.course_schema_version != COURSE_SCHEMA_VERSION
            or (self.objective_version != OBJECTIVE_VERSION)
            or (self.specificity_objective_version != SPECIFICITY_OBJECTIVE_VERSION)
            or (self.causal_objective_schema_version != CAUSAL_OBJECTIVE_SCHEMA_VERSION)
            or (self.geometry_schema_version != "recursive-t1-b64-k64-answer-only")
            or (geometry is None)
            or (
                int(geometry.group(1)) * int(geometry.group(2))
                != int(geometry.group(3))
            )
            or (self.attn_implementation != "sdpa")
        ):
            raise ValueError("checkpoint is not an exact ThinkBridge checkpoint")
        if self.phase not in ("A",) or self.method != "bridge":
            raise ValueError("checkpoint must belong to ThinkBridge R training")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or int(self.step) < 0
        ):
            raise ValueError("checkpoint seed/step mismatch")
        if (
            isinstance(self.world_size, bool)
            or not isinstance(self.world_size, int)
            or self.world_size <= 0
        ):
            raise ValueError("checkpoint world size must be positive")
        active_geometry = (
            self.route1_local_samples,
            self.route1_gradient_accumulation_steps,
        )
        if any(
            (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in active_geometry
            )
        ):
            raise ValueError("checkpoint active-route batch geometry is invalid")
        if self.optimizer_backend == "replicated_ddp":
            if (
                int(self.zero_stage) != 0
                or set(self.deepspeed_config_sha256) != {"0"}
                or set(self.deepspeed_state_semantics_sha256) != {"0"}
                or (set(self.resume_artifact_sha256) != {"0"})
            ):
                raise ValueError("replicated checkpoint carries a ZeRO identity")
        elif self.optimizer_backend == "deepspeed_zero1":
            if (
                int(self.zero_stage) != 1
                or set(self.deepspeed_state_semantics_sha256) == {"0"}
                or set(self.resume_artifact_sha256) == {"0"}
            ):
                raise ValueError("DeepSpeed checkpoint is not a sharded ZeRO-1 resume")
        else:
            raise ValueError("checkpoint optimizer backend is unsupported")
        validate_boundary_token_identity(
            self.boundary_token_ids,
            boundary_token_count=self.boundary_token_count,
            boundary_ids_sha256=self.boundary_ids_sha256,
        )
        for name in (
            "selected_r_sha256",
            "model_state_schema_sha256",
            "owned_state_sha256",
            "optimizer_state_sha256",
            "scheduler_state_sha256",
            "rng_state_sha256",
            "sampler_state_sha256",
            "cache_identity_sha256",
            "exact_resume_identity_sha256",
            "tokenizer_sha256",
            "boundary_ids_sha256",
            "deepspeed_config_sha256",
            "deepspeed_state_semantics_sha256",
            "resume_artifact_sha256",
        ):
            require_sha256(getattr(self, name), name)
        from think_bridge.model.contract import NEED_Z_COHORT_DEFINITION

        if self.need_z_cohort_definition != NEED_Z_COHORT_DEFINITION:
            raise ValueError("checkpoint exact need-z cohort definition mismatch")
        provenance_matches = (
            self.d_answer_provenance_schema == D_ANSWER_PROVENANCE_SCHEMA
        )
        eos_rule_matches = self.d_answer_eos_rule == D_ANSWER_EOS_RULE
        if not provenance_matches:
            raise ValueError("checkpoint D-answer provenance schema mismatch")
        if not eos_rule_matches:
            raise ValueError("checkpoint D-answer EOS rule mismatch")
        if set(self.selected_r_sha256) != {"0"}:
            raise ValueError("Route1 checkpoint cannot claim selected-R lineage")

    def assert_resume_compatible(self, candidate: Mapping[str, Any]) -> None:
        other = self.from_mapping(candidate)
        if other != self:
            differing = [
                field.name
                for field in fields(self)
                if getattr(self, field.name) != getattr(other, field.name)
            ]
            raise ValueError(f"resume identity differs in exact fields: {differing}")


def _validate_checkpoint_metadata_semantics(
    root: Path,
    *,
    metadata: Mapping[str, Any],
    route: str,
    step: int,
    ledger: Mapping[str, str],
) -> BridgeCheckpointIdentity:
    required = {
        "owner",
        "world_size",
        "identity",
        "runtime_ledger",
        "zero1_manifest",
        "portable_identity_seed",
        "exact_resume_identity_sha256",
        "model_safetensors_sha256",
        "active_owner_config_sha256",
        "tensor_scope",
        "contains_frozen_parameters",
        "excluded_frozen_parameter_names",
        "frozen_parameter_source",
    }
    missing = sorted(required.difference(metadata))
    if missing:
        raise ValueError(f"checkpoint metadata is incomplete: {missing}")
    if not isinstance(metadata["identity"], Mapping):
        raise ValueError("checkpoint metadata identity is malformed")
    identity = BridgeCheckpointIdentity.from_mapping(metadata["identity"])
    expected_phase = {"route1": "A"}.get(route)
    expected_owner = {"route1": "R"}.get(route)
    if expected_phase is None or expected_owner is None:
        raise ValueError("checkpoint metadata route is invalid")
    if (
        identity.phase != expected_phase
        or identity.step != int(step)
        or metadata.get("owner") != expected_owner
        or (int(metadata.get("world_size", -1)) != int(identity.world_size))
        or (
            metadata.get("exact_resume_identity_sha256")
            != identity.exact_resume_identity_sha256
        )
        or (metadata.get("tensor_scope") != "active_owner_trainable_parameters_only")
        or (metadata.get("contains_frozen_parameters") is not False)
    ):
        raise ValueError("checkpoint metadata owner/route/step identity mismatch")
    model_safetensors_sha256 = require_sha256(
        metadata.get("model_safetensors_sha256"), "model_safetensors_sha256"
    )
    active_owner_config_sha256 = require_sha256(
        metadata.get("active_owner_config_sha256"), "active_owner_config_sha256"
    )
    if (
        ledger.get("model.safetensors") != model_safetensors_sha256
        or ledger.get("active_owner_config.json") != active_owner_config_sha256
    ):
        raise ValueError(
            "checkpoint owner/config digests differ from the sealed ledger"
        )
    owner_config = _read_json_object(
        Path(root) / "active_owner_config.json", label="checkpoint active-owner config"
    )
    expected_owner_config_fields = {
        "artifact_type",
        "schema_version",
        "model_type",
        "objective_version",
        "specificity_objective_version",
        "route",
        "phase",
        "method",
        "owner",
        "step",
        "weight_file",
        "weight_format",
        "tensor_dtype",
        "contains_frozen_executor",
        "tensor_scope",
        "contains_frozen_parameters",
        "excluded_frozen_parameter_names",
        "frozen_parameter_source",
        "model_state_schema_sha256",
        "owned_state_sha256",
        "exact_resume_identity_sha256",
        "selected_r_sha256",
        "tokenizer_sha256",
        "tokenizer_files",
    }
    if (
        set(owner_config) != expected_owner_config_fields
        or owner_config.get("artifact_type") != ACTIVE_OWNER_CONFIG
        or owner_config.get("schema_version") != 1
        or (owner_config.get("model_type") != "think_bridge_active_owner")
        or (owner_config.get("objective_version") != OBJECTIVE_VERSION)
        or (
            owner_config.get("specificity_objective_version")
            != identity.specificity_objective_version
        )
        or (owner_config.get("route") != route)
        or (owner_config.get("phase") != expected_phase)
        or (owner_config.get("method") != identity.method)
        or (owner_config.get("owner") != expected_owner)
        or (owner_config.get("step") != int(step))
        or (owner_config.get("weight_file") != "model.safetensors")
        or (owner_config.get("weight_format") != "safetensors")
        or (owner_config.get("tensor_dtype") != "float32")
        or (owner_config.get("contains_frozen_executor") is not False)
        or (
            owner_config.get("tensor_scope") != "active_owner_trainable_parameters_only"
        )
        or (owner_config.get("contains_frozen_parameters") is not False)
        or (
            owner_config.get("model_state_schema_sha256")
            != identity.model_state_schema_sha256
        )
        or (owner_config.get("owned_state_sha256") != identity.owned_state_sha256)
        or (
            owner_config.get("exact_resume_identity_sha256")
            != identity.exact_resume_identity_sha256
        )
        or (owner_config.get("selected_r_sha256") != identity.selected_r_sha256)
        or (owner_config.get("tokenizer_sha256") != identity.tokenizer_sha256)
    ):
        raise ValueError("checkpoint active-owner config identity mismatch")
    expected_frozen_names = {"route1": []}[route]
    if owner_config.get("excluded_frozen_parameter_names") != expected_frozen_names:
        raise ValueError("checkpoint excluded-frozen parameter identity mismatch")
    frozen_source = owner_config.get("frozen_parameter_source")
    if frozen_source is not None:
        raise ValueError("Route1 checkpoint cannot declare a frozen tensor source")
    if (
        metadata.get("excluded_frozen_parameter_names") != expected_frozen_names
        or metadata.get("frozen_parameter_source") != frozen_source
    ):
        raise ValueError("checkpoint seal frozen-parameter provenance mismatch")
    tokenizer_files = owner_config.get("tokenizer_files")
    if not isinstance(tokenizer_files, Mapping) or not tokenizer_files:
        raise ValueError("checkpoint active-owner tokenizer ledger is missing")
    observed_tokenizer: dict[str, str] = {}
    tokenizer_root = Path(root) / "tokenizer"
    if not tokenizer_root.is_dir() or tokenizer_root.is_symlink():
        raise ValueError("checkpoint tokenizer assets are missing or unsafe")
    for path in sorted(tokenizer_root.rglob("*")):
        if path.is_symlink():
            raise ValueError("checkpoint tokenizer assets cannot contain symlinks")
        if path.is_file():
            observed_tokenizer[path.relative_to(tokenizer_root).as_posix()] = (
                file_sha256(path)
            )
    if observed_tokenizer != dict(tokenizer_files):
        raise ValueError("checkpoint tokenizer asset ledger changed")
    portable_identity_seed = require_sha256(
        str(metadata.get("portable_identity_seed", "")), "portable_identity_seed"
    )
    runtime_ledger = metadata.get("runtime_ledger")
    if not isinstance(runtime_ledger, Mapping):
        raise ValueError("checkpoint runtime ledger is malformed")
    expected_runtime = {
        f"rank-{rank:05d}": ledger.get(f"runtime/rank-{rank:05d}.pt")
        for rank in range(int(identity.world_size))
    }
    if dict(runtime_ledger) != expected_runtime or any(
        (value is None for value in expected_runtime.values())
    ):
        raise ValueError("checkpoint runtime ledger does not bind every rank")
    require_sha256(portable_identity_seed, "portable_identity_seed")
    zero_paths = {name for name in ledger if name.startswith("zero1/")}
    if identity.optimizer_backend == "replicated_ddp":
        if metadata.get("zero1_manifest") is not None or zero_paths:
            raise ValueError("replicated checkpoint carries ZeRO resume files")
    else:
        manifest = metadata.get("zero1_manifest")
        if not isinstance(manifest, Mapping) or not zero_paths:
            raise ValueError("ZeRO checkpoint lacks its sharded resume subtree")
        from think_bridge.training.runtime_backend import (
            validate_sharded_resume_manifest,
        )

        validate_sharded_resume_manifest(
            manifest,
            expected_world_size=int(identity.world_size),
            checkpoint_dir=Path(root),
            sealed_file_ledger=ledger,
        )
        if manifest.get("artifact_sha256") != identity.resume_artifact_sha256:
            raise ValueError("ZeRO checkpoint manifest differs from resume identity")
    return identity


_CHECKPOINT_PAYLOAD_FIELDS = frozenset(
    {
        "identity",
        "owner",
        "owned_state",
        "sampler_state",
    }
)
_EVALUATION_SAMPLER_CONTRACT = {"route1": ("A", ROUTE1_OCCURRENCE_SAMPLER_SCHEMA)}


def validate_evaluation_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    route: str,
) -> tuple[BridgeCheckpointIdentity, Mapping[str, Any]]:
    """Validate the cheap, identity-bearing checkpoint surface used by eval."""

    if not isinstance(payload, Mapping):
        raise ValueError("evaluation checkpoint payload must be a mapping")
    missing = sorted(_CHECKPOINT_PAYLOAD_FIELDS.difference(payload))
    unknown = sorted(set(payload).difference(_CHECKPOINT_PAYLOAD_FIELDS))
    if missing or unknown:
        raise ValueError(
            "evaluation checkpoint payload schema mismatch; "
            f"missing={missing}, unknown={unknown}"
        )
    identity_raw = payload["identity"]
    if not isinstance(identity_raw, Mapping):
        raise ValueError("evaluation checkpoint identity must be a mapping")
    identity = BridgeCheckpointIdentity.from_mapping(identity_raw)
    if route not in _EVALUATION_SAMPLER_CONTRACT:
        raise ValueError("evaluation checkpoint route is unknown")
    expected_phase, expected_schema = _EVALUATION_SAMPLER_CONTRACT[route]
    if identity.phase != expected_phase:
        raise ValueError("evaluation checkpoint sampler route/phase mismatch")
    sampler_state = payload["sampler_state"]
    if not isinstance(sampler_state, Mapping):
        raise ValueError("evaluation checkpoint sampler_state must be a mapping")
    if identity.sampler_state_sha256 != canonical_json_sha256(sampler_state):
        raise ValueError("evaluation checkpoint sampler-state hash mismatch")
    if (
        sampler_state.get("artifact_type") != expected_schema
        or sampler_state.get("schema_version") != 1
    ):
        raise ValueError("evaluation checkpoint sampler schema mismatch")
    epoch = sampler_state.get("epoch")
    # Split-stage runs number epochs from zero; older combined runs offset D
    # by the R course. Route ownership is established by phase/schema above,
    # not by a hard-coded epoch range. Exact resume checks its full course.
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("evaluation checkpoint sampler epoch is outside its route")
    sampler_seed = sampler_state.get("seed")
    if (
        isinstance(sampler_seed, bool)
        or not isinstance(sampler_seed, int)
        or sampler_seed != int(identity.seed)
    ):
        raise ValueError("evaluation checkpoint sampler seed differs from identity")
    batch_index = sampler_state.get("global_batch_index")
    if (
        isinstance(batch_index, bool)
        or not isinstance(batch_index, int)
        or batch_index < -1
    ):
        raise ValueError("evaluation checkpoint sampler frontier is invalid")
    return identity, sampler_state
