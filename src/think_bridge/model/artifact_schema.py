"""Stable envelopes for persisted ThinkBridge artifacts."""

from __future__ import annotations

from typing import Any, Mapping


SCHEMA_VERSION = 1
STAGE_RESOLVED_CONFIG = "think-bridge.stage.resolved-config"
STAGE_ARTIFACT_LOCATOR = "think-bridge.stage.artifact-locator"
CHECKPOINT_CONFIG_REFERENCE = "think-bridge.checkpoint.config-reference"
CHECKPOINT_IDENTITY = "think-bridge.checkpoint.identity"
CHECKPOINT_DIRECTORY = "think-bridge.checkpoint.directory"
CHECKPOINT_COMPLETE = "think-bridge.checkpoint.complete"
STARTUP_INVARIANT_IDENTITY = "think-bridge.startup-invariant.identity"
STARTUP_INVARIANT_SEAL = "think-bridge.startup-invariant.seal"
RUNTIME_SIDECAR = "think-bridge.runtime.sidecar"
TOKENIZER_RUNTIME_SIDECAR = "think-bridge.runtime.tokenizer-sidecar"
TOKENIZER_RUNTIME_IDENTITY = "think-bridge.runtime.tokenizer-identity"
TRAINER_STATE = "think-bridge.training.state"
COMPLETED_EVALUATION = "think-bridge.evaluation.completed-record"
LIFECYCLE_EVENT = "think-bridge.training.lifecycle-event"
RESUME_ROLLBACK = "think-bridge.training.resume-rollback"
RESUME_AUDIT_EVENT = "think-bridge.training.resume-audit-event"
DISTRIBUTED_TRANSACTION = "think-bridge.distributed.transaction"
DISTRIBUTED_JSON_SHARD = "think-bridge.distributed.json-shard"
DISTRIBUTED_TENSOR_SHARD = "think-bridge.distributed.tensor-shard"
OCCURRENCE_SAMPLER_STATE = "think-bridge.training.occurrence-sampler-state"
ROUTE1_SELECTION = "think-bridge.selection.route1"
STAGE1_ARTIFACT_SUMMARY = "think-bridge.stage1.artifact-summary"
ACTIVE_OWNER_WEIGHTS = "think-bridge.checkpoint.active-owner-weights"
ACTIVE_OWNER_CONFIG = "think-bridge.checkpoint.active-owner-config"
RANK_RUNTIME = "think-bridge.checkpoint.rank-runtime"
ZERO1_OPTIMIZER_RESUME = "think-bridge.checkpoint.zero1-optimizer-resume"
ZERO1_SCHEDULER_RESUME = "think-bridge.checkpoint.zero1-scheduler-resume"
CHECKPOINT_TRAINER_STATE = "think-bridge.checkpoint.trainer-state"
STEP_AUDIT = "think-bridge.training.step-audit"
BACKEND_IDENTITY = "think-bridge.runtime.backend-identity"
DEEPSPEED_CHECKPOINT_SEMANTICS = "think-bridge.runtime.deepspeed-semantics"
BACKEND_COMPATIBILITY = "think-bridge.runtime.backend-compatibility"
ZERO1_RUNTIME_COMPATIBILITY = "think-bridge.checkpoint.zero1-runtime-compatibility"
ZERO1_PARTICIPANT = "think-bridge.checkpoint.zero1-participant"
ZERO1_MANIFEST = "think-bridge.checkpoint.zero1-manifest"
RANKED_RUNTIME_STATE = "think-bridge.checkpoint.ranked-runtime-state"
STAGE1_SHARED_MANIFEST = "think-bridge.stage1.shared-manifest"
STAGE1_TARGET_INDEX = "think-bridge.stage1.target-index"
STAGE1_HARD_DONOR_MANIFEST = "think-bridge.stage1.hard-donor-manifest"


def artifact_header(artifact_type: str) -> dict[str, Any]:
    """Return the required type/version fields for one persisted artifact."""

    if not isinstance(artifact_type, str) or not artifact_type.strip():
        raise ValueError("artifact_type must be a non-empty string")
    return {"artifact_type": artifact_type, "schema_version": SCHEMA_VERSION}


def require_artifact_header(
    payload: Mapping[str, Any],
    artifact_type: str,
    *,
    label: str,
) -> None:
    """Fail closed unless a payload carries the exact stable envelope."""

    if payload.get("artifact_type") != artifact_type:
        raise ValueError(f"{label} artifact_type mismatch")
    schema_version = payload.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != SCHEMA_VERSION
    ):
        raise ValueError(f"{label} schema_version must be integer 1")
