"""ThinkBridge R training with native supervision and checkpoint identities."""

from __future__ import annotations

from think_bridge.training.reasoner_fixed_state import (
    load_owner_fixed_state,
    restore_fixed_state,
    save_fixed_state,
)

import hashlib
import json
import math
import os
import os
from pathlib import Path
import random
import shutil
import sys
import time
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.distributed as torch_distributed
import torch.nn as nn

from think_bridge.model.artifact_schema import (
    ACTIVE_OWNER_CONFIG,
    ACTIVE_OWNER_WEIGHTS,
    CHECKPOINT_IDENTITY,
    CHECKPOINT_TRAINER_STATE,
    RANK_RUNTIME,
    STEP_AUDIT,
    ZERO1_OPTIMIZER_RESUME,
    ZERO1_SCHEDULER_RESUME,
    artifact_header,
)
from think_bridge.training.specificity_sampling import (
    SpecificityDonorSampler,
    SPECIFICITY_SAMPLING_POLICY,
    validate_donor_limit,
)
from think_bridge.model.checkpoint_policy import (
    SealedCheckpoint,
    BridgeCheckpointIdentity,
    checkpoint_artifact_sha256,
    checkpoint_building_path,
    checkpoint_path as route_checkpoint_path,
    checkpoint_run_contract_kind,
    parse_checkpoint_path,
    prepare_checkpoint_building_directory,
    read_checkpoint_pointer_target,
    seal_checkpoint_directory,
    validate_route1_gate_metric_schema,
    validate_checkpoint_directory,
    validate_checkpoint_directory_rank0,
    validate_checkpoint_seal,
    write_checkpoint_config_reference,
)
from think_bridge.model.training_config import TrainingConfig
from think_bridge.model.contract import (
    ANSWER_CAPACITY,
    CAUSAL_OBJECTIVE_SCHEMA_VERSION,
    CHECKPOINT_SCHEMA_VERSION,
    COURSE_SCHEMA_VERSION,
    D_ANSWER_EOS_RULE,
    D_ANSWER_PROVENANCE_SCHEMA,
    NEED_Z_COHORT_DEFINITION,
    OBJECTIVE_VERSION,
    SPECIFICITY_OBJECTIVE_VERSION,
    ROUTE1_VALIDATION_REPORT_SCHEMA_VERSION,
    canonical_json_sha256,
    file_sha256,
    is_bridge_isolated_path,
    normalize_route1_null_mode,
    require_manifest_fields,
    require_sha256,
    resolve_boundary_token_ids,
    route1_course_population_side,
    route1_course_reduction,
    validate_compiled_cot_capacity,
    validate_bridge_behavior_count,
    validate_route1_batch_geometry,
    validation_randomness_identity,
    write_atomic_json,
)
from think_bridge.training.timing import finalize_timing_sink, merge_timing_sink
from think_bridge.model.parallel_model import (
    Route1SpecificityStats,
    BridgeParallelModel,
    build_owner_optimizer,
    set_bridge_phase_ownership,
)
from think_bridge.training.progress import (
    BridgeStageReporter,
    BridgeLoggingWindow,
    TRAINING_WINDOW_METRICS,
    set_bridge_training_stage,
    suspend_bridge_progress,
    bridge_progress,
    bridge_training_stage,
    bridge_training_postfix,
)
from think_bridge.training.lifecycle_log import (
    LifecycleLogger,
    append_step_audit,
    reconcile_completed_evaluation_events,
)
from think_bridge.training.validation_reports import (
    canonical_validation_report_paths,
)
from think_bridge.training.distributed_protocol import (
    encode_route1_component_audit_reduction,
    finalize_route1_component_audit_reduction,
    finalize_route1_log_reduction,
    owner_z_vjp_views_strictly_aligned,
)
from think_bridge.training.occurrence_sampler import (
    CostBalancedRankAssignment,
    OccurrenceBatchSampler,
    OCCURRENCE_BATCH_UNIT,
    OCCURRENCE_WEIGHTING_UNIT,
    ROUTE1_OCCURRENCE_SAMPLER_SCHEMA,
    occurrence_steps_per_epoch,
)
from think_bridge.training.route1_preprocessing import (
    Route1NormalizedRowCache,
    exact_route1_optimizer_steps,
    resolve_route1_normalized_rows,
)
from think_bridge.training.objective_window_plans import (
    ROUTE1_ACTIVE_DOMAIN_SCHEMA,
    ROUTE1_PHYSICAL_COST_POLICY_SCHEMA,
    complete_optimizer_window_count,
    optimizer_course_clock,
    route1_active_physical_length_key,
)
from think_bridge.training.phase_runner import (
    ControlSchedule,
    PhaseRunner,
    build_long_lived_evaluation_arguments,
    training_evaluation_service_arguments,
    configure_control_evaluation_arguments,
    materialize_control_checkpoint,
    resume_evaluation_action,
    resume_sampler_epoch_end,
    resume_step_requires_evaluation,
)
from think_bridge.training.static_batch_prefetch import (
    DepthOneBatchPrefetch,
)
from think_bridge.training.runtime_backend import (
    broadcast_checkpoint_preflight,
    broadcast_rank0_result,
    checkpoint_backend_compatibility,
    collect_rank_failures,
    run_rank0_control,
    run_route1_accumulation_update,
)
from think_bridge.training.runtime_sidecars import (
    load_route_runtime_identity,
    load_runtime_tokenizer,
    tokenizer_identity_sha256,
)
from think_bridge.training.resume_audit import append_checkpoint_resume_event
from think_bridge.stage1.methods.bridge.recipe import (
    answer_course_geometry,
    phase_for_epoch,
    route1_curriculum_cut,
    route1_curriculum_value,
    route1_specificity_scale,
)

_BASE_RECORD_FIELDS = frozenset(
    {
        "record_id",
        "prompt_group_id",
        "problem_id",
        "semantic_group_id",
        "self_cot_id",
        "self_cot_source_rollout_id",
        "provenance_valid",
        "prompt_ids",
        "direct_prompt_ids",
        "cot_ids",
        "readout_cot_ids",
        "route2_eligible",
        "answer_ids",
        "answer_source",
        "split_id",
        "population",
        "n_correct",
        "need_z",
        "direct_correct",
        "locked_full_correct",
        "need_z_definition",
    }
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} is not an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"prepared record file is empty: {path}")
    return rows


def _validate_token_ids(
    value: Any, field: str, *, permit_empty: bool = False
) -> list[int]:
    if not isinstance(value, list) or (not value and not permit_empty):
        raise ValueError(f"{field} must be a non-empty token-id list")
    if any(not isinstance(token, int) or token < 0 for token in value):
        raise ValueError(f"{field} contains an invalid token id")
    return value


def validate_prepared_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    route: str,
    expected_split: str = "train",
    z_width: int | None = None,
    eos_token_id: int | None = None,
    z_finite_prevalidated: bool = False,
    route2_content_capacity: int,
) -> None:
    if route not in {"base", "route1"}:
        raise ValueError("prepared-record validation route is invalid")
    if (
        isinstance(route2_content_capacity, bool)
        or not isinstance(route2_content_capacity, int)
        or route2_content_capacity <= 0
    ):
        raise ValueError("Route2 content capacity must be a positive integer")
    identities: set[str] = set()
    for index, row in enumerate(rows):
        missing = sorted(_BASE_RECORD_FIELDS.difference(row))
        if missing:
            raise ValueError(f"record {index} missing fields: {missing}")
        if row["provenance_valid"] is not True:
            raise ValueError(f"record {index} provenance is not sealed valid")
        record_id = str(row["record_id"])
        if not record_id or record_id in identities:
            raise ValueError("record ids must be unique and non-empty")
        identities.add(record_id)
        for field in ("prompt_group_id", "problem_id", "semantic_group_id", "split_id"):
            if not isinstance(row[field], str) or not row[field]:
                raise ValueError(f"record {index} has an invalid {field}")
        if row["split_id"] != expected_split:
            raise ValueError(
                f"record {index} split is {row['split_id']!r}; expected {expected_split!r}"
            )
        _validate_token_ids(row["prompt_ids"], "prompt_ids")
        _validate_token_ids(row["direct_prompt_ids"], "direct_prompt_ids")
        population = row["population"]
        if population not in {"native", "d_answer_fallback", "reference"}:
            raise ValueError("prepared population is invalid")
        direct_only = population == "d_answer_fallback"
        unknown_validation_cot = (
            expected_split == "validation" and row.get("cot_length_known") is False
        )
        cot_ids = _validate_token_ids(
            row["cot_ids"],
            "cot_ids",
            permit_empty=direct_only or expected_split == "validation",
        )
        readout_cot_ids = _validate_token_ids(
            row["readout_cot_ids"],
            "readout_cot_ids",
            permit_empty=direct_only or unknown_validation_cot,
        )
        answer_ids = _validate_token_ids(row["answer_ids"], "answer_ids")
        if not isinstance(row["answer_source"], str) or not row["answer_source"]:
            raise ValueError("prepared answer source is not provenance-bound")
        answer_only = expected_split == "train" and row.get("answer_source") in {
            "explicit-answer-only-boxed-reference",
            "explicit-answer-only-selfgen",
            "dataset-reference-cot-and-answer",
        }
        n_correct = row["n_correct"]
        if answer_only:
            if (
                direct_only
                or n_correct is not None
                or any(
                    (
                        row[field] is not None
                        for field in ("need_z", "direct_correct", "locked_full_correct")
                    )
                )
                or (
                    row["answer_source"]
                    not in {
                        "explicit-answer-only-boxed-reference",
                        "explicit-answer-only-selfgen",
                        "dataset-reference-cot-and-answer",
                    }
                )
            ):
                raise ValueError(
                    "prepared answer-only row forged Stage0 behavior labels"
                )
        else:
            if not all(
                (
                    isinstance(row[field], bool)
                    for field in ("need_z", "direct_correct", "locked_full_correct")
                )
            ):
                raise ValueError("prepared population labels must be explicit booleans")
            if bool(row["need_z"]) != (
                bool(row["locked_full_correct"]) and (not bool(row["direct_correct"]))
            ):
                raise ValueError("prepared direct/no-think need-z cohort differs")
            try:
                validate_bridge_behavior_count(
                    n_correct,
                    split=expected_split,
                    locked_full_correct=row["locked_full_correct"],
                )
            except ValueError as exc:
                raise ValueError(
                    "prepared n_correct differs from locked native label"
                ) from exc
        if row["need_z_definition"] != NEED_Z_COHORT_DEFINITION:
            raise ValueError("prepared need-z cohort definition differs")
        if direct_only:
            if (
                expected_split != "train"
                or n_correct != 0
                or row["direct_correct"] is not True
                or (row["need_z"] is not False)
                or cot_ids
                or readout_cot_ids
                or (row["route2_eligible"] is not False)
                or (row["self_cot_id"] is not None)
                or (row["self_cot_source_rollout_id"] is not None)
            ):
                raise ValueError("prepared direct-only population contract differs")
        elif not unknown_validation_cot:
            for field in ("self_cot_id", "self_cot_source_rollout_id"):
                if not isinstance(row[field], str) or not row[field]:
                    raise ValueError(f"record {index} has an invalid {field}")
            if readout_cot_ids[:-1] != cot_ids:
                raise ValueError(
                    "readout CoT must be the exact source CoT plus one terminal token"
                )
        route2_eligible = row["route2_eligible"]
        if not isinstance(route2_eligible, bool):
            raise ValueError("route2_eligible must be boolean")
        if unknown_validation_cot:
            if cot_ids or readout_cot_ids or route2_eligible:
                raise ValueError(
                    "unknown validation CoT cannot supply a readout target"
                )
        elif not direct_only:
            validate_compiled_cot_capacity(
                content_token_count=len(cot_ids),
                readout_token_count=len(readout_cot_ids),
                route2_eligible=route2_eligible,
                content_capacity=route2_content_capacity,
            )
        if expected_split == "train" and len(answer_ids) > ANSWER_CAPACITY:
            raise ValueError(
                "prepared native self-answer exceeds the 2048-token horizon"
            )


def _pad_token_rows_cpu(
    rows: Sequence[Sequence[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = max(len(row) for row in rows)
    ids = torch.zeros((len(rows), maximum), dtype=torch.long)
    mask = torch.zeros((len(rows), maximum), dtype=torch.bool)
    for index, row in enumerate(rows):
        length = len(row)
        ids[index, :length] = torch.as_tensor(row, dtype=torch.long)
        mask[index, :length] = True
    return ids, mask


def _pad_optional_token_rows_cpu(
    rows: Sequence[Sequence[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not rows:
        return (
            torch.zeros((0, 1), dtype=torch.long),
            torch.zeros((0, 1), dtype=torch.bool),
        )
    return _pad_token_rows_cpu(rows)


_ROUTE1_FORWARD_TENSOR_FIELDS = frozenset(
    {
        "prompt_ids",
        "prompt_mask",
        "direct_prompt_ids",
        "direct_prompt_mask",
        "course_cot_ids",
        "course_cot_mask",
        "distill_cot_ids",
        "distill_cot_mask",
        "course_answer_ids",
        "course_answer_mask",
        "course_view_to_sample",
        "course_view_to_course_sample",
        "course_sample_side",
        "c_native_cot_ids",
        "c_native_cot_mask",
        "c_view_to_sample",
        "c_view_to_c_sample",
        "c_sample_indices",
        "distill_row_weights",
        "wrong_candidate_mask",
        "specificity_positive_mask",
        "specificity_donor_prompt_ids",
        "specificity_donor_prompt_mask",
    }
)
_ROUTE1_FORWARD_BATCH_FIELDS = _ROUTE1_FORWARD_TENSOR_FIELDS | frozenset(
    {
        "answer_course_geometry",
        "course_reduction",
        "global_course_c_sample_count",
        "global_course_direct_side_sample_count",
        "global_match_sample_count",
        "global_specific_sample_count",
        "course_weight",
        "match_weight",
        "specific_weight",
        "specificity_tau",
        "physical_chunk_size",
        "wrong_control_chunk_size",
    }
)
_ROUTE1_PREPARED_TELEMETRY_FIELDS = frozenset(
    {
        "curriculum",
        "course_cut_sum",
        "course_remaining_token_count",
        "course_target_token_count",
    }
)
_ROUTE1_PREPARED_BATCH_FIELDS = (
    _ROUTE1_FORWARD_BATCH_FIELDS
    | _ROUTE1_PREPARED_TELEMETRY_FIELDS
    | frozenset({"_pinned_memory"})
)


def _training_compute_context(device: Any) -> Any:
    """Use the same compute precision before and inside the public forward."""
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    )


def _prepare_route1_microstep_for_training(
    model: Any, arguments: Mapping[str, Any], **kwargs: Any
) -> dict[str, Any]:
    # Prefetch bypasses train_model's (and DeepSpeed's) forward autocast scope.
    # Keep FP32 owners, but materialize the exact same BF16-compute R graph as
    # non-prefetched training and selected-R inference on CUDA.
    with _training_compute_context(arguments["prompt_ids"].device):
        return model.prepare_route1_microstep(arguments, **kwargs)


def _route1_execution_course_side(quadrant: str, epoch: int) -> str | None:
    """Resolve staged quadrants plus the label-free answer-only population."""

    if quadrant == "G":
        return "c"
    if quadrant == "A":
        return "direct_correct_side"
    return route1_course_population_side(quadrant, epoch)


def _prepare_route1_batch_cpu(
    rows: Sequence[Mapping[str, Any]],
    *,
    epoch: int,
    optimizer_step: int,
    optimizer_steps_per_epoch: int,
    global_course_c_sample_count: int,
    global_course_direct_side_sample_count: int,
    global_match_sample_count: int,
    global_specific_sample_count: int,
    global_native_b_sample_count: int,
    global_native_c_sample_count: int,
    distillation_populations: Sequence[str] = ("C",),
    wrong_candidate_mask: Sequence[Sequence[bool]],
    specificity_donor_rows: Sequence[Mapping[str, Any]] = (),
    physical_chunk_size: int,
    wrong_control_chunk_size: int,
    pin_memory: bool,
    course_weight: float,
    match_weight: float,
    specific_weight: float,
    specificity_tau: float,
    seed: int = 42,
    course_epochs: float = 0.0,
    course_updates: int | None = None,
) -> dict[str, Any]:
    """Collate B/C/D samples and the optimizer-window donor prompt bank."""

    if not rows or len({str(row["record_id"]) for row in rows}) != len(rows):
        raise RuntimeError("Route1 microbatch requires distinct record identities")
    if any(row.get("quadrant") not in {"A", "B", "C", "D", "G"} for row in rows):
        raise ValueError("Route1 sample batch contains a non-training quadrant")
    prompt_ids, prompt_mask = _pad_token_rows_cpu([row["prompt_ids"] for row in rows])
    direct_prompt_ids, direct_prompt_mask = _pad_token_rows_cpu(
        [row["direct_prompt_ids"] for row in rows]
    )
    curriculum = route1_curriculum_value(
        int(optimizer_step),
        int(optimizer_steps_per_epoch),
        course_epochs=course_epochs,
        course_steps=course_updates,
    )
    geometry = "deployed_z_cot_suffix" if curriculum < 1.0 else "deployed_z"
    specific_weight = float(specific_weight) * route1_specificity_scale(
        int(optimizer_step),
        int(optimizer_steps_per_epoch),
        course_epochs=course_epochs,
        course_steps=course_updates,
    )
    course_cot_rows: list[list[int]] = []
    course_answer_rows: list[list[int]] = []
    course_view_to_sample: list[int] = []
    course_view_to_course_sample: list[int] = []
    course_sample_side: list[int] = []
    course_cuts: list[int] = []
    c_cot_rows: list[list[int]] = []
    distill_cot_rows: list[list[int]] = []
    c_view_to_sample: list[int] = []
    c_view_to_c_sample: list[int] = []
    c_sample_indices: list[int] = []
    c_objectives_active = (
        int(global_match_sample_count) > 0 or int(global_specific_sample_count) > 0
    )
    active_populations = frozenset(
        str(value).upper()
        for value in (
            distillation_populations
            if not isinstance(distillation_populations, str)
            else tuple(distillation_populations)
        )
    )
    if active_populations not in {
        frozenset({"C"}),
        frozenset({"B", "C"}),
        frozenset({"G"}),
    }:
        raise ValueError("distillation_populations must select C or B+C")
    counts = (global_native_b_sample_count, global_native_c_sample_count)
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in counts):
        raise ValueError("native population counts must be nonnegative integers")
    population_counts = {
        "B": int(global_native_b_sample_count),
        "C": int(global_native_c_sample_count),
        "G": int(global_native_c_sample_count),
    }
    native_total = sum(population_counts[side] for side in active_populations)
    if (
        (match_weight > 0 or specific_weight > 0)
        and native_total == 0
        and any(row["quadrant"] in active_populations for row in rows)
    ):
        raise ValueError("local native owner has no global population count")
    if global_match_sample_count != (native_total if match_weight > 0 else 0):
        raise ValueError("Match count differs from its configured population")
    if not 0 <= global_specific_sample_count <= global_native_c_sample_count:
        raise ValueError("Specificity count exceeds eligible C/reference owners")
    if specific_weight <= 0 and global_specific_sample_count != 0:
        raise ValueError("inactive Specificity has a nonzero count")
    distill_row_weights: list[float] = []
    compute_course = float(course_weight) > 0.0
    if not compute_course and (
        int(global_course_c_sample_count) != 0
        or int(global_course_direct_side_sample_count) != 0
    ):
        raise ValueError("zero-weight Route1 course received active course counts")
    for sample_index, row in enumerate(rows):
        views = row.get("answer_views")
        if not isinstance(views, list) or len(views) != 1:
            raise ValueError("Route1 sample must carry exactly one legal answer view")
        c_sample_index = -1
        if row["quadrant"] in active_populations and c_objectives_active:
            c_sample_index = len(c_sample_indices)
            c_sample_indices.append(sample_index)
            side_count = population_counts[row["quadrant"]]
            if side_count <= 0:
                raise ValueError("local native owner has no global population count")
            distill_row_weights.append(1.0 / native_total)
        course_side = (
            _route1_execution_course_side(row["quadrant"], int(epoch))
            if compute_course
            else None
        )
        course_active = course_side is not None
        course_sample_index = -1
        if course_active:
            course_sample_index = len(course_sample_side)
            course_sample_side.append(0 if course_side == "c" else 1)
        for view in views:
            kind = view.get("kind")
            cot = [int(token) for token in view.get("cot_ids", ())]
            answer = [int(token) for token in view.get("answer_ids", ())]
            if not answer or kind not in {
                "answer_only",
                "native",
                "direct",
                "reference",
            }:
                raise ValueError("Route1 answer view is malformed")
            if kind == "direct":
                if row["quadrant"] != "D" or cot:
                    raise ValueError("only D may carry one suffix-free direct view")
                cut = 0
            else:
                if row["quadrant"] not in {"B", "C", "G"} or not cot:
                    raise ValueError("native course view lacks a paired CoT")
                cut = (
                    route1_curriculum_cut(
                        len(cot),
                        curriculum=float(curriculum),
                        optimizer_step=int(optimizer_step),
                        prompt_group_key=str(row["prompt_group_id"]),
                        prompt_ids=list(row["prompt_ids"]),
                        seed=int(seed),
                    )
                    if course_active or c_objectives_active
                    else 0
                )
            retained_cot = cot[cut:]
            if course_active:
                course_cot_rows.append(retained_cot)
                course_answer_rows.append(answer)
                course_view_to_sample.append(sample_index)
                course_view_to_course_sample.append(course_sample_index)
                course_cuts.append(cut)
            if row["quadrant"] in active_populations and c_objectives_active:
                expected_kind = "reference" if row["quadrant"] == "G" else "native"
                if kind != expected_kind:
                    raise ValueError(
                        "distillation view source disagrees with its population"
                    )
                c_cot_rows.append(cot)
                distill_cot_rows.append([])
                c_view_to_sample.append(sample_index)
                c_view_to_c_sample.append(c_sample_index)
    course_cot_ids, course_cot_mask = _pad_optional_token_rows_cpu(course_cot_rows)
    course_answer_ids, course_answer_mask = _pad_optional_token_rows_cpu(
        course_answer_rows
    )
    c_native_cot_ids, c_native_cot_mask = _pad_optional_token_rows_cpu(c_cot_rows)
    distill_cot_ids, distill_cot_mask = _pad_optional_token_rows_cpu(distill_cot_rows)
    candidate_mask = torch.as_tensor(wrong_candidate_mask, dtype=torch.bool)
    if candidate_mask.ndim != 2 or candidate_mask.size(0) != len(rows):
        raise ValueError("Route1 wrong candidate mask must be [local,global]")
    if specificity_donor_rows and int(candidate_mask.size(1)) != len(
        specificity_donor_rows
    ):
        raise ValueError(
            "Route1 wrong candidate mask width differs from optimizer-window donor bank"
        )
    specificity_donor_prompt_ids, specificity_donor_prompt_mask = (
        _pad_optional_token_rows_cpu(
            [row["prompt_ids"] for row in specificity_donor_rows]
        )
    )
    positive_mask = torch.tensor(
        [
            [
                str(owner["record_id"]) != str(donor["record_id"])
                and str(owner["prompt_group_id"]) == str(donor["prompt_group_id"])
                for donor in specificity_donor_rows
            ]
            for owner in rows
        ],
        dtype=torch.bool,
    ).reshape(len(rows), len(specificity_donor_rows))
    if positive_mask.shape == candidate_mask.shape and bool(
        (positive_mask & candidate_mask).any()
    ):
        raise ValueError("same-prompt positive cannot also be a wrong donor")
    tensors = {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "direct_prompt_ids": direct_prompt_ids,
        "direct_prompt_mask": direct_prompt_mask,
        "course_cot_ids": course_cot_ids,
        "course_cot_mask": course_cot_mask,
        "distill_cot_ids": distill_cot_ids,
        "distill_cot_mask": distill_cot_mask,
        "course_answer_ids": course_answer_ids,
        "course_answer_mask": course_answer_mask,
        "course_view_to_sample": torch.tensor(course_view_to_sample, dtype=torch.long),
        "course_view_to_course_sample": torch.tensor(
            course_view_to_course_sample, dtype=torch.long
        ),
        "course_sample_side": torch.tensor(course_sample_side, dtype=torch.long),
        "c_native_cot_ids": c_native_cot_ids,
        "c_native_cot_mask": c_native_cot_mask,
        "c_view_to_sample": torch.tensor(c_view_to_sample, dtype=torch.long),
        "c_view_to_c_sample": torch.tensor(c_view_to_c_sample, dtype=torch.long),
        "c_sample_indices": torch.tensor(c_sample_indices, dtype=torch.long),
        "distill_row_weights": torch.tensor(distill_row_weights, dtype=torch.float32),
        "wrong_candidate_mask": candidate_mask,
        "specificity_positive_mask": positive_mask,
        "specificity_donor_prompt_ids": specificity_donor_prompt_ids,
        "specificity_donor_prompt_mask": specificity_donor_prompt_mask,
    }
    pinned = False
    if pin_memory:
        try:
            tensors = {name: tensor.pin_memory() for name, tensor in tensors.items()}
            pinned = True
        except RuntimeError:
            # Pinned allocation is an optional transport optimization.  A
            # missing accelerator allocator keeps the exact pageable path.
            pinned = False
    return {
        **tensors,
        "_pinned_memory": pinned,
        "answer_course_geometry": geometry,
        "curriculum": float(curriculum),
        "course_cut_sum": int(sum(course_cuts)),
        "course_remaining_token_count": int(sum(map(len, course_cot_rows))),
        "course_target_token_count": int(
            sum(map(len, course_answer_rows)) + sum(map(len, course_cot_rows))
        ),
        "course_reduction": route1_course_reduction(int(epoch)),
        "global_course_c_sample_count": int(global_course_c_sample_count),
        "global_course_direct_side_sample_count": int(
            global_course_direct_side_sample_count
        ),
        "global_match_sample_count": int(global_match_sample_count),
        "global_specific_sample_count": int(global_specific_sample_count),
        "course_weight": float(course_weight),
        "match_weight": float(match_weight),
        "specific_weight": float(specific_weight),
        "specificity_tau": float(specificity_tau),
        "physical_chunk_size": int(physical_chunk_size),
        "wrong_control_chunk_size": int(wrong_control_chunk_size),
    }


def _route1_batch_to_device(
    prepared: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    observed_fields = set(prepared)
    missing_fields = sorted(_ROUTE1_PREPARED_BATCH_FIELDS - observed_fields)
    unexpected_fields = sorted(observed_fields - _ROUTE1_PREPARED_BATCH_FIELDS)
    if missing_fields or unexpected_fields:
        raise ValueError(
            "Route1 prepared batch field contract differs; "
            f"missing={missing_fields}, unexpected={unexpected_fields}"
        )
    pinned = bool(prepared["_pinned_memory"])
    non_blocking = bool(pinned and device.type == "cuda")
    return {
        name: (
            value.to(device=device, non_blocking=non_blocking)
            if name in _ROUTE1_FORWARD_TENSOR_FIELDS
            else value
        )
        for name, value in prepared.items()
        if name in _ROUTE1_FORWARD_BATCH_FIELDS
    }


def _distributed_context() -> tuple[int, int, torch.device]:
    from think_bridge.training.distributed_artifacts import initialize_distributed_gpu

    rank, world_size, _, device = initialize_distributed_gpu()
    if torch_distributed.is_initialized():
        world_size = torch_distributed.get_world_size()
        rank = torch_distributed.get_rank()
    return world_size, rank, device


def _head_count(width: int) -> int:
    for candidate in (16, 8, 4, 2, 1):
        if width % candidate == 0:
            return candidate
    raise RuntimeError("executor width has no legal attention head count")


def _build_model(
    config: TrainingConfig,
    manifest: Mapping[str, Any],
    *,
    route: str,
    device: torch.device,
    runtime_sidecar_dir: Path | None = None,
    no_progress: bool = False,
) -> tuple[nn.Module, Any]:
    if route not in {"route1"}:
        raise ValueError("Bridge model route must be route1")
    if runtime_sidecar_dir is None:
        raise ValueError(
            "model construction requires a sealed frozen executor identity"
        )
    from think_bridge.model.executor_identity import assert_executor_identity

    frozen_identity = load_route_runtime_identity(runtime_sidecar_dir, route=route)[
        "frozen_executor_identity"
    ]
    if runtime_sidecar_dir is None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required on the GPU server") from exc
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path)
    else:
        tokenizer = load_runtime_tokenizer(runtime_sidecar_dir)
    boundary_ids = resolve_boundary_token_ids(tokenizer, config.boundary_text)
    if list(boundary_ids) != list(manifest["boundary_token_ids"]):
        raise RuntimeError(
            "runtime tokenizer boundary ids differ from the manifest seal"
        )
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("transformers is required on the GPU server") from exc
    executor_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    executor = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=executor_dtype,
        attn_implementation=config.attn_implementation,
    ).to(device)
    assert_executor_identity(executor, frozen_identity, no_progress=no_progress)
    reasoner = BridgeParallelModel.build_reasoner(
        executor,
        latent_steps=int(config.latent_steps),
        latents_per_step=int(config.latents_per_step),
        loop_steps=int(config.reasoner_loop_steps),
        num_layers=int(config.emitter_depth),
        num_heads=_head_count(int(executor.config.hidden_size)),
        dim_feedforward=4 * int(executor.config.hidden_size),
        bound_scale_init=float(config.bound_scale_init),
        zero_residual_init=bool(config.reasoner_zero_init),
        output_normalization=str(config.reasoner_output_normalization),
        input_mode=config.reasoner_input_mode,
        dropout_p=float(config.reasoner_dropout_p),
        dropout_views=int(config.reasoner_dropout_views),
    )
    model = BridgeParallelModel(
        executor=executor,
        reasoner=reasoner,
        alignment_capacity=int(manifest["alignment_capacity"]),
        boundary_ids=torch.tensor(boundary_ids, dtype=torch.long, device=device),
        eos_token_id=int(tokenizer.eos_token_id),
        method=config.method,
        vocab_chunk_size=config.lm_head_vocab_chunk_size,
        trajectory_max_steps=config.trajectory_max_steps,
        route1_generation_temperature=config.route1_generation_temperature,
        route1_specificity_wrong_gradient=config.route1_specificity_wrong_gradient,
        route1_specificity_include_direct=config.route1_specificity_include_direct,
        route1_specificity_loss=config.route1_specificity_loss,
        route1_specificity_margin=config.route1_specificity_margin,
        route1_specificity_temperature=config.route1_specificity_temperature,
        route1_specificity_negative_kl_cap=config.route1_specificity_negative_kl_cap,
        generation_seed=config.generation_seed,
        route1_gradient_checkpointing=config.route1_gradient_checkpointing,
    ).to(device)
    from think_bridge.model.feedback_precision import configure_feedback

    configure_feedback(model, config)
    model._frozen_executor_identity = frozen_identity
    _validate_loaded_runtime(model, tokenizer, config=config, manifest=manifest)
    return (model, tokenizer)


def _tensor_tree_sha256(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(json.dumps(list(tensor.shape)).encode())
            # Optimizer state contains 0-D tensors (for example Adam's
            # scalar step).  A dtype-changing view requires at least one
            # dimension, so flatten first while preserving the raw bytes.
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, Mapping):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def _state_schema_sha256(module: nn.Module) -> str:
    schema = [
        (name, list(value.shape), str(value.dtype))
        for name, value in module.named_parameters()
        if value.requires_grad
    ]
    if not schema:
        raise ValueError("active owner has no trainable parameter schema")
    return canonical_json_sha256(schema)


def _tokenizer_sha256(tokenizer: Any) -> str:
    return tokenizer_identity_sha256(tokenizer)


def _architecture_sha256(
    config: TrainingConfig, *, component: str, hidden_size: int
) -> str:
    value = {
        "name": f"bridge-recurrent-feedback-emitter-t{config.latent_steps}-b{config.latents_per_step}-k{config.latent_slots}",
        "hidden_size": hidden_size,
        "latent_steps": int(config.latent_steps),
        "latents_per_step": int(config.latents_per_step),
        "emitter_depth": int(config.emitter_depth),
        "tap_count": int(config.tap_count),
        "full_bptt": True,
        "real_frozen_f_appends": int(config.real_frozen_f_appends),
        "bound": "rms-radial-bound",
        "bound_scale_init": config.bound_scale_init,
        "num_heads": _head_count(hidden_size),
        "dim_feedforward": 4 * hidden_size,
        "owner_dtype": "float32",
    }
    value["input_mode"] = config.reasoner_input_mode
    if config.reasoner_loop_steps != 1:
        value["loop_steps"] = int(config.reasoner_loop_steps)
    return canonical_json_sha256(value)


def _validate_loaded_runtime(
    model: BridgeParallelModel,
    tokenizer: Any,
    *,
    config: TrainingConfig,
    manifest: Mapping[str, Any],
) -> None:
    hidden = int(model.executor.config.hidden_size)
    observed = {
        "tokenizer_sha256": _tokenizer_sha256(tokenizer),
        "template_sha256": canonical_json_sha256(tokenizer.chat_template or ""),
        "boundary_ids_sha256": canonical_json_sha256(
            [int(value) for value in model.boundary_ids.detach().cpu().tolist()]
        ),
    }
    mismatched = [
        field for field, value in observed.items() if value != manifest[field]
    ]
    if mismatched:
        raise ValueError(
            f"runtime model/tokenizer/template/architecture hash mismatch: {mismatched}"
        )
    if int(manifest["z_width"]) != hidden:
        raise ValueError("manifest dF differs from the loaded executor width")
    if int(model.boundary_ids.numel()) != int(manifest["boundary_token_count"]):
        raise ValueError("runtime boundary token count differs from manifest")


def _owner_module(model: nn.Module, owner: str) -> nn.Module:
    return model.reasoner


def _fp32_owner_state(state: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Materialize one portable CPU FP32 active-owner tensor mapping."""

    tensors: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if not isinstance(name, str) or not name or not torch.is_tensor(value):
            raise TypeError("active-owner state must contain named tensors only")
        if not value.is_floating_point() or value.dtype != torch.float32:
            raise TypeError(f"active-owner checkpoint tensor is not FP32: {name}")
        tensors[name] = value.detach().to(device="cpu").contiguous()
    if not tensors:
        raise ValueError("active-owner checkpoint state is empty")
    return tensors


def _owner_checkpoint_state(module: nn.Module) -> dict[str, torch.Tensor]:
    """Return only FP32 trainable parameters owned by the active route."""

    return _fp32_owner_state(
        {
            name: parameter
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }
    )


def _frozen_owner_parameter_names(module: nn.Module) -> tuple[str, ...]:
    return tuple(
        sorted(
            name
            for name, parameter in module.named_parameters()
            if not parameter.requires_grad
        )
    )


def _checkpoint_excluded_frozen_parameter_names(
    model: nn.Module, *, owner: str
) -> tuple[str, ...]:
    return _frozen_owner_parameter_names(_owner_module(model, owner))


def _restore_owner_checkpoint_state(
    module: nn.Module, state: Mapping[str, Any]
) -> None:
    """Strictly restore the trainable subset and retain rebuilt frozen assets."""

    expected_parameters = {
        name: parameter
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    if not expected_parameters:
        raise ValueError("active owner has no trainable checkpoint parameters")
    if set(state) != set(expected_parameters):
        missing = sorted(set(expected_parameters).difference(state))
        unexpected = sorted(set(state).difference(expected_parameters))
        raise ValueError(
            "active-owner trainable checkpoint schema mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for name, parameter in expected_parameters.items():
        value = state[name]
        if (
            not torch.is_tensor(value)
            or value.dtype != torch.float32
            or tuple(value.shape) != tuple(parameter.shape)
        ):
            raise ValueError(
                f"active-owner trainable checkpoint tensor mismatch: {name}"
            )
    incompatible = module.load_state_dict(dict(state), strict=False)
    expected_missing = set(module.state_dict()).difference(expected_parameters)
    if (
        set(incompatible.missing_keys) != expected_missing
        or incompatible.unexpected_keys
    ):
        raise ValueError("active-owner subset restore did not preserve rebuilt state")


def _save_owner_safetensors(
    path: Path,
    state: Mapping[str, Any],
    *,
    route: str,
    owner: str,
    identity: BridgeCheckpointIdentity,
    excluded_frozen_parameter_names: Sequence[str],
) -> None:
    from safetensors.torch import save_file

    save_file(
        _fp32_owner_state(state),
        str(path),
        metadata={
            "artifact_type": ACTIVE_OWNER_WEIGHTS,
            "objective_version": OBJECTIVE_VERSION,
            "specificity_objective_version": SPECIFICITY_OBJECTIVE_VERSION,
            "route": route,
            "owner": owner,
            "phase": identity.phase,
            "method": identity.method,
            "step": str(identity.step),
            "owned_state_sha256": identity.owned_state_sha256,
            "contains_frozen_executor": "false",
            "tensor_scope": "active_owner_trainable_parameters_only",
            "contains_frozen_parameters": "false",
            "excluded_frozen_parameter_count": str(
                len(tuple(excluded_frozen_parameter_names))
            ),
        },
    )


def _load_owner_safetensors(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    state = dict(load_file(str(Path(path) / "model.safetensors"), device="cpu"))
    if not state:
        raise ValueError("active-owner model.safetensors is empty")
    for name, tensor in state.items():
        if not torch.is_tensor(tensor) or tensor.dtype != torch.float32:
            raise TypeError(f"active-owner safetensors value is not FP32: {name}")
    return state


def _write_checkpoint_portable_assets(
    building: Path,
    *,
    run_dir: Path,
    route: str,
    owner: str,
    identity: BridgeCheckpointIdentity,
    excluded_frozen_parameter_names: Sequence[str],
) -> None:
    contract_kind = checkpoint_run_contract_kind(run_dir)
    runtime_index = _read_json(run_dir / "runtime_sidecars.json")
    tokenizer_relative = Path(str(runtime_index.get("tokenizer_path", "")))
    if tokenizer_relative.is_absolute() or ".." in tokenizer_relative.parts:
        raise ValueError("checkpoint tokenizer source escaped the run directory")
    tokenizer_source = (run_dir / tokenizer_relative).resolve(strict=True)
    try:
        tokenizer_source.relative_to(run_dir.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(
            "checkpoint tokenizer source escaped the run directory"
        ) from exc
    if contract_kind != "split" or not tokenizer_source.is_dir():
        raise FileNotFoundError(
            "checkpoint portable run contract/tokenizer assets are missing"
        )
    tokenizer_ledger: dict[str, str] = {}
    for source in sorted(tokenizer_source.rglob("*")):
        if source.is_symlink():
            raise ValueError("checkpoint tokenizer source cannot contain symlinks")
        if source.is_file():
            tokenizer_ledger[source.relative_to(tokenizer_source).as_posix()] = (
                file_sha256(source)
            )
    if not tokenizer_ledger:
        raise ValueError("checkpoint tokenizer source is empty")
    frozen_names = tuple(
        sorted((str(name) for name in excluded_frozen_parameter_names))
    )
    expected_frozen_names = {"route1": ()}.get(route)
    if expected_frozen_names is None:
        raise ValueError("checkpoint portable route is invalid")
    if frozen_names != expected_frozen_names:
        raise ValueError("checkpoint excluded-frozen parameter schema mismatch")
    frozen_source = None
    shutil.copytree(
        tokenizer_source, building / "tokenizer", copy_function=shutil.copy2
    )
    write_atomic_json(
        building / "active_owner_config.json",
        {
            **artifact_header(ACTIVE_OWNER_CONFIG),
            "model_type": "think_bridge_active_owner",
            "objective_version": OBJECTIVE_VERSION,
            "specificity_objective_version": SPECIFICITY_OBJECTIVE_VERSION,
            "route": route,
            "phase": identity.phase,
            "method": identity.method,
            "owner": owner,
            "step": int(identity.step),
            "weight_file": "model.safetensors",
            "weight_format": "safetensors",
            "tensor_dtype": "float32",
            "contains_frozen_executor": False,
            "tensor_scope": "active_owner_trainable_parameters_only",
            "contains_frozen_parameters": False,
            "excluded_frozen_parameter_names": list(frozen_names),
            "frozen_parameter_source": frozen_source,
            "model_state_schema_sha256": identity.model_state_schema_sha256,
            "owned_state_sha256": identity.owned_state_sha256,
            "exact_resume_identity_sha256": identity.exact_resume_identity_sha256,
            "selected_r_sha256": identity.selected_r_sha256,
            "tokenizer_sha256": identity.tokenizer_sha256,
            "tokenizer_files": tokenizer_ledger,
        },
        replace_mismatch=False,
    )


class _WarmupCosine:
    def __init__(
        self, optimizer: torch.optim.Optimizer, *, warmup: int, total: int
    ) -> None:
        if total <= 0 or warmup < 0:
            raise ValueError("scheduler update geometry is invalid")
        self.warmup = int(warmup)
        self.total = int(total)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self._factor)

    def _factor(self, step: int) -> float:
        if self.warmup and step < self.warmup:
            return float(step + 1) / float(self.warmup)
        span = max(self.total - self.warmup, 1)
        progress = min(max((step - self.warmup) / span, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def step(self) -> None:
        self.scheduler.step()

    def rebase_current_step(self, step: int) -> None:
        """Apply the new horizon before the first resumed optimizer update."""
        if self.scheduler.last_epoch != step:
            raise ValueError("restored scheduler step differs from checkpoint update")
        rates = [base * self._factor(step) for base in self.scheduler.base_lrs]
        for group, rate in zip(self.scheduler.optimizer.param_groups, rates):
            group["lr"] = rate
        self.scheduler._last_lr = rates

    def state_dict(self) -> dict[str, Any]:
        return self.scheduler.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.scheduler.load_state_dict(dict(state))


def _rng_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "torch_cpu": torch.random.get_rng_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        device_index = int(torch.cuda.current_device())
        payload["torch_cuda_device_index"] = device_index
        payload["torch_cuda"] = torch.cuda.get_rng_state(device_index)
    return payload


def _restore_rng(payload: Mapping[str, Any]) -> None:
    torch.random.set_rng_state(payload["torch_cpu"])
    random.setstate(payload["python"])
    if torch.cuda.is_available() and "torch_cuda" in payload:
        current_device = int(torch.cuda.current_device())
        if int(payload.get("torch_cuda_device_index", -1)) != current_device:
            raise ValueError("rank RNG sidecar belongs to another CUDA device")
        torch.cuda.set_rng_state(payload["torch_cuda"], device=current_device)


def _portable_sampler_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return the rank-invariant resume frontier stored in the portable file.

    Local prompt/occurrence shards are deterministic consequences of the
    sealed global sampler state plus rank/world size.  Persisting rank 0's
    local fields in a file loaded by every rank makes exact distributed resume
    impossible, because ranks 1..N correctly derive different local shards.
    ZeRO rank-runtime files still retain the complete per-rank state.
    """

    portable = {
        str(key): value
        for key, value in dict(state).items()
        if not str(key).startswith("local_")
    }
    if not portable or "epoch" not in portable or "global_batch_index" not in portable:
        raise ValueError("sampler state lacks a portable global frontier")
    return portable


def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    owner: str,
    backend: Any,
    sampler_state: Mapping[str, Any],
    phase: str,
    method: str,
    geometry_schema_version: str,
    attn_implementation: str,
    seed: int,
    selected_r_sha256: str,
    manifest: Mapping[str, Any],
    cache_identity_sha256: str,
    exact_resume_identity_sha256: str,
    step: int,
    recover_incomplete_transactions: bool = False,
) -> tuple[BridgeCheckpointIdentity, SealedCheckpoint]:
    expected_route = {"A": "route1"}.get(phase)
    if expected_route is None:
        raise ValueError("checkpoint phase is invalid")
    run_dir = Path(path).parent
    if Path(path) != route_checkpoint_path(run_dir, expected_route, int(step)):
        raise ValueError("checkpoint directory route/step differs from its owner")
    route = expected_route
    building = checkpoint_building_path(path)
    local_conflict: str | None = None
    if backend.rank == 0:
        try:
            prepare_checkpoint_building_directory(
                path, recover_incomplete=bool(recover_incomplete_transactions)
            )
        except Exception as exc:
            local_conflict = f"{type(exc).__name__}: {exc}"
    conflict = broadcast_checkpoint_preflight(
        torch_distributed, rank=int(backend.rank), local_conflict=local_conflict
    )
    if conflict is not None:
        raise FileExistsError(conflict)
    if torch_distributed.is_initialized():
        torch_distributed.barrier()
    owner_module = _owner_module(model, owner)
    owned_state = _owner_checkpoint_state(owner_module)
    excluded_frozen_parameter_names = _checkpoint_excluded_frozen_parameter_names(
        model, owner=owner
    )
    rng_state = _rng_payload()
    portable_sampler_state = _portable_sampler_state(sampler_state)
    portable_identity_seed = canonical_json_sha256(
        {
            "phase": phase,
            "method": method,
            "geometry_schema_version": geometry_schema_version,
            "seed": int(seed),
            "owner": owner,
            "step": int(step),
            "selected_r_sha256": selected_r_sha256,
            "cache_identity_sha256": cache_identity_sha256,
            "exact_resume_identity_sha256": exact_resume_identity_sha256,
            "sampler_state_sha256": canonical_json_sha256(portable_sampler_state),
            "runtime_backend_identity_sha256": backend.identity.compatibility_sha256(),
        }
    )
    runtime_state = {
        **artifact_header(RANK_RUNTIME),
        "rank": int(backend.rank),
        "world_size": int(backend.world_size),
        "portable_identity_seed": portable_identity_seed,
        "rng_state": rng_state,
        "sampler_state": dict(sampler_state),
    }
    runtime_path = building / "runtime" / f"rank-{int(backend.rank):05d}.pt"
    runtime_failure: str | None = None
    try:
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(runtime_state, runtime_path)
    except Exception as exc:
        runtime_failure = f"rank runtime save: {type(exc).__name__}: {exc}"
    runtime_failures = collect_rank_failures(
        torch_distributed,
        world_size=int(backend.world_size),
        local_failure=runtime_failure,
    )
    if runtime_failures:
        raise RuntimeError("; ".join(runtime_failures))
    sharded_manifest: dict[str, Any] | None = None
    backend_failure: str | None = None
    try:
        sharded_manifest = backend.save_checkpoint(building)
    except Exception as exc:
        backend_failure = f"backend checkpoint save: {type(exc).__name__}: {exc}"
    backend_failures = collect_rank_failures(
        torch_distributed,
        world_size=int(backend.world_size),
        local_failure=backend_failure,
    )
    if backend_failures:
        raise RuntimeError("; ".join(backend_failures))
    if backend.is_zero1:
        if sharded_manifest is None:
            raise RuntimeError(
                "DeepSpeed save did not return its all-rank shard ledger"
            )
        resume_artifact_sha256 = str(sharded_manifest["artifact_sha256"])
        optimizer_state = {
            **artifact_header(ZERO1_OPTIMIZER_RESUME),
            "portable_identity_seed": portable_identity_seed,
            "sharded_resume_manifest": sharded_manifest,
        }
        scheduler_state = {
            **artifact_header(ZERO1_SCHEDULER_RESUME),
            "portable_identity_seed": portable_identity_seed,
            "resume_artifact_sha256": resume_artifact_sha256,
        }
    else:
        resume_artifact_sha256 = "0" * 64
        optimizer_state = backend.optimizer.state_dict()
        scheduler_state = backend.scheduler.state_dict()
    local_failure: str | None = None
    published_checkpoint: dict[str, Any] | None = None
    if backend.rank == 0:
        try:
            runtime_ledger = {
                f"rank-{rank:05d}": file_sha256(
                    building / "runtime" / f"rank-{rank:05d}.pt"
                )
                for rank in range(int(backend.world_size))
            }
            identity = BridgeCheckpointIdentity(
                artifact_type=CHECKPOINT_IDENTITY,
                schema_version=CHECKPOINT_SCHEMA_VERSION,
                course_schema_version=COURSE_SCHEMA_VERSION,
                objective_version=OBJECTIVE_VERSION,
                specificity_objective_version=SPECIFICITY_OBJECTIVE_VERSION,
                causal_objective_schema_version=CAUSAL_OBJECTIVE_SCHEMA_VERSION,
                geometry_schema_version=str(geometry_schema_version),
                attn_implementation=str(attn_implementation),
                tokenizer_sha256=str(manifest["tokenizer_sha256"]),
                boundary_token_ids=tuple(
                    (int(value) for value in manifest["boundary_token_ids"])
                ),
                boundary_token_count=int(manifest["boundary_token_count"]),
                boundary_ids_sha256=str(manifest["boundary_ids_sha256"]),
                phase=phase,
                method=method,
                seed=int(seed),
                selected_r_sha256=selected_r_sha256,
                need_z_cohort_definition=str(manifest["need_z_cohort_definition"]),
                d_answer_provenance_schema=str(manifest["d_answer_provenance_schema"]),
                d_answer_eos_rule=str(manifest["d_answer_eos_rule"]),
                model_state_schema_sha256=_state_schema_sha256(
                    _owner_module(model, owner)
                ),
                owned_state_sha256=_tensor_tree_sha256(owned_state),
                optimizer_state_sha256=_tensor_tree_sha256(optimizer_state),
                scheduler_state_sha256=_tensor_tree_sha256(scheduler_state),
                rng_state_sha256=canonical_json_sha256(runtime_ledger),
                sampler_state_sha256=canonical_json_sha256(portable_sampler_state),
                cache_identity_sha256=cache_identity_sha256,
                exact_resume_identity_sha256=exact_resume_identity_sha256,
                optimizer_backend=backend.identity.optimizer_backend,
                zero_stage=backend.identity.zero_stage,
                deepspeed_config_sha256=backend.identity.deepspeed_config_sha256,
                deepspeed_state_semantics_sha256=backend.identity.deepspeed_state_semantics_sha256,
                world_size=backend.identity.world_size,
                route1_local_samples=backend.identity.route1_local_samples,
                route1_gradient_accumulation_steps=backend.identity.route1_gradient_accumulation_steps,
                resume_artifact_sha256=resume_artifact_sha256,
                step=int(step),
            )
            identity.validate()
            _save_owner_safetensors(
                building / "model.safetensors",
                owned_state,
                route=route,
                owner=owner,
                identity=identity,
                excluded_frozen_parameter_names=excluded_frozen_parameter_names,
            )
            save_fixed_state(building, _owner_module(model, owner))
            _write_checkpoint_portable_assets(
                building,
                run_dir=run_dir,
                route=route,
                owner=owner,
                identity=identity,
                excluded_frozen_parameter_names=excluded_frozen_parameter_names,
            )
            torch.save(optimizer_state, building / "optimizer.pt")
            torch.save(scheduler_state, building / "scheduler.pt")
            retained_occurrences = int(
                portable_sampler_state["retained_occurrence_count"]
            )
            logical_global_batch = (
                int(backend.identity.route1_local_samples)
                * int(backend.identity.world_size)
                * int(backend.identity.route1_gradient_accumulation_steps)
            )
            steps_per_epoch = occurrence_steps_per_epoch(
                retained_occurrences,
                global_batch_size=logical_global_batch,
                drop_last=True,
            )
            write_atomic_json(
                building / "trainer_state.json",
                {
                    **artifact_header(CHECKPOINT_TRAINER_STATE),
                    "route": route,
                    "step": int(step),
                    "epoch": int(portable_sampler_state["epoch"]),
                    "global_batch_index": int(
                        portable_sampler_state["global_batch_index"]
                    ),
                    "retained_occurrence_count": retained_occurrences,
                    "steps_per_epoch": steps_per_epoch,
                    "epoch_end": int(portable_sampler_state["global_batch_index"])
                    == steps_per_epoch - 1,
                    "phase_terminal": bool(
                        portable_sampler_state.get("phase_terminal", False)
                    ),
                    "control_frontier": "terminal"
                    if portable_sampler_state.get("phase_terminal") is True
                    else "periodic",
                    "sampler_state": portable_sampler_state,
                },
                replace_mismatch=False,
            )
            write_checkpoint_config_reference(
                building, run_dir=run_dir, include_startup_invariant=method != "bridge"
            )
            checkpoint_frozen_source = None
            seal = seal_checkpoint_directory(
                building,
                path,
                route=route,
                step=int(step),
                metadata={
                    "owner": owner,
                    "world_size": int(backend.world_size),
                    "identity": dict(identity.__dict__),
                    "runtime_ledger": runtime_ledger,
                    "zero1_manifest": sharded_manifest,
                    "portable_identity_seed": portable_identity_seed,
                    "exact_resume_identity_sha256": exact_resume_identity_sha256,
                    "model_safetensors_sha256": file_sha256(
                        building / "model.safetensors"
                    ),
                    "active_owner_config_sha256": file_sha256(
                        building / "active_owner_config.json"
                    ),
                    "tensor_scope": "active_owner_trainable_parameters_only",
                    "contains_frozen_parameters": False,
                    "excluded_frozen_parameter_names": list(
                        excluded_frozen_parameter_names
                    ),
                    "frozen_parameter_source": checkpoint_frozen_source,
                },
            )
            published_checkpoint = {
                "identity": dict(identity.__dict__),
                "seal": seal.to_mapping(),
            }
        except Exception as exc:
            local_failure = (
                f"rank0 checkpoint-directory publication: {type(exc).__name__}: {exc}"
            )
    failures = collect_rank_failures(
        torch_distributed,
        world_size=int(backend.world_size),
        local_failure=local_failure,
    )
    if failures:
        raise RuntimeError("; ".join(failures))
    checkpoint_payload = broadcast_rank0_result(
        torch_distributed, rank=int(backend.rank), local_result=published_checkpoint
    )
    if (
        not isinstance(checkpoint_payload, Mapping)
        or set(checkpoint_payload) != {"identity", "seal"}
        or (not isinstance(checkpoint_payload["identity"], Mapping))
        or (not isinstance(checkpoint_payload["seal"], Mapping))
    ):
        raise RuntimeError("checkpoint publication did not reach every rank")
    return (
        BridgeCheckpointIdentity.from_mapping(checkpoint_payload["identity"]),
        SealedCheckpoint.from_mapping(checkpoint_payload["seal"]),
    )


def _resume_sampler_matches(
    saved: Mapping[str, Any], expected: Mapping[str, Any], *, extended: bool
) -> bool:
    (left, right) = (dict(saved), dict(expected))
    if extended:
        for key in ("phase_terminal", "route1_max_optimizer_updates"):
            left.pop(key, None)
            right.pop(key, None)
    return left == right


def _load_resume(
    path: Path,
    *,
    model: nn.Module,
    owner: str,
    backend: Any,
    phase: str,
    method: str,
    seed: int,
    manifest: Mapping[str, Any],
    selected_r_sha256: str,
    cache_identity_sha256: str,
    exact_resume_identity_sha256: str,
    expected_sampler_state: Mapping[str, Any],
    checkpoint_metadata_payload: Mapping[str, Any],
    sealed_checkpoint: SealedCheckpoint,
    extended: bool = False,
) -> tuple[int, dict[str, Any]]:
    if not is_bridge_isolated_path(path):
        raise ValueError("resume checkpoint is outside the ThinkBridge namespace")
    sealed_checkpoint = validate_checkpoint_seal(sealed_checkpoint)
    if sealed_checkpoint.path != path.resolve(strict=True):
        raise ValueError("resume checkpoint differs from its rank-0 seal")
    from think_bridge.model.checkpoint_policy import checkpoint_owner_run_directory

    saved_frozen_identity = load_route_runtime_identity(
        checkpoint_owner_run_directory(path), route="route1"
    )["frozen_executor_identity"]
    if getattr(model, "_frozen_executor_identity", None) != saved_frozen_identity:
        raise ValueError("resume frozen executor content identity mismatch")
    metadata = dict(checkpoint_metadata_payload)
    if metadata != _read_json(sealed_checkpoint.path / "checkpoint.json"):
        raise ValueError("resume metadata differs from its rank-0 seal")
    identity = BridgeCheckpointIdentity.from_mapping(metadata["identity"])
    owned_state = _load_owner_safetensors(path)
    optimizer_state = torch.load(path / "optimizer.pt", map_location="cpu")
    scheduler_state = torch.load(path / "scheduler.pt", map_location="cpu")
    trainer_state = _read_json(path / "trainer_state.json")
    expected = (
        phase,
        method,
        int(seed),
        selected_r_sha256,
        cache_identity_sha256,
        exact_resume_identity_sha256,
    )
    observed = (
        identity.phase,
        identity.method,
        identity.seed,
        identity.selected_r_sha256,
        identity.cache_identity_sha256,
        identity.exact_resume_identity_sha256,
    )
    if observed != expected or metadata.get("owner") != owner:
        raise ValueError("resume phase/arm/seed/run/owner identity mismatch")
    backend_observed = checkpoint_backend_compatibility(identity)
    backend_expected = checkpoint_backend_compatibility(backend.identity)
    if backend_observed != backend_expected:
        raise ValueError(
            "resume optimizer backend/world/route geometry identity mismatch"
        )
    if (
        identity.tokenizer_sha256 != manifest["tokenizer_sha256"]
        or list(identity.boundary_token_ids) != list(manifest["boundary_token_ids"])
        or identity.boundary_token_count != manifest["boundary_token_count"]
        or (identity.boundary_ids_sha256 != manifest["boundary_ids_sha256"])
    ):
        raise ValueError("resume tokenizer/boundary identity differs from manifest")
    if identity.model_state_schema_sha256 != _state_schema_sha256(
        _owner_module(model, owner)
    ):
        raise ValueError("resume model state schema mismatch")
    checks = (
        ("owned_state_sha256", _tensor_tree_sha256(owned_state)),
        ("optimizer_state_sha256", _tensor_tree_sha256(optimizer_state)),
        ("scheduler_state_sha256", _tensor_tree_sha256(scheduler_state)),
        ("rng_state_sha256", canonical_json_sha256(metadata["runtime_ledger"])),
        ("sampler_state_sha256", canonical_json_sha256(trainer_state["sampler_state"])),
    )
    for field, actual in checks:
        if getattr(identity, field) != actual:
            raise ValueError(f"resume payload hash mismatch: {field}")
    portable_expected_sampler = _portable_sampler_state(expected_sampler_state)
    if not _resume_sampler_matches(
        trainer_state["sampler_state"], portable_expected_sampler, extended=extended
    ):
        raise ValueError(
            "resume global sampler frontier does not match the deterministic course"
        )
    restored_fixed_state = load_owner_fixed_state(path, owner, owned_state)
    _restore_owner_checkpoint_state(_owner_module(model, owner), owned_state)
    restore_fixed_state(_owner_module(model, owner), restored_fixed_state)
    local_runtime = torch.load(
        path / "runtime" / f"rank-{int(backend.rank):05d}.pt", map_location="cpu"
    )
    portable_identity_seed = require_sha256(
        str(metadata.get("portable_identity_seed", "")), "portable_identity_seed"
    )
    if (
        not isinstance(local_runtime, Mapping)
        or local_runtime.get("artifact_type") != RANK_RUNTIME
        or local_runtime.get("schema_version") != 1
        or (int(local_runtime.get("rank", -1)) != int(backend.rank))
        or (int(local_runtime.get("world_size", -1)) != int(backend.world_size))
        or (local_runtime.get("portable_identity_seed") != portable_identity_seed)
    ):
        raise ValueError("resume rank runtime sidecar identity is malformed")
    if backend.is_zero1:
        optimizer_marker = optimizer_state
        scheduler_marker = scheduler_state
        if (
            not isinstance(optimizer_marker, dict)
            or optimizer_marker.get("artifact_type") != ZERO1_OPTIMIZER_RESUME
            or optimizer_marker.get("schema_version") != 1
            or (not isinstance(scheduler_marker, dict))
            or (scheduler_marker.get("artifact_type") != ZERO1_SCHEDULER_RESUME)
            or (scheduler_marker.get("schema_version") != 1)
        ):
            raise ValueError("resume ZeRO optimizer/scheduler marker mismatch")
        sharded_manifest = optimizer_marker.get("sharded_resume_manifest")
        if (
            not isinstance(sharded_manifest, dict)
            or sharded_manifest.get("artifact_sha256")
            != identity.resume_artifact_sha256
            or scheduler_marker.get("resume_artifact_sha256")
            != identity.resume_artifact_sha256
        ):
            raise ValueError("resume portable/sharded artifact identity mismatch")
        if metadata.get("zero1_manifest") != sharded_manifest:
            raise ValueError("checkpoint metadata/optimizer ZeRO manifest mismatch")
        backend.load_checkpoint(
            path, sharded_manifest, sealed_checkpoint=sealed_checkpoint
        )
        restore_fixed_state(_owner_module(model, owner), restored_fixed_state)
        active_owner = _owner_module(model, owner)
        owner_device = next(active_owner.parameters()).device
        if not _all_ranks_true(
            _tensor_tree_sha256(_owner_checkpoint_state(active_owner))
            == identity.owned_state_sha256,
            device=owner_device,
        ):
            raise ValueError(
                "ZeRO sharded module restore differs from portable owner identity"
            )
        portable_seed = optimizer_marker.get("portable_identity_seed")
        if (
            portable_identity_seed != portable_seed
            or scheduler_marker.get("portable_identity_seed") != portable_seed
        ):
            raise ValueError("resume shard does not bind the portable identity")
        restored_sampler = local_runtime.get("sampler_state")
        restored_rng = local_runtime.get("rng_state")
        if not isinstance(restored_sampler, Mapping) or not isinstance(
            restored_rng, Mapping
        ):
            raise ValueError("resume rank runtime RNG/sampler state is missing")
        if not _resume_sampler_matches(
            restored_sampler, expected_sampler_state, extended=extended
        ):
            raise ValueError("resume rank sampler differs from deterministic course")
        _restore_rng(restored_rng)
        return (int(identity.step), dict(restored_sampler))
    backend.optimizer.load_state_dict(optimizer_state)
    backend.scheduler.load_state_dict(scheduler_state)
    if not _resume_sampler_matches(
        local_runtime.get("sampler_state", {}),
        expected_sampler_state,
        extended=extended,
    ):
        raise ValueError("resume DDP rank sampler differs from deterministic course")
    restored_rng = local_runtime.get("rng_state")
    if not isinstance(restored_rng, Mapping):
        raise ValueError("resume DDP rank RNG state is missing")
    _restore_rng(restored_rng)
    return (int(identity.step), dict(local_runtime["sampler_state"]))


def _power_two_length_bucket(length: int) -> int:
    if int(length) <= 0:
        raise ValueError("physical sequence length must be positive")
    return 1 << (int(length) - 1).bit_length()


def _route1_length_key(
    row: Mapping[str, Any],
    *,
    epoch: int,
    boundary_token_count: int,
    compute_course: bool,
    compute_match: bool,
    compute_specific: bool,
) -> tuple[int, int, int]:
    """Bucket only physical branches enabled by the active Route1 leaves."""

    return route1_active_physical_length_key(
        row,
        epoch=int(epoch),
        boundary_token_count=int(boundary_token_count),
        compute_course=compute_course,
        compute_match=compute_match,
        compute_specific=compute_specific,
    )


def _route1_optimizer_steps_for_epoch(
    rows: Sequence[Mapping[str, Any]] | Route1NormalizedRowCache,
    *,
    epoch: int,
    seed: int,
    world_size: int,
    local_samples: int,
    gradient_accumulation_steps: int,
    boundary_token_count: int,
    population: str,
    compute_course: bool,
    compute_match: bool,
    compute_specific: bool,
) -> int:
    """Count exact retained GAS windows for one active Route1 epoch domain."""

    cache = resolve_route1_normalized_rows(rows, population=population)
    active_rows = cache.active_epoch_rows(
        epoch=int(epoch),
        compute_course=bool(compute_course),
        compute_match=bool(compute_match),
        compute_specific=bool(compute_specific),
    )
    return exact_route1_optimizer_steps(
        len(active_rows),
        local_samples=int(local_samples),
        world_size=int(world_size),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
    )


def _route1_optimizer_course_clock(
    rows: Sequence[Mapping[str, Any]] | Route1NormalizedRowCache,
    *,
    epochs: Sequence[int],
    seed: int,
    world_size: int,
    local_samples: int,
    gradient_accumulation_steps: int,
    boundary_token_count: int,
    population: str,
    compute_course: bool,
    compute_match: bool,
    compute_specific: bool,
    max_optimizer_updates: int | None,
) -> Any:
    steps = [
        (
            int(epoch),
            _route1_optimizer_steps_for_epoch(
                rows,
                epoch=int(epoch),
                seed=int(seed),
                world_size=int(world_size),
                local_samples=int(local_samples),
                gradient_accumulation_steps=int(gradient_accumulation_steps),
                boundary_token_count=int(boundary_token_count),
                population=population,
                compute_course=bool(compute_course),
                compute_match=bool(compute_match),
                compute_specific=bool(compute_specific),
            ),
        )
        for epoch in epochs
    ]
    return optimizer_course_clock(steps, max_optimizer_updates=max_optimizer_updates)


def _epoch_batches(
    rows: Sequence[Mapping[str, Any]] | Route1NormalizedRowCache,
    *,
    epoch: int,
    seed: int,
    rank: int,
    world_size: int,
    local_samples: int,
    gradient_accumulation_steps: int,
    boundary_token_count: int,
    population: str,
    compute_course: bool = True,
    compute_match: bool,
    compute_specific: bool,
    eval_null_mode: str,
    max_optimizer_updates: int | None = None,
    terminal_epoch: int = 1,
    cursor: int = 0,
    optimizer_update_offset: int = 0,
    terminal_optimizer_update: int | None = None,
    specificity_donors_per_owner: int = 2,
    distillation_populations: Sequence[str] = ("C",),
    course_epochs: float = 0.0,
    course_updates: int | None = None,
    course_steps_per_epoch: int | None = None,
) -> Iterable[tuple[list[Mapping[str, Any]], dict[str, Any]]]:
    cache = resolve_route1_normalized_rows(rows, population=population)
    local_microbatch = int(local_samples)
    accumulation = int(gradient_accumulation_steps)
    local_optimizer_occurrences = local_microbatch * accumulation
    global_optimizer_occurrences = local_optimizer_occurrences * int(world_size)
    if global_optimizer_occurrences <= 0:
        raise RuntimeError("Route1 logical optimizer batch must be positive")
    if population not in {"staged", "answer-only", "gold-reference"}:
        raise ValueError("Route1 population must be staged or answer-only")
    if not all(
        isinstance(value, bool)
        for value in (compute_course, compute_match, compute_specific)
    ):
        raise TypeError("Route1 leaf switches must be boolean")
    eval_null_mode = normalize_route1_null_mode(eval_null_mode)
    rows = list(
        cache.active_epoch_rows(
            epoch=int(epoch),
            compute_course=bool(compute_course),
            compute_match=bool(compute_match),
            compute_specific=bool(compute_specific),
        )
    )
    if (
        complete_optimizer_window_count(
            len(rows),
            local_microbatch=local_microbatch,
            world_size=int(world_size),
            gradient_accumulation_steps=accumulation,
        )
        == 0
    ):
        return
    active_distillation_populations = frozenset(
        str(value).upper()
        for value in (
            distillation_populations
            if not isinstance(distillation_populations, str)
            else tuple(distillation_populations)
        )
    )
    if active_distillation_populations not in {
        frozenset({"C"}),
        frozenset({"B", "C"}),
        frozenset({"G"}),
    }:
        raise ValueError("distillation_populations must select C or B+C")

    validate_donor_limit(specificity_donors_per_owner)
    assignment_cache = {}

    def sequential_assignment(
        batch_index: int,
        batch: Any,
        rows_by_id: Mapping[str, Mapping[str, Any]],
    ) -> CostBalancedRankAssignment:
        key = tuple(batch.record_ids)
        if key in assignment_cache:
            return assignment_cache[key]

        shards = tuple(
            (
                tuple(
                    batch.record_ids[r * local_microbatch : (r + 1) * local_microbatch]
                ),
            )
            for r in range(int(world_size))
        )
        assignment = CostBalancedRankAssignment(
            rank_microsteps=shards,
            rank_costs=tuple(0 for _ in shards),
            assignment_sha256=canonical_json_sha256(shards),
        )
        assignment_cache[key] = assignment
        return assignment

    sampler = OccurrenceBatchSampler(
        rows,
        shuffle_records=True,
        seed=seed,
        rank=rank,
        world_size=world_size,
        length_key=lambda row: _route1_length_key(
            row,
            epoch=int(epoch),
            boundary_token_count=int(boundary_token_count),
            compute_course=bool(compute_course),
            compute_match=bool(compute_match),
            compute_specific=bool(compute_specific),
        ),
        global_batch_size=local_microbatch * int(world_size),
        rank_assignment=sequential_assignment,
        complete_batch_multiple=accumulation,
    )
    sampler.set_epoch(int(epoch))
    sampler.set_cursor(int(cursor) * accumulation)
    plan = sampler.plan
    rows_by_id = {str(row["record_id"]): row for row in rows}
    steps_per_epoch = len(plan.batches) // accumulation
    if steps_per_epoch * accumulation != len(plan.batches):
        raise RuntimeError("Route1 sampler retained an incomplete GAS window")
    course_steps = (
        steps_per_epoch
        if course_steps_per_epoch is None
        else int(course_steps_per_epoch)
    )
    donor_sampler = SpecificityDonorSampler(
        seed=seed, donors_per_owner=specificity_donors_per_owner
    )
    # Advance the independent epoch RNG over skipped microbatches on resume.
    # The epoch membership/order is already reconstructed by the data sampler.
    for previous_window in range(int(cursor)):
        previous_update = int(optimizer_update_offset) + previous_window
        enabled = bool(
            compute_specific
            and route1_specificity_scale(
                previous_update,
                course_steps,
                course_epochs=course_epochs,
                course_steps=course_updates,
            )
            > 0.0
        )
        previous_ids = [
            record_id
            for previous_batch in plan.batches[
                previous_window * accumulation : (previous_window + 1) * accumulation
            ]
            for record_id in previous_batch.record_ids
        ]
        donor_sampler.sample(
            [rows_by_id[key] for key in previous_ids],
            enabled=enabled,
            active_populations=active_distillation_populations,
            owner_populations=(("C",)),
        )
    sampler_iterator = iter(sampler)
    for batch_index in range(int(cursor), steps_per_epoch):
        completed_updates = int(optimizer_update_offset) + int(batch_index)
        specificity_scale = route1_specificity_scale(
            completed_updates,
            course_steps,
            course_epochs=course_epochs,
            course_steps=course_updates,
        )
        effective_specific = bool(compute_specific and specificity_scale > 0.0)
        microbatches = [
            plan.batches[batch_index * accumulation + microstep]
            for microstep in range(accumulation)
        ]
        local_index_microbatches = [
            next(sampler_iterator) for _microstep in range(accumulation)
        ]
        global_groups_by_microstep = [
            list(batch.prompt_group_ids) for batch in microbatches
        ]
        microstep_global_count = local_microbatch * int(world_size)
        if any(
            len(groups) != microstep_global_count
            for groups in global_groups_by_microstep
        ):
            raise RuntimeError(
                "Route1 distributed microbatch has incorrect sample count"
            )
        global_groups = [
            group for groups in global_groups_by_microstep for group in groups
        ]
        global_occurrence_microbatches = [
            [rows_by_id[record_id] for record_id in batch.record_ids]
            for batch in microbatches
        ]
        global_occurrences = [
            row
            for microbatch_rows in global_occurrence_microbatches
            for row in microbatch_rows
        ]
        global_length_keys = [
            key for batch in microbatches for key in batch.length_keys
        ]
        assignments = [
            sequential_assignment(
                batch_index * accumulation + microstep,
                batch,
                {record_id: rows_by_id[record_id] for record_id in batch.record_ids},
            )
            for microstep, batch in enumerate(microbatches)
        ]
        record_microbatches = [
            [rows[index] for index in local_indices]
            for local_indices in local_index_microbatches
        ]
        records = [
            row for microbatch_rows in record_microbatches for row in microbatch_rows
        ]
        if any(len(values) != local_microbatch for values in record_microbatches):
            raise RuntimeError("rank-local Route1 microbatch size differs from config")
        global_occurrence_ids = [str(row["record_id"]) for row in global_occurrences]
        if (
            len(global_occurrence_ids) != global_optimizer_occurrences
            or len(set(global_occurrence_ids)) != global_optimizer_occurrences
        ):
            raise RuntimeError("global Route1 sample identities are not one-to-one")
        local_occurrence_ids = [str(row["record_id"]) for row in records]
        expected_local_ids = [
            record_id
            for assignment in assignments
            for record_id in assignment.rank_microsteps[rank][0]
        ]
        if local_occurrence_ids != expected_local_ids:
            raise RuntimeError("Route1 sampler and balanced assignment differ")
        local_groups = [str(row["prompt_group_id"]) for row in records]
        native_population_counts = {
            side: sum(row["quadrant"] == side for row in global_occurrences)
            if population != "answer-only" and (compute_match or effective_specific)
            else 0
            for side in ("B", "C", "G")
        }
        native_objective_sample_count = sum(
            native_population_counts[side] for side in active_distillation_populations
        )
        match_objective_sample_count = (
            native_objective_sample_count if compute_match else 0
        )
        specific_objective_sample_count = (
            native_objective_sample_count if effective_specific else 0
        )
        global_microstep_native_sample_counts = [
            (
                0
                if population == "answer-only"
                or not (compute_match or effective_specific)
                else sum(
                    int(row["quadrant"] in active_distillation_populations)
                    for row in microbatch_rows
                )
            )
            for microbatch_rows in global_occurrence_microbatches
        ]
        course_sides = [
            (
                _route1_execution_course_side(row["quadrant"], int(epoch))
                if compute_course
                else None
            )
            for row in global_occurrences
        ]
        course_c_sample_count = sum(side == "c" for side in course_sides)
        course_direct_side_sample_count = sum(
            side == "direct_correct_side" for side in course_sides
        )
        course_sample_count = course_c_sample_count + course_direct_side_sample_count
        if not any(
            (
                int(course_sample_count) > 0,
                int(match_objective_sample_count) > 0,
                int(specific_objective_sample_count) > 0,
            )
        ):
            raise RuntimeError(
                "Route1 active-domain planner produced an inactive optimizer window"
            )
        optimizer_update = int(optimizer_update_offset) + int(batch_index) + 1
        if max_optimizer_updates is not None and optimizer_update > int(
            max_optimizer_updates
        ):
            break
        # Specificity candidates are sampled once from the complete optimizer
        # window.  This keeps the objective independent of GAS/microbatch
        # partitioning; the model receives the corresponding prompt bank and
        # recomputes selected donor z's under the current parameters.
        window_selection = donor_sampler.sample(
            global_occurrences,
            enabled=effective_specific,
            active_populations=active_distillation_populations,
            owner_populations=(("C",)),
        )
        specific_objective_sample_count = (
            sum(bool(ids) for ids in window_selection.donors.values())
            if effective_specific
            else 0
        )
        window_donor_record_ids = global_occurrence_ids if effective_specific else []
        local_wrong_candidate_mask: list[list[bool]] = []
        for owner_rows in record_microbatches:
            local_wrong_candidate_mask.extend(
                [
                    donor_id in window_selection.donors[str(owner["record_id"])]
                    for donor_id in window_donor_record_ids
                ]
                for owner in owner_rows
            )
        positive_record_pairs = (
            sum(
                1
                for owner in global_occurrences
                if window_selection.donors[str(owner["record_id"])]
                for candidate in global_occurrences
                if owner["record_id"] != candidate["record_id"]
                and owner["prompt_group_id"] == candidate["prompt_group_id"]
            )
            if effective_specific
            else 0
        )
        state = {
            "specificity_same_prompt_record_pair_count": positive_record_pairs,
            "specificity_positive_policy": "own-and-same-prompt-two-dropout-views-owner-teacher-prefix",
            "record_order_policy": "sorted-record-id-python-random-seed-plus-epoch-shuffle",
            "rank_assignment_policy": "contiguous-slices-no-length-reordering",
            "specificity_eligible_pair_count": int(
                window_selection.eligible_pair_count
            ),
            "specificity_selected_pair_count": int(
                window_selection.selected_pair_count
            ),
            "specificity_weight_scale": float(specificity_scale),
            "course_retained_fraction": 1.0
            - (
                route1_curriculum_value(
                    completed_updates,
                    course_steps,
                    course_epochs=course_epochs,
                    course_steps=course_updates,
                )
            ),
            **artifact_header(ROUTE1_OCCURRENCE_SAMPLER_SCHEMA),
            "active_domain_schema": ROUTE1_ACTIVE_DOMAIN_SCHEMA,
            "physical_cost_policy_schema": ROUTE1_PHYSICAL_COST_POLICY_SCHEMA,
            "weighting_unit": OCCURRENCE_WEIGHTING_UNIT,
            "batch_unit": OCCURRENCE_BATCH_UNIT,
            "epoch": int(epoch),
            "global_batch_index": int(batch_index),
            "seed": int(seed),
            "specificity_sampling_policy": SPECIFICITY_SAMPLING_POLICY,
            "specificity_donors_per_owner": validate_donor_limit(
                specificity_donors_per_owner
            ),
            "global_prompt_group_ids": list(global_groups),
            "global_prompt_group_ids_sha256": canonical_json_sha256(global_groups),
            "global_microstep_prompt_group_ids": global_groups_by_microstep,
            "global_microstep_prompt_group_ids_sha256": canonical_json_sha256(
                global_groups_by_microstep
            ),
            "local_prompt_group_ids": list(local_groups),
            "local_prompt_group_ids_sha256": canonical_json_sha256(local_groups),
            "global_occurrence_record_ids": global_occurrence_ids,
            "global_occurrence_record_ids_sha256": canonical_json_sha256(
                global_occurrence_ids
            ),
            "local_occurrence_record_ids": local_occurrence_ids,
            "local_occurrence_record_ids_sha256": canonical_json_sha256(
                local_occurrence_ids
            ),
            "global_length_bucket_keys": [list(key) for key in global_length_keys],
            "global_length_bucket_keys_sha256": canonical_json_sha256(
                [list(key) for key in global_length_keys]
            ),
            "retained_occurrence_record_ids_sha256": plan.retained_record_ids_sha256,
            "dropped_occurrence_record_ids_sha256": plan.dropped_record_ids_sha256,
            "dropped_occurrence_count": len(plan.dropped_record_ids),
            "epoch_occurrence_count": len(rows),
            "retained_occurrence_count": len(plan.retained_record_ids),
            "epoch_batch_order_sha256": plan.batch_order_sha256,
            "rank_assignment_sha256": canonical_json_sha256(
                [assignment.assignment_sha256 for assignment in assignments]
            ),
            "rank_estimated_costs": [
                sum(assignment.rank_costs[owner_rank] for assignment in assignments)
                for owner_rank in range(int(world_size))
            ],
            "global_prompt_group_count": len(set(global_groups)),
            "global_sample_count": global_optimizer_occurrences,
            "global_occurrence_count": global_optimizer_occurrences,
            "optimizer_global_batch": global_optimizer_occurrences,
            "route1_local_samples": local_microbatch,
            "route1_gradient_accumulation_steps": accumulation,
            "route1_microstep_global_samples": local_microbatch * int(world_size),
            "local_microstep_sample_record_ids": [
                local_occurrence_ids[
                    micro_step * local_microbatch : (micro_step + 1) * local_microbatch
                ]
                for micro_step in range(accumulation)
            ],
            "route1_population_phase": (
                "c_native_distillation" if not compute_course else "fixed_b_c_d_course"
            ),
            "answer_course_geometry": (
                "deployed_z_cot_suffix"
                if route1_curriculum_value(
                    completed_updates,
                    course_steps,
                    course_epochs=course_epochs,
                    course_steps=course_updates,
                )
                < 1.0
                else "deployed_z"
            ),
            "answer_course_batch_count": steps_per_epoch,
            "optimizer_update_offset": int(optimizer_update_offset),
            "compute_course": bool(compute_course),
            "compute_match": bool(compute_match),
            "compute_specific": bool(compute_specific),
            "global_course_sample_count": int(course_sample_count),
            "global_course_c_sample_count": int(course_c_sample_count),
            "global_course_direct_side_sample_count": int(
                course_direct_side_sample_count
            ),
            "course_reduction": route1_course_reduction(int(epoch)),
            "global_native_b_sample_count": int(native_population_counts["B"]),
            "global_native_c_sample_count": int(
                native_population_counts["C"] + native_population_counts["G"]
            ),
            "distillation_reduction": "global-configured-native-window-empirical-proportion",
            "distillation_populations": sorted(active_distillation_populations),
            "global_match_sample_count": int(match_objective_sample_count),
            "global_specific_sample_count": int(specific_objective_sample_count),
            "global_microstep_native_sample_counts": global_microstep_native_sample_counts,
            "optimizer_update": optimizer_update,
            "optimizer_step": optimizer_update - 1,
            "route1_population": str(population),
            "route1_max_optimizer_updates": max_optimizer_updates,
            "d_answer_provenance_schema": D_ANSWER_PROVENANCE_SCHEMA,
            "d_answer_eos_rule": D_ANSWER_EOS_RULE,
            "epoch_b_sample_count": sum(int(row["quadrant"] == "B") for row in rows),
            "epoch_c_sample_count": sum(int(row["quadrant"] == "C") for row in rows),
            "epoch_d_sample_count": sum(int(row["quadrant"] == "D") for row in rows),
            "epoch_answer_only_sample_count": sum(
                int(row["quadrant"] == "A") for row in rows
            ),
            "route1_eval_null_mode": eval_null_mode,
            "local_wrong_candidate_mask": local_wrong_candidate_mask,
            "local_wrong_candidate_mask_sha256": canonical_json_sha256(
                local_wrong_candidate_mask
            ),
            "specificity_donor_record_ids": list(window_donor_record_ids),
            "specificity_donor_record_ids_sha256": canonical_json_sha256(
                window_donor_record_ids
            ),
            # Compatibility aliases for older resume/report readers.  The
            # donor bank is now the same optimizer-window bank for every GAS
            # microstep; it is no longer a per-microstep z snapshot.
            "global_microstep_z_bank_record_ids": [
                list(window_donor_record_ids) for _ in range(accumulation)
            ],
            "global_microstep_z_bank_record_ids_sha256": canonical_json_sha256(
                [list(window_donor_record_ids) for _ in range(accumulation)]
            ),
            "phase_terminal": bool(
                optimizer_update
                == int(
                    terminal_optimizer_update
                    if terminal_optimizer_update is not None
                    else (
                        max_optimizer_updates
                        if max_optimizer_updates is not None
                        else int(optimizer_update_offset) + steps_per_epoch
                    )
                )
            ),
            "boundary_token_count": int(boundary_token_count),
        }
        yield records, state


def _assert_runtime_sample_microbatches(
    global_group_microbatches: Sequence[Sequence[str]],
    *,
    expected_microstep_global_batch: int,
    global_record_ids: Sequence[str],
) -> None:
    # Different trajectory records may share a prompt in the shuffled window.
    # Only record identities, not prompt identities, must remain unique.
    if (
        int(expected_microstep_global_batch) <= 0
        or not global_group_microbatches
        or any(
            len(groups) != int(expected_microstep_global_batch)
            for groups in global_group_microbatches
        )
    ):
        raise RuntimeError(
            "Route1 distributed sample microbatch has an invalid sample count"
        )
    expected_records = len(global_group_microbatches) * int(
        expected_microstep_global_batch
    )
    if len(global_record_ids) != expected_records:
        raise RuntimeError("Route1 optimizer window has an invalid record count")
    if len(set(global_record_ids)) != expected_records:
        raise RuntimeError("Route1 optimizer window repeats a record")


def _initial_route1_sampler_state(
    *,
    epoch: int,
    seed: int,
    eval_null_mode: str,
    population: str = "staged",
    compute_course: bool = True,
    compute_match: bool = True,
    compute_specific: bool = True,
    specificity_donors_per_owner: int = 2,
    distillation_populations: Sequence[str] = ("B", "C"),
) -> dict[str, Any]:
    if population not in {"staged", "answer-only", "gold-reference"}:
        raise ValueError("Route1 population must be staged or answer-only")
    active_distillation_populations = frozenset(
        str(value).upper()
        for value in (
            distillation_populations
            if not isinstance(distillation_populations, str)
            else tuple(distillation_populations)
        )
    )
    if active_distillation_populations not in {
        frozenset({"C"}),
        frozenset({"B", "C"}),
        frozenset({"G"}),
    }:
        raise ValueError("distillation_populations must select C or B+C")
    return {
        **artifact_header(ROUTE1_OCCURRENCE_SAMPLER_SCHEMA),
        "active_domain_schema": ROUTE1_ACTIVE_DOMAIN_SCHEMA,
        "physical_cost_policy_schema": ROUTE1_PHYSICAL_COST_POLICY_SCHEMA,
        "weighting_unit": OCCURRENCE_WEIGHTING_UNIT,
        "batch_unit": OCCURRENCE_BATCH_UNIT,
        "epoch": int(epoch),
        "global_batch_index": -1,
        "seed": int(seed),
        "specificity_sampling_policy": SPECIFICITY_SAMPLING_POLICY,
        "specificity_donors_per_owner": validate_donor_limit(
            specificity_donors_per_owner
        ),
        "global_prompt_group_ids": [],
        "global_prompt_group_ids_sha256": canonical_json_sha256([]),
        "global_microstep_prompt_group_ids": [],
        "global_microstep_prompt_group_ids_sha256": canonical_json_sha256([]),
        "local_prompt_group_ids": [],
        "local_prompt_group_ids_sha256": canonical_json_sha256([]),
        "global_course_sample_count": 0,
        "global_course_c_sample_count": 0,
        "global_course_direct_side_sample_count": 0,
        "course_reduction": route1_course_reduction(int(epoch)),
        "global_native_b_sample_count": 0,
        "global_native_c_sample_count": 0,
        "distillation_reduction": "global-configured-native-window-empirical-proportion",
        "distillation_populations": sorted(active_distillation_populations),
        "global_match_sample_count": 0,
        "global_specific_sample_count": 0,
        "specificity_eligible_pair_count": 0,
        "specificity_selected_pair_count": 0,
        "specificity_weight_scale": 0.0,
        "course_retained_fraction": 1.0 if int(epoch) == 0 else 0.0,
        "global_microstep_native_sample_counts": [],
        "optimizer_update": 0,
        "optimizer_step": -1,
        "optimizer_update_offset": 0,
        "compute_course": bool(compute_course),
        "compute_match": bool(compute_match),
        "compute_specific": bool(compute_specific),
        "local_wrong_candidate_mask": [],
        "local_wrong_candidate_mask_sha256": canonical_json_sha256([]),
        "specificity_donor_record_ids": [],
        "specificity_donor_record_ids_sha256": canonical_json_sha256([]),
        "global_microstep_z_bank_record_ids": [],
        "global_microstep_z_bank_record_ids_sha256": canonical_json_sha256([]),
        "route1_population_phase": (
            "c_native_distillation" if not compute_course else "fixed_b_c_d_course"
        ),
        "answer_course_geometry": answer_course_geometry(int(epoch)),
        "answer_course_batch_count": 0,
        "route1_population": population,
        "d_answer_provenance_schema": D_ANSWER_PROVENANCE_SCHEMA,
        "d_answer_eos_rule": D_ANSWER_EOS_RULE,
        "epoch_b_sample_count": 0,
        "epoch_c_sample_count": 0,
        "epoch_d_sample_count": 0,
        "epoch_answer_only_sample_count": 0,
        "route1_eval_null_mode": normalize_route1_null_mode(eval_null_mode),
        "epoch_occurrence_count": 0,
        "retained_occurrence_count": 0,
        "global_occurrence_record_ids": [],
        "global_occurrence_record_ids_sha256": canonical_json_sha256([]),
        "local_occurrence_record_ids": [],
        "local_occurrence_record_ids_sha256": canonical_json_sha256([]),
        "global_length_bucket_keys": [],
        "global_length_bucket_keys_sha256": canonical_json_sha256([]),
        "retained_occurrence_record_ids_sha256": canonical_json_sha256([]),
        "dropped_occurrence_record_ids_sha256": canonical_json_sha256([]),
        "dropped_occurrence_count": 0,
        "epoch_batch_order_sha256": canonical_json_sha256([]),
        "rank_assignment_sha256": canonical_json_sha256([]),
        "rank_estimated_costs": [],
        "global_prompt_group_count": 0,
        "global_sample_count": 0,
        "global_occurrence_count": 0,
        "route1_local_samples": 0,
        "route1_gradient_accumulation_steps": 0,
        "route1_microstep_global_samples": 0,
        "local_microstep_sample_record_ids": [],
        "phase_terminal": False,
    }


def _expected_route1_sampler_state(
    rows: Sequence[Mapping[str, Any]],
    *,
    epochs: Sequence[int],
    completed_updates: int,
    seed: int,
    rank: int,
    world_size: int,
    local_samples: int,
    gradient_accumulation_steps: int,
    boundary_token_count: int,
    population: str,
    compute_course: bool,
    compute_match: bool,
    compute_specific: bool,
    eval_null_mode: str,
    max_optimizer_updates: int | None = None,
    specificity_donors_per_owner: int = 2,
    distillation_populations: Sequence[str] = ("C",),
    course_epochs: float = 0.0,
    course_updates: int | None = None,
) -> dict[str, Any]:
    if completed_updates < 0:
        raise ValueError("completed update count is negative")
    clock = _route1_optimizer_course_clock(
        rows,
        epochs=epochs,
        seed=seed,
        world_size=world_size,
        local_samples=local_samples,
        gradient_accumulation_steps=gradient_accumulation_steps,
        boundary_token_count=boundary_token_count,
        population=population,
        compute_course=compute_course,
        compute_match=compute_match,
        compute_specific=compute_specific,
        max_optimizer_updates=max_optimizer_updates,
    )
    if completed_updates > int(clock.total_updates):
        raise ValueError("resume step exceeds the active Route1 course")
    state = _initial_route1_sampler_state(
        specificity_donors_per_owner=specificity_donors_per_owner,
        epoch=int(epochs[0]),
        seed=seed,
        eval_null_mode=eval_null_mode,
        population=population,
        compute_course=compute_course,
        compute_match=compute_match,
        compute_specific=compute_specific,
        distillation_populations=distillation_populations,
    )
    observed = 0
    for epoch in epochs:
        for _, candidate in _epoch_batches(
            rows,
            epoch=epoch,
            seed=seed,
            specificity_donors_per_owner=specificity_donors_per_owner,
            distillation_populations=distillation_populations,
            rank=rank,
            world_size=world_size,
            local_samples=local_samples,
            gradient_accumulation_steps=gradient_accumulation_steps,
            boundary_token_count=boundary_token_count,
            population=population,
            compute_course=compute_course,
            compute_match=compute_match,
            compute_specific=compute_specific,
            eval_null_mode=eval_null_mode,
            max_optimizer_updates=max_optimizer_updates,
            course_epochs=course_epochs,
            course_updates=course_updates,
            course_steps_per_epoch=clock.steps_for_epoch(int(epochs[0])),
            terminal_epoch=int(epochs[-1]),
            optimizer_update_offset=clock.offset_for_epoch(int(epoch)),
            terminal_optimizer_update=clock.terminal_optimizer_update,
        ):
            if observed == completed_updates:
                return state
            state = candidate
            observed += 1
    if observed != completed_updates:
        raise ValueError("resume step is outside the deterministic sampler course")
    return state


def _all_ranks_true(value: bool, *, device: torch.device) -> bool:
    flag = torch.tensor([int(bool(value))], dtype=torch.int32, device=device)
    if torch_distributed.is_initialized():
        torch_distributed.all_reduce(flag, op=torch_distributed.ReduceOp.MIN)
    return bool(flag.item())


def _clip_owner_gradients_for_update(
    backend: Any,
    owner_parameters: Sequence[Any],
    *,
    max_norm: float,
    device: torch.device,
    preclip_observer: Any | None = None,
) -> tuple[float, float]:
    """Admit one backend-native owner clip before the optimizer update.

    The pre-clip owner check is the sole full-gradient scan.  Both maintained
    backends then clip without an intervening user callback: replicated DDP
    uses ``error_if_nonfinite=True`` and ZeRO-1 derives its coefficient from a
    finite adapter-owned global norm.  Consequently the post-clip boundary
    needs one fused scalar/bound verdict, not another tensor scan plus three
    separate collectives.
    """

    if not backend.owner_gradients_finite(owner_parameters):
        raise FloatingPointError(
            "ThinkBridge missing/non-finite owner gradient at pre-clip; "
            "optimizer step withheld"
        )
    if preclip_observer is not None:
        preclip_observer()
    maximum_norm = float(max_norm)
    preclip, postclip = backend.clip_owner_gradients(
        owner_parameters,
        max_norm=maximum_norm,
    )
    preclip_value = float(preclip)
    postclip_value = float(postclip)
    norm_tolerance = max(1.0e-6, maximum_norm * 1.0e-6)
    locally_admissible = (
        math.isfinite(preclip_value)
        and math.isfinite(postclip_value)
        and 0.0 <= postclip_value <= maximum_norm + norm_tolerance
    )
    if not _all_ranks_true(locally_admissible, device=device):
        raise FloatingPointError(
            "ThinkBridge non-finite or unbounded owner gradient norm after clipping; "
            "optimizer step withheld"
        )
    return preclip_value, postclip_value


def _synchronized_tensors_finite_or_raise(
    tensors: Sequence[torch.Tensor], *, device: torch.device, label: str
) -> None:
    if tensors:
        flag = (
            torch.stack(
                [
                    torch.isfinite(value.detach()).all().to(device=device)
                    for value in tensors
                ]
            )
            .all()
            .to(dtype=torch.int32)
            .reshape(1)
        )
    else:
        flag = torch.zeros((1,), dtype=torch.int32, device=device)
    if torch_distributed.is_initialized():
        torch_distributed.all_reduce(flag, op=torch_distributed.ReduceOp.MIN)
    if not flag.item():
        raise FloatingPointError(
            f"ThinkBridge non-finite {label}; optimizer step withheld"
        )


def _synchronized_route1_service_failure_or_raise(
    local_error: BaseException | None,
    *,
    device: torch.device,
) -> None:
    """Propagate service or overlapped trainer failure before prefix scoring."""

    if local_error is not None:
        from think_bridge.training.vllm_client import route1_service_failure_event

        rank = (
            int(torch_distributed.get_rank())
            if torch_distributed.is_initialized()
            else 0
        )
        print(
            json.dumps(
                route1_service_failure_event(local_error, rank=rank),
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )

    failed = torch.tensor(
        [int(local_error is not None)], dtype=torch.int32, device=device
    )
    if torch_distributed.is_initialized():
        torch_distributed.all_reduce(failed, op=torch_distributed.ReduceOp.MAX)
    if failed.item():
        raise RuntimeError(
            "Bridge Route1 generation/prefix preparation failed on at least one trainer rank "
            "(inspect the original exception: this may be a trainer-side failure); "
            "all ranks withheld prefix scoring and optimizer progress"
        ) from local_error


def _nested_tensors_finite(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value.detach()).all())
    if isinstance(value, Mapping):
        return all(_nested_tensors_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_nested_tensors_finite(item) for item in value)
    return True


def _synchronized_update_state_finite_or_raise(
    parameters: Sequence[nn.Parameter],
    optimizer: Any,
    *,
    device: torch.device,
    label: str,
) -> None:
    local = bool(parameters) and all(
        bool(torch.isfinite(parameter.detach()).all()) for parameter in parameters
    )
    candidate = optimizer
    state = getattr(candidate, "state", None)
    if state is None and getattr(candidate, "optimizer", None) is not None:
        candidate = candidate.optimizer
        state = getattr(candidate, "state", None)
    if state is not None:
        local = local and _nested_tensors_finite(state)
    groups = getattr(candidate, "param_groups", ())
    local = local and all(
        math.isfinite(float(group.get("lr", float("nan")))) for group in groups
    )
    if not _all_ranks_true(local, device=device):
        raise FloatingPointError(
            f"ThinkBridge non-finite parameter/optimizer state at {label}; "
            "checkpoint and selection withheld"
        )


def _training_step_metrics(
    *,
    result: Any,
    route: str,
    sampler_state: Mapping[str, Any],
    route1_global_metrics: Mapping[str, float | int] | None = None,
) -> dict[str, float | int | None]:
    if route1_global_metrics is None:
        raise RuntimeError("Route1 detached global training metrics are missing")
    metrics: dict[str, float | int | None] = {
        "loss_total": float(route1_global_metrics["loss_total"]),
        "loss_ce": float(route1_global_metrics["loss_answer_course"]),
        "loss_match": float(route1_global_metrics["loss_match"]),
        "loss_specific": float(route1_global_metrics["loss_specific"]),
        "valid_course_tokens": int(route1_global_metrics["valid_course_tokens"]),
        "batch_sample_count": int(sampler_state["global_sample_count"]),
        "batch_unique_group_count": int(sampler_state["global_prompt_group_count"]),
        "course_active_count": int(result.course_active_count),
        "course_c_active_count": int(result.course_c_active_count),
        "course_b_d_side_active_count": int(result.course_b_d_side_active_count),
        "match_active_count": int(result.match_active_count),
        "specific_active_count": int(result.specific_active_count),
    }
    for side in ("b", "c"):
        count = sampler_state.get(f"global_native_{side}_sample_count")
        metrics[f"distill_{side}_sample_count"] = None if count is None else int(count)
    for name in (
        "specificity_eligible_pair_count",
        "specificity_selected_pair_count",
        "specificity_donors_per_owner",
        "specificity_same_prompt_record_pair_count",
    ):
        value = sampler_state.get(name)
        metrics[name] = None if value is None else int(value)
    for name in ("specificity_weight_scale", "course_retained_fraction"):
        value = sampler_state.get(name)
        metrics[name] = None if value is None else float(value)
    return metrics


def _route1_component_gradient_audit(
    *,
    result: Any,
    model: BridgeParallelModel,
    step: int,
) -> dict[str, Any]:
    """Diagnose the first active microstep's role VJPs, not a GAS-window direction."""

    failure_flags: dict[str, bool] = {}
    graph = getattr(result, "audit_graph", None)
    if graph is None:
        failure_flags["missing_audit_graph"] = True
        return {
            "grad_norm_course_z_vjp": 0.0,
            "grad_norm_match_owner_z_vjp": 0.0,
            "grad_norm_specific_owner_z_vjp": 0.0,
            "grad_norm_specific_wrong_donor_z_vjp": 0.0,
            "grad_dot_match_vs_specific_owner_z_vjp": 0.0,
            "specific_objective_active": False,
            "specific_wrong_donor_branch_present": False,
            "owner_z_vjp_cosine_aligned": False,
            "frozen_F_embedding_D": False,
            "frozen_teacher_direct_prefix": False,
            "component_gradient_audit_method": "intermediate_z_vjp",
            "component_gradient_audit_status": "local-collected",
            "component_gradient_audit_step": int(step),
            "_failure_flags": failure_flags,
        }

    def gradients(
        loss: torch.Tensor, tensors: Sequence[torch.Tensor]
    ) -> tuple[tuple[torch.Tensor | None, ...], float]:
        if not tensors:
            return (), 0.0
        values = torch.autograd.grad(
            loss,
            tuple(tensors),
            retain_graph=True,
            allow_unused=True,
        )
        squared = loss.new_zeros((), dtype=torch.float64)
        for value in values:
            if value is not None:
                squared = squared + value.detach().double().square().sum()
        return tuple(values), float(squared.sqrt().item())

    try:
        _course_gradients, course_norm = gradients(
            result.loss_answer_course,
            graph.course_z if result.course_active_count > 0 else (),
        )
        match_gradients, match_norm = gradients(
            result.loss_match,
            graph.match_owner_z if result.match_active_count > 0 else (),
        )
        specific_gradients, specific_norm = gradients(
            result.loss_specific,
            (graph.specific_owner_z if result.specific_active_count > 0 else ()),
        )
        _donor_gradients, donor_norm = gradients(
            result.loss_specific,
            graph.specific_donor_z,
        )
        local_owner_z_views_present = bool(
            graph.match_owner_z or graph.specific_owner_z
        )
        owner_z_vjp_cosine_expected = bool(
            local_owner_z_views_present
            and result.match_active_count > 0
            and result.specific_active_count > 0
        )
        owner_z_vjp_cosine_aligned = (
            owner_z_vjp_views_strictly_aligned(
                graph.match_owner_z, graph.specific_owner_z
            )
            if owner_z_vjp_cosine_expected
            else False
        )
        if owner_z_vjp_cosine_expected and not owner_z_vjp_cosine_aligned:
            failure_flags["owner_z_vjp_alignment_unproven"] = True
        match_specific_dot_tensor = result.loss.new_zeros((), dtype=torch.float64)
        if owner_z_vjp_cosine_aligned:
            for match_gradient, specific_gradient in zip(
                match_gradients, specific_gradients
            ):
                if match_gradient is not None and specific_gradient is not None:
                    match_specific_dot_tensor = (
                        match_specific_dot_tensor
                        + (
                            match_gradient.detach().double()
                            * specific_gradient.detach().double()
                        ).sum()
                    )
        match_specific_dot = float(match_specific_dot_tensor.item())
        specific_objective_active = bool(result.specific_active_count > 0)
        specific_wrong_donor_branch_present = bool(graph.specific_donor_z)
        frozen_modules = all(
            not parameter.requires_grad for parameter in model.executor.parameters()
        ) and all(
            not parameter.requires_grad
            for parameter in model.executor.get_input_embeddings().parameters()
        )
        failure_flags.update(
            course_z_vjp_missing=(bool(graph.course_z) and course_norm <= 0.0),
            match_owner_z_vjp_missing=(bool(graph.match_owner_z) and match_norm <= 0.0),
            specific_owner_z_vjp_missing=(
                bool(graph.specific_owner_z)
                and any(g is None for g in specific_gradients)
            ),
            specific_wrong_donor_z_vjp_missing=(
                bool(graph.specific_donor_z)
                and any(g is None for g in _donor_gradients)
            ),
            forbidden_frozen_owner=(
                not graph.reference_tensors_frozen or not frozen_modules
            ),
        )
    except Exception as error:
        course_norm = match_norm = specific_norm = 0.0
        donor_norm = match_specific_dot = 0.0
        specific_objective_active = False
        specific_wrong_donor_branch_present = False
        owner_z_vjp_cosine_aligned = False
        frozen_modules = False
        failure_flags["local_computation_error"] = True
        local_error_type = type(error).__name__
    else:
        local_error_type = None
    return {
        "grad_norm_course_z_vjp": course_norm,
        "grad_norm_match_owner_z_vjp": match_norm,
        "grad_norm_specific_owner_z_vjp": specific_norm,
        "grad_norm_specific_wrong_donor_z_vjp": donor_norm,
        "grad_dot_match_vs_specific_owner_z_vjp": match_specific_dot,
        "specific_objective_active": specific_objective_active,
        "specific_wrong_donor_branch_present": (specific_wrong_donor_branch_present),
        "owner_z_vjp_cosine_aligned": owner_z_vjp_cosine_aligned,
        "frozen_F_embedding_D": bool(frozen_modules),
        "frozen_teacher_direct_prefix": bool(graph.reference_tensors_frozen),
        "component_gradient_audit_method": "intermediate_z_vjp",
        "component_gradient_audit_status": "local-collected",
        "component_gradient_audit_step": int(step),
        "_failure_flags": failure_flags,
        "_local_error_type": local_error_type,
    }


def _globalize_route1_component_gradient_audit(
    audit: Mapping[str, Any], *, device: torch.device
) -> dict[str, Any]:
    """Reduce one fixed state vector, then make the verdict on every rank."""

    try:
        encoded = encode_route1_component_audit_reduction(
            norm_values=(
                float(audit["grad_norm_course_z_vjp"]),
                float(audit["grad_norm_match_owner_z_vjp"]),
                float(audit["grad_norm_specific_owner_z_vjp"]),
            ),
            specific_objective_active=bool(audit["specific_objective_active"]),
            failure_flags=dict(audit.get("_failure_flags", {})),
            match_specific_owner_z_vjp_dot=float(
                audit["grad_dot_match_vs_specific_owner_z_vjp"]
            ),
            specific_wrong_donor_z_vjp_norm=float(
                audit["grad_norm_specific_wrong_donor_z_vjp"]
            ),
            specific_wrong_donor_branch_present=bool(
                audit["specific_wrong_donor_branch_present"]
            ),
            owner_z_vjp_cosine_aligned=bool(audit["owner_z_vjp_cosine_aligned"]),
        )
    except Exception:
        encoded = encode_route1_component_audit_reduction(
            norm_values=(0.0, 0.0, 0.0),
            specific_objective_active=False,
            failure_flags={"local_computation_error": True},
            match_specific_owner_z_vjp_dot=0.0,
            specific_wrong_donor_z_vjp_norm=0.0,
            specific_wrong_donor_branch_present=False,
            owner_z_vjp_cosine_aligned=False,
        )
    values = torch.tensor(encoded, dtype=torch.float64, device=device)
    if torch_distributed.is_initialized():
        torch_distributed.all_reduce(values, op=torch_distributed.ReduceOp.SUM)
    report = dict(
        finalize_route1_component_audit_reduction(
            values.tolist(),
            step=int(audit.get("component_gradient_audit_step", 0)),
        )
    )
    report["component_gradient_audit_method"] = "intermediate_z_vjp"
    return report


def _route1_global_step_metrics(
    *, result: Any, device: torch.device, loss_weights: Sequence[float]
) -> dict[str, Any]:
    """Reconstruct detached global log values without touching backward."""

    counts = (
        int(result.course_active_count),
        int(result.match_active_count),
        int(result.specific_active_count),
    )
    world_size = (
        int(torch_distributed.get_world_size())
        if torch_distributed.is_initialized()
        else 1
    )
    if world_size <= 0:
        values = torch.tensor(
            (0.0, 0.0, 0.0, 0.0, 1.0),
            dtype=torch.float64,
            device=device,
        )
    else:
        losses = torch.stack(
            (
                result.loss_answer_course.detach().reshape(()),
                result.loss_match.detach().reshape(()),
                result.loss_specific.detach().reshape(()),
            )
        ).to(device=device, dtype=torch.float64)
        count_values = torch.tensor(counts, dtype=torch.int64, device=device)
        local_valid_course_tokens = (
            result.valid_course_tokens.detach()
            .to(device=device, dtype=torch.float64)
            .reshape(())
        )
        finite = torch.isfinite(losses)
        positive_count = count_values > 0
        invalid = ~finite | (count_values < 0) | ((count_values == 0) & losses.ne(0.0))
        means = torch.where(
            finite & positive_count,
            losses / float(world_size),
            torch.zeros_like(losses),
        )
        values = torch.cat(
            (
                means,
                local_valid_course_tokens.reshape((1,)),
                (
                    invalid.any()
                    | ~torch.isfinite(local_valid_course_tokens)
                    | local_valid_course_tokens.lt(0.0)
                )
                .to(dtype=torch.float64)
                .reshape((1,)),
            )
        )
    if torch_distributed.is_initialized():
        torch_distributed.all_reduce(values, op=torch_distributed.ReduceOp.SUM)
    return dict(
        finalize_route1_log_reduction(
            values.tolist(), global_counts=counts, loss_weights=loss_weights
        )
    )


def _globalize_route1_specificity_step_stats(
    *, result: Any, device: torch.device
) -> dict[str, float | int | None]:
    """Report sampled donor coverage and teacher-relative distances."""
    stats = result.specificity_stats
    if not isinstance(stats, Route1SpecificityStats):
        raise RuntimeError("Route1 result lacks specificity coverage")
    world = (
        torch_distributed.get_world_size() if torch_distributed.is_initialized() else 1
    )
    names = (
        "true_kl",
        "no_z_kl",
        "wrong_kl",
        "margin_active_fraction",
        "soft_weight_mean",
        "negative_cap_fraction",
        "effective_wrong_kl",
    )
    scalar = lambda value: (
        torch.as_tensor(value, device=device, dtype=torch.float64).detach().reshape(())
    )
    fields = [getattr(stats, name) for name in names]
    values = torch.stack(
        [
            scalar(value)
            for value in (
                stats.owner_count,
                stats.covered_owner_count,
                stats.pair_count,
                *(value if value is not None else 0.0 for value in fields),
                *(int(value is not None) for value in fields),
            )
        ]
    )
    if torch_distributed.is_initialized():
        torch_distributed.all_reduce(values, op=torch_distributed.ReduceOp.SUM)
    raw = values.tolist()
    owners, covered, pairs = (int(x) for x in raw[:3])
    if owners != int(result.specific_active_count):
        raise RuntimeError(
            "specificity coverage differs from global eligible owner count"
        )
    presence = raw[3 + len(names) :]
    if any(count not in (0, world) for count in presence):
        raise RuntimeError("specificity audit differs between ranks")
    output = {
        "specificity_owner_count": owners,
        "specificity_covered_owner_count": covered,
        "specificity_pair_count": pairs,
        "specificity_owner_coverage": covered / owners if owners else 0.0,
    }
    output.update(
        {
            "specificity_" + name: value if count else None
            for name, value, count in zip(names, raw[3 : 3 + len(names)], presence)
        }
    )
    return output


def _aggregate_route1_microstep_results(results: Sequence[Any]) -> Any:
    """Build one detached optimizer-window result for logging/auditing."""

    values = tuple(results)
    if not values:
        raise ValueError("Route1 optimizer window has no microstep results")
    first = values[0]
    count_fields = (
        "course_active_count",
        "course_c_active_count",
        "course_b_d_side_active_count",
        "match_active_count",
        "specific_active_count",
    )
    if any(
        any(
            int(getattr(value, field)) != int(getattr(first, field)) for value in values
        )
        for field in count_fields
    ):
        raise RuntimeError(
            "Route1 microsteps disagree on optimizer-window denominators"
        )

    def detached_sum(field: str) -> Any:
        return sum(
            (getattr(value, field).detach() for value in values[1:]),
            getattr(first, field).detach(),
        )

    specificity_values = tuple(
        getattr(value, "specificity_stats", None) for value in values
    )
    if not all(
        isinstance(value, Route1SpecificityStats) for value in specificity_values
    ):
        raise RuntimeError("Route1 microstep lacks executable specificity statistics")

    def aggregate_specificity_distance(field):
        tensors = [getattr(value, field) for value in specificity_values]
        if all(value is None for value in tensors):
            return None
        if any(value is None for value in tensors):
            raise RuntimeError(
                "Specificity distance audit differs between GAS microsteps"
            )
        return sum((value.detach() for value in tensors[1:]), tensors[0].detach())

    specificity_stats = Route1SpecificityStats(
        owner_count=sum(value.owner_count for value in specificity_values),
        covered_owner_count=sum(
            value.covered_owner_count for value in specificity_values
        ),
        pair_count=sum(value.pair_count for value in specificity_values),
        true_kl=aggregate_specificity_distance("true_kl"),
        no_z_kl=aggregate_specificity_distance("no_z_kl"),
        wrong_kl=aggregate_specificity_distance("wrong_kl"),
        margin_active_fraction=aggregate_specificity_distance("margin_active_fraction"),
        soft_weight_mean=aggregate_specificity_distance("soft_weight_mean"),
        negative_cap_fraction=aggregate_specificity_distance("negative_cap_fraction"),
        effective_wrong_kl=aggregate_specificity_distance("effective_wrong_kl"),
    )

    return SimpleNamespace(
        loss=detached_sum("loss"),
        loss_answer_course=detached_sum("loss_answer_course"),
        loss_match=detached_sum("loss_match"),
        loss_specific=detached_sum("loss_specific"),
        **{field: int(getattr(first, field)) for field in count_fields},
        valid_course_tokens=detached_sum("valid_course_tokens"),
        physical_chunk_count=sum(int(value.physical_chunk_count) for value in values),
        c_view_count=sum(int(value.c_view_count) for value in values),
        legal_wrong_pair_count=sum(
            int(value.legal_wrong_pair_count) for value in values
        ),
        wrong_physical_chunk_count=sum(
            int(value.wrong_physical_chunk_count) for value in values
        ),
        specificity_stats=specificity_stats,
        rollout_request_count=sum(int(value.rollout_request_count) for value in values),
        rollout_token_count=sum(int(value.rollout_token_count) for value in values),
        rollout_service_batch_rows=sum(
            int(value.rollout_service_batch_rows) for value in values
        ),
        rollout_service_real_rows=sum(
            int(value.rollout_service_real_rows) for value in values
        ),
        audit_graph=None,
    )


def _write_step_audit(
    *,
    epoch: int,
    step: int,
    total_steps: int,
    owner: str,
    result: Any,
    route: str,
    components: Mapping[str, Any],
    latest_component_audit: Mapping[str, Any] | None,
    latest_component_audit_step: int | None,
    specificity_detail: Mapping[str, float | int | None] | None,
    current_metrics: Mapping[str, float | int | None],
    window_means: Mapping[str, float | int | None],
    step_audit_jsonl: Path | None,
) -> None:
    row: dict[str, Any] = {
        **artifact_header(STEP_AUDIT),
        "event": "step_audit",
        "route": route,
        "weighting_unit": OCCURRENCE_WEIGHTING_UNIT,
        "batch_unit": OCCURRENCE_BATCH_UNIT,
        "epoch": int(epoch),
        "step": int(step),
        "total_steps": int(total_steps),
        "active_owner": owner,
        **{f"instant_{name}": value for (name, value) in current_metrics.items()},
        **window_means,
        **components,
    }
    if latest_component_audit is not None:
        row["latest_component_gradient_audit"] = dict(latest_component_audit)
        row["latest_component_gradient_audit_step"] = int(
            latest_component_audit_step
            if latest_component_audit_step is not None
            else latest_component_audit["component_gradient_audit_step"]
        )
    if specificity_detail is None:
        raise RuntimeError("Route1 step audit lacks specificity detail")
    row.update(
        course_global_valid_tokens=int(current_metrics["valid_course_tokens"]),
        route1_course_sample_count=result.course_active_count,
        route1_course_c_sample_count=result.course_c_active_count,
        route1_course_b_d_side_sample_count=result.course_b_d_side_active_count,
        route1_match_sample_count=result.match_active_count,
        route1_specific_sample_count=result.specific_active_count,
        route1_physical_chunks=result.physical_chunk_count,
        **dict(specificity_detail),
    )
    if step_audit_jsonl is not None:
        append_step_audit(Path(step_audit_jsonl), row)


def _validate_terminal_best_reference(
    ledger: Sequence[Mapping[str, Any]],
    best: tuple[Path, SealedCheckpoint],
) -> str:
    """Match the retained best to validated reports after checkpoint rotation.

    Completed reports outlive non-best weights. Only the selected checkpoint
    must still exist; resolving historical locators must not load old weights.
    The caller validates the report ledger and reads the sealed best pointer.
    """
    selected = best[0].resolve(strict=True)
    if selected != best[1].path:
        raise ValueError("best checkpoint path differs from its sealed reference")
    matching = [
        row
        for row in ledger
        if Path(str(row["checkpoint_path"])).resolve(strict=False) == selected
        and str(row["checkpoint_sha256"]) == best[1].artifact_sha256
    ]
    if len(matching) != 1:
        raise ValueError(
            "training-time best pointer does not resolve to one committed report"
        )
    return str(selected)


def _validate_validation_barrier(
    path: Path,
    *,
    checkpoint_path: Path,
    route: str,
    seed: int,
    epoch: int,
    generation_seed: int,
    metric_policy: Mapping[str, Any],
    checkpoint_sha256: str | None = None,
    completed_step: int | None = None,
) -> None:
    from think_bridge.training.checkpoint_manager import (
        checkpoint_run_directory,
        resolve_run_artifact_locator,
        validate_recorded_checkpoint_location,
    )

    report = _read_json(path)
    if completed_step is None:
        (checkpoint_route, checkpoint_step) = parse_checkpoint_path(checkpoint_path)
        checkpoint_run = checkpoint_run_directory(checkpoint_path)
    else:
        from think_bridge.model.checkpoint_policy import checkpoint_owner_run_directory

        if checkpoint_sha256 is None:
            raise ValueError(
                "completed validation requires its recorded checkpoint hash"
            )
        checkpoint_run = validate_recorded_checkpoint_location(
            checkpoint_owner_run_directory(checkpoint_path),
            checkpoint_path,
            route=route,
            step=completed_step,
        )
        (checkpoint_route, checkpoint_step) = (route, completed_step)
    if route not in {"route1"} or checkpoint_route != route:
        raise ValueError("validation barrier checkpoint route mismatch")
    schema = {"route1": ROUTE1_VALIDATION_REPORT_SCHEMA_VERSION}[route]
    expected = {
        **artifact_header(schema),
        "objective_version": OBJECTIVE_VERSION,
        "method": "bridge",
        "route": route,
        "seed": int(seed),
        "epoch": int(epoch),
        "step": int(checkpoint_step),
        "checkpoint_sha256": checkpoint_artifact_sha256(checkpoint_path)
        if checkpoint_sha256 is None
        else require_sha256(checkpoint_sha256, "checkpoint_sha256"),
        "split": "validation",
        "free_generation": True,
    }
    for field, value in expected.items():
        if report.get(field) != value:
            raise ValueError(f"validation barrier report mismatch: {field}")
    from think_bridge.training.metric_registry import (
        report_metric_policy,
        require_report_metric_policy,
    )

    if require_report_metric_policy(report, route=route) != report_metric_policy(
        route=route, value=metric_policy
    ):
        raise ValueError("validation barrier report metric policy mismatch")
    reported_checkpoint = resolve_run_artifact_locator(
        checkpoint_run,
        report.get("checkpoint_path"),
        label="validation barrier report checkpoint path",
    )
    if reported_checkpoint != Path(checkpoint_path).resolve(strict=False):
        raise ValueError("validation barrier report mismatch: checkpoint_path")
    if (
        isinstance(generation_seed, bool)
        or not isinstance(generation_seed, int)
        or generation_seed < 0
    ):
        raise ValueError("free-generation validation requires a generation seed")
    expected_randomness = validation_randomness_identity(
        route=route, base_seed=generation_seed
    )
    if report.get("validation_randomness") != expected_randomness:
        raise ValueError("validation barrier evaluation-protocol identity mismatch")
    metrics = report.get("gate_metrics")
    if not isinstance(metrics, dict):
        raise ValueError("validation barrier lacks real free metrics")
    metric_route = "Route1"
    try:
        validate_route1_gate_metric_schema(metrics)
    except ValueError as error:
        raise ValueError(
            f"validation barrier {metric_route} metric schema mismatch: {error}"
        ) from error
    if metrics["step"] != report.get("step"):
        raise ValueError("validation barrier metric/checkpoint step mismatch")
    if metrics["seed"] != seed:
        raise ValueError("validation barrier metric seed mismatch")
    if metrics["judge_path"] != "think_bridge.eval.answer_match.judge_answer":
        raise ValueError("validation barrier judge path mismatch")
    diagnostics = report.get("diagnostics_executed", True)
    if diagnostics != metrics.get("diagnostics_executed", True):
        raise ValueError("validation barrier diagnostic modes differ")
    actual_counts = metrics.get("wrong_donor_counts")
    if (
        actual_counts is not None
        and int(metrics["donor_count"]) != max(actual_counts, default=0)
        or metrics["bootstrap_ci_generated"]
        is not (metrics.get("robust_g1_ci_low") is not None)
    ):
        raise ValueError("validation barrier donor/bootstrap identity mismatch")
    paired_rows = report.get("paired_rows")
    if not isinstance(paired_rows, list) or not paired_rows:
        raise ValueError("validation barrier lacks paired free-evaluation rows")
    paired_domain = [
        {
            "record_id": row["record_id"],
            "prompt_group_id": row["prompt_group_id"],
            "need_z": row["need_z"],
            "direct_correct": row["direct_correct"],
            "population": row["population"],
            "need_z_definition": NEED_Z_COHORT_DEFINITION,
        }
        for row in paired_rows
    ]
    paired_sha256 = canonical_json_sha256(paired_domain)
    require_sha256(str(metrics["paired_row_sha256"]), "paired_row_sha256")
    if (
        metrics["paired_row_sha256"] != paired_sha256
        or report.get("paired_row_sha256") != paired_sha256
    ):
        raise ValueError("validation barrier paired-row identity mismatch")
    if report.get("need_z_cohort_definition") != NEED_Z_COHORT_DEFINITION:
        raise ValueError("validation barrier lacks direct/no-think cohort identity")


def _route1_diagnostic_mode(
    *, route: str, max_optimizer_updates: int | None, requested: bool
) -> bool:
    """Keep the Route1 duration cap orthogonal to selector policy."""

    diagnostic_only = bool(requested)
    if diagnostic_only and (route != "route1" or max_optimizer_updates is None):
        raise ValueError(
            "Route1 diagnostic-only mode requires Route1 and an explicit update cap"
        )
    return diagnostic_only


def _active_route1_specificity_tau(specific_weight: float, raw_tau: Any) -> float:
    """Validate the stored scalar; active InfoNCE uses specificity_temperature."""
    tau = float(raw_tau)
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("specificity_tau must be finite and positive")
    return tau


def run_training(config: TrainingConfig, arguments: Any) -> int:
    from think_bridge.training.distributed_artifacts import (
        run_owned_process_group_entry,
    )

    def run_body(context: Any, resource_stack: Any) -> int:
        world_size, rank, device = context
        return _run_training_owned_process_group(
            config,
            arguments,
            world_size=world_size,
            rank=rank,
            device=device,
            resource_stack=resource_stack,
        )

    return run_owned_process_group_entry(
        torch_distributed,
        initialize=_distributed_context,
        body=run_body,
    )


def _run_training_owned_process_group(
    config: TrainingConfig,
    arguments: Any,
    *,
    world_size: int,
    rank: int,
    device: torch.device,
    resource_stack: Any,
) -> int:
    route = str(arguments.route)
    if route not in {"route1"}:
        raise ValueError("internal Bridge training route is invalid")
    route1_like = True
    route1_population = str(getattr(arguments, "route1_population", "staged"))

    def resolved_route1_float(name: str, default: float) -> float:
        value = getattr(arguments, name, None)
        return float(default if value is None else value)

    route1_course_weight = resolved_route1_float(
        "route1_course_weight", getattr(config, "route1_course_weight", 1.0)
    )
    route1_match_weight = resolved_route1_float(
        "route1_match_weight", getattr(config, "route1_match_weight", 1.0)
    )
    route1_specific_weight = resolved_route1_float(
        "route1_specific_weight", getattr(config, "route1_specific_weight", 1.0)
    )
    raw_route1_specificity_tau = getattr(arguments, "route1_specificity_tau", None)
    if raw_route1_specificity_tau is None:
        raw_route1_specificity_tau = getattr(config, "route1_specificity_tau", 0.1)
    if any(
        (
            not math.isfinite(value) or value < 0.0
            for value in (
                route1_course_weight,
                route1_match_weight,
                route1_specific_weight,
            )
        )
    ):
        raise ValueError("Route1 loss weights must be finite and nonnegative")
    route1_course_epochs = float(getattr(config, "route1_course_epochs", 0.0))
    route1_course_steps = getattr(config, "route1_course_steps", None)
    route1_compute_course = route1_course_weight > 0.0
    route1_compute_match = route1_match_weight > 0.0
    route1_compute_specific = route1_specific_weight > 0.0
    route1_specificity_tau = _active_route1_specificity_tau(
        route1_specific_weight, raw_route1_specificity_tau
    )
    route1_distillation_populations = str(
        getattr(config, "route1_distillation_populations", "BC")
    ).upper()
    if route1_distillation_populations not in {"C", "BC", "G"}:
        raise ValueError("Route1 distillation populations must be C or BC")
    route1_eval_null_mode = normalize_route1_null_mode(
        getattr(arguments, "route1_eval_null_mode", "direct")
    )
    route1_max_optimizer_updates = getattr(
        arguments, "route1_max_optimizer_updates", None
    )
    if route1_max_optimizer_updates is not None:
        route1_max_optimizer_updates = int(route1_max_optimizer_updates)
        if route1_max_optimizer_updates <= 0:
            raise ValueError("Route1 max optimizer updates must be positive")
    if route1_population not in {"staged", "answer-only", "gold-reference"}:
        raise ValueError("Route1 population must be staged or answer-only")
    if not any((route1_compute_course, route1_compute_match, route1_compute_specific)):
        raise ValueError("Route1 requires at least one positive loss weight")
    diagnostic_only = _route1_diagnostic_mode(
        route=route,
        max_optimizer_updates=route1_max_optimizer_updates,
        requested=bool(getattr(arguments, "route1_diagnostic_only", False)),
    )
    manifest_path = Path(arguments.manifest)
    exact_resume_identity_sha256 = str(arguments.exact_resume_identity_sha256)
    require_sha256(exact_resume_identity_sha256, "exact_resume_identity_sha256")
    shared_manifest = require_manifest_fields(_read_json(manifest_path))
    if arguments.target_index is None:
        raise ValueError("Bridge training requires the explicit target-index authority")
    if arguments.checkpoint_dir is None:
        raise ValueError("internal ThinkBridge training requires checkpoint-dir")
    runtime_sidecar_run = Path(
        getattr(arguments, "runtime_sidecar_run", None) or arguments.checkpoint_dir
    )
    from think_bridge.training.manifest_identity import load_target_index_binding

    manifest = load_target_index_binding(
        Path(arguments.target_index),
        shared_manifest,
        model_family=config.model_family,
        route2_content_capacity=int(config.cot_content_capacity),
        runtime_identity=load_route_runtime_identity(runtime_sidecar_run, route=route),
    )
    local_samples = int(
        config.route1_local_samples
        if arguments.route1_local_samples is None
        else arguments.route1_local_samples
    )
    route1_gas = int(
        config.route1_gradient_accumulation_steps
        if arguments.route1_gradient_accumulation_steps is None
        else arguments.route1_gradient_accumulation_steps
    )
    geometry = validate_route1_batch_geometry(
        local_samples=local_samples,
        world_size=world_size,
        gradient_accumulation_steps=route1_gas,
    )
    if geometry.optimizer_global_samples != int(config.route1_optimizer_global_batch):
        raise ValueError("runtime Route1 sample geometry differs from config")
    if arguments.report_dir is None:
        raise ValueError(
            "internal ThinkBridge training requires orchestrator-owned report-dir"
        )
    output = Path(arguments.checkpoint_dir)
    report_output = Path(arguments.report_dir)
    if not is_bridge_isolated_path(output):
        raise ValueError("ThinkBridge checkpoint directory is not run isolated")
    try:
        report_output.resolve(strict=False).relative_to(output.resolve(strict=False))
    except ValueError as exc:
        raise ValueError(
            "ThinkBridge report directory escaped its run directory"
        ) from exc
    if arguments.logging_jsonl is None:
        raise ValueError(
            "internal ThinkBridge training requires orchestrator-owned logging-jsonl"
        )
    try:
        Path(arguments.logging_jsonl).resolve(strict=False).relative_to(
            output.resolve(strict=False)
        )
    except ValueError as exc:
        raise ValueError(
            "ThinkBridge structured log escaped its run directory"
        ) from exc
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        report_output.mkdir(parents=True, exist_ok=True)
    if torch_distributed.is_initialized():
        torch_distributed.barrier()
    records_path = Path(manifest["target_index_artifacts"]["train"]["path"])
    rows = _read_jsonl(records_path)
    validate_prepared_records(
        rows,
        route=route,
        expected_split="train",
        route2_content_capacity=int(manifest["route2_content_capacity"]),
    )
    route1_row_cache = resolve_route1_normalized_rows(
        rows, population=route1_population
    )
    rows_for_training = route1_row_cache if route1_row_cache is not None else rows
    if arguments.resume is None:
        torch.manual_seed(int(arguments.seed))
        random.seed(int(arguments.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(int(arguments.seed))
    (model, tokenizer) = _build_model(
        config,
        manifest,
        route=route,
        device=device,
        runtime_sidecar_dir=runtime_sidecar_run,
        no_progress=bool(getattr(arguments, "no_progress", False)),
    )
    free_generation_provider = None
    if str(arguments.route1_generation_backend) == "vllm":
        if arguments.generation_seed is None or arguments.answer_max_tokens is None:
            raise ValueError(
                "Route1 vLLM generation requires generation_seed and answer_max_tokens"
            )
        from think_bridge.training.vllm_client import (
            Route1AsyncRequestPool,
            BridgeRoute1AsyncPrefixProvider,
            BridgeRoute1ServiceClient,
        )
        from think_bridge.training.vllm_runtime import ROUTE1_MAX_HTTP_RESPONSE_BYTES

        service_client = BridgeRoute1ServiceClient(
            host=str(arguments.route1_vllm_host),
            port=int(arguments.route1_vllm_port),
            timeout_seconds=float(arguments.route1_vllm_request_timeout_seconds),
            teardown_timeout_seconds=float(
                arguments.route1_vllm_shutdown_timeout_seconds
            ),
            max_response_bytes=ROUTE1_MAX_HTTP_RESPONSE_BYTES,
            max_in_flight=int(arguments.route1_vllm_max_in_flight),
            watchdog_interval_seconds=float(
                arguments.route1_vllm_watchdog_interval_seconds
            ),
            wait_reporter=(lambda message: print(message, flush=True))
            if rank == 0
            else None,
        )
        health_error = None
        if not service_client.health():
            health_error = RuntimeError(
                "Route1 vLLM service is not healthy at trainer startup"
            )
        _synchronized_route1_service_failure_or_raise(health_error, device=device)
        request_pool = Route1AsyncRequestPool(
            client=service_client,
            max_pending_microsteps=int(arguments.route1_vllm_max_pending_microsteps),
            request_timeout_seconds=float(
                arguments.route1_vllm_request_timeout_seconds
            ),
            backpressure_timeout_seconds=float(
                arguments.route1_vllm_backpressure_timeout_seconds
            ),
            teardown_timeout_seconds=float(
                arguments.route1_vllm_shutdown_timeout_seconds
            ),
        )
        free_generation_provider = BridgeRoute1AsyncPrefixProvider(
            pool=request_pool,
            run_id=exact_resume_identity_sha256,
            rank=rank,
            physical_chunk_size=int(arguments.route1_vllm_physical_chunk_size),
            generation_seed=int(arguments.generation_seed),
            max_prefix_tokens=int(arguments.answer_max_tokens),
            temperature=model.route1_generation_temperature,
            boundary_ids=tuple(
                (int(value) for value in model.boundary_ids.detach().cpu().tolist())
            ),
            eos_token_id=int(tokenizer.eos_token_id),
            synchronize_failure=lambda error: (
                _synchronized_route1_service_failure_or_raise(error, device=device)
            ),
        )
        resource_stack.callback(free_generation_provider.close)
    if rank == 0:
        route1_gas = int(
            config.route1_gradient_accumulation_steps
            if arguments.route1_gradient_accumulation_steps is None
            else arguments.route1_gradient_accumulation_steps
        )
        print(
            f"[bridge-route1] backend=pytorch-frozen-f world={world_size} logical_local_samples={local_samples} GAS={route1_gas} local_chunk={config.route1_local_chunk_size} prefetch_microsteps={min(int(arguments.route1_vllm_max_pending_microsteps), route1_gas)} wrong_chunk={config.route1_wrong_control_chunk_size} course_steps={route1_course_steps} course_epochs={route1_course_epochs:g} specificity_donors_per_owner={config.route1_specificity_donors_per_owner} specificity_wrong_gradient={config.route1_specificity_wrong_gradient} specificity_loss={config.route1_specificity_loss} specificity_negative_kl_cap={config.route1_specificity_negative_kl_cap} specificity_temperature={config.route1_specificity_temperature:g} r_output={config.reasoner_output_normalization} r_input={config.reasoner_input_mode} f_taps={config.tap_count} r_layers={config.emitter_depth} r_internal_loops={config.reasoner_loop_steps} r_schedule={model.reasoner.layer_schedule} specificity_include_direct={config.route1_specificity_include_direct} distillation_populations={route1_distillation_populations} gradient_checkpointing={config.route1_gradient_checkpointing} eval_backend=hf eval_row_batch={int(arguments.route1_eval_local_row_batch)} eval_r_batch=1 r_compute={config.reasoner_compute_dtype} r_eval_group={config.reasoner_eval_group_size} tf32_matmul={torch.backends.cuda.matmul.allow_tf32} tf32_cudnn={torch.backends.cudnn.allow_tf32} generation_backend={arguments.route1_generation_backend} training_temperature={model.route1_generation_temperature:g} eval_temperature=0 executable_donor_bank=optimizer-window",
            flush=True,
        )
    selected_r_sha256 = "0" * 64
    cache_identity_sha256 = "0" * 64
    phase = "A"
    owner = "R"
    epoch_indices = tuple(range(0, int(config.route1_epochs)))
    if not rows:
        raise RuntimeError("training has no retained correct native occurrences")
    evaluation_bundle: (
        tuple[
            list[dict[str, Any]],
            dict[str, list[dict[str, Any]]] | None,
            dict[str, Any] | None,
        ]
        | None
    ) = None
    evaluation_arguments: SimpleNamespace | None = None
    validation_records = getattr(arguments, "validation_records", None)
    generation_seed = getattr(arguments, "generation_seed", None)
    answer_max_tokens = getattr(arguments, "answer_max_tokens", None)
    max_eval_samples = getattr(arguments, "max_eval_samples", None)
    donor_path = getattr(arguments, "donors", None)
    route_owned_evaluation_fields = (generation_seed, answer_max_tokens, donor_path)
    if validation_records is None:
        if (
            any((value is not None for value in route_owned_evaluation_fields))
            or max_eval_samples is not None
        ):
            raise ValueError("long-lived evaluation inputs require validation records")
    else:
        if any((value is None for value in route_owned_evaluation_fields)):
            raise ValueError(
                "Route1 validation requires generation seed, answer horizon and donors"
            )
        evaluation_arguments = build_long_lived_evaluation_arguments(
            manifest=Path(arguments.manifest),
            target_index=Path(arguments.target_index),
            records=Path(validation_records),
            donors=Path(donor_path),
            route=route,
            metric_policy=arguments.metric_policy,
            seed=int(arguments.seed),
            generation_seed=int(generation_seed),
            route1_eval_local_row_batch=int(
                getattr(arguments, "route1_eval_local_row_batch", 1)
            ),
            no_progress=bool(arguments.no_progress),
            **{"answer_max_tokens": int(answer_max_tokens)},
            route1_null_mode=route1_eval_null_mode,
            **training_evaluation_service_arguments(config, arguments),
            max_eval_samples=None
            if max_eval_samples is None
            else int(max_eval_samples),
        )
        from think_bridge.training.eval import _load_bound_evaluation_data

        (eval_rows, eval_donors, eval_domain) = _load_bound_evaluation_data(
            config, evaluation_arguments, manifest=manifest
        )
        evaluation_bundle = (eval_rows, eval_donors, eval_domain)
    first_epoch = next(iter(epoch_indices))
    ownership = set_bridge_phase_ownership(
        model.executor,
        model.reasoner,
        epoch=first_epoch,
        route1_epochs=int(config.route1_epochs),
    )
    if ownership.active_owner != owner:
        raise RuntimeError("active parameter owner does not match the course phase")
    learning_rate = config.lr_r
    optimizer = build_owner_optimizer(
        model.reasoner,
        owner=owner,
        learning_rate=learning_rate,
        weight_decay=config.weight_decay,
    )
    optimizer_global_batch = int(config.route1_optimizer_global_batch)
    active_max_optimizer_updates = route1_max_optimizer_updates
    course_clock = _route1_optimizer_course_clock(
        rows_for_training,
        epochs=epoch_indices,
        seed=int(arguments.seed),
        world_size=int(world_size),
        local_samples=int(local_samples),
        gradient_accumulation_steps=int(
            config.route1_gradient_accumulation_steps
            if arguments.route1_gradient_accumulation_steps is None
            else arguments.route1_gradient_accumulation_steps
        ),
        boundary_token_count=int(manifest["boundary_token_count"]),
        population=route1_population,
        compute_course=route1_compute_course,
        compute_match=route1_compute_match,
        compute_specific=route1_compute_specific,
        max_optimizer_updates=route1_max_optimizer_updates,
    )
    total_updates = int(course_clock.total_updates)
    progress_update_budget = (
        total_updates if active_max_optimizer_updates is not None else None
    )
    phase_runner = PhaseRunner(
        ControlSchedule(
            save_steps=int(config.save_steps),
            eval_steps=int(config.eval_steps),
            logging_steps=int(config.logging_steps),
        )
    )
    warmup = config.warmup_updates_r
    scheduler = _WarmupCosine(optimizer, warmup=warmup, total=max(total_updates, 1))
    from think_bridge.training.runtime_backend import initialize_training_backend

    backend = initialize_training_backend(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        route=route,
        rank=rank,
        world_size=world_size,
        device=device,
        project_root=Path(os.environ.get("THINK_BRIDGE_PROJECT_ROOT", Path.cwd())),
    )
    from think_bridge.training.checkpoint_manager import CheckpointManager

    checkpoint_manager = (
        CheckpointManager(
            output, route=route, save_total_limit=int(config.save_total_limit)
        )
        if rank == 0
        else None
    )
    start_step = 0
    resume_sampler: dict[str, Any] | None = None
    resume_control_decision: Any | None = None
    resume_evaluation_plan: str | None = None
    resume_validation_report: Path | None = None
    resume_seal: SealedCheckpoint | None = None
    pending_resume_evaluation = broadcast_rank0_result(
        torch_distributed,
        rank=rank,
        local_result=None
        if checkpoint_manager is None
        else checkpoint_manager.pending_evaluation(),
    )
    resume_schedule = phase_runner.schedule
    resume_plan = getattr(arguments, "resume_training_plan", None)
    if resume_plan:
        old_control = resume_plan["previous_training"]
        resume_schedule = ControlSchedule(
            save_steps=int(old_control["save_steps"]),
            eval_steps=int(old_control["eval_steps"]),
            logging_steps=int(old_control["logging_steps"]),
        )
    if arguments.resume is not None:
        (resume_header, resume_seal) = validate_checkpoint_directory_rank0(
            Path(arguments.resume), distributed=torch_distributed, rank=rank
        )
        resume_identity = BridgeCheckpointIdentity.from_mapping(
            resume_header["identity"]
        )
        candidate_step = int(resume_identity.step)
        expected_sampler = _expected_route1_sampler_state(
            rows_for_training,
            course_epochs=route1_course_epochs,
            course_updates=route1_course_steps,
            specificity_donors_per_owner=config.route1_specificity_donors_per_owner,
            distillation_populations=route1_distillation_populations,
            epochs=epoch_indices,
            completed_updates=candidate_step,
            seed=arguments.seed,
            rank=rank,
            world_size=world_size,
            local_samples=local_samples,
            gradient_accumulation_steps=int(
                config.route1_gradient_accumulation_steps
                if arguments.route1_gradient_accumulation_steps is None
                else arguments.route1_gradient_accumulation_steps
            ),
            boundary_token_count=int(manifest["boundary_token_count"]),
            population=route1_population,
            compute_course=route1_compute_course,
            compute_match=route1_compute_match,
            compute_specific=route1_compute_specific,
            eval_null_mode=route1_eval_null_mode,
            max_optimizer_updates=route1_max_optimizer_updates,
        )
        (start_step, resume_sampler) = _load_resume(
            Path(arguments.resume),
            model=model,
            owner=owner,
            backend=backend,
            phase=phase,
            method=config.method,
            seed=arguments.seed,
            manifest=manifest,
            selected_r_sha256=selected_r_sha256,
            cache_identity_sha256=cache_identity_sha256,
            exact_resume_identity_sha256=exact_resume_identity_sha256,
            expected_sampler_state=expected_sampler,
            checkpoint_metadata_payload=resume_header,
            sealed_checkpoint=resume_seal,
            extended=bool(getattr(arguments, "resume_training_plan", None)),
        )
        if getattr(arguments, "resume_training_plan", None):
            if start_step >= total_updates:
                raise ValueError(
                    "extended training budget must exceed the restored step"
                )
            old_rates = [float(g["lr"]) for g in backend.optimizer.param_groups]
            scheduler.rebase_current_step(start_step)
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "event": "resume-training-plan",
                            "step": start_step,
                            "total_updates": total_updates,
                            "previous_learning_rates": old_rates,
                            "learning_rates": [
                                float(g["lr"]) for g in backend.optimizer.param_groups
                            ],
                            "plan": arguments.resume_training_plan,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        if start_step:
            resume_epoch_end = resume_sampler_epoch_end(
                step=start_step,
                sampler_state=resume_sampler,
                route1_epochs=int(config.route1_epochs),
            )
            resume_control_decision = resume_schedule.at(
                step=start_step,
                epoch_end=resume_epoch_end,
                terminal=bool(resume_sampler["phase_terminal"]),
            )
            if (
                resume_control_decision.should_evaluate
                != resume_step_requires_evaluation(
                    resume_schedule,
                    step=start_step,
                    sampler_state=resume_sampler,
                    route1_epochs=int(config.route1_epochs),
                )
            ):
                raise AssertionError("resume control schedule reconstruction drifted")
            pending_matches = bool(
                isinstance(pending_resume_evaluation, Mapping)
                and int(pending_resume_evaluation.get("step", -1)) == start_step
                and (
                    int(pending_resume_evaluation.get("epoch", -1))
                    == int(resume_sampler["epoch"])
                )
                and (
                    Path(
                        str(pending_resume_evaluation.get("checkpoint_path", ""))
                    ).resolve(strict=False)
                    == Path(arguments.resume).resolve(strict=False)
                )
                and (
                    pending_resume_evaluation.get("checkpoint_sha256")
                    == resume_seal.artifact_sha256
                )
            )
            if isinstance(pending_resume_evaluation, Mapping) and (not pending_matches):
                raise ValueError(
                    "pending evaluation route/step/epoch/checkpoint identity mismatch"
                )
            canonical_report = (
                output
                / "reports"
                / route
                / "eval"
                / f"bridge-{route}-validation-step-{start_step}.json"
            )
            supplied_report = (
                None
                if arguments.validation_report is None
                else Path(arguments.validation_report)
            )
            if supplied_report is not None and supplied_report != canonical_report:
                raise ValueError(
                    "resume validation report is not the canonical run/route/step path"
                )
            resume_validation_report = supplied_report or (
                canonical_report if canonical_report.is_file() else None
            )
            resume_evaluation_plan = resume_evaluation_action(
                resume_schedule,
                step=start_step,
                sampler_state=resume_sampler,
                pending_evaluation=pending_matches,
                completed_report=resume_validation_report is not None,
                route1_epochs=int(config.route1_epochs),
            )
    owner_parameters = tuple(
        (
            parameter
            for parameter in _owner_module(model, owner).parameters()
            if parameter.requires_grad
        )
    )
    integrity_scan_total_seconds = 0.0
    startup_integrity_started = time.perf_counter()
    _synchronized_update_state_finite_or_raise(
        owner_parameters,
        backend.integrity_optimizer,
        device=device,
        label="startup/resume integrity boundary",
    )
    integrity_scan_total_seconds += time.perf_counter() - startup_integrity_started
    train_model: nn.Module = backend.train_model
    train_model.train()
    step = start_step
    last_sampler_state = resume_sampler or _initial_route1_sampler_state(
        specificity_donors_per_owner=config.route1_specificity_donors_per_owner,
        epoch=first_epoch,
        seed=arguments.seed,
        eval_null_mode=route1_eval_null_mode,
        population=route1_population,
        compute_course=route1_compute_course,
        compute_match=route1_compute_match,
        compute_specific=route1_compute_specific,
        distillation_populations=tuple(route1_distillation_populations),
    )
    update_limit = total_updates
    test_interrupt = getattr(arguments, "test_interrupt_after_updates", None)
    if test_interrupt is not None:
        if int(test_interrupt) <= 0:
            raise ValueError("test interruption update count must be positive")
        update_limit = min(total_updates, start_step + int(test_interrupt))
    completed_epoch_boundary = False
    record_total = 0
    component_audit_report: dict[str, Any] | None = None
    component_audit_step: int | None = None
    component_audit_pending = bool(
        getattr(arguments, "route1_component_gradient_audit", False)
    )
    initial_progress_stage = bridge_training_stage(
        route=route,
        epoch=int(last_sampler_state["epoch"]),
        route1_epochs=int(config.route1_epochs),
        update_budget=progress_update_budget,
    )
    progress_bar = bridge_progress(
        total=total_updates,
        initial=start_step,
        desc=initial_progress_stage.description,
        unit="update",
        disabled=bool(arguments.no_progress or rank != 0),
    )
    resource_stack.callback(progress_bar.close)
    stage_reporter = (
        BridgeStageReporter(progress_bar, label="Bridge Stage1 training")
        if rank == 0
        else None
    )
    logging_window = (
        BridgeLoggingWindow(metric_names=TRAINING_WINDOW_METRICS) if rank == 0 else None
    )
    lifecycle_logger = (
        LifecycleLogger(Path(arguments.logging_jsonl)) if rank == 0 else None
    )
    lifecycle_started = time.perf_counter()

    def rank0_control(action: Any) -> Mapping[str, Any]:
        return run_rank0_control(torch_distributed, rank=rank, action=action)

    def write_control_checkpoint(
        *, checkpoint_directory: Path, state: Mapping[str, Any], current_step: int
    ) -> tuple[BridgeCheckpointIdentity, SealedCheckpoint]:
        (identity, seal) = _save_checkpoint(
            checkpoint_directory,
            model=model,
            owner=owner,
            backend=backend,
            sampler_state=state,
            phase=phase,
            method=config.method,
            geometry_schema_version=config.geometry_schema_version,
            attn_implementation=config.attn_implementation,
            seed=arguments.seed,
            selected_r_sha256=selected_r_sha256,
            manifest=manifest,
            cache_identity_sha256=cache_identity_sha256,
            exact_resume_identity_sha256=exact_resume_identity_sha256,
            step=current_step,
            recover_incomplete_transactions=arguments.resume is not None,
        )
        return (identity, seal)

    def validation_report_paths() -> list[Path]:
        return list(
            canonical_validation_report_paths(
                output / "reports" / route / "eval", route=route
            )
        )

    def validate_completed_report_ledger(
        *, exact_report_paths: Sequence[Path] | None = None
    ) -> tuple[dict[str, Any], ...]:
        if checkpoint_manager is None:
            raise AssertionError("rank-0 checkpoint manager is missing")
        rows = checkpoint_manager.validate_completed_evaluations(
            report_paths=None
            if exact_report_paths is None
            else tuple((Path(path) for path in exact_report_paths))
        )
        for row in rows:
            _validate_validation_barrier(
                Path(str(row["report_path"])),
                checkpoint_path=Path(str(row["checkpoint_path"])),
                route=route,
                seed=arguments.seed,
                epoch=int(row["epoch"]),
                generation_seed=int(arguments.generation_seed),
                metric_policy=arguments.metric_policy,
                checkpoint_sha256=str(row["checkpoint_sha256"]),
                completed_step=int(row["step"]),
            )
            report = _read_json(Path(str(row["report_path"])))
            if report.get("validation_randomness") != row["validation_randomness"]:
                raise ValueError(
                    "completed evaluation randomness differs from its ledger"
                )
        return rows

    def validate_completed_resume() -> str:
        if arguments.resume is None or resume_validation_report is None:
            raise AssertionError("completed evaluation resume lacks its artifacts")
        rows = validate_completed_report_ledger(
            exact_report_paths=validation_report_paths()
        )
        matches = [row for row in rows if int(row["step"]) == int(start_step)]
        if (
            len(matches) != 1
            or rows[-1:] != tuple(matches)
            or Path(str(matches[0]["report_path"])).resolve(strict=True)
            != Path(resume_validation_report).resolve(strict=True)
            or (
                Path(str(matches[0]["checkpoint_path"])).resolve(strict=True)
                != Path(arguments.resume).resolve(strict=True)
            )
            or (matches[0]["checkpoint_sha256"] != resume_seal.artifact_sha256)
        ):
            raise ValueError(
                "completed evaluation resume differs from its sealed ledger frontier"
            )
        return "completed-evaluation-ledger-validated"

    def selected_step(paths: Sequence[Path]) -> int:
        reports = [_read_json(path) for path in paths]
        from think_bridge.training.metric_registry import (
            metric_policy_from_mapping,
            select_best_candidate,
        )

        policy = metric_policy_from_mapping(arguments.metric_policy)
        evaluator = {"route1": "stage1.route1"}[route]
        return int(
            select_best_candidate(reports, evaluator=evaluator, policy=policy).step
        )

    def pointer_reference(name: str) -> tuple[Path, SealedCheckpoint] | None:
        path = output / name
        if not path.is_file():
            return None
        target = read_checkpoint_pointer_target(path, run_dir=output)
        return (target, validate_checkpoint_directory(target))

    def prune_committed_checkpoints() -> None:
        if checkpoint_manager is None:
            raise AssertionError("rank-0 checkpoint manager is missing")
        checkpoint_manager.prune_committed()

    def append_checkpoint_lifecycle(
        *,
        current_step: int,
        current_epoch: int,
        reason: str,
        checkpoint: Path,
        seal: SealedCheckpoint,
    ) -> None:
        if lifecycle_logger is None:
            raise AssertionError("rank-0 lifecycle logger is missing")
        latest = pointer_reference(f"{route}_latest_checkpoint.txt")
        best = pointer_reference(f"{route}_best_checkpoint.txt")
        if latest is None:
            raise RuntimeError("committed checkpoint lacks the latest pointer")
        lifecycle_logger.append_checkpoint(
            route=route,
            owner=owner,
            phase=phase_for_epoch(
                int(current_epoch), route1_epochs=int(config.route1_epochs)
            ).name,
            step=int(current_step),
            reason=reason,
            checkpoint_path=checkpoint,
            checkpoint_sha256=seal.artifact_sha256,
            latest_checkpoint_path=latest[0],
            latest_checkpoint_sha256=latest[1].artifact_sha256,
            best_checkpoint_path=None if best is None else best[0],
            best_checkpoint_sha256=None if best is None else best[1].artifact_sha256,
        )

    def finish_save(
        current_step: int,
        current_epoch: int,
        sealed_checkpoint: SealedCheckpoint | None = None,
    ) -> str:
        if checkpoint_manager is None:
            raise AssertionError("rank-0 checkpoint manager is missing")
        published = checkpoint_manager.publish_after_update(
            step=current_step,
            finite=True,
            materialize=lambda _path: None,
            sealed_checkpoint=sealed_checkpoint,
        )
        seal = validate_checkpoint_directory(published)
        append_checkpoint_lifecycle(
            current_step=current_step,
            current_epoch=current_epoch,
            reason="scheduled_save",
            checkpoint=published,
            seal=seal,
        )
        return str(published)

    def finish_evaluation(
        current_step: int,
        report_path: Path,
        sealed_checkpoint: SealedCheckpoint | None = None,
    ) -> None:
        if checkpoint_manager is None:
            raise AssertionError("rank-0 checkpoint manager is missing")
        pending = checkpoint_manager.pending_evaluation()
        if pending is None or int(pending.get("step", -1)) != int(current_step):
            raise RuntimeError("evaluation finish lacks its pending transaction")
        _validate_validation_barrier(
            report_path,
            checkpoint_path=Path(str(pending["checkpoint_path"])),
            route=route,
            seed=arguments.seed,
            epoch=int(pending["epoch"]),
            generation_seed=int(arguments.generation_seed),
            metric_policy=arguments.metric_policy,
            checkpoint_sha256=str(pending["checkpoint_sha256"]),
        )
        completed = validate_completed_report_ledger()
        paths = validation_report_paths()
        expected_paths = {
            Path(str(row["report_path"])).resolve(strict=True) for row in completed
        }
        expected_paths.add(Path(report_path).resolve(strict=True))
        observed_paths = [Path(path).resolve(strict=True) for path in paths]
        if (
            len(observed_paths) != len(set(observed_paths))
            or set(observed_paths) != expected_paths
        ):
            raise ValueError(
                "best selection report set differs from completed+pending ledger"
            )
        best_step = None if diagnostic_only else selected_step(paths)
        checkpoint_manager.commit_evaluation(
            step=current_step,
            report_path=report_path,
            is_best=best_step == current_step if best_step is not None else False,
            sealed_checkpoint=sealed_checkpoint,
        )
        committed_seal = validate_checkpoint_directory(
            Path(str(pending["checkpoint_path"]))
        )
        append_checkpoint_lifecycle(
            current_step=current_step,
            current_epoch=int(pending["epoch"]),
            reason="evaluation_commit",
            checkpoint=Path(str(pending["checkpoint_path"])),
            seal=committed_seal,
        )
        committed = checkpoint_manager.validate_completed_evaluations()
        matching = [row for row in committed if int(row["step"]) == int(current_step)]
        if len(matching) != 1 or lifecycle_logger is None:
            raise RuntimeError("committed evaluation lacks one lifecycle receipt")
        lifecycle_logger.append_evaluation(
            receipt=matching[0], manifest_path=Path(arguments.manifest)
        )

    def evaluate_control_checkpoint(
        *,
        checkpoint: Path,
        identity: BridgeCheckpointIdentity,
        state: Mapping[str, Any],
        current_step: int,
        sealed_checkpoint: SealedCheckpoint,
    ) -> Path:
        if evaluation_arguments is None or evaluation_bundle is None:
            raise RuntimeError(
                "formal long-lived phase lacks its in-process validation inputs"
            )
        (eval_rows, eval_donors, eval_domain) = evaluation_bundle
        report_path = (
            output
            / "reports"
            / route
            / "eval"
            / f"bridge-{route}-validation-step-{current_step}.json"
        )
        configure_control_evaluation_arguments(
            evaluation_arguments,
            checkpoint=checkpoint,
            report_path=report_path,
            generation_seed=int(arguments.generation_seed),
        )
        evaluation_arguments.eval_diagnostics = True
        from think_bridge.training.eval import evaluate_loaded_model

        with suspend_bridge_progress(progress_bar):
            try:
                evaluate_loaded_model(
                    config,
                    evaluation_arguments,
                    model=model,
                    tokenizer=tokenizer,
                    checkpoint_identity=identity,
                    checkpoint_payload={
                        "sampler_state": _portable_sampler_state(state)
                    },
                    manifest=manifest,
                    rows=eval_rows,
                    donors=eval_donors,
                    evaluation_domain=eval_domain,
                    rank=rank,
                    world_size=world_size,
                    checkpoint_seal=sealed_checkpoint,
                    capture_live_feedback=True,
                )
                rank0_control(
                    lambda: finish_evaluation(
                        current_step, report_path, sealed_checkpoint
                    )
                )
            finally:
                train_model.train()
        return report_path

    def reconcile_completed_lifecycle_events() -> int:
        if lifecycle_logger is None:
            raise AssertionError("rank-0 lifecycle logger is missing")
        completed = validate_completed_report_ledger()
        return reconcile_completed_evaluation_events(
            lifecycle_logger, receipts=completed, manifest_path=Path(arguments.manifest)
        )

    rank0_control(reconcile_completed_lifecycle_events)
    if resume_evaluation_plan == "validate_completed_evaluation":
        rank0_control(
            lambda: (
                validate_completed_resume(),
                prune_committed_checkpoints(),
                "completed-evaluation-pruned",
            )[2]
        )
    if isinstance(pending_resume_evaluation, Mapping):
        if resume_evaluation_plan != "finish_pending_evaluation":
            raise AssertionError(
                "pending resume did not classify as pending evaluation"
            )
        if arguments.resume is None or resume_sampler is None:
            raise ValueError(
                "pending evaluation exists without an explicit exact resume"
            )
        pending_step = int(pending_resume_evaluation["step"])
        report_path = (
            output
            / "reports"
            / route
            / "eval"
            / f"bridge-{route}-validation-step-{pending_step}.json"
        )
        if report_path.is_file():
            rank0_control(
                lambda: finish_evaluation(pending_step, report_path, resume_seal)
            )
        else:
            evaluate_control_checkpoint(
                checkpoint=Path(arguments.resume),
                identity=resume_identity,
                state=resume_sampler,
                current_step=pending_step,
                sealed_checkpoint=resume_seal,
            )
    elif resume_evaluation_plan == "begin_pending_evaluation":
        if arguments.resume is None or resume_sampler is None:
            raise AssertionError(
                "resume evaluation repair lacks its checkpoint frontier"
            )
        if resume_control_decision is None:
            raise AssertionError("resume evaluation repair lacks its control decision")
        rank0_control(
            lambda: (
                str(
                    checkpoint_manager.repair_evaluation_frontier(
                        step=start_step,
                        epoch=int(resume_sampler["epoch"]),
                        checkpoint=Path(arguments.resume),
                        ordinary_checkpoint=bool(resume_control_decision.should_save),
                        sealed_checkpoint=resume_seal,
                    )
                )
                if checkpoint_manager is not None
                else None
            )
        )
        evaluate_control_checkpoint(
            checkpoint=Path(arguments.resume),
            identity=resume_identity,
            state=resume_sampler,
            current_step=start_step,
            sealed_checkpoint=resume_seal,
        )
    if resume_evaluation_plan == "resume_update":
        if arguments.resume is None or resume_sampler is None:
            raise AssertionError("save-only resume lacks its checkpoint frontier")
        epoch_end = resume_sampler_epoch_end(
            step=start_step,
            sampler_state=resume_sampler,
            route1_epochs=int(config.route1_epochs),
        )
        if not resume_schedule.at(
            step=start_step,
            epoch_end=epoch_end,
            terminal=bool(resume_sampler["phase_terminal"]),
        ).should_save:
            raise ValueError(
                "resume checkpoint was neither a save nor eval control event"
            )
        rank0_control(
            lambda: finish_save(start_step, int(resume_sampler["epoch"]), resume_seal)
        )
    if start_step:
        may_continue = rank0_control(
            lambda: (
                checkpoint_manager.may_begin_update(start_step + 1)
                if checkpoint_manager is not None
                else False
            )
        )["value"]
        if may_continue is not True:
            raise RuntimeError(
                "checkpoint manager frontier is not committed for the next update"
            )
        if arguments.resume is None or resume_seal is None:
            raise AssertionError("accepted resume lacks its sealed checkpoint")
        rank0_control(
            lambda: (
                lifecycle_logger.rollback_training_after(
                    route=route,
                    frontier_step=start_step,
                    step_audit_path=output / "step_audit.jsonl",
                )
                if lifecycle_logger is not None
                else None
            )
        )
        rank0_control(
            lambda: (
                append_checkpoint_resume_event(
                    output,
                    Path(arguments.resume),
                    checkpoint_sha256=resume_seal.artifact_sha256,
                    status="accepted",
                ),
                "resume-accepted-after-semantic-validation",
            )[1]
        )
    for epoch in epoch_indices:
        steps_per_epoch = int(course_clock.steps_for_epoch(int(epoch)))
        if rank == 0:
            set_bridge_training_stage(
                progress_bar,
                route=route,
                epoch=int(epoch),
                route1_epochs=int(config.route1_epochs),
                update_budget=progress_update_budget,
            )
        epoch_offset = int(course_clock.offset_for_epoch(int(epoch)))
        epoch_cursor = min(steps_per_epoch, max(0, start_step - epoch_offset))
        set_bridge_phase_ownership(
            model.executor,
            model.reasoner,
            epoch=epoch,
            route1_epochs=int(config.route1_epochs),
        )
        route1_rows_by_record_id = {
            str(row["record_id"]): row
            for row in resolve_route1_normalized_rows(
                rows_for_training, population=route1_population
            ).rows
        }
        route1_batch_function = _epoch_batches
        route1_batch_options: dict[str, Any] = {}
        route1_source_batches = route1_batch_function(
            rows_for_training,
            course_epochs=route1_course_epochs,
            course_updates=route1_course_steps,
            course_steps_per_epoch=course_clock.steps_for_epoch(first_epoch),
            specificity_donors_per_owner=config.route1_specificity_donors_per_owner,
            distillation_populations=route1_distillation_populations,
            epoch=epoch,
            seed=arguments.seed,
            rank=rank,
            world_size=world_size,
            local_samples=local_samples,
            gradient_accumulation_steps=int(
                config.route1_gradient_accumulation_steps
                if arguments.route1_gradient_accumulation_steps is None
                else arguments.route1_gradient_accumulation_steps
            ),
            boundary_token_count=int(manifest["boundary_token_count"]),
            population=route1_population,
            compute_course=route1_compute_course,
            compute_match=route1_compute_match,
            compute_specific=route1_compute_specific,
            eval_null_mode=route1_eval_null_mode,
            max_optimizer_updates=route1_max_optimizer_updates,
            terminal_epoch=int(epoch_indices[-1]),
            cursor=epoch_cursor,
            optimizer_update_offset=epoch_offset,
            terminal_optimizer_update=int(course_clock.terminal_optimizer_update),
            **route1_batch_options,
        )

        def prepare_static_route1_batch(
            source_batch: tuple[Sequence[Mapping[str, Any]], Mapping[str, Any]],
        ) -> tuple[tuple[dict[str, Any], ...], Mapping[str, Any]]:
            (batch_rows, sampler_state) = source_batch
            microbatch = int(sampler_state["route1_local_samples"])
            accumulation = int(sampler_state["route1_gradient_accumulation_steps"])
            if len(batch_rows) != microbatch * accumulation:
                raise RuntimeError(
                    "Route1 optimizer batch cannot split into GAS microsteps"
                )

            def prepare_microstep(micro_step: int) -> dict[str, Any]:
                rows_for_microstep = batch_rows[
                    micro_step * microbatch : (micro_step + 1) * microbatch
                ]
                common_arguments = {
                    "epoch": int(sampler_state["epoch"]),
                    "optimizer_step": int(sampler_state["optimizer_step"]),
                    "optimizer_steps_per_epoch": course_clock.steps_for_epoch(
                        first_epoch
                    ),
                    "course_epochs": route1_course_epochs,
                    "course_updates": route1_course_steps,
                    "global_course_c_sample_count": int(
                        sampler_state["global_course_c_sample_count"]
                    ),
                    "global_course_direct_side_sample_count": int(
                        sampler_state["global_course_direct_side_sample_count"]
                    ),
                    "global_match_sample_count": int(
                        sampler_state["global_match_sample_count"]
                    ),
                    "global_specific_sample_count": int(
                        sampler_state["global_specific_sample_count"]
                    ),
                    "global_native_b_sample_count": int(
                        sampler_state["global_native_b_sample_count"]
                    ),
                    "global_native_c_sample_count": int(
                        sampler_state["global_native_c_sample_count"]
                    ),
                    "distillation_populations": tuple(
                        sampler_state.get(
                            "distillation_populations",
                            tuple(route1_distillation_populations),
                        )
                    ),
                    "wrong_candidate_mask": sampler_state["local_wrong_candidate_mask"][
                        micro_step * microbatch : (micro_step + 1) * microbatch
                    ],
                    "specificity_donor_rows": tuple(
                        (
                            route1_rows_by_record_id[record_id]
                            for record_id in sampler_state.get(
                                "specificity_donor_record_ids", ()
                            )
                        )
                    ),
                    "physical_chunk_size": config.route1_local_chunk_size,
                    "wrong_control_chunk_size": config.route1_wrong_control_chunk_size,
                    "pin_memory": device.type == "cuda",
                    "course_weight": route1_course_weight,
                    "match_weight": route1_match_weight,
                    "specific_weight": route1_specific_weight,
                    "specificity_tau": route1_specificity_tau,
                    "seed": int(arguments.seed),
                }
                return _prepare_route1_batch_cpu(rows_for_microstep, **common_arguments)

            return (
                tuple(
                    (
                        prepare_microstep(micro_step)
                        for micro_step in range(accumulation)
                    )
                ),
                sampler_state,
            )

        first_prefetch_wait_active = True

        def report_first_prefetch_wait(
            phase: str, _phase_elapsed: float, wait_elapsed: float
        ) -> None:
            if not first_prefetch_wait_active or stage_reporter is None:
                return
            stage_reporter.report(
                {
                    "source": "epoch-plan",
                    "prepare": "cpu-prepare/pin",
                    "ready": "first-prefetch-ready",
                }.get(phase, f"first-prefetch/{phase}"),
                elapsed_seconds=wait_elapsed,
                important=True,
            )

        route1_prefetch = DepthOneBatchPrefetch(
            route1_source_batches,
            prepare=prepare_static_route1_batch,
            close_timeout_seconds=30.0,
            wait_reporter=report_first_prefetch_wait
            if stage_reporter is not None
            else None,
        )
        resource_stack.callback(route1_prefetch.close)
        batches: Iterable[Any] = route1_prefetch
        epoch_had_update = False
        first_batch_in_epoch = True
        for batch_item in batches:
            startup_timings: dict[str, float] | None = None
            (batch_arguments_cpu, sampler_state) = batch_item.value
            batch_prepare_wait_seconds = float(batch_item.wait_seconds)
            if first_batch_in_epoch:
                startup_timings = {
                    "epoch_plan_seconds": float(batch_item.source_seconds),
                    "first_prefetch_wait_seconds": float(batch_item.wait_seconds),
                    "first_batch_prepare_seconds": float(batch_item.prepare_seconds),
                }
                if stage_reporter is not None:
                    stage_reporter.report(
                        "first-prefetch-ready",
                        elapsed_seconds=float(batch_item.wait_seconds),
                        important=True,
                    )
                first_prefetch_wait_active = False
                first_batch_in_epoch = False
            batch_rows = ()
            if step >= update_limit:
                break
            update_started = time.perf_counter()
            _assert_runtime_sample_microbatches(
                sampler_state["global_microstep_prompt_group_ids"],
                global_record_ids=sampler_state["global_occurrence_record_ids"],
                expected_microstep_global_batch=int(
                    config.route1_microstep_global_batch
                ),
            )
            if int(sampler_state["optimizer_update"]) != step + 1:
                raise RuntimeError(
                    "Route1 sampler/update schedule frontier is inconsistent"
                )
            epoch_end = int(sampler_state["global_batch_index"]) == steps_per_epoch - 1
            planned_decision = phase_runner.schedule.at(
                step=step + 1,
                epoch_end=epoch_end,
                terminal=bool(sampler_state["phase_terminal"]),
            )
            collect_route1_diagnostics = bool(
                planned_decision.should_log
                or planned_decision.should_save
                or planned_decision.should_evaluate
                or diagnostic_only
                or component_audit_pending
            )
            if batch_arguments_cpu is None:
                raise AssertionError("Route1 static batch preparation is missing")
            batch_arguments = tuple(
                (
                    _route1_batch_to_device(prepared, device)
                    for prepared in batch_arguments_cpu
                )
            )

            def forward_once(
                arguments_for_forward: Mapping[str, Any] | None = None,
            ) -> Any:
                with _training_compute_context(device):
                    return train_model(
                        route="route1_bridge",
                        **batch_arguments
                        if arguments_for_forward is None
                        else arguments_for_forward,
                    )

            component_audit: dict[str, Any] = {}
            forward_timings: dict[str, Any] = {"backward_seconds": 0.0}
            integrity_scan_seconds = 0.0
            backend.zero_grad()

            def finalize_owner_update(
                _results: Sequence[Any],
            ) -> tuple[float, float, float]:
                if free_generation_provider is not None:
                    free_generation_provider.finish_update(step + 1)
                maximum_norm = config.max_grad_norm_r
                (preclip, postclip) = _clip_owner_gradients_for_update(
                    backend,
                    owner_parameters,
                    max_norm=float(maximum_norm),
                    device=device,
                    preclip_observer=None,
                )
                optimizer_started = time.perf_counter()
                optimizer_events = None
                if device.type == "cuda" and collect_route1_diagnostics:
                    optimizer_events = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    optimizer_events[0].record()
                backend.step()
                if optimizer_events is not None:
                    optimizer_events[1].record()
                    optimizer_events[1].synchronize()
                    optimizer_seconds = (
                        float(optimizer_events[0].elapsed_time(optimizer_events[1]))
                        / 1000.0
                    )
                else:
                    optimizer_seconds = time.perf_counter() - optimizer_started
                return (float(preclip), float(postclip), float(optimizer_seconds))

            c_counts = tuple(
                (
                    int(value)
                    for value in sampler_state["global_microstep_native_sample_counts"]
                )
            )
            audit_microstep = next(
                (index for (index, count) in enumerate(c_counts) if count > 0), 0
            )
            audited_result: Any | None = None

            def route1_prepare(
                prepared_arguments: Mapping[str, Any], *, microstep: int
            ) -> dict[str, Any]:
                prepared = dict(prepared_arguments)
                timings: dict[str, float] = {}
                keys = tuple(
                    (
                        str(value)
                        for value in sampler_state["local_microstep_sample_record_ids"][
                            microstep
                        ]
                    )
                )
                prepared["prepared_state"] = _prepare_route1_microstep_for_training(
                    model,
                    prepared,
                    provider=free_generation_provider,
                    update=step + 1,
                    microstep=microstep,
                    sample_keys=keys,
                    timing_sink=timings if collect_route1_diagnostics else None,
                    progress_sink=(
                        lambda stage: stage_reporter.report(
                            f"gas{microstep + 1}:{stage}",
                            elapsed_seconds=max(
                                time.perf_counter() - update_started, 0.0
                            ),
                        )
                    )
                    if stage_reporter is not None
                    else None,
                )
                for name, value in timings.items():
                    forward_timings[name] = forward_timings.get(name, 0.0) + float(
                        value
                    )
                return prepared

            def route1_discard(prepared: dict[str, Any]) -> None:
                state = prepared.pop("prepared_state", None)
                if state is not None:
                    try:
                        if not state["resolved"] and state["ticket"] is not None:
                            free_generation_provider.discard(state["ticket"])
                    finally:
                        state.clear()

            def route1_forward(
                prepared_arguments: Mapping[str, Any], *, microstep: int
            ) -> Any:
                nonlocal component_audit_pending
                nonlocal component_audit_report, component_audit_step
                nonlocal audited_result
                arguments_for_forward = dict(prepared_arguments)
                perform_component_audit = bool(
                    component_audit_pending
                    and microstep == audit_microstep
                    and (
                        not route1_compute_specific
                        or (
                            float(arguments_for_forward["specific_weight"]) > 0.0
                            and int(
                                arguments_for_forward["global_specific_sample_count"]
                            )
                            > 0
                        )
                    )
                )
                microstep_timings: dict[str, Any] = {}
                arguments_for_forward.update(
                    component_gradient_audit=perform_component_audit,
                    specificity_step_audit=bool(planned_decision.should_log),
                    free_generation_provider=free_generation_provider,
                    rollout_update=step + 1,
                    rollout_micro_step=int(microstep),
                    rollout_sample_keys=tuple(
                        (
                            str(value)
                            for value in sampler_state[
                                "local_microstep_sample_record_ids"
                            ][microstep]
                        )
                    ),
                    timing_sink=microstep_timings
                    if collect_route1_diagnostics
                    else None,
                    progress_sink=(
                        lambda stage: stage_reporter.report(
                            f"gas{microstep + 1}:{stage}",
                            elapsed_seconds=max(
                                time.perf_counter() - update_started, 0.0
                            ),
                        )
                    )
                    if stage_reporter is not None
                    else None,
                )
                microstep_result = forward_once(arguments_for_forward)
                merge_timing_sink(forward_timings, microstep_timings)
                _synchronized_tensors_finite_or_raise(
                    (microstep_result.loss,),
                    device=device,
                    label=f"Route1 GAS microstep {microstep + 1} total loss",
                )
                if perform_component_audit:
                    component_audit.update(
                        _globalize_route1_component_gradient_audit(
                            _route1_component_gradient_audit(
                                result=microstep_result, model=model, step=step + 1
                            ),
                            device=device,
                        )
                    )
                    audited_result = microstep_result
                    component_audit_pending = False
                    component_audit_report = dict(component_audit)
                    component_audit_step = step + 1
                return microstep_result

            def route1_backward(
                microstep_result: Any, *, synchronize_gradients: bool
            ) -> None:
                nonlocal audited_result
                backward_started = time.perf_counter()
                backward_events = None
                if collect_route1_diagnostics and device.type == "cuda":
                    backward_events = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    backward_events[0].record()
                backend.backward(
                    microstep_result.loss, synchronize_gradients=synchronize_gradients
                )
                if microstep_result is audited_result:
                    frozen_parameters = list(model.executor.parameters())
                    frozen_gradient_free = not any(
                        (parameter.grad is not None for parameter in frozen_parameters)
                    )
                    if not _all_ranks_true(frozen_gradient_free, device=device):
                        component_audit["frozen_F_embedding_D"] = False
                        component_audit["component_gradient_audit_status"] = (
                            "measured_with_findings"
                        )
                        prior = str(
                            component_audit.get("component_gradient_audit_findings", "")
                        )
                        component_audit["component_gradient_audit_findings"] = ",".join(
                            (
                                value
                                for value in (
                                    prior,
                                    "formal_backward_touched_frozen_owner",
                                )
                                if value
                            )
                        )
                    audited_result = None
                if backward_events is not None:
                    backward_events[1].record()
                    backward_events[1].synchronize()
                    elapsed = (
                        float(backward_events[0].elapsed_time(backward_events[1]))
                        / 1000.0
                    )
                elif collect_route1_diagnostics:
                    elapsed = time.perf_counter() - backward_started
                else:
                    elapsed = 0.0
                if collect_route1_diagnostics:
                    forward_timings["backward_seconds"] += elapsed

            (microstep_results, finalize_values) = run_route1_accumulation_update(
                batch_arguments,
                synchronization_context=backend.synchronization_context,
                forward=route1_forward,
                backward=route1_backward,
                finalize=finalize_owner_update,
                prepare=route1_prepare
                if free_generation_provider is not None
                else None,
                discard=route1_discard,
                max_prepared_microsteps=int(
                    arguments.route1_vllm_max_pending_microsteps
                ),
            )
            finalize_timing_sink(forward_timings)
            result = _aggregate_route1_microstep_results(microstep_results)
            (preclip_norm, postclip_norm, optimizer_step_seconds) = finalize_values
            if (
                diagnostic_only
                or bool(component_audit)
                or planned_decision.should_save
                or planned_decision.should_evaluate
                or bool(getattr(arguments, "full_state_audit", False))
            ):
                integrity_started = time.perf_counter()
                _synchronized_update_state_finite_or_raise(
                    owner_parameters,
                    backend.integrity_optimizer,
                    device=device,
                    label="checkpoint/evaluation integrity boundary",
                )
                integrity_scan_seconds = time.perf_counter() - integrity_started
                integrity_scan_total_seconds += integrity_scan_seconds
            if component_audit:
                component_audit["total_reasoner_R_parameter_grad_norm_preclip"] = float(
                    preclip_norm
                )
                component_audit_report = dict(component_audit)
            rank_max_route1_metrics: dict[str, float] = {}
            route1_global_metrics: dict[str, float | int] | None = None
            route1_specificity_detail: dict[str, float | int | None] | None = None
            collect_route1_rank_metrics = False
            if collect_route1_diagnostics:
                collect_route1_rank_metrics = True
            if collect_route1_rank_metrics:
                forward_timings["optimizer_seconds"] = float(optimizer_step_seconds)
                timing_names = (
                    "live_z_seconds",
                    "answer_course_seconds",
                    "rollout_submit_seconds",
                    "rollout_queue_seconds",
                    "rollout_service_seconds",
                    "rollout_generation_seconds",
                    "rollout_overlap_seconds",
                    "rollout_wait_seconds",
                    "need_objectives_seconds",
                    "need_objectives_rollout_seconds",
                    "specificity_teacher_seconds",
                    "specificity_student_seconds",
                    "specificity_no_z_seconds",
                    "specificity_donor_student_seconds",
                    "specificity_head_seconds",
                    "specificity_donor_backward_seconds",
                    "specificity_owner_replay_seconds",
                    "backward_seconds",
                    "optimizer_seconds",
                )
                count_fields = (
                    "c_view_count",
                    "legal_wrong_pair_count",
                    "wrong_physical_chunk_count",
                    "rollout_request_count",
                    "rollout_token_count",
                    "rollout_service_batch_rows",
                    "rollout_service_real_rows",
                )
                rollout_service_seconds = float(
                    forward_timings.get("rollout_service_seconds", 0.0)
                )
                rollout_overlap_seconds = float(
                    forward_timings.get("rollout_overlap_seconds", 0.0)
                )
                rollout_batch_rows = int(result.rollout_service_batch_rows)
                rollout_real_rows = int(result.rollout_service_real_rows)
                derived_values = (
                    min(
                        1.0, max(0.0, rollout_overlap_seconds / rollout_service_seconds)
                    )
                    if rollout_service_seconds > 0.0
                    else 0.0,
                    float(rollout_real_rows) / float(rollout_batch_rows)
                    if rollout_batch_rows > 0
                    else 0.0,
                )
                rank_values = torch.tensor(
                    [float(forward_timings.get(name, 0.0)) for name in timing_names]
                    + list(derived_values)
                    + [
                        float(torch.cuda.max_memory_allocated(device))
                        if device.type == "cuda"
                        else 0.0,
                        float(torch.cuda.max_memory_reserved(device))
                        if device.type == "cuda"
                        else 0.0,
                    ]
                    + [float(getattr(result, name)) for name in count_fields],
                    dtype=torch.float64,
                    device=device,
                )
                if torch_distributed.is_initialized():
                    torch_distributed.all_reduce(
                        rank_values, op=torch_distributed.ReduceOp.MAX
                    )
                metric_names = tuple((f"rank_max_{name}" for name in timing_names)) + (
                    "rank_max_trainer_overlap_efficiency",
                    "rank_max_service_batch_occupancy",
                    "rank_max_peak_memory_bytes",
                    "rank_max_reserved_memory_bytes",
                    "rank_max_c_view_count",
                    "rank_max_legal_wrong_pair_count",
                    "rank_max_wrong_physical_chunk_count",
                    "rank_max_rollout_request_count",
                    "rank_max_rollout_token_count",
                    "rank_max_rollout_service_batch_rows",
                    "rank_max_rollout_service_real_rows",
                )
                rank_max_route1_metrics = {
                    name: float(value)
                    for (name, value) in zip(metric_names, rank_values.tolist())
                }
            effective_specific_weight = float(batch_arguments[0]["specific_weight"])
            if any(
                (
                    float(values["specific_weight"]) != effective_specific_weight
                    for values in batch_arguments
                )
            ):
                raise RuntimeError("Specificity schedule differs inside a GAS update")
            route1_global_metrics = _route1_global_step_metrics(
                result=result,
                device=device,
                loss_weights=(
                    route1_course_weight,
                    route1_match_weight,
                    effective_specific_weight,
                ),
            )
            if planned_decision.should_log:
                route1_specificity_detail = _globalize_route1_specificity_step_stats(
                    result=result, device=device
                )
                if (
                    route1_specificity_detail["specificity_pair_count"]
                    != sampler_state["specificity_selected_pair_count"]
                ):
                    raise RuntimeError(
                        "Specificity executed pair count differs from the sampled global plan"
                    )
            step += 1
            epoch_had_update = True
            last_sampler_state = sampler_state
            update_seconds = max(time.perf_counter() - update_started, 1e-12)
            current_metrics = None
            if rank == 0:
                current_metrics = _training_step_metrics(
                    result=result,
                    route=route,
                    sampler_state=sampler_state,
                    route1_global_metrics=route1_global_metrics,
                )
                if startup_timings is not None:
                    current_metrics.update(startup_timings)
                current_metrics["optimizer_seconds"] = (
                    optimizer_step_seconds if collect_route1_diagnostics else None
                )
                current_metrics["integrity_scan_seconds"] = float(
                    integrity_scan_seconds
                )
                current_metrics.update(rank_max_route1_metrics)
                progress_bar.set_postfix(
                    bridge_training_postfix(
                        route=route,
                        arm=config.method,
                        metrics=current_metrics,
                        learning_rate=float(backend.optimizer.param_groups[0]["lr"]),
                    ),
                    refresh=False,
                )
            progress_bar.update(1)
            decision = phase_runner.after_finite_update(
                step=step,
                epoch=int(sampler_state["epoch"]),
                epoch_end=epoch_end,
                terminal=bool(sampler_state["phase_terminal"]),
            )
            if decision != planned_decision:
                raise AssertionError(
                    "post-update control decision changed within one batch"
                )
            if rank == 0:
                if logging_window is None:
                    raise AssertionError("rank-0 logging window is missing")
                if current_metrics is None:
                    raise AssertionError("rank-0 training metrics are missing")
                current_metrics["throughput_samples_per_second"] = (
                    float(sampler_state["global_occurrence_count"]) / update_seconds
                )
                current_metrics["update_seconds"] = update_seconds
                current_metrics["cuda_allocated_peak_bytes"] = (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda" and collect_route1_diagnostics
                    else None
                )
                current_metrics["cuda_reserved_peak_bytes"] = (
                    int(torch.cuda.max_memory_reserved(device))
                    if device.type == "cuda" and collect_route1_diagnostics
                    else None
                )
                if collect_route1_diagnostics:
                    previous = getattr(model, "_training_memory_high_water", (0, 0))
                    peaks = tuple(
                        (
                            max(int(old), int(rank_max_route1_metrics[name]))
                            for (old, name) in zip(
                                previous,
                                (
                                    "rank_max_peak_memory_bytes",
                                    "rank_max_reserved_memory_bytes",
                                ),
                            )
                        )
                    )
                    model._training_memory_high_water = peaks
                    (
                        current_metrics["cuda_allocated_peak_bytes"],
                        current_metrics["cuda_reserved_peak_bytes"],
                    ) = peaks
                current_metrics["clip_coefficient"] = (
                    1.0
                    if preclip_norm <= 0.0
                    else min(1.0, float(postclip_norm) / float(preclip_norm))
                )
                current_metrics.update(
                    {
                        "lr_R": float(backend.optimizer.param_groups[0]["lr"]),
                        "preclip_grad_R": preclip_norm,
                        "postclip_grad_R": postclip_norm,
                    }
                )
                logging_window.add(current_metrics)
                if decision.should_log:
                    window_means = logging_window.snapshot(
                        reset=bool(decision.should_log)
                    )
                    _write_step_audit(
                        epoch=int(sampler_state["epoch"]),
                        step=step,
                        total_steps=total_updates,
                        owner=owner,
                        result=result,
                        route=route,
                        components=component_audit,
                        latest_component_audit=component_audit_report,
                        latest_component_audit_step=component_audit_step,
                        specificity_detail=route1_specificity_detail,
                        current_metrics=current_metrics,
                        window_means=window_means,
                        step_audit_jsonl=output / "step_audit.jsonl",
                    )
                    if lifecycle_logger is None:
                        raise AssertionError("rank-0 lifecycle logger is missing")
                    lifecycle_logger.append_train(
                        route=route,
                        owner=owner,
                        phase=phase_for_epoch(
                            int(sampler_state["epoch"]),
                            route1_epochs=int(config.route1_epochs),
                        ).name,
                        step=step,
                        total_steps=total_updates,
                        epoch=float(sampler_state["epoch"])
                        + (int(sampler_state["global_batch_index"]) + 1)
                        / steps_per_epoch,
                        window_metrics=window_means,
                        specificity_detail=route1_specificity_detail,
                        component_audit=component_audit or None,
                        elapsed_seconds=time.perf_counter() - lifecycle_started,
                    )
            control_checkpoint: Path | None = None
            control_identity: BridgeCheckpointIdentity | None = None
            control_seal: SealedCheckpoint | None = None
            if decision.should_save or decision.should_evaluate:
                (control_checkpoint, control_identity, control_seal) = (
                    materialize_control_checkpoint(
                        run_dir=output,
                        route=route,
                        step=step,
                        materialize=lambda checkpoint_directory: (
                            write_control_checkpoint(
                                checkpoint_directory=checkpoint_directory,
                                state=sampler_state,
                                current_step=step,
                            )
                        ),
                    )
                )
            if decision.should_save:
                if control_checkpoint is None or control_seal is None:
                    raise AssertionError("save event lacks a checkpoint")
                rank0_control(
                    lambda: finish_save(step, int(sampler_state["epoch"]), control_seal)
                )
            if decision.should_evaluate:
                if (
                    control_checkpoint is None
                    or control_identity is None
                    or control_seal is None
                ):
                    raise AssertionError("evaluation event lacks its exact checkpoint")
                rank0_control(
                    lambda: (
                        str(
                            checkpoint_manager.begin_evaluation(
                                step=step,
                                epoch=int(sampler_state["epoch"]),
                                checkpoint=control_checkpoint,
                                ordinary_checkpoint=bool(decision.should_save),
                                materialize=lambda _path: None,
                                sealed_checkpoint=control_seal,
                            )
                        )
                        if checkpoint_manager is not None
                        else None
                    )
                )
                evaluate_control_checkpoint(
                    checkpoint=control_checkpoint,
                    identity=control_identity,
                    state=sampler_state,
                    current_step=step,
                    sealed_checkpoint=control_seal,
                )
        if route1_prefetch is not None:
            route1_prefetch.close()
        completed_epoch_boundary = bool(
            epoch_had_update
            and int(last_sampler_state["global_batch_index"]) == steps_per_epoch - 1
        )
        if epoch_had_update and (not completed_epoch_boundary):
            if step < update_limit:
                raise RuntimeError("phase stopped before a declared control frontier")
            break
        if step >= update_limit and step < total_updates:
            break
    terminal_integrity_started = time.perf_counter()
    _synchronized_update_state_finite_or_raise(
        owner_parameters,
        backend.integrity_optimizer,
        device=device,
        label="terminal integrity boundary",
    )
    integrity_scan_total_seconds += time.perf_counter() - terminal_integrity_started
    if step == total_updates:
        if evaluation_arguments is None or evaluation_bundle is None:
            raise RuntimeError("completed formal phase lacks validation reports")

        def validate_terminal_frontier() -> str:
            ledger = validate_completed_report_ledger()
            if not ledger or int(ledger[-1]["step"]) != int(total_updates):
                raise ValueError(
                    "completed phase lacks a terminal committed validation report"
                )
            if diagnostic_only:
                latest = pointer_reference(f"{route}_latest_checkpoint.txt")
                if latest is None:
                    raise ValueError("diagnostic phase lacks its latest pointer")
                return str(latest[0])
            best = pointer_reference(f"{route}_best_checkpoint.txt")
            if best is None:
                raise ValueError("completed phase lacks its training-time best pointer")
            return _validate_terminal_best_reference(ledger, best)

        rank0_control(validate_terminal_frontier)

    def publish_final_report() -> None:
        final_entry = pointer_reference(f"{route}_latest_checkpoint.txt")
        final_reference = (
            None
            if final_entry is None
            else {"path": str(final_entry[0]), "sha256": final_entry[1].artifact_sha256}
        )
        report_path = report_output / f"bridge-train-step-{step}.json"
        write_atomic_json(
            report_path,
            {
                **artifact_header("think-bridge.training.final-report"),
                "objective_version": OBJECTIVE_VERSION,
                "phase": phase,
                "route": route,
                "method": config.method,
                "seed": int(arguments.seed),
                "run_label": str(arguments.run_label),
                "active_owner": owner,
                "updates": int(step),
                "integrity_scan_total_seconds": float(integrity_scan_total_seconds),
                "steps_per_epoch": int(steps_per_epoch),
                "optimizer_steps_per_epoch": [
                    [int(epoch_value), int(step_count)]
                    for (
                        epoch_value,
                        step_count,
                    ) in course_clock.optimizer_steps_per_epoch
                ],
                "optimizer_step_offsets": [
                    [int(epoch_value), int(step_offset)]
                    for (
                        epoch_value,
                        step_offset,
                    ) in course_clock.optimizer_step_offsets
                ],
                "phase_total_updates": int(total_updates),
                "epoch_sample_count": int(last_sampler_state["epoch_occurrence_count"]),
                "retained_sample_count": int(
                    last_sampler_state["retained_occurrence_count"]
                ),
                "route1_occurrence_sampler_schema": ROUTE1_OCCURRENCE_SAMPLER_SCHEMA,
                "occurrence_weighting_unit": OCCURRENCE_WEIGHTING_UNIT,
                "occurrence_batch_unit": OCCURRENCE_BATCH_UNIT,
                "route1_population": route1_population,
                "route1_course_epochs": route1_course_epochs,
                "route1_course_steps": route1_course_steps,
                "route1_course_weight": route1_course_weight,
                "route1_match_weight": route1_match_weight,
                "route1_specific_weight": route1_specific_weight,
                "route1_specificity_loss": config.route1_specificity_loss,
                "route1_specificity_negative_kl_cap": config.route1_specificity_negative_kl_cap,
                "route1_specificity_temperature": config.route1_specificity_temperature,
                "route1_specificity_include_direct": bool(
                    config.route1_specificity_include_direct
                ),
                "route1_eval_null_mode": route1_eval_null_mode,
                "route1_max_optimizer_updates": route1_max_optimizer_updates,
                "specificity_sample_count_at_last_update": int(
                    last_sampler_state.get("global_specific_sample_count", 0)
                ),
                "selected_r_sha256": selected_r_sha256,
                "cache_identity_sha256": cache_identity_sha256,
                "checkpoint": final_reference,
                "status": "phase_complete" if step == total_updates else "interrupted",
                **(
                    {
                        "component_gradient_audit_step": int(component_audit_step),
                        "component_gradient_audit": dict(component_audit_report),
                    }
                    if component_audit_report is not None
                    and component_audit_step is not None
                    else {}
                ),
            },
            replace_mismatch=True,
        )

    rank0_control(publish_final_report)
    return 0
