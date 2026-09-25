"""Dependency-light fixed-vector protocols for Route1 distributed diagnostics."""

from __future__ import annotations

import math
import zlib
from typing import Mapping, Sequence


ROUTE1_COMPONENT_AUDIT_NORM_KEYS = (
    "grad_norm_course_z_vjp",
    "grad_norm_match_owner_z_vjp",
    "grad_norm_specific_owner_z_vjp",
)

ROUTE1_COMPONENT_AUDIT_FAILURE_KEYS = (
    "missing_audit_graph",
    "local_computation_error",
    "nonfinite_z_vjp_norm",
    "course_z_vjp_missing",
    "match_owner_z_vjp_missing",
    "specific_owner_z_vjp_missing",
    "specific_wrong_donor_z_vjp_missing",
    "owner_z_vjp_alignment_unproven",
    "forbidden_frozen_owner",
)


def route1_rendezvous_tag(label: str) -> int:
    """Return a stable bounded tag for one Route1 collective boundary."""

    if not isinstance(label, str) or not label:
        raise ValueError("Route1 rendezvous label must be a non-empty string")
    # Twenty-four bits keep the sum-of-squares phase check inside signed int64
    # for substantially more ranks than the maintained 4/8-GPU topologies.
    return int(zlib.crc32(label.encode("utf-8")) & 0xFFFFFF)


def route1_rendezvous_moments_aligned(
    *, world_size: int, tag_sum: int, tag_square_sum: int
) -> bool:
    """Return whether one reduced tag population contains a single value."""

    if isinstance(world_size, bool) or int(world_size) <= 0:
        raise ValueError("Route1 rendezvous world size must be positive")
    if int(tag_sum) < 0 or int(tag_square_sum) < 0:
        raise ValueError("Route1 rendezvous moments must be nonnegative")
    return int(tag_sum) * int(tag_sum) == int(world_size) * int(tag_square_sum)


def owner_z_vjp_views_strictly_aligned(
    match_views: Sequence[object], specific_views: Sequence[object]
) -> bool:
    """Return whether both VJPs target the same ordered owner-z objects."""

    return (
        bool(match_views)
        and len(match_views) == len(specific_views)
        and all(
            match_view is specific_view
            for match_view, specific_view in zip(match_views, specific_views)
        )
    )


def encode_route1_component_audit_reduction(
    *,
    norm_values: Sequence[float],
    specific_objective_active: bool,
    failure_flags: Mapping[str, bool],
    match_specific_owner_z_vjp_dot: float = 0.0,
    specific_wrong_donor_z_vjp_norm: float = 0.0,
    specific_wrong_donor_branch_present: bool = False,
    owner_z_vjp_cosine_aligned: bool = False,
) -> tuple[float, ...]:
    """Encode one rank's z-space audit into the fixed SUM layout."""

    if len(norm_values) != len(ROUTE1_COMPONENT_AUDIT_NORM_KEYS):
        raise ValueError("Route1 audit norm vector has the wrong length")
    unknown = set(failure_flags).difference(ROUTE1_COMPONENT_AUDIT_FAILURE_KEYS)
    if unknown:
        raise ValueError(f"unknown Route1 audit failure flags: {sorted(unknown)}")
    flags = {
        key: bool(failure_flags.get(key, False))
        for key in ROUTE1_COMPONENT_AUDIT_FAILURE_KEYS
    }
    squared_norms: list[float] = []
    for value in norm_values:
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            flags["nonfinite_z_vjp_norm"] = True
            number = 0.0
        squared_norms.append(number * number)
    dot = float(match_specific_owner_z_vjp_dot)
    donor = float(specific_wrong_donor_z_vjp_norm)
    if not math.isfinite(dot):
        flags["nonfinite_z_vjp_norm"] = True
        dot = 0.0
    if not math.isfinite(donor) or donor < 0.0:
        flags["nonfinite_z_vjp_norm"] = True
        donor = 0.0
    if not owner_z_vjp_cosine_aligned:
        dot = 0.0
    return tuple(
        squared_norms
        + [dot, donor * donor]
        + [
            float(bool(specific_objective_active)),
            float(bool(specific_wrong_donor_branch_present)),
            float(bool(owner_z_vjp_cosine_aligned)),
        ]
        + [float(flags[key]) for key in ROUTE1_COMPONENT_AUDIT_FAILURE_KEYS]
    )


def finalize_route1_component_audit_reduction(
    reduced: Sequence[float], *, step: int
) -> dict[str, float | bool | int | str]:
    """Return one global z-space pressure report without gating training."""

    norm_count = len(ROUTE1_COMPONENT_AUDIT_NORM_KEYS)
    geometry_count = 2
    activity_count = 3
    expected = (
        norm_count
        + geometry_count
        + activity_count
        + len(ROUTE1_COMPONENT_AUDIT_FAILURE_KEYS)
    )
    if len(reduced) != expected:
        raise ValueError("reduced Route1 audit vector has the wrong length")
    values = tuple(float(value) for value in reduced)
    dot = values[norm_count]
    donor_squared = values[norm_count + 1]
    activity_offset = norm_count + geometry_count
    failure_offset = activity_offset + activity_count
    failures = [
        key
        for index, key in enumerate(ROUTE1_COMPONENT_AUDIT_FAILURE_KEYS)
        if values[failure_offset + index] > 0.0
    ]
    if any(not math.isfinite(value) or value < 0.0 for value in values[:norm_count]):
        failures.append("nonfinite_z_vjp_norm")
    specific_objective_active = bool(values[activity_offset] > 0.0)
    specific_wrong_donor_branch_present = bool(values[activity_offset + 1] > 0.0)
    owner_z_vjp_cosine_aligned = (
        bool(values[activity_offset + 2] > 0.0)
        and "owner_z_vjp_alignment_unproven" not in failures
    )
    if specific_objective_active and not specific_wrong_donor_branch_present:
        failures.append("specific_wrong_donor_z_vjp_missing")
    failures = list(dict.fromkeys(failures))
    result: dict[str, float | bool | int | str] = {
        key: math.sqrt(values[index])
        for index, key in enumerate(ROUTE1_COMPONENT_AUDIT_NORM_KEYS)
    }
    match_squared = values[1]
    specific_squared = values[2]
    cosine_denominator = math.sqrt(match_squared * specific_squared)
    if owner_z_vjp_cosine_aligned and cosine_denominator > 0.0:
        raw_cosine = dot / cosine_denominator
        result["grad_cosine_match_vs_specific_owner_z_vjp"] = max(
            -1.0, min(1.0, raw_cosine)
        )
    result["grad_norm_specific_wrong_donor_z_vjp"] = math.sqrt(max(donor_squared, 0.0))
    result.update(
        specific_objective_active=specific_objective_active,
        specific_wrong_donor_branch_present=(specific_wrong_donor_branch_present),
        owner_z_vjp_cosine_aligned=owner_z_vjp_cosine_aligned,
        frozen_F_embedding_D=("forbidden_frozen_owner" not in failures),
        frozen_teacher_direct_prefix=("forbidden_frozen_owner" not in failures),
        executable_wrong_donor_z_vjp_active=(
            specific_wrong_donor_branch_present
            and "specific_wrong_donor_z_vjp_missing" not in failures
        ),
        component_gradient_audit_status=(
            "measured_with_findings" if failures else "measured"
        ),
        component_gradient_audit_step=int(step),
    )
    if failures:
        result["component_gradient_audit_findings"] = ",".join(failures)
    return result


def finalize_route1_log_reduction(
    reduced: Sequence[float],
    *,
    global_counts: Sequence[int],
    loss_weights: Sequence[float],
) -> dict[str, float | int]:
    """Convert globally summed DDP contributions into scientific losses."""

    if len(reduced) != 5 or len(global_counts) != 3 or len(loss_weights) != 3:
        raise ValueError("reduced Route1 log vector has the wrong length")
    values = tuple(float(value) for value in reduced)
    counts = tuple(int(value) for value in global_counts)
    weights = tuple(float(value) for value in loss_weights)
    if (
        values[4] > 0.0
        or counts[0] < 0
        or any(value < 0 for value in counts)
        or any(not math.isfinite(value) or value < 0.0 for value in weights)
    ):
        raise RuntimeError("Route1 global log reduction protocol is invalid")
    means = tuple(
        values[index] if count > 0 else 0.0 for index, count in enumerate(counts)
    )
    if any(not math.isfinite(value) for value in means):
        raise RuntimeError("Route1 global log reduction is non-finite")
    valid_tokens = values[3]
    if not math.isfinite(valid_tokens) or valid_tokens < 0.0:
        raise RuntimeError("Route1 global valid-answer token count is invalid")
    return {
        "loss_answer_course": means[0],
        "loss_match": means[1],
        "loss_specific": means[2],
        "loss_total": sum(weight * mean for weight, mean in zip(weights, means)),
        "valid_course_tokens": int(round(valid_tokens)),
    }
