"""Dependency-light deterministic objective-window plans for Bridge Stage1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


ROUTE1_ACTIVE_DOMAIN_SCHEMA = "bridge-route1-fixed-population-epoch-domain"
ROUTE1_PHYSICAL_COST_POLICY_SCHEMA = (
    "bridge-route1-active-physical-branches-cost-policy"
)


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def complete_optimizer_window_count(
    active_record_count: int,
    *,
    local_microbatch: int,
    world_size: int,
    gradient_accumulation_steps: int,
) -> int:
    """Return the maximum count of complete logical optimizer windows."""

    if (
        isinstance(active_record_count, bool)
        or not isinstance(active_record_count, int)
        or active_record_count < 0
    ):
        raise ValueError("active_record_count must be a nonnegative integer")
    optimizer_batch = (
        _positive_integer(local_microbatch, "local_microbatch")
        * _positive_integer(world_size, "world_size")
        * _positive_integer(gradient_accumulation_steps, "gradient_accumulation_steps")
    )
    return int(active_record_count) // optimizer_batch


def _power_two_bucket(length: int) -> int:
    return 1 << (_positive_integer(length, "physical sequence length") - 1).bit_length()


@dataclass(frozen=True)
class Route1PhysicalCostEstimate:
    """Planner-only work estimate for the Route1 branches that will execute."""

    answer_course_tokens: int
    stopped_rollout_tokens: int


def route1_active_physical_cost(
    row: Mapping[str, Any],
    *,
    epoch: int,
    boundary_token_count: int,
    compute_course: bool,
    compute_match: bool,
    compute_specific: bool,
    legal_wrong_candidate_count: int = 0,
    trajectory_max_steps: int = 2048,
) -> Route1PhysicalCostEstimate:
    """Estimate only enabled course and need-objective physical branches."""

    if not all(
        isinstance(value, bool)
        for value in (compute_course, compute_match, compute_specific)
    ):
        raise TypeError("Route1 physical cost switches must be boolean")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("Route1 physical cost epoch must be nonnegative")
    boundary = _positive_integer(boundary_token_count, "boundary_token_count")
    trajectory = _positive_integer(trajectory_max_steps, "trajectory_max_steps")
    if (
        isinstance(legal_wrong_candidate_count, bool)
        or not isinstance(legal_wrong_candidate_count, int)
        or legal_wrong_candidate_count < 0
    ):
        raise ValueError("legal_wrong_candidate_count must be nonnegative")
    prompt = row.get("prompt_ids")
    views = row.get("answer_views")
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("Route1 physical cost requires a nonempty prompt")
    if not isinstance(views, list) or not views:
        raise ValueError("Route1 physical cost requires legal answer views")
    if any(
        not isinstance(view, Mapping)
        or not isinstance(view.get("cot_ids"), list)
        or not isinstance(view.get("answer_ids"), list)
        or not view["answer_ids"]
        for view in views
    ):
        raise ValueError("Route1 physical cost received a malformed answer view")
    course_tokens = (
        max(len(prompt) + 64 + boundary + len(view["answer_ids"]) for view in views)
        if compute_course
        else 0
    )
    need_objectives_active = bool(
        (row.get("quadrant") in {"B", "C", "G"} and compute_match)
        or (row.get("quadrant") in {"C", "G"} and compute_specific)
    )
    stopped_tokens = 0
    if need_objectives_active:
        cot_tokens = sum(len(view["cot_ids"]) for view in views)
        if compute_course:
            # Course-on placement estimates each currently active B/C/D branch;
            # only the native suffix geometry changes across the curriculum.
            stopped_tokens = trajectory + cot_tokens
            if compute_specific:
                stopped_tokens += trajectory * (1 + int(legal_wrong_candidate_count))
        else:
            deployed_prefix = len(prompt) + 64 + boundary
            rollout_tokens = deployed_prefix + trajectory
            native_reference_tokens = deployed_prefix + cot_tokens + trajectory
            true_stopped_tokens = deployed_prefix + trajectory
            stopped_tokens = (
                rollout_tokens + native_reference_tokens + true_stopped_tokens
            )
            if compute_specific:
                # One frozen direct condition and each selected live donor.
                # This remains a placement heuristic, not an exact FLOP budget.
                stopped_tokens += (deployed_prefix + trajectory) * (
                    1 + int(legal_wrong_candidate_count)
                )
    if course_tokens == 0 and stopped_tokens == 0:
        raise ValueError("Route1 physical cost row has no enabled execution branch")
    return Route1PhysicalCostEstimate(
        answer_course_tokens=int(course_tokens),
        stopped_rollout_tokens=int(stopped_tokens),
    )


def route1_active_physical_length_key(
    row: Mapping[str, Any],
    *,
    epoch: int,
    boundary_token_count: int,
    compute_course: bool,
    compute_match: bool,
    compute_specific: bool,
    trajectory_max_steps: int = 2048,
) -> tuple[int, int, int]:
    """Bucket the actual enabled branch geometry without inactive-course data."""

    estimate = route1_active_physical_cost(
        row,
        epoch=int(epoch),
        boundary_token_count=int(boundary_token_count),
        compute_course=compute_course,
        compute_match=compute_match,
        compute_specific=compute_specific,
        trajectory_max_steps=int(trajectory_max_steps),
    )
    views = row["answer_views"]
    if estimate.answer_course_tokens > 0:
        cot_length = 0
        answer_length = max(len(view["answer_ids"]) for view in views)
        deployed_prefix = (
            len(row["prompt_ids"]) + 64 + cot_length + int(boundary_token_count)
        )
        return (
            _power_two_bucket(deployed_prefix + answer_length),
            _power_two_bucket(deployed_prefix),
            _power_two_bucket(answer_length),
        )
    cot_length = max(len(view["cot_ids"]) for view in views)
    deployed_prefix = (
        len(row["prompt_ids"]) + 64 + cot_length + int(boundary_token_count)
    )
    physical_lengths = [
        deployed_prefix + int(trajectory_max_steps),
        len(row["prompt_ids"])
        + 64
        + int(boundary_token_count)
        + int(trajectory_max_steps),
    ]
    physical_prefixes = [
        deployed_prefix,
        len(row["prompt_ids"]) + 64 + int(boundary_token_count),
    ]
    return (
        _power_two_bucket(max(physical_lengths)),
        _power_two_bucket(max(physical_prefixes)),
        _power_two_bucket(int(trajectory_max_steps)),
    )


def route1_active_epoch_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    population: str,
    epoch: int,
    compute_course: bool,
    compute_match: bool,
    compute_specific: bool,
) -> tuple[Mapping[str, Any], ...]:
    """Return the rows with at least one genuinely active Route1 leaf."""

    if population not in {"staged", "answer-only", "gold-reference"}:
        raise ValueError("Route1 population must be staged or answer-only")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("Route1 epoch must be a nonnegative integer")
    switches = (compute_course, compute_match, compute_specific)
    if not all(isinstance(value, bool) for value in switches):
        raise TypeError("Route1 active-leaf switches must be boolean")
    if not any(switches):
        raise ValueError("Route1 requires at least one active loss leaf")
    normalized = tuple(rows)
    record_ids = [str(row.get("record_id", "")) for row in normalized]
    if any(not value for value in record_ids) or len(set(record_ids)) != len(
        record_ids
    ):
        raise ValueError("Route1 active domain requires unique record ids")

    allowed: set[str] = set()
    if compute_course:
        allowed.update(("B", "C", "D"))
    if compute_match:
        allowed.update(("B", "C"))
    if compute_specific:
        allowed.add("C")
    if any(row.get("quadrant") not in {"B", "C", "D"} for row in normalized):
        raise ValueError("staged Route1 domain must contain only B/C/D rows")
    return tuple(row for row in normalized if str(row["quadrant"]) in allowed)


@dataclass(frozen=True)
class OptimizerCourseClock:
    optimizer_steps_per_epoch: tuple[tuple[int, int], ...]
    optimizer_step_offsets: tuple[tuple[int, int], ...]
    full_total_updates: int
    total_updates: int
    terminal_optimizer_update: int

    def steps_for_epoch(self, epoch: int) -> int:
        values = dict(self.optimizer_steps_per_epoch)
        if int(epoch) not in values:
            raise ValueError("epoch is outside the optimizer course")
        return int(values[int(epoch)])

    def offset_for_epoch(self, epoch: int) -> int:
        values = dict(self.optimizer_step_offsets)
        if int(epoch) not in values:
            raise ValueError("epoch is outside the optimizer course")
        return int(values[int(epoch)])

    def cursor_for_step(self, *, epoch: int, completed_updates: int) -> int:
        offset = self.offset_for_epoch(int(epoch))
        steps = self.steps_for_epoch(int(epoch))
        completed = int(completed_updates)
        if completed < offset or completed > offset + steps:
            raise ValueError("completed update is outside this epoch")
        return completed - offset


def optimizer_course_clock(
    optimizer_steps_per_epoch: Sequence[tuple[int, int]],
    *,
    max_optimizer_updates: int | None,
) -> OptimizerCourseClock:
    """Build prefix offsets for a course whose epoch domains may differ."""

    values = tuple(
        (int(epoch), int(steps)) for epoch, steps in optimizer_steps_per_epoch
    )
    if not values or len({epoch for epoch, _ in values}) != len(values):
        raise ValueError("optimizer course requires unique epochs")
    if any(epoch < 0 or steps < 0 for epoch, steps in values):
        raise ValueError("optimizer course epochs/steps must be nonnegative")
    offsets: list[tuple[int, int]] = []
    total = 0
    for epoch, steps in values:
        offsets.append((epoch, total))
        total += steps
    if total <= 0:
        raise ValueError("optimizer course has no complete active optimizer window")
    if max_optimizer_updates is None:
        capped = total
    else:
        capped = min(
            total,
            _positive_integer(max_optimizer_updates, "max_optimizer_updates"),
        )
    return OptimizerCourseClock(
        optimizer_steps_per_epoch=values,
        optimizer_step_offsets=tuple(offsets),
        full_total_updates=total,
        total_updates=capped,
        terminal_optimizer_update=capped,
    )
