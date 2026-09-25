"""Route1 response CE, forward-KL Match, and question-specific supervision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from think_bridge.model.contract import ROUTE1_COURSE_REDUCTIONS

try:
    import torch
    import torch.distributed as torch_distributed
    import torch.nn.functional as torch_functional
except ModuleNotFoundError:  # dependency-free contract tests still import this module
    torch = None
    torch_distributed = None
    torch_functional = None


# Bound the physical LM-head matrix without changing the exact full-vocabulary

# Stage1 runner; padding rows never enter one of these projections.
ANSWER_LOGIT_ROW_BUDGET = 512


@dataclass(frozen=True)
class PerOccurrenceGoldLoss:
    loss: Any
    per_occurrence: Any
    valid_token_count: int
    occurrence_count: int


@dataclass(frozen=True)
class Route1PopulationLoss:
    loss: Any
    loss_answer_course: Any
    loss_match: Any
    loss_specific: Any
    course_active_count: int
    course_c_active_count: int
    course_b_d_side_active_count: int
    match_active_count: int
    specific_active_count: int


@dataclass(frozen=True)
class Route1CourseReductionScales:
    """DDP-local gradient scales and detached global side weights."""

    c_local_sum_scale: float
    direct_side_local_sum_scale: float
    detached_c_weight: float
    detached_direct_side_weight: float
    active_sample_count: int


def route1_course_reduction_scales(
    *,
    reduction: str,
    global_c_sample_count: int,
    global_direct_side_sample_count: int,
    world_size: int,
) -> Route1CourseReductionScales:
    """Return exact no-collective scales for the active course populations.

    Every epoch gives equal weight to the global C mean and the global
    direct-correct-side (B union D) mean when both exist.  The local-sum scales
    include ``world_size`` because DDP averages gradients.
    """

    counts = (global_c_sample_count, global_direct_side_sample_count)
    if reduction not in ROUTE1_COURSE_REDUCTIONS:
        raise ValueError(f"unknown Route1 course reduction: {reduction!r}")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in counts
    ):
        raise ValueError("Route1 course side counts must be nonnegative integers")
    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size <= 0
    ):
        raise ValueError("Route1 course world size must be a positive integer")
    c_count, direct_count = map(int, counts)
    active = c_count + direct_count
    if active == 0:
        return Route1CourseReductionScales(0.0, 0.0, 0.0, 0.0, 0)
    if c_count and direct_count:
        return Route1CourseReductionScales(
            0.5 * float(world_size) / float(c_count),
            0.5 * float(world_size) / float(direct_count),
            0.5,
            0.5,
            active,
        )
    if c_count:
        return Route1CourseReductionScales(
            float(world_size) / float(c_count), 0.0, 1.0, 0.0, active
        )
    return Route1CourseReductionScales(
        0.0, float(world_size) / float(direct_count), 0.0, 1.0, active
    )


@dataclass(frozen=True)
class StreamedForwardKL:
    loss: Any
    per_occurrence: Any
    valid_token_count: int
    occurrence_count: int
    teacher_prefix_entropy: Any | None = None
    teacher_prefix_token_nll: Any | None = None
    teacher_prefix_token_top1: Any | None = None


@dataclass(frozen=True)
class StreamedForwardKLTeacherReference:
    """Stopped per-token teacher statistics reusable by many students.

    ``expected_lm_head_weight`` and ``expected_lm_head_bias`` are the exact
    first moments of the frozen LM head under the native teacher distribution.
    Together with ``teacher_entropy`` they are sufficient for
    ``KL(q_teacher || p_student)`` without retaining ``[tokens, vocabulary]``
    teacher probabilities between calls.  The stored tensors retain the
    teacher's padded ``[B,T,...]`` geometry so a repeated donor batch can
    select owner occurrences without re-projecting the teacher.
    """

    token_mask: Any
    occurrence_token_counts: Any
    expected_lm_head_weight: Any
    expected_lm_head_bias: Any
    teacher_entropy: Any

    def index_select(
        self, occurrence_indices: Any
    ) -> "StreamedForwardKLTeacherReference":
        if occurrence_indices.ndim != 1 or occurrence_indices.dtype != torch.long:
            raise ValueError("teacher reference indices must be one int64 vector")
        return StreamedForwardKLTeacherReference(
            token_mask=self.token_mask.index_select(0, occurrence_indices),
            occurrence_token_counts=self.occurrence_token_counts.index_select(
                0, occurrence_indices
            ),
            expected_lm_head_weight=self.expected_lm_head_weight.index_select(
                0, occurrence_indices
            ),
            expected_lm_head_bias=self.expected_lm_head_bias.index_select(
                0, occurrence_indices
            ),
            teacher_entropy=self.teacher_entropy.index_select(0, occurrence_indices),
        )

    def index_select_and_trim(
        self,
        occurrence_indices: Any,
        token_mask: Any,
    ) -> "StreamedForwardKLTeacherReference":
        """Select repeated owners and remove only verified padding columns."""

        selected = self.index_select(occurrence_indices)
        if (
            token_mask.ndim != 2
            or selected.token_mask.ndim != 2
            or token_mask.size(0) != selected.token_mask.size(0)
            or token_mask.size(1) <= 0
            or token_mask.size(1) > selected.token_mask.size(1)
            or selected.occurrence_token_counts.ndim != 1
            or selected.occurrence_token_counts.size(0) != token_mask.size(0)
            or selected.expected_lm_head_weight.ndim != 3
            or selected.expected_lm_head_weight.shape[:2] != selected.token_mask.shape
            or selected.expected_lm_head_bias.shape != selected.token_mask.shape
            or selected.teacher_entropy.shape != selected.token_mask.shape
        ):
            raise ValueError("teacher reference trim geometry is invalid")
        width = int(token_mask.size(1))
        valid = token_mask.bool()
        selected_valid = selected.token_mask.bool()
        counts = valid.long().sum(dim=1)
        if (
            not torch.equal(selected_valid[:, :width], valid)
            or bool(selected_valid[:, width:].any())
            or not torch.equal(selected.occurrence_token_counts, counts)
        ):
            raise ValueError(
                "teacher reference trim would discard a valid nonpadding token"
            )
        return StreamedForwardKLTeacherReference(
            token_mask=selected_valid[:, :width],
            occurrence_token_counts=selected.occurrence_token_counts,
            expected_lm_head_weight=(selected.expected_lm_head_weight[:, :width]),
            expected_lm_head_bias=selected.expected_lm_head_bias[:, :width],
            teacher_entropy=selected.teacher_entropy[:, :width],
        )

    def _index_select_and_trim_validated(
        self,
        occurrence_indices: Any,
        token_mask: Any,
    ) -> "StreamedForwardKLTeacherReference":
        """Select a prefix already verified by the stopped-prefix frontier."""

        selected = self.index_select(occurrence_indices)
        width = int(token_mask.size(1))
        return StreamedForwardKLTeacherReference(
            token_mask=token_mask.bool(),
            occurrence_token_counts=selected.occurrence_token_counts,
            expected_lm_head_weight=(selected.expected_lm_head_weight[:, :width]),
            expected_lm_head_bias=selected.expected_lm_head_bias[:, :width],
            teacher_entropy=selected.teacher_entropy[:, :width],
        )


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("torch is required for ThinkBridge tensor objectives")


def _distributed_active() -> bool:
    return bool(
        torch_distributed is not None
        and torch_distributed.is_available()
        and torch_distributed.is_initialized()
    )


def _global_detached_scalar(value: Any) -> Any:
    detached = value.detach().clone().float()
    if _distributed_active():
        torch_distributed.all_reduce(detached, op=torch_distributed.ReduceOp.SUM)
    return detached


def distributed_global_mean(
    local_sum: Any,
    local_count: Any,
    *,
    global_count_override: Any | None = None,
) -> Any:
    """Exact global value with DDP-average-correct local gradient scaling."""

    _require_torch()
    if not torch.is_tensor(local_sum) or local_sum.numel() != 1:
        raise ValueError("local_sum must be a scalar tensor")
    count = torch.as_tensor(local_count, dtype=torch.float32, device=local_sum.device)
    if count.numel() != 1 or bool((count < 0).any()):
        raise ValueError("local_count must be one non-negative scalar")
    global_sum = _global_detached_scalar(local_sum)
    if global_count_override is None:
        global_count = _global_detached_scalar(count)
    else:
        global_count = torch.as_tensor(
            global_count_override, dtype=torch.float32, device=local_sum.device
        ).detach()
        if global_count.numel() != 1:
            raise ValueError("global_count_override must be one scalar")
    if float(global_count) <= 0.0:
        raise RuntimeError("global denominator is zero")
    world_size = torch_distributed.get_world_size() if _distributed_active() else 1
    backward_value = local_sum.float() * (float(world_size) / global_count)
    global_value = global_sum / global_count
    return backward_value + (global_value - backward_value.detach())


def ddp_global_mean_local_contribution(
    local_sum: Any,
    *,
    global_count: int,
    world_size_override: int | None = None,
) -> Any:
    """Return one no-collective contribution to a DDP global mean.

    DDP averages owner gradients at the final GAS microstep.  Therefore each
    rank must backpropagate ``world_size * local_sum / global_count``.  The
    caller reconstructs the detached global scalar once, after every physical
    chunk; this hot-path helper intentionally performs no collective.
    """

    _require_torch()
    if not torch.is_tensor(local_sum) or local_sum.numel() != 1:
        raise ValueError("local_sum must be a scalar tensor")
    if isinstance(global_count, bool) or int(global_count) <= 0:
        raise ValueError("global_count must be a positive integer")
    if world_size_override is None:
        world_size = (
            int(torch_distributed.get_world_size()) if _distributed_active() else 1
        )
    else:
        if isinstance(world_size_override, bool) or int(world_size_override) <= 0:
            raise ValueError("world_size_override must be a positive integer")
        world_size = int(world_size_override)
    return local_sum.float() * (float(world_size) / float(global_count))


def _per_occurrence_mean_from_flat_validated(flat_values: Any, counts: Any) -> Any:
    """Restore rows after the caller has validated prefix counts once."""

    if counts.numel() == 0:
        return flat_values.new_zeros((0,))
    owners = torch.repeat_interleave(
        torch.arange(counts.numel(), device=counts.device),
        counts,
        output_size=int(flat_values.numel()),
    )
    sums = flat_values.new_zeros((counts.numel(),)).index_add(0, owners, flat_values)
    return sums / counts.to(dtype=sums.dtype)


def _streamed_forward_kl_rows_from_hidden_validated(
    teacher_hidden: Any,
    student_hidden: Any,
    token_mask: Any,
    occurrence_token_counts: Any,
    *,
    lm_head_weight: Any,
    lm_head_bias: Any | None = None,
) -> Any:
    """Exact hidden-state KL rows after one caller-owned mask validation."""

    from think_bridge.model.stage1_losses import (
        forward_kl_from_teacher_logprob,
    )

    teacher = teacher_hidden.detach()
    weight = lm_head_weight.detach()
    bias = None if lm_head_bias is None else lm_head_bias.detach().to(weight.dtype)
    if int(weight.size(0)) <= 0:
        raise ValueError("vocabulary must be positive")
    if bias is not None and tuple(bias.shape) != (weight.size(0),):
        raise ValueError("LM head bias shape mismatch")
    valid = token_mask.bool()
    flat_teacher = teacher[valid]
    flat_student = student_hidden[valid]
    token_kl_parts: list[Any] = []
    for start in range(0, int(flat_teacher.size(0)), ANSWER_LOGIT_ROW_BUDGET):
        stop = min(start + ANSWER_LOGIT_ROW_BUDGET, int(flat_teacher.size(0)))
        with torch.no_grad():
            teacher_logprob = torch_functional.log_softmax(
                torch_functional.linear(
                    flat_teacher[start:stop].to(weight.dtype), weight, bias
                ).float(),
                dim=-1,
            )
        student_logits = torch_functional.linear(
            flat_student[start:stop].to(weight.dtype), weight, bias
        ).float()
        token_kl_parts.append(
            forward_kl_from_teacher_logprob(teacher_logprob, student_logits)
        )
    flat_token_kl = (
        torch.cat(token_kl_parts)
        if token_kl_parts
        else student_hidden.sum().reshape(()).mul(0.0).expand((0,))
    )
    return _per_occurrence_mean_from_flat_validated(
        flat_token_kl, occurrence_token_counts
    )


def _per_occurrence_gold_ce_rows_from_hidden_validated(
    hidden: Any,
    target_ids: Any,
    target_mask: Any,
    occurrence_token_counts: Any,
    *,
    lm_head_weight: Any,
    lm_head_bias: Any | None = None,
) -> Any:
    """Exact CE rows after one caller-owned target-mask validation."""

    states = hidden
    weight = lm_head_weight.detach()
    bias = None if lm_head_bias is None else lm_head_bias.detach().to(weight.dtype)
    if int(weight.size(0)) <= 0:
        raise ValueError("vocabulary must be positive")
    if bias is not None and tuple(bias.shape) != (weight.size(0),):
        raise ValueError("LM head bias shape mismatch")
    valid = target_mask.bool()
    flat_states = states[valid]
    flat_targets = target_ids.long()[valid]
    token_nll_parts: list[Any] = []
    for start in range(0, int(flat_states.size(0)), ANSWER_LOGIT_ROW_BUDGET):
        stop = min(start + ANSWER_LOGIT_ROW_BUDGET, int(flat_states.size(0)))
        logits = torch_functional.linear(
            flat_states[start:stop].to(weight.dtype), weight, bias
        ).float()
        token_nll_parts.append(
            torch_functional.cross_entropy(
                logits, flat_targets[start:stop], reduction="none"
            )
        )
    flat_token_nll = (
        torch.cat(token_nll_parts)
        if token_nll_parts
        else states.sum().reshape(()).mul(0.0).expand((0,))
    )
    return _per_occurrence_mean_from_flat_validated(
        flat_token_nll, occurrence_token_counts
    )
