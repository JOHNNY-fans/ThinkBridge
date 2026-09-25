"""Small Bridge-specific control-event scheduler around custom scientific steps."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Protocol

from think_bridge.model.checkpoint_policy import checkpoint_path

from think_bridge.model.contract import (
    ROUTE1_OCCURRENCE_SAMPLER_SCHEMA,
    normalize_route1_null_mode,
)
from think_bridge.training.occurrence_sampler import occurrence_steps_per_epoch


@dataclass(frozen=True)
class ControlDecision:
    should_log: bool
    should_save: bool
    should_evaluate: bool

    def as_tuple(self) -> tuple[bool, bool, bool]:
        return self.should_log, self.should_save, self.should_evaluate


@dataclass(frozen=True)
class ControlSchedule:
    save_steps: int
    eval_steps: int
    logging_steps: int

    def __post_init__(self) -> None:
        for name, value in (
            ("save_steps", self.save_steps),
            ("eval_steps", self.eval_steps),
            ("logging_steps", self.logging_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def at(
        self, *, step: int, epoch_end: bool, terminal: bool = False
    ) -> ControlDecision:
        if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
            raise ValueError("control step must be a positive integer")
        if not isinstance(epoch_end, bool):
            raise ValueError("epoch_end must be boolean")
        if not isinstance(terminal, bool):
            raise ValueError("terminal must be boolean")
        return ControlDecision(
            should_log=bool(step == 1 or epoch_end or step % self.logging_steps == 0),
            should_save=bool(terminal or step % self.save_steps == 0),
            should_evaluate=bool(terminal or step % self.eval_steps == 0),
        )


def materialize_control_checkpoint(
    *,
    run_dir: Path,
    route: str,
    step: int,
    materialize: Callable[[Path], tuple[Any, Any]],
) -> tuple[Path, Any, Any]:
    """Build one canonical control checkpoint and preserve its typed payload."""

    checkpoint_directory = checkpoint_path(run_dir, route, step)
    identity, seal = materialize(checkpoint_directory)
    return checkpoint_directory, identity, seal


def build_long_lived_evaluation_arguments(
    *,
    manifest: Path,
    target_index: Path,
    records: Path,
    donors: Path | None,
    route: str,
    metric_policy: Mapping[str, Any],
    seed: int,
    generation_seed: int,
    route1_eval_local_row_batch: int,
    no_progress: bool,
    answer_max_tokens: int | None = None,
    route1_null_mode: str = "direct",
    route1_eval_backend: str = "torch",
    eval_vllm_host: str = "",
    eval_vllm_port: int = 0,
    eval_vllm_request_timeout_seconds: float = 0.0,
    eval_vllm_physical_chunk_size: int = 0,
    eval_vllm_data_parallel_size: int | None = None,
    eval_vllm_max_num_seqs: int = 64,
    eval_vllm_gpu_memory_utilization: float | None = None,
    max_eval_samples: int | None = None,
) -> SimpleNamespace:
    """Build the complete train-to-eval handoff from explicit artifact paths."""
    if route not in {"route1"}:
        raise ValueError("long-lived evaluation route must be route1")
    route1_like = True
    from think_bridge.training.metric_registry import report_metric_policy

    sealed_metric_policy = report_metric_policy(
        route=route, value=metric_policy
    ).as_dict()
    if donors is None:
        raise ValueError("long-lived Route1 evaluation requires donors")
    if (
        isinstance(answer_max_tokens, bool)
        or not isinstance(answer_max_tokens, int)
        or answer_max_tokens <= 0
    ):
        raise ValueError("long-lived Route1 evaluation requires an answer horizon")
    if (
        isinstance(generation_seed, bool)
        or not isinstance(generation_seed, int)
        or generation_seed < 0
    ):
        raise ValueError("long-lived evaluation requires a generation seed")
    if route1_eval_backend not in {"torch", "vllm"}:
        raise ValueError("long-lived Route1 eval backend must be torch or vllm")
    if route1_eval_backend == "vllm" and (
        eval_vllm_host != "127.0.0.1"
        or isinstance(eval_vllm_port, bool)
        or int(eval_vllm_port) <= 0
        or (not math.isfinite(float(eval_vllm_request_timeout_seconds)))
        or (float(eval_vllm_request_timeout_seconds) <= 0.0)
        or isinstance(eval_vllm_physical_chunk_size, bool)
        or (int(eval_vllm_physical_chunk_size) <= 0)
    ):
        raise ValueError("long-lived Route1 vLLM eval runtime is invalid")
    if max_eval_samples is not None and (
        isinstance(max_eval_samples, bool)
        or not isinstance(max_eval_samples, int)
        or max_eval_samples <= 0
    ):
        raise ValueError("max_eval_samples must be a positive validation limit or None")
    values = dict(
        manifest=Path(manifest),
        target_index=Path(target_index),
        records=Path(records),
        donors=None if donors is None else Path(donors),
        split="validation",
        route=route,
        metric_policy=sealed_metric_policy,
        seed=int(seed),
        answer_max_tokens=int(answer_max_tokens),
        route1_eval_local_row_batch=int(route1_eval_local_row_batch),
        route1_null_mode=normalize_route1_null_mode(route1_null_mode),
        route1_eval_backend=route1_eval_backend,
        eval_vllm_host=eval_vllm_host,
        eval_vllm_port=eval_vllm_port,
        eval_vllm_request_timeout_seconds=eval_vllm_request_timeout_seconds,
        eval_vllm_physical_chunk_size=eval_vllm_physical_chunk_size,
        eval_vllm_data_parallel_size=eval_vllm_data_parallel_size,
        eval_vllm_max_num_seqs=eval_vllm_max_num_seqs,
        eval_vllm_gpu_memory_utilization=eval_vllm_gpu_memory_utilization,
        generation_seed=int(generation_seed),
        max_eval_samples=int(max_eval_samples)
        if max_eval_samples is not None
        else None,
        no_progress=bool(no_progress),
        report=None,
        checkpoint=None,
        selected_r=None,
    )
    return SimpleNamespace(**values)


def training_evaluation_service_arguments(
    config: Any, arguments: Any
) -> dict[str, Any]:
    """Validation uses local HF replicas; vLLM remains training-generation only."""
    return {"route1_eval_backend": "torch"}


def configure_control_evaluation_arguments(
    evaluation_arguments: Any,
    *,
    checkpoint: Path,
    report_path: Path,
    generation_seed: int,
) -> Any:
    """Bind report identity while keeping checkpoint validation RNG run-common."""

    if (
        isinstance(generation_seed, bool)
        or not isinstance(generation_seed, int)
        or generation_seed < 0
    ):
        raise ValueError("validation generation seed must be a non-negative integer")
    evaluation_arguments.generation_seed = int(generation_seed)
    evaluation_arguments.report = Path(report_path)
    evaluation_arguments.checkpoint = Path(checkpoint)
    return evaluation_arguments


def _resume_sampler_position(
    *, step: int, sampler_state: Mapping[str, Any], route1_epochs: int
) -> tuple[int, bool]:
    global_batch_index = sampler_state.get("global_batch_index")
    retained_occurrences = sampler_state.get("retained_occurrence_count")
    epoch = sampler_state.get("epoch")
    artifact_type = sampler_state.get("artifact_type")
    schema_version = sampler_state.get("schema_version")
    if "optimizer_global_batch" not in sampler_state:
        raise ValueError("resume sampler state lacks optimizer_global_batch")
    optimizer_global_batch = sampler_state["optimizer_global_batch"]
    if (
        isinstance(global_batch_index, bool)
        or not isinstance(global_batch_index, int)
        or global_batch_index < 0
        or isinstance(retained_occurrences, bool)
        or (not isinstance(retained_occurrences, int))
        or (retained_occurrences <= 0)
        or isinstance(epoch, bool)
        or (not isinstance(epoch, int))
        or isinstance(optimizer_global_batch, bool)
        or (not isinstance(optimizer_global_batch, int))
        or (optimizer_global_batch <= 0)
    ):
        raise ValueError("resume sampler state cannot reconstruct an epoch boundary")
    if (
        isinstance(route1_epochs, bool)
        or not isinstance(route1_epochs, int)
        or route1_epochs < 0
    ):
        raise ValueError("resume Route1 epoch count must be a nonnegative integer")
    if schema_version != 1:
        raise ValueError("resume sampler schema_version must be integer 1")
    if artifact_type == ROUTE1_OCCURRENCE_SAMPLER_SCHEMA:
        if route1_epochs == 0:
            raise ValueError("Route1 sampler requires a positive Route1 epoch count")
        first_epoch = 0
        drop_last = True
        if retained_occurrences % optimizer_global_batch != 0:
            raise ValueError(
                "Route1 resume frontier is not a strict logical-batch domain"
            )
        optimizer_update_offset = sampler_state.get("optimizer_update_offset")
        if (
            isinstance(optimizer_update_offset, bool)
            or not isinstance(optimizer_update_offset, int)
            or optimizer_update_offset < 0
        ):
            raise ValueError("Route1 resume frontier lacks its epoch step prefix")
    else:
        raise ValueError("resume sampler schema is not a current Bridge route")
    steps_per_epoch = occurrence_steps_per_epoch(
        retained_occurrences,
        global_batch_size=optimizer_global_batch,
        drop_last=drop_last,
    )
    if global_batch_index >= steps_per_epoch:
        raise ValueError("resume sampler cursor exceeds its retained occurrence domain")
    expected_step = (
        int(optimizer_update_offset) + global_batch_index + 1
        if artifact_type == ROUTE1_OCCURRENCE_SAMPLER_SCHEMA
        else (epoch - first_epoch) * steps_per_epoch + global_batch_index + 1
    )
    if epoch < first_epoch or int(step) != expected_step:
        raise ValueError("resume step does not match its exact sampler frontier")
    epoch_end = global_batch_index == steps_per_epoch - 1
    return (steps_per_epoch, epoch_end)


def resume_sampler_epoch_end(
    *,
    step: int,
    sampler_state: Mapping[str, Any],
    route1_epochs: int,
) -> bool:
    """Return the sealed epoch-boundary status for either current route schema."""

    _steps_per_epoch, epoch_end = _resume_sampler_position(
        step=step,
        sampler_state=sampler_state,
        route1_epochs=route1_epochs,
    )
    return epoch_end


def resume_step_requires_evaluation(
    schedule: ControlSchedule,
    *,
    step: int,
    sampler_state: Mapping[str, Any],
    route1_epochs: int,
) -> bool:
    """Reconstruct the exact post-update eval event for a saved frontier.

    The sampler state is the checkpointed scientific frontier. Ordinary epoch
    ends affect logging only; evaluation is due at its configured cadence or at
    the route's sealed terminal update.
    """

    epoch_end = resume_sampler_epoch_end(
        step=step,
        sampler_state=sampler_state,
        route1_epochs=route1_epochs,
    )
    terminal = sampler_state.get("phase_terminal")
    if not isinstance(terminal, bool):
        raise ValueError("resume sampler state lacks the terminal frontier seal")
    return schedule.at(
        step=step, epoch_end=epoch_end, terminal=terminal
    ).should_evaluate


def resume_evaluation_action(
    schedule: ControlSchedule,
    *,
    step: int,
    sampler_state: Mapping[str, Any],
    pending_evaluation: bool,
    completed_report: bool,
    route1_epochs: int,
) -> str:
    """Classify the only legal resume transaction without replaying an update."""

    if not isinstance(pending_evaluation, bool) or not isinstance(
        completed_report, bool
    ):
        raise ValueError("resume evaluation transaction flags must be boolean")
    required = resume_step_requires_evaluation(
        schedule,
        step=step,
        sampler_state=sampler_state,
        route1_epochs=route1_epochs,
    )
    if not required:
        if pending_evaluation or completed_report:
            raise ValueError("save-only resume cannot carry evaluation state")
        return "resume_update"
    if pending_evaluation:
        return "finish_pending_evaluation"
    if completed_report:
        return "validate_completed_evaluation"
    return "begin_pending_evaluation"


class BridgeControlCallback(Protocol):
    """Control-only callback surface; it never owns loss/backward/optimizer."""

    def on_log(self, *, step: int, epoch: int) -> None: ...

    def on_save(self, *, step: int, epoch: int) -> None: ...

    def on_evaluate(self, *, step: int, epoch: int) -> None: ...

    def on_epoch_end(self, *, step: int, epoch: int) -> None: ...


class PhaseRunner:
    """Dispatch post-update control events without touching scientific state."""

    def __init__(
        self,
        schedule: ControlSchedule,
        callbacks: tuple[BridgeControlCallback, ...] = (),
    ) -> None:
        self.schedule = schedule
        self.callbacks = tuple(callbacks)

    def after_finite_update(
        self, *, step: int, epoch: int, epoch_end: bool, terminal: bool = False
    ) -> ControlDecision:
        decision = self.schedule.at(step=step, epoch_end=epoch_end, terminal=terminal)
        for callback in self.callbacks:
            if decision.should_log:
                callback.on_log(step=step, epoch=epoch)
            if decision.should_save:
                callback.on_save(step=step, epoch=epoch)
            if decision.should_evaluate:
                callback.on_evaluate(step=step, epoch=epoch)
            if epoch_end:
                callback.on_epoch_end(step=step, epoch=epoch)
        return decision
