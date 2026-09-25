"""Concise, idempotent rank-0 lifecycle logging for ThinkBridge.

The full scientific report and per-step audit streams remain separate.  This
module only publishes small operator-facing events after their owning durable
transaction has committed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from think_bridge.model.artifact_schema import (
    LIFECYCLE_EVENT,
    RESUME_ROLLBACK,
    STAGE_ARTIFACT_LOCATOR,
    STAGE_RESOLVED_CONFIG,
    artifact_header,
    require_artifact_header,
)

from think_bridge.training.metric_registry import (
    MetricPolicy,
    report_metric_policy,
    require_report_metric_policy,
    select_best_candidate,
)
from think_bridge.training.progress import write_progress


_RESUME_ROLLBACK_MARKER = ".resume-rollback.transaction.json"
_RESUME_ROLLBACK_LOG_STAGE = ".resume-rollback.logging.jsonl.building"
_RESUME_ROLLBACK_AUDIT_STAGE = ".resume-rollback.step-audit.jsonl.building"
_ROUTES = {"route1"}
_LIFECYCLE_EVENTS = {
    "train",
    "eval",
    "checkpoint",
    "selection",
    "phase_end",
    "run_end",
}
_TRAIN_STAGES = {"Route1": "route1", "Route1-A": "route1", "Route1-B": "route1"}
_STAGE_ROUTES = {
    **_TRAIN_STAGES,
    "Route1": "route1",
    "Stage1": None,
}
_ROUTE1_CONDITIONS = (
    "true_z",
    "direct",
    "wrong_z_1",
    "wrong_z_2",
    "wrong_z_3",
    "wrong_z_4",
    "wrong_z_5",
    "wrong_z_6",
    "wrong_z_7",
    "wrong_z_8",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_fsynced(path: Path, payload: bytes) -> None:
    target = Path(path)
    descriptor = os.open(target, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o644)
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError("short transaction write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        return b""
    return (
        "".join(
            json.dumps(
                dict(row), sort_keys=False, separators=(",", ":"), allow_nan=False
            )
            + "\n"
            for row in rows
        )
    ).encode("utf-8")


def _recover_resume_rollback(run_dir: Path) -> None:
    """Finish a published two-file resume rollback after process failure."""

    run = Path(run_dir)
    marker_path = run / _RESUME_ROLLBACK_MARKER
    if not marker_path.exists():
        return
    if marker_path.is_symlink() or not marker_path.is_file():
        raise ValueError("resume log rollback marker is not a regular file")
    marker = _read_json(marker_path)
    expected_fields = {
        "artifact_type",
        "schema_version",
        "route",
        "frontier_step",
        "logging",
        "step_audit",
    }
    if (
        set(marker) != expected_fields
        or marker.get("artifact_type") != RESUME_ROLLBACK
        or marker.get("schema_version") != 1
        or marker.get("route") not in _ROUTES
        or isinstance(marker.get("frontier_step"), bool)
        or not isinstance(marker.get("frontier_step"), int)
        or int(marker["frontier_step"]) < 0
    ):
        raise ValueError("resume log rollback marker schema mismatch")
    specifications = (
        (
            "logging",
            "logging.jsonl",
            _RESUME_ROLLBACK_LOG_STAGE,
        ),
        (
            "step_audit",
            "step_audit.jsonl",
            _RESUME_ROLLBACK_AUDIT_STAGE,
        ),
    )
    for field, target_name, staged_name in specifications:
        row = marker.get(field)
        if not isinstance(row, Mapping) or set(row) != {
            "before_sha256",
            "after_sha256",
        }:
            raise ValueError("resume log rollback file identity is malformed")
        before = _require_sha256(
            row.get("before_sha256"), f"resume rollback {field} before_sha256"
        )
        after = _require_sha256(
            row.get("after_sha256"), f"resume rollback {field} after_sha256"
        )
        target = run / target_name
        staged = run / staged_name
        if target.is_symlink() or not target.is_file():
            raise ValueError("resume log rollback target is missing or unsafe")
        current = _file_sha256(target)
        if current == after:
            continue
        if current != before:
            raise ValueError("resume log rollback target changed during transaction")
        if staged.is_symlink() or not staged.is_file() or _file_sha256(staged) != after:
            raise ValueError("resume log rollback staged file is missing or changed")
        os.replace(staged, target)
        _fsync_directory(run)
    for staged_name in (_RESUME_ROLLBACK_LOG_STAGE, _RESUME_ROLLBACK_AUDIT_STAGE):
        staged = run / staged_name
        if staged.exists():
            if staged.is_symlink() or not staged.is_file():
                raise ValueError("resume log rollback staging path is unsafe")
            staged.unlink()
    marker_path.unlink()
    _fsync_directory(run)


def _require_sha256(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _require_route(route: str) -> str:
    if route not in _ROUTES:
        raise ValueError("lifecycle route must be route1")
    return route


def _train_stage(*, route: str, phase: str) -> str:
    mapping = {
        ("route1", "route1"): "Route1",
        ("route1", "route1_phase_a"): "Route1-A",
        ("route1", "route1_phase_b"): "Route1-B",
        ("J",): "Joint",
    }
    try:
        return mapping[route, phase]
    except KeyError as exc:
        raise ValueError(
            "lifecycle train phase differs from the Bridge recipe"
        ) from exc


def _event_kind(row: Mapping[str, Any]) -> str | None:
    event = row.get("event")
    if isinstance(event, str):
        return event
    if row.get("stage") in _TRAIN_STAGES:
        return "train"
    return None


def _event_route(row: Mapping[str, Any]) -> str | None:
    route = row.get("route")
    if route in _ROUTES:
        return str(route)
    stage = row.get("stage")
    return _STAGE_ROUTES.get(str(stage))


def _event_identity(row: Mapping[str, Any]) -> str:
    event_id = row.get("event_id")
    if isinstance(event_id, str) and event_id:
        return event_id
    event = _event_kind(row)
    route = _event_route(row)
    step = row.get("step")
    if event in {"train", "checkpoint", "eval"}:
        if (
            route not in _ROUTES
            or isinstance(step, bool)
            or not isinstance(step, int)
            or int(step) <= 0
        ):
            raise ValueError(f"compact {event} row identity is malformed")
    if event == "train":
        return f"train:{route}:{int(step)}"
    if event == "checkpoint":
        reason = row.get("reason")
        if reason not in {"scheduled_save", "evaluation_commit"}:
            raise ValueError("compact checkpoint row identity is malformed")
        return f"checkpoint:{route}:{int(step)}:{reason}"
    if event == "eval":
        return f"eval:{route}:{int(step)}"
    if event == "selection":
        if route not in _ROUTES:
            raise ValueError("compact selection row identity is malformed")
        return f"selection:{route}"
    if event == "phase_end":
        updates = row.get("updates")
        status = row.get("status")
        if (
            route not in _ROUTES
            or isinstance(updates, bool)
            or not isinstance(updates, int)
            or int(updates) <= 0
            or not isinstance(status, str)
            or not status
        ):
            raise ValueError("compact phase-end row identity is malformed")
        return f"phase_end:{route}:{int(updates)}:{status}"
    if event == "run_end":
        if row.get("stage") != "Stage1":
            raise ValueError("compact run-end row identity is malformed")
        return "run_end:stage1"
    raise ValueError("lifecycle event identity is missing")


def _require_metric_policy(report: Mapping[str, Any], *, route: str) -> MetricPolicy:
    """Require the exact canonical policy used to select this report."""

    return require_report_metric_policy(report, route=_require_route(route))


def _require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _require_nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return int(value)


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(_require_number(seconds, "duration"))))
    days, remainder = divmod(total, 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    if minutes or hours or days:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)


def _rounded_number(value: Any, label: str, *, digits: int = 8) -> float:
    return round(_require_number(value, label), int(digits))


def _rounded_optional(value: Any, label: str, *, digits: int = 8) -> float | None:
    return None if value is None else _rounded_number(value, label, digits=digits)


def _lifecycle_console_message(row: Mapping[str, Any]) -> str | None:
    """Return one short operator summary; machine JSON stays file-only."""
    event = _event_kind(row)
    if event == "train":
        summary: dict[str, Any] = {
            "loss": row.get("loss"),
            "grad_norm": row.get("grad_norm"),
            "learning_rate": row.get("learning_rate"),
            "iteration": row.get("iteration"),
            "epoch": row.get("epoch"),
            "elapsed_time": row.get("elapsed_time"),
            "remaining_time": row.get("remaining_time"),
            "memory(GiB)": row.get("memory(GiB)"),
            "train_speed(s/it)": row.get("train_speed(s/it)"),
            "throughput(samples/s)": row.get("throughput(samples/s)"),
        }
        route = _event_route(row)
        summary.update(
            loss_ce=row.get("loss_ce"),
            loss_match=row.get("loss_match"),
            loss_specific=row.get("loss_specific"),
            distill_b_sample_count=row.get("distill_b_sample_count"),
            distill_c_sample_count=row.get("distill_c_sample_count"),
            distill_reference_sample_count=row.get("distill_reference_sample_count"),
            specificity_eligible_pair_count=row.get("specificity_eligible_pair_count"),
            specificity_selected_pair_count=row.get("specificity_selected_pair_count"),
            specificity_same_prompt_record_pair_count=row.get(
                "specificity_same_prompt_record_pair_count"
            ),
        )
        return "Train: " + repr(
            {name: value for (name, value) in summary.items() if value is not None}
        )
    if event == "checkpoint":
        return "Saving: " + repr(
            {
                "stage": row.get("stage"),
                "step": row.get("step"),
                "reason": row.get("reason"),
            }
        )
    if event == "eval":
        if row.get("stage") == "Route1":
            summary = {
                "stage": row.get("stage"),
                "step": row.get("step"),
                "true_z_full_accuracy": row.get("true_z_full_accuracy"),
                "robust_g1": row.get("robust_g1"),
                "robust_g1_ci_low": row.get("robust_g1_ci_low"),
                "b_retention": row.get("b_retention"),
                "d_retention": row.get("d_retention"),
                "c_paired_count": row.get("c_paired_count"),
                "wrong_pair_count": row.get("wrong_pair_count"),
            }
        elif row.get("stage") == "Joint":
            summary = {
                "stage": row.get("stage"),
                "step": row.get("step"),
                "true_z_full_accuracy": row.get("true_z_full_accuracy"),
            }
        else:
            raise ValueError("evaluation lifecycle stage is invalid")
        return "Evaluate: " + repr(
            {name: value for (name, value) in summary.items() if value is not None}
        )
    if event == "selection":
        message = f"bridge lifecycle selection stage={row.get('stage')} selected_step={row.get('selected_step')}"
        if row.get("primary_metrics") is not None:
            message += f" primary_metrics={row.get('primary_metrics')} aggregate_score={row.get('aggregate_score')}"
        return message
    if event == "phase_end":
        return f"bridge lifecycle phase_end stage={row.get('stage')} updates={row.get('updates')} status={row.get('status')}"
    if event == "run_end":
        return f"bridge lifecycle run_end status={row.get('status')}"
    raise ValueError("lifecycle console event kind is invalid")


def _append_json_line(path: Path, row: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        dict(row), sort_keys=False, separators=(",", ":"), allow_nan=False
    )
    payload = (serialized + "\n").encode("utf-8")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError("short JSONL append")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return serialized


def append_step_audit(path: Path, row: Mapping[str, Any]) -> None:
    """Append one detailed scientific audit row without console duplication."""

    if not isinstance(row, Mapping) or not row:
        raise ValueError("step audit row must be a non-empty mapping")
    _append_json_line(Path(path), row)


def _count_condition(
    rows: Sequence[Mapping[str, Any]], condition: str
) -> dict[str, int | float]:
    numerator = sum(bool(row["correct"][condition]) for row in rows)
    denominator = len(rows)
    if denominator <= 0:
        raise ValueError("evaluation population cannot be empty")
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "accuracy": float(numerator / denominator),
    }


def _without_accuracy(count: Mapping[str, int | float]) -> dict[str, int]:
    return {
        "numerator": int(count["numerator"]),
        "denominator": int(count["denominator"]),
    }


def _assert_same_float(observed: Any, expected: float, label: str) -> None:
    value = _require_number(observed, label)
    if not math.isclose(value, float(expected), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"sealed report summary differs from paired rows: {label}")


class LifecycleLogger:
    """Append compact lifecycle events exactly once per stable identity.

    Instantiate this class only on rank 0.  Reopening it during exact resume
    reconstructs the durable event-id set from ``logging.jsonl``.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.run_dir = self.path.parent
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _recover_resume_rollback(self.run_dir)
        self.path.touch(exist_ok=True)
        self._events: dict[str, dict[str, Any]] = {}
        self._reload_events()

    def _reload_events(self) -> None:
        self._events: dict[str, dict[str, Any]] = {}
        with self.path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"lifecycle log has an incomplete row at line {line_number}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(
                        f"lifecycle log row schema mismatch at line {line_number}"
                    )
                if (
                    row.get("artifact_type") != LIFECYCLE_EVENT
                    or row.get("schema_version") != 1
                    or _event_kind(row) not in _LIFECYCLE_EVENTS
                ):
                    raise ValueError(
                        f"lifecycle log row schema mismatch at line {line_number}"
                    )
                try:
                    event_id = _event_identity(row)
                except ValueError as exc:
                    raise ValueError(
                        f"lifecycle log row schema mismatch at line {line_number}: {exc}"
                    ) from exc
                if event_id in self._events:
                    raise ValueError(
                        "lifecycle log contains a duplicate event identity"
                    )
                self._events[event_id] = row

    def rollback_training_after(
        self, *, route: str, frontier_step: int, step_audit_path: Path
    ) -> dict[str, int]:
        """Remove only replayable train/audit rows beyond a sealed resume frontier.

        Evaluation, selection, phase, and run events are committed evidence and
        are never deleted here.  A small durable transaction marker lets the
        next process finish both file replacements if this process exits in
        between them.
        """
        route = _require_route(route)
        if (
            isinstance(frontier_step, bool)
            or not isinstance(frontier_step, int)
            or frontier_step < 0
        ):
            raise ValueError("resume log rollback frontier must be non-negative")
        audit_path = Path(step_audit_path)
        if audit_path.parent.resolve(strict=False) != self.run_dir.resolve(strict=True):
            raise ValueError(
                "step audit rollback path must remain in the run directory"
            )
        if self.path.name != "logging.jsonl" or audit_path.name != "step_audit.jsonl":
            raise ValueError("resume rollback requires the standard lifecycle paths")
        _recover_resume_rollback(self.run_dir)
        audit_path.touch(exist_ok=True)
        if self.path.is_symlink() or audit_path.is_symlink():
            raise ValueError("resume rollback refuses symlinked JSONL targets")
        lifecycle_rows = list(self._events.values())
        kept_lifecycle = [
            row
            for row in lifecycle_rows
            if not (
                _event_kind(row) == "train"
                and _event_route(row) == route
                and (int(row.get("step", -1)) > frontier_step)
            )
        ]
        removed_lifecycle = len(lifecycle_rows) - len(kept_lifecycle)
        audit_rows: list[dict[str, Any]] = []
        with audit_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"step audit has an incomplete row at line {line_number}"
                    ) from exc
                if (
                    not isinstance(row, dict)
                    or row.get("event") != "step_audit"
                    or row.get("route") not in _ROUTES
                    or isinstance(row.get("step"), bool)
                    or (not isinstance(row.get("step"), int))
                    or (int(row["step"]) <= 0)
                ):
                    raise ValueError(
                        "step audit row schema mismatch during resume rollback"
                    )
                audit_rows.append(row)
        kept_audit = [
            row
            for row in audit_rows
            if not (row["route"] == route and int(row["step"]) > frontier_step)
        ]
        removed_audit = len(audit_rows) - len(kept_audit)
        if removed_lifecycle == 0 and removed_audit == 0:
            self._reload_events()
            return {"logging_train_rows": 0, "step_audit_rows": 0}
        before_logging = self.path.read_bytes()
        before_audit = audit_path.read_bytes()
        after_logging = _jsonl_bytes(kept_lifecycle)
        after_audit = _jsonl_bytes(kept_audit)
        log_stage = self.run_dir / _RESUME_ROLLBACK_LOG_STAGE
        audit_stage = self.run_dir / _RESUME_ROLLBACK_AUDIT_STAGE
        marker_path = self.run_dir / _RESUME_ROLLBACK_MARKER
        marker_building = marker_path.with_name(f"{marker_path.name}.building")
        for scratch in (log_stage, audit_stage, marker_building):
            if scratch.exists():
                if scratch.is_symlink() or not scratch.is_file():
                    raise ValueError("resume rollback scratch path is unsafe")
                scratch.unlink()
        _write_fsynced(log_stage, after_logging)
        _write_fsynced(audit_stage, after_audit)
        marker = {
            **artifact_header(RESUME_ROLLBACK),
            "route": route,
            "frontier_step": int(frontier_step),
            "logging": {
                "before_sha256": _bytes_sha256(before_logging),
                "after_sha256": _bytes_sha256(after_logging),
            },
            "step_audit": {
                "before_sha256": _bytes_sha256(before_audit),
                "after_sha256": _bytes_sha256(after_audit),
            },
        }
        _write_fsynced(
            marker_building,
            (json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            ),
        )
        os.replace(marker_building, marker_path)
        _fsync_directory(self.run_dir)
        _recover_resume_rollback(self.run_dir)
        self._reload_events()
        return {
            "logging_train_rows": removed_lifecycle,
            "step_audit_rows": removed_audit,
        }

    def _append(self, row: Mapping[str, Any], *, console: bool = True) -> bool:
        payload = {**artifact_header(LIFECYCLE_EVENT), **dict(row)}
        event = _event_kind(payload)
        if (
            payload.get("artifact_type") != LIFECYCLE_EVENT
            or payload.get("schema_version") != 1
        ):
            raise ValueError("lifecycle event schema mismatch")
        event_id = _event_identity(payload)
        if event not in _LIFECYCLE_EVENTS:
            raise ValueError("lifecycle event kind is invalid")
        # Evaluation can include per-question donor counts. JSON preserves them
        # directly; the compact console summary still selects scalar fields.
        if any(
            (isinstance(value, str) and "\x1b" in value for value in payload.values())
        ):
            raise ValueError("canonical lifecycle rows cannot contain ANSI escapes")
        existing = self._events.get(event_id)
        if existing is not None:
            if existing != payload:
                raise ValueError(
                    "lifecycle event identity was reused with different content"
                )
            return False
        _append_json_line(self.path, payload)
        self._events[event_id] = payload
        if console:
            message = _lifecycle_console_message(payload)
            if message is not None:
                write_progress(message)
        return True

    def append_train(
        self,
        *,
        route: str,
        owner: str,
        phase: str,
        step: int,
        total_steps: int,
        epoch: float,
        window_metrics: Mapping[str, float | int | None],
        elapsed_seconds: float,
        component_audit: Mapping[str, Any] | None = None,
        specificity_detail: Mapping[str, Any] | None = None,
    ) -> bool:
        route = _require_route(route)
        expected_owner = {"route1": "R"}[route]
        if owner != expected_owner:
            raise ValueError("lifecycle train owner differs from route")
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or step <= 0
            or isinstance(total_steps, bool)
            or (not isinstance(total_steps, int))
            or (step > total_steps)
        ):
            raise ValueError("lifecycle train step range is invalid")
        prefix = "R"
        expected_phases = {"route1": {"route1"}}[route]
        if phase not in expected_phases:
            raise ValueError("lifecycle train phase differs from the Bridge recipe")
        speed = _require_number(
            window_metrics["window_mean_update_seconds"], "train_speed(s/it)"
        )
        allocated_peak_bytes = _require_number(
            window_metrics.get(
                "window_max_cuda_allocated_peak_bytes",
                window_metrics["window_mean_cuda_allocated_peak_bytes"],
            ),
            "CUDA allocated peak memory",
        )
        reserved_peak_bytes = _require_number(
            window_metrics.get(
                "window_max_cuda_reserved_peak_bytes",
                window_metrics["window_mean_cuda_reserved_peak_bytes"],
            ),
            "CUDA reserved peak memory",
        )
        audit_fields = {
            "grad_norm_course_z_vjp": None,
            "grad_norm_match_owner_z_vjp": None,
            "grad_norm_specific_owner_z_vjp": None,
            "grad_norm_specific_wrong_donor_z_vjp": None,
            "grad_cosine_match_vs_specific_owner_z_vjp": None,
            "total_reasoner_R_parameter_grad_norm_preclip": None,
            "specific_objective_active": None,
            "specific_wrong_donor_branch_present": None,
            "owner_z_vjp_cosine_aligned": None,
            "frozen_F_embedding_D": None,
            "frozen_teacher_direct_prefix": None,
            "executable_wrong_donor_z_vjp_active": None,
            "component_gradient_audit_method": None,
            "component_gradient_audit_status": None,
            "component_gradient_audit_step": None,
        }
        if component_audit is not None:
            optional_audit_fields = {"grad_cosine_match_vs_specific_owner_z_vjp"}
            allowed_audit_fields = set(audit_fields) | {
                "component_gradient_audit_findings"
            }
            if (
                route != "route1"
                or not set(audit_fields)
                .difference(optional_audit_fields)
                .issubset(component_audit)
                or (not set(component_audit).issubset(allowed_audit_fields))
            ):
                raise ValueError("component gradient audit lifecycle schema mismatch")
            audit_step = component_audit.get("component_gradient_audit_step")
            method = component_audit.get("component_gradient_audit_method")
            status = component_audit.get("component_gradient_audit_status")
            if (
                isinstance(audit_step, bool)
                or not isinstance(audit_step, int)
                or audit_step != step
                or (method != "intermediate_z_vjp")
                or (status not in {"measured", "measured_with_findings"})
            ):
                raise ValueError("component gradient audit lifecycle identity mismatch")
            for name in (
                "grad_norm_course_z_vjp",
                "grad_norm_match_owner_z_vjp",
                "grad_norm_specific_owner_z_vjp",
                "grad_norm_specific_wrong_donor_z_vjp",
                "total_reasoner_R_parameter_grad_norm_preclip",
            ):
                value = component_audit.get(name)
                audit_fields[name] = _require_number(value, f"component audit {name}")
            cosine_name = "grad_cosine_match_vs_specific_owner_z_vjp"
            if cosine_name in component_audit:
                audit_fields[cosine_name] = _require_number(
                    component_audit[cosine_name], f"component audit {cosine_name}"
                )
            for name in (
                "specific_objective_active",
                "specific_wrong_donor_branch_present",
                "owner_z_vjp_cosine_aligned",
                "frozen_F_embedding_D",
                "frozen_teacher_direct_prefix",
                "executable_wrong_donor_z_vjp_active",
            ):
                if not isinstance(component_audit.get(name), bool):
                    raise ValueError(f"component audit {name} must be boolean")
                audit_fields[name] = bool(component_audit[name])
            audit_fields["component_gradient_audit_status"] = status
            audit_fields["component_gradient_audit_step"] = int(audit_step)
            audit_fields["component_gradient_audit_method"] = method
            findings = component_audit.get("component_gradient_audit_findings")
            if findings is not None:
                if not isinstance(findings, str) or not findings:
                    raise ValueError(
                        "component audit findings must be a nonempty string"
                    )
                audit_fields["component_gradient_audit_findings"] = findings
        row = {
            "stage": _train_stage(route=route, phase=phase),
            "step": int(step),
            "loss": _rounded_number(
                window_metrics["window_mean_loss_total"], "window loss"
            ),
            "grad_norm": _rounded_number(
                window_metrics[f"window_mean_preclip_grad_{prefix}"],
                "pre-clip gradient norm",
            ),
            "learning_rate": _rounded_number(
                window_metrics[f"window_mean_lr_{prefix}"], "learning rate"
            ),
            "iteration": f"{step}/{total_steps}",
            "epoch": _rounded_number(epoch, "epoch"),
            "elapsed_time": _format_duration(elapsed_seconds),
            "remaining_time": _format_duration((total_steps - step) * speed),
            "memory(GiB)": round(
                max(allocated_peak_bytes, reserved_peak_bytes) / 1024**3, 2
            ),
            "train_speed(s/it)": round(speed, 6),
            "throughput(samples/s)": _rounded_number(
                window_metrics["window_mean_throughput_samples_per_second"],
                "sample throughput",
            ),
            "loss_ce": _rounded_optional(
                window_metrics.get("window_mean_loss_ce"), "window answer-course loss"
            ),
            **{
                name: _rounded_optional(window_metrics.get(f"window_mean_{name}"), name)
                for name in (
                    "loss_route1",
                    "distill_b_sample_count",
                    "distill_c_sample_count",
                    "distill_reference_sample_count",
                    "specificity_eligible_pair_count",
                    "specificity_selected_pair_count",
                    "specificity_donors_per_owner",
                    "specificity_same_prompt_record_pair_count",
                )
            },
            "loss_match": _rounded_optional(
                window_metrics.get("window_mean_loss_match"), "window match loss"
            ),
            "loss_specific": _rounded_optional(
                window_metrics.get("window_mean_loss_specific"),
                "window specificity loss",
            ),
            **(
                {
                    f"{name}_current_update": _rounded_optional(
                        specificity_detail.get(name), name
                    )
                    for name in (
                        "specificity_true_kl",
                        "specificity_wrong_kl",
                        "specificity_margin_active_fraction",
                        "specificity_soft_weight_mean",
                        "specificity_negative_cap_fraction",
                        "specificity_effective_wrong_kl",
                    )
                    if specificity_detail.get(name) is not None
                }
                if specificity_detail
                and specificity_detail.get("specificity_covered_owner_count", 0) > 0
                else {}
            ),
            "course_active_count": _rounded_optional(
                window_metrics.get("window_mean_course_active_count"),
                "window course sample count",
            ),
            "course_c_active_count": _rounded_optional(
                window_metrics.get("window_mean_course_c_active_count"),
                "window course C sample count",
            ),
            "course_b_d_side_active_count": _rounded_optional(
                window_metrics.get("window_mean_course_b_d_side_active_count"),
                "window course B union D side sample count",
            ),
            "match_active_count": _rounded_optional(
                window_metrics.get("window_mean_match_active_count"),
                "window match sample count",
            ),
            "specific_active_count": _rounded_optional(
                window_metrics.get("window_mean_specific_active_count"),
                "window specificity sample count",
            ),
            **{
                name: _rounded_optional(value, f"audit {name}")
                if isinstance(value, (int, float)) and (not isinstance(value, bool))
                else value
                for (name, value) in audit_fields.items()
                if name
                not in {
                    "component_gradient_audit_status",
                    "component_gradient_audit_step",
                    "component_gradient_audit_method",
                }
            },
            "component_gradient_audit_status": audit_fields[
                "component_gradient_audit_status"
            ],
            "component_gradient_audit_step": audit_fields[
                "component_gradient_audit_step"
            ],
            "component_gradient_audit_method": audit_fields[
                "component_gradient_audit_method"
            ],
        }
        return self._append(
            {name: value for (name, value) in row.items() if value is not None}
        )

    def append_checkpoint(
        self,
        *,
        route: str,
        owner: str,
        phase: str,
        step: int,
        reason: str,
        checkpoint_path: Path,
        checkpoint_sha256: str,
        latest_checkpoint_path: Path,
        latest_checkpoint_sha256: str,
        best_checkpoint_path: Path | None,
        best_checkpoint_sha256: str | None,
    ) -> bool:
        route = _require_route(route)
        expected_owner = {"route1": "R"}[route]
        if owner != expected_owner:
            raise ValueError("checkpoint owner differs from route")
        expected_phases = {"route1": {"route1"}}[route]
        if phase not in expected_phases:
            raise ValueError("checkpoint phase differs from the Bridge recipe")
        if reason not in {"scheduled_save", "evaluation_commit"}:
            raise ValueError("checkpoint lifecycle reason is invalid")
        if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
            raise ValueError("checkpoint lifecycle step is invalid")
        checkpoint = Path(checkpoint_path).resolve(strict=True)
        latest = Path(latest_checkpoint_path).resolve(strict=True)
        checkpoint_hash = _require_sha256(checkpoint_sha256, "checkpoint_sha256")
        latest_hash = _require_sha256(
            latest_checkpoint_sha256, "latest_checkpoint_sha256"
        )
        if checkpoint != latest or checkpoint_hash != latest_hash:
            raise ValueError("committed checkpoint differs from the latest pointer")
        if (best_checkpoint_path is None) != (best_checkpoint_sha256 is None):
            raise ValueError("best checkpoint pointer/hash must be present together")
        if best_checkpoint_path is not None:
            Path(best_checkpoint_path).resolve(strict=True)
            _require_sha256(best_checkpoint_sha256, "best_checkpoint_sha256")
        return self._append(
            {
                "event": "checkpoint",
                "stage": _train_stage(route=route, phase=phase),
                "step": int(step),
                "reason": reason,
            },
            console=False,
        )

    def _validated_evaluation_inputs(
        self, *, receipt: Mapping[str, Any], manifest_path: Path
    ) -> tuple[str, Path, dict[str, Any], str, dict[str, Any]]:
        from think_bridge.training.checkpoint_manager import (
            resolve_run_artifact_locator,
        )

        route = _require_route(str(receipt.get("route", "")))
        report_path = Path(str(receipt.get("report_path", "")))
        if report_path.is_symlink() or not report_path.is_file():
            raise FileNotFoundError(
                f"sealed evaluation report is missing: {report_path}"
            )
        report_path = report_path.resolve(strict=True)
        report_sha256 = _file_sha256(report_path)
        if report_sha256 != _require_sha256(
            receipt.get("report_sha256"), "evaluation report_sha256"
        ):
            raise ValueError("evaluation report differs from its committed receipt")
        report = _read_json(report_path)
        reported_checkpoint = resolve_run_artifact_locator(
            self.run_dir,
            report.get("checkpoint_path"),
            label="evaluation report checkpoint",
        )
        receipt_checkpoint = resolve_run_artifact_locator(
            self.run_dir,
            receipt.get("checkpoint_path"),
            label="evaluation receipt checkpoint",
        )
        step = receipt.get("step")
        epoch = receipt.get("epoch")
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or isinstance(epoch, bool)
            or (not isinstance(epoch, int))
            or (report.get("step") != step)
            or (report.get("epoch") != epoch)
            or (reported_checkpoint != receipt_checkpoint)
            or (report.get("checkpoint_sha256") != receipt.get("checkpoint_sha256"))
        ):
            raise ValueError("evaluation report differs from its committed identity")
        manifest = Path(manifest_path).resolve(strict=True)
        manifest_payload = _read_json(manifest)
        resolved_path = self.run_dir / "resolved_config.json"
        if resolved_path.is_symlink():
            raise ValueError("resolved stage config must be a regular file")
        if resolved_path.is_file():
            resolved = _read_json(resolved_path)
            require_artifact_header(
                resolved, STAGE_RESOLVED_CONFIG, label="resolved stage config"
            )
            stage = str(resolved.get("stage", ""))
            expected_roles = {"reasoner-sft": "eval_behavior"}
            if stage not in expected_roles:
                raise ValueError("resolved split stage is invalid")
            expected_role = expected_roles[stage]
            scientific_identity = _require_sha256(
                resolved.get("scientific_identity_sha256"),
                "resolved stage scientific_identity_sha256",
            )
            artifacts = _read_json(self.run_dir / "artifacts.json")
            provenance = artifacts.get("validation_provenance")
            if (
                artifacts.get("artifact_type") != STAGE_ARTIFACT_LOCATOR
                or artifacts.get("schema_version") != 1
                or artifacts.get("stage") != stage
                or (artifacts.get("scientific_identity_sha256") != scientific_identity)
                or (not isinstance(provenance, Mapping))
                or (set(provenance) != {"role", "path"})
            ):
                raise ValueError("split validation provenance binding mismatch")
            role = str(provenance["role"])
            resolved_inputs = resolved.get("inputs")
            resolved_input = (
                resolved_inputs.get(expected_role)
                if isinstance(resolved_inputs, Mapping)
                else None
            )
            if role != expected_role or not isinstance(resolved_input, Mapping):
                raise ValueError(
                    "split validation provenance differs from resolved input"
                )
            resolved_behavior = Path(str(resolved_input.get("path", ""))).resolve(
                strict=True
            )
            provenance_behavior = Path(str(provenance["path"])).resolve(strict=True)
            if (
                resolved_behavior != provenance_behavior
                or not provenance_behavior.is_file()
            ):
                raise ValueError(
                    "split validation provenance differs from resolved input"
                )
        elif resolved_path.exists():
            raise ValueError("resolved stage config must be a regular file")
        else:
            args = _read_json(self.run_dir / "args.json")
            arguments = args.get("arguments")
            if not isinstance(arguments, Mapping):
                raise ValueError("run arguments do not identify validation behavior")
            raise ValueError("run lacks the resolved supervised-training configuration")
        return (route, report_path, report, report_sha256, manifest_payload)

    def append_evaluation(
        self, *, receipt: Mapping[str, Any], manifest_path: Path
    ) -> bool:
        (route, _report_path, report, _report_sha256, _manifest_payload) = (
            self._validated_evaluation_inputs(
                receipt=receipt, manifest_path=manifest_path
            )
        )
        common = {
            "event": "eval",
            "stage": {"route1": "Route1"}[route],
            "step": int(receipt["step"]),
            "epoch": int(receipt["epoch"]),
        }
        summary = self._route1_evaluation_summary(report, route=route)
        metric_policy = summary["metric_policy"]
        row = {
            **common,
            "primary_metrics": ",".join(metric_policy["primary_metrics"]),
            "metric_aggregation": metric_policy["aggregation"],
            "true_z_full_accuracy": _rounded_number(
                summary["true_z_full_accuracy"], "true_z_full_accuracy"
            ),
            "diagnostics_executed": summary["diagnostics_executed"],
            **{
                name: _rounded_optional(summary[name], name)
                for name in ("robust_g1", "robust_g1_ci_low")
                if name in summary
            },
            "b_retention": _rounded_optional(summary["b_retention"], "b_retention"),
            "d_retention": _rounded_optional(summary["d_retention"], "d_retention"),
            "control_available_count": summary["control_available_count"],
            "control_unavailable_count": summary["control_unavailable_count"],
            "wrong_donor_counts": summary["wrong_donor_counts"],
            "c_paired_count": int(summary["c_paired_count"]),
            "wrong_pair_count": int(summary["wrong_pair_count"]),
            "eval_sample_count": int(summary["eval_sample_count"]),
            "max_eval_samples": summary["max_eval_samples"],
            "evaluation_subset_sha256": summary["evaluation_subset_sha256"],
            "direct_baseline_identity_sha256": summary[
                "direct_baseline_identity_sha256"
            ],
            "generation_request_total": summary["generation_request_counts"]["total"],
        }
        return self._append(row)

    @staticmethod
    def _route1_evaluation_summary(
        report: Mapping[str, Any], *, route: str
    ) -> dict[str, Any]:
        rows = report.get("paired_rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError("Route1 report lacks paired rows")
        diagnostics = report.get("diagnostics_executed", True)
        if not isinstance(diagnostics, bool):
            raise ValueError("Route1 diagnostics flag must be boolean")
        clean_rows: list[Mapping[str, Any]] = []
        for row in rows:
            correct = row.get("correct") if isinstance(row, Mapping) else None
            population = row.get("population") if isinstance(row, Mapping) else None
            expected_conditions = (
                (
                    {"true_z", "direct"}
                    | {
                        f"wrong_z_{i}"
                        for i in range(
                            1,
                            1
                            + sum(
                                (str(k).startswith("wrong_z_") for k in correct or {})
                            ),
                        )
                    }
                    if diagnostics
                    else {"true_z", "direct"}
                )
                if population == "C"
                else {"true_z"}
            )
            if (
                not isinstance(row, Mapping)
                or not isinstance(row.get("direct_correct"), bool)
                or population not in {"B", "C", "D", "E"}
                or (not isinstance(correct, Mapping))
                or (set(correct) != expected_conditions)
                or any((not isinstance(value, bool) for value in correct.values()))
            ):
                raise ValueError("Route1 paired-row summary schema mismatch")
            expected_direct = row["population"] in {"B", "D"}
            if bool(row["direct_correct"]) != expected_direct:
                raise ValueError("Route1 population/direct label mismatch")
            clean_rows.append(row)
        c_rows = [row for row in clean_rows if row["population"] == "C"]
        full_counts = {"true_z": _count_condition(clean_rows, "true_z")}
        gate = report.get("gate_metrics")
        if not isinstance(gate, Mapping):
            raise ValueError("Route1 report summary schema mismatch")
        _assert_same_float(
            gate.get("true_z_full_accuracy"),
            float(full_counts["true_z"]["accuracy"]),
            "true_z_full_accuracy",
        )
        true_full_correct = _require_nonnegative_integer(
            gate.get("true_full_correct"), "true_full_correct"
        )
        true_full_total = _require_nonnegative_integer(
            gate.get("true_full_total"), "true_full_total"
        )
        if true_full_total != len(clean_rows):
            raise ValueError("true full denominator differs from paired rows")
        if true_full_correct != int(full_counts["true_z"]["numerator"]):
            raise ValueError("true_full_correct differs from paired rows")
        population_summary: dict[str, dict[str, Any]] = {}
        for quadrant in "BCDE":
            quadrant_rows = [row for row in clean_rows if row["population"] == quadrant]
            denominator = len(quadrant_rows)
            numerator = sum((bool(row["correct"]["true_z"]) for row in quadrant_rows))
            accuracy = numerator / denominator if denominator else None
            if gate.get(f"{quadrant.lower()}_denominator") != denominator:
                raise ValueError(f"{quadrant} denominator differs from paired rows")
            if accuracy is None:
                if gate.get(f"{quadrant.lower()}_accuracy") is not None:
                    raise ValueError("missing population accuracy must be null")
            else:
                _assert_same_float(
                    gate.get(f"{quadrant.lower()}_accuracy"),
                    accuracy,
                    f"{quadrant} true-z accuracy",
                )
            population_summary[quadrant] = {
                "numerator": int(numerator),
                "denominator": int(denominator),
                "accuracy": accuracy,
            }
        for quadrant, retention_field in (("B", "b_retention"), ("D", "d_retention")):
            expected_retention = population_summary[quadrant]["accuracy"]
            if expected_retention is None:
                if gate.get(retention_field) is not None:
                    raise ValueError("missing population retention must be null")
            else:
                _assert_same_float(
                    gate.get(retention_field), expected_retention, retention_field
                )
        from think_bridge.model.contract import route1_selector_metrics

        expected_metrics = route1_selector_metrics(
            clean_rows,
            generation_seed=int(report["generation"]["seed"]),
            run_seed=int(report["seed"]),
            include_controls=diagnostics,
        )
        for name, value in expected_metrics.items():
            if value is None or isinstance(value, list):
                if gate.get(name) != value:
                    raise ValueError(
                        f"Route1 summary differs from actual controls: {name}"
                    )
            else:
                _assert_same_float(gate.get(name), value, name)
        if gate.get("donor_count") != max(
            expected_metrics.get("wrong_donor_counts", []), default=0
        ):
            raise ValueError("Route1 control donor count mismatch")
        if gate.get("diagnostics_executed", True) != diagnostics:
            raise ValueError("Route1 report/gate diagnostic modes differ")
        generation = report.get("generation")
        if not isinstance(generation, Mapping):
            raise ValueError("Route1 report lacks generation identity")
        generation_seed = generation.get("seed")
        answer_max_tokens = generation.get("answer_max_tokens")
        if (
            isinstance(generation_seed, bool)
            or not isinstance(generation_seed, int)
            or isinstance(answer_max_tokens, bool)
            or (not isinstance(answer_max_tokens, int))
            or (answer_max_tokens <= 0)
        ):
            raise ValueError("Route1 report generation seed/cap is invalid")
        evaluation_domain = report.get("evaluation_domain")
        direct_baseline = report.get("direct_baseline")
        if (
            not isinstance(evaluation_domain, Mapping)
            or evaluation_domain.get("actual_sample_count") != len(clean_rows)
            or (
                not isinstance(
                    evaluation_domain.get("max_eval_samples"), (int, type(None))
                )
            )
            or isinstance(evaluation_domain.get("max_eval_samples"), bool)
            or (not isinstance(direct_baseline, Mapping))
        ):
            raise ValueError("Route1 report evaluation/direct identity is invalid")
        evaluation_subset_sha256 = _require_sha256(
            evaluation_domain.get("evaluation_subset_sha256"),
            "evaluation_subset_sha256",
        )
        direct_baseline_sha256 = _require_sha256(
            direct_baseline.get("identity_sha256"), "direct_baseline_identity_sha256"
        )
        actual_counts = [
            sum((k.startswith("wrong_z_") for k in row["correct"])) for row in c_rows
        ]
        expected_wrong_requests = sum(actual_counts)
        expected_request_counts = {
            "protocol": "bridge-route1-sparse-selector-request-count",
            "true_z": len(clean_rows),
            "direct": 0,
            "wrong_z": expected_wrong_requests,
            "c_prompt_count": len(c_rows),
            "requested_wrong_z_per_c_prompt": 8 if diagnostics else 0,
            "wrong_z_per_c_prompt": actual_counts[0]
            if actual_counts and len(set(actual_counts)) == 1
            else None,
            "wrong_z_per_c_prompt_counts": actual_counts,
            "total": len(clean_rows) + expected_wrong_requests,
        }
        if report.get("generation_request_counts") != expected_request_counts:
            raise ValueError("Route1 request counts differ from actual paired rows")
        measured_conditions = sorted({key for row in c_rows for key in row["correct"]})
        c_counts = {
            condition: _count_condition(
                [row for row in c_rows if condition in row["correct"]], condition
            )
            for condition in measured_conditions
        }
        metric_policy = _require_metric_policy(report, route=route)
        return {
            "metric_policy": metric_policy.as_dict(),
            "true_full": _without_accuracy(full_counts["true_z"]),
            "true_z_full_accuracy": _require_number(
                gate["true_z_full_accuracy"], "true_z_full_accuracy"
            ),
            "a_true": None
            if gate["a_true"] is None
            else _require_number(gate["a_true"], "a_true"),
            "a_direct": None
            if gate["a_direct"] is None
            else _require_number(gate["a_direct"], "a_direct"),
            "raw_g1": None
            if gate["raw_g1"] is None
            else _require_number(gate["raw_g1"], "raw_g1"),
            "diagnostics_executed": diagnostics,
            **{
                name: None if gate[name] is None else _require_number(gate[name], name)
                for name in ("a_wrong", "robust_g1", "robust_g1_ci_low")
                if diagnostics
            },
            "b_retention": None
            if gate["b_retention"] is None
            else _require_number(gate["b_retention"], "b_retention"),
            "d_retention": None
            if gate["d_retention"] is None
            else _require_number(gate["d_retention"], "d_retention"),
            "control_available_count": expected_metrics.get(
                "control_available_count", 0
            ),
            "control_unavailable_count": expected_metrics.get(
                "control_unavailable_count", 0
            ),
            "wrong_donor_counts": expected_metrics.get("wrong_donor_counts", []),
            "c_paired_count": len(c_rows),
            "wrong_pair_count": expected_wrong_requests,
            "eval_sample_count": len(clean_rows),
            "max_eval_samples": evaluation_domain.get("max_eval_samples"),
            "evaluation_subset_sha256": evaluation_subset_sha256,
            "direct_baseline_identity_sha256": direct_baseline_sha256,
            "generation_request_counts": expected_request_counts,
            "population": population_summary,
            "selector_condition_counts": {
                "true_z_full": _without_accuracy(full_counts["true_z"]),
                **{
                    f"c_{condition}": _without_accuracy(c_counts[condition])
                    for condition in measured_conditions
                },
            },
            "generation": {
                "seed": generation_seed,
                "answer_max_tokens": answer_max_tokens,
                "temperature": _require_number(
                    generation.get("temperature"), "generation temperature"
                ),
                "top_p": _require_number(generation.get("top_p"), "generation top_p"),
            },
            "paired_row_sha256": _require_sha256(
                report.get("paired_row_sha256"), "paired_row_sha256"
            ),
            "paired_output_sha256": _require_sha256(
                report.get("paired_output_sha256"), "paired_output_sha256"
            ),
        }

    def _selection_event(self, *, route: str, selection_path: Path) -> dict[str, Any]:
        route = _require_route(route)
        selection = Path(selection_path).resolve(strict=True)
        payload = _read_json(selection)
        candidates = payload.get("candidate_reports")
        metrics = payload.get("selection_metrics")
        if (
            not isinstance(candidates, list)
            or not candidates
            or (not isinstance(metrics, Mapping))
        ):
            raise ValueError("selection seal summary is incomplete")
        metric_policy_value = payload.get("metric_policy")
        if not isinstance(metric_policy_value, Mapping):
            raise ValueError("selection seal lacks its metric policy")
        policy = report_metric_policy(route=route, value=metric_policy_value)
        audit_fields = {
            "raw_metrics",
            "normalized_metrics",
            "aggregate_score",
            "tie_break",
        }
        missing_audit = sorted(audit_fields.difference(payload))
        if missing_audit:
            raise ValueError(f"selection score audit is incomplete: {missing_audit}")
        selected_hash_field = {"route1": "selected_r_sha256"}[route]
        selected_sha256 = _require_sha256(
            payload.get(selected_hash_field), selected_hash_field
        )
        candidate_summaries: list[dict[str, Any]] = []
        candidate_report_payloads: list[dict[str, Any]] = []
        selected_report: dict[str, Any] | None = None
        for raw in candidates:
            if not isinstance(raw, Mapping):
                raise ValueError("selection candidate ledger row is malformed")
            candidate_path = Path(str(raw.get("path", "")))
            if candidate_path.is_absolute() or ".." in candidate_path.parts:
                raise ValueError("selection candidate report escaped its run")
            report_path = (self.run_dir / candidate_path).resolve(strict=True)
            report_sha256 = _file_sha256(report_path)
            if report_sha256 != _require_sha256(
                raw.get("sha256"), "selection candidate report sha256"
            ):
                raise ValueError("selection candidate report changed")
            report = _read_json(report_path)
            if (
                report.get("step") != raw.get("step")
                or report.get("epoch") != raw.get("epoch")
                or (not isinstance(report.get("gate_metrics"), Mapping))
            ):
                raise ValueError("selection candidate identity mismatch")
            if _require_metric_policy(report, route=route) != policy:
                raise ValueError(
                    "candidate report metric policy differs from selection seal"
                )
            summary = {
                "step": int(raw["step"]),
                "epoch": int(raw["epoch"]),
                "report_path": str(report_path),
                "report_sha256": report_sha256,
                "checkpoint_path": str(report["checkpoint_path"]),
                "checkpoint_sha256": _require_sha256(
                    report["checkpoint_sha256"], "candidate checkpoint sha256"
                ),
                "metrics": dict(report["gate_metrics"]),
            }
            candidate_summaries.append(summary)
            candidate_report_payloads.append(report)
            if summary["checkpoint_sha256"] == selected_sha256:
                if selected_report is not None:
                    raise ValueError("selection hash identifies multiple candidates")
                selected_report = dict(summary)
        if selected_report is None or dict(metrics) != selected_report["metrics"]:
            raise ValueError("selection metrics do not identify the selected candidate")
        if payload.get("validation_report_sha256") != selected_report["report_sha256"]:
            raise ValueError("selection report hash differs from selected candidate")
        decision = select_best_candidate(
            candidate_report_payloads, evaluator=policy.evaluator, policy=policy
        )
        if (
            decision.step != int(selected_report["step"])
            or payload.get("raw_metrics") != decision.raw_metrics
            or payload.get("normalized_metrics") != decision.normalized_metrics
            or (payload.get("tie_break") != decision.tie_break)
            or (
                not math.isclose(
                    _require_number(
                        payload.get("aggregate_score"), "selection aggregate score"
                    ),
                    decision.aggregate_score,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
        ):
            raise ValueError(
                "selection score/tie-break audit differs from metric policy"
            )
        result = {
            **artifact_header(LIFECYCLE_EVENT),
            "event": "selection",
            "event_id": f"selection:{route}",
            "route": route,
            "owner": {"route1": "R"}[route],
            "candidate_count": len(candidate_summaries),
            "selected_step": int(selected_report["step"]),
            "selected_checkpoint_path": str(selected_report["checkpoint_path"]),
            "selected_checkpoint_sha256": selected_sha256,
            "selected_report_path": str(selected_report["report_path"]),
            "selected_report_sha256": str(selected_report["report_sha256"]),
            "selection_path": str(selection),
            "selection_sha256": _file_sha256(selection),
            "primary_metrics": ",".join(policy.primary_metrics),
            "metric_aggregation": policy.aggregation,
            "aggregate_score": decision.aggregate_score,
            **{f"raw::{name}": value for (name, value) in decision.raw_metrics.items()},
            **{
                f"normalized::{name}": value
                for (name, value) in decision.normalized_metrics.items()
            },
        }
        return result

    def append_selection(
        self, *, route: str, selection_path: Path
    ) -> dict[str, Any] | None:
        validated = self._selection_event(route=route, selection_path=selection_path)
        row = {
            "event": "selection",
            "stage": {"route1": "Route1"}[route],
            "candidate_count": int(validated["candidate_count"]),
            "selected_step": int(validated["selected_step"]),
            **{
                name: value
                for (name, value) in validated.items()
                if name in {"primary_metrics", "metric_aggregation", "aggregate_score"}
                or name.startswith("raw::")
                or name.startswith("normalized::")
            },
        }
        return row if self._append(row) else None

    def append_phase_end(
        self,
        *,
        route: str,
        owner: str,
        updates: int,
        status: str,
        selection_path: Path,
        last_checkpoint_path: Path,
        last_checkpoint_sha256: str,
    ) -> dict[str, Any] | None:
        selection = self._selection_event(route=route, selection_path=selection_path)
        if owner != selection["owner"]:
            raise ValueError("phase-end owner differs from selected route")
        row = {
            "event": "phase_end",
            "stage": {"route1": "Route1"}[route],
            "updates": int(updates),
            "status": str(status),
        }
        Path(last_checkpoint_path).resolve(strict=False)
        _require_sha256(last_checkpoint_sha256, "last checkpoint sha256")
        return row if self._append(row) else None

    def append_run_end(
        self,
        *,
        status: str,
        summary_path: Path,
        route_summaries: Mapping[str, Mapping[str, Any]],
    ) -> bool:
        routes = set(route_summaries)
        if routes not in ({"route1"}, _ROUTES):
            raise ValueError("run-end summary must contain route1")
        summary = Path(summary_path).resolve(strict=True)
        update_counts: dict[str, int] = {}
        for route in ("route1",):
            if route not in route_summaries:
                continue
            route_summary = route_summaries[route]
            updates = route_summary.get("updates")
            route_status = route_summary.get("status")
            if (
                isinstance(updates, bool)
                or not isinstance(updates, int)
                or updates <= 0
                or (not isinstance(route_status, str))
                or (not route_status)
            ):
                raise ValueError("run-end route summary is malformed")
            update_counts[route] = int(updates)
        row = {
            "event": "run_end",
            "stage": "Stage1",
            "status": str(status),
            "route1_updates": update_counts["route1"],
        }
        _file_sha256(summary)
        return self._append(row)


def reconcile_completed_evaluation_events(
    logger: LifecycleLogger,
    *,
    receipts: Sequence[Mapping[str, Any]],
    manifest_path: Path,
) -> int:
    """Append missing committed eval events without re-running evaluation."""

    if not isinstance(logger, LifecycleLogger):
        raise TypeError(
            "completed evaluation reconciliation requires a lifecycle logger"
        )
    appended = 0
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            raise TypeError("completed evaluation receipt must be a mapping")
        appended += int(
            logger.append_evaluation(
                receipt=receipt,
                manifest_path=Path(manifest_path),
            )
        )
    return appended
