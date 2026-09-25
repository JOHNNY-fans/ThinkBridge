"""Atomic checkpoint/evaluation transaction manager for ThinkBridge phases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Union

from think_bridge.model.artifact_schema import (
    COMPLETED_EVALUATION,
    TRAINER_STATE,
    artifact_header,
)

from think_bridge.model.checkpoint_policy import (
    SealedCheckpoint,
    checkpoint_owner_run_directory,
    checkpoint_seal_from_marker,
    checkpoint_path as route_checkpoint_path,
    parse_checkpoint_path,
    prune_checkpoint_namespace,
    read_checkpoint_pointer_target,
    validate_checkpoint_directory,
    validate_checkpoint_seal,
    write_checkpoint_pointer,
)
from think_bridge.model.contract import (
    ROUTE1_VALIDATION_REPORT_SCHEMA_VERSION,
    require_sha256,
    write_atomic_json,
)


MaterializeCheckpoint = Callable[[Path], Union[SealedCheckpoint, Path, None]]

_COMPLETED_EVALUATION_FIELDS = {
    "artifact_type",
    "schema_version",
    "route",
    "step",
    "epoch",
    "report_path",
    "report_sha256",
    "checkpoint_path",
    "checkpoint_sha256",
    "validation_randomness",
    "ordinary_checkpoint",
}


def resolve_run_artifact_locator(
    run_dir: str | Path,
    raw: Any,
    *,
    label: str,
) -> Path:
    """Resolve a canonical run-relative locator or an absolute path."""

    raw_text = str(raw)
    if not raw_text:
        raise ValueError(f"{label} is empty")
    locator = Path(raw_text)
    run = Path(run_dir).resolve(strict=False)
    if locator.is_absolute():
        resolved = locator.resolve(strict=False)
    else:
        if locator == Path(".") or ".." in locator.parts:
            raise ValueError(f"{label} escaped the run directory")
        resolved = (run / locator).resolve(strict=False)
    try:
        relative = resolved.relative_to(run)
    except ValueError as exc:
        raise ValueError(f"{label} escaped the run directory") from exc
    if relative == Path("."):
        raise ValueError(f"{label} must identify a run artifact")
    if not locator.is_absolute() and locator.as_posix() != relative.as_posix():
        raise ValueError(f"{label} is not a canonical run-relative locator")
    return resolved


def run_relative_artifact_locator(
    run_dir: str | Path,
    path: str | Path,
    *,
    label: str,
) -> str:
    """Encode one contained artifact without binding the run's mount path."""

    run = Path(run_dir).resolve(strict=False)
    resolved = Path(path).resolve(strict=False)
    try:
        relative = resolved.relative_to(run)
    except ValueError as exc:
        raise ValueError(f"{label} escaped the run directory") from exc
    if relative == Path("."):
        raise ValueError(f"{label} must identify a run artifact")
    return relative.as_posix()


def checkpoint_run_directory(checkpoint: str | Path) -> Path:
    """Recover and validate the owning run from a canonical checkpoint path."""

    resolved = Path(checkpoint).resolve(strict=False)
    route, step = parse_checkpoint_path(resolved)
    if len(resolved.parents) < 3:
        raise ValueError("checkpoint path lacks its owning run directory")
    run = checkpoint_owner_run_directory(resolved)
    expected = route_checkpoint_path(run, route, step).resolve(strict=False)
    if resolved != expected:
        raise ValueError("checkpoint path is not canonical under its owning run")
    return run


def validate_recorded_checkpoint_location(
    run_dir: str | Path, checkpoint: str | Path, *, route: str, step: int
) -> Path:
    """Validate a completed ledger locator even after retention removes weights.

    This does not authorize loading a missing checkpoint. Its caller must first
    verify the completed report/ledger identity. Existing checkpoints still need
    their metadata; an incomplete directory is not a pruned checkpoint.
    """
    run = Path(run_dir).resolve(strict=False)
    path = Path(checkpoint)
    expected = route_checkpoint_path(run, route, step)
    if path.is_symlink():
        raise ValueError("recorded checkpoint must not be a symlink")
    resolved = path.resolve(strict=False)
    if resolved != expected:
        raise ValueError("recorded checkpoint location differs from route/step")
    if resolved.exists() and parse_checkpoint_path(resolved) != (route, step):
        raise ValueError("recorded checkpoint metadata differs from route/step")
    return run


def checkpoint_run_relative_locator(checkpoint: str | Path) -> str:
    run = checkpoint_run_directory(checkpoint)
    return run_relative_artifact_locator(
        run,
        checkpoint,
        label="validation report checkpoint path",
    )


class CheckpointManager:
    """Publish complete artifacts, pending-eval state, and protected pointers."""

    def __init__(
        self, run_dir: str | Path, *, route: str, save_total_limit: int = 10
    ) -> None:
        if route not in {"route1"}:
            raise ValueError("checkpoint route must be route1")
        if (
            isinstance(save_total_limit, bool)
            or not isinstance(save_total_limit, int)
            or save_total_limit < 1
        ):
            raise ValueError("save_total_limit must be a positive integer")
        self.run_dir = Path(run_dir)
        self.route = route
        self.save_total_limit = int(save_total_limit)
        self.state_path = self.run_dir / "trainer_state.json"
        self._sealed_checkpoints: dict[Path, SealedCheckpoint] = {}

    def checkpoint_path(self, step: int) -> Path:
        return route_checkpoint_path(self.run_dir, self.route, step)

    def _ledger_routes(self) -> tuple[str, ...]:
        return ("route1",)

    def _empty_completed_evaluations(self) -> dict[str, list[dict[str, Any]]]:
        return {route: [] for route in self._ledger_routes()}

    def _resolved_run_path(self, raw: Any, *, label: str) -> Path:
        return resolve_run_artifact_locator(self.run_dir, raw, label=label)

    def _relative_run_path(self, path: str | Path, *, label: str) -> str:
        return run_relative_artifact_locator(self.run_dir, path, label=label)

    def _canonical_report_path(self, *, route: str, step: int) -> Path:
        return (
            self.run_dir
            / "reports"
            / route
            / "eval"
            / f"bridge-{route}-validation-step-{int(step)}.json"
        ).resolve(strict=False)

    def _normalize_completed_evaluations(
        self, value: Any
    ) -> dict[str, list[dict[str, Any]]]:
        routes = self._ledger_routes()
        if not isinstance(value, Mapping) or set(value) != set(routes):
            raise ValueError("completed evaluation ledger route schema mismatch")
        normalized: dict[str, list[dict[str, Any]]] = {}
        for route in routes:
            rows = value[route]
            if not isinstance(rows, list):
                raise ValueError("completed evaluation ledger must be a list per route")
            clean: list[dict[str, Any]] = []
            prior_step = 0
            for raw in rows:
                if (
                    not isinstance(raw, Mapping)
                    or set(raw) != _COMPLETED_EVALUATION_FIELDS
                ):
                    raise ValueError("completed evaluation ledger row schema mismatch")
                step = raw.get("step")
                epoch = raw.get("epoch")
                ordinary = raw.get("ordinary_checkpoint")
                randomness = raw.get("validation_randomness")
                if (
                    raw.get("artifact_type") != COMPLETED_EVALUATION
                    or raw.get("schema_version") != 1
                    or raw.get("route") != route
                    or isinstance(step, bool)
                    or (not isinstance(step, int))
                    or (step <= prior_step)
                    or isinstance(epoch, bool)
                    or (not isinstance(epoch, int))
                    or (not isinstance(ordinary, bool))
                    or (not isinstance(randomness, Mapping))
                ):
                    raise ValueError("completed evaluation ledger identity mismatch")
                report_path = self._resolved_run_path(
                    raw.get("report_path"), label="completed evaluation report path"
                )
                checkpoint_path = self._resolved_run_path(
                    raw.get("checkpoint_path"),
                    label="completed evaluation checkpoint path",
                )
                if report_path != self._canonical_report_path(route=route, step=step):
                    raise ValueError(
                        "completed evaluation report path is not canonical"
                    )
                expected_checkpoint = route_checkpoint_path(
                    self.run_dir, route, step
                ).resolve(strict=False)
                if checkpoint_path != expected_checkpoint:
                    raise ValueError(
                        "completed evaluation checkpoint path is not canonical"
                    )
                clean.append(
                    {
                        **artifact_header(COMPLETED_EVALUATION),
                        "route": route,
                        "step": int(step),
                        "epoch": int(epoch),
                        "report_path": self._relative_run_path(
                            report_path, label="completed evaluation report path"
                        ),
                        "report_sha256": require_sha256(
                            raw.get("report_sha256"), "evaluation report_sha256"
                        ),
                        "checkpoint_path": self._relative_run_path(
                            checkpoint_path,
                            label="completed evaluation checkpoint path",
                        ),
                        "checkpoint_sha256": require_sha256(
                            raw.get("checkpoint_sha256"), "evaluation checkpoint_sha256"
                        ),
                        "validation_randomness": dict(randomness),
                        "ordinary_checkpoint": ordinary,
                    }
                )
                prior_step = int(step)
            normalized[route] = clean
        return normalized

    def _fresh_state(self) -> dict[str, Any]:
        return {
            **artifact_header(TRAINER_STATE),
            "route": self.route,
            "committed_step": 0,
            "pending_evaluation": None,
            "completed_evaluations": self._empty_completed_evaluations(),
        }

    def _normalize_pending_evaluation(self, pending: Any) -> Any:
        if pending is None or not isinstance(pending, Mapping):
            return pending
        clean = dict(pending)
        checkpoint = self._resolved_run_path(
            clean.get("checkpoint_path"), label="pending evaluation checkpoint path"
        )
        clean["checkpoint_path"] = self._relative_run_path(
            checkpoint, label="pending evaluation checkpoint path"
        )
        return clean

    def _validate_state_frontier(
        self, payload: Mapping[str, Any], completed: Mapping[str, list[dict[str, Any]]]
    ) -> None:
        route = payload.get("route")
        committed = payload.get("committed_step")
        pending = payload.get("pending_evaluation")
        if route not in set(self._ledger_routes()):
            raise ValueError("trainer state route is invalid")
        if (
            isinstance(committed, bool)
            or not isinstance(committed, int)
            or committed < 0
        ):
            raise ValueError("trainer state committed frontier is invalid")
        rows = completed[route]
        if rows and int(rows[-1]["step"]) > int(committed):
            raise ValueError("completed evaluation ledger exceeds committed frontier")
        if pending is None:
            return
        expected_fields = {
            "step",
            "epoch",
            "checkpoint_path",
            "checkpoint_sha256",
            "ordinary_checkpoint",
        }
        if not isinstance(pending, Mapping) or set(pending) != expected_fields:
            raise ValueError("pending evaluation state schema mismatch")
        step = pending.get("step")
        epoch = pending.get("epoch")
        ordinary = pending.get("ordinary_checkpoint")
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or step <= 0
            or (step != committed)
            or isinstance(epoch, bool)
            or (not isinstance(epoch, int))
            or (not isinstance(ordinary, bool))
        ):
            raise ValueError("pending evaluation frontier identity mismatch")
        checkpoint = self._resolved_run_path(
            pending.get("checkpoint_path"), label="pending evaluation checkpoint path"
        )
        if checkpoint != route_checkpoint_path(self.run_dir, route, step).resolve(
            strict=False
        ):
            raise ValueError("pending evaluation checkpoint path is not canonical")
        require_sha256(pending.get("checkpoint_sha256"), "pending checkpoint_sha256")

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self._fresh_state()
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("artifact_type") != TRAINER_STATE
            or payload.get("schema_version") != 1
            or ("completed_evaluations" not in payload)
        ):
            raise ValueError("trainer state identity differs from the active phase")
        completed = self._normalize_completed_evaluations(
            payload["completed_evaluations"]
        )
        payload["pending_evaluation"] = self._normalize_pending_evaluation(
            payload.get("pending_evaluation")
        )
        self._validate_state_frontier(payload, completed)
        if payload.get("route") != self.route:
            if payload.get("pending_evaluation") is not None:
                raise ValueError("cannot enter the next route with pending evaluation")
            return {
                **artifact_header(TRAINER_STATE),
                "route": self.route,
                "previous_route": payload.get("route"),
                "previous_route_committed_step": payload.get("committed_step"),
                "committed_step": 0,
                "pending_evaluation": None,
                "completed_evaluations": completed,
            }
        payload["completed_evaluations"] = completed
        return payload

    def _write_state(self, state: Mapping[str, Any]) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        payload = dict(state)
        if (
            payload.get("artifact_type") != TRAINER_STATE
            or payload.get("schema_version") != 1
        ):
            raise ValueError("trainer state write uses a foreign schema")
        payload["completed_evaluations"] = self._normalize_completed_evaluations(
            payload.get("completed_evaluations")
        )
        payload["pending_evaluation"] = self._normalize_pending_evaluation(
            payload.get("pending_evaluation")
        )
        self._validate_state_frontier(payload, payload["completed_evaluations"])
        write_atomic_json(self.state_path, payload, replace_mismatch=True)

    def _published_frontier_step(self, state: Mapping[str, Any]) -> int:
        committed = state.get("committed_step", 0)
        if (
            isinstance(committed, bool)
            or not isinstance(committed, int)
            or committed < 0
        ):
            raise ValueError("trainer state committed_step is invalid")
        pointer = self.run_dir / f"{self.route}_latest_checkpoint.txt"
        if not pointer.is_file():
            return int(committed)
        latest = read_checkpoint_pointer_target(pointer, run_dir=self.run_dir)
        (_, latest_step) = parse_checkpoint_path(latest)
        return max(int(committed), int(latest_step))

    def _reject_rollback(self, *, step: int, state: Mapping[str, Any]) -> None:
        frontier = self._published_frontier_step(state)
        if int(step) < frontier:
            raise ValueError(
                f"checkpoint transaction step {step} would roll back frontier {frontier}"
            )

    def _remember_seal(
        self,
        seal: SealedCheckpoint,
        *,
        checkpoint: Path | None = None,
        validate_marker: bool,
    ) -> SealedCheckpoint:
        observed = validate_checkpoint_seal(seal) if validate_marker else seal
        expected = self.checkpoint_path(observed.step).resolve(strict=True)
        if (
            observed.route != self.route
            or observed.path != expected
            or (
                checkpoint is not None
                and Path(checkpoint).resolve(strict=True) != observed.path
            )
        ):
            raise ValueError("checkpoint seal differs from manager route/path")
        self._sealed_checkpoints[observed.path] = observed
        return observed

    def _seal_for_checkpoint(
        self, checkpoint: Path, *, supplied: SealedCheckpoint | None = None
    ) -> SealedCheckpoint:
        resolved = Path(checkpoint).resolve(strict=True)
        cached = self._sealed_checkpoints.get(resolved)
        if supplied is not None:
            if cached is not None and cached == supplied:
                return cached
            return self._remember_seal(
                supplied, checkpoint=checkpoint, validate_marker=True
            )
        if cached is not None:
            return cached
        seal = validate_checkpoint_directory(resolved)
        return self._remember_seal(seal, checkpoint=checkpoint, validate_marker=False)

    def _pointer(self, name: str, seal: SealedCheckpoint, *, step: int) -> None:
        (_, parsed_step) = parse_checkpoint_path(seal.path)
        if parsed_step != int(step):
            raise ValueError("checkpoint pointer step differs from its directory")
        write_checkpoint_pointer(
            self.run_dir / name, run_dir=self.run_dir, checkpoint=seal
        )

    def _materialize(
        self,
        *,
        step: int,
        materialize: MaterializeCheckpoint,
        sealed_checkpoint: SealedCheckpoint | None = None,
    ) -> tuple[Path, SealedCheckpoint]:
        checkpoint = self.checkpoint_path(step)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        materialized: SealedCheckpoint | Path | None = None
        if not checkpoint.exists():
            materialized = materialize(checkpoint)
        if isinstance(materialized, Path):
            if materialized.resolve(strict=True) != checkpoint.resolve(strict=True):
                raise ValueError("materializer returned a different checkpoint path")
            materialized = None
        if materialized is not None and (
            not isinstance(materialized, SealedCheckpoint)
        ):
            raise TypeError("checkpoint materializer returned an unsupported value")
        if (
            sealed_checkpoint is not None
            and materialized is not None
            and (sealed_checkpoint != materialized)
        ):
            raise ValueError(
                "materializer and caller supplied different checkpoint seals"
            )
        seal = self._seal_for_checkpoint(
            checkpoint, supplied=sealed_checkpoint or materialized
        )
        return (checkpoint, seal)

    def prune_committed(self) -> tuple[Path, ...]:
        protected: list[Path] = []
        for name in (
            f"{self.route}_best_checkpoint.txt",
            f"{self.route}_latest_checkpoint.txt",
        ):
            pointer = self.run_dir / name
            if pointer.is_file():
                protected.append(
                    read_checkpoint_pointer_target(pointer, run_dir=self.run_dir)
                )
        return prune_checkpoint_namespace(
            self.run_dir,
            route=self.route,
            protected=tuple(protected),
            limit=self.save_total_limit,
        )

    def publish_after_update(
        self,
        *,
        step: int,
        finite: bool,
        materialize: MaterializeCheckpoint,
        sealed_checkpoint: SealedCheckpoint | None = None,
    ) -> Path:
        if not finite:
            raise FloatingPointError(
                "non-finite update cannot publish a selectable checkpoint"
            )
        state = self._load_state()
        if state.get("pending_evaluation") is not None:
            raise RuntimeError("pending evaluation must finish before checkpointing")
        self._reject_rollback(step=step, state=state)
        (checkpoint, seal) = self._materialize(
            step=step, materialize=materialize, sealed_checkpoint=sealed_checkpoint
        )
        self._pointer(f"{self.route}_latest_checkpoint.txt", seal, step=step)
        self._pointer("latest_checkpoint.txt", seal, step=step)
        state["committed_step"] = int(step)
        self._write_state(state)
        self.prune_committed()
        return checkpoint

    def begin_evaluation(
        self,
        *,
        step: int,
        epoch: int,
        checkpoint: Path | None,
        ordinary_checkpoint: bool,
        materialize: MaterializeCheckpoint,
        sealed_checkpoint: SealedCheckpoint | None = None,
    ) -> Path:
        if not isinstance(ordinary_checkpoint, bool):
            raise ValueError("evaluation checkpoint kind must be explicit")
        state = self._load_state()
        pending = state.get("pending_evaluation")
        if pending is not None:
            if int(pending["step"]) != int(step):
                raise RuntimeError("a different evaluation transaction is pending")
            recovered = self._resolved_run_path(
                pending["checkpoint_path"], label="pending evaluation checkpoint path"
            )
            recovered_seal = self._seal_for_checkpoint(
                recovered, supplied=sealed_checkpoint
            )
            if recovered_seal.artifact_sha256 != pending["checkpoint_sha256"]:
                raise RuntimeError(
                    "pending evaluation checkpoint is incomplete or changed"
                )
            return recovered
        self._reject_rollback(step=step, state=state)
        if checkpoint is None:
            (resolved, resolved_seal) = self._materialize(
                step=step, materialize=materialize, sealed_checkpoint=sealed_checkpoint
            )
        else:
            resolved = Path(checkpoint)
            resolved_seal = self._seal_for_checkpoint(
                resolved, supplied=sealed_checkpoint
            )
        (resolved_route, resolved_step) = parse_checkpoint_path(resolved)
        if resolved_route != self.route or resolved_step != int(step):
            raise ValueError("evaluation checkpoint route/step mismatch")
        state["committed_step"] = int(step)
        state["pending_evaluation"] = {
            "step": int(step),
            "epoch": int(epoch),
            "checkpoint_path": self._relative_run_path(
                resolved, label="pending evaluation checkpoint path"
            ),
            "checkpoint_sha256": resolved_seal.artifact_sha256,
            "ordinary_checkpoint": ordinary_checkpoint,
        }
        self._write_state(state)
        return resolved

    def pending_evaluation(self) -> dict[str, Any] | None:
        pending = self._load_state().get("pending_evaluation")
        if pending is None:
            return None
        external = dict(pending)
        external["checkpoint_path"] = str(
            self._resolved_run_path(
                pending["checkpoint_path"], label="pending evaluation checkpoint path"
            )
        )
        return external

    def _evaluation_receipt(
        self, *, pending: Mapping[str, Any], report_path: Path, completed: bool = False
    ) -> dict[str, Any]:
        step = pending.get("step")
        epoch = pending.get("epoch")
        ordinary = pending.get("ordinary_checkpoint")
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or step <= 0
            or isinstance(epoch, bool)
            or (not isinstance(epoch, int))
            or (not isinstance(ordinary, bool))
        ):
            raise ValueError("pending evaluation identity is malformed")
        report = Path(report_path)
        if report.is_symlink() or not report.is_file() or report.stat().st_size <= 0:
            raise RuntimeError("evaluation report is not completely published")
        report = report.resolve(strict=True)
        if report != self._canonical_report_path(route=self.route, step=step):
            raise ValueError("evaluation report path is not canonical for its step")
        raw = report.read_bytes()
        report_sha256 = hashlib.sha256(raw).hexdigest()
        payload = json.loads(raw)
        expected_schema = {"route1": ROUTE1_VALIDATION_REPORT_SCHEMA_VERSION}[
            self.route
        ]
        checkpoint = self._resolved_run_path(
            pending.get("checkpoint_path"), label="pending evaluation checkpoint path"
        )
        if completed:
            validate_recorded_checkpoint_location(
                self.run_dir, checkpoint, route=self.route, step=step
            )
            (checkpoint_route, checkpoint_step) = (self.route, step)
        else:
            (checkpoint_route, checkpoint_step) = parse_checkpoint_path(checkpoint)
        reported_checkpoint = self._resolved_run_path(
            payload.get("checkpoint_path") if isinstance(payload, Mapping) else None,
            label="validation report checkpoint path",
        )
        reported_step = payload.get("step") if isinstance(payload, Mapping) else None
        reported_epoch = payload.get("epoch") if isinstance(payload, Mapping) else None
        randomness = (
            payload.get("validation_randomness")
            if isinstance(payload, Mapping)
            else None
        )
        checkpoint_sha256 = require_sha256(
            pending.get("checkpoint_sha256"), "pending checkpoint_sha256"
        )
        if (
            not isinstance(payload, Mapping)
            or payload.get("artifact_type") != expected_schema
            or payload.get("schema_version") != 1
            or (checkpoint_route != self.route)
            or (checkpoint_step != int(step))
            or isinstance(reported_step, bool)
            or (reported_step != int(step))
            or isinstance(reported_epoch, bool)
            or (reported_epoch != int(epoch))
            or (reported_checkpoint != checkpoint)
            or (payload.get("checkpoint_sha256") != checkpoint_sha256)
            or (not isinstance(randomness, Mapping))
        ):
            raise ValueError(
                "validation report step/epoch/checkpoint/randomness identity mismatch"
            )
        return {
            **artifact_header(COMPLETED_EVALUATION),
            "route": self.route,
            "step": int(step),
            "epoch": int(epoch),
            "report_path": self._relative_run_path(
                report, label="completed evaluation report path"
            ),
            "report_sha256": report_sha256,
            "checkpoint_path": self._relative_run_path(
                checkpoint, label="completed evaluation checkpoint path"
            ),
            "checkpoint_sha256": checkpoint_sha256,
            "validation_randomness": dict(randomness),
            "ordinary_checkpoint": ordinary,
        }

    def _validate_completed_evaluation_rows(
        self, state: Mapping[str, Any], *, report_paths: tuple[Path, ...] | None
    ) -> tuple[dict[str, Any], ...]:
        completed = self._normalize_completed_evaluations(
            state.get("completed_evaluations")
        )
        rows = tuple((dict(row) for row in completed[self.route]))
        observed_paths: list[Path] = []
        for row in rows:
            report = self._resolved_run_path(
                row["report_path"], label="completed evaluation report path"
            )
            if report.is_symlink() or not report.is_file():
                raise ValueError("completed evaluation report is missing or unsafe")
            raw = report.read_bytes()
            if hashlib.sha256(raw).hexdigest() != row["report_sha256"]:
                raise ValueError("completed evaluation report hash mismatch")
            payload = json.loads(raw)
            replay_pending = {
                "step": row["step"],
                "epoch": row["epoch"],
                "checkpoint_path": row["checkpoint_path"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "ordinary_checkpoint": row["ordinary_checkpoint"],
            }
            observed = self._evaluation_receipt(
                pending=replay_pending, report_path=report, completed=True
            )
            if observed != row:
                raise ValueError("completed evaluation ledger/report identity mismatch")
            checkpoint = self._resolved_run_path(
                row["checkpoint_path"], label="completed evaluation checkpoint path"
            )
            if checkpoint.exists():
                seal = checkpoint_seal_from_marker(checkpoint)
                if seal.artifact_sha256 != row["checkpoint_sha256"]:
                    raise ValueError(
                        "completed evaluation checkpoint marker hash mismatch"
                    )
            observed_paths.append(report.resolve(strict=True))
        if report_paths is not None:
            supplied = [Path(path).resolve(strict=True) for path in report_paths]
            if len(supplied) != len(set(supplied)) or set(supplied) != set(
                observed_paths
            ):
                raise ValueError(
                    "completed evaluation report set differs from its ledger"
                )
        return rows

    def validate_completed_evaluations(
        self, *, report_paths: tuple[Path, ...] | None = None
    ) -> tuple[dict[str, Any], ...]:
        """Validate the append-only report ledger before resume or selection."""
        rows = self._validate_completed_evaluation_rows(
            self._load_state(), report_paths=report_paths
        )
        external: list[dict[str, Any]] = []
        for row in rows:
            value = dict(row)
            value["report_path"] = str(
                self._resolved_run_path(
                    row["report_path"], label="completed evaluation report path"
                )
            )
            value["checkpoint_path"] = str(
                self._resolved_run_path(
                    row["checkpoint_path"], label="completed evaluation checkpoint path"
                )
            )
            external.append(value)
        return tuple(external)

    def repair_evaluation_frontier(
        self,
        *,
        step: int,
        epoch: int,
        checkpoint: Path,
        ordinary_checkpoint: bool,
        sealed_checkpoint: SealedCheckpoint | None = None,
    ) -> Path:
        """Idempotently publish a missed save before repairing its evaluation."""
        if ordinary_checkpoint:
            self.publish_after_update(
                step=step,
                finite=True,
                materialize=lambda _path: None,
                sealed_checkpoint=sealed_checkpoint,
            )
        return self.begin_evaluation(
            step=step,
            epoch=epoch,
            checkpoint=checkpoint,
            ordinary_checkpoint=ordinary_checkpoint,
            materialize=lambda _path: None,
            sealed_checkpoint=sealed_checkpoint,
        )

    def may_begin_update(self, step: int) -> bool:
        state = self._load_state()
        return (
            state.get("pending_evaluation") is None
            and int(step) == int(state.get("committed_step", 0)) + 1
        )

    def commit_evaluation(
        self,
        *,
        step: int,
        report_path: Path,
        is_best: bool,
        sealed_checkpoint: SealedCheckpoint | None = None,
    ) -> None:
        state = self._load_state()
        pending = state.get("pending_evaluation")
        if pending is None or int(pending["step"]) != int(step):
            raise RuntimeError("evaluation commit has no matching pending transaction")
        self._validate_completed_evaluation_rows(state, report_paths=None)
        report = Path(report_path)
        receipt = self._evaluation_receipt(pending=pending, report_path=report)
        checkpoint = self._resolved_run_path(
            pending["checkpoint_path"], label="pending evaluation checkpoint path"
        )
        seal = self._seal_for_checkpoint(checkpoint, supplied=sealed_checkpoint)
        if seal.artifact_sha256 != pending["checkpoint_sha256"]:
            raise RuntimeError("evaluation checkpoint changed before report commit")
        self._pointer(f"{self.route}_latest_checkpoint.txt", seal, step=step)
        self._pointer("latest_checkpoint.txt", seal, step=step)
        if is_best:
            self._pointer(f"{self.route}_best_checkpoint.txt", seal, step=step)
        completed = self._normalize_completed_evaluations(
            state.get("completed_evaluations")
        )
        route_rows = completed[self.route]
        if route_rows and int(receipt["step"]) <= int(route_rows[-1]["step"]):
            raise ValueError("completed evaluation ledger cannot move backwards")
        route_rows.append(receipt)
        state["completed_evaluations"] = completed
        state["pending_evaluation"] = None
        self._write_state(state)
        self.prune_committed()
