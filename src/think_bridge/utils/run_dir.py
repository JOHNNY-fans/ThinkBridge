"""Atomic run-directory allocation and lifecycle locators."""

from __future__ import annotations

import datetime as _dt
from contextlib import contextmanager
import fcntl
import os
import re
from pathlib import Path
import threading
import uuid

_RUN_RE = re.compile(r"^(?:run-|v)(\d+)-\d{8}-\d{6}$")
_RUN_LOCATOR_ARTIFACT_TYPE = "think-bridge.stage.run-locator"
_RUN_LOCATOR_STAGES = frozenset({"reasoner-sft"})
_RUN_LOCATOR_STATES = frozenset({"allocated", "resumed", "completed"})


def _now_ts() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


@contextmanager
def _project_lifecycle_lock(project_dir: Path):
    """Serialize project metadata updates while mkdir remains the final claim."""

    project = Path(project_dir).expanduser().resolve(strict=False)
    project.mkdir(parents=True, exist_ok=True)
    lock_path = project / ".stage-run-lifecycle.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield project
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _run_index(path: Path) -> int:
    match = _RUN_RE.fullmatch(Path(path).name)
    if match is None:
        raise ValueError("run directory must contain an allocation index and timestamp")
    return int(match.group(1))


def require_project_run(
    project_dir: str | Path, run_dir: str | Path, *, require_exists: bool = True
) -> Path:
    """Validate one canonical immediate run child of a stage project."""

    project = Path(project_dir).expanduser().resolve(strict=False)
    run = Path(run_dir).expanduser().resolve(strict=False)
    if run.parent != project:
        raise ValueError("run directory does not belong to the requested stage project")
    _run_index(run)
    if require_exists and not run.is_dir():
        raise FileNotFoundError("stage run directory is missing")
    return run


def allocate_run_dir(project_dir: Path) -> Path:
    """Atomically reserve the next run-{N}-{ts} child under project_dir.

    The project lock makes local contenders deterministic, while the final
    ``mkdir(exist_ok=False)`` remains the atomic filesystem claim.  Matching
    files as well as directories consume their allocation index so a malformed
    stale entry cannot make allocation spin on the same candidate.
    """

    with _project_lifecycle_lock(Path(project_dir)) as project:
        max_n = -1
        for child in project.iterdir():
            match = _RUN_RE.fullmatch(child.name)
            if match is not None:
                max_n = max(max_n, int(match.group(1)))
        candidate = project / f"run-{max_n + 1}-{_now_ts()}"
        candidate.mkdir(exist_ok=False)
        (candidate / "runs").mkdir(exist_ok=False)
        return candidate


def _atomic_write_text(path: Path, text: str) -> None:
    target = Path(path).expanduser().resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.tmp-{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex}"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_locator_path(locator: Path, *, project_dir: Path) -> Path:
    project = Path(project_dir).expanduser().resolve(strict=False)
    exact = Path(locator).expanduser().resolve(strict=False)
    allowed = {project, (project / "invocations").resolve(strict=False)}
    if exact.parent not in allowed:
        raise ValueError("stage run locator escaped the project metadata namespace")
    return exact


def _run_locator_text(*, stage: str, state: str, run_dir: Path) -> str:
    if stage not in _RUN_LOCATOR_STAGES:
        raise ValueError("stage run locator has an invalid stage")
    if state not in _RUN_LOCATOR_STATES:
        raise ValueError("stage run locator has an invalid state")
    return (
        f"artifact_type={_RUN_LOCATOR_ARTIFACT_TYPE}\n"
        "schema_version=1\n"
        f"stage={stage}\n"
        f"state={state}\n"
        f"path={run_dir.name}\n"
    )


def write_stage_run_locator(
    locator: str | Path,
    *,
    project_dir: str | Path,
    run_dir: str | Path,
    stage: str,
    state: str,
) -> Path:
    """Atomically bind one operational locator to an exact project run."""

    project = Path(project_dir).expanduser().resolve(strict=False)
    run = require_project_run(project, run_dir)
    target = _validate_locator_path(Path(locator), project_dir=project)
    _atomic_write_text(target, _run_locator_text(stage=stage, state=state, run_dir=run))
    return target


def _read_stage_run_locator(locator: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw in Path(locator).read_text(encoding="utf-8").splitlines():
        if not raw or "=" not in raw:
            raise ValueError("stage run locator is malformed")
        key, value = raw.split("=", 1)
        if not key or not value or key in fields:
            raise ValueError("stage run locator is malformed")
        fields[key] = value
    expected = {"artifact_type", "schema_version", "stage", "state", "path"}
    if (
        set(fields) != expected
        or fields["artifact_type"] != _RUN_LOCATOR_ARTIFACT_TYPE
        or fields["schema_version"] != "1"
    ):
        raise ValueError("stage run locator schema is invalid")
    return fields


def resolve_stage_run_locator(
    locator: str | Path,
    *,
    project_dir: str | Path,
    stage: str,
    require_completed: bool = True,
) -> Path:
    """Resolve and validate one exact invocation or completed-run locator."""

    project = Path(project_dir).expanduser().resolve(strict=False)
    target = _validate_locator_path(Path(locator), project_dir=project)
    fields = _read_stage_run_locator(target)
    if fields["stage"] != stage:
        raise ValueError("stage run locator belongs to a different stage")
    if fields["state"] not in _RUN_LOCATOR_STATES:
        raise ValueError("stage run locator state is invalid")
    if require_completed and fields["state"] != "completed":
        raise ValueError("stage run locator is not completed")
    relative = Path(fields["path"])
    if (
        relative.is_absolute()
        or len(relative.parts) != 1
        or relative.name != fields["path"]
    ):
        raise ValueError("stage run locator path is not canonical")
    return require_project_run(project, project / relative)


def publish_stage_run_locator(
    project_dir: str | Path, run_dir: str | Path, *, stage: str
) -> Path:
    """Publish the most recently allocated completed run as ``latest-run.txt``."""

    project = Path(project_dir).expanduser().resolve(strict=False)
    run = require_project_run(project, run_dir)
    latest = project / "latest-run.txt"
    with _project_lifecycle_lock(project):
        if latest.is_file():
            observed = resolve_stage_run_locator(
                latest,
                project_dir=project,
                stage=stage,
                require_completed=True,
            )
            if _run_index(observed) > _run_index(run):
                return latest
        _atomic_write_text(
            latest,
            _run_locator_text(stage=stage, state="completed", run_dir=run),
        )
    return latest


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
