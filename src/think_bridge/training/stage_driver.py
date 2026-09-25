"""Single-parse public drivers for the split Bridge Stage1 commands."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
from think_bridge.training.evaluation_dependencies import check_evaluation_imports
from think_bridge.training.process_cleanup import (
    run_managed_process,
    termination_signals,
)

from think_bridge.model.checkpoint_policy import (
    checkpoint_seal_from_marker,
    parse_checkpoint_path,
    read_checkpoint_pointer,
)
from think_bridge.model.contract import write_atomic_json
from think_bridge.utils.run_dir import (
    allocate_run_dir,
    publish_stage_run_locator,
    require_project_run,
    write_stage_run_locator,
)
from think_bridge.training.checkpoint_manager import checkpoint_run_directory
from think_bridge.model.artifact_schema import (
    STAGE_ARTIFACT_LOCATOR,
    STAGE_RESOLVED_CONFIG,
    artifact_header,
    require_artifact_header,
)
from think_bridge.training.staged_training import (
    STAGE1_VLLM_REQUEST_TIMEOUT_SECONDS,
    build_reasoner_parser,
    build_worker_launch,
    resolve_reasoner_config,
    stage_artifact_paths,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"stage JSON must be an object: {path}")
    return value


def _reasoner_run_dir(checkpoint: Path) -> Path:
    exact = Path(checkpoint).expanduser().resolve(strict=True)
    for candidate in (exact, *exact.parents):
        if (candidate / "runtime_sidecars.json").is_file():
            return candidate
    raise ValueError(
        "explicit reasoner checkpoint has no owning runtime_sidecars.json ancestor"
    )


def _activate_stage_run(resolved: Mapping[str, Any]) -> dict[str, Any]:
    """Turn one public stage project into an exact fresh or resumed run."""

    activated = dict(resolved)
    runtime = dict(resolved.get("runtime", {}))
    project = Path(str(resolved["output_dir"])).expanduser().resolve(strict=False)
    resume_raw = resolved.get("resume_from_checkpoint")
    if resume_raw is None or not str(resume_raw).strip():
        run = allocate_run_dir(project)
        resume_existing = False
        locator_state = "allocated"
    else:
        checkpoint = Path(str(resume_raw)).expanduser().resolve(strict=True)
        run = checkpoint_run_directory(checkpoint).resolve(strict=True)
        require_project_run(project, run)
        resume_existing = True
        locator_state = "resumed"
    (run / "runs").mkdir(exist_ok=True)
    activated["output_dir"] = str(run)
    runtime.update(
        project_dir=str(project),
        resume_existing_run=resume_existing,
    )
    activated["runtime"] = runtime
    locator_raw = runtime.get("run_locator")
    if locator_raw is not None and str(locator_raw).strip():
        write_stage_run_locator(
            Path(str(locator_raw)),
            project_dir=project,
            run_dir=run,
            stage=str(activated["stage"]),
            state=locator_state,
        )
    return activated


def _publish_completed_stage_run(resolved: Mapping[str, Any]) -> None:
    """Publish operational locators only after selection-owned best is valid."""
    stage = str(resolved["stage"])
    expected_route = {"reasoner-sft": "route1"}[stage]
    run = Path(str(resolved["output_dir"])).resolve(strict=True)
    runtime = resolved.get("runtime", {})
    project = Path(str(runtime["project_dir"])).resolve(strict=True)
    require_project_run(project, run)
    paths = stage_artifact_paths(resolved)
    selection = Path(str(paths["selection"]))
    if not selection.is_file():
        raise FileNotFoundError(
            "completed stage publication requires its run-internal selection"
        )
    pointer = Path(str(paths["best_checkpoint_pointer"]))
    checkpoint = read_checkpoint_pointer(pointer, run_dir=run)
    (observed_route, _) = parse_checkpoint_path(checkpoint)
    if observed_route != expected_route:
        raise ValueError("completed stage best pointer belongs to the wrong route")
    locator_raw = runtime.get("run_locator")
    if locator_raw is not None and str(locator_raw).strip():
        write_stage_run_locator(
            Path(str(locator_raw)),
            project_dir=project,
            run_dir=run,
            stage=stage,
            state="completed",
        )
    publish_stage_run_locator(project, run, stage=stage)


def _existing_training_frontier(
    output: Path,
    *,
    paths: Mapping[str, str],
    artifact_path: Path,
) -> tuple[str, ...]:
    """Return only mutable trainer/selection state, never reusable data caches."""

    observed: list[str] = []
    for event_log in (output / "logging.jsonl", output / "step_audit.jsonl"):
        if event_log.is_file() and event_log.stat().st_size > 0:
            observed.append(str(event_log))
    trainer_state = output / "trainer_state.json"
    if trainer_state.exists():
        observed.append(str(trainer_state))
    for pointer in sorted(output.glob("*checkpoint*.txt")):
        observed.append(str(pointer))
    for checkpoint in sorted(output.glob("checkpoint-*")):
        if checkpoint.exists():
            observed.append(str(checkpoint))
    report_root = output / "reports"
    if report_root.exists() and any(report_root.glob("*/eval/*.json")):
        observed.append(str(report_root))
    selection = Path(str(paths["selection"]))
    if selection.exists():
        observed.append(str(selection))
    if artifact_path.is_file():
        locator = _read_json(artifact_path)
        if any(
            name in locator
            for name in (
                "selected_checkpoint",
                "selected_checkpoint_sha256",
                "selection_sha256",
            )
        ):
            observed.append(str(artifact_path))
    return tuple(dict.fromkeys(observed))


def _validate_stage_start(
    resolved: Mapping[str, Any],
    *,
    output: Path,
    paths: Mapping[str, str],
    resolved_path: Path,
    artifact_path: Path,
) -> bool:
    """Fail fast for accidental fixed-output reruns; validate exact resumes."""
    resume_raw = resolved.get("resume_from_checkpoint")
    if resume_raw is None or not str(resume_raw).strip():
        frontier = _existing_training_frontier(
            output, paths=paths, artifact_path=artifact_path
        )
        if frontier:
            preview = ", ".join(frontier[:4])
            raise ValueError(
                f"fresh split stage found an existing training frontier in output_dir; pass its exact --resume_from_checkpoint or use a new output_dir: {preview}"
            )
        return False
    checkpoint = Path(str(resume_raw)).expanduser().resolve(strict=True)
    run_dir = output.resolve(strict=True)
    if checkpoint_run_directory(checkpoint) != run_dir:
        raise ValueError(
            "resume checkpoint does not belong to the exact owning output_dir"
        )
    expected_route = {"reasoner-sft": "route1"}[str(resolved["stage"])]
    (route, _) = parse_checkpoint_path(checkpoint)
    if route != expected_route:
        raise ValueError("resume checkpoint belongs to the wrong split stage")
    if not resolved_path.is_file() or not artifact_path.is_file():
        raise ValueError(
            "resume requires the owning run's resolved config and artifact locator"
        )
    checkpoint_seal_from_marker(checkpoint)
    metadata = _read_json(checkpoint / "checkpoint.json")
    if metadata.get("exact_resume_identity_sha256") != resolved.get(
        "scientific_identity_sha256"
    ):
        raise ValueError("resume checkpoint exact stage identity differs")
    return True


def _write_stage_contract(resolved: Mapping[str, Any]) -> tuple[Path, Path]:
    require_artifact_header(
        resolved, STAGE_RESOLVED_CONFIG, label="resolved stage config"
    )
    output = Path(str(resolved["output_dir"]))
    output.mkdir(parents=True, exist_ok=True)
    paths = stage_artifact_paths(resolved)
    resolved_path = Path(paths["resolved_config"])
    artifact_path = Path(paths["artifacts"])
    locator: dict[str, Any] = {
        **artifact_header(STAGE_ARTIFACT_LOCATOR),
        "stage": resolved["stage"],
        "scientific_identity_sha256": resolved["scientific_identity_sha256"],
        "paths": paths,
    }
    validation_role = (
        "eval_behavior" if resolved["stage"] in {"reasoner-sft"} else "eval_dataset"
    )
    validation_input = resolved.get("inputs", {}).get(validation_role)
    if isinstance(validation_input, Mapping):
        locator["validation_provenance"] = {
            "role": validation_role,
            "path": str(validation_input.get("path", "")),
        }
    identity = str(resolved["scientific_identity_sha256"])
    if resolved_path.is_file():
        existing_resolved = _read_json(resolved_path)
        if (
            existing_resolved.get("artifact_type") != STAGE_RESOLVED_CONFIG
            or existing_resolved.get("schema_version") != 1
            or existing_resolved.get("stage") != resolved["stage"]
            or (existing_resolved.get("scientific_identity_sha256") != identity)
        ):
            raise ValueError(
                "output_dir already owns a different resolved stage identity"
            )
    if artifact_path.is_file():
        existing_locator = _read_json(artifact_path)
        if (
            existing_locator.get("artifact_type") != STAGE_ARTIFACT_LOCATOR
            or existing_locator.get("schema_version") != 1
            or existing_locator.get("stage") != resolved["stage"]
            or (existing_locator.get("scientific_identity_sha256") != identity)
        ):
            raise ValueError(
                "output_dir already owns a different artifact locator identity"
            )
    exact_resume = _validate_stage_start(
        resolved,
        output=output,
        paths=paths,
        resolved_path=resolved_path,
        artifact_path=artifact_path,
    )
    if exact_resume:
        locator = {**_read_json(artifact_path), **locator}
    write_atomic_json(resolved_path, dict(resolved), replace_mismatch=True)
    write_atomic_json(artifact_path, locator, replace_mismatch=True)
    return (resolved_path, artifact_path)


def _worker_environment(
    resolved: Mapping[str, Any], *, trainer_only: bool
) -> dict[str, str]:
    environment = os.environ.copy()
    if trainer_only:
        devices = resolved["runtime"]["trainer_devices"]
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in devices)
    environment["NPROC_PER_NODE"] = str(resolved["training"]["world_size"])
    return environment


def _run_worker(
    action: str,
    *,
    resolved: Mapping[str, Any],
    resolved_path: Path,
    artifact_path: Path,
    distributed: bool,
) -> None:
    world = int(resolved["training"]["world_size"])
    port = int(resolved["runtime"]["master_port"])
    command = build_worker_launch(
        action=action,
        resolved_config=resolved_path,
        artifacts=artifact_path,
        world_size=(world if distributed else 1),
        master_port=port,
    )
    completed = run_managed_process(
        command,
        env=_worker_environment(resolved, trainer_only=distributed),
    )
    if completed.returncode:
        raise SystemExit(completed.returncode)


def _start_vllm(resolved: Mapping[str, Any]) -> Any | None:
    if resolved["runtime"].get("generation_backend") != "vllm":
        return None
    from think_bridge.training.vllm_runtime import (
        LaunchSpec,
        VLLMServiceSupervisor,
        new_private_service_instance_token,
        nvidia_smi_gpu_compute_processes,
        private_service_ready,
        spawn_private_service,
    )

    runtime = resolved["runtime"]
    devices = tuple(str(value) for value in runtime["vllm_devices"])
    request_timeout_seconds = STAGE1_VLLM_REQUEST_TIMEOUT_SECONDS
    launch = LaunchSpec(
        command=(
            sys.executable,
            "-m",
            "think_bridge.training.vllm_service",
            "serve",
            "--model",
            str(resolved["model"]),
            "--seed",
            str(resolved["training"]["seed"]),
            "--host",
            str(runtime["vllm_host"]),
            "--port",
            str(runtime["vllm_port"]),
            # URL-safe nonces may start with '-'; bind the value to its option.
            f"--instance-token={new_private_service_instance_token()}",
            "--data-parallel-size",
            str(len(devices)),
            "--tensor-parallel-size",
            "1",
            "--physical-chunk-size",
            str(runtime["vllm_physical_batch_size"]),
            "--request-timeout-seconds",
            str(request_timeout_seconds),
            "--worker-extension-cls",
            "think_bridge.training.vllm_worker.BridgeRoute1WorkerExtension",
            "--gpu-memory-utilization",
            str(runtime["vllm_gpu_memory_utilization"]),
            "--enforce-eager",
            "false",
            "--max-num-seqs",
            str(max(64, int(runtime["vllm_physical_batch_size"]))),
            "--max-in-flight",
            str(runtime["vllm_max_in_flight"]),
            "--max-queued-requests",
            str(int(runtime["vllm_service_max_queued_requests"])),
            "--queue-timeout-seconds",
            str(request_timeout_seconds),
        ),
        environment={"CUDA_VISIBLE_DEVICES": ",".join(devices)},
    )
    supervisor = VLLMServiceSupervisor(
        spawn=spawn_private_service,
        ready=private_service_ready,
        gpu_processes=nvidia_smi_gpu_compute_processes,
    )
    supervisor.start(launch, startup_timeout_seconds=1800.0)
    return supervisor


def _report_vllm_service_status(supervisor: Any) -> None:
    """Emit bounded process/health state without touching structured train logs."""

    try:
        status = dict(supervisor.status())
    except BaseException as exc:
        status = {
            "active": False,
            "exit_code": None,
            "healthy": False,
            "status_error": f"{type(exc).__name__}: {str(exc)[:512]}",
        }
    payload = {
        "event": "bridge-stage1-vllm-service-status",
        "active": bool(status.get("active", False)),
        "exit_code": status.get("exit_code"),
        "healthy": bool(status.get("healthy", False)),
    }
    if "status_error" in status:
        payload["status_error"] = str(status["status_error"])[:512]
    print(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        file=sys.stderr,
        flush=True,
    )


@termination_signals()
def run_reasoner_driver(argv: Sequence[str] | None = None) -> int:
    parsed = build_reasoner_parser().parse_args(argv)
    check_evaluation_imports()
    resolved = _activate_stage_run(resolve_reasoner_config(parsed))
    resolved_path, artifact_path = _write_stage_contract(resolved)
    _run_worker(
        "prepare-targets",
        resolved=resolved,
        resolved_path=resolved_path,
        artifact_path=artifact_path,
        distributed=False,
    )
    supervisor = _start_vllm(resolved)
    try:
        _run_worker(
            "train",
            resolved=resolved,
            resolved_path=resolved_path,
            artifact_path=artifact_path,
            distributed=True,
        )
    except BaseException:
        if supervisor is not None:
            _report_vllm_service_status(supervisor)
        raise
    finally:
        if supervisor is not None:
            supervisor.close(shutdown_timeout_seconds=60.0)
    _run_worker(
        "select",
        resolved=resolved,
        resolved_path=resolved_path,
        artifact_path=artifact_path,
        distributed=False,
    )
    _publish_completed_stage_run(resolved)
    return 0
