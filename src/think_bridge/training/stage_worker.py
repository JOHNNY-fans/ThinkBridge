"""Private resolved-config worker for split Bridge stage drivers.

The public CLI is parsed exactly once by the driver.  This worker accepts only
the sealed resolved config and artifact locator, then calls route-owned leaves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
from think_bridge.training.evaluation_dependencies import check_evaluation_imports

from think_bridge.model.checkpoint_policy import (
    checkpoint_artifact_sha256,
    write_checkpoint_pointer,
)
from think_bridge.model.training_config import load_training_config
from think_bridge.model.contract import (
    file_sha256,
    require_sha256,
    write_atomic_json,
)
from think_bridge.model.artifact_schema import (
    STAGE_ARTIFACT_LOCATOR,
    STAGE_RESOLVED_CONFIG,
    require_artifact_header,
)
from think_bridge.training.staged_training import (
    STAGE1_VLLM_REQUEST_TIMEOUT_SECONDS,
)
from think_bridge.training.validation_reports import (
    canonical_validation_report_paths,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m think_bridge.training.stage_worker", allow_abbrev=False
    )
    parser.add_argument("action", choices=("prepare-targets", "train", "select"))
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"stage JSON must be an object: {path}")
    return value


def _load_contract(
    config_path: Path, artifact_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = _read_json(config_path)
    artifacts = _read_json(artifact_path)
    require_artifact_header(
        resolved, STAGE_RESOLVED_CONFIG, label="resolved stage config"
    )
    require_artifact_header(
        artifacts, STAGE_ARTIFACT_LOCATOR, label="stage artifact locator"
    )
    identity = require_sha256(
        str(resolved.get("scientific_identity_sha256")),
        "scientific_identity_sha256",
    )
    if (
        artifacts.get("scientific_identity_sha256") != identity
        or artifacts.get("stage") != resolved.get("stage")
        or not isinstance(artifacts.get("paths"), dict)
    ):
        raise ValueError("stage config/artifact locator binding mismatch")
    return resolved, artifacts


def _gpu_ids(resolved: Mapping[str, Any]) -> tuple[int, ...]:
    values = tuple(int(value) for value in resolved["runtime"]["trainer_devices"])
    world = int(resolved["training"]["world_size"])
    if len(values) != world or len(set(values)) != world:
        raise ValueError("resolved trainer device topology is malformed")
    return values


def _runtime_config(resolved: Mapping[str, Any]) -> Any:
    training = resolved["training"]
    runtime = resolved["runtime"]
    stage = str(resolved["stage"])
    reasoner = True
    route1_like = reasoner
    world = int(training["world_size"])
    devices = _gpu_ids(resolved)
    overrides: dict[str, Any] = {
        "seed": int(training["seed"]),
        "generation_seed": int(training["generation_seed"]),
        "route1_generation_temperature": float(
            training.get("route1_generation_temperature", 0.0)
        ),
        "save_steps": int(training["save_steps"]),
        "eval_steps": int(training["eval_steps"]),
        "logging_steps": int(training["logging_steps"]),
        "save_total_limit": int(training["save_total_limit"]),
        "route1_epochs": int(training["num_train_epochs"]),
        "world_size": world,
        "trainer_gpu_ids": devices,
        "route1_world_size": world,
        "route1_trainer_gpu_ids": devices,
        "weight_decay": float(training["weight_decay"]),
        "route1_local_samples": int(training["per_device_train_batch_size"]),
        "route1_local_chunk_size": int(runtime["physical_batch_size"]),
        "route1_wrong_control_chunk_size": int(runtime["wrong_control_chunk_size"]),
        "route1_gradient_accumulation_steps": int(
            training["gradient_accumulation_steps"]
        ),
        "route1_gradient_checkpointing": bool(runtime["gradient_checkpointing"]),
        "route1_eval_local_row_batch": int(
            (resolved.get("answer_evaluation") or {}).get(
                "answer_group_size", training["per_device_eval_batch_size"]
            )
        ),
    }
    from think_bridge.model.feedback_precision import (
        feedback_policy,
        saved_output_normalization,
        saved_reasoner_loop_steps,
        saved_reasoner_layers,
    )
    from think_bridge.model.feedback_precision import saved_reasoner_input

    overrides.update(saved_reasoner_input(resolved))
    overrides["emitter_depth"] = saved_reasoner_layers(resolved)
    overrides["reasoner_loop_steps"] = saved_reasoner_loop_steps(resolved)
    policy = feedback_policy(resolved)
    overrides.update(
        {
            name: policy[name]
            for name in ("reasoner_compute_dtype", "reasoner_eval_group_size")
        }
    )
    geometry = resolved.get("reasoner_geometry")
    if not isinstance(geometry, Mapping):
        raise ValueError(f"{stage} resolved config lacks reasoner geometry")
    overrides.update(
        latent_slots=int(geometry["latent_slots"]),
        latent_steps=int(geometry["latent_steps"]),
        latents_per_step=int(geometry["latents_per_step"]),
        real_frozen_f_appends=int(geometry["real_frozen_f_appends"]),
        geometry_schema_version=str(geometry["geometry_schema_version"]),
        reasoner_zero_init=bool(geometry.get("zero_init", False)),
        reasoner_output_normalization=saved_output_normalization(resolved),
        reasoner_dropout_p=float(geometry.get("dropout_p", 0.0)),
        reasoner_dropout_views=int(geometry.get("dropout_views", 1)),
    )
    objective = resolved["objective"]
    route1_weights = (
        float(objective["course_weight"]),
        float(objective["match_weight"]),
        float(objective["specific_weight"]),
    )
    if any((value < 0.0 for value in route1_weights)) or sum(route1_weights) <= 0.0:
        raise ValueError(f"{stage} requires at least one positive Route1 loss weight")
    overrides.update(
        lr_r=float(training["learning_rate"]),
        max_grad_norm_r=float(training["max_grad_norm"]),
        warmup_updates_r=int(training["warmup_steps"]),
        route1_course_epochs=float(objective.get("course_epoch_fraction", 0.0)),
        route1_course_steps=objective.get("course_steps", 0),
        route1_course_weight=float(objective["course_weight"]),
        route1_match_weight=float(objective["match_weight"]),
        route1_specific_weight=float(objective["specific_weight"]),
        route1_specificity_donors_per_owner=int(
            objective["specificity_donors_per_owner"]
        ),
        route1_specificity_tau=float(objective["specificity_tau"]),
        route1_specificity_loss=str(
            objective.get("specificity_loss", "same-prompt-capped-soft-infonce")
        ),
        route1_specificity_margin=float(objective.get("specificity_margin", 0.1)),
        route1_specificity_temperature=float(
            objective.get("specificity_temperature", 1.0)
        ),
        route1_specificity_negative_kl_cap=objective.get("specificity_negative_kl_cap"),
        route1_specificity_wrong_gradient=objective.get(
            "specificity_wrong_gradient", "live"
        ),
        route1_specificity_include_direct=bool(
            objective.get("specificity_include_direct", False)
        ),
        route1_distillation_populations=str(
            objective.get("route1_distillation_populations", "BC")
        ),
        route1_generation_backend=str(runtime["generation_backend"]),
        route1_vllm_gpu_ids=tuple(
            (int(value) for value in runtime.get("vllm_devices", ()))
        ),
        route1_vllm_data_parallel_size=max(1, len(runtime.get("vllm_devices", ()))),
        route1_vllm_tensor_parallel_size=1,
        route1_vllm_host=str(runtime["vllm_host"]),
        route1_vllm_port=int(runtime["vllm_port"]),
        route1_vllm_physical_chunk_size=int(runtime["vllm_physical_batch_size"]),
        route1_vllm_max_in_flight=int(runtime["vllm_max_in_flight"]),
        route1_vllm_max_queued_requests=int(
            runtime["vllm_service_max_queued_requests"]
        ),
        route1_vllm_max_pending_microsteps=int(runtime["vllm_max_pending_microsteps"]),
        route1_vllm_gpu_memory_utilization=float(
            runtime["vllm_gpu_memory_utilization"]
        ),
    )
    return load_training_config(
        resolved["method_config"],
        model_name_or_path=str(resolved["model"]),
        tokenizer_name_or_path=str(resolved["tokenizer"]),
        runtime_overrides=overrides,
        stage_kind=stage,
    )


def _target_path(
    config: Any, resolved: Mapping[str, Any], artifacts: Mapping[str, Any], split: str
) -> Path:
    from think_bridge.training.prepare import resolve_target_artifact

    return resolve_target_artifact(
        Path(artifacts["paths"]["target_index"]),
        split,
        model_family=config.model_family,
    )


def _prepare_targets(
    config: Any, resolved: Mapping[str, Any], artifacts: Mapping[str, Any]
) -> int:
    from think_bridge.training.prepare import prepare_targets

    paths = artifacts["paths"]
    inputs = resolved["inputs"]
    objective = resolved.get("target_preparation", resolved["objective"])
    argument_values = dict(
        input_references=inputs,
        train_source=Path(inputs["train_dataset"]["path"]),
        train_behavior=None
        if inputs["train_behavior"].get("path") is None
        else Path(inputs["train_behavior"]["path"]),
        train_direct_answer_source=None
        if inputs["train_direct_answer_source"].get("path") is None
        else Path(inputs["train_direct_answer_source"]["path"]),
        route1_population=objective.get("population", "staged"),
        answer_only_target_source=objective.get(
            "answer_target_source", "paired-stage0-self-answer"
        ),
        validation_source=Path(inputs["eval_dataset"]["path"]),
        validation_behavior=Path(inputs["eval_behavior"]["path"]),
        output_index=Path(paths["target_index"]),
        manifest=Path(paths["parent_manifest"]),
        donors=Path(paths["donor_manifest"]),
        run_dir=Path(resolved["output_dir"]),
        world_size=int(resolved["training"]["world_size"]),
        route1_local_samples=int(resolved["training"]["per_device_train_batch_size"]),
        route1_gradient_accumulation_steps=int(
            resolved["training"]["gradient_accumulation_steps"]
        ),
        local_files_only=bool(resolved["runtime"]["local_files_only"]),
        no_progress=bool(resolved["runtime"]["no_progress"]),
        metric_policy=resolved["selection"],
        overwrite_cache=bool(resolved["runtime"].get("overwrite_cache", False)),
    )
    return prepare_targets(config, SimpleNamespace(**argument_values))


def _train(
    config: Any, resolved: Mapping[str, Any], artifacts: Mapping[str, Any]
) -> int:
    from think_bridge.training.input_identity import assert_target_input_binding

    assert_target_input_binding(
        Path(artifacts["paths"]["target_index"]), resolved["inputs"]
    )
    from think_bridge.training.train import run_training

    stage = str(resolved["stage"])
    route = "route1"
    route1_like = True
    paths = artifacts["paths"]
    manifest = Path(paths["parent_manifest"])
    max_steps = int(resolved["training"]["max_steps"])
    argument_values = dict(
        route=route,
        seed=int(resolved["training"]["seed"]),
        run_label=f"{resolved['model_family']}-{resolved['stage']}",
        exact_resume_identity_sha256=resolved["scientific_identity_sha256"],
        resume_training_plan=resolved.get("resume_training_plan"),
        manifest=manifest,
        target_index=Path(paths["target_index"]),
        resume=None
        if resolved["resume_from_checkpoint"] is None
        else Path(resolved["resume_from_checkpoint"]),
        selected_r=None,
        validation_report=None,
        checkpoint_dir=Path(resolved["output_dir"]),
        runtime_sidecar_run=Path(resolved["output_dir"]),
        report_dir=Path(paths["report_dir"]) / route / "train",
        logging_jsonl=Path(paths["structured_log"]),
        validation_records=_target_path(config, resolved, artifacts, "validation"),
        donors=Path(paths["donor_manifest"]),
        selection_output=Path(paths["selection"]),
        metric_policy=resolved["selection"],
        answer_max_tokens=int(config.answer_capacity),
        route1_eval_local_row_batch=int(config.route1_eval_local_row_batch),
        route1_generation_backend=str(
            resolved["runtime"].get("generation_backend", "torch")
        ),
        route1_vllm_host=str(resolved["runtime"].get("vllm_host", "127.0.0.1")),
        route1_vllm_port=int(resolved["runtime"].get("vllm_port", 29601)),
        route1_vllm_request_timeout_seconds=STAGE1_VLLM_REQUEST_TIMEOUT_SECONDS,
        route1_vllm_shutdown_timeout_seconds=60.0,
        route1_vllm_backpressure_timeout_seconds=60.0,
        route1_vllm_watchdog_interval_seconds=60.0,
        route1_vllm_physical_chunk_size=int(
            resolved["runtime"].get("vllm_physical_batch_size", 1)
        ),
        route1_vllm_max_in_flight=int(resolved["runtime"].get("vllm_max_in_flight", 1)),
        route1_vllm_max_pending_microsteps=int(
            resolved["runtime"].get("vllm_max_pending_microsteps", 1)
        ),
        route1_local_samples=int(resolved["training"]["per_device_train_batch_size"]),
        route1_gradient_accumulation_steps=int(
            resolved["training"]["gradient_accumulation_steps"]
        ),
        route1_population=str(
            resolved.get("objective", {}).get("population", "staged")
        ),
        route1_course_weight=resolved.get("objective", {}).get("course_weight"),
        route1_match_weight=resolved.get("objective", {}).get("match_weight"),
        route1_specific_weight=resolved.get("objective", {}).get("specific_weight"),
        route1_specificity_tau=resolved.get("objective", {}).get("specificity_tau"),
        route1_specificity_include_direct=bool(
            resolved.get("objective", {}).get("specificity_include_direct", False)
        ),
        route1_distillation_populations=str(
            resolved.get("objective", {}).get("route1_distillation_populations", "BC")
        ),
        route1_eval_null_mode=str(
            resolved.get("evaluation", {}).get("null_mode", "direct")
        ),
        max_eval_samples=resolved.get("evaluation", {}).get("max_eval_samples"),
        route1_max_optimizer_updates=max_steps if max_steps > 0 else None,
        route1_diagnostic_only=False,
        route1_component_gradient_audit=False,
        test_interrupt_after_updates=None,
        full_state_audit=bool(resolved["runtime"]["full_state_audit"]),
        no_progress=bool(resolved["runtime"]["no_progress"]),
    )
    argument_values["generation_seed"] = int(resolved["training"]["generation_seed"])
    arguments = SimpleNamespace(**argument_values)
    return run_training(config, arguments)


def _selection_reports(output: Path, route: str) -> tuple[Path, ...]:
    reports = canonical_validation_report_paths(
        output / "reports" / route / "eval",
        route=route,
    )
    if not reports:
        raise ValueError(f"{route} selection has no validation reports")
    return reports


def _select(
    resolved: Mapping[str, Any], artifacts: dict[str, Any], artifact_path: Path
) -> int:
    from think_bridge.training.eval import run_selection
    from think_bridge.training.checkpoint_manager import resolve_run_artifact_locator

    route = "route1"
    route1_like = True
    paths = artifacts["paths"]
    output = Path(resolved["output_dir"])
    reports = _selection_reports(output, route)
    selection = Path(paths["selection"])
    selection_arguments = {
        "command": f"select-{route}",
        "seed": int(resolved["training"]["seed"]),
        "manifest": Path(paths["parent_manifest"]),
        "selected_r": None,
        "reports": list(reports),
        "output": selection,
        "metric_policy": resolved["selection"],
    }
    selection_arguments["generation_seed"] = int(
        resolved["training"]["generation_seed"]
    )
    arguments = SimpleNamespace(**selection_arguments)
    result = run_selection(arguments)
    seal = _read_json(selection)
    selected_field = {"route1": "selected_r_sha256"}[route]
    selected_sha = require_sha256(str(seal[selected_field]), f"selected_{route}_sha256")
    selected_checkpoint = None
    for report_path in reports:
        report = _read_json(report_path)
        if report.get("checkpoint_sha256") == selected_sha:
            selected_checkpoint = resolve_run_artifact_locator(
                output,
                report["checkpoint_path"],
                label="selected validation checkpoint path",
            )
            break
    if selected_checkpoint is None:
        raise ValueError("selection SHA does not identify one validation checkpoint")
    if checkpoint_artifact_sha256(selected_checkpoint) != selected_sha:
        raise ValueError("selected checkpoint changed after validation")
    write_checkpoint_pointer(
        Path(paths["best_checkpoint_pointer"]),
        run_dir=output,
        checkpoint=selected_checkpoint,
    )
    artifacts.update(
        selected_checkpoint=str(selected_checkpoint.resolve()),
        selected_checkpoint_sha256=selected_sha,
        selection_sha256=file_sha256(selection),
    )
    write_atomic_json(artifact_path, artifacts, replace_mismatch=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.action != "select":
        check_evaluation_imports()
    resolved, artifacts = _load_contract(arguments.resolved_config, arguments.artifacts)
    if arguments.action == "select":
        return _select(resolved, artifacts, arguments.artifacts)
    config = _runtime_config(resolved)
    if arguments.action == "prepare-targets":
        if resolved["stage"] not in {"reasoner-sft"}:
            raise ValueError("only reasoner-sft prepares targets")
        return _prepare_targets(config, resolved, artifacts)
    if arguments.action == "train":
        return _train(config, resolved, artifacts)
    raise RuntimeError(f"unreachable worker action: {arguments.action}")


if __name__ == "__main__":
    raise SystemExit(main())
