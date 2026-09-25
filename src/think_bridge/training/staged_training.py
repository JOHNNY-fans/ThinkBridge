"""Public contracts for Bridge reasoner training.

The public commands parse ordinary trainer vocabulary exactly once.  They
persist a resolved config plus a small artifact locator; private workers read
those files instead of receiving a second copy of every training argument.
"""

from __future__ import annotations


import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping

from think_bridge.training.input_identity import (
    file_reference,
    input_contract,
    dataset_identity,
)
from think_bridge.arguments.train_args import validate_stage1_deepspeed_config
from think_bridge.model.contract import (
    ROUTE1_OCCURRENCE_SAMPLER_SCHEMA,
    canonical_json_sha256,
    require_sha256,
    bridge_stage1_model_root,
)
from think_bridge.training.objective_window_plans import (
    ROUTE1_ACTIVE_DOMAIN_SCHEMA,
    ROUTE1_PHYSICAL_COST_POLICY_SCHEMA,
)
from think_bridge.model.artifact_schema import STAGE_RESOLVED_CONFIG, artifact_header
from think_bridge.model.training_config import load_training_config


PERFORMANCE_OBSERVABILITY_SCHEMA = "bridge-training-performance-observability"
STAGE1_VLLM_REQUEST_TIMEOUT_SECONDS = 1800.0
_REASONER_OBJECTIVE = "answer-ce-bc-forward-kl-same-prompt-specificity"
_SAFE_PROFILE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class BatchGeometry:
    """One real per-process microbatch and its optimizer-window geometry."""

    per_device_train_batch_size: int
    world_size: int
    gradient_accumulation_steps: int

    def validate(self) -> None:
        values = (
            self.per_device_train_batch_size,
            self.world_size,
            self.gradient_accumulation_steps,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in values
        ):
            raise ValueError("batch, world size, and GAS must be positive integers")

    @property
    def effective_global_batch(self) -> int:
        self.validate()
        return (
            self.per_device_train_batch_size
            * self.world_size
            * self.gradient_accumulation_steps
        )


def _strict_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _add_common_training_arguments(
    parser: argparse.ArgumentParser,
    *,
    gradient_checkpointing_help: str | None = None,
) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoner_compute_dtype", choices=("bfloat16",))
    parser.add_argument("--reasoner_eval_group_size", type=int)
    parser.add_argument("--tokenizer")
    parser.add_argument("--model_family", choices=("qwen3-0.6b", "qwen3-4b"))
    parser.add_argument("--train_dataset", type=Path, required=True)
    parser.add_argument("--eval_dataset", type=Path, required=True)
    parser.add_argument("--per_device_train_batch_size", type=int)
    parser.add_argument("--per_device_eval_batch_size", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=10)
    parser.add_argument("--metric_for_best_model")
    parser.add_argument(
        "--metric_aggregation",
        choices=("mean",),
        default="mean",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--run_locator",
        type=Path,
        help=(
            "Optional invocation-unique operational locator written with the "
            "allocated run; excluded from scientific identity."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=Path)
    parser.add_argument("--physical_batch_size", type=int)
    parser.add_argument(
        "--gradient_checkpointing",
        type=_strict_bool,
        help=gradient_checkpointing_help,
    )
    parser.add_argument("--full_state_audit", type=_strict_bool, default=False)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help=(
            "Optional deterministic validation limit applied after the full "
            "validation domain is checked; unset evaluates all rows."
        ),
    )


def _course_contract(arguments: Any, template: Mapping[str, Any]) -> dict[str, Any]:
    values = [
        getattr(arguments, "course_steps", None),
        getattr(arguments, "course_epochs", None),
        template.get("route1_course_steps", 0),
        template.get("route1_course_epochs", 0),
    ]
    if any(value not in (None, 0) for value in values):
        raise ValueError("Route1 uses answer-only CE with curriculum disabled")
    resume = getattr(arguments, "resume_from_checkpoint", None)
    if resume:
        objective = json.loads((Path(resume) / "resolved_config.json").read_text())[
            "objective"
        ]
        if (
            objective.get("course_steps") != 0
            or objective.get("course_epoch_fraction", 0) != 0
            or objective.get("ce_mask") != "answer-eos-only"
        ):
            raise ValueError("resume requires the same answer-only CE contract")
    return {"course_steps": 0}


def _ce_condition_contract(
    arguments: Any, template: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        **_course_contract(arguments, template),
        "ce_mask": "answer-eos-only",
        "course_ce_mask": "answer-eos-only",
        "course_removal": "disabled",
        "course_student_input": "prompt-live-z-boundary-answer",
    }


def _specificity_gradient_contract(
    arguments: Any, template: Mapping[str, Any]
) -> dict[str, Any]:
    mode = getattr(arguments, "specificity_wrong_gradient", None)
    if mode is None and getattr(arguments, "resume_from_checkpoint", None):
        saved = json.loads(
            (
                Path(arguments.resume_from_checkpoint) / "resolved_config.json"
            ).read_text()
        )
        mode = saved["objective"].get("specificity_wrong_gradient", "live")
    if mode is None:
        mode = template.get("route1_specificity_wrong_gradient", "live")
    if mode != "live":
        raise ValueError("specificity_wrong_gradient must be live")
    return {
        "specificity_wrong_gradient": mode,
        "specificity_objective": f"{mode}-same-prompt-capped-soft-infonce-teacher-forward-kl",
    }


def _specificity_objective_contract(
    arguments: Any, template: Mapping[str, Any]
) -> dict[str, Any]:
    if getattr(arguments, "specificity_margin", None) is not None:
        raise ValueError("Use specificity_negative_kl_cap")
    cap = getattr(arguments, "specificity_negative_kl_cap", None)
    temperature = getattr(arguments, "specificity_temperature", None)
    if getattr(arguments, "resume_from_checkpoint", None):
        saved = json.loads(
            (
                Path(arguments.resume_from_checkpoint) / "resolved_config.json"
            ).read_text()
        )
        if (
            saved["objective"].get("specificity_loss")
            != "same-prompt-capped-soft-infonce"
        ):
            raise ValueError("Resume requires the same specificity objective")
        if cap is None:
            cap = saved["objective"]["specificity_negative_kl_cap"]
        if temperature is None:
            temperature = saved["objective"]["specificity_temperature"]
    cap = float(
        template.get("route1_specificity_negative_kl_cap", 0.2) if cap is None else cap
    )
    temperature = float(
        template.get("route1_specificity_temperature", 0.1)
        if temperature is None
        else temperature
    )
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError("specificity_negative_kl_cap must be finite and positive")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("specificity_temperature must be finite and positive")
    return {
        "specificity_loss": "same-prompt-capped-soft-infonce",
        "specificity_negative_kl_cap": cap,
        "specificity_temperature": temperature,
        "specificity_temperature_placement": "negative-forward-kl-scores",
        "specificity_cap_placement": "raw-negative-token-mean-kl-zero-gradient-at-boundary",
        "specificity_soft_weight": "detached-frozen-prompt-mean-one-plus-cos-over-two-floor-1e-3-legal-mean-one",
        "specificity_pair_reduction": "same-prompt-positive-mean-vs-weighted-negative-mean-owner-mean",
        "specificity_positive_scope": "optimizer-window-same-prompt-records-two-dropout-views",
        "specificity_positive_teacher": "owner-teacher-and-stopped-owner-answer-prefix",
        "route1_record_order": "sorted-record-id-python-random-seed-plus-epoch-shuffle",
        "route1_rank_assignment": "contiguous-slices-no-length-reordering",
    }


def _specificity_sampling_contract(
    arguments: Any, template: Mapping[str, Any]
) -> dict[str, Any]:
    from think_bridge.training.specificity_sampling import (
        SPECIFICITY_SAMPLING_POLICY,
        validate_donor_limit,
    )

    requested = getattr(arguments, "specificity_donors_per_owner", None)
    limit = (
        template.get("route1_specificity_donors_per_owner", 2)
        if requested is None
        else requested
    )
    return {
        "specificity_donors_per_owner": validate_donor_limit(limit),
        "specificity_sampling_policy": SPECIFICITY_SAMPLING_POLICY,
    }


def _specificity_control_contract(
    arguments: Any, template: Mapping[str, Any]
) -> dict[str, Any]:
    """Require direct/no-z to remain a validation-only control."""
    requested = getattr(arguments, "specificity_include_direct", None)
    if requested is None and getattr(arguments, "resume_from_checkpoint", None):
        saved = json.loads(
            (
                Path(arguments.resume_from_checkpoint) / "resolved_config.json"
            ).read_text()
        )
        requested = saved.get("objective", {}).get("specificity_include_direct", False)
    if requested is None:
        requested = template.get("route1_specificity_include_direct", False)
    if requested is not False:
        raise ValueError("specificity_include_direct must be false")
    return {"specificity_include_direct": bool(requested)}


def _distillation_population_contract(
    arguments: Any, template: Mapping[str, Any]
) -> dict[str, Any]:
    """Resolve the native B/C populations used by Match and Specificity."""
    requested = getattr(arguments, "distillation_populations", None)
    if requested is None and getattr(arguments, "resume_from_checkpoint", None):
        saved = json.loads(
            (
                Path(arguments.resume_from_checkpoint) / "resolved_config.json"
            ).read_text()
        )
        values = saved.get("objective", {}).get("distillation_populations")
        if isinstance(values, list):
            requested = "BC" if set(values) == {"B", "C"} else "C"
        else:
            requested = saved.get("objective", {}).get(
                "route1_distillation_populations"
            )
    if requested is None:
        requested = template.get("route1_distillation_populations", "BC")
    normalized = str(requested).upper().replace(",", "")
    if normalized != "BC":
        raise ValueError("distillation_populations must be BC")
    populations = ["B", "C"]
    return {
        "route1_distillation_populations": normalized,
        "distillation_populations": populations,
        "specificity_owner_populations": ["C"],
        "specificity_donor_populations": populations,
    }


def build_reasoner_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="think-bridge reasoner-sft",
        description=(
            "Train only Bridge reasoner R with the three approved Route1 losses; "
            "F, token embeddings, and D remain frozen and outside the optimizer."
        ),
        allow_abbrev=False,
    )
    _add_common_training_arguments(parser)
    parser.set_defaults(
        specificity_margin=None,
        specificity_tau=0.1,
        answer_target_source="selfgen",
        reasoner_zero_init=False,
    )
    parser.set_defaults(
        metric_for_best_model="route1.true_z_full_accuracy",
        eval_steps=50,
        save_steps=50,
    )
    parser.add_argument("--generation_seed", type=int, default=None)
    parser.add_argument(
        "--latent_steps",
        type=int,
        help="Number of recurrent frozen-F feedback steps; defaults to the model template.",
    )
    parser.add_argument("--reasoner_layers", type=int, default=None)
    parser.add_argument(
        "--reasoner_loop_steps",
        type=int,
        default=None,
        help="Shared R stack repetitions per fixed F context; independent of feedback rounds.",
    )
    parser.add_argument(
        "--latents_per_step",
        type=int,
        help="Latent slots emitted per feedback step; defaults to the model template.",
    )
    parser.add_argument(
        "--reasoner_dropout_p",
        type=float,
        default=None,
        help="Training-only dropout probability in R; evaluation always disables it.",
    )
    parser.add_argument(
        "--reasoner_dropout_views",
        type=int,
        default=None,
        help="Number of stochastic R views requested by the Route1 objective (1-8).",
    )
    parser.add_argument(
        "--wrong_control_chunk_size",
        type=int,
        default=None,
        help="Physical wrong-z pair batch; capped by physical_batch_size.",
    )
    parser.add_argument(
        "--population",
        choices=("staged",),
        default="staged",
        help="Use the B/C/D training populations.",
    )
    parser.add_argument("--train_behavior", type=Path)
    parser.add_argument("--eval_behavior", type=Path, required=True)
    parser.add_argument("--train_direct_answer_source", type=Path)
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Explicitly rebuild the shared compiled-target cache.",
    )
    parser.add_argument(
        "--ce_weight", "--course_weight", dest="course_weight", type=float, default=1.0
    )
    parser.add_argument("--match_weight", type=float, default=1.0)
    parser.add_argument(
        "--specific_weight",
        dest="specific_weight",
        type=float,
        default=1.0,
        help="Capped soft InfoNCE specificity weight; active from the first update, 0 disables it",
    )
    parser.add_argument(
        "--specificity_donors_per_owner",
        type=int,
        default=None,
        help="Uniform optimizer-window donors per owner; default 2, or 0 for all legal B+C donors",
    )
    parser.add_argument(
        "--specificity_negative_kl_cap",
        type=float,
        default=None,
        help="Raw negative token-mean FKL saturation threshold; hyperparameter default 0.2, resume inherits if omitted",
    )
    parser.add_argument(
        "--specificity_temperature",
        type=float,
        default=None,
        help="Temperature of -KL logits; fresh default 0.1, exact resume inherits if omitted",
    )
    parser.add_argument(
        "--specificity_wrong_gradient",
        choices=("live",),
        default=None,
        help="Keep donor z gradients live.",
    )
    parser.add_argument(
        "--specificity_include_direct",
        type=_strict_bool,
        default=None,
        help="Include detached direct/no-z in Specificity candidates; validation control only, default false",
    )
    parser.add_argument(
        "--distillation_populations",
        choices=("BC",),
        default=None,
        help="Use native B+C rows for Match and donors, and C owners for Specificity.",
    )
    parser.add_argument(
        "--generation_backend", choices=("torch", "vllm"), default="torch"
    )
    parser.add_argument("--vllm_host", default="127.0.0.1")
    parser.add_argument("--vllm_port", type=int, default=29601)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--vllm_physical_batch_size", type=int, default=4)
    parser.add_argument("--vllm_max_in_flight", type=int, default=4)
    parser.add_argument("--vllm_max_pending_microsteps", type=int, default=1)
    return parser


def infer_model_family(model: str, explicit: str | None = None) -> str:
    if explicit is not None:
        return explicit
    normalized = str(model).lower().replace("_", "-")
    if "qwen3-0.6b" in normalized or "qwen3-06b" in normalized:
        return "qwen3-0.6b"
    if "qwen3-4b" in normalized:
        return "qwen3-4b"
    raise ValueError(
        "cannot infer maintained model family; pass --model_family explicitly"
    )


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, name)


def _world_size(environ: Mapping[str, str]) -> int:
    raw = str(environ.get("NPROC_PER_NODE", "1")).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("NPROC_PER_NODE must be a positive integer") from exc
    return _positive_int(value, "NPROC_PER_NODE")


def _visible_devices(environ: Mapping[str, str], *, world_size: int) -> tuple[str, ...]:
    raw = str(environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if not raw:
        return tuple(str(index) for index in range(world_size))
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if len(values) < world_size or len(set(values)) != len(values):
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must contain unique devices for the stage"
        )
    return values


def _template_path(model_family: str) -> Path:
    name = {
        "qwen3-0.6b": "stage1_qwen3_0.6b.yaml",
        "qwen3-4b": "stage1_qwen3_4b.yaml",
    }[model_family]
    return Path(__file__).resolve().parents[1] / "configs" / name


def _load_template(model_family: str) -> dict[str, Any]:
    path = _template_path(model_family)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("method") != "bridge":
        raise ValueError(f"invalid maintained Bridge template: {path}")
    if payload.get("model_family") != model_family:
        raise ValueError(
            "Bridge template model_family differs from the requested model"
        )
    if payload.get("deepspeed_config"):
        payload["deepspeed_config"] = str(
            path.parent / Path(payload["deepspeed_config"]).name
        )
    return payload


def _validate_optimizer_runtime(runtime: Mapping[str, Any]) -> None:
    """Validate the selected optimizer backend before expensive stage startup."""

    if str(runtime.get("optimizer_backend")) != "deepspeed_zero1":
        return
    raw_path = runtime.get("deepspeed_config")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("DeepSpeed ZeRO-1 runtime requires a config path")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    validate_stage1_deepspeed_config(str(path.resolve()))


def _reasoner_geometry(
    arguments: argparse.Namespace,
    template: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve feedback rounds and slots per round into the supported slot budget."""

    latent_steps = _positive_int(
        int(template["latent_steps"])
        if getattr(arguments, "latent_steps", None) is None
        else arguments.latent_steps,
        "latent_steps",
    )
    latents_per_step = _positive_int(
        int(template["latents_per_step"])
        if getattr(arguments, "latents_per_step", None) is None
        else arguments.latents_per_step,
        "latents_per_step",
    )
    loop_steps = _positive_int(
        template.get("reasoner_loop_steps", 1)
        if getattr(arguments, "reasoner_loop_steps", None) is None
        else arguments.reasoner_loop_steps,
        "reasoner_loop_steps",
    )
    layers = _positive_int(
        template.get("emitter_depth", 1)
        if getattr(arguments, "reasoner_layers", None) is None
        else arguments.reasoner_layers,
        "reasoner_layers",
    )
    if loop_steps > 1 and layers < 2:
        raise ValueError("looped R requires at least two independent layers")
    from think_bridge.model.feedback_precision import QUESTION_READER_INPUT_MODES

    if (
        template.get("reasoner_input_mode") in QUESTION_READER_INPUT_MODES
        and latent_steps != 1
    ):
        raise ValueError("question reader requires a single external R call")
    latent_slots = latent_steps * latents_per_step
    if latent_slots not in {32, 64, 128}:
        raise ValueError(
            "latent_steps * latents_per_step must equal the supported latent budgets 32, 64 or 128"
        )
    dropout_p = float(
        getattr(arguments, "reasoner_dropout_p", None)
        if getattr(arguments, "reasoner_dropout_p", None) is not None
        else template.get("reasoner_dropout_p", 0.0)
    )
    dropout_views = int(
        getattr(arguments, "reasoner_dropout_views", None)
        if getattr(arguments, "reasoner_dropout_views", None) is not None
        else template.get("reasoner_dropout_views", 1)
    )
    if not math.isfinite(dropout_p) or not 0.0 <= dropout_p < 1.0:
        raise ValueError("reasoner_dropout_p must be finite in [0, 1)")
    if not 1 <= dropout_views <= 8:
        raise ValueError("reasoner_dropout_views must be in [1, 8]")
    return {
        "architecture": "shared-loop-state",
        "input_mode": str(
            template.get("reasoner_input_mode", "last-query-reader-self-loop")
        ),
        "tap_count": int(template.get("tap_count", 6)),
        "loop_steps": loop_steps,
        "num_layers": layers,
        "latent_slots": latent_slots,
        "latent_steps": latent_steps,
        "latents_per_step": latents_per_step,
        "real_frozen_f_appends": latent_steps - 1,
        "zero_init": bool(
            getattr(arguments, "reasoner_zero_init", None)
            if getattr(arguments, "reasoner_zero_init", None) is not None
            else template.get("reasoner_zero_init", True)
        ),
        "output_normalization": str(
            template.get("reasoner_output_normalization", "residual")
        ),
        "dropout_p": dropout_p,
        "dropout_views": dropout_views,
        "geometry_schema_version": (
            f"recursive-t{latent_steps}-b{latents_per_step}-k{latent_slots}-answer-only"
        ),
    }


def _file_reference(path: Path, label: str) -> dict[str, Any]:
    return file_reference(path, label)


def _stage1_cache_root(
    model_family: str,
    inputs: Mapping[str, Mapping[str, Any]],
    *,
    cache_root: str | Path,
) -> Path:
    return (
        bridge_stage1_model_root(model_family, cache_root=cache_root)
        / "datasets"
        / dataset_identity(inputs)
    )


def _input_locator_contract(inputs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return input_contract(inputs)


def _reasoner_behavior_contract(
    inputs: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Validate behavior artifacts structurally without content-hash admission."""

    schemas: list[dict[str, Any]] = []
    for role in ("train_behavior", "eval_behavior"):
        if not inputs[role].get("path"):
            continue
        payload = json.loads(Path(inputs[role]["path"]).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(
            payload.get("prompts"), list
        ):
            raise ValueError(f"{role} lacks a behavior prompt mapping")
        schemas.append(
            {
                "role": role,
                "schema_version": payload.get("schema_version"),
                "status": payload.get("status"),
            }
        )
    if not schemas:
        raise ValueError("reasoner inputs lack behavior labels")
    return {"validation": "schema-and-required-fields", "artifacts": schemas}


def _runtime_sidecar_content_proof(run_dir: Path) -> dict[str, Any]:
    payload = json.loads(
        (Path(run_dir) / "runtime_sidecars.json").read_text(encoding="utf-8")
    )
    identity = payload.get("identity") if isinstance(payload, dict) else None
    if not isinstance(identity, dict):
        raise ValueError("R runtime sidecar lacks its tokenizer identity")
    fields = (
        "tokenizer_sha256",
        "template_sha256",
        "boundary_ids_sha256",
    )
    proof: dict[str, Any] = {
        name: require_sha256(str(identity.get(name)), name) for name in fields
    }
    for name in (
        "hidden_size",
        "vocab_size",
        "eos_token_id",
        "pad_token_id",
        "boundary_token_count",
    ):
        value = identity.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"runtime sidecar {name} is invalid")
        proof[name] = int(value)
    return proof


def _model_semantic_identity(
    resolved: Mapping[str, Any], *, component: str, content_proof: Mapping[str, Any]
) -> dict[str, Any]:
    """Project only validated model/tokenizer and component architecture semantics."""
    from think_bridge.model.feedback_precision import (
        feedback_policy,
        saved_reasoner_loop_steps,
        saved_reasoner_layers,
    )

    geometry = resolved.get("reasoner_geometry")
    geometry_overrides = {"reasoner_loop_steps": saved_reasoner_loop_steps(resolved)}
    if isinstance(geometry, Mapping):
        geometry_overrides = {
            name: geometry[name]
            for name in (
                "latent_slots",
                "latent_steps",
                "latents_per_step",
                "real_frozen_f_appends",
                "geometry_schema_version",
            )
            if name in geometry
        }
        geometry_overrides["reasoner_loop_steps"] = saved_reasoner_loop_steps(resolved)
        geometry_overrides["reasoner_output_normalization"] = str(
            geometry["output_normalization"]
        )
        if "zero_init" in geometry:
            geometry_overrides["reasoner_zero_init"] = bool(geometry["zero_init"])
        if "dropout_p" in geometry:
            geometry_overrides["reasoner_dropout_p"] = float(geometry["dropout_p"])
        if "dropout_views" in geometry:
            geometry_overrides["reasoner_dropout_views"] = int(
                geometry["dropout_views"]
            )
    from think_bridge.model.feedback_precision import saved_reasoner_input

    geometry_overrides.update(saved_reasoner_input(resolved))
    geometry_overrides["emitter_depth"] = saved_reasoner_layers(resolved)
    config = load_training_config(
        Path(str(resolved["method_config"])),
        model_name_or_path=str(resolved["model"]),
        tokenizer_name_or_path=str(resolved["tokenizer"]),
        runtime_overrides={
            **(geometry_overrides or {}),
            "generation_seed": int(resolved["training"]["generation_seed"]),
        },
    )
    architecture = {
        "name": f"bridge-recurrent-feedback-emitter-t{config.latent_steps}-b{config.latents_per_step}-k{config.latent_slots}",
        "latent_slots": int(config.latent_slots),
        "latent_steps": int(config.latent_steps),
        "latents_per_step": int(config.latents_per_step),
        "real_frozen_f_appends": int(config.real_frozen_f_appends),
        "bound_scale_init": float(config.bound_scale_init),
        "trajectory_max_steps": int(config.trajectory_max_steps),
        "attn_implementation": str(config.attn_implementation),
        "owner_dtype": "float32",
        "dropout_p": float(config.reasoner_dropout_p),
        "dropout_views": int(config.reasoner_dropout_views),
    }
    architecture["input_mode"] = config.reasoner_input_mode
    architecture["tap_count"] = config.tap_count
    if config.emitter_depth != 1:
        architecture["num_layers"] = int(config.emitter_depth)
    if config.reasoner_loop_steps != 1:
        architecture["loop_steps"] = int(config.reasoner_loop_steps)
    if isinstance(geometry, Mapping) and "output_normalization" in geometry:
        architecture["output_normalization"] = config.reasoner_output_normalization
    return {
        **artifact_header("think-bridge.stage1.model-semantic-identity"),
        "model_family": str(resolved["model_family"]),
        "component": component,
        "content_proof": dict(content_proof),
        "architecture": architecture,
        "feedback_policy": feedback_policy(resolved),
        "answer_evaluation": resolved.get("answer_evaluation"),
    }


def vllm_service_max_queued_requests(
    *,
    trainer_world_size: int,
    per_rank_in_flight: int,
    service_active: int,
    pending_microsteps: int = 1,
) -> int:
    """Bound the full trainer burst after the service's active slots fill."""

    world = _positive_int(trainer_world_size, "trainer_world_size")
    per_rank = _positive_int(per_rank_in_flight, "per_rank_in_flight")
    active = _positive_int(service_active, "service_active")
    # Each prefetched microstep owns an independent client.stream() thread
    # pool. Account for their combined burst without raising GPU concurrency.
    pending = _positive_int(pending_microsteps, "pending_microsteps")
    return max(1, world * per_rank * pending - active)


def scientific_identity_sha256(
    scientific: Mapping[str, Any], *, operational: Mapping[str, Any] | None = None
) -> str:
    """Hash only scientific/recovery state; operational placement is diagnostic."""

    if not isinstance(scientific, Mapping) or not scientific:
        raise ValueError("scientific stage identity must be a nonempty mapping")
    if operational is not None and not isinstance(operational, Mapping):
        raise TypeError("operational diagnostics must be a mapping")
    return canonical_json_sha256(dict(scientific))


def _bind_resume_training_plan(
    resolved: dict[str, Any], scientific: Mapping[str, Any]
) -> None:
    """Keep a run's recovery identity while explicitly recording a longer plan.

    Only budget and evaluation/save cadence may change. Model, loss, course,
    inputs, optimizer settings and effective batch remain compatibility checks.
    """
    resume = resolved.get("resume_from_checkpoint")
    if not resume:
        return
    source = Path(resume) / "resolved_config.json"
    previous = json.loads(source.read_text(encoding="utf-8"))
    if previous.get("answer_evaluation") != resolved.get("answer_evaluation"):
        raise ValueError(
            "Resume evaluation protocol differs; start a fresh run so best scores remain comparable"
        )
    old_training, new_training = previous["training"], resolved["training"]
    import copy

    candidate = copy.deepcopy(scientific)
    candidate_training = dict(candidate["training"])
    for key in ("num_train_epochs", "max_steps", "eval_steps", "save_steps"):
        candidate_training[key] = old_training[key]
    candidate["training"] = candidate_training
    if "model_semantics" in candidate:
        old_feedback = previous.get("feedback_policy", {})
        semantics = candidate["model_semantics"]
        components = semantics.get("components", {"R": semantics})
        for component in components.values():
            if old_feedback:
                component["feedback_policy"]["reasoner_eval_group_size"] = old_feedback[
                    "reasoner_eval_group_size"
                ]
                component["feedback_policy"]["tf32"] = old_feedback.get(
                    "tf32", "framework-default"
                )
            component["answer_evaluation"] = previous.get("answer_evaluation")
    old_plan = previous.get(
        "requested_plan_sha256", previous["scientific_identity_sha256"]
    )
    if scientific_identity_sha256(candidate) != old_plan:
        raise ValueError(
            "resume model/data/loss/course/optimizer contract differs; only budget and eval/save cadence may change"
        )
    old_epochs, new_epochs = (
        int(old_training["num_train_epochs"]),
        int(new_training["num_train_epochs"]),
    )
    old_max, new_max = int(old_training["max_steps"]), int(new_training["max_steps"])
    if new_epochs < old_epochs or (new_max > 0 and (old_max < 0 or new_max < old_max)):
        raise ValueError("resume training budget may be extended, not shortened")
    resolved["requested_plan_sha256"] = resolved["scientific_identity_sha256"]
    resolved["scientific_identity_sha256"] = previous["scientific_identity_sha256"]
    if old_training != new_training:
        resolved["resume_training_plan"] = {
            "checkpoint": str(resume),
            "previous_training": dict(old_training),
            "requested_training": dict(new_training),
            "scheduler_policy": "cosine-new-total-at-restored-global-step",
        }


def _scientific_evaluation_identity(
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    """Exclude the executor transport while retaining evaluation semantics."""

    projected = dict(evaluation)
    projected.pop("generation_backend", None)
    return projected


def _training_identity(training: Mapping[str, Any], *, stage: str) -> dict[str, Any]:
    """Project resolved training config onto trajectory/recovery semantics."""
    fields = (
        "effective_global_batch",
        "num_train_epochs",
        "max_steps",
        "learning_rate",
        "weight_decay",
        "warmup_steps",
        "max_grad_norm",
        "eval_steps",
        "save_steps",
        "seed",
    )
    projected = {name: training[name] for name in fields}
    projected["generation_seed"] = training["generation_seed"]
    projected["route1_generation_temperature"] = training.get(
        "route1_generation_temperature", 0.0
    )
    return projected


def _base_resolution(
    arguments: argparse.Namespace,
    *,
    stage: str,
    environ: Mapping[str, str],
    profile: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if stage not in {"reasoner-sft"}:
        raise ValueError(f"unknown maintained Stage1 stage: {stage}")
    route1_like = True
    family = infer_model_family(arguments.model, arguments.model_family)
    method_config = _template_path(family)
    template = _load_template(family)
    reasoner_geometry = _reasoner_geometry(arguments, template)
    world = _world_size(environ)
    selected = dict(profile or {})
    per_device = _positive_int(
        arguments.per_device_train_batch_size
        if arguments.per_device_train_batch_size is not None
        else int(
            selected.get(
                "per_device_train_batch_size", template["route1_local_samples"]
            )
        ),
        "per_device_train_batch_size",
    )
    gas = _positive_int(
        arguments.gradient_accumulation_steps
        if arguments.gradient_accumulation_steps is not None
        else int(
            selected.get(
                "gradient_accumulation_steps",
                template["route1_gradient_accumulation_steps"],
            )
        ),
        "gradient_accumulation_steps",
    )
    geometry = BatchGeometry(per_device, world, gas)
    geometry.validate()
    epochs_default = int(template["route1_epochs"])
    epochs = _positive_int(
        epochs_default
        if arguments.num_train_epochs is None
        else arguments.num_train_epochs,
        "num_train_epochs",
    )
    if arguments.max_steps == 0 or arguments.max_steps < -1:
        raise ValueError("max_steps must be -1 or a positive integer")
    eval_batch = _positive_int(
        arguments.per_device_eval_batch_size
        if arguments.per_device_eval_batch_size is not None
        else int(template["route1_eval_local_row_batch"]),
        "per_device_eval_batch_size",
    )
    learning_rate = float(
        arguments.learning_rate
        if arguments.learning_rate is not None
        else template["lr_r"]
    )
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    warmup_default = template["warmup_updates_r"]
    warmup = int(
        warmup_default if arguments.warmup_steps is None else arguments.warmup_steps
    )
    if warmup < 0:
        raise ValueError("warmup_steps must be nonnegative")
    for name in ("eval_steps", "save_steps", "logging_steps", "save_total_limit"):
        _positive_int(getattr(arguments, name), name)
    visible = _visible_devices(environ, world_size=world)
    output = Path(arguments.output_dir).expanduser().resolve()
    tokenizer = arguments.tokenizer or arguments.model
    training = {
        "per_device_train_batch_size": per_device,
        "per_device_eval_batch_size": eval_batch,
        "gradient_accumulation_steps": gas,
        "world_size": world,
        "effective_global_batch": geometry.effective_global_batch,
        "num_train_epochs": epochs,
        "max_steps": int(arguments.max_steps),
        "learning_rate": learning_rate,
        "weight_decay": float(arguments.weight_decay),
        "warmup_steps": warmup,
        "max_grad_norm": float(arguments.max_grad_norm),
        "eval_steps": int(arguments.eval_steps),
        "save_steps": int(arguments.save_steps),
        "logging_steps": int(arguments.logging_steps),
        "save_total_limit": int(arguments.save_total_limit),
        "seed": int(arguments.seed),
    }
    if arguments.generation_seed is not None and int(arguments.generation_seed) != int(
        arguments.seed
    ):
        raise ValueError("generation_seed must equal seed; omit it to use seed")
    training["generation_seed"] = int(arguments.seed)
    training["route1_generation_temperature"] = float(
        template.get("route1_generation_temperature", 0.0)
    )
    default_physical_batch = template["route1_local_chunk_size"]
    default_checkpointing = template["route1_gradient_checkpointing"]
    profile_checkpointing = selected.get(
        "gradient_checkpointing", default_checkpointing
    )
    if arguments.gradient_checkpointing is None and profile_checkpointing is None:
        raise ValueError(
            "gradient_checkpointing must have an explicit or template default"
        )
    runtime = {
        "profile": selected.get("name", f"{stage}-{family}-world{world}"),
        "profile_status": selected.get("status", "stage-owned"),
        "physical_batch_size": _positive_int(
            arguments.physical_batch_size
            if arguments.physical_batch_size is not None
            else int(selected.get("physical_batch_size", default_physical_batch)),
            "physical_batch_size",
        ),
        "gradient_checkpointing": bool(arguments.gradient_checkpointing)
        if arguments.gradient_checkpointing is not None
        else bool(profile_checkpointing),
        "prefetch_batches": int(
            getattr(arguments, "prefetch_batches", None)
            if getattr(arguments, "prefetch_batches", None) is not None
            else selected.get("prefetch_batches", 1)
        ),
        "optimizer_backend": str(
            selected.get("optimizer_backend", template["optimizer_backend"])
        ),
        "zero_stage": int(selected.get("zero_stage", template["zero_stage"])),
        "deepspeed_config": template["deepspeed_config"],
        "cuda_visible_devices": ",".join(visible),
        "trainer_devices": list(visible[-world:]),
        "master_port": int(environ.get("MASTER_PORT", "29500")),
        "local_files_only": bool(arguments.local_files_only),
        "no_progress": bool(arguments.no_progress),
        "performance_observability_schema": PERFORMANCE_OBSERVABILITY_SCHEMA,
        "full_state_audit": bool(arguments.full_state_audit),
        "run_locator": None
        if getattr(arguments, "run_locator", None) is None
        else str(Path(arguments.run_locator).expanduser().resolve()),
    }
    runtime["wrong_control_chunk_size"] = min(
        _positive_int(
            getattr(arguments, "wrong_control_chunk_size", None)
            if getattr(arguments, "wrong_control_chunk_size", None) is not None
            else int(template["route1_wrong_control_chunk_size"]),
            "wrong_control_chunk_size",
        ),
        int(runtime["physical_batch_size"]),
    )
    if runtime["prefetch_batches"] not in {0, 1}:
        raise ValueError("prefetch_batches must be zero or one")
    _validate_optimizer_runtime(runtime)
    resolved = {
        **artifact_header(STAGE_RESOLVED_CONFIG),
        "stage": stage,
        "method": "bridge",
        "model_family": family,
        "model": str(arguments.model),
        "tokenizer": str(tokenizer),
        "method_config": str(method_config),
        "output_dir": str(output),
        "training": training,
        "runtime": runtime,
        "resume_from_checkpoint": None
        if arguments.resume_from_checkpoint is None
        or not str(arguments.resume_from_checkpoint).strip()
        else str(Path(arguments.resume_from_checkpoint).expanduser().resolve()),
    }
    from think_bridge.model.feedback_precision import feedback_policy

    policy_inputs = dict(template)
    for name in ("reasoner_compute_dtype", "reasoner_eval_group_size"):
        if getattr(arguments, name, None) is not None:
            policy_inputs[name] = getattr(arguments, name)
    resolved["feedback_policy"] = feedback_policy(policy_inputs)
    from think_bridge.eval.hf_protocol import hf_evaluation_protocol

    resolved["answer_evaluation"] = hf_evaluation_protocol(
        eval_batch,
        reasoner_batch_size=resolved["feedback_policy"]["reasoner_eval_group_size"],
    )
    if reasoner_geometry is not None:
        resolved["reasoner_geometry"] = reasoner_geometry
    from think_bridge.training.metric_registry import resolve_metric_policy

    resolved["selection"] = resolve_metric_policy(
        entrance=stage,
        metric_for_best_model=arguments.metric_for_best_model,
        metric_aggregation=arguments.metric_aggregation,
    ).as_dict()
    return (resolved, template)


def resolve_reasoner_config(
    arguments: argparse.Namespace, *, environ: Mapping[str, str] | None = None
) -> dict[str, Any]:
    environment = dict(os.environ if environ is None else environ)
    max_eval_samples = _optional_positive_int(
        getattr(arguments, "max_eval_samples", None), "max_eval_samples"
    )
    resolved, template = _base_resolution(
        arguments, stage="reasoner-sft", environ=environment
    )
    resolved["runtime"]["overwrite_cache"] = bool(arguments.overwrite_cache)
    population = str(arguments.population).replace("_", "-")
    if population == "staged":
        if arguments.train_behavior is None:
            raise ValueError("staged reasoner-sft requires --train_behavior")
        if arguments.train_direct_answer_source is None:
            raise ValueError(
                "staged reasoner-sft requires --train_direct_answer_source"
            )
    elif (
        arguments.train_behavior is not None
        or arguments.train_direct_answer_source is not None
    ):
        raise ValueError(
            "answer_only reasoner-sft must omit train behavior/direct-answer artifacts"
        )
    inputs = {
        "train_dataset": _file_reference(arguments.train_dataset, "train dataset"),
        "eval_dataset": _file_reference(arguments.eval_dataset, "eval dataset"),
        "train_behavior": (
            _file_reference(arguments.train_behavior, "train behavior")
            if arguments.train_behavior is not None
            else {
                "path": None,
                "role": "explicit-answer-only",
            }
        ),
        "eval_behavior": _file_reference(arguments.eval_behavior, "eval behavior"),
        "train_direct_answer_source": (
            _file_reference(
                arguments.train_direct_answer_source, "train direct-answer source"
            )
            if arguments.train_direct_answer_source is not None
            else {
                "path": None,
                "role": "not-applicable-to-answer-only",
            }
        ),
    }
    if not math.isfinite(arguments.specificity_tau) or arguments.specificity_tau <= 0:
        raise ValueError("specificity_tau must be finite and positive")
    specificity_control = _specificity_control_contract(arguments, template)
    if specificity_control["specificity_include_direct"]:
        raise ValueError("capped soft InfoNCE excludes direct/no-z candidates")
    losses = {
        "three_loss": _REASONER_OBJECTIVE,
        "execution_revision": "ce-match-specificity",
        "ce_weight": float(arguments.course_weight),
        "active_domain_schema": ROUTE1_ACTIVE_DOMAIN_SCHEMA,
        "physical_cost_policy_schema": ROUTE1_PHYSICAL_COST_POLICY_SCHEMA,
        "occurrence_sampler_schema": ROUTE1_OCCURRENCE_SAMPLER_SCHEMA,
        "course_weight": float(arguments.course_weight),
        "match_weight": float(arguments.match_weight),
        "specific_weight": float(arguments.specific_weight),
        "specificity_tau": float(arguments.specificity_tau),
        **_specificity_gradient_contract(arguments, template),
        **_specificity_objective_contract(arguments, template),
        **specificity_control,
        **_distillation_population_contract(arguments, template),
        "specificity_reduction": "eligible-owner-mean-C-or-reference-window",
        "match_reduction": "token-mean-B-C-empirical-window-mean",
        **_specificity_sampling_contract(arguments, template),
        **_ce_condition_contract(arguments, template),
        "distillation_input": "pure-z-student-full-cot-teacher-shared-student-answer-prefix",
        "distillation_sample_source": "same-CE-batch-Match-BC-Specificity-C-or-reference",
        "specificity_control": (
            "unsupported-direct-candidate"
            if specificity_control["specificity_include_direct"]
            else "excluded-direct-prompt-from-specificity-candidates"
        ),
        "specificity_weight_schedule": "constant-from-first-update",
        "population": population,
        "answer_target_source": ("paired-stage0-self-answer"),
    }
    if any(
        not math.isfinite(losses[name]) or losses[name] < 0.0
        for name in ("course_weight", "match_weight", "specific_weight")
    ):
        raise ValueError("reasoner loss weights must be nonnegative")
    if (
        sum(
            losses[name]
            for name in ("course_weight", "match_weight", "specific_weight")
        )
        <= 0.0
    ):
        raise ValueError("at least one reasoner loss weight must be positive")
    generation_backend = str(arguments.generation_backend)
    visible = tuple(resolved["runtime"]["cuda_visible_devices"].split(","))
    world = int(resolved["training"]["world_size"])
    if generation_backend == "vllm":
        if len(visible) <= world:
            raise ValueError(
                "vLLM reasoner evaluation requires service GPUs outside trainer NPROC"
            )
        service_devices = visible[:-world]
        trainer_devices = visible[-world:]
    else:
        if len(visible) != world:
            visible = visible[-world:]
        service_devices = ()
        trainer_devices = visible
    resolved["runtime"].update(
        generation_backend=generation_backend,
        trainer_devices=list(trainer_devices),
        vllm_devices=list(service_devices),
        vllm_host=str(arguments.vllm_host),
        vllm_port=int(arguments.vllm_port),
        vllm_gpu_memory_utilization=float(arguments.vllm_gpu_memory_utilization),
        vllm_physical_batch_size=_positive_int(
            arguments.vllm_physical_batch_size, "vllm_physical_batch_size"
        ),
        vllm_max_in_flight=_positive_int(
            arguments.vllm_max_in_flight, "vllm_max_in_flight"
        ),
        vllm_max_pending_microsteps=_positive_int(
            arguments.vllm_max_pending_microsteps, "vllm_max_pending_microsteps"
        ),
    )
    resolved["runtime"]["vllm_service_max_queued_requests"] = (
        vllm_service_max_queued_requests(
            trainer_world_size=world,
            per_rank_in_flight=int(resolved["runtime"]["vllm_max_in_flight"]),
            service_active=int(resolved["runtime"]["vllm_max_in_flight"]),
            pending_microsteps=min(
                int(resolved["runtime"]["vllm_max_pending_microsteps"]),
                int(resolved["training"]["gradient_accumulation_steps"]),
            ),
        )
    )
    resolved.update(
        inputs=inputs,
        objective=losses,
        evaluation={
            "scope": "free-true-control-route1",
            "null_mode": "direct",
            "generation_backend": "torch",
            "max_eval_samples": max_eval_samples,
        },
    )
    scientific = {
        key: resolved[key]
        for key in (
            "stage",
            "method",
            "model_family",
            "objective",
            "evaluation",
            "selection",
        )
    }
    scientific["inputs"] = _input_locator_contract(inputs)
    scientific["evaluation"] = _scientific_evaluation_identity(resolved["evaluation"])
    scientific["model_semantics"] = _model_semantic_identity(
        resolved,
        component="R",
        content_proof=_reasoner_behavior_contract(inputs),
    )
    scientific["training"] = _training_identity(
        resolved["training"], stage="reasoner-sft"
    )
    scientific["execution"] = {
        name: resolved["runtime"][name] for name in ("optimizer_backend", "zero_stage")
    }
    resolved["scientific_identity_sha256"] = scientific_identity_sha256(
        scientific, operational=resolved["runtime"]
    )
    _bind_resume_training_plan(resolved, scientific)
    resolved["runtime"]["base_optimizer_backend"] = template["optimizer_backend"]
    return resolved


def build_worker_launch(
    *,
    action: str,
    resolved_config: Path,
    artifacts: Path,
    world_size: int,
    master_port: int,
) -> list[str]:
    if action not in {"prepare-targets", "materialize-z", "train", "select"}:
        raise ValueError(f"unknown stage worker action: {action}")
    command = [sys.executable]
    if action in {"materialize-z", "train"}:
        _positive_int(world_size, "world_size")
        _positive_int(master_port, "master_port")
        command.extend(
            [
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={world_size}",
                "--nnodes=1",
                "--rdzv-backend=c10d",
                f"--rdzv-endpoint=127.0.0.1:{master_port}",
                f"--rdzv-id=bridge-stage-{master_port}",
            ]
        )
    command.extend(
        [
            "-m",
            "think_bridge.training.stage_worker",
            action,
            "--resolved-config",
            str(resolved_config),
            "--artifacts",
            str(artifacts),
        ]
    )
    return command


def stage_artifact_paths(resolved: Mapping[str, Any]) -> dict[str, str]:
    family = str(resolved["model_family"])
    template = _load_template(family)
    dataset_root = _stage1_cache_root(
        family, resolved["inputs"], cache_root=template["cache_root"]
    )
    output = Path(str(resolved["output_dir"]))
    common = {
        "resolved_config": str(output / "resolved_config.json"),
        "artifacts": str(output / "artifacts.json"),
        "target_index": str(dataset_root / "targets" / "index.json"),
        "parent_manifest": str(dataset_root / "manifests" / "stage1.json"),
        "donor_manifest": str(dataset_root / "validation-donors.json"),
        "validation_records": str(dataset_root / "targets" / "validation.jsonl"),
        "structured_log": str(output / "logging.jsonl"),
        "report_dir": str(output / "reports"),
    }
    common.update(
        donor_manifest=str(dataset_root / "validation-donors.json"),
        selection=str(output / "checkpoint_selection_route1.json"),
        best_checkpoint_pointer=str(output / "best_reasoner_checkpoint.txt"),
    )
    return common
