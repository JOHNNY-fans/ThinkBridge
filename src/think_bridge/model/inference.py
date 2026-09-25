"""Load the frozen base model and the selected R checkpoint."""

from __future__ import annotations


from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class CheckpointComponent:
    checkpoint: Path
    component: str
    checkpoint_kind: str
    owner: str
    weight_path: Path
    resolved_config_path: Path
    resolved_config: Mapping[str, Any]
    expected_tensor_names: tuple[str, ...] | None


@dataclass(frozen=True)
class ReasonerGeometry:
    latent_slots: int
    latent_steps: int
    latents_per_step: int
    loop_steps: int = 1
    num_layers: int = 1


@dataclass(frozen=True)
class RuntimeBundle:
    model: Any
    tokenizer: Any
    device: Any
    reasoner_source: CheckpointComponent
    reasoner_geometry: ReasonerGeometry
    model_family: str


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def resolve_checkpoint_component(
    checkpoint: Path, component: str
) -> CheckpointComponent:
    """Validate a sealed R checkpoint before loading its owner weights."""
    from think_bridge.model.checkpoint_policy import validate_checkpoint_directory

    root = Path(checkpoint).expanduser().resolve(strict=True)
    validate_checkpoint_directory(root)
    metadata = _read_json_object(root / "checkpoint.json", label="checkpoint metadata")
    expected_owner = {"reasoner": "R"}.get(component)
    if expected_owner is None or metadata.get("owner") != expected_owner:
        raise ValueError("checkpoint owner does not match requested component")
    resolved_path = root / "resolved_config.json"
    resolved = _read_json_object(resolved_path, label="resolved config")
    from think_bridge.model.artifact_schema import (
        require_artifact_header,
        STAGE_RESOLVED_CONFIG,
    )

    require_artifact_header(resolved, STAGE_RESOLVED_CONFIG, label="resolved config")
    if resolved.get("model_family") not in {"qwen3-0.6b", "qwen3-4b"}:
        raise ValueError("unsupported checkpoint model family")
    weight = root / "model.safetensors"
    if not weight.is_file():
        raise FileNotFoundError("checkpoint lacks owner weights")
    return CheckpointComponent(
        root,
        component,
        resolved["stage"],
        expected_owner,
        weight,
        resolved_path,
        resolved,
        None,
    )


def project_component_state(
    state: Mapping[str, Any], *, owner: str, component: str
) -> dict[str, Any]:
    if {"reasoner": "R"}.get(component) != owner or not state:
        raise ValueError("invalid checkpoint owner or empty tensor state")
    return dict(state)


def _load_component_state(source: CheckpointComponent) -> dict[str, Any]:
    from safetensors.torch import load_file

    return project_component_state(
        load_file(str(source.weight_path), device="cpu"),
        owner=source.owner,
        component=source.component,
    )


def resolve_reasoner_geometry(
    resolved: Mapping[str, Any], state: Mapping[str, Any]
) -> ReasonerGeometry:
    """Recover saved feedback geometry and cross-check checkpoint tensors."""

    from think_bridge.model.feedback_precision import (
        saved_reasoner_loop_steps,
        saved_reasoner_layers,
    )

    loop_steps = saved_reasoner_loop_steps(resolved)
    num_layers = saved_reasoner_layers(resolved)
    from think_bridge.model.feedback_precision import (
        saved_reasoner_input,
        QUESTION_READER_INPUT_MODES,
    )

    saved_input = saved_reasoner_input(resolved)
    has_reader = any(name.startswith("input_reader.") for name in state)
    if has_reader != (
        saved_input["reasoner_input_mode"] in QUESTION_READER_INPUT_MODES
    ):
        raise ValueError("R input_reader tensors disagree with saved input_mode")
    query = state.get("query_base")
    shape = tuple(int(value) for value in getattr(query, "shape", ()))
    if len(shape) != 3 or shape[0] != 1 or shape[1] <= 0 or shape[2] <= 0:
        raise ValueError(
            "Reasoner checkpoint lacks a valid query_base tensor shaped [1,B,dF]"
        )
    tensor_latents_per_step = int(shape[1])
    raw = resolved.get("reasoner_geometry")
    latent_slots = int(raw.get("latent_slots", 64)) if isinstance(raw, Mapping) else 64
    if latent_slots not in {32, 64, 128}:
        raise ValueError("unsupported checkpoint slot count")
    if latent_slots % tensor_latents_per_step:
        raise ValueError(
            "Reasoner query_base width does not divide the saved slot budget"
        )
    tensor_steps = latent_slots // tensor_latents_per_step
    raw = resolved.get("reasoner_geometry")
    if isinstance(raw, Mapping):
        try:
            geometry = ReasonerGeometry(
                latent_slots=int(raw["latent_slots"]),
                latent_steps=int(raw["latent_steps"]),
                latents_per_step=int(raw["latents_per_step"]),
                loop_steps=loop_steps,
                num_layers=num_layers,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("resolved reasoner_geometry is incomplete") from exc
    else:
        geometry = ReasonerGeometry(
            latent_slots=latent_slots,
            latent_steps=tensor_steps,
            latents_per_step=tensor_latents_per_step,
        )
    if (
        geometry.latent_slots != latent_slots
        or geometry.latent_steps <= 0
        or geometry.latents_per_step <= 0
        or geometry.latent_steps * geometry.latents_per_step != latent_slots
        or geometry.latent_steps != tensor_steps
        or geometry.latents_per_step != tensor_latents_per_step
    ):
        raise ValueError(
            "resolved reasoner geometry differs from checkpoint query_base shape: "
            f"resolved={geometry.latent_steps}x{geometry.latents_per_step}, "
            f"tensor={tensor_steps}x{tensor_latents_per_step}"
        )
    # Parameter names distinguish independent layer counts even when widths match.
    block_indices = {
        int(key.split(".")[1]) for key in state if key.startswith("blocks.")
    }
    if block_indices and block_indices != set(range(geometry.num_layers)):
        raise ValueError("saved R num_layers differs from checkpoint blocks")
    if geometry.num_layers > 1 and (
        not block_indices or any(key.startswith("block.") for key in state)
    ):
        raise ValueError("multi-layer R checkpoint lacks matching blocks")
    if geometry.loop_steps > 1 and geometry.num_layers < 2:
        raise ValueError("looped R requires at least two independent layers")
    return geometry


def _restore_component(module: Any, state: Mapping[str, Any], *, label: str) -> None:
    import torch

    parameters = {
        name: parameter
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    if set(state) != set(parameters):
        missing = sorted(set(parameters).difference(state))[:8]
        unexpected = sorted(set(state).difference(parameters))[:8]
        raise ValueError(
            f"{label} checkpoint tensor names differ from the runtime module: "
            f"missing={missing}, unexpected={unexpected}"
        )
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = state[name]
            if value.dtype != torch.float32:
                raise TypeError(f"{label} checkpoint tensor is not FP32: {name}")
            if tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"{label} checkpoint tensor shape differs: {name}; "
                    f"checkpoint={tuple(value.shape)} runtime={tuple(parameter.shape)}"
                )
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def _config_path(model_family: str) -> Path:
    name = {
        "qwen3-0.6b": "stage1_qwen3_0.6b.yaml",
        "qwen3-4b": "stage1_qwen3_4b.yaml",
    }[model_family]
    return Path(__file__).resolve().parents[1] / "configs" / name


def record_reasoner_input(model: Any, resolved: Mapping[str, Any]) -> None:
    """Record actual saved semantics without rewriting the loaded architecture."""
    from think_bridge.model.feedback_precision import saved_reasoner_input

    saved_mode = saved_reasoner_input(resolved)["reasoner_input_mode"]
    if model.reasoner.input_mode != saved_mode:
        raise ValueError("R input mode differs from saved checkpoint")
    model._inference_reasoner_input = {
        "saved_input_mode": saved_mode,
        "effective_input_mode": model.reasoner.input_mode,
        "inference_intervention": False,
    }


def _build_runtime(arguments: Any) -> RuntimeBundle:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from think_bridge.data.templates import THINK_BOUNDARY_TEXT
    from think_bridge.model.contract import (
        ANSWER_CAPACITY,
        COT_CONTENT_CAPACITY,
        resolve_boundary_token_ids,
    )
    from think_bridge.model.parallel_model import BridgeParallelModel
    from think_bridge.model.training_config import load_training_config
    from think_bridge.model.feedback_precision import saved_output_normalization
    from think_bridge.training.train import _head_count

    reasoner_source = resolve_checkpoint_component(
        arguments.reasoner_checkpoint, "reasoner"
    )
    family = str(reasoner_source.resolved_config["model_family"])
    reasoner_state = _load_component_state(reasoner_source)
    geometry = resolve_reasoner_geometry(
        reasoner_source.resolved_config, reasoner_state
    )
    seed = int(arguments.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(
        "cuda"
        if arguments.local_device == "auto" and torch.cuda.is_available()
        else "cpu"
        if arguments.local_device == "auto"
        else arguments.local_device
    )
    tokenizer_name = arguments.tokenizer or str(
        reasoner_source.checkpoint / "tokenizer"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, local_files_only=arguments.local_files_only
    )
    executor_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    executor = AutoModelForCausalLM.from_pretrained(
        arguments.model,
        torch_dtype=executor_dtype,
        attn_implementation=arguments.attn_implementation,
        local_files_only=arguments.local_files_only,
    ).to(device)
    from think_bridge.training.runtime_sidecars import (
        _runtime_identity,
        load_route_runtime_identity,
    )
    from think_bridge.model.checkpoint_policy import checkpoint_owner_run_directory

    observed_identity = _runtime_identity(
        tokenizer=tokenizer,
        executor=executor,
        attn_implementation=arguments.attn_implementation,
        boundary_text=THINK_BOUNDARY_TEXT,
        no_progress=bool(getattr(arguments, "no_progress", False)),
    )
    for component in (reasoner_source,):
        if component is None:
            continue
        saved_identity = load_route_runtime_identity(
            checkpoint_owner_run_directory(component.checkpoint),
            route="route1",
        )
        differences = [
            name
            for name, value in observed_identity.items()
            if saved_identity.get(name) != value
        ]
        if differences:
            raise ValueError(
                f"{component.owner} tokenizer/runtime identity mismatch: {differences}"
            )
    from think_bridge.model.feedback_precision import saved_reasoner_input

    config = load_training_config(
        _config_path(family),
        model_name_or_path=str(arguments.model),
        tokenizer_name_or_path=str(tokenizer_name),
        runtime_overrides={
            **saved_reasoner_input(reasoner_source.resolved_config),
            "reasoner_loop_steps": geometry.loop_steps,
            "emitter_depth": geometry.num_layers,
            "latent_slots": geometry.latent_slots,
            "latent_steps": geometry.latent_steps,
            "latents_per_step": geometry.latents_per_step,
            "real_frozen_f_appends": geometry.latent_steps - 1,
            "geometry_schema_version": (
                f"recursive-t{geometry.latent_steps}-b{geometry.latents_per_step}-"
                f"k{geometry.latent_slots}-answer-only"
            ),
            "reasoner_output_normalization": saved_output_normalization(
                reasoner_source.resolved_config
            ),
            "reasoner_zero_init": bool(
                isinstance(
                    reasoner_source.resolved_config.get("reasoner_geometry"), Mapping
                )
                and reasoner_source.resolved_config["reasoner_geometry"].get(
                    "zero_init", False
                )
            ),
        },
    )
    width = int(executor.config.hidden_size)
    checkpoint_width = int(reasoner_state["query_base"].shape[2])
    expected_family_width = {"qwen3-0.6b": 1024, "qwen3-4b": 2560}[family]
    if width != checkpoint_width or width != expected_family_width:
        raise ValueError(
            "base model, checkpoint tensor width, and resolved model_family differ: "
            f"model={width}, checkpoint={checkpoint_width}, "
            f"expected_{family}={expected_family_width}"
        )
    reasoner = BridgeParallelModel.build_reasoner(
        executor,
        latent_steps=geometry.latent_steps,
        latents_per_step=geometry.latents_per_step,
        loop_steps=geometry.loop_steps,
        num_layers=geometry.num_layers,
        num_heads=_head_count(width),
        dim_feedforward=4 * width,
        bound_scale_init=float(config.bound_scale_init),
        zero_residual_init=bool(config.reasoner_zero_init),
        output_normalization=str(config.reasoner_output_normalization),
        input_mode=config.reasoner_input_mode,
        dropout_p=float(getattr(config, "reasoner_dropout_p", 0.0)),
        dropout_views=1,
    )
    boundary_ids = resolve_boundary_token_ids(tokenizer, THINK_BOUNDARY_TEXT)
    model = BridgeParallelModel(
        executor=executor,
        reasoner=reasoner,
        alignment_capacity=COT_CONTENT_CAPACITY,
        boundary_ids=torch.tensor(boundary_ids, dtype=torch.long, device=device),
        eos_token_id=int(tokenizer.eos_token_id),
        trajectory_max_steps=ANSWER_CAPACITY,
        route1_gradient_checkpointing=False,
    ).to(device)
    from think_bridge.training.reasoner_fixed_state import (
        load_fixed_state,
        restore_fixed_state,
    )

    fixed = load_fixed_state(reasoner_source.weight_path.parent, reasoner_state)
    _restore_component(model.reasoner, reasoner_state, label="Reasoner")
    restore_fixed_state(model.reasoner, fixed)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    from think_bridge.model.feedback_precision import (
        configure_feedback,
        feedback_policy,
    )

    policy = feedback_policy(reasoner_source.resolved_config)
    for name in ("reasoner_compute_dtype", "reasoner_eval_group_size"):
        if getattr(arguments, name, None) is not None:
            policy[name] = getattr(arguments, name)
    configure_feedback(model, policy)
    model.eval()
    # Question prefill and R use bounded HF/BF16 batches even
    # when the already materialized z is passed to vLLM for answer generation.
    from think_bridge.eval.hf_protocol import configure_hf_evaluation

    configure_hf_evaluation(
        model,
        reasoner_batch_size=(
            getattr(arguments, "reasoner_eval_group_size", None) or 64
        ),
    )
    record_reasoner_input(model, reasoner_source.resolved_config)
    return RuntimeBundle(
        model=model,
        tokenizer=tokenizer,
        device=device,
        reasoner_source=reasoner_source,
        reasoner_geometry=geometry,
        model_family=family,
    )
