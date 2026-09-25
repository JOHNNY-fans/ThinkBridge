"""ThinkBridge optimizer backend and exact ZeRO-1 resume artifacts.

This module deliberately keeps configuration and shard-manifest validation
dependency-light.  Torch and DeepSpeed are imported only by the formal runtime
initializer, so local contract validation does not pretend the GPU backend is
installed.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from importlib import metadata
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

from think_bridge.model.artifact_schema import (
    BACKEND_COMPATIBILITY,
    BACKEND_IDENTITY,
    DEEPSPEED_CHECKPOINT_SEMANTICS,
    RANKED_RUNTIME_STATE,
    ZERO1_MANIFEST,
    ZERO1_PARTICIPANT,
    ZERO1_RUNTIME_COMPATIBILITY,
    artifact_header,
)

from think_bridge.model.training_config import TrainingConfig
from think_bridge.model.contract import canonical_json_sha256, file_sha256


_ZERO_SHA256 = "0" * 64
_ZERO1_RESUME_TAG = "resume"
_RANK_PARTICIPANT = re.compile(r"bridge-rank-(\d{5})-participant\.json")
_ZERO1_OPTIMIZER_SHARD = re.compile(
    r"zero_pp_rank_(0|[1-9]\d*)_mp_rank_00_optim_states\.pt"
)


def _run_accumulation_update(
    microbatches: Sequence[Any],
    *,
    synchronization_context: Any,
    forward: Any,
    backward: Any,
    finalize: Any,
) -> tuple[tuple[Any, ...], Any]:
    """Run one optimizer window with exactly one finalization.

    Losses are already normalized by optimizer-window sample denominators, so
    this executor deliberately applies no additional GAS scale.  Every
    microstep runs the same forward/backward path; only the final microstep
    synchronizes replicated gradients (or marks the ZeRO accumulation
    boundary).  ``finalize`` owns clipping and the single optimizer step.
    """

    values = tuple(microbatches)
    if not values:
        raise ValueError("gradient accumulation requires at least one microbatch")
    results: list[Any] = []
    for microstep, microbatch in enumerate(values):
        synchronize_gradients = microstep == len(values) - 1
        with synchronization_context(synchronize_gradients=synchronize_gradients):
            result = forward(microbatch, microstep=microstep)
            backward(
                result,
                synchronize_gradients=synchronize_gradients,
            )
        detach_result = getattr(result, "detached", None)
        results.append(detach_result() if callable(detach_result) else result)
    sealed_results = tuple(results)
    return sealed_results, finalize(sealed_results)


def run_route1_accumulation_update(
    microbatches: Sequence[Any],
    *,
    synchronization_context: Any,
    forward: Any,
    backward: Any,
    finalize: Any,
    prepare: Any = None,
    discard: Any = None,
    max_prepared_microsteps: int = 2,
) -> tuple[tuple[Any, ...], Any]:
    """Run a Route1 window with bounded, same-update generation prefetch.

    Preparation runs on the caller thread. Only detached service requests run
    in background workers; live autograd graphs never cross Python threads.
    Live graphs are bounded by max_prepared_microsteps; none crosses finalize.
    Each consume/backward retains the original DDP/ZeRO accumulation boundary.
    """

    if prepare is not None:
        if type(max_prepared_microsteps) is not int or max_prepared_microsteps < 1:
            raise ValueError("prepared Route1 depth must be a positive integer")
        if discard is None:
            raise ValueError("prepared Route1 accumulation requires cleanup")
        values = tuple(microbatches)
        if not values:
            raise ValueError("gradient accumulation requires at least one microbatch")
        pending = []
        results = []
        current = None
        try:
            next_prepare = 0
            for microstep in range(len(values)):
                # Fill only after the preceding backward released its graph.
                # The request pool must have at least this many ticket slots:
                # filling beyond its bound would block before the first resolve.
                while (
                    next_prepare < len(values)
                    and len(pending) < max_prepared_microsteps
                ):
                    pending.append(
                        prepare(values[next_prepare], microstep=next_prepare)
                    )
                    next_prepare += 1
                current = pending.pop(0)
                synchronize = microstep == len(values) - 1
                with synchronization_context(synchronize_gradients=synchronize):
                    result = forward(current, microstep=microstep)
                    backward(result, synchronize_gradients=synchronize)
                detach_result = getattr(result, "detached", None)
                results.append(detach_result() if callable(detach_result) else result)
                discard(current)
                current = None
                # The next prepare must not keep the preceding live graph alive.
                del result, detach_result
            sealed_results = tuple(results)
            return sealed_results, finalize(sealed_results)
        finally:
            primary_error = sys.exc_info()[1]
            cleanup_errors = []
            for prepared in ([current] if current is not None else []) + pending:
                try:
                    discard(prepared)
                except BaseException as error:
                    cleanup_errors.append(error)
            if cleanup_errors:
                if primary_error is None:
                    raise cleanup_errors[0]
                annotate = getattr(primary_error, "add_note", None)
                if annotate is not None:
                    annotate(f"Route1 pipeline cleanup also failed: {cleanup_errors!r}")

    return _run_accumulation_update(
        microbatches,
        synchronization_context=synchronization_context,
        forward=forward,
        backward=backward,
        finalize=finalize,
    )


@dataclass(frozen=True)
class BridgeBackendIdentity:
    artifact_type: str
    schema_version: int
    optimizer_backend: str
    zero_stage: int
    deepspeed_config_path: str
    deepspeed_config_sha256: str
    deepspeed_version_spec: str
    world_size: int
    route1_local_samples: int
    route1_local_chunk_size: int
    route1_gradient_accumulation_steps: int
    route1_gradient_checkpointing: bool
    route: str = "unspecified"
    trainer_gpu_ids: tuple[int, ...] = ()
    deepspeed_state_semantics_sha256: str = _ZERO_SHA256

    def sha256(self) -> str:
        """Hash the full diagnostic runtime, including resource tuning."""
        return canonical_json_sha256(asdict(self))

    def compatibility_sha256(self) -> str:
        """Hash only state-layout and optimizer-backend semantic identity.

        Physical chunks, device placement, memory budgets, checkpointing and
        GEMM grouping are execution diagnostics.  They do not alter owned
        parameters, optimizer state, or effective reductions and therefore
        cannot invalidate a checkpoint.
        """
        return backend_compatibility_sha256(self)


def _backend_value(identity: Any, name: str, default: Any = None) -> Any:
    if isinstance(identity, Mapping):
        return identity.get(name, default)
    return getattr(identity, name, default)


def _backend_route(identity: Any) -> str:
    route = str(_backend_value(identity, "route", ""))
    return "route1"


def _deepspeed_offload_semantics(value: Any) -> dict[str, Any]:
    """Project an offload section onto optimizer-state placement semantics."""

    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError("DeepSpeed offload semantics must be an object")
    return {
        "device": str(value.get("device", "none")),
        "ratio": float(value.get("ratio", 1.0)),
    }


def deepspeed_checkpoint_semantics(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return only DeepSpeed fields that can alter update/state semantics.

    Queue depths, bucket sizes, overlap, contiguous buffers, wall-clock
    diagnostics and printing cadence are deliberately absent.  Effective
    microbatch/GAS/world geometry is sealed separately by the backend identity.
    """

    if not isinstance(config, Mapping):
        raise ValueError("DeepSpeed checkpoint semantics require an object")
    zero = config.get("zero_optimization", {})
    if not isinstance(zero, Mapping):
        raise ValueError("DeepSpeed zero_optimization must be an object")
    fp16 = config.get("fp16", {})
    bf16 = config.get("bf16", {})
    torch_autocast = config.get("torch_autocast", {})
    if (
        not isinstance(fp16, Mapping)
        or not isinstance(bf16, Mapping)
        or not isinstance(torch_autocast, Mapping)
    ):
        raise ValueError("DeepSpeed precision semantics must be objects")
    safe_modules = torch_autocast.get("lower_precision_safe_modules", ())
    if not isinstance(safe_modules, Sequence) or isinstance(
        safe_modules, (str, bytes, bytearray)
    ):
        raise ValueError(
            "DeepSpeed torch_autocast lower_precision_safe_modules must be a list"
        )
    return {
        **artifact_header(DEEPSPEED_CHECKPOINT_SEMANTICS),
        "zero": {
            "stage": int(zero.get("stage", 0)),
            "legacy_stage1": bool(zero.get("legacy_stage1", False)),
            "zero_hpz_partition_size": int(zero.get("zero_hpz_partition_size", 1)),
            "offload_optimizer": _deepspeed_offload_semantics(
                zero.get("offload_optimizer")
            ),
            "offload_param": _deepspeed_offload_semantics(zero.get("offload_param")),
            "allgather_partitions": bool(zero.get("allgather_partitions", True)),
            "reduce_scatter": bool(zero.get("reduce_scatter", True)),
            "zero_quantized_weights": bool(zero.get("zero_quantized_weights", False)),
            "zero_quantized_nontrainable_weights": bool(
                zero.get("zero_quantized_nontrainable_weights", False)
            ),
            "zero_quantized_gradients": bool(
                zero.get("zero_quantized_gradients", False)
            ),
        },
        "communication_data_type": config.get("communication_data_type"),
        "gradient_predivide_factor": float(
            config.get("gradient_predivide_factor", 1.0)
        ),
        "prescale_gradients": bool(config.get("prescale_gradients", False)),
        "fp16_enabled": fp16.get("enabled", False),
        "bf16_enabled": bf16.get("enabled", False),
        "torch_autocast": {
            "enabled": torch_autocast.get("enabled", False),
            "dtype": torch_autocast.get("dtype"),
            "lower_precision_safe_modules": list(safe_modules),
        },
        "gradient_clipping": config.get("gradient_clipping", 0.0),
        "optimizer": config.get("optimizer"),
        "scheduler": config.get("scheduler"),
    }


def deepspeed_checkpoint_semantics_sha256(config: Mapping[str, Any]) -> str:
    return canonical_json_sha256(deepspeed_checkpoint_semantics(config))


def _backend_deepspeed_state_semantics_sha256(identity: Any) -> Any:
    semantic = _backend_value(identity, "deepspeed_state_semantics_sha256", "")
    if isinstance(semantic, str) and len(semantic) == 64:
        return semantic
    # Old checkpoint identities did not carry the semantic projection.  Keep

    # checkpoints always publish the explicit projection.
    return _backend_value(identity, "deepspeed_config_sha256")


def backend_compatibility_sha256(identity: Any) -> str:
    """Hash only optimizer-state layout and logical distributed reductions."""
    route = _backend_route(identity)
    route_geometry = {
        "route1_local_samples": int(
            _backend_value(identity, "route1_local_samples", -1)
        ),
        "route1_gradient_accumulation_steps": int(
            _backend_value(identity, "route1_gradient_accumulation_steps", -1)
        ),
    }
    return canonical_json_sha256(
        {
            **artifact_header(BACKEND_COMPATIBILITY),
            "optimizer_backend": _backend_value(identity, "optimizer_backend"),
            "zero_stage": int(_backend_value(identity, "zero_stage", -1)),
            "deepspeed_state_semantics_sha256": _backend_deepspeed_state_semantics_sha256(
                identity
            ),
            "world_size": int(_backend_value(identity, "world_size", -1)),
            "route": route,
            **route_geometry,
        }
    )


def checkpoint_backend_compatibility(identity: Any) -> tuple[Any, ...]:
    """Project a runtime/checkpoint identity onto resume-critical geometry."""
    route = _backend_route(identity)
    common = (
        _backend_value(identity, "optimizer_backend"),
        int(_backend_value(identity, "zero_stage", -1)),
        _backend_deepspeed_state_semantics_sha256(identity),
        int(_backend_value(identity, "world_size", -1)),
    )
    return common + (
        route,
        int(_backend_value(identity, "route1_local_samples", -1)),
        int(_backend_value(identity, "route1_gradient_accumulation_steps", -1)),
    )


def _resolve_config_path(value: str, *, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _load_zero1_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"DeepSpeed config does not exist: {path}")
    from think_bridge.arguments.train_args import validate_stage1_deepspeed_config

    if validate_stage1_deepspeed_config(str(path)) != 1:
        raise ValueError(
            "ThinkBridge DeepSpeed config must resolve to exact ZeRO stage 1"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("DeepSpeed config root must be an object")
    zero = value.get("zero_optimization")
    if not isinstance(zero, dict) or zero.get("stage") != 1:
        raise ValueError("ThinkBridge DeepSpeed backend requires ZeRO stage 1")
    offload = zero.get("offload_optimizer", {})
    if not isinstance(offload, dict) or offload.get("device", "none") != "none":
        raise ValueError("ThinkBridge ZeRO stage 1 forbids optimizer offload")
    if "offload_param" in zero:
        param_offload = zero["offload_param"]
        if (
            not isinstance(param_offload, dict)
            or param_offload.get("device", "none") != "none"
        ):
            raise ValueError("ThinkBridge ZeRO stage 1 forbids parameter offload")
    for precision in ("fp16", "bf16"):
        section = value.get(precision, {})
        if not isinstance(section, dict) or section.get("enabled", False) is not False:
            raise ValueError(
                "ThinkBridge keeps FP32 owner/Adam state; native DeepSpeed "
                "fp16/bf16 parameter modes must be disabled"
            )
    if value.get("gradient_clipping") != 0.0:
        raise ValueError("ThinkBridge performs owner clipping outside DeepSpeed")
    if "optimizer" in value or "scheduler" in value:
        raise ValueError("ThinkBridge supplies the exact owner optimizer and scheduler")
    return value


def resolve_backend_identity(
    config: TrainingConfig, *, project_root: Path, route: str | None = None
) -> BridgeBackendIdentity:
    if config.optimizer_backend == "replicated_ddp":
        if config.zero_stage != 0 or config.deepspeed_config is not None:
            raise ValueError("replicated DDP cannot carry a DeepSpeed/ZeRO identity")
        path_text = ""
        digest = _ZERO_SHA256
        state_semantics_digest = _ZERO_SHA256
    elif config.optimizer_backend == "deepspeed_zero1":
        if config.zero_stage != 1 or not config.deepspeed_config:
            raise ValueError("DeepSpeed backend requires exact ZeRO stage 1 config")
        path = _resolve_config_path(config.deepspeed_config, project_root=project_root)
        zero1_config = _load_zero1_config(path)
        path_text = str(path)
        digest = file_sha256(path)
        state_semantics_digest = deepspeed_checkpoint_semantics_sha256(zero1_config)
    else:
        raise ValueError(
            f"unsupported ThinkBridge optimizer backend: {config.optimizer_backend}"
        )
    if route not in {"route1", None}:
        raise ValueError("runtime backend route is invalid")
    world_size = int(config.route1_world_size)
    trainer_gpu_ids = tuple(config.route1_trainer_gpu_ids)
    return BridgeBackendIdentity(
        artifact_type=BACKEND_IDENTITY,
        schema_version=1,
        optimizer_backend=config.optimizer_backend,
        zero_stage=int(config.zero_stage),
        deepspeed_config_path=path_text,
        deepspeed_config_sha256=digest,
        deepspeed_version_spec=config.deepspeed_version_spec,
        world_size=world_size,
        route=str(route or "unspecified"),
        trainer_gpu_ids=trainer_gpu_ids,
        route1_local_samples=int(config.route1_local_samples),
        route1_local_chunk_size=int(config.route1_local_chunk_size),
        route1_gradient_accumulation_steps=int(
            config.route1_gradient_accumulation_steps
        ),
        route1_gradient_checkpointing=bool(config.route1_gradient_checkpointing),
        deepspeed_state_semantics_sha256=state_semantics_digest,
    )


def bind_bridge_zero1_owner_adapter(ds_optimizer: Any, *, optimizer: Any) -> Any:
    """Bind the exact private ZeRO-1 owner protocol Bridge consumes.

    Installed versions are diagnostic evidence only.  Structural capabilities
    and owner/optimizer identity are the runtime admission authority.
    """

    from think_bridge.train.deepspeed_zero1_adapter import (
        DeepSpeedZero1OwnerAdapter,
    )

    groups = getattr(optimizer, "param_groups", None)
    if (
        not isinstance(groups, Sequence)
        or isinstance(groups, (str, bytes, bytearray))
        or not groups
    ):
        raise RuntimeError(
            "ThinkBridge ZeRO base optimizer lacks nonempty parameter groups"
        )
    adapter = DeepSpeedZero1OwnerAdapter.bind(
        ds_optimizer,
        optimizer=optimizer,
        group_count=len(groups),
    )
    if not callable(getattr(ds_optimizer, "scaled_global_norm", None)):
        raise RuntimeError(
            "ThinkBridge ZeRO-1 missing required owner capabilities: "
            "('scaled_global_norm',)"
        )
    return adapter


def _normalize_installed_deepspeed_version(value: Any) -> str:
    """Normalize installed-version evidence without imposing an allow-list."""

    raw = str(value).strip()
    try:
        from packaging.version import Version
    except ImportError:
        # Dependency-light contract tests do not install packaging.  DeepSpeed
        # releases observed through importlib.metadata already use this
        # canonical numeric-release form; retain any local/pre-release suffix
        # while normalizing case, a leading ``v``, and leading zeroes.
        match = re.fullmatch(r"[vV]?(\d+(?:\.\d+){1,3})([A-Za-z0-9.+_-]*)", raw)
        if match is None:
            raise RuntimeError(
                f"DeepSpeed reported an invalid installed version: {value!r}"
            )
        release = ".".join(str(int(part)) for part in match.group(1).split("."))
        normalized = release + match.group(2).lower()
    else:
        try:
            normalized = str(Version(raw))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"DeepSpeed reported an invalid installed version: {value!r}"
            ) from exc
    if not normalized:
        raise RuntimeError("DeepSpeed reported an empty installed version")
    return normalized


def _zero1_runtime_compatibility(installed_version: Any) -> dict[str, Any]:
    """Return resume-only runtime evidence sealed with ZeRO optimizer shards."""

    return {
        **artifact_header(ZERO1_RUNTIME_COMPATIBILITY),
        "deepspeed_version": _normalize_installed_deepspeed_version(installed_version),
    }


def _validate_zero1_runtime_compatibility(
    payload: Any, *, expected: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    required = {"artifact_type", "schema_version", "deepspeed_version"}
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("ZeRO runtime compatibility schema mismatch")
    normalized = _zero1_runtime_compatibility(payload["deepspeed_version"])
    if (
        payload["artifact_type"] != ZERO1_RUNTIME_COMPATIBILITY
        or payload["schema_version"] != 1
    ):
        raise ValueError("ZeRO runtime compatibility schema mismatch")
    if str(payload["deepspeed_version"]) != normalized["deepspeed_version"]:
        raise ValueError("ZeRO runtime compatibility version is not normalized")
    if expected is not None:
        expected_normalized = _validate_zero1_runtime_compatibility(expected)
        if normalized != expected_normalized:
            raise ValueError(
                "resume DeepSpeed installed version differs from the sealed "
                "ZeRO runtime compatibility"
            )
    return normalized


def resolved_deepspeed_config(
    config: TrainingConfig, *, project_root: Path, route: str
) -> dict[str, Any]:
    identity = resolve_backend_identity(config, project_root=project_root, route=route)
    if identity.optimizer_backend != "deepspeed_zero1":
        raise ValueError("resolved DeepSpeed config requested for replicated DDP")
    value = _load_zero1_config(Path(identity.deepspeed_config_path))
    micro = config.route1_local_samples
    gas = config.route1_gradient_accumulation_steps
    global_batch = config.route1_optimizer_global_batch
    value["train_micro_batch_size_per_gpu"] = int(micro)
    value["gradient_accumulation_steps"] = int(gas)
    value["train_batch_size"] = int(global_batch)
    return value


def _validate_zero1_rank_artifacts(root: Path, *, world_size: int) -> list[int]:
    """Prove that every sealed DP rank wrote its own resume artifacts."""

    expected_ranks = set(range(int(world_size)))
    resume_root = root / _ZERO1_RESUME_TAG
    participants: dict[int, list[Path]] = {}
    optimizer_shards: dict[int, list[Path]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        participant_match = _RANK_PARTICIPANT.fullmatch(path.name)
        participant_like = path.name.startswith("bridge-rank-") and path.name.endswith(
            "-participant.json"
        )
        if participant_like and participant_match is None:
            raise ValueError(
                f"ZeRO participant filename does not match the sealed schema: {path.name}"
            )
        if participant_match is not None:
            file_rank = int(participant_match.group(1))
            participants.setdefault(file_rank, []).append(path)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"ZeRO participant payload is unreadable: {path}"
                ) from exc
            required = {"artifact_type", "schema_version", "rank", "world_size"}
            if not isinstance(payload, dict) or set(payload) != required:
                raise ValueError("ZeRO participant payload schema mismatch")
            if (
                payload["artifact_type"] != ZERO1_PARTICIPANT
                or payload["schema_version"] != 1
            ):
                raise ValueError("ZeRO participant payload schema version mismatch")
            if (
                type(payload["rank"]) is not int
                or type(payload["world_size"]) is not int
            ):
                raise ValueError("ZeRO participant rank/world size must be integers")
            if payload["rank"] != file_rank:
                raise ValueError(
                    "ZeRO participant filename rank differs from payload rank"
                )
            if payload["world_size"] != int(world_size):
                raise ValueError("ZeRO participant payload world size mismatch")

        optimizer_like = path.name.endswith("_optim_states.pt")
        optimizer_match = _ZERO1_OPTIMIZER_SHARD.fullmatch(path.name)
        if optimizer_like and optimizer_match is None:
            raise ValueError(
                f"ZeRO optimizer shard filename does not match the sealed schema: {path.name}"
            )
        if optimizer_match is not None:
            rank = int(optimizer_match.group(1))
            optimizer_shards.setdefault(rank, []).append(path)

    duplicate_participants = {
        rank: paths for rank, paths in participants.items() if len(paths) != 1
    }
    if duplicate_participants:
        raise ValueError("ZeRO checkpoint contains duplicate participant rank files")
    duplicate_optimizer_shards = {
        rank: paths for rank, paths in optimizer_shards.items() if len(paths) != 1
    }
    if duplicate_optimizer_shards:
        raise ValueError("ZeRO checkpoint contains duplicate optimizer shard ranks")
    if any(paths[0].parent != resume_root for paths in participants.values()):
        raise ValueError("ZeRO participant file is outside the sealed resume tag")
    if any(paths[0].parent != resume_root for paths in optimizer_shards.values()):
        raise ValueError("ZeRO optimizer shard is outside the sealed resume tag")
    if set(participants) != expected_ranks:
        raise ValueError(
            "ZeRO checkpoint does not prove every rank participated in save"
        )
    if set(optimizer_shards) != expected_ranks:
        raise ValueError(
            "ZeRO checkpoint does not contain exactly one optimizer shard for every rank"
        )
    return sorted(expected_ranks)


def build_sharded_resume_manifest(
    root: Path,
    *,
    world_size: int,
    runtime_compatibility: Mapping[str, Any],
    root_relative: str = "zero1",
) -> dict[str, Any]:
    root = Path(root).resolve()
    if world_size <= 0 or not root.is_dir():
        raise ValueError("ZeRO sharded resume root/world size is invalid")
    files: list[dict[str, Any]] = []
    ranks = _validate_zero1_rank_artifacts(root, world_size=world_size)
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        files.append(
            {
                "path": relative,
                "sha256": file_sha256(path),
                "bytes": int(path.stat().st_size),
            }
        )
    files_sha256 = canonical_json_sha256(files)
    runtime_compatibility = _validate_zero1_runtime_compatibility(runtime_compatibility)
    return {
        **artifact_header(ZERO1_MANIFEST),
        "root_relative": str(root_relative),
        "world_size": int(world_size),
        "participating_ranks": ranks,
        "files": files,
        "files_sha256": files_sha256,
        "runtime_compatibility": runtime_compatibility,
        "artifact_sha256": canonical_json_sha256(
            {
                **artifact_header(ZERO1_MANIFEST),
                "world_size": int(world_size),
                "participating_ranks": ranks,
                "files_sha256": files_sha256,
                "runtime_compatibility": runtime_compatibility,
            }
        ),
    }


def validate_sharded_resume_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_world_size: int,
    checkpoint_dir: Path,
    sealed_file_ledger: Mapping[str, str] | None = None,
) -> Path:
    """Validate a ZeRO manifest, hashing files unless a trusted ledger is held.

    ``sealed_file_ledger`` is an internal capability: callers may supply it only
    after the checkpoint policy has either computed that ledger from the files
    in this boundary or received its rank-0 validated ``SealedCheckpoint``.
    """

    required = {
        "artifact_type",
        "schema_version",
        "root_relative",
        "world_size",
        "participating_ranks",
        "files",
        "files_sha256",
        "runtime_compatibility",
        "artifact_sha256",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != required:
        raise ValueError("ZeRO sharded resume manifest schema mismatch")
    if (
        manifest["artifact_type"] != ZERO1_MANIFEST
        or manifest["schema_version"] != 1
        or int(manifest["world_size"]) != int(expected_world_size)
    ):
        raise ValueError("ZeRO sharded resume world size/schema mismatch")
    expected_ranks = list(range(int(expected_world_size)))
    if manifest["participating_ranks"] != expected_ranks:
        raise ValueError("ZeRO sharded resume is missing rank participation")
    files = manifest["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("ZeRO sharded resume file ledger is empty")
    if manifest["files_sha256"] != canonical_json_sha256(files):
        raise ValueError("ZeRO sharded resume file-ledger hash mismatch")
    relative = Path(str(manifest["root_relative"]))
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != "zero1"
    ):
        raise ValueError("ZeRO sharded resume root escaped its checkpoint directory")
    checkpoint_root = Path(checkpoint_dir).resolve(strict=True)
    root = (checkpoint_root / relative).resolve(strict=True)
    try:
        root.relative_to(checkpoint_root)
    except ValueError as exc:
        raise ValueError(
            "ZeRO sharded resume root escaped its checkpoint directory"
        ) from exc
    sealed_zero_paths: set[str] | None = None
    if sealed_file_ledger is not None:
        if not isinstance(sealed_file_ledger, Mapping):
            raise ValueError("sealed checkpoint file ledger is malformed")
        sealed_zero_paths = {
            str(path).removeprefix("zero1/")
            for path in sealed_file_ledger
            if isinstance(path, str) and path.startswith("zero1/")
        }
    declared_paths: set[str] = set()
    for row in files:
        if not isinstance(row, dict) or set(row) != {"path", "sha256", "bytes"}:
            raise ValueError("ZeRO sharded resume file row mismatch")
        row_value = str(row["path"])
        row_relative = Path(row_value)
        if (
            row_relative.is_absolute()
            or ".." in row_relative.parts
            or row_relative.as_posix() != row_value
            or row_value in declared_paths
        ):
            raise ValueError("ZeRO sharded resume file path escaped or duplicated")
        declared_paths.add(row_value)
        path = (root / row_relative).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("ZeRO sharded resume file escaped its root") from exc
        if not path.is_file() or path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"ZeRO sharded resume file hash mismatch: {path}")
        observed_sha256 = (
            sealed_file_ledger.get(f"zero1/{row_value}")
            if sealed_file_ledger is not None
            else file_sha256(path)
        )
        if observed_sha256 != row["sha256"]:
            raise ValueError(f"ZeRO sharded resume file hash mismatch: {path}")
    actual_paths = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if actual_paths != declared_paths:
        raise ValueError("ZeRO sharded resume file set differs from its ledger")
    if sealed_zero_paths is not None and sealed_zero_paths != declared_paths:
        raise ValueError("ZeRO manifest differs from sealed checkpoint file ledger")
    _validate_zero1_rank_artifacts(root, world_size=expected_world_size)
    expected_artifact = canonical_json_sha256(
        {
            "artifact_type": manifest["artifact_type"],
            "schema_version": manifest["schema_version"],
            "world_size": int(manifest["world_size"]),
            "participating_ranks": manifest["participating_ranks"],
            "files_sha256": manifest["files_sha256"],
            "runtime_compatibility": _validate_zero1_runtime_compatibility(
                manifest["runtime_compatibility"]
            ),
        }
    )
    if manifest["artifact_sha256"] != expected_artifact:
        raise ValueError("ZeRO sharded resume artifact hash mismatch")
    return root


def broadcast_checkpoint_preflight(
    distributed: Any,
    *,
    rank: int,
    local_conflict: str | None,
) -> str | None:
    """Make a rank-0 namespace conflict fail identically on every rank."""

    if not distributed.is_initialized():
        return local_conflict
    values = [local_conflict if int(rank) == 0 else None]
    distributed.broadcast_object_list(values, src=0)
    value = values[0]
    return None if value is None else str(value)


def broadcast_rank0_result(
    distributed: Any,
    *,
    rank: int,
    local_result: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Broadcast one rank-0 post-collective result before any rank returns."""

    if not distributed.is_initialized():
        return local_result
    values = [local_result if int(rank) == 0 else None]
    distributed.broadcast_object_list(values, src=0)
    value = values[0]
    if value is not None and not isinstance(value, Mapping):
        raise RuntimeError("rank-0 checkpoint result is not a mapping")
    return value


def run_rank0_control(distributed: Any, *, rank: int, action: Any) -> Mapping[str, Any]:
    """Finish rank-0 filesystem work on every rank, including failed writes."""
    local_result = None
    if int(rank) == 0:
        try:
            local_result = {"error": None, "value": action()}
        except Exception as exc:
            local_result = {"error": f"{type(exc).__name__}: {exc}", "value": None}
    result = broadcast_rank0_result(distributed, rank=rank, local_result=local_result)
    if not isinstance(result, Mapping) or result.get("error") is not None:
        raise RuntimeError(f"rank-0 control event failed: {result}")
    return result


def collect_rank_failures(
    distributed: Any,
    *,
    world_size: int,
    local_failure: str | None,
) -> tuple[str, ...]:
    """Collect post-collective failures so every rank takes the same exit."""

    if not distributed.is_initialized():
        return () if local_failure is None else (str(local_failure),)
    gathered: list[str | None] = [None] * int(world_size)
    distributed.all_gather_object(gathered, local_failure)
    return tuple(str(value) for value in gathered if value is not None)


class BridgeTrainingBackend:
    """Small owner-preserving facade over replicated DDP or DeepSpeed ZeRO-1."""

    def __init__(
        self,
        *,
        identity: BridgeBackendIdentity,
        train_model: Any,
        optimizer: Any,
        scheduler: Any,
        engine: Any | None,
        rank: int,
        world_size: int,
        zero1_owner_adapter: Any | None = None,
        fresh_owner_parameter_count: int | None = None,
        fresh_owner_group_count: int | None = None,
        operational_evidence: Mapping[str, Any] | None = None,
    ) -> None:
        self.identity = identity
        self.train_model = train_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.engine = engine
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.zero1_owner_adapter = zero1_owner_adapter
        self.fresh_owner_parameter_count = fresh_owner_parameter_count
        self.fresh_owner_group_count = fresh_owner_group_count
        self.operational_evidence = dict(operational_evidence or {})

    @property
    def is_zero1(self) -> bool:
        return self.engine is not None

    def zero1_runtime_compatibility(self) -> dict[str, str]:
        """Return resume-only evidence for the installed ZeRO implementation."""

        if not self.is_zero1:
            raise RuntimeError(
                "ZeRO runtime compatibility requested for replicated DDP"
            )
        return _zero1_runtime_compatibility(
            self.operational_evidence.get("deepspeed_version")
        )

    @property
    def integrity_optimizer(self) -> Any:
        """Optimizer whose FP32 state is authoritative at integrity audits."""

        if not self.is_zero1:
            return self.optimizer
        if self.zero1_owner_adapter is None:
            raise RuntimeError("ThinkBridge ZeRO-1 owner adapter is not bound")
        return self.zero1_owner_adapter.optimizer

    def pristine_owner_capability(self) -> dict[str, int]:
        """Describe fresh Adam ownership through the active backend adapter.

        DeepSpeed ZeRO-1 is allowed to replace the public optimizer parameter
        groups with rank partitions/master tensors.  The exact pre-wrap owner
        cardinality is therefore sealed before ``deepspeed.initialize`` while
        Adam moments and published gradient groups are inspected through the
        capability-bound base optimizer/ZeRO adapter.
        """

        groups = getattr(self.integrity_optimizer, "param_groups", None)
        state = getattr(self.integrity_optimizer, "state", None)
        if (
            not isinstance(groups, Sequence)
            or isinstance(groups, (str, bytes, bytearray))
            or not isinstance(state, Mapping)
            or isinstance(self.fresh_owner_parameter_count, bool)
            or not isinstance(self.fresh_owner_parameter_count, int)
            or self.fresh_owner_parameter_count <= 0
            or isinstance(self.fresh_owner_group_count, bool)
            or not isinstance(self.fresh_owner_group_count, int)
            or self.fresh_owner_group_count <= 0
            or len(groups) != self.fresh_owner_group_count
        ):
            raise RuntimeError(
                "ThinkBridge fresh owner optimizer capability is invalid"
            )
        published = 0
        if self.is_zero1:
            adapter = self.zero1_owner_adapter
            if (
                adapter is None
                or int(adapter.group_count) != self.fresh_owner_group_count
            ):
                raise RuntimeError(
                    "ThinkBridge fresh ZeRO owner group capability differs"
                )
            gradients = adapter.averaged_gradients
            if not isinstance(gradients, Mapping):
                raise RuntimeError(
                    "ThinkBridge fresh ZeRO gradient capability is invalid"
                )
            published = len(gradients)
        else:
            observed_parameters = [
                parameter for group in groups for parameter in group.get("params", ())
            ]
            if len({id(parameter) for parameter in observed_parameters}) != len(
                observed_parameters
            ):
                raise RuntimeError(
                    "ThinkBridge fresh replicated owner parameters overlap"
                )
            if len(observed_parameters) != self.fresh_owner_parameter_count:
                raise RuntimeError(
                    "ThinkBridge fresh replicated owner parameters differ"
                )
        return {
            "parameter_group_count": int(self.fresh_owner_group_count),
            "owner_parameter_count": int(self.fresh_owner_parameter_count),
            "optimizer_state_entries": len(state),
            "published_gradient_groups": int(published),
        }

    def zero_grad(self) -> None:
        if self.is_zero1:
            self.engine.zero_grad()
        else:
            self.optimizer.zero_grad(set_to_none=True)

    def set_gradient_accumulation_boundary(self, value: bool) -> None:
        if not self.is_zero1:
            return
        setter = getattr(self.engine, "set_gradient_accumulation_boundary", None)
        if not callable(setter):
            raise RuntimeError(
                "DeepSpeed engine lacks the GAS-boundary accumulation API"
            )
        setter(bool(value))

    def backward(self, loss: Any, *, synchronize_gradients: bool = True) -> None:
        """Backpropagate one route microstep contribution.

        DDP synchronizes only the final GAS microstep; ZeRO-1 receives the same
        explicit boundary.  Callers either pass an already normalized
        optimizer-window contribution or apply one backend-native gradient
        scale before clipping; DeepSpeed must never add a second GAS division.
        """

        if self.is_zero1:
            self.set_gradient_accumulation_boundary(synchronize_gradients)
            # Each route loss is already normalized by its optimizer-window
            # sample denominator.  Keep DeepSpeed from applying a second GAS
            # scale.
            self.engine.backward(loss, scale_wrt_gas=False)
        else:
            loss.backward()

    def synchronization_context(self, *, synchronize_gradients: bool) -> Any:
        """Wrap DDP forward+backward so only the final GAS microstep reduces."""

        if self.is_zero1 or synchronize_gradients:
            return nullcontext()
        no_sync = getattr(self.train_model, "no_sync", None)
        if no_sync is None:
            if self.world_size == 1:
                return nullcontext()
            raise RuntimeError("replicated multi-rank backend lacks DDP no_sync")
        if not callable(no_sync):
            raise RuntimeError("replicated DDP no_sync surface is not callable")
        return no_sync()

    def _zero1_gradient_tensors(self) -> list[Any]:
        adapter = self.zero1_owner_adapter
        averaged = None if adapter is None else adapter.averaged_gradients
        if not isinstance(averaged, Mapping) or not averaged:
            raise RuntimeError("ZeRO-1 did not publish partitioned gradients")
        if set(averaged) != set(range(int(adapter.group_count))):
            raise RuntimeError("ZeRO-1 partitioned gradient groups are incomplete")
        gradients: list[Any] = []
        for group_index in sorted(averaged):
            group = averaged[group_index]
            if (
                group is None
                or not isinstance(group, Sequence)
                or isinstance(group, (str, bytes, bytearray))
                or not group
                or any(value is None for value in group)
            ):
                raise RuntimeError("ZeRO-1 partitioned gradient group is missing")
            gradients.extend(group)
        if not gradients:
            raise RuntimeError("ZeRO-1 partitioned gradient set is empty")
        return gradients

    def snapshot_owner_gradients(self, parameters: Sequence[Any]) -> tuple[Any, ...]:
        """Clone the active owner's backend-native gradient representation."""

        import torch

        gradients = (
            self._zero1_gradient_tensors()
            if self.is_zero1
            else [parameter.grad for parameter in parameters]
        )
        if not gradients or any(gradient is None for gradient in gradients):
            raise RuntimeError("component audit owner gradients are incomplete")
        snapshots = tuple(gradient.detach().float().clone() for gradient in gradients)
        if not snapshots or any(
            not bool(torch.isfinite(value).all()) for value in snapshots
        ):
            raise FloatingPointError("component audit owner gradients are non-finite")
        return snapshots

    def component_gradient_metrics(
        self,
        gold: Sequence[Any],
        causal: Sequence[Any] | None,
    ) -> dict[str, float | None]:
        """Compute exact full-owner norms/cosine from DDP or ZeRO-1 shards."""

        import torch
        import torch.distributed as distributed

        if not gold:
            raise RuntimeError("gold component gradient snapshot is empty")
        device = gold[0].device
        gold_sq = sum(
            (value.detach().double().square().sum() for value in gold),
            torch.zeros((), dtype=torch.float64, device=device),
        )
        if causal is None:
            statistics = torch.stack(
                (gold_sq, torch.zeros_like(gold_sq), torch.zeros_like(gold_sq))
            )
        else:
            if len(gold) != len(causal) or any(
                left.shape != right.shape
                for left, right in zip(gold, causal, strict=True)
            ):
                raise RuntimeError("component gradient shard schemas differ")
            causal_sq = sum(
                (value.detach().double().square().sum() for value in causal),
                torch.zeros((), dtype=torch.float64, device=device),
            )
            dot = sum(
                (
                    left.detach().double().mul(right.detach().double()).sum()
                    for left, right in zip(gold, causal, strict=True)
                ),
                torch.zeros((), dtype=torch.float64, device=device),
            )
            statistics = torch.stack((gold_sq, causal_sq, dot))
        if self.is_zero1 and distributed.is_initialized():
            distributed.all_reduce(statistics, op=distributed.ReduceOp.SUM)
        gold_norm = statistics[0].clamp_min(0.0).sqrt()
        if causal is None:
            return {
                "grad_norm_gold": float(gold_norm),
                "grad_norm_causal": None,
                "grad_cosine": None,
            }
        causal_norm = statistics[1].clamp_min(0.0).sqrt()
        denominator = gold_norm * causal_norm
        cosine = (
            None if float(denominator) == 0.0 else float(statistics[2] / denominator)
        )
        return {
            "grad_norm_gold": float(gold_norm),
            "grad_norm_causal": float(causal_norm),
            "grad_cosine": cosine,
        }

    def owner_gradients_finite(self, parameters: Sequence[Any]) -> bool:
        import torch
        import torch.distributed as distributed

        owner_parameters = tuple(parameters)
        reference = next(
            (value for value in owner_parameters if torch.is_tensor(value)),
            None,
        )
        if reference is None:
            return False

        gradients: list[Any] = []
        complete = bool(owner_parameters)
        if self.is_zero1:
            adapter = self.zero1_owner_adapter
            averaged = None if adapter is None else adapter.averaged_gradients
            complete = isinstance(averaged, Mapping) and bool(averaged)
            if complete:
                if set(averaged) != set(range(int(adapter.group_count))):
                    complete = False
                for group in averaged.values():
                    if (
                        group is None
                        or not isinstance(group, Sequence)
                        or isinstance(group, (str, bytes, bytearray))
                        or not group
                    ):
                        complete = False
                        continue
                    for gradient in group:
                        if gradient is None or not torch.is_tensor(gradient):
                            complete = False
                        else:
                            gradients.append(gradient)
        else:
            for parameter in owner_parameters:
                gradient = parameter.grad
                if gradient is None or not torch.is_tensor(gradient):
                    complete = False
                else:
                    gradients.append(gradient)

        # Launch per-tensor checks without reading any scalar on the host, then
        # merge them on device.  The one reduced scalar below is the sole
        # host/collective decision for this owner-gradient boundary.
        finite_flags = [reference.detach().new_tensor(bool(complete), dtype=torch.bool)]
        finite_flags.extend(
            torch.isfinite(gradient.detach()).all().to(device=reference.device)
            for gradient in gradients
        )
        reduced = torch.stack(finite_flags).all().to(dtype=torch.int32)
        if distributed.is_initialized():
            distributed.all_reduce(reduced, op=distributed.ReduceOp.MIN)
        return bool(reduced.item())

    def scale_owner_gradients(
        self, parameters: Sequence[Any], *, coefficient: float
    ) -> None:
        """Scale the backend-native active-owner gradients exactly once."""

        scale = float(coefficient)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("owner gradient scale must be finite and positive")
        if self.is_zero1:
            gradients = self._zero1_gradient_tensors()
        else:
            gradients = [
                parameter.grad for parameter in parameters if parameter.grad is not None
            ]
            if not gradients:
                raise RuntimeError("replicated owner gradient set is empty")
        for gradient in gradients:
            gradient.mul_(scale)

    def clip_owner_gradients(
        self,
        parameters: Sequence[Any],
        *,
        max_norm: float,
        allow_norm_overflow_recovery: bool = True,
    ) -> tuple[float, float]:
        """Clip the active owner in its backend-native gradient storage."""

        maximum = float(max_norm)
        if not math.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("owner max gradient norm must be finite and positive")
        if self.is_zero1:
            gradients = self._zero1_gradient_tensors()
            if self.zero1_owner_adapter is None:
                raise RuntimeError("ThinkBridge ZeRO-1 owner adapter is not bound")
            loss_scale = float(self.zero1_owner_adapter.loss_scale)
            if loss_scale != 1.0:
                raise RuntimeError(
                    "ThinkBridge ZeRO-1 owner clipping requires unit loss scale"
                )
            preclip = self.zero1_owner_adapter.scaled_global_norm()
            if not math.isfinite(preclip):
                raise FloatingPointError("ZeRO-1 global gradient norm is non-finite")
            coefficient = min(1.0, maximum / (preclip + 1e-6))
            for gradient in gradients:
                gradient.mul_(coefficient)
            return preclip, preclip * coefficient

        import torch

        if not parameters:
            raise RuntimeError("replicated owner parameter set is empty")
        try:
            preclip = float(
                torch.nn.utils.clip_grad_norm_(
                    parameters,
                    max_norm=maximum,
                    error_if_nonfinite=True,
                )
            )
        except RuntimeError as exc:
            if not allow_norm_overflow_recovery:
                raise
            # With error_if_nonfinite=True PyTorch rejects the norm BEFORE
            # mutating any gradient. A finite FP32 vector can nevertheless
            # overflow its FP32 norm reduction (e.g. [1e20, 1e20]). Preserve
            # the normal clipping path exactly, and retry only that overflow.
            if "non-finite" not in str(exc):
                raise
            gradients = [p.grad for p in parameters if p.grad is not None]
            if not gradients or any(
                gradient.dtype != torch.float32
                or not bool(torch.isfinite(gradient).all())
                for gradient in gradients
            ):
                raise
            norms = torch.stack(
                [gradient.detach().double().norm(2) for gradient in gradients]
            )
            preclip = float(norms.norm(2))
            if not math.isfinite(preclip):
                raise FloatingPointError(
                    "FP64 owner gradient norm is non-finite"
                ) from exc
            coefficient = min(1.0, maximum / (preclip + 1.0e-6))
            for gradient in gradients:
                # The coefficient can be an FP32 subnormal even though the
                # clipped result is normal. Multiply in FP64 before copying
                # back so flush-to-zero cannot erase a finite large gradient.
                gradient.copy_(gradient.detach().double().mul_(coefficient))
        # PyTorch applies this exact finite coefficient in clip_grad_norm_.
        # Derive the diagnostic norm from its returned preclip norm rather
        # than rereading every gradient after the real in-place clip.
        coefficient = min(1.0, maximum / (preclip + 1.0e-6))
        return preclip, preclip * coefficient

    def step(self) -> None:
        if self.is_zero1:
            self.engine.step()
        else:
            self.optimizer.step()
            self.scheduler.step()

    def collect_rank_runtime_state(
        self, runtime_state: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Seal per-rank RNG/sampler state at a checkpoint control event.

        Object collection is intentionally confined to checkpoint publication;
        Route1's training hot path gathers only precompiled tensors.
        """

        import torch.distributed as distributed

        local = {"rank": self.rank, **dict(runtime_state)}
        if distributed.is_initialized():
            gathered: list[dict[str, Any] | None] = [None] * self.world_size
            distributed.all_gather_object(gathered, local)
        else:
            gathered = [local]
        if any(not isinstance(value, Mapping) for value in gathered):
            raise RuntimeError("checkpoint rank runtime collection is incomplete")
        states = sorted(
            (dict(value) for value in gathered), key=lambda row: row["rank"]
        )
        if [int(row["rank"]) for row in states] != list(range(self.world_size)):
            raise RuntimeError("checkpoint rank runtime identities are incomplete")
        return {
            **artifact_header(RANKED_RUNTIME_STATE),
            "world_size": self.world_size,
            "states": states,
        }

    def local_rank_runtime_state(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if (
            not isinstance(payload, Mapping)
            or payload.get("artifact_type") != RANKED_RUNTIME_STATE
            or payload.get("schema_version") != 1
            or int(payload.get("world_size", -1)) != self.world_size
            or not isinstance(payload.get("states"), list)
            or len(payload["states"]) != self.world_size
        ):
            raise ValueError("checkpoint ranked runtime state schema mismatch")
        state = payload["states"][self.rank]
        if not isinstance(state, Mapping) or int(state.get("rank", -1)) != self.rank:
            raise ValueError("checkpoint local rank runtime state is missing")
        return dict(state)

    def save_checkpoint(self, checkpoint_building: Path) -> dict[str, Any] | None:
        import torch.distributed as distributed

        if not self.is_zero1:
            return None

        root = Path(checkpoint_building) / "zero1"
        tag = _ZERO1_RESUME_TAG
        runtime_compatibility = self.zero1_runtime_compatibility()
        root_failure: str | None = None
        if self.rank == 0:
            try:
                root.parent.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                root_failure = f"ZeRO root creation: {type(exc).__name__}: {exc}"
        root_failure = broadcast_checkpoint_preflight(
            distributed,
            rank=self.rank,
            local_conflict=root_failure,
        )
        if root_failure is not None:
            raise RuntimeError(root_failure)
        # DeepSpeed requires every data-parallel rank to enter this call.
        self.engine.save_checkpoint(
            str(root),
            tag=tag,
            client_state={
                "bridge_backend_identity": asdict(self.identity),
                "bridge_backend_compatibility_sha256": (
                    self.identity.compatibility_sha256()
                ),
                "bridge_zero1_runtime_compatibility": runtime_compatibility,
                "bridge_zero1_runtime_compatibility_sha256": (
                    canonical_json_sha256(runtime_compatibility)
                ),
            },
            exclude_frozen_parameters=True,
        )
        participant = root / tag / f"bridge-rank-{self.rank:05d}-participant.json"
        participant_failure: str | None = None
        try:
            participant.parent.mkdir(parents=True, exist_ok=True)
            participant.write_text(
                json.dumps(
                    {
                        **artifact_header(ZERO1_PARTICIPANT),
                        "rank": self.rank,
                        "world_size": self.world_size,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            participant_failure = f"ZeRO participant save: {type(exc).__name__}: {exc}"
        participant_failures = collect_rank_failures(
            distributed,
            world_size=self.world_size,
            local_failure=participant_failure,
        )
        if participant_failures:
            raise RuntimeError("; ".join(participant_failures))
        rank0_result: dict[str, Any] | None = None
        if self.rank == 0:
            try:
                rank0_result = {
                    "error": None,
                    "manifest": build_sharded_resume_manifest(
                        root,
                        world_size=self.world_size,
                        runtime_compatibility=runtime_compatibility,
                        root_relative="zero1",
                    ),
                }
            except Exception as exc:
                rank0_result = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "manifest": None,
                }
        result = broadcast_rank0_result(
            distributed, rank=self.rank, local_result=rank0_result
        )
        if not isinstance(result, Mapping):
            raise RuntimeError("ZeRO checkpoint did not publish a rank-0 result")
        if result.get("error") is not None:
            raise RuntimeError(
                f"ZeRO checkpoint shard-manifest publication failed: {result['error']}"
            )
        manifest = result.get("manifest")
        if not isinstance(manifest, Mapping):
            raise RuntimeError("ZeRO checkpoint shard manifest is missing")
        return dict(manifest)

    def load_checkpoint(
        self,
        checkpoint_dir: Path,
        manifest: Mapping[str, Any],
        *,
        sealed_checkpoint: Any | None = None,
    ) -> None:
        if not self.is_zero1:
            raise ValueError("sharded checkpoint requested for replicated DDP")
        expected_runtime_compatibility = self.zero1_runtime_compatibility()
        _validate_zero1_runtime_compatibility(
            manifest.get("runtime_compatibility")
            if isinstance(manifest, Mapping)
            else None,
            expected=expected_runtime_compatibility,
        )
        if sealed_checkpoint is None:
            root = validate_sharded_resume_manifest(
                manifest,
                expected_world_size=self.world_size,
                checkpoint_dir=checkpoint_dir,
            )
        else:
            # The capability was created by rank-0 full validation and
            # broadcast to this process group.  Recheck its small immutable
            # marker and bind the ZeRO manifest to that sealed ledger without
            # rehashing every shard independently on every rank.
            from think_bridge.model.checkpoint_policy import (
                CHECKPOINT_METADATA_NAME,
                validate_checkpoint_seal,
            )

            seal = validate_checkpoint_seal(sealed_checkpoint)
            checkpoint = Path(checkpoint_dir).resolve(strict=True)
            if seal.path != checkpoint:
                raise ValueError("ZeRO resume seal differs from checkpoint path")
            metadata = json.loads(
                (checkpoint / CHECKPOINT_METADATA_NAME).read_text(encoding="utf-8")
            )
            if (
                not isinstance(metadata, Mapping)
                or metadata.get("zero1_manifest") != dict(manifest)
                or not isinstance(metadata.get("file_ledger"), Mapping)
            ):
                raise ValueError("ZeRO resume manifest differs from checkpoint seal")
            root = validate_sharded_resume_manifest(
                manifest,
                expected_world_size=self.world_size,
                checkpoint_dir=checkpoint,
                sealed_file_ledger=metadata["file_ledger"],
            )
        loaded, client_state = self.engine.load_checkpoint(
            str(root),
            tag=_ZERO1_RESUME_TAG,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
            load_module_strict=False,
        )
        if loaded is None or not isinstance(client_state, dict):
            raise RuntimeError("DeepSpeed did not restore the exact ZeRO-1 checkpoint")
        stored_identity = client_state.get("bridge_backend_identity")
        stored_compatibility = client_state.get("bridge_backend_compatibility_sha256")
        stored_runtime_compatibility = _validate_zero1_runtime_compatibility(
            client_state.get("bridge_zero1_runtime_compatibility"),
            expected=expected_runtime_compatibility,
        )
        if (
            client_state.get("bridge_zero1_runtime_compatibility_sha256")
            != canonical_json_sha256(stored_runtime_compatibility)
            or stored_runtime_compatibility != manifest["runtime_compatibility"]
        ):
            raise ValueError("DeepSpeed checkpoint runtime compatibility mismatch")
        if not isinstance(stored_identity, Mapping):
            raise ValueError("DeepSpeed checkpoint backend identity mismatch")
        projected_stored = backend_compatibility_sha256(stored_identity)
        if stored_compatibility is not None and (
            not isinstance(stored_compatibility, str)
            or stored_compatibility != projected_stored
        ):
            raise ValueError("DeepSpeed checkpoint backend identity mismatch")
        if projected_stored != self.identity.compatibility_sha256():
            raise ValueError("DeepSpeed checkpoint backend identity mismatch")


def initialize_training_backend(
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    config: TrainingConfig,
    route: str,
    rank: int,
    world_size: int,
    device: Any,
    project_root: Path,
) -> BridgeTrainingBackend:
    identity = resolve_backend_identity(config, project_root=project_root, route=route)
    if int(world_size) != identity.world_size:
        raise ValueError("runtime world size differs from optimizer backend identity")
    original_groups = getattr(optimizer, "param_groups", None)
    if (
        not isinstance(original_groups, Sequence)
        or isinstance(original_groups, (str, bytes, bytearray))
        or not original_groups
    ):
        raise RuntimeError("ThinkBridge owner optimizer lacks parameter groups")
    original_parameters = [
        parameter for group in original_groups for parameter in group.get("params", ())
    ]
    if not original_parameters or len(
        {id(value) for value in original_parameters}
    ) != len(original_parameters):
        raise RuntimeError("ThinkBridge owner optimizer parameter ownership is invalid")
    fresh_owner_parameter_count = len(original_parameters)
    fresh_owner_group_count = len(original_groups)
    if identity.optimizer_backend == "replicated_ddp":
        import torch
        import torch.distributed as distributed

        train_model = model
        if distributed.is_initialized():
            train_model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=False,
            )
        return BridgeTrainingBackend(
            identity=identity,
            train_model=train_model,
            optimizer=optimizer,
            scheduler=scheduler,
            engine=None,
            rank=rank,
            world_size=world_size,
            fresh_owner_parameter_count=fresh_owner_parameter_count,
            fresh_owner_group_count=fresh_owner_group_count,
            operational_evidence={
                "optimizer_backend": "replicated_ddp",
                "torch_version": str(torch.__version__),
            },
        )

    try:
        import deepspeed
    except ImportError as exc:
        raise RuntimeError(
            "ThinkBridge ZeRO backend requires DeepSpeed stage 1"
        ) from exc
    engine_config = resolved_deepspeed_config(
        config, project_root=project_root, route=route
    )
    engine, ds_optimizer, _, ds_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler.scheduler,
        config=engine_config,
    )
    if ds_optimizer is None or ds_scheduler is None:
        raise RuntimeError("DeepSpeed did not retain the sealed optimizer/scheduler")
    zero_stage = int(engine.zero_optimization_stage())
    if zero_stage != 1:
        raise RuntimeError(f"DeepSpeed initialized unexpected ZeRO stage {zero_stage}")
    zero1_owner_adapter = bind_bridge_zero1_owner_adapter(
        ds_optimizer, optimizer=optimizer
    )
    deepspeed_version = _normalize_installed_deepspeed_version(
        metadata.version("deepspeed")
    )
    return BridgeTrainingBackend(
        identity=identity,
        train_model=engine,
        optimizer=ds_optimizer,
        scheduler=scheduler,
        engine=engine,
        rank=rank,
        world_size=world_size,
        zero1_owner_adapter=zero1_owner_adapter,
        fresh_owner_parameter_count=fresh_owner_parameter_count,
        fresh_owner_group_count=fresh_owner_group_count,
        operational_evidence={
            "optimizer_backend": "deepspeed_zero1",
            "deepspeed_version": deepspeed_version,
            "zero1_owner_adapter": zero1_owner_adapter.signature,
        },
    )
