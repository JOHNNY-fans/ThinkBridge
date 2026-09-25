"""Topology-neutral sharding and exact merge for ThinkBridge GPU artifacts."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import timedelta
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Callable, Iterator, Mapping, Sequence
import uuid

from think_bridge.model.artifact_schema import (
    DISTRIBUTED_JSON_SHARD,
    DISTRIBUTED_TENSOR_SHARD,
    DISTRIBUTED_TRANSACTION,
    artifact_header,
)

from think_bridge.model.contract import (
    canonical_json_sha256,
    is_bridge_isolated_path,
)


OFFLINE_COLLECTIVE_TIMEOUT = timedelta(hours=2)
_TRANSACTION_ID = re.compile(r"[0-9a-f]{32}")
_TRANSACTION_LABEL = re.compile(r"[a-z][a-z0-9-]*")


@dataclass(frozen=True)
class DistributedArtifactTransaction:
    """One torchrun-owned filesystem namespace, never an artifact identity."""

    transaction_id: str
    root: Path
    label: str
    world_size: int


def _require_transaction_id(value: str) -> str:
    normalized = str(value)
    if _TRANSACTION_ID.fullmatch(normalized) is None:
        raise ValueError("distributed artifact transaction id is invalid")
    return normalized


def begin_distributed_artifact_transaction(
    distributed: Any,
    *,
    rank: int,
    world_size: int,
    parent: Path,
    label: str,
) -> DistributedArtifactTransaction:
    """Create one rank-0-owned root and broadcast its exact descriptor."""

    rank = int(rank)
    world_size = int(world_size)
    parent = Path(parent).resolve(strict=False)
    label = str(label)
    if (
        world_size <= 0
        or rank < 0
        or rank >= world_size
        or _TRANSACTION_LABEL.fullmatch(label) is None
        or not is_bridge_isolated_path(parent)
    ):
        raise ValueError("distributed artifact transaction geometry is invalid")
    local: dict[str, Any] | None = None
    if rank == 0:
        try:
            transaction_id = uuid.uuid4().hex
            root = parent / transaction_id
            parent.mkdir(parents=True, exist_ok=True)
            root.mkdir(exist_ok=False)
            local = {
                **artifact_header(DISTRIBUTED_TRANSACTION),
                "transaction_id": transaction_id,
                "root": str(root),
                "label": label,
                "world_size": world_size,
                "error": None,
            }
        except Exception as exc:
            local = {
                **artifact_header(DISTRIBUTED_TRANSACTION),
                "transaction_id": None,
                "root": None,
                "label": label,
                "world_size": world_size,
                "error": f"{type(exc).__name__}: {exc}",
            }
    values = [local if rank == 0 else None]
    if distributed.is_initialized():
        distributed.broadcast_object_list(values, src=0)
    descriptor = values[0]
    if not isinstance(descriptor, dict) or descriptor.get("error") is not None:
        raise RuntimeError(
            f"distributed artifact transaction creation failed: {descriptor}"
        )
    required = {
        "artifact_type",
        "schema_version",
        "transaction_id",
        "root",
        "label",
        "world_size",
        "error",
    }
    transaction_id = _require_transaction_id(descriptor.get("transaction_id"))
    expected_root = parent / transaction_id
    if (
        set(descriptor) != required
        or descriptor["artifact_type"] != DISTRIBUTED_TRANSACTION
        or descriptor["schema_version"] != 1
        or descriptor["label"] != label
        or int(descriptor["world_size"]) != world_size
        or Path(descriptor["root"]) != expected_root
        or not expected_root.is_dir()
    ):
        raise RuntimeError("distributed artifact transaction descriptor mismatch")
    return DistributedArtifactTransaction(
        transaction_id=transaction_id,
        root=expected_root,
        label=label,
        world_size=world_size,
    )


def cleanup_artifact_transaction(transaction: DistributedArtifactTransaction) -> None:
    """Remove only the invocation-owned root; shared parents are never swept."""

    root = Path(transaction.root)
    _require_transaction_id(transaction.transaction_id)
    if root.name != transaction.transaction_id or not is_bridge_isolated_path(root):
        raise ValueError("artifact transaction cleanup escaped its owned root")
    if root.exists():
        shutil.rmtree(root)


@contextmanager
def distributed_artifact_transaction(
    distributed: Any,
    *,
    rank: int,
    world_size: int,
    parent: Path,
    label: str,
) -> Iterator[DistributedArtifactTransaction]:
    transaction = begin_distributed_artifact_transaction(
        distributed,
        rank=rank,
        world_size=world_size,
        parent=parent,
        label=label,
    )
    try:
        yield transaction
    finally:
        if int(rank) == 0:
            cleanup_artifact_transaction(transaction)


@contextmanager
def owned_process_group(distributed: Any, *, owned: bool = True) -> Iterator[None]:
    """Gracefully destroy a command-owned process group only after success.

    A distributed worker failure must reach torchrun/elastic immediately.  A
    destroy in exception unwinding can wait for peers that are still blocked in
    a collective and can therefore hide the first rank's original traceback.
    """

    try:
        yield
    except BaseException:
        raise
    else:
        if owned and distributed.is_initialized():
            distributed.destroy_process_group()


def run_owned_process_group_entry(
    distributed: Any,
    *,
    initialize: Callable[[], Any],
    body: Callable[[Any, ExitStack], Any],
) -> Any:
    """Run one entry with caller-aware PG ownership and inner resource cleanup."""

    owns_process_group = not distributed.is_initialized()
    with owned_process_group(distributed, owned=owns_process_group):
        initialized_context = initialize()
        resource_stack = ExitStack()
        try:
            result = body(initialized_context, resource_stack)
        except BaseException as primary_error:
            try:
                resource_stack.__exit__(
                    type(primary_error),
                    primary_error,
                    primary_error.__traceback__,
                )
            except BaseException as cleanup_error:
                try:
                    primary_error.add_note(
                        "suppressed local resource cleanup failure while preserving "
                        f"the original worker exception: {type(cleanup_error).__name__}: "
                        f"{cleanup_error}"
                    )
                except BaseException:
                    pass
            raise
        else:
            resource_stack.close()
            return result


def publish_file_once(
    path: Path,
    *,
    transaction_id: str,
    write_temporary: Callable[[Path], Any],
    validate: Callable[[Path], Any],
) -> str:
    """Atomically create a final file or validate/reuse a concurrent winner."""

    path = Path(path)
    transaction_id = _require_transaction_id(transaction_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        validate(path)
        return "reuse"
    if path.exists():
        raise FileExistsError(f"artifact destination is not a file: {path}")
    temporary = path.with_name(f".{path.name}.{transaction_id}.building")
    if temporary.exists():
        raise FileExistsError(f"transaction temporary already exists: {temporary}")
    try:
        write_temporary(temporary)
        if not temporary.is_file():
            raise RuntimeError("artifact writer did not create its temporary file")
        validate(temporary)
        try:
            os.link(temporary, path)
            action = "build"
        except FileExistsError:
            if not path.is_file():
                raise
            action = "reuse"
        validate(path)
        return action
    finally:
        temporary.unlink(missing_ok=True)


def local_rank_from_environment(environment: Mapping[str, str]) -> tuple[int, int, int]:
    try:
        rank = int(environment.get("RANK", "0"))
        world_size = int(environment.get("WORLD_SIZE", "1"))
        local_rank = int(environment.get("LOCAL_RANK", "0"))
    except ValueError as exc:
        raise ValueError("distributed rank environment is not integral") from exc
    if world_size <= 0 or rank < 0 or rank >= world_size or local_rank < 0:
        raise ValueError("distributed rank environment is outside its legal domain")
    return rank, world_size, local_rank


def initialize_distributed_gpu() -> tuple[int, int, int, Any]:
    """Select LOCAL_RANK before process-group init and before model loading."""

    import torch
    import torch.distributed as distributed

    rank, world_size, local_rank = local_rank_from_environment(os.environ)
    if torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise ValueError("LOCAL_RANK exceeds visible CUDA devices")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if world_size > 1 and not distributed.is_initialized():
        init_kwargs: dict[str, Any] = {
            "backend": backend,
            "init_method": "env://",
            "timeout": OFFLINE_COLLECTIVE_TIMEOUT,
        }
        if backend == "nccl":
            init_kwargs["device_id"] = device
        distributed.init_process_group(**init_kwargs)
    return rank, world_size, local_rank, device


def contiguous_shard_indices(
    total: int, *, rank: int, world_size: int
) -> tuple[int, ...]:
    if total < 0 or world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError("invalid distributed shard geometry")
    start = total * rank // world_size
    stop = total * (rank + 1) // world_size
    return tuple(range(start, stop))


def _shard_path(root: Path, rank: int) -> Path:
    return Path(root) / f"bridge-rank-{int(rank):05d}.json"


def write_json_shard(
    root: Path,
    *,
    rank: int,
    world_size: int,
    transaction_id: str,
    identity_sha256: str,
    indices: Sequence[int],
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    root = Path(root)
    if not is_bridge_isolated_path(root):
        raise ValueError("distributed shard root is outside ThinkBridge")
    if len(indices) != len(rows):
        raise ValueError("distributed shard index/payload cardinality mismatch")
    normalized_rows = [dict(row) for row in rows]
    transaction_id = _require_transaction_id(transaction_id)
    payload = {
        **artifact_header(DISTRIBUTED_JSON_SHARD),
        "transaction_id": transaction_id,
        "rank": int(rank),
        "world_size": int(world_size),
        "identity_sha256": str(identity_sha256),
        "indices": [int(index) for index in indices],
        "rows": normalized_rows,
        "rows_sha256": canonical_json_sha256(normalized_rows),
    }
    path = _shard_path(root, rank)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".building")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"distributed worker shard already exists: {path}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def merge_json_shards(
    root: Path,
    *,
    world_size: int,
    expected_count: int,
    transaction_id: str,
    identity_sha256: str,
    expected_indices_by_rank: Sequence[Sequence[int]] | None = None,
) -> list[dict[str, Any]]:
    root = Path(root)
    transaction_id = _require_transaction_id(transaction_id)
    if expected_indices_by_rank is None:
        expected_indices_by_rank = [
            contiguous_shard_indices(expected_count, rank=rank, world_size=world_size)
            for rank in range(world_size)
        ]
    plan = [list(indices) for indices in expected_indices_by_rank]
    planned_indices = [index for indices in plan for index in indices]
    if (
        len(plan) != world_size
        or any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in planned_indices
        )
        or sorted(planned_indices) != list(range(expected_count))
    ):
        raise ValueError(
            "distributed shard plan must cover each canonical index exactly once"
        )
    paths = sorted(root.glob("bridge-rank-*.json"))
    expected_paths = [_shard_path(root, rank) for rank in range(world_size)]
    if paths != expected_paths:
        raise ValueError("distributed shard rank set is incomplete or foreign")
    indexed: dict[int, dict[str, Any]] = {}
    for rank, path in enumerate(paths):
        value = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "artifact_type",
            "schema_version",
            "transaction_id",
            "rank",
            "world_size",
            "identity_sha256",
            "indices",
            "rows",
            "rows_sha256",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError("distributed shard schema mismatch")
        if (
            value["artifact_type"] != DISTRIBUTED_JSON_SHARD
            or value["schema_version"] != 1
            or value["transaction_id"] != transaction_id
            or int(value["rank"]) != rank
            or int(value["world_size"]) != int(world_size)
            or value["identity_sha256"] != identity_sha256
        ):
            raise ValueError("distributed shard transaction/identity mismatch")
        indices = value["indices"]
        rows = value["rows"]
        if (
            not isinstance(indices, list)
            or not isinstance(rows, list)
            or len(indices) != len(rows)
        ):
            raise ValueError(f"distributed shard row/index count mismatch: rank={rank}")
        if indices != plan[rank]:
            raise ValueError(
                f"distributed shard indices differ from evaluation plan: rank={rank}"
            )
        if value["rows_sha256"] != canonical_json_sha256(rows):
            raise ValueError(f"distributed shard payload hash mismatch: rank={rank}")
        for index, row in zip(indices, rows):
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index in indexed
                or not isinstance(row, dict)
                or row.get("index") != index
            ):
                raise ValueError(
                    "distributed shard has duplicate/invalid canonical index"
                )
            indexed[index] = row
    if sorted(indexed) != list(range(int(expected_count))):
        raise ValueError("distributed shard indices are missing or out of range")
    return [indexed[index] for index in range(int(expected_count))]


def write_tensor_shard(
    root: Path,
    *,
    rank: int,
    world_size: int,
    transaction_id: str,
    identity_sha256: str,
    indices: Sequence[int],
    group_ids: Sequence[str],
    tensors: Any,
) -> Path:
    import torch

    root = Path(root)
    if not is_bridge_isolated_path(root) or len(indices) != len(group_ids):
        raise ValueError("distributed tensor shard geometry is invalid")
    tensor = tensors.detach().float().cpu().contiguous()
    transaction_id = _require_transaction_id(transaction_id)
    if (
        not torch.is_tensor(tensor)
        or tensor.ndim < 2
        or tensor.size(0) != len(indices)
        or not bool(torch.isfinite(tensor).all())
    ):
        raise ValueError("distributed tensor shard is nonfinite or misshaped")
    payload = {
        **artifact_header(DISTRIBUTED_TENSOR_SHARD),
        "transaction_id": transaction_id,
        "rank": int(rank),
        "world_size": int(world_size),
        "identity_sha256": str(identity_sha256),
        "indices": [int(index) for index in indices],
        "group_ids": [str(group) for group in group_ids],
        "tensors": tensor,
    }
    path = Path(root) / f"bridge-rank-{int(rank):05d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".building")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"distributed tensor shard already exists: {path}")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def merge_tensor_shards(
    root: Path,
    *,
    world_size: int,
    transaction_id: str,
    group_ids: Sequence[str],
    tensor_shape_tail: Sequence[int],
    identity_sha256: str,
) -> Any:
    import torch

    root = Path(root)
    transaction_id = _require_transaction_id(transaction_id)
    paths = sorted(root.glob("bridge-rank-*.pt"))
    expected_paths = [root / f"bridge-rank-{rank:05d}.pt" for rank in range(world_size)]
    if paths != expected_paths:
        raise ValueError("distributed tensor shard rank set is incomplete or foreign")
    indexed: dict[int, Any] = {}
    for rank, path in enumerate(paths):
        value = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(value, dict) or set(value) != {
            "artifact_type",
            "schema_version",
            "transaction_id",
            "rank",
            "world_size",
            "identity_sha256",
            "indices",
            "group_ids",
            "tensors",
        }:
            raise ValueError("distributed tensor shard schema mismatch")
        if (
            value["artifact_type"] != DISTRIBUTED_TENSOR_SHARD
            or value["schema_version"] != 1
            or value["transaction_id"] != transaction_id
            or int(value["rank"]) != rank
            or int(value["world_size"]) != int(world_size)
            or value["identity_sha256"] != identity_sha256
        ):
            raise ValueError("distributed tensor shard transaction/identity mismatch")
        indices = value["indices"]
        local_groups = value["group_ids"]
        tensors = value["tensors"]
        if (
            len(indices) != len(local_groups)
            or indices
            != list(
                contiguous_shard_indices(
                    len(group_ids), rank=rank, world_size=int(world_size)
                )
            )
            or not torch.is_tensor(tensors)
            or tuple(tensors.shape)
            != (len(indices), *(int(value) for value in tensor_shape_tail))
            or tensors.dtype != torch.float32
            or tensors.requires_grad
            or not tensors.is_contiguous()
            or not bool(torch.isfinite(tensors).all())
        ):
            raise ValueError("distributed tensor shard payload mismatch")
        for offset, index in enumerate(indices):
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index in indexed
                or index < 0
                or index >= len(group_ids)
                or local_groups[offset] != group_ids[index]
            ):
                raise ValueError("distributed tensor shard index/group mismatch")
            indexed[index] = tensors[offset]
    if sorted(indexed) != list(range(len(group_ids))):
        raise ValueError("distributed tensor shards omit canonical prompt groups")
    return torch.stack([indexed[index] for index in range(len(group_ids))]).contiguous()
