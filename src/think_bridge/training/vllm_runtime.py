"""Dependency-light contracts for the optional Route1 generation service.

The Bridge Route1 optimizer and every scorer stay in the torch/HF trainer. This
module exposes only free-generation transport and lifecycle surfaces; importing
it requires neither torch nor vLLM.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
import re
import secrets
import signal
import socket
import struct
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

from think_bridge.model.artifact_schema import artifact_header
from think_bridge.training.process_cleanup import (
    OwnedProcessTree,
    uninterrupted_cleanup,
    cleanup_message,
    spawn_owned_process,
)


_REQUEST_MAGIC = b"TBBridgeQ1"
_RESPONSE_MAGIC = b"TBBridgeS1"
_TENSOR_MAGIC = b"TBBridgeT1"
_HEADER_LENGTH = struct.Struct("<I")
ROUTE1_MAX_HTTP_REQUEST_BYTES = 128 * 1024 * 1024
ROUTE1_MAX_HTTP_RESPONSE_BYTES = 256 * 1024 * 1024
ROUTE1_GENERATION_PROBE_STATUS = "generation-only-native-generate-ok"
_TRANSACTION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{7,127}")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{7,127}")
_INSTANCE_TOKEN = re.compile(r"[A-Za-z0-9_-]{32,128}")
_TENSOR_ELEMENT_BYTES = {"float32": 4, "bfloat16": 2}


def parse_strict_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError("boolean value must be exactly 'true' or 'false'")


def new_private_service_instance_token() -> str:
    """Create a non-scientific nonce binding one parent to one service child."""

    token = secrets.token_urlsafe(32)
    if _INSTANCE_TOKEN.fullmatch(token) is None:
        raise RuntimeError("generated private service instance token is invalid")
    return token


def validate_private_service_instance_token(value: Any) -> str:
    token = str(value)
    if _INSTANCE_TOKEN.fullmatch(token) is None:
        raise ValueError("private service instance token is invalid")
    return token


def _gpu_tuple(values: Sequence[int], label: str) -> tuple[int, ...]:
    normalized = tuple(int(value) for value in values)
    if not normalized or any(value < 0 for value in normalized):
        raise ValueError(f"{label} must contain non-negative GPU ids")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} contains duplicate GPU ids")
    return normalized


def _shape_elements(shape: Sequence[int]) -> int:
    elements = 1
    for value in shape:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("tensor payload shape is invalid")
        elements *= int(value)
    return elements


def _validate_tensor_layout(
    descriptors: Sequence[Mapping[str, Any]], payload: bytes
) -> None:
    offset = 0
    for descriptor in descriptors:
        if not isinstance(descriptor, Mapping):
            raise ValueError("tensor payload descriptor is invalid")
        if descriptor.get("offset") != offset:
            raise ValueError("tensor payload descriptors are not contiguous")
        nbytes = descriptor.get("nbytes")
        if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 0:
            raise ValueError("tensor payload descriptor size is invalid")
        offset += nbytes
    if offset != len(payload):
        raise ValueError("tensor payload layout does not cover its exact bytes")


@dataclass(frozen=True)
class TensorPayload:
    dtype: str
    shape: tuple[int, ...]
    data: bytes
    sha256: str

    def __post_init__(self) -> None:
        if self.dtype not in _TENSOR_ELEMENT_BYTES:
            raise ValueError("tensor payload dtype is invalid")
        if not isinstance(self.shape, tuple) or not self.shape:
            raise ValueError("tensor payload shape is invalid")
        expected = _shape_elements(self.shape) * _TENSOR_ELEMENT_BYTES[self.dtype]
        if not isinstance(self.data, bytes) or len(self.data) != expected:
            raise ValueError("tensor payload byte length differs from shape/dtype")
        if self.sha256 != hashlib.sha256(self.data).hexdigest():
            raise ValueError("tensor payload digest mismatch")

    @property
    def nbytes(self) -> int:
        return len(self.data)

    @classmethod
    def from_bytes(
        cls, *, dtype: str, shape: Sequence[int], data: bytes
    ) -> "TensorPayload":
        raw = bytes(data)
        return cls(
            dtype=str(dtype),
            shape=tuple(int(value) for value in shape),
            data=raw,
            sha256=hashlib.sha256(raw).hexdigest(),
        )

    @classmethod
    def empty(cls, *, dtype: str, shape: Sequence[int]) -> "TensorPayload":
        return cls.from_bytes(dtype=dtype, shape=shape, data=b"")

    @classmethod
    def from_float32_values(
        cls, values: Sequence[float], *, shape: Sequence[int]
    ) -> "TensorPayload":
        packed = array("f", (float(value) for value in values))
        if sys.byteorder != "little":
            packed.byteswap()
        return cls.from_bytes(dtype="float32", shape=shape, data=packed.tobytes())

    @classmethod
    def from_torch(
        cls, tensor: Any, *, wire_dtype: str | None = None
    ) -> "TensorPayload":
        import torch

        if not torch.is_tensor(tensor):
            raise TypeError("tensor payload source must be a torch tensor")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("tensor payload source contains non-finite values")
        target = wire_dtype or (
            "bfloat16" if tensor.dtype == torch.bfloat16 else "float32"
        )
        if target == "float32":
            value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
            raw = value.numpy().tobytes(order="C")
        elif target == "bfloat16":
            value = tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
            raw = value.view(torch.uint16).numpy().tobytes(order="C")
        else:
            raise ValueError("tensor payload torch wire dtype is invalid")
        return cls.from_bytes(dtype=target, shape=tuple(value.shape), data=raw)

    def to_torch(self, *, device: Any = None, dtype: Any = None) -> Any:
        import torch

        raw = bytearray(self.data)
        if self.dtype == "float32":
            value = torch.frombuffer(raw, dtype=torch.float32).clone()
        else:
            value = torch.frombuffer(raw, dtype=torch.uint16).clone()
            value = value.view(torch.bfloat16)
        value = value.reshape(self.shape)
        if not bool(torch.isfinite(value).all()):
            raise ValueError("tensor payload contains non-finite values")
        if dtype is not None or device is not None:
            value = value.to(device=device, dtype=dtype)
        return value

    def descriptor(self, *, offset: int) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "shape": list(self.shape),
            "offset": int(offset),
            "nbytes": self.nbytes,
            "sha256": self.sha256,
        }

    @classmethod
    def from_descriptor(
        cls, descriptor: Mapping[str, Any], payload: bytes
    ) -> "TensorPayload":
        if not isinstance(descriptor, Mapping):
            raise ValueError("tensor payload descriptor is invalid")
        offset = descriptor.get("offset")
        nbytes = descriptor.get("nbytes")
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
            or isinstance(nbytes, bool)
            or not isinstance(nbytes, int)
            or nbytes < 0
            or offset + nbytes > len(payload)
        ):
            raise ValueError("tensor payload descriptor bounds are invalid")
        return cls(
            dtype=str(descriptor.get("dtype")),
            shape=tuple(int(value) for value in descriptor.get("shape", ())),
            data=bytes(payload[offset : offset + nbytes]),
            sha256=str(descriptor.get("sha256")),
        )

    def to_wire(self) -> bytes:
        header = json.dumps(
            self.descriptor(offset=0), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return _TENSOR_MAGIC + _HEADER_LENGTH.pack(len(header)) + header + self.data

    @classmethod
    def from_wire(cls, payload: bytes) -> "TensorPayload":
        if not isinstance(payload, bytes) or not payload.startswith(_TENSOR_MAGIC):
            raise ValueError("tensor payload wire magic mismatch")
        start = len(_TENSOR_MAGIC) + _HEADER_LENGTH.size
        if len(payload) < start:
            raise ValueError("tensor payload wire header is truncated")
        length = _HEADER_LENGTH.unpack_from(payload, len(_TENSOR_MAGIC))[0]
        stop = start + int(length)
        if stop > len(payload):
            raise ValueError("tensor payload wire JSON is truncated")
        try:
            descriptor = json.loads(payload[start:stop].decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("tensor payload wire JSON is invalid") from exc
        tensor_bytes = payload[stop:]
        _validate_tensor_layout((descriptor,), tensor_bytes)
        return cls.from_descriptor(descriptor, tensor_bytes)


@dataclass(frozen=True)
class LaunchSpec:
    command: tuple[str, ...]
    environment: Mapping[str, str]


@dataclass(frozen=True)
class Route1ServiceRequest:
    """Generation-only request; no scorer/wave fields exist in this schema."""

    run_id: str
    transaction_id: str
    rank: int
    step: int
    micro_step: int
    chunk_id: int
    sample_ids: tuple[int, ...]
    sample_keys: tuple[str, ...]
    sample_is_padding: tuple[bool, ...]
    prompt_ids: tuple[tuple[int, ...], ...]
    prompt_lengths: tuple[int, ...]
    detached_z: TensorPayload
    boundary_ids: tuple[int, ...]
    generation_seed: int
    max_prefix_tokens: int
    z_present: tuple[bool, ...]
    append_boundary: tuple[bool, ...]
    eos_token_id: int
    temperature: float = 0.0  # Evaluation is greedy unless training opts in.

    def __post_init__(self) -> None:
        if _RUN_ID.fullmatch(str(self.run_id)) is None:
            raise ValueError("Route1 run id is invalid")
        if _TRANSACTION_ID.fullmatch(str(self.transaction_id)) is None:
            raise ValueError("Route1 transaction id is invalid")
        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank < 0
        ):
            raise ValueError("Route1 request rank is invalid")
        for value, label in (
            (self.step, "step"),
            (self.micro_step, "micro-step"),
            (self.chunk_id, "chunk id"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Route1 request {label} is invalid")
        count = len(self.sample_ids)
        if count <= 0 or len(set(self.sample_ids)) != count:
            raise ValueError("Route1 request sample ids must be non-empty and unique")
        if (
            len(self.sample_keys) != count
            or len(set(self.sample_keys)) != count
            or any(
                not isinstance(value, str) or not value for value in self.sample_keys
            )
            or len(self.sample_is_padding) != count
            or any(not isinstance(value, bool) for value in self.sample_is_padding)
            or all(self.sample_is_padding)
        ):
            raise ValueError("Route1 request sample identity rows are invalid")
        if len(self.prompt_ids) != count or len(self.prompt_lengths) != count:
            raise ValueError("Route1 request rows are not aligned")
        if len(self.z_present) != count or any(
            not isinstance(value, bool) for value in self.z_present
        ):
            raise ValueError("Route1 request z-presence rows are not aligned")
        if len(self.append_boundary) != count or any(
            not isinstance(value, bool) for value in self.append_boundary
        ):
            raise ValueError("Route1 request boundary-policy rows are not aligned")
        if any(
            present and not append
            for present, append in zip(self.z_present, self.append_boundary)
        ):
            raise ValueError("Route1 z rows must append the sealed boundary")
        if (
            not isinstance(self.detached_z, TensorPayload)
            or self.detached_z.dtype != "float32"
            or len(self.detached_z.shape) != 3
            or self.detached_z.shape[0] != count
            or self.detached_z.shape[1] not in {32, 64, 128}
            or self.detached_z.shape[2] <= 0
        ):
            raise ValueError("Route1 request detached-z tensor is invalid")
        for row, length in zip(self.prompt_ids, self.prompt_lengths):
            if (
                isinstance(length, bool)
                or int(length) != len(row)
                or not row
                or any(int(value) < 0 for value in row)
            ):
                raise ValueError("Route1 prompt ids/length are invalid")
        if not self.boundary_ids or any(int(value) < 0 for value in self.boundary_ids):
            raise ValueError("Route1 boundary ids are invalid")
        if isinstance(self.generation_seed, bool) or not isinstance(
            self.generation_seed, int
        ):
            raise ValueError("Route1 generation seed must be an integer")
        if not math.isfinite(float(self.temperature)) or float(self.temperature) < 0:
            raise ValueError(
                "Route1 generation temperature must be finite and nonnegative"
            )
        if (
            isinstance(self.max_prefix_tokens, bool)
            or not isinstance(self.max_prefix_tokens, int)
            or self.max_prefix_tokens <= 0
        ):
            raise ValueError("Route1 max prefix tokens must be positive")
        if (
            isinstance(self.eos_token_id, bool)
            or not isinstance(self.eos_token_id, int)
            or self.eos_token_id < 0
        ):
            raise ValueError("Route1 EOS token id is invalid")

    @property
    def identity(self) -> tuple[Any, ...]:
        return (
            self.run_id,
            int(self.rank),
            self.transaction_id,
            int(self.step),
            int(self.micro_step),
            int(self.chunk_id),
        )

    def with_chunk_id(self, chunk_id: int) -> "Route1ServiceRequest":
        return replace(self, chunk_id=int(chunk_id))

    def to_wire(self) -> bytes:
        header = {
            **artifact_header("think-bridge.runtime.route1-generation-request"),
            "run_id": self.run_id,
            "transaction_id": self.transaction_id,
            "rank": int(self.rank),
            "step": int(self.step),
            "micro_step": int(self.micro_step),
            "chunk_id": int(self.chunk_id),
            "sample_ids": list(self.sample_ids),
            "sample_keys": list(self.sample_keys),
            "sample_is_padding": list(self.sample_is_padding),
            "prompt_ids": [list(row) for row in self.prompt_ids],
            "prompt_lengths": list(self.prompt_lengths),
            "boundary_ids": list(self.boundary_ids),
            "generation_seed": int(self.generation_seed),
            "max_prefix_tokens": int(self.max_prefix_tokens),
            "temperature": float(self.temperature),
            "z_present": list(self.z_present),
            "append_boundary": list(self.append_boundary),
            "eos_token_id": int(self.eos_token_id),
            "detached_z": self.detached_z.descriptor(offset=0),
        }
        header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return (
            _REQUEST_MAGIC
            + _HEADER_LENGTH.pack(len(header_bytes))
            + header_bytes
            + self.detached_z.data
        )

    @classmethod
    def from_wire(cls, payload: bytes) -> "Route1ServiceRequest":
        if not isinstance(payload, bytes) or not payload.startswith(_REQUEST_MAGIC):
            raise ValueError("Route1 request wire magic mismatch")
        start = len(_REQUEST_MAGIC) + _HEADER_LENGTH.size
        if len(payload) < start:
            raise ValueError("Route1 request wire header is truncated")
        length = _HEADER_LENGTH.unpack_from(payload, len(_REQUEST_MAGIC))[0]
        stop = start + int(length)
        if stop > len(payload):
            raise ValueError("Route1 request wire JSON is truncated")
        try:
            header = json.loads(payload[start:stop].decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Route1 request wire JSON is invalid") from exc
        if (
            not isinstance(header, dict)
            or header.get("artifact_type")
            != "think-bridge.runtime.route1-generation-request"
            or header.get("schema_version") != 1
        ):
            raise ValueError("Route1 request wire schema mismatch")
        tensor_bytes = payload[stop:]
        descriptor = header.get("detached_z")
        _validate_tensor_layout((descriptor,), tensor_bytes)
        return cls(
            run_id=str(header["run_id"]),
            transaction_id=str(header["transaction_id"]),
            rank=int(header["rank"]),
            step=int(header["step"]),
            micro_step=int(header["micro_step"]),
            chunk_id=int(header["chunk_id"]),
            sample_ids=tuple(int(value) for value in header["sample_ids"]),
            sample_keys=tuple(str(value) for value in header["sample_keys"]),
            sample_is_padding=tuple(
                bool(value) for value in header["sample_is_padding"]
            ),
            prompt_ids=tuple(
                tuple(int(value) for value in row) for row in header["prompt_ids"]
            ),
            prompt_lengths=tuple(int(value) for value in header["prompt_lengths"]),
            detached_z=TensorPayload.from_descriptor(descriptor, tensor_bytes),
            boundary_ids=tuple(int(value) for value in header["boundary_ids"]),
            generation_seed=int(header["generation_seed"]),
            temperature=float(header.get("temperature", 0.0)),
            max_prefix_tokens=int(header["max_prefix_tokens"]),
            z_present=tuple(header["z_present"]),
            append_boundary=tuple(header["append_boundary"]),
            eos_token_id=int(header["eos_token_id"]),
        )


@dataclass(frozen=True)
class Route1ServiceResponse:
    """Generation-only response with no scorer or optimizer payload."""

    run_id: str
    transaction_id: str
    rank: int
    step: int
    micro_step: int
    chunk_id: int
    sample_ids: tuple[int, ...]
    sample_keys: tuple[str, ...]
    sample_is_padding: tuple[bool, ...]
    status: str
    prefix_token_ids: tuple[tuple[int, ...], ...]
    prefix_token_masks: tuple[tuple[bool, ...], ...]
    timings: Mapping[str, float]
    counts: Mapping[str, int]
    error: str | None = None

    @classmethod
    def fake_for(cls, request: Route1ServiceRequest) -> "Route1ServiceResponse":
        prefixes = tuple((int(sample_id) + 100,) for sample_id in request.sample_ids)
        return cls(
            run_id=request.run_id,
            transaction_id=request.transaction_id,
            rank=request.rank,
            step=request.step,
            micro_step=request.micro_step,
            chunk_id=request.chunk_id,
            sample_ids=request.sample_ids,
            sample_keys=request.sample_keys,
            sample_is_padding=request.sample_is_padding,
            status="ok",
            prefix_token_ids=prefixes,
            prefix_token_masks=tuple(tuple(True for _ in row) for row in prefixes),
            timings={
                "service_queue_seconds": 0.0,
                "rollout_generation_seconds": 0.0,
                "service_request_seconds": 0.0,
            },
            counts={
                "service_batch_rows": len(request.sample_ids),
                "service_real_rows": sum(
                    not value for value in request.sample_is_padding
                ),
                "service_active_requests": 1,
            },
        )

    @property
    def identity(self) -> tuple[Any, ...]:
        return (
            self.run_id,
            int(self.rank),
            self.transaction_id,
            int(self.step),
            int(self.micro_step),
            int(self.chunk_id),
        )

    def to_wire(self) -> bytes:
        header = {
            **artifact_header("think-bridge.runtime.route1-generation-response"),
            "run_id": self.run_id,
            "transaction_id": self.transaction_id,
            "rank": int(self.rank),
            "step": int(self.step),
            "micro_step": int(self.micro_step),
            "chunk_id": int(self.chunk_id),
            "sample_ids": list(self.sample_ids),
            "sample_keys": list(self.sample_keys),
            "sample_is_padding": list(self.sample_is_padding),
            "status": self.status,
            "prefix_token_ids": [list(row) for row in self.prefix_token_ids],
            "prefix_token_masks": [list(row) for row in self.prefix_token_masks],
            "timings": dict(self.timings),
            "counts": dict(self.counts),
            "error": self.error,
        }
        header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return _RESPONSE_MAGIC + _HEADER_LENGTH.pack(len(header_bytes)) + header_bytes

    @classmethod
    def from_wire(cls, payload: bytes) -> "Route1ServiceResponse":
        if not isinstance(payload, bytes) or not payload.startswith(_RESPONSE_MAGIC):
            raise ValueError("Route1 response wire magic mismatch")
        start = len(_RESPONSE_MAGIC) + _HEADER_LENGTH.size
        if len(payload) < start:
            raise ValueError("Route1 response wire header is truncated")
        length = _HEADER_LENGTH.unpack_from(payload, len(_RESPONSE_MAGIC))[0]
        stop = start + int(length)
        if stop != len(payload):
            raise ValueError("Route1 generation response carries trailing payload")
        try:
            header = json.loads(payload[start:stop].decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Route1 response wire JSON is invalid") from exc
        if (
            not isinstance(header, dict)
            or header.get("artifact_type")
            != "think-bridge.runtime.route1-generation-response"
            or header.get("schema_version") != 1
        ):
            raise ValueError("Route1 response wire schema mismatch")
        return cls(
            run_id=str(header["run_id"]),
            transaction_id=str(header["transaction_id"]),
            rank=int(header["rank"]),
            step=int(header["step"]),
            micro_step=int(header["micro_step"]),
            chunk_id=int(header["chunk_id"]),
            sample_ids=tuple(int(value) for value in header["sample_ids"]),
            sample_keys=tuple(str(value) for value in header["sample_keys"]),
            sample_is_padding=tuple(
                bool(value) for value in header["sample_is_padding"]
            ),
            status=str(header["status"]),
            prefix_token_ids=tuple(
                tuple(int(value) for value in row) for row in header["prefix_token_ids"]
            ),
            prefix_token_masks=tuple(
                tuple(bool(value) for value in row)
                for row in header["prefix_token_masks"]
            ),
            timings={
                str(key): float(value) for key, value in header["timings"].items()
            },
            counts={str(key): int(value) for key, value in header["counts"].items()},
            error=None if header.get("error") is None else str(header["error"]),
        )

    def assert_matches(self, request: Route1ServiceRequest) -> None:
        if (
            self.identity != request.identity
            or self.sample_ids != request.sample_ids
            or self.sample_keys != request.sample_keys
            or self.sample_is_padding != request.sample_is_padding
        ):
            raise RuntimeError(
                "Route1 service response transaction/sample identity mismatch"
            )
        if self.status != "ok":
            raise RuntimeError(
                "Route1 service failed: " + (self.error or "unknown service error")
            )
        count = len(request.sample_ids)
        if len(self.prefix_token_ids) != count or len(self.prefix_token_masks) != count:
            raise RuntimeError("Route1 service response shape mismatch")
        for ids, mask in zip(self.prefix_token_ids, self.prefix_token_masks):
            if (
                len(ids) != len(mask)
                or not ids
                or not all(mask)
                or len(ids) > request.max_prefix_tokens
                or any(int(token_id) < 0 for token_id in ids)
            ):
                raise RuntimeError("Route1 service prefix mask is invalid")
        required_timings = {
            "service_queue_seconds",
            "rollout_generation_seconds",
            "service_request_seconds",
        }
        if set(self.timings) != required_timings or any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in self.timings.values()
        ):
            raise RuntimeError("Route1 service response timing telemetry is invalid")
        required_counts = {
            "service_batch_rows",
            "service_real_rows",
            "service_active_requests",
        }
        if (
            set(self.counts) != required_counts
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in self.counts.values()
            )
            or int(self.counts["service_batch_rows"]) != count
            or int(self.counts["service_real_rows"])
            != sum(not value for value in request.sample_is_padding)
            or int(self.counts["service_active_requests"]) <= 0
        ):
            raise RuntimeError("Route1 service response count telemetry is invalid")


class VLLMServiceSupervisor:
    """Own one private service process and prove bounded final teardown."""

    def __init__(
        self,
        *,
        spawn: Callable[[LaunchSpec], Any],
        ready: Callable[[str, int, str], bool],
        gpu_processes: (
            Callable[[Sequence[int]], Mapping[int, Sequence[int]]] | None
        ) = None,
        signal_process_group: Callable[[int, int], None] = os.killpg,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._spawn = spawn
        self._ready = ready
        self._gpu_processes = gpu_processes or nvidia_smi_gpu_compute_processes
        self._signal_process_group = signal_process_group
        self._monotonic = monotonic
        self._sleep = sleep
        self._process: Any | None = None
        self._launch: LaunchSpec | None = None
        self._gpu_ids: tuple[int, ...] = ()
        self._gpu_pid_baseline: dict[int, tuple[int, ...]] = {}
        self._process_group_id: int | None = None
        self._closed = False
        self._host = ""
        self._port = 0
        self._instance_token = ""

    def _snapshot_gpu_processes(self) -> dict[int, tuple[int, ...]]:
        observed = self._gpu_processes(self._gpu_ids)
        if not isinstance(observed, Mapping):
            raise RuntimeError("GPU compute-process query returned a non-mapping")
        snapshot: dict[int, tuple[int, ...]] = {}
        for gpu_id in self._gpu_ids:
            if gpu_id not in observed:
                raise RuntimeError(
                    f"GPU compute-process query omitted service GPU {gpu_id}"
                )
            pids = tuple(int(value) for value in observed[gpu_id])
            if any(pid <= 0 for pid in pids) or len(set(pids)) != len(pids):
                raise RuntimeError(
                    f"GPU compute-process query returned invalid PIDs for GPU {gpu_id}"
                )
            snapshot[gpu_id] = tuple(sorted(pids))
        return snapshot

    def owned_gpu_processes(self) -> dict[int, tuple[int, ...]]:
        """Return service-lifetime GPU PIDs, excluding the launch baseline."""

        if self._closed:
            return {}
        current = self._snapshot_gpu_processes()
        owned = {
            gpu_id: tuple(
                sorted(set(current[gpu_id]) - set(self._gpu_pid_baseline[gpu_id]))
            )
            for gpu_id in self._gpu_ids
        }
        return {gpu_id: pids for gpu_id, pids in owned.items() if pids}

    def ownership(self) -> dict[str, Any]:
        """Return JSON-safe process ownership evidence for durable recovery."""

        return {
            "service_process_group_id": self._process_group_id,
            "service_gpu_pid_baseline": {
                str(gpu_id): list(self._gpu_pid_baseline.get(gpu_id, ()))
                for gpu_id in self._gpu_ids
            },
        }

    def _wait_for_leader(self, process: Any, *, timeout: float) -> bool:
        deadline = self._monotonic() + timeout
        while process.poll() is None:
            remaining = deadline - self._monotonic()
            if remaining <= 0.0:
                return False
            self._sleep(min(0.1, remaining))
        return True

    def _wait_for_reclaim(
        self, process: Any, *, timeout: float
    ) -> tuple[bool, dict[int, tuple[int, ...]], Exception | None]:
        deadline = self._monotonic() + timeout
        last_owned: dict[int, tuple[int, ...]] = {}
        last_error: Exception | None = None
        while True:
            leader_exited = process.poll() is not None
            try:
                last_owned = self.owned_gpu_processes()
                last_error = None
            except Exception as exc:
                last_error = exc
            if leader_exited and last_error is None and not last_owned:
                return True, {}, None
            remaining = deadline - self._monotonic()
            if remaining <= 0.0:
                return False, last_owned, last_error
            self._sleep(min(0.1, remaining))

    def _signal_group(self, process: Any, signum: int) -> Exception | None:
        try:
            if self._process_group_id is not None:
                self._signal_process_group(self._process_group_id, signum)
            elif signum == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()
        except ProcessLookupError:
            return None
        except Exception as exc:
            return exc
        return None

    def start(self, launch: LaunchSpec, *, startup_timeout_seconds: float) -> None:
        if self._process is not None:
            raise RuntimeError("vLLM service supervisor is already active")
        try:
            self._start(launch, startup_timeout_seconds=startup_timeout_seconds)
        except BaseException as error:
            # start() can be interrupted before a driver acquires the supervisor.
            # It must release its own partially started service in that case.
            try:
                self.close(shutdown_timeout_seconds=5.0)
            except Exception as cleanup_error:
                if hasattr(error, "add_note"):
                    error.add_note(f"vLLM startup cleanup failed: {cleanup_error}")
            raise

    def _start(self, launch: LaunchSpec, *, startup_timeout_seconds: float) -> None:
        if self._process is not None:
            raise RuntimeError("vLLM service supervisor is already active")
        timeout = float(startup_timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("vLLM startup timeout must be finite and positive")
        command = list(launch.command)
        host = command[command.index("--host") + 1]
        port = int(command[command.index("--port") + 1])
        token_options = [
            i
            for i, value in enumerate(command)
            if value == "--instance-token" or value.startswith("--instance-token=")
        ]
        if len(token_options) != 1:
            raise ValueError("service launch requires exactly one --instance-token")
        token_index = token_options[0]
        if command[token_index] == "--instance-token":
            if token_index + 1 >= len(command):
                raise ValueError("service launch --instance-token is missing its value")
            instance_token = validate_private_service_instance_token(
                command[token_index + 1]
            )

            command[token_index : token_index + 2] = [
                f"--instance-token={instance_token}"
            ]
        else:
            instance_token = validate_private_service_instance_token(
                command[token_index].split("=", 1)[1]
            )
        launch = replace(launch, command=tuple(command))
        gpu_ids = tuple(
            int(value)
            for value in launch.environment["CUDA_VISIBLE_DEVICES"].split(",")
        )
        self._launch = launch
        self._gpu_ids = gpu_ids
        self._host = host
        self._port = port
        self._instance_token = instance_token
        self._closed = False
        if self._gpu_processes is not None:
            self._gpu_pid_baseline = self._snapshot_gpu_processes()
        self._process = self._spawn(launch)
        process_pid = getattr(self._process, "pid", None)
        self._process_group_id = (
            int(process_pid)
            if isinstance(process_pid, int)
            and not isinstance(process_pid, bool)
            and process_pid > 0
            else None
        )
        started = self._monotonic()
        while True:
            returncode = self._process.poll()
            if returncode is not None:
                try:
                    self.close()
                except Exception as teardown_error:
                    raise RuntimeError(
                        "vLLM service exited before readiness and teardown failed "
                        f"exit_code={returncode} host={host} port={port}: "
                        f"{teardown_error}"
                    ) from teardown_error
                raise RuntimeError(
                    "vLLM service exited before readiness "
                    f"exit_code={returncode} host={host} port={port}; inspect "
                    "bridge-route1-service-startup-error above"
                )
            if self._ready(host, port, instance_token):
                return
            if self._monotonic() - started >= timeout:
                try:
                    self.close()
                except Exception as teardown_error:
                    raise TimeoutError(
                        "vLLM service readiness timed out and teardown failed: "
                        f"{teardown_error}"
                    ) from teardown_error
                raise TimeoutError("vLLM service readiness timed out")
            self._sleep(min(0.25, timeout))

    @uninterrupted_cleanup()
    def close(self, *, shutdown_timeout_seconds: float = 30.0) -> None:
        process = self._process
        timeout = float(shutdown_timeout_seconds)
        if isinstance(sys.exc_info()[1], (KeyboardInterrupt, SystemExit)):
            timeout = min(timeout, 5.0)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("vLLM shutdown timeout must be finite and positive")
        if self._closed:
            return
        if process is None:
            self._closed = True
            return

        cleanup_message("[cleanup] stopping owned vLLM service and workers")
        # EngineCore/torchrun workers may create their own sessions. Snapshot
        # ancestry before the service leader exits and those workers reparent.
        tree = (
            OwnedProcessTree(
                process.pid, getattr(process, "_think_bridge_owner_tag", None)
            )
            if self._process_group_id is not None
            else None
        )

        signal_errors: list[str] = []
        if process.poll() is None:
            # Let the HTTP leader drain handlers and call backend/AsyncLLM close
            # before signalling its DP worker group directly.
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            except Exception as exc:
                signal_errors.append(f"leader_SIGTERM={type(exc).__name__}: {exc}")
            self._wait_for_leader(process, timeout=timeout)

        reclaimed, owned, query_error = self._wait_for_reclaim(
            process, timeout=min(timeout, 0.1)
        )
        if not reclaimed:
            error = self._signal_group(process, signal.SIGTERM)
            if tree is not None:
                tree.signal_descendants(signal.SIGTERM)
            if error is not None:
                signal_errors.append(f"group_SIGTERM={type(error).__name__}: {error}")
            reclaimed, owned, query_error = self._wait_for_reclaim(
                process, timeout=timeout
            )
        if not reclaimed:
            error = self._signal_group(process, signal.SIGKILL)
            if tree is not None:
                tree.signal_descendants(signal.SIGKILL)
            if error is not None:
                signal_errors.append(f"group_SIGKILL={type(error).__name__}: {error}")
            reclaimed, owned, query_error = self._wait_for_reclaim(
                process, timeout=timeout
            )
        if reclaimed:
            # GPU reclamation does not account for CPU-only coordinators or
            # multiprocessing helpers left after the service leader exits.
            error = self._signal_group(process, signal.SIGKILL)
            if tree is not None:
                tree.signal_descendants(signal.SIGKILL)
                tree.wait_for_descendants(timeout)
            if error is not None:
                raise RuntimeError(
                    f"vLLM residual process-group cleanup failed: {error}"
                )
            self._closed = True
            self._process = None
            return

        owned_detail = (
            ", ".join(
                f"gpu={gpu_id} pids={list(pids)}"
                for gpu_id, pids in sorted(owned.items())
            )
            or "owned_gpu_pids=unknown"
        )
        detail = [
            "vLLM private service was not reclaimed",
            f"process_group={self._process_group_id}",
            owned_detail,
            "baseline="
            + json.dumps(self.ownership()["service_gpu_pid_baseline"], sort_keys=True),
        ]
        if query_error is not None:
            detail.append(
                f"gpu_query={type(query_error).__name__}: {str(query_error)[:512]}"
            )
        detail.extend(signal_errors)
        raise RuntimeError("; ".join(detail))

    @property
    def active(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def status(self) -> dict[str, Any]:
        """Return bounded runtime process and health state for failure logs."""

        process = self._process
        returncode = None if process is None else process.poll()
        active = process is not None and returncode is None
        healthy = bool(
            active and self._ready(self._host, self._port, self._instance_token)
        )
        return {
            "active": active,
            "exit_code": returncode,
            "healthy": healthy,
        }


def spawn_private_service(launch: LaunchSpec) -> subprocess.Popen[str]:
    environment = dict(os.environ)
    for name in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        environment.pop(name, None)
    environment.update(
        {str(key): str(value) for key, value in launch.environment.items()}
    )
    return spawn_owned_process(
        list(launch.command),
        env=environment,
        text=True,
    )


def private_service_ready(host: str, port: int, instance_token: str) -> bool:
    if host != "127.0.0.1":
        return False
    try:
        expected_token = validate_private_service_instance_token(instance_token)
    except ValueError:
        return False
    try:
        with socket.create_connection((host, int(port)), timeout=0.25):
            pass
    except OSError:
        return False
    try:
        from think_bridge.training.vllm_client import BridgeRoute1ServiceClient

        return BridgeRoute1ServiceClient(
            host=host,
            port=int(port),
            timeout_seconds=1.0,
            max_response_bytes=4096,
            max_in_flight=1,
        ).health(expected_token)
    except (RuntimeError, ValueError):
        return False


def nvidia_smi_gpu_compute_processes(
    gpu_ids: Sequence[int],
) -> dict[int, tuple[int, ...]]:
    """Return exact compute PIDs per physical GPU or fail closed."""

    snapshot: dict[int, tuple[int, ...]] = {}
    for gpu_id in _gpu_tuple(gpu_ids, "compute-process GPU ids"):
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    str(gpu_id),
                    "--query-compute-apps=pid",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                text=True,
                capture_output=True,
                timeout=5.0,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                f"nvidia-smi compute-process query failed for GPU {gpu_id}"
            ) from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip()[:512]
            raise RuntimeError(
                f"nvidia-smi compute-process query failed for GPU {gpu_id} "
                f"exit_code={completed.returncode} detail={detail!r}"
            )
        active: list[int] = []
        for line in completed.stdout.splitlines():
            value = line.strip()
            if not value or "No running" in value:
                continue
            try:
                pid = int(value)
            except ValueError as exc:
                raise RuntimeError(
                    f"nvidia-smi returned an invalid compute PID for GPU {gpu_id}: "
                    f"{value!r}"
                ) from exc
            if pid <= 0:
                raise RuntimeError(
                    f"nvidia-smi returned a non-positive compute PID for GPU {gpu_id}"
                )
            active.append(pid)
        snapshot[gpu_id] = tuple(sorted(set(active)))
    return snapshot
