"""Capability-checked generation-only vLLM backend for Route1 evaluation.

Training, teacher scoring, direct/wrong controls, and all optimizer work remain
inside the torch/HF trainer. GPU dependencies stay lazy so protocol tests can
import this module on dependency-light development hosts.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping, Sequence

from think_bridge.training.vllm_runtime import (
    ROUTE1_GENERATION_PROBE_STATUS,
    Route1ServiceRequest,
    Route1ServiceResponse,
    TensorPayload,
)


_KZ = 64
_MAX_COLLECTIVE_REQUEST_BYTES = 128 * 1024 * 1024
_EMBED_PROBE_IDS = (0, 1, 2, 17)


def _installed_vllm_version() -> str:
    """Record the installed version without using it as an admission gate."""

    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("Bridge Route1 requires vLLM on the GPU server") from exc
    if not str(version).strip():
        raise RuntimeError("installed vLLM version metadata is empty")
    return str(version)


class _AsyncLoopThread:
    def __init__(self) -> None:
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread = threading.Thread(
            target=self._run_loop,
            name="bridge-route1-async-vllm",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=30.0) or self._loop is None:
            raise RuntimeError("vLLM asyncio runtime did not start")

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    def run(self, coroutine: Any, *, timeout: float) -> Any:
        loop = self._loop
        if loop is None or not self._thread.is_alive():
            raise RuntimeError("vLLM asyncio runtime is closed")
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        try:
            return future.result(timeout=float(timeout))
        except BaseException:
            future.cancel()
            raise

    def close(self) -> None:
        loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(loop.stop)
        self._thread.join(timeout=30.0)
        self._loop = None
        if self._thread.is_alive():
            raise RuntimeError("vLLM asyncio runtime did not stop")


class _AsyncUtilityFanout:
    """Capability-bound utility RPC across one or more internal DP engines."""

    def __init__(self, llm: Any, *, expected_dp_size: int) -> None:
        client = getattr(llm, "engine_core", None)
        utility = (
            None if client is None else getattr(client, "_call_utility_async", None)
        )
        if not callable(utility):
            raise RuntimeError(
                "vLLM prompt-embedding capability lacks EngineCore utility RPC"
            )
        expected = int(expected_dp_size)
        if expected <= 0:
            raise ValueError("vLLM data parallel size must be positive")
        core_engines = tuple(getattr(client, "core_engines", ()))
        try:
            signature = inspect.signature(utility)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "vLLM capability probe cannot inspect EngineCore utility RPC"
            ) from exc

        if core_engines:
            if len(core_engines) != expected or len(set(core_engines)) != expected:
                raise RuntimeError(
                    "vLLM internal-DP utility topology differs from requested DP: "
                    f"engines={len(core_engines)} requested={expected}"
                )
            try:
                signature.bind(
                    "collective_rpc",
                    "bridge_route1_probe",
                    1.0,
                    (),
                    {},
                    engine=core_engines[0],
                )
            except TypeError as exc:
                raise RuntimeError(
                    "vLLM internal-DP utility RPC lacks the engine capability"
                ) from exc
            self._engines: tuple[Any | None, ...] = core_engines
        else:
            if expected != 1:
                raise RuntimeError(
                    "vLLM requested multi-DP but exposes only a single EngineCore"
                )
            try:
                signature.bind("collective_rpc", "bridge_route1_probe", 1.0, (), {})
            except TypeError as exc:
                raise RuntimeError(
                    "vLLM single-DP utility RPC call shape is unsupported"
                ) from exc
            self._engines = (None,)
        self._client = client

    @property
    def dp_size(self) -> int:
        return len(self._engines)

    async def collective_rpc(
        self,
        method: str,
        *,
        timeout: float,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> list[Any]:
        if not isinstance(method, str) or not method:
            raise ValueError("DP utility RPC method must be a non-empty name")
        timeout_value = float(timeout)
        if not math.isfinite(timeout_value) or timeout_value <= 0.0:
            raise ValueError("DP utility RPC timeout must be finite and positive")
        calls = []
        for engine in self._engines:
            call_args = (
                "collective_rpc",
                method,
                timeout_value,
                tuple(args),
                dict(kwargs or {}),
            )
            calls.append(
                self._client._call_utility_async(*call_args)
                if engine is None
                else self._client._call_utility_async(*call_args, engine=engine)
            )
        values = await asyncio.wait_for(asyncio.gather(*calls), timeout=timeout_value)
        flattened: list[Any] = []
        for value in values:
            if not isinstance(value, list):
                raise RuntimeError(
                    "vLLM EngineCore collective RPC returned a non-list result"
                )
            flattened.extend(value)
        return flattened


def _strict_eos_token_ids(value: Any, *, source: str) -> tuple[int, ...]:
    if value is None:
        return ()
    raw_values = value if isinstance(value, (list, tuple)) else (value,)
    values: list[int] = []
    for item in raw_values:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise RuntimeError(f"vLLM {source} EOS identity is invalid")
        values.append(item)
    return tuple(values)


def _effective_model_eos_token_ids(
    model_config: Any, hf_config: Any
) -> tuple[int, ...]:
    getter = getattr(model_config, "try_get_generation_config", None)
    if not callable(getter):
        raise RuntimeError("vLLM generation-config EOS identity is unavailable")
    generation_config = getter()
    if not isinstance(generation_config, Mapping):
        raise RuntimeError("vLLM generation-config EOS identity is invalid")
    primary = _strict_eos_token_ids(
        getattr(hf_config, "eos_token_id", None), source="model-config"
    )
    configured = _strict_eos_token_ids(
        generation_config.get("eos_token_id"), source="generation-config"
    )
    merged = tuple(dict.fromkeys(primary + configured))
    if not merged:
        raise RuntimeError("vLLM model-native EOS identity is unavailable")
    return merged


def _normalize_native_prefix(
    token_ids: Sequence[int],
    *,
    finish_reason: str | None,
    stop_reason: Any,
    eos_token_id: int,
    model_eos_token_ids: Sequence[int],
    max_tokens: int,
) -> tuple[int, ...]:
    values = tuple(int(value) for value in token_ids)
    native_eos = tuple(int(value) for value in model_eos_token_ids)
    if (
        not native_eos
        or len(set(native_eos)) != len(native_eos)
        or any(value < 0 for value in native_eos)
        or int(eos_token_id) not in native_eos
    ):
        raise RuntimeError("vLLM model-native EOS identity is invalid")
    if finish_reason == "length":
        if len(values) != int(max_tokens):
            raise RuntimeError("vLLM length finish does not contain max_tokens")
        return values
    if finish_reason != "stop":
        raise RuntimeError("vLLM native generation lacks a terminal finish reason")
    if stop_reason is not None and (
        isinstance(stop_reason, bool)
        or not isinstance(stop_reason, int)
        or int(stop_reason) != int(eos_token_id)
    ):
        raise RuntimeError(
            "vLLM native generation stopped on a foreign token: "
            f"finish_reason={finish_reason!r} stop_reason={stop_reason!r}"
        )
    if values and values[-1] in native_eos:
        return values[:-1] + (int(eos_token_id),)
    if len(values) >= int(max_tokens):
        raise RuntimeError("vLLM EOS restoration would exceed max_tokens")
    return values + (int(eos_token_id),)


def _worker_model(worker: Any) -> tuple[Any, Any, Any]:
    runner = getattr(worker, "model_runner", None)
    if runner is None:
        runner = getattr(worker, "runner", None)
    get_model = None if runner is None else getattr(runner, "get_model", None)
    model = (
        get_model()
        if callable(get_model)
        else None
        if runner is None
        else getattr(runner, "model", None)
    )
    config = getattr(worker, "vllm_config", None)
    if runner is None or model is None or config is None:
        raise RuntimeError("vLLM worker does not expose model_runner/model/vllm_config")
    embed = getattr(model, "embed_input_ids", None)
    if not callable(embed):
        embed = getattr(model, "get_input_embeddings", None)
    if not callable(embed):
        raise RuntimeError("loaded vLLM model lacks the prompt-embedding primitive")
    return runner, model, config


def _parallel_identity(worker: Any, config: Any) -> tuple[int, int, int, int]:
    parallel = getattr(config, "parallel_config", None)
    dp_size = int(getattr(parallel, "data_parallel_size", 0))
    tp_size = int(getattr(parallel, "tensor_parallel_size", 0))
    global_rank = int(getattr(worker, "rank", -1))
    dp_value = getattr(parallel, "data_parallel_rank", None)
    tp_value = getattr(parallel, "tensor_parallel_rank", None)
    dp_rank = (
        int(dp_value)
        if dp_value is not None
        else (global_rank // tp_size if global_rank >= 0 and tp_size > 0 else -1)
    )
    # Modern vLLM isolates dense-model engines into DP=1 worker groups.
    # The engine index retains the global replica identity in that case.
    dp_index = getattr(parallel, "data_parallel_index", None)
    if dp_index is not None:
        index = int(dp_index)
        if (
            index < 0
            or (dp_size == 1 and dp_rank != 0)
            or (dp_size != 1 and index != dp_rank)
        ):
            raise RuntimeError("vLLM worker engine and parallel ranks disagree")
        dp_rank = index
    tp_rank = (
        int(tp_value)
        if tp_value is not None
        else (global_rank % tp_size if global_rank >= 0 and tp_size > 0 else 0)
    )
    if (
        dp_size <= 0
        or tp_size <= 0
        or dp_rank < 0
        or ((dp_size != 1 or dp_index is None) and dp_rank >= dp_size)
        or not 0 <= tp_rank < tp_size
    ):
        raise RuntimeError("vLLM worker parallel identity is unavailable")
    return dp_rank, dp_size, tp_rank, tp_size


def _embed_ids(torch: Any, model: Any, values: Sequence[int], device: Any) -> Any:
    ids = torch.tensor(tuple(int(value) for value in values), device=device).long()
    embed = getattr(model, "embed_input_ids", None)
    if not callable(embed):
        embed = getattr(model, "get_input_embeddings", None)
    if not callable(embed):
        raise RuntimeError("vLLM Qwen embedding primitive is unavailable")
    try:
        embedded = embed(ids)
    except TypeError:
        embedded = embed(input_ids=ids)
    if embedded.ndim != 2 or embedded.size(0) != ids.numel():
        raise RuntimeError("vLLM Qwen embedding output is misaligned")
    return embedded


def _prompt_embedding_segments(
    prompt: Any,
    z: Any | None,
    boundary: Any,
    *,
    append_boundary: bool,
) -> tuple[Any, ...]:
    segments = (prompt,) if z is None else (prompt, z)
    return (*segments, boundary) if append_boundary else segments


def _generation_data_parallel_rank(
    request: Route1ServiceRequest,
    row: int,
    *,
    data_parallel_size: int,
) -> int:
    size = int(data_parallel_size)
    row_index = int(row)
    if size <= 0:
        raise ValueError("vLLM data parallel size must be positive")
    if row_index < 0 or row_index >= len(request.sample_ids):
        raise IndexError("generation row is outside the request")
    return int(request.sample_ids[row_index]) % size


def _resolve_local_model_dir(model_name_or_path: str) -> Path:
    candidate = Path(model_name_or_path).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(repo_id=model_name_or_path, local_files_only=True)
    ).resolve()


def _tensor_sha256(torch: Any, tensor: Any) -> str:
    digest = hashlib.sha256()
    cpu = tensor.detach().to(device="cpu").contiguous()
    rows = max(1, min(4096, int(cpu.shape[0])))
    for start in range(0, int(cpu.shape[0]), rows):
        value = cpu[start : start + rows]
        if value.dtype == torch.bfloat16:
            value = value.view(torch.uint16)
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _load_cpu_embedding_table(
    model_name_or_path: str, *, prompt_dtype: str
) -> tuple[Any, Mapping[str, Any]]:
    import torch
    from safetensors import safe_open

    model_dir = _resolve_local_model_dir(model_name_or_path)
    aliases = ("model.embed_tokens.weight", "transformer.wte.weight")
    index_path = model_dir / "model.safetensors.index.json"
    shard: Path | None = None
    key: str | None = None
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map", {})
        for alias in aliases:
            if alias in weight_map:
                key = alias
                shard = model_dir / str(weight_map[alias])
                break
    else:
        for path in sorted(model_dir.glob("*.safetensors")):
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                for alias in aliases:
                    if alias in handle.keys():
                        key = alias
                        shard = path
                        break
            if shard is not None:
                break
    if shard is None or key is None or not shard.is_file():
        raise RuntimeError("model safetensors lack the input embedding weight")
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        source = handle.get_tensor(key).contiguous()
    if source.ndim != 2:
        raise RuntimeError("input embedding weight is not rank two")
    source_dtype = str(source.dtype).removeprefix("torch.")
    source_sha256 = _tensor_sha256(torch, source)
    dtype = torch.bfloat16 if prompt_dtype == "bfloat16" else torch.float32
    table = source.to(dtype=dtype).contiguous()
    revision = model_dir.name if model_dir.parent.name == "snapshots" else "local"
    identity = {
        "model_source": str(model_name_or_path),
        "resolved_model_dir": str(model_dir),
        "revision": revision,
        "weight_key": key,
        "weight_shard": shard.name,
        "source_dtype": source_dtype,
        "prompt_dtype": prompt_dtype,
        "vocab_size": int(table.size(0)),
        "hidden_size": int(table.size(1)),
        "source_sha256": source_sha256,
    }
    return table, identity


def _embedding_probe_payload(torch: Any, table: Any) -> TensorPayload:
    ids = torch.tensor(_EMBED_PROBE_IDS, dtype=torch.long)
    return TensorPayload.from_torch(table.index_select(0, ids))


class BridgeRoute1WorkerExtension:
    """Minimal worker extension used only for prompt-embedding capability probes."""

    def bridge_route1_probe(self) -> Mapping[str, Any]:
        import torch

        _runner, model, config = _worker_model(self)
        dp_rank, dp_size, tp_rank, tp_size = _parallel_identity(self, config)
        device = next(model.parameters()).device
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Bridge Route1 worker probe requires CUDA")
        embedded = _embed_ids(torch, model, _EMBED_PROBE_IDS, device)
        model_config = getattr(config, "model_config", None)
        hf_config = getattr(model_config, "hf_config", None)
        vocab_size = int(getattr(hf_config, "vocab_size", 0))
        if vocab_size <= max(_EMBED_PROBE_IDS):
            raise RuntimeError("vLLM worker vocabulary identity is invalid")
        payload = TensorPayload.from_torch(embedded)
        return {
            "dp_rank": dp_rank,
            "dp_size": dp_size,
            "dp_identity_source": (
                "data_parallel_index"
                if getattr(config.parallel_config, "data_parallel_index", None)
                is not None
                else "data_parallel_rank"
            ),
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "device": str(device),
            "hidden_size": int(embedded.size(-1)),
            "vocab_size": vocab_size,
            "prompt_dtype": payload.dtype,
            "embedding_probe": payload.to_wire(),
            "eos_token_ids": _effective_model_eos_token_ids(model_config, hf_config),
        }

    def bridge_route1_release(self) -> Mapping[str, Any]:
        import gc
        import torch

        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return {"rank": int(getattr(self, "rank", -1)), "released": True}


class BridgeRoute1VLLMBackend:
    """Private frontend for configurable generation-only DP/TP topology."""

    def __init__(
        self,
        *,
        model_name_or_path: str,
        data_parallel_size: int,
        tensor_parallel_size: int,
        worker_extension_cls: str,
        gpu_memory_utilization: float,
        max_num_seqs: int,
        request_timeout_seconds: float,
        physical_chunk_size: int,
        enforce_eager: bool,
        seed: int = 42,
    ) -> None:
        # Respect an explicitly supplied environment; do not force an extra
        # determinism policy on training or evaluation engines.
        self.batch_invariant = os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1"
        self.vllm_version = _installed_vllm_version()
        dp_size = int(data_parallel_size)
        tp_size = int(tensor_parallel_size)
        if dp_size <= 0 or tp_size <= 0:
            raise ValueError("vLLM DP/TP sizes must be positive")
        if not 0.0 < float(gpu_memory_utilization) < 1.0:
            raise ValueError("vLLM GPU memory utilization is invalid")
        if not isinstance(enforce_eager, bool):
            raise ValueError("vLLM enforce-eager setting must be boolean")
        if (
            int(max_num_seqs) <= 0
            or not math.isfinite(request_timeout_seconds)
            or float(request_timeout_seconds) <= 0.0
        ):
            raise ValueError("vLLM service bounds are invalid")
        # A request may queue more sequences than the engine runs concurrently.
        if int(physical_chunk_size) <= 0:
            raise ValueError("vLLM service batch must be positive")
        self._seed = int(seed)
        self._data_parallel_size = dp_size
        self._tensor_parallel_size = tp_size
        self._timeout = float(request_timeout_seconds)
        self._physical_chunk_size = int(physical_chunk_size)
        self._request_condition = threading.Condition()
        self._active_requests = 0
        self._closing = False
        self._runtime = _AsyncLoopThread()
        self._llm: Any | None = None
        self._embedding_table: Any | None = None
        startup_timings: dict[str, float] = {}
        try:
            started = time.perf_counter()
            self._llm = self._runtime.run(
                self._build_async_llm(
                    model_name_or_path=str(model_name_or_path),
                    data_parallel_size=dp_size,
                    tensor_parallel_size=tp_size,
                    worker_extension_cls=str(worker_extension_cls),
                    gpu_memory_utilization=float(gpu_memory_utilization),
                    max_num_seqs=int(max_num_seqs),
                    enforce_eager=enforce_eager,
                ),
                timeout=self._timeout,
            )
            startup_timings["async_engine"] = round(time.perf_counter() - started, 3)
            self._dp_rpc = _AsyncUtilityFanout(self._llm, expected_dp_size=dp_size)
            started = time.perf_counter()
            try:
                probes = self._runtime.run(
                    self._dp_rpc.collective_rpc(
                        "bridge_route1_probe", timeout=self._timeout
                    ),
                    timeout=self._timeout,
                )
                self._validate_worker_probes(
                    probes,
                    data_parallel_size=dp_size,
                    tensor_parallel_size=tp_size,
                )
            except Exception as exc:
                topology = f"DP={dp_size}/TP={tp_size}"
                raise RuntimeError(
                    "vLLM prompt-embedding private capability probe failed for "
                    f"{topology}: {exc}"
                ) from exc
            startup_timings["worker_identity"] = round(time.perf_counter() - started, 3)
            widths = {int(probe["hidden_size"]) for probe in probes}
            vocabs = {int(probe["vocab_size"]) for probe in probes}
            dtypes = {str(probe["prompt_dtype"]) for probe in probes}
            eos_identities = {
                tuple(int(value) for value in probe["eos_token_ids"])
                for probe in probes
            }
            probe_bytes = {bytes(probe["embedding_probe"]) for probe in probes}
            prompt_dtype = next(iter(dtypes))
            self._model_eos_token_ids = next(iter(eos_identities))
            started = time.perf_counter()
            self._embedding_table, identity = _load_cpu_embedding_table(
                str(model_name_or_path), prompt_dtype=prompt_dtype
            )
            if int(identity["hidden_size"]) != next(iter(widths)) or int(
                identity["vocab_size"]
            ) != next(iter(vocabs)):
                raise RuntimeError(
                    "frontend embedding table and vLLM model geometry differ"
                )
            import torch

            frontend_probe = _embedding_probe_payload(torch, self._embedding_table)
            if any(
                TensorPayload.from_wire(value) != frontend_probe
                for value in probe_bytes
            ):
                raise RuntimeError(
                    "frontend embedding bytes and vLLM replica bytes differ"
                )
            self.embedding_identity = dict(identity)
            startup_timings["frontend_embedding"] = round(
                time.perf_counter() - started, 3
            )
            probe_request = self._startup_probe_request(
                hidden_size=int(identity["hidden_size"]),
                eos_token_id=int(self._model_eos_token_ids[0]),
            )
            started = time.perf_counter()
            probe_prefixes = self._probe_native_generate(probe_request)
            startup_timings["native_rollout_probe"] = round(
                time.perf_counter() - started, 3
            )
            self._generation_response(
                probe_request,
                probe_prefixes,
                rollout_seconds=0.0,
                service_queue_seconds=0.0,
            ).assert_matches(probe_request)
            self.startup_timings = startup_timings
            self.probe_status = ROUTE1_GENERATION_PROBE_STATUS
        except BaseException:
            with contextlib.suppress(Exception):
                self.close()
            raise

    @staticmethod
    def _validate_worker_probes(
        probes: Any, *, data_parallel_size: int, tensor_parallel_size: int
    ) -> None:
        if not isinstance(probes, list):
            raise RuntimeError("vLLM worker capability probe returned no result list")
        expected_pairs = {
            (dp_rank, tp_rank)
            for dp_rank in range(int(data_parallel_size))
            for tp_rank in range(int(tensor_parallel_size))
        }
        observed_pairs = {
            (int(probe["dp_rank"]), int(probe["tp_rank"])) for probe in probes
        }
        topology = [
            {
                key: probe.get(key)
                for key in (
                    "dp_rank",
                    "dp_size",
                    "dp_identity_source",
                    "tp_rank",
                    "tp_size",
                    "device",
                )
            }
            for probe in probes
        ]
        if (
            len(probes) != len(expected_pairs)
            or observed_pairs != expected_pairs
            or len({int(probe["dp_size"]) for probe in probes}) != 1
            or any(
                int(probe["dp_size"]) != int(data_parallel_size)
                and not (
                    int(probe["dp_size"]) == 1
                    and probe.get("dp_identity_source") == "data_parallel_index"
                )
                for probe in probes
            )
            or any(
                int(probe["tp_size"]) != int(tensor_parallel_size) for probe in probes
            )
        ):
            raise RuntimeError(
                "vLLM worker topology mismatch: "
                f"expected={sorted(expected_pairs)} observed={json.dumps(topology, sort_keys=True)}"
            )
        for field in (
            "hidden_size",
            "vocab_size",
            "prompt_dtype",
            "eos_token_ids",
            "embedding_probe",
        ):
            values = [
                tuple(probe[field])
                if field == "eos_token_ids"
                else bytes(probe[field])
                if field == "embedding_probe"
                else probe[field]
                for probe in probes
            ]
            if any(value != values[0] for value in values[1:]):
                details = (
                    [hashlib.sha256(value).hexdigest() for value in values]
                    if field == "embedding_probe"
                    else values
                )
                raise RuntimeError(
                    f"vLLM worker {field} mismatch: values={details} "
                    f"workers={json.dumps(topology, sort_keys=True)}"
                )

    async def _build_async_llm(
        self,
        *,
        model_name_or_path: str,
        data_parallel_size: int,
        tensor_parallel_size: int,
        worker_extension_cls: str,
        gpu_memory_utilization: float,
        max_num_seqs: int,
        enforce_eager: bool,
    ) -> Any:
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        engine_args = AsyncEngineArgs(
            model=model_name_or_path,
            seed=self._seed,
            tensor_parallel_size=int(tensor_parallel_size),
            data_parallel_size=int(data_parallel_size),
            worker_extension_cls=worker_extension_cls,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            enforce_eager=enforce_eager,
            enable_prompt_embeds=True,
            disable_log_stats=False,
            enable_log_requests=False,
        )
        return AsyncLLM.from_engine_args(engine_args)

    def _startup_probe_request(
        self, *, hidden_size: int, eos_token_id: int
    ) -> Route1ServiceRequest:
        import torch

        count = self._data_parallel_size
        zeros = torch.zeros((count, _KZ, hidden_size), dtype=torch.float32)
        return Route1ServiceRequest(
            run_id="bridge-route1-startup-probe",
            transaction_id="bridge-route1-startup-probe",
            rank=0,
            step=0,
            micro_step=0,
            chunk_id=0,
            sample_ids=tuple(range(count)),
            sample_keys=tuple(f"startup-probe-{row}" for row in range(count)),
            sample_is_padding=(False,) * count,
            prompt_ids=((0,),) * count,
            prompt_lengths=(1,) * count,
            detached_z=TensorPayload.from_torch(zeros),
            boundary_ids=(0,),
            generation_seed=self._seed,
            max_prefix_tokens=1,
            z_present=(True,) * count,
            append_boundary=(True,) * count,
            eos_token_id=int(eos_token_id),
        )

    def _scheduler_prompts(
        self, request: Route1ServiceRequest
    ) -> list[Mapping[str, Any]]:
        import torch

        z = request.detached_z.to_torch(dtype=self._embedding_table.dtype)
        if (
            z.ndim != 3
            or z.shape[1] not in {32, 64, 128}
            or z.shape[2] != self._embedding_table.shape[1]
        ):
            raise RuntimeError("Route1 scheduler z geometry is invalid")
        boundary = self._embedding_table.index_select(
            0, torch.tensor(request.boundary_ids, dtype=torch.long)
        )
        prompts: list[Mapping[str, Any]] = []
        for row, token_ids in enumerate(request.prompt_ids):
            prompt = self._embedding_table.index_select(
                0, torch.tensor(token_ids, dtype=torch.long)
            )
            prompt_embeds = torch.cat(
                _prompt_embedding_segments(
                    prompt,
                    (z[row] if request.z_present[row] else None),
                    boundary,
                    append_boundary=(request.append_boundary[row]),
                ),
                dim=0,
            ).contiguous()
            prompts.append({"prompt_embeds": prompt_embeds})
        return prompts

    async def _native_generate_async(
        self, request: Route1ServiceRequest
    ) -> tuple[tuple[int, ...], ...]:
        from vllm import SamplingParams

        prompts = self._scheduler_prompts(request)
        sampling = SamplingParams(
            temperature=float(request.temperature),
            top_p=1.0,
            max_tokens=int(request.max_prefix_tokens),
            seed=int(request.generation_seed),
            # Torch greedy_true_z_prefix stops only on the sealed tokenizer
            # EOS.  Disable vLLM's possibly wider model-native EOS set and
            # restore that one exact stop token explicitly.
            ignore_eos=True,
            stop_token_ids=[int(request.eos_token_id)],
        )

        async def generate_one(row: int, prompt: Mapping[str, Any]) -> Any:
            output = None
            request_id = (
                f"{request.transaction_id}:s{request.step}:m{request.micro_step}:"
                f"c{request.chunk_id}:r{row}"
            )
            async for current in self._llm.generate(
                prompt,
                sampling,
                request_id,
                data_parallel_rank=_generation_data_parallel_rank(
                    request, row, data_parallel_size=self._data_parallel_size
                ),
            ):
                output = current
            if output is None or not bool(getattr(output, "finished", False)):
                raise RuntimeError("vLLM native generation did not finish its request")
            return output

        outputs = await asyncio.gather(
            *(generate_one(row, prompt) for row, prompt in enumerate(prompts))
        )
        prefixes: list[tuple[int, ...]] = []
        for output in outputs:
            candidates = getattr(output, "outputs", None)
            if not candidates or len(candidates) != 1:
                raise RuntimeError("vLLM native generation returned invalid candidates")
            candidate = candidates[0]
            token_ids = _normalize_native_prefix(
                candidate.token_ids,
                finish_reason=getattr(candidate, "finish_reason", None),
                stop_reason=getattr(candidate, "stop_reason", None),
                eos_token_id=int(request.eos_token_id),
                model_eos_token_ids=self._model_eos_token_ids,
                max_tokens=int(request.max_prefix_tokens),
            )
            if not token_ids or len(token_ids) > request.max_prefix_tokens:
                raise RuntimeError("vLLM native generation returned an invalid prefix")
            prefixes.append(token_ids)
        return tuple(prefixes)

    def _probe_native_generate(
        self, request: Route1ServiceRequest
    ) -> tuple[tuple[int, ...], ...]:
        return self._runtime.run(
            self._native_generate_async(request), timeout=self._timeout
        )

    @contextlib.contextmanager
    def _request_scope(self) -> Any:
        with self._request_condition:
            if self._closing or self._llm is None:
                raise RuntimeError("vLLM Route1 backend is closed")
            self._active_requests += 1
        try:
            yield
        finally:
            with self._request_condition:
                self._active_requests -= 1
                if self._active_requests < 0:
                    raise RuntimeError("vLLM active request accounting underflow")
                self._request_condition.notify_all()

    def _generate_request(
        self, request: Route1ServiceRequest, *, timeout: float
    ) -> tuple[tuple[tuple[int, ...], ...], float, float]:
        remaining = float(timeout)
        if not math.isfinite(remaining) or remaining <= 0.0:
            raise TimeoutError("Route1 generation has no remaining request budget")
        started = time.perf_counter()
        prefixes = self._runtime.run(
            self._native_generate_async(request), timeout=remaining
        )
        return prefixes, time.perf_counter() - started, 0.0

    @staticmethod
    def _generation_response(
        request: Route1ServiceRequest,
        prefixes: Sequence[Sequence[int]],
        *,
        rollout_seconds: float,
        service_queue_seconds: float,
        service_active_requests: int = 1,
    ) -> Route1ServiceResponse:
        response = Route1ServiceResponse(
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
            prefix_token_ids=tuple(
                tuple(int(value) for value in row) for row in prefixes
            ),
            prefix_token_masks=tuple(tuple(True for _ in row) for row in prefixes),
            timings={
                "service_queue_seconds": float(service_queue_seconds),
                "rollout_generation_seconds": float(rollout_seconds),
                "service_request_seconds": float(
                    service_queue_seconds + rollout_seconds
                ),
            },
            counts={
                "service_batch_rows": len(request.sample_ids),
                "service_real_rows": sum(
                    not value for value in request.sample_is_padding
                ),
                "service_active_requests": int(service_active_requests),
            },
        )
        response.assert_matches(request)
        return response

    def execute(
        self,
        request: Route1ServiceRequest,
        *,
        service_queue_seconds: float = 0.0,
        service_active_requests: int = 1,
    ) -> Route1ServiceResponse:
        # Greedy evaluation has its own canonical request width, independent of
        # the sampled training microbatch. Wire shapes and byte limits remain checked.
        if (
            request.temperature != 0.0
            and len(request.sample_ids) != self._physical_chunk_size
        ):
            raise RuntimeError("Sampled Route1 request does not fill its service batch")
        if int(request.eos_token_id) not in self._model_eos_token_ids:
            raise RuntimeError("Route1 request EOS differs from vLLM model identity")
        request_bytes = len(request.to_wire())
        if request_bytes > _MAX_COLLECTIVE_REQUEST_BYTES:
            raise RuntimeError(
                "Route1 request exceeds the collective byte bound: "
                f"request_bytes={request_bytes} "
                f"limit_bytes={_MAX_COLLECTIVE_REQUEST_BYTES} "
                f"rows={len(request.sample_ids)} z_shape={request.detached_z.shape}. "
                "For benchmark evaluation, reduce --batch_size "
                "(EVAL_VLLM_BATCH_SIZE in batch evaluation scripts)."
            )
        with self._request_scope():
            prefixes, rollout_seconds, queue_seconds = self._generate_request(
                request, timeout=self._timeout
            )
            return self._generation_response(
                request,
                prefixes,
                rollout_seconds=rollout_seconds,
                service_queue_seconds=(
                    float(service_queue_seconds) + float(queue_seconds)
                ),
                service_active_requests=int(service_active_requests),
            )

    def close(self) -> None:
        condition = getattr(self, "_request_condition", None)
        if condition is not None:
            deadline = time.monotonic() + float(self._timeout)
            with condition:
                self._closing = True
                while self._active_requests:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        raise RuntimeError(
                            "vLLM Route1 teardown timed out with active requests"
                        )
                    condition.wait(timeout=remaining)
        llm = getattr(self, "_llm", None)
        runtime = getattr(self, "_runtime", None)
        if llm is None and runtime is None:
            return
        teardown_errors: list[tuple[str, Exception]] = []
        if llm is not None and hasattr(self, "_dp_rpc"):
            try:
                runtime.run(
                    self._dp_rpc.collective_rpc(
                        "bridge_route1_release", timeout=self._timeout
                    ),
                    timeout=self._timeout,
                )
            except Exception as exc:
                teardown_errors.append(("worker_release", exc))
        if llm is not None:

            async def shutdown_async() -> None:
                llm.shutdown()
                await asyncio.sleep(0)

            try:
                runtime.run(shutdown_async(), timeout=self._timeout)
            except Exception as exc:
                teardown_errors.append(("async_llm_shutdown", exc))
        self._llm = None
        self._embedding_table = None
        if runtime is not None:
            try:
                runtime.close()
            except Exception as exc:
                teardown_errors.append(("runtime_close", exc))
        self._runtime = None
        if teardown_errors:
            detail = "; ".join(
                f"{phase}={type(error).__name__}: {str(error)[:512]}"
                for phase, error in teardown_errors
            )
            raise RuntimeError(
                f"vLLM backend teardown failed: {detail}"
            ) from teardown_errors[0][1]
