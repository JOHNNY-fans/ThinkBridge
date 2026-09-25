"""Public benchmark execution, migrated from the maintained research evaluator.

Uses the public sealed-checkpoint loader; HF and single-device vLLM answer execution.
"""

from __future__ import annotations
from dataclasses import dataclass
import json, math, statistics, time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from think_bridge.model.inference import (
    RuntimeBundle,
    CheckpointComponent,
    _build_runtime,
)
from think_bridge.model.reasoner_inference import reasoner_inference_metadata
from think_bridge.data.dataset import load_data_file as load_json_records


@dataclass(frozen=True)
class AnswerBatchResult:
    z: Any
    outputs: tuple[Mapping[str, Any], ...]
    warmed: bool
    ttft_seconds: float | None = None
    reasoner_seconds: float | None = None
    answer_seconds: float | None = None
    input_preparation_seconds: float = 0.0
    response_censored: bool = False
    first_response_token_ids: tuple[int, ...] = ()
    row_ttft_seconds: tuple[float | None, ...] = ()
    row_first_response_token_ids: tuple[tuple[int, ...], ...] = ()


def answer_generation_limit(measurement: str, max_new_tokens: int) -> int:
    if measurement in {"ttft", "accuracy"} and int(max_new_tokens) > 0:
        return int(max_new_tokens)
    raise ValueError("answer generation measurement or horizon is invalid")


def _execute_answer_batch(
    *,
    measurement: str,
    max_new_tokens: int,
    warmup_samples: int,
    already_warmed: bool,
    reason: Callable[[], Any],
    generate: Callable[[Any, int], Sequence[Mapping[str, Any]]],
    synchronize: Callable[[], None],
    clock: Callable[[], float] = time.perf_counter,
    prepare: Callable[[], None] | None = None,
    request_started: float | None = None,
    generate_timed: Callable[
        [Any, int, Callable[[int, int], bool | None]], Sequence[Mapping[str, Any]]
    ]
    | None = None,
    response_tokenizer: Any = None,
    response_started: bool = True,
    stop_after_response: bool = False,
    response_batch_size: int = 1,
) -> AnswerBatchResult:
    """Execute one answer batch with an explicit end-to-end TTFT boundary."""

    limit = answer_generation_limit(measurement, max_new_tokens)
    if measurement == "ttft":
        from think_bridge.eval.response_timing import ResponseTokenTimer

        if response_batch_size < 1:
            raise ValueError("response timing requires a positive batch size")

        if generate_timed is None or response_tokenizer is None:
            raise RuntimeError(
                "response TTFT requires an incremental token callback backend"
            )

    def observer(timers):
        def observe(row_index, token_id):
            if not 0 <= row_index < len(timers):
                raise RuntimeError("response callback row outside the request batch")
            timer = timers[row_index]
            timer(0, token_id)
            return stop_after_response and timer.seconds is not None

        return observe

    warmed = bool(already_warmed)
    if measurement == "ttft" and not warmed:
        if request_started is not None and int(warmup_samples) > 0:
            raise ValueError("warmup must finish before the request arrives")
        if prepare is not None:
            prepare()
        for _ in range(int(warmup_samples)):
            synchronize()
            warm_z = reason()
            if stop_after_response:
                warm_started = clock()
                warm_timers = [
                    ResponseTokenTimer(
                        response_tokenizer,
                        started=warm_started,
                        synchronize=synchronize,
                        clock=clock,
                        response_started=response_started,
                    )
                    for _ in range(response_batch_size)
                ]
                generate_timed(warm_z, limit, observer(warm_timers))
            else:
                generate(warm_z, limit)
            synchronize()
        warmed = True

    if measurement == "ttft":
        synchronize()
        started = clock() if request_started is None else request_started
        preparation_started = clock() if prepare is not None else None
        if prepare is not None:
            prepare()
            synchronize()
        input_seconds = (
            clock() - preparation_started if preparation_started is not None else 0.0
        )
        timers = [
            ResponseTokenTimer(
                response_tokenizer,
                started=started,
                synchronize=synchronize,
                clock=clock,
                response_started=response_started,
            )
            for _ in range(response_batch_size)
        ]
        z = reason()
        outputs = tuple(generate_timed(z, limit, observer(timers)))
        for timer in timers:
            timer.finish()
        synchronize()
        if len(outputs) != response_batch_size:
            raise RuntimeError("response TTFT output batch size changed")
        for timer, output in zip(timers, outputs):
            if timer.observed_tokens != len(output.get("ids", ())):
                raise RuntimeError(
                    "response TTFT token callback did not observe the full output"
                )
        timer = timers[0]
        return AnswerBatchResult(
            z=z,
            outputs=outputs,
            warmed=warmed,
            ttft_seconds=timer.seconds if response_batch_size == 1 else None,
            input_preparation_seconds=input_seconds,
            response_censored=all(t.seconds is None for t in timers),
            first_response_token_ids=timer.first_response_token_ids
            if response_batch_size == 1
            else (),
            row_ttft_seconds=tuple(t.seconds for t in timers),
            row_first_response_token_ids=tuple(
                t.first_response_token_ids for t in timers
            ),
        )

    if prepare is not None:
        prepare()
    synchronize()
    reasoner_started = clock()
    z = reason()
    synchronize()
    reasoner_seconds = clock() - reasoner_started
    synchronize()
    answer_started = clock()
    outputs = tuple(generate(z, limit))
    synchronize()
    answer_seconds = clock() - answer_started
    if any(
        value < 0.0 or not math.isfinite(value)
        for value in (reasoner_seconds, answer_seconds)
    ):
        raise RuntimeError("accuracy timing clock returned an invalid latency")
    return AnswerBatchResult(
        z=z,
        outputs=outputs,
        warmed=warmed,
        reasoner_seconds=reasoner_seconds,
        answer_seconds=answer_seconds,
    )


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered or not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("percentile input is empty or invalid")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(quantile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_ttft(values: Sequence[float]) -> dict[str, float | int]:
    normalized = [float(value) for value in values]
    if not normalized or any(
        value < 0.0 or not math.isfinite(value) for value in normalized
    ):
        raise ValueError("TTFT samples must be nonempty, finite, and nonnegative")
    return {
        "samples": len(normalized),
        "mean_seconds": statistics.fmean(normalized),
        "median_seconds": statistics.median(normalized),
        "p90_seconds": _percentile(normalized, 0.90),
        "p95_seconds": _percentile(normalized, 0.95),
        "p99_seconds": _percentile(normalized, 0.99),
    }


def _reason_batch_in_chunks(
    prompt_ids: Any,
    prompt_mask: Any,
    *,
    chunk_size: int,
    reason: Callable[[Any, Any], Any],
    concatenate: Callable[[Sequence[Any]], Any],
) -> Any:
    """Construct one logical answer batch without coupling R memory to vLLM concurrency."""

    rows = len(prompt_ids)
    if rows <= 0 or int(chunk_size) <= 0:
        raise ValueError("reasoner batch rows and chunk size must be positive")
    chunks = [
        reason(
            prompt_ids[start : start + int(chunk_size)],
            prompt_mask[start : start + int(chunk_size)],
        )
        for start in range(0, rows, int(chunk_size))
    ]
    return chunks[0] if len(chunks) == 1 else concatenate(chunks)


def _materialize_reasoner_batch(
    model: Any,
    prompt_ids: Any,
    prompt_mask: Any,
    *,
    chunk_size: int,
    reader_context_mask: Any = None,
) -> Any:
    """Use the validation R path; legacy chunk_size remains a memory upper bound."""
    from think_bridge.model.reasoner_inference import reason_eval_padded

    if int(chunk_size) <= 0:
        raise ValueError("reasoner chunk size must be positive")
    return reason_eval_padded(
        model,
        prompt_ids,
        prompt_mask,
        chunk_size=chunk_size,
        **(
            {"reader_context_mask": reader_context_mask}
            if reader_context_mask is not None
            else {}
        ),
    )


def _run_dataset_evaluations(
    paths: Sequence[Path],
    *,
    backend_factory: Callable[[], Any],
    evaluate: Callable[[Path, Any], Any],
) -> list[Any]:
    """Keep one answer backend alive for the complete multi-dataset run."""

    backend = backend_factory()
    try:
        return [evaluate(path, backend) for path in paths]
    finally:
        backend.close()


def _dataset_rows(path: Path, maximum: int | None) -> list[dict[str, Any]]:
    rows = load_json_records(path)
    selected = rows if maximum is None else rows[:maximum]
    result: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer", row.get("reference_answer", ""))).strip()
        if not question or not answer:
            raise ValueError(
                f"benchmark file {path} row {index} lacks question or answer"
            )
        result.append(
            {
                "row_index": index,
                "question": question,
                "reference_answer": answer,
                "task_type": str(row.get("type") or "math").lower(),
            }
        )
    if not result:
        raise ValueError(f"benchmark file has no usable rows: {path}")
    return result


def _tokenize_rows(
    rows: Sequence[Mapping[str, Any]], tokenizer: Any
) -> list[dict[str, Any]]:
    from think_bridge.data.templates import THINK_OPEN_TEXT, build_prompt

    result: list[dict[str, Any]] = []
    for row in rows:
        prompt = (
            build_prompt(
                tokenizer,
                str(row["question"]),
                task_type=str(row["task_type"]),
                think=True,
            )
            + THINK_OPEN_TEXT
        )
        result.append(
            {
                **dict(row),
                "prompt_ids": tuple(
                    int(value)
                    for value in tokenizer.encode(prompt, add_special_tokens=False)
                ),
            }
        )
    return result


def _pad(rows: Sequence[Sequence[int]], *, device: Any) -> tuple[Any, Any]:
    import torch

    width = max(len(row) for row in rows)
    ids = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    mask = torch.zeros((len(rows), width), dtype=torch.bool, device=device)
    for index, row in enumerate(rows):
        ids[index, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        mask[index, : len(row)] = True
    return ids, mask


def _sync(device: Any) -> None:
    if getattr(device, "type", None) == "cuda":
        import torch

        torch.cuda.synchronize(device)


class _HFAnswerBackend:
    def __init__(self, runtime: RuntimeBundle, arguments: Any) -> None:
        self.runtime = runtime
        self.arguments = arguments
        self.warmed = False

    def generate(
        self,
        rows: Sequence[Mapping[str, Any]],
        z: Any,
        *,
        max_tokens: int,
        offset: int,
        on_token: Callable[[int, int], bool | None] | None = None,
    ) -> list[dict[str, Any]]:
        from think_bridge.training.eval import CausalEvalRequest, _free_answers

        requests = [
            CausalEvalRequest(
                prompt_ids=tuple(row["prompt_ids"]),
                z=z[index],
                seed=int(self.arguments.seed),
            )
            for index, row in enumerate(rows)
        ]
        return list(
            _free_answers(
                self.runtime.model,
                self.runtime.tokenizer,
                requests,
                max_tokens=int(max_tokens),
                temperature=float(self.arguments.temperature),
                top_p=float(self.arguments.top_p),
                token_callback=on_token,
            )
        )

    def generate_timed(self, rows, z, *, max_tokens, offset, on_token):
        return self.generate(
            rows, z, max_tokens=max_tokens, offset=offset, on_token=on_token
        )

    def close(self) -> None:
        return None


class _VLLMAnswerBackend:
    def __init__(self, runtime: RuntimeBundle, arguments: Any) -> None:
        from think_bridge.training.vllm_worker import BridgeRoute1VLLMBackend

        self.runtime = runtime
        self.arguments = arguments
        self.physical = int(arguments.batch_size)
        self.warmed = False
        self.request_index = 0
        self.service = BridgeRoute1VLLMBackend(
            model_name_or_path=str(arguments.model),
            data_parallel_size=int(arguments.vllm_data_parallel_size),
            tensor_parallel_size=int(arguments.vllm_tensor_parallel_size),
            worker_extension_cls="think_bridge.training.vllm_worker.BridgeRoute1WorkerExtension",
            gpu_memory_utilization=float(arguments.vllm_gpu_memory_utilization),
            max_num_seqs=int(
                getattr(arguments, "vllm_max_num_seqs", None) or self.physical
            ),
            request_timeout_seconds=float(arguments.vllm_request_timeout_seconds),
            physical_chunk_size=self.physical,
            enforce_eager=bool(arguments.vllm_enforce_eager),
            seed=int(arguments.seed),
        )
        print(
            f"[vLLM eval] batch_invariant={self.service.batch_invariant} "
            f"seed={int(arguments.seed)} temperature=0",
            flush=True,
        )

    def generate(
        self,
        rows: Sequence[Mapping[str, Any]],
        z: Any,
        *,
        max_tokens: int,
        offset: int,
    ) -> list[dict[str, Any]]:
        from think_bridge.eval.answer_protocol import true_z_request, answer_outputs

        real = len(rows)
        request_index = self.request_index
        self.request_index += 1
        request = true_z_request(
            rows,
            z,
            boundary_ids=self.runtime.model.boundary_ids.tolist(),
            eos_token_id=self.runtime.model.eos_token_id,
            seed=self.arguments.seed,
            max_tokens=max_tokens,
            offset=offset,
            physical=self.physical,
            namespace="benchmark",
            request_index=request_index,
        )
        observer = getattr(self, "request_observer", None)
        if observer is not None:
            observer(request, self.service._scheduler_prompts(request), real)
        response = self.service.execute(request)
        return answer_outputs(response, request, self.runtime.tokenizer, real)

    def close(self) -> None:
        self.service.close()


def _answer_backend_factory(
    runtime: RuntimeBundle, arguments: Any
) -> Callable[[], Any]:
    backend_type = (
        _VLLMAnswerBackend if arguments.backend == "vllm" else _HFAnswerBackend
    )
    return lambda: backend_type(runtime, arguments)


def _public_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "prompt_ids"}


def _evaluate_dataset(
    path: Path, backend: Any, *, runtime: RuntimeBundle, arguments: Any
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch
    from think_bridge.cli.benchmark import score_prediction_rows
    from think_bridge.training.progress import bridge_progress

    rows = _dataset_rows(path, arguments.max_samples)
    fixed_rows = fixed_groups = None
    if getattr(runtime.model, "_feedback_fixed_groups", False):
        if arguments.measurement != "accuracy":
            raise ValueError(
                "Fixed-group R evaluation is an accuracy protocol, not singleton TTFT"
            )
        from think_bridge.model.evaluation_groups import FixedReasonerGroups

        fixed_rows = _tokenize_rows(rows, runtime.tokenizer)

        def fixed_reason(prompts):
            (ids, mask) = _pad(prompts, device=runtime.device)
            return _materialize_reasoner_batch(
                runtime.model,
                ids,
                mask,
                chunk_size=runtime.model._feedback_eval_batch_size,
            )

        fixed_groups = FixedReasonerGroups(
            [r["prompt_ids"] for r in fixed_rows],
            fixed_reason,
            group_size=runtime.model._feedback_eval_batch_size,
        )
    progress = bridge_progress(
        total=len(rows),
        desc=f"Benchmark {path.stem}",
        unit="row",
        disabled=arguments.no_progress,
    )
    limit = answer_generation_limit(arguments.measurement, arguments.max_new_tokens)
    generated_rows: list[dict[str, Any]] = []
    answer_latencies: list[float] = []
    z_rows: list[Any] = []
    answer_seconds = 0.0
    reasoner_seconds = 0.0
    try:
        for start in range(0, len(rows), int(arguments.batch_size)):
            raw_batch = rows[start : start + int(arguments.batch_size)]
            batch = []
            prompt_ids = prompt_mask = None

            def prepare() -> None:
                nonlocal batch, prompt_ids, prompt_mask
                batch = (
                    fixed_rows[start : start + len(raw_batch)]
                    if fixed_rows is not None
                    else _tokenize_rows(raw_batch, runtime.tokenizer)
                )
                (prompt_ids, prompt_mask) = _pad(
                    [row["prompt_ids"] for row in batch], device=runtime.device
                )

            def reason() -> Any:
                if fixed_groups is not None:
                    return torch.stack(
                        fixed_groups.select(range(start, start + len(raw_batch)))
                    ).to(runtime.device)
                return _materialize_reasoner_batch(
                    runtime.model,
                    prompt_ids,
                    prompt_mask,
                    chunk_size=int(
                        arguments.reasoner_batch_size or arguments.batch_size
                    ),
                )

            result = _execute_answer_batch(
                measurement=arguments.measurement,
                max_new_tokens=int(arguments.max_new_tokens),
                warmup_samples=int(arguments.warmup_samples),
                already_warmed=bool(backend.warmed),
                reason=reason,
                generate=lambda z, max_tokens: backend.generate(
                    batch, z, max_tokens=max_tokens, offset=start
                ),
                synchronize=lambda: _sync(runtime.device),
                prepare=prepare,
                response_tokenizer=runtime.tokenizer,
                stop_after_response=arguments.measurement == "ttft",
                response_batch_size=len(raw_batch),
                generate_timed=(
                    lambda z, max_tokens, on_token: backend.generate_timed(
                        batch, z, max_tokens=max_tokens, offset=start, on_token=on_token
                    )
                )
                if arguments.measurement == "ttft"
                else None,
            )
            backend.warmed = result.warmed
            outputs = result.outputs
            if len(outputs) != len(batch):
                raise RuntimeError("answer generation changed the logical batch size")
            if arguments.measurement == "ttft":
                answer_latencies.extend(
                    (t for t in result.row_ttft_seconds if t is not None)
                )
            else:
                if result.reasoner_seconds is None or result.answer_seconds is None:
                    raise RuntimeError("accuracy execution did not return timings")
                reasoner_seconds += result.reasoner_seconds
                answer_seconds += result.answer_seconds
            for row_index, (row, output) in enumerate(zip(batch, outputs)):
                public = _public_row(row)
                if arguments.measurement == "accuracy":
                    if hasattr(runtime.model, "boundary_ids") and hasattr(
                        runtime.model, "eos_token_id"
                    ):
                        from think_bridge.eval.answer_protocol import input_signature

                        public["reasoner_input"] = input_signature(
                            row["prompt_ids"],
                            result.z[row_index],
                            runtime.model.boundary_ids.tolist(),
                            runtime.model.eos_token_id,
                        )
                    public.update(
                        prediction=str(output["text"]),
                        answer_token_ids=list(output["ids"]),
                        terminated=bool(output["terminated"]),
                        cap_hit=bool(output["cap_hit"]),
                    )
                else:
                    public.update(
                        first_token_text=runtime.tokenizer.decode(
                            list(result.row_first_response_token_ids[row_index]),
                            skip_special_tokens=True,
                        ),
                        first_token_ids=list(
                            result.row_first_response_token_ids[row_index]
                        ),
                        prediction=str(output["text"]),
                        answer_token_ids=list(output["ids"]),
                        terminated=bool(output["terminated"]),
                        cap_hit=bool(output["cap_hit"]),
                        response_censored=result.row_ttft_seconds[row_index] is None,
                        input_preparation_seconds=result.input_preparation_seconds,
                        answer_ttft_seconds=result.row_ttft_seconds[row_index],
                        configured_batch_size=int(arguments.batch_size),
                        request_batch_size=len(batch),
                        request_batch_id=start // int(arguments.batch_size),
                        generation_policy="stop-after-first-response-token",
                    )
                generated_rows.append(public)
            progress.update(len(batch))
    finally:
        progress.close()
    if arguments.measurement == "accuracy":
        (metric, persisted) = score_prediction_rows(generated_rows, task_type="math")
        metric.update(
            reasoner_seconds=reasoner_seconds,
            reasoner_throughput_rows_per_second=len(rows) / reasoner_seconds
            if reasoner_seconds > 0.0
            else 0.0,
            answer_generation_seconds=answer_seconds,
            answer_throughput_rows_per_second=len(rows) / answer_seconds
            if answer_seconds > 0.0
            else 0.0,
        )
    else:
        from think_bridge.eval.response_timing import (
            SINGLE_TURN_TTFT_CONTRACT,
            SINGLE_TURN_BATCH_TTFT_CONTRACT,
        )

        metric = (
            summarize_ttft(answer_latencies)
            if answer_latencies
            else {
                "samples": 0,
                "mean_seconds": None,
                "median_seconds": None,
                "p90_seconds": None,
                "p95_seconds": None,
                "p99_seconds": None,
            }
        )
        metric.update(
            timing_contract=SINGLE_TURN_TTFT_CONTRACT
            if int(arguments.batch_size) == 1
            else SINGLE_TURN_BATCH_TTFT_CONTRACT,
            configured_per_gpu_batch_size=int(arguments.batch_size),
            observed_batch_sizes=sorted(
                {r["request_batch_size"] for r in generated_rows}
            ),
            request_count=len(rows),
            response_observed_count=len(answer_latencies),
            response_censored_count=len(rows) - len(answer_latencies),
            generation_policy="stop-after-first-response-token",
            response_search_max_tokens=int(arguments.max_new_tokens),
        )
        persisted = generated_rows
    summary: dict[str, Any] = {
        "dataset": str(path),
        "measurement": arguments.measurement,
        "backend": arguments.backend,
        "components": ["reasoner"],
        "checkpoint_sources": {"reasoner": _source_summary(runtime.reasoner_source)},
        "sampling": {
            "seed": int(arguments.seed),
            "answer_max_tokens": limit,
            "answer_temperature": float(arguments.temperature),
            "answer_top_p": float(arguments.top_p),
            "batch_size": int(arguments.batch_size),
            "reasoner_batch_size": reasoner_inference_metadata(runtime.model)[
                "physical_batch_size"
            ]
            or int(arguments.reasoner_batch_size),
        },
        "reasoner_inference": reasoner_inference_metadata(runtime.model),
        arguments.measurement: metric,
    }
    if arguments.backend == "hf" and arguments.measurement == "accuracy":
        from think_bridge.eval.hf_protocol import hf_evaluation_protocol

        summary["answer_evaluation"] = {
            **hf_evaluation_protocol(
                int(arguments.batch_size), runtime.model._feedback_eval_batch_size
            ),
            "seed": int(arguments.seed),
            "answer_max_tokens": limit,
        }
    if arguments.backend == "vllm" and arguments.measurement == "accuracy":
        from think_bridge.eval.answer_protocol import answer_protocol

        summary["answer_evaluation"] = {
            **answer_protocol(),
            "request_batch_size": int(arguments.batch_size),
            "schema": "vllm-bf16-batched-hf-r-v2",
            "reasoner_batch_size": runtime.model._feedback_eval_batch_size,
            "backend": "vllm",
            "reasoner_backend": "hf",
            "vllm_version": getattr(
                getattr(backend, "service", None), "vllm_version", None
            ),
            "batch_invariant": getattr(
                getattr(backend, "service", None), "batch_invariant", None
            ),
            "max_num_seqs": int(
                getattr(arguments, "vllm_max_num_seqs", None) or arguments.batch_size
            ),
            "data_parallel_size": getattr(arguments, "vllm_data_parallel_size", None),
            "tensor_parallel_size": getattr(
                arguments, "vllm_tensor_parallel_size", None
            ),
            "gpu_memory_utilization": getattr(
                arguments, "vllm_gpu_memory_utilization", None
            ),
            "enforce_eager": bool(getattr(arguments, "vllm_enforce_eager", False)),
            "answer_max_tokens": limit,
            "seed": int(arguments.seed),
        }
    return (summary, persisted)


def _write_json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _source_summary(source: CheckpointComponent | None) -> dict[str, Any] | None:
    if source is None:
        return None
    return {
        "checkpoint": str(source.checkpoint),
        "kind": source.checkpoint_kind,
        "owner": source.owner,
        "weight_file": str(source.weight_path),
        "resolved_config": str(source.resolved_config_path),
    }


def run_benchmark(arguments: Any) -> int:
    output = arguments.output_dir.expanduser()
    if output.exists():
        raise FileExistsError(f"benchmark output directory already exists: {output}")
    dataset_paths = tuple(
        (path.expanduser().resolve(strict=True) for path in arguments.dataset)
    )
    stems = [path.stem for path in dataset_paths]
    if len(stems) != len(set(stems)):
        raise ValueError("benchmark dataset stems must be unique within one run")
    runtime = _build_runtime(arguments)
    output.mkdir(parents=True, exist_ok=False)

    def evaluate(
        path: Path, backend: Any
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        (summary, predictions) = _evaluate_dataset(
            path, backend, runtime=runtime, arguments=arguments
        )
        _write_json_new(
            output / f"{path.stem}.predictions.json",
            {"metadata": summary, "rows": predictions},
        )
        _write_json_new(output / f"{path.stem}.summary.json", summary)
        return (summary, predictions)

    results = _run_dataset_evaluations(
        dataset_paths,
        backend_factory=_answer_backend_factory(runtime, arguments),
        evaluate=evaluate,
    )
    summaries = [summary for (summary, _predictions) in results]
    run_summary = {
        "mode": "single-turn",
        "measurement": arguments.measurement,
        "backend": arguments.backend,
        "components": ["reasoner"],
        "checkpoint_sources": {"reasoner": _source_summary(runtime.reasoner_source)},
        "model": str(arguments.model),
        "tokenizer": str(runtime.tokenizer.name_or_path),
        "model_family": runtime.model_family,
        "reasoner_geometry": {
            "latent_slots": runtime.reasoner_geometry.latent_slots,
            "latent_steps": runtime.reasoner_geometry.latent_steps,
            "latents_per_step": runtime.reasoner_geometry.latents_per_step,
            "loop_steps": runtime.reasoner_geometry.loop_steps,
            "num_layers": runtime.reasoner_geometry.num_layers,
        },
        "device_layout": {
            "backend": "hf",
            "local_device": str(runtime.device),
            "replicas": getattr(arguments, "_hf_world_size", 1),
            "group_size": int(arguments.batch_size),
        }
        if arguments.backend == "hf"
        else {
            "backend": "vllm",
            "reasoner_backend": "hf",
            "local_device": str(runtime.device),
            "vllm_data_parallel_size": int(arguments.vllm_data_parallel_size),
            "vllm_tensor_parallel_size": int(arguments.vllm_tensor_parallel_size),
            "vllm_gpu_memory_utilization": float(arguments.vllm_gpu_memory_utilization),
        },
        "sampling": {
            "seed": int(arguments.seed),
            "answer_max_tokens": answer_generation_limit(
                arguments.measurement, arguments.max_new_tokens
            ),
            "answer_temperature": float(arguments.temperature),
            "answer_top_p": float(arguments.top_p),
            "batch_size": int(arguments.batch_size),
            "reasoner_batch_size": reasoner_inference_metadata(runtime.model)[
                "physical_batch_size"
            ]
            or int(arguments.reasoner_batch_size),
        },
        "datasets": summaries,
        "reasoner_inference": reasoner_inference_metadata(runtime.model),
    }
    _write_json_new(output / "benchmark_summary.json", run_summary)
    return 0
