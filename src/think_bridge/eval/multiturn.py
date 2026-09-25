"""Response-only multi-turn evaluation with complete text history and fresh R latents."""

from __future__ import annotations
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence
from think_bridge.eval import benchmark as single
from think_bridge.training.progress import bridge_progress

PROTOCOL = "think-bridge-response-history-multiturn-v1"
TTFT_CONTRACT = "online-request-to-first-response-token-raw-history-v2"
BATCH_TTFT_CONTRACT = "batched-request-to-first-response-token-raw-history-v1"
ROW_FIELDS = (
    "id",
    "conversation_id",
    "turn_index",
    "is_final_turn",
    "question",
    "reference_answer",
    "task_type",
    "row_index",
)


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _seal(row: Mapping[str, Any]) -> dict[str, Any]:
    value = {key: item for key, item in row.items() if key != "row_sha256"}
    return {**value, "row_sha256": _digest(value)}


def _prompt_binding(row: Mapping[str, Any], source_name: str, identity_sha: str) -> str:
    return _digest(
        {
            "identity_sha256": identity_sha,
            "id": row["id"],
            "history_source": source_name,
            "messages": row["messages"],
            "prompt_ids": row["prompt_ids"],
            "reader_context": row.get("reader_context", "full"),
            "reader_context_start": row.get("reader_context_start", 0),
        }
    )


def _bind_row(
    row: Mapping[str, Any], source_name: str, identity_sha: str
) -> dict[str, Any]:
    return _seal(
        {
            **row,
            "history_source": source_name,
            "execution_identity_sha256": identity_sha,
            "prompt_binding_sha256": _prompt_binding(row, source_name, identity_sha),
        }
    )


def _reply_dict(parts: Any) -> dict[str, Any]:
    # ReplyParts diagnostics may contain tuples; persist and compare JSON values.
    return json.loads(json.dumps(asdict(parts)))


def _think_tags(token_ids: Sequence[int], tokenizer: Any) -> dict[str, int]:
    raw = tokenizer.decode(token_ids, skip_special_tokens=False)
    return {"open": raw.count("<think>"), "close": raw.count("</think>")}


def _component_identity(source: Any, hashes: dict[Path, str]) -> dict[str, Any]:
    def hashed(path: Path) -> str:
        resolved = path.resolve(strict=True)
        if resolved not in hashes:
            hashes[resolved] = _file_sha(resolved)
        return hashes[resolved]

    root = source.checkpoint
    files = [source.weight_path, source.resolved_config_path]
    files.extend(
        root / name for name in ("checkpoint.json",) if (root / name).is_file()
    )
    if source.component == "reasoner":
        fixed = root / "reasoner_fixed_state.safetensors"
        if not fixed.is_file():
            raise ValueError(
                "multi-turn identity requires the portable R fixed-state artifact"
            )
        files.append(fixed)
    return {
        **single._source_summary(source),
        "files": [
            {"path": str(path.resolve()), "sha256": hashed(path)} for path in files
        ],
    }


def _runtime_identity(runtime: Any, arguments: Any) -> dict[str, Any]:
    """Bind actual lexical/model artifacts and the portable fixed R state."""
    config = runtime.model.executor.config
    config_dict = dict(config.to_dict())
    # Loading-device/attention choices may differ between ACC and HF TTFT.
    for name in (
        "_name_or_path",
        "_attn_implementation_internal",
        "torch_dtype",
        "dtype",
    ):
        config_dict.pop(name, None)
    model_root = Path(arguments.model).expanduser()
    model_files = (
        sorted(
            {
                path
                for pattern in (
                    "*.safetensors",
                    "pytorch_model*.bin",
                    "*.index.json",
                    "config.json",
                    "generation_config.json",
                )
                for path in model_root.glob(pattern)
            }
        )
        if model_root.is_dir()
        else []
    )
    if model_root.is_dir() and not any(
        path.suffix in {".safetensors", ".bin"} for path in model_files
    ):
        raise ValueError("local frozen model identity requires weight files")
    revision = getattr(config, "_commit_hash", None)
    if not model_files and not revision:
        raise ValueError("remote frozen model identity requires a resolved commit hash")
    tokenizer = runtime.tokenizer
    hashes: dict[Path, str] = {}
    return {
        "model": {
            "name": str(model_root.resolve())
            if model_root.is_dir()
            else str(arguments.model),
            "revision": revision,
            "config_sha256": _digest(config_dict),
            "files": [
                {"name": path.name, "sha256": _file_sha(path)} for path in model_files
            ],
        },
        "tokenizer": {
            "name": str(runtime.tokenizer.name_or_path),
            "class": type(tokenizer).__name__,
            "vocabulary_sha256": _digest(tokenizer.get_vocab()),
            "special_tokens_sha256": _digest(tokenizer.special_tokens_map),
            "backend_sha256": _digest(
                tokenizer.backend_tokenizer.to_str()
                if hasattr(tokenizer, "backend_tokenizer")
                else tokenizer.init_kwargs
            ),
        },
        "checkpoints": {
            "reasoner": _component_identity(runtime.reasoner_source, hashes),
        },
        "model_family": runtime.model_family,
        "reasoner_geometry": asdict(runtime.reasoner_geometry),
        "reasoner_inference": single.reasoner_inference_metadata(runtime.model),
        "reasoner_input": getattr(runtime.model, "_inference_reasoner_input", {}),
        "reasoner_compute": {
            "owners": "float32",
            "cuda_compute": "bfloat16",
            "cpu_compute": "float32",
        },
    }


def _render_turn(
    row: Mapping[str, Any],
    history: Sequence[Mapping[str, str]],
    tokenizer: Any,
    *,
    reader_context: str = "full",
) -> dict[str, Any]:
    from think_bridge.data.templates import (
        THINK_OPEN_TEXT,
        build_chat_messages,
        build_prompt,
    )

    # Historical user questions and selected assistant text are preserved verbatim.
    messages = build_chat_messages(
        messages=[*history, {"role": "user", "content": row["question"]}],
        task_type=row["task_type"],
        think=True,
    )
    prompt = (
        build_prompt(
            tokenizer,
            messages=messages,
            think_instruction="",
            preserve_history_thinking=True,
        )
        + THINK_OPEN_TEXT
    )
    ids = [int(token) for token in tokenizer.encode(prompt, add_special_tokens=False)]
    if reader_context != "full":
        raise ValueError("multi-turn R requires full history")
    return {
        **{key: row[key] for key in ROW_FIELDS},
        "history": [dict(message) for message in history],
        "messages": messages,
        "prompt_ids": ids,
        "context_tokens": len(ids),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    }


def _check_context(row: Mapping[str, Any], runtime: Any, max_new_tokens: int) -> None:
    limit = int(runtime.model.executor.config.max_position_embeddings)
    # R feedback appends at most K latents; deployed F adds z and the think boundary.
    boundary = len(runtime.model.boundary_ids)
    required = (
        len(row["prompt_ids"])
        + runtime.reasoner_geometry.latent_slots
        + boundary
        + max_new_tokens
    )
    if not row["prompt_ids"] or required > limit:
        raise ValueError(
            f"context exceeds max_position_embeddings for {row['id']}: "
            f"prompt={len(row['prompt_ids'])}, z={runtime.reasoner_geometry.latent_slots}, "
            f"boundary={boundary}, generation={max_new_tokens}, limit={limit}; no truncation"
        )


def _answer_batch(
    rows: Sequence[Mapping[str, Any]],
    *,
    runtime: Any,
    backend: Any,
    arguments: Any,
    offset: int,
    request_started: float | None = None,
    fixed_reasoner_groups: Any = None,
    on_latents: Any = None,
) -> Any:
    import torch

    def prepare_inputs():
        ids, mask = single._pad(
            [row["prompt_ids"] for row in rows], device=runtime.device
        )
        model_rows = [{"prompt_ids": row["prompt_ids"]} for row in rows]
        return ids, mask, model_rows

    if arguments.measurement == "ttft":
        (ids, mask, model_rows), input_seconds = _timed_preparation(
            runtime, prepare_inputs
        )
    else:
        ids, mask, model_rows = prepare_inputs()
        input_seconds = 0.0

    def reason() -> Any:
        if fixed_reasoner_groups is not None:
            return torch.stack(
                fixed_reasoner_groups.lookup([r["prompt_ids"] for r in rows])
            ).to(runtime.device)
        return single._materialize_reasoner_batch(
            runtime.model,
            ids,
            mask,
            chunk_size=int(arguments.reasoner_batch_size or arguments.batch_size),
        )

    def generate(z, max_tokens):
        if on_latents is not None:
            on_latents(z)
        return backend.generate(model_rows, z, max_tokens=max_tokens, offset=offset)

    # Gold and future questions never enter either model-facing callable.
    with torch.no_grad():
        result = single._execute_answer_batch(
            measurement=arguments.measurement,
            max_new_tokens=int(arguments.max_new_tokens),
            warmup_samples=int(arguments.warmup_samples),
            already_warmed=bool(backend.warmed),
            reason=reason,
            generate=generate,
            synchronize=lambda: single._sync(runtime.device),
            **(
                {
                    "request_started": request_started,
                    "response_tokenizer": runtime.tokenizer,
                    "response_batch_size": len(rows),
                    "generate_timed": lambda z, max_tokens, on_token: (
                        backend.generate_timed(
                            model_rows,
                            z,
                            max_tokens=max_tokens,
                            offset=offset,
                            on_token=on_token,
                        )
                    ),
                }
                if arguments.measurement == "ttft"
                else {}
            ),
        )
    backend.warmed = result.warmed
    if len(result.outputs) != len(rows):
        raise RuntimeError(
            "multi-turn answer generation changed the logical batch size"
        )
    return replace(result, input_preparation_seconds=input_seconds)


def _history_record(parts: Any, source: str) -> dict[str, Any]:
    from think_bridge.eval.conversation_history import history_content

    text = history_content(parts, source)
    return {
        "source": source,
        "text": text,
        "protocol": "verbatim-selected-history-v1",
        "identity_sha256": _digest({"source": source, "text": text}),
    }


def _next_history(row: Mapping[str, Any], source: str) -> list[dict[str, str]]:
    retained = row.get("retained_history")
    if not isinstance(retained, dict) or retained.get("source") != source:
        raise ValueError("history source is missing its selected original content")
    return [
        *row["history"],
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": retained["text"]},
    ]


def _timed_preparation(runtime: Any, operation: Any) -> tuple[Any, float]:
    """Synchronized serial wall time including CPU preparation and device work."""
    single._sync(runtime.device)
    began = time.perf_counter()
    value = operation()
    single._sync(runtime.device)
    seconds = time.perf_counter() - began
    if not math.isfinite(seconds) or seconds < 0:
        raise RuntimeError("TTFT preparation clock returned an invalid latency")
    return value, seconds


def _score_timed_predictions(predictions, *, no_progress):
    """Use the same judge for ACC and TTFT, after all timed generation."""
    from think_bridge.cli.benchmark import score_prediction_rows

    progress = bridge_progress(
        total=sum(map(len, predictions.values())),
        desc="Multi-turn judgment",
        unit="answer",
        disabled=no_progress,
    )
    try:
        for rows in predictions.values():
            for index, row in enumerate(rows):
                _, scored = score_prediction_rows([row], task_type=row["task_type"])
                rows[index] = _seal(scored[0])
                progress.update(1)
    finally:
        progress.close()


def _timed_accuracy_summary(rows):
    """ACC for the exact complete outputs used in online TTFT and history."""
    from think_bridge.cli.benchmark import summarize_scored_prediction_rows

    groups = {
        f"turn_{turn}": [row for row in rows if row["turn_index"] == turn - 1]
        for turn in (1, 2, 3)
    }
    groups.update(
        follow_up=[row for row in rows if row["turn_index"] > 0], overall=list(rows)
    )
    summary = {
        name: summarize_scored_prediction_rows(selected)
        for name, selected in groups.items()
    }
    conversations = {}
    for row in rows:
        cid = row["conversation_id"]
        conversations[cid] = conversations.get(cid, True) and row["correct"]
    count = sum(conversations.values())
    summary["conversation_all_correct"] = {
        "correct": count,
        "total": len(conversations),
        "accuracy": count / len(conversations),
    }
    return summary


def _context_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [int(row["context_tokens"]) for row in rows]
    return {
        "total": sum(values),
        "rows": len(values),
        "mean": statistics.fmean(values),
        "p50": single._percentile(values, 0.5),
        "p95": single._percentile(values, 0.95),
        "maximum": max(values),
    }


def _generation_identity(arguments):
    return {
        name: getattr(arguments, name)
        for name in (
            "seed",
            "max_new_tokens",
            "temperature",
            "top_p",
            "batch_size",
            "reasoner_batch_size",
            "warmup_samples",
        )
    }


def _summarize(rows, measurement):
    accuracy = _timed_accuracy_summary(rows)
    if measurement == "accuracy":
        return accuracy
    groups = {
        f"turn_{turn + 1}": [r for r in rows if r["turn_index"] == turn]
        for turn in range(3)
    }
    groups.update(
        follow_up=[r for r in rows if r["turn_index"] > 0], overall=list(rows)
    )
    summary = {}
    for name, selected in groups.items():
        contracts = {r["ttft_contract"] for r in selected}
        sizes = {r["configured_batch_size"] for r in selected}
        if len(contracts) != 1 or len(sizes) != 1:
            raise ValueError("cannot combine different batch sizes or TTFT protocols")
        values = [
            r["answer_ttft_seconds"]
            for r in selected
            if r["answer_ttft_seconds"] is not None
        ]
        metric = (
            single.summarize_ttft(values)
            if values
            else dict(
                samples=0,
                mean_seconds=None,
                median_seconds=None,
                p90_seconds=None,
                p95_seconds=None,
                p99_seconds=None,
            )
        )
        metric.update(
            requests=len(selected),
            response_censored=len(selected) - len(values),
            ttft_contract=next(iter(contracts)),
            configured_per_gpu_batch_size=next(iter(sizes)),
            observed_batch_sizes=sorted({r["request_batch_size"] for r in selected}),
            context_tokens=_context_stats(selected),
        )
        summary[name] = metric
    return summary


def _unique_execution_costs(predictions, measurement):
    rows = [row for values in predictions.values() for row in values]
    result = {"requests": len(rows)}
    if measurement == "accuracy":
        for name in ("reasoner", "answer"):
            result[f"{name}_seconds"] = sum(
                r[f"{name}_seconds_amortized"] for r in rows
            )
    else:
        result.update(
            ttft_seconds=sum(
                r["answer_ttft_seconds"]
                for r in rows
                if r["answer_ttft_seconds"] is not None
            ),
            response_censored=sum(r["answer_ttft_seconds"] is None for r in rows),
        )
    return result


def _conversation_batches(conversations, *, runtime, backend, arguments, progress):
    """Keep each conversation in one fixed batch; record individual response events.

    Next-turn arrival is the preceding full answer batch completion plus the
    configured delay. Text history assembly and input preparation after arrival
    are part of latency. Warmups finish before the first measured arrival.
    """
    from think_bridge.eval.conversation_history import make_reply_parts

    size = int(arguments.batch_size)
    timed = arguments.measurement == "ttft"
    delay = float(arguments.inter_turn_delay_seconds or 0)
    if timed and not backend.warmed:
        backend.warmed = True
        warm_rows = [
            _render_turn(c[0], [], runtime.tokenizer) for c in conversations[:size]
        ]
        for row in warm_rows:
            _check_context(row, runtime, arguments.max_new_tokens)
        for _ in range(arguments.warmup_samples):
            _answer_batch(
                warm_rows,
                runtime=runtime,
                backend=backend,
                arguments=arguments,
                offset=0,
            )
    output, offset = [], 0
    for start in range(0, len(conversations), size):
        group = conversations[start : start + size]
        previous = None
        arrival = time.perf_counter()
        for turn in range(3):
            remaining = arrival - time.perf_counter()
            if timed and remaining > 0:
                time.sleep(remaining)

            def render():
                batch = []
                for index, conversation in enumerate(group):
                    history = (
                        _next_history(previous[index], "response_only")
                        if previous
                        else []
                    )
                    row = _render_turn(conversation[turn], history, runtime.tokenizer)
                    _check_context(row, runtime, arguments.max_new_tokens)
                    batch.append(row)
                return batch

            rows, render_seconds = _timed_preparation(runtime, render)
            result = _answer_batch(
                rows,
                runtime=runtime,
                backend=backend,
                arguments=arguments,
                offset=offset,
                request_started=arrival if timed else None,
            )
            finished = time.perf_counter()
            for index, (row, answer) in enumerate(zip(rows, result.outputs)):
                parts = make_reply_parts("", str(answer["text"]))
                row.update(
                    prediction=str(answer["text"]),
                    answer_token_ids=list(answer["ids"]),
                    terminated=bool(answer["terminated"]),
                    cap_hit=bool(answer["cap_hit"]),
                    missing_box=not bool(parts.final_answer),
                    reply_parts=_reply_dict(parts),
                    retained_history=_history_record(parts, "response_only")
                    if turn < 2
                    else None,
                    history_source="response_only",
                    generation_offset=offset + index,
                    request_batch_size=len(rows),
                    configured_batch_size=size,
                    shared_first_turn_execution=False,
                )
                if timed:
                    first_ids = list(result.row_first_response_token_ids[index])
                    row.update(
                        answer_ttft_seconds=result.row_ttft_seconds[index],
                        first_token_ids=first_ids,
                        first_token_text=runtime.tokenizer.decode(
                            first_ids, skip_special_tokens=True
                        ),
                        response_censored=result.row_ttft_seconds[index] is None,
                        ttft_contract=TTFT_CONTRACT
                        if size == 1
                        else BATCH_TTFT_CONTRACT,
                        input_preparation_seconds=render_seconds
                        + result.input_preparation_seconds,
                        full_generation_seconds=finished - arrival,
                        source_missing_box=not bool(parts.final_answer),
                        source_cap_hit=bool(answer["cap_hit"]),
                        inter_turn_delay_seconds=delay if turn else 0.0,
                    )
                else:
                    row.update(
                        reasoner_seconds_amortized=result.reasoner_seconds / len(rows),
                        answer_seconds_amortized=result.answer_seconds / len(rows),
                    )
                output.append(row)
            previous = rows
            arrival = finished + delay
            offset += len(rows)
            progress.update(len(rows))
            del result  # No latent state crosses turns.
    order = {
        str(row["id"]): i for i, row in enumerate(r for c in conversations for r in c)
    }
    output.sort(key=lambda row: order[str(row["id"])])
    return {"response_only": output}


def run_multiturn_benchmark(arguments):
    from think_bridge.eval.conversation_history import load_conversations
    from think_bridge.eval.hf_parallel import shard_groups

    if arguments.backend != "hf" or arguments.measurement not in {"accuracy", "ttft"}:
        raise ValueError("multi-turn supports HF accuracy and TTFT")
    if arguments.reasoner_batch_size != 1:
        raise ValueError("multi-turn uses singleton R with full history")
    output = Path(arguments.output_dir).expanduser()
    paths = [Path(p).expanduser().resolve(strict=True) for p in arguments.dataset]
    if not paths or len({p.stem for p in paths}) != len(paths):
        raise ValueError("dataset names must be nonempty and distinct")
    if output.exists():
        raise FileExistsError(f"benchmark output directory already exists: {output}")
    datasets = [
        (
            p,
            shard_groups(
                load_conversations(p, maximum=arguments.max_conversations),
                rank=getattr(arguments, "_hf_rank", 0),
                world_size=getattr(arguments, "_hf_world_size", 1),
                size=arguments.batch_size,
            ),
        )
        for p in paths
    ]
    runtime = single._build_runtime(arguments)
    history_contract = dict(
        protocol="verbatim-response-history-v1",
        assistant="complete generated F response",
        users="original questions verbatim",
        summarization=False,
        context_overflow="raise; no truncation",
    )
    reader_contract = dict(
        scope="full",
        f_encoding="full-context",
        answer_input="full-context",
        inference_intervention=False,
    )
    timing = dict(
        contract=TTFT_CONTRACT if arguments.batch_size == 1 else BATCH_TTFT_CONTRACT,
        per_gpu_batch_size=arguments.batch_size,
        inter_turn_delay_seconds=arguments.inter_turn_delay_seconds,
        arrival="previous full answer batch completion plus delay",
        warmup_samples=arguments.warmup_samples,
    )
    identity = dict(
        protocol=PROTOCOL,
        runtime=_runtime_identity(runtime, arguments),
        generation=_generation_identity(arguments),
        timing=timing,
        history_contract=history_contract,
        reader_context=reader_contract,
    )
    output.mkdir(parents=True, exist_ok=False)
    reports, predictions_paths, costs = [], {}, {}
    backend = single._answer_backend_factory(runtime, arguments)()
    try:
        for path, conversations in datasets:
            progress = bridge_progress(
                total=3 * len(conversations),
                desc=f"Multi-turn {path.stem}",
                unit="request",
                disabled=arguments.no_progress,
            )
            try:
                predictions = _conversation_batches(
                    conversations,
                    runtime=runtime,
                    backend=backend,
                    arguments=arguments,
                    progress=progress,
                )
            finally:
                progress.close()
            _score_timed_predictions(predictions, no_progress=arguments.no_progress)
            rows = predictions["response_only"]
            report = dict(
                dataset=str(path),
                dataset_sha256=_file_sha(path),
                protocol=PROTOCOL,
                mode="multi-turn",
                measurement=arguments.measurement,
                history_source="response_only",
                history_contract=history_contract,
                reader_context=reader_contract,
                identity_sha256=_digest(identity),
                **{arguments.measurement: _summarize(rows, arguments.measurement)},
            )
            if arguments.measurement == "ttft":
                report["accuracy"] = _timed_accuracy_summary(rows)
            name = f"{path.stem}.response_only"
            single._write_json_new(
                output / f"{name}.predictions.json", dict(metadata=report, rows=rows)
            )
            single._write_json_new(output / f"{name}.summary.json", report)
            reports.append(report)
            predictions_paths[str(path)] = {
                "response_only": str(output / f"{name}.predictions.json")
            }
            for key, value in _unique_execution_costs(
                predictions, arguments.measurement
            ).items():
                costs[key] = costs.get(key, 0) + value
    finally:
        backend.close()
    single._write_json_new(
        output / "benchmark_summary.json",
        dict(
            mode="multi-turn",
            measurement=arguments.measurement,
            backend="hf",
            complete=True,
            history_sources=["response_only"],
            history_generation="online-full-response",
            history_contract=history_contract,
            reader_context=reader_contract,
            identity=identity,
            identity_sha256=_digest(identity),
            datasets=reports,
            prediction_files=predictions_paths,
            execution=dict(
                latent_state="fresh z each turn from complete history",
                unique_costs=costs,
            ),
        ),
    )
    return 0
