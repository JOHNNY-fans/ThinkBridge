"""Precompute prompt-level deterministic frozen-F behavior labels.

``native_label_source=generate`` obtains direct and native labels from the same
greedy inference backend.  ``native_label_source=stage0`` is the formal Stage1
training shortcut: HF still generates exact greedy direct labels, while the
already validated Stage0 paired rollouts supply the existential native label.
Neither path builds R or enables gradients.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_VLLM_CHILD_ENV = "TB_BEHAVIOR_VLLM_CHILD"
_VLLM_RANK_ENV = "TB_BEHAVIOR_VLLM_RANK"
_VLLM_WORLD_ENV = "TB_BEHAVIOR_VLLM_WORLD_SIZE"


class _PromptRenderer:
    """Render the exact hidden and direct prompts required by behavior labels."""

    def __init__(self, tokenizer: Any, contract: Any) -> None:
        self.tokenizer = tokenizer
        self.task_type = str(contract.task_type)
        self.cfg = contract

    def rendered_prompt_ids(
        self, record: Mapping[str, Any], *, mode: str = "hidden"
    ) -> list[int]:
        from think_bridge.data.templates import (
            THINK_OPEN_TEXT,
            build_prompt,
            build_thinking_prompt,
        )

        if mode not in {"hidden", "direct"}:
            raise ValueError("mode must be hidden or direct")
        question = str(record.get("question", "")).strip()
        messages = record.get("messages")
        if not question and not messages:
            raise ValueError("each record requires question or messages")
        task_type = str(record.get("type") or self.task_type).lower()
        prompt_builder = build_thinking_prompt if mode == "hidden" else build_prompt
        kwargs = {} if mode == "hidden" else {"think": False}
        prompt = prompt_builder(
            self.tokenizer,
            question,
            task_type=task_type,
            **kwargs,
            messages=messages,
            preserve_history_thinking=(
                bool(record.get("preserve_history_thinking", False))
                if mode != "direct"
                else False
            ),
        )
        if mode == "direct":
            prompt += THINK_OPEN_TEXT
            from think_bridge.data.templates import THINK_BOUNDARY_TEXT

            prompt += THINK_BOUNDARY_TEXT
        return list(self.tokenizer.encode(prompt, add_special_tokens=False))

    def prompt_group_key(self, record: Mapping[str, Any]) -> str:
        from think_bridge.data.behavior_manifest import prompt_token_ids_key

        return prompt_token_ids_key(self.rendered_prompt_ids(record, mode="hidden"))


def _last_subsequence_start(
    values: Sequence[int], needle: Sequence[int], *, end: int | None = None
) -> int | None:
    """Return the last exact subsequence start before ``end``."""

    haystack = [int(value) for value in values]
    target = [int(value) for value in needle]
    stop = len(haystack) if end is None else min(int(end), len(haystack))
    if not target or stop < len(target):
        return None
    for start in range(stop - len(target), -1, -1):
        if haystack[start : start + len(target)] == target:
            return start
    return None


def _native_cot_token_count(
    native_tokens: Sequence[int], *, tokenizer: Any
) -> tuple[bool, int | None]:
    """Measure the generated native-think span, excluding think markers.

    The primary path operates on the actual generated token ids.  A decoded
    fallback handles tokenizers whose close marker is context-tokenized
    differently; incomplete generations remain explicit ``(False, None)``.
    """

    tokens = [int(value) for value in native_tokens]
    close_ids = list(tokenizer.encode("</think>", add_special_tokens=False))
    close_start = _last_subsequence_start(tokens, close_ids)
    if close_start is not None:
        open_ids = list(tokenizer.encode("<think>", add_special_tokens=False))
        open_start = _last_subsequence_start(tokens, open_ids, end=close_start)
        reasoning_start = open_start + len(open_ids) if open_start is not None else 0
        return True, max(close_start - reasoning_start, 0)

    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        return False, None
    text = str(decode(tokens, skip_special_tokens=False))
    if "</think>" not in text:
        return False, None
    head, _, _ = text.rpartition("</think>")
    reasoning = head.rsplit("<think>", 1)[-1]
    fallback_ids = tokenizer.encode(reasoning, add_special_tokens=False)
    return True, len(fallback_ids)


def _split_exact_native_pair(
    native_tokens: Sequence[int], *, tokenizer: Any, hit_eos: bool
) -> tuple[
    bool,
    list[int] | None,
    list[int] | None,
    list[int] | None,
]:
    """Split the generated token stream without decode/re-tokenize drift.

    The hidden prompt already ends in ``<think>\n``. A reusable native
    trajectory is delimited by Qwen's ``</think>`` special-token sequence and
    must terminate normally.  Prefix whitespace is context-tokenized and is
    therefore not required to equal a separately tokenized deployment
    boundary.  V2 preserves the actual close marker plus following whitespace
    as ``native_boundary_ids``.  Thus ``cot + boundary + answer`` reconstructs
    the generated IDs exactly, without a text fallback or re-tokenization.
    """

    tokens = [int(value) for value in native_tokens]
    close_ids = list(tokenizer.encode("</think>", add_special_tokens=False))
    close_start = _last_subsequence_start(tokens, close_ids)
    if close_start is None or not close_ids or not hit_eos:
        return False, None, None, None
    boundary_end = close_start + len(close_ids)
    while boundary_end < len(tokens):
        token_id = int(tokens[boundary_end])
        try:
            decoded = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        if not decoded or not str(decoded).isspace():
            break
        boundary_end += 1
    cot_ids = tokens[:close_start]
    boundary_ids = tokens[close_start:boundary_end]
    answer_ids = tokens[boundary_end:]
    if [*cot_ids, *boundary_ids, *answer_ids] != tokens:
        raise RuntimeError("native pair split did not reconstruct generated tokens")
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None and answer_ids and answer_ids[-1] == int(eos_id):
        answer_ids.pop()
    if not cot_ids or not boundary_ids or not answer_ids:
        return False, None, None, None
    return True, cot_ids, boundary_ids, answer_ids


def _contiguous_shard_bounds(
    total: int,
    *,
    world_size: int,
    rank: int,
) -> tuple[int, int]:
    if total < 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid behavior-manifest shard context")
    base, remainder = divmod(total, world_size)
    start = rank * base + min(rank, remainder)
    return start, start + base + int(rank < remainder)


def _hf_shard_indices(
    total: int,
    *,
    world_size: int,
    rank: int,
    native_label_source: str,
) -> list[int]:
    """Match generated held-out behavior to periodic DP evaluation batches.

    Stage0-backed train behavior keeps cost-local contiguous shards.  Generated
    held-out behavior uses the same interleaved ``rank, rank+world, ...``
    partition as the training evaluator so greedy BF16 batches are
    byte-for-byte reproducible at the behavior-label gate.
    """

    if native_label_source not in {"generate", "stage0"}:
        raise ValueError("native_label_source must be 'generate' or 'stage0'")
    if total < 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid behavior-manifest shard context")
    if native_label_source == "generate":
        return list(range(rank, total, world_size))
    start, end = _contiguous_shard_bounds(total, world_size=world_size, rank=rank)
    return list(range(start, end))


def _generation_layout(
    args: argparse.Namespace,
    *,
    world_size: int,
) -> dict[str, int | str] | None:
    """Bind only generated HF labels to their live-eval physical layout."""

    if args.backend != "hf" or args.native_label_source != "generate":
        return None
    if world_size <= 0 or int(args.generation_batch_size) <= 0:
        raise ValueError("generation world size and batch size must be positive")
    return {
        "world_size": int(world_size),
        "batch_size": int(args.generation_batch_size),
        "rank_partition": "interleaved-v1",
    }


def _part_path(output: Path, *, world_size: int, rank: int) -> Path:
    return output.with_name(
        f".{output.name}.rank-{rank:05d}-of-{world_size:05d}.part.json"
    )


def _spec_path(output: Path, *, world_size: int, rank: int) -> Path:
    return output.with_name(
        f".{output.name}.rank-{rank:05d}-of-{world_size:05d}.spec.json"
    )


def _write_json(path: Path, payload: Any, *, indent: int | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=indent),
        encoding="utf-8",
    )
    temporary.replace(path)


def build_prompt_specs(
    records: Sequence[Mapping[str, Any]],
    *,
    collator: Any,
    default_task_type: str,
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    """Collapse rollouts only for prompt-level behavior generation."""

    from think_bridge.data.behavior_manifest import (
        gold_answer_key,
        prompt_token_ids_key,
    )
    from think_bridge.data.rollout_identity import (
        validate_stage0_rollout_identities,
    )

    specs: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}
    stage0_source = any("correct" in record for record in records)
    validated_prompt_keys = (
        validate_stage0_rollout_identities(
            records,
            rendered_prompt_key_fn=collator.prompt_group_key,
            source_name="Stage0 behavior source",
            show_progress=show_progress,
        )
        if stage0_source
        else ()
    )
    record_iter: Any = records
    if show_progress:
        try:
            from tqdm.auto import tqdm
        except ImportError:
            pass
        else:
            record_iter = tqdm(
                records,
                total=len(records),
                desc="Stage0 behavior: render/group prompts",
                unit="record",
                dynamic_ncols=True,
            )
    for index, record in enumerate(record_iter):
        correct = record.get("correct")
        if stage0_source and not isinstance(correct, bool):
            raise ValueError(
                f"every Stage 0 behavior record requires Boolean correct: index={index}"
            )
        if correct is not None and not isinstance(correct, bool):
            raise ValueError(
                "optional behavior-manifest correct must be a JSON Boolean: "
                f"index={index}"
            )
        prompt_ids = collator.rendered_prompt_ids(record, mode="hidden")
        direct_prompt_ids = collator.rendered_prompt_ids(record, mode="direct")
        prompt_key = (
            validated_prompt_keys[index]
            if stage0_source
            else prompt_token_ids_key(prompt_ids)
        )
        direct_key = prompt_token_ids_key(direct_prompt_ids)
        task_type = str(record.get("type") or default_task_type).strip().lower()
        answer = str(record.get("answer", "")).strip()
        signature = {
            "prompt_group_key": prompt_key,
            "direct_prompt_key": direct_key,
            "prompt_token_count": len(prompt_ids),
            "direct_prompt_token_count": len(direct_prompt_ids),
            "gold_answer_key": gold_answer_key(answer, task_type),
            "prompt_ids": list(prompt_ids),
            "direct_prompt_ids": list(direct_prompt_ids),
            "gold_answer": answer,
            "task_type": task_type,
        }
        previous = by_key.get(prompt_key)
        if previous is None:
            spec = {
                **signature,
                # Audit-only display fields come from the first occurrence.
                # They are not model inputs and do not redefine exact rendered
                # prompt identity when a messages-based record has stale
                # convenience text in ``question``.
                "source_id": str(record.get("id", "")),
                "question": str(record.get("question", "")),
                "n_correct": int(correct is True),
            }
            by_key[prompt_key] = spec
            specs.append(spec)
            continue
        for name, value in signature.items():
            if previous[name] != value:
                raise ValueError(
                    "direct render, gold, or task conflicts for one hidden prompt: "
                    f"prompt={prompt_key} field={name}"
                )
        previous["n_correct"] = int(previous["n_correct"]) + int(correct is True)
    return specs


def _configure_vllm_worker_multiprocessing() -> None:
    """Force CUDA-safe EngineCore creation before importing vLLM."""

    previous = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", "").strip().lower()
    if previous and previous != "spawn":
        print(
            "[behavior-vllm][WARN] overriding "
            f"VLLM_WORKER_MULTIPROC_METHOD={previous!r} with 'spawn'",
            flush=True,
        )
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def _vllm_token_prompts(
    prompt_ids: Sequence[Sequence[int]],
) -> list[dict[str, list[int]]]:
    """Pass exact rendered ids to vLLM; never decode/re-tokenize prompts."""

    return [{"prompt_token_ids": [int(token) for token in ids]} for ids in prompt_ids]


def _vllm_sampling_kwargs(
    *,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "n": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": int(max_new_tokens),
    }
    if eos_token_id is not None:
        # stop_token_ids is additive unless model-derived EOS handling is
        # disabled.  This matches the reference HF decoder's sole EOS token.
        kwargs["ignore_eos"] = True
        kwargs["stop_token_ids"] = [int(eos_token_id)]
    return kwargs


def _extract_vllm_token_ids(
    outputs: Sequence[Any], *, expected: int
) -> tuple[list[list[int]], list[bool]]:
    if len(outputs) != int(expected):
        raise RuntimeError(
            "vLLM behavior prompt count mismatch: "
            f"outputs={len(outputs)} expected={expected}"
        )
    rows: list[list[int]] = []
    hit_eos: list[bool] = []
    for prompt_index, request in enumerate(outputs):
        completions = list(getattr(request, "outputs", ()) or ())
        if len(completions) != 1:
            raise RuntimeError(
                "vLLM behavior requires exactly one completion per prompt: "
                f"prompt_index={prompt_index} outputs={len(completions)}"
            )
        token_ids = getattr(completions[0], "token_ids", None)
        if not isinstance(token_ids, (list, tuple)):
            raise RuntimeError(
                f"vLLM behavior completion lacks token ids: prompt_index={prompt_index}"
            )
        rows.append([int(token) for token in token_ids])
        # vLLM reports ``stop`` for EOS/explicit stop-token termination and
        # ``length`` for an exhausted generation budget.
        hit_eos.append(str(getattr(completions[0], "finish_reason", "")) == "stop")
    return rows, hit_eos


def _backend_identity(name: str) -> dict[str, str]:
    distribution = "vllm" if name == "vllm" else "transformers"
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"behavior backend dependency is missing: {distribution}"
        ) from exc
    return {"name": name, "version": str(version)}


def _offline() -> bool:
    return str(os.environ.get("HF_HUB_OFFLINE", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
    }


def _prepare_context(
    args: argparse.Namespace,
) -> dict[str, Any]:
    from transformers import AutoConfig, AutoTokenizer

    from think_bridge.data.behavior_manifest import (
        behavior_prompt_fingerprint,
        deterministic_decode_contract,
        fingerprint_behavior_source,
    )
    from think_bridge.data.dataset import load_data_file
    from think_bridge.data.templates import PromptContract

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=_offline(),
    )
    model_config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=_offline(),
    )
    records = load_data_file(args.input)
    if not records:
        raise ValueError("behavior manifest input is empty")
    if args.native_label_source == "stage0" and any(
        not isinstance(record.get("correct"), bool) for record in records
    ):
        raise ValueError(
            "native_label_source=stage0 requires Boolean correct on every "
            "Stage 0 record"
        )
    prompt_contract = PromptContract(
        task_type=str(args.task_type),
        max_position_embeddings=int(model_config.max_position_embeddings),
    )
    collator = _PromptRenderer(tokenizer, prompt_contract)
    specs = build_prompt_specs(
        records,
        collator=collator,
        default_task_type=args.task_type,
        show_progress=(not args.no_progress and int(os.environ.get("RANK", "0")) == 0),
    )
    if not specs:
        raise ValueError("behavior manifest produced no rendered prompts")
    return {
        "tokenizer": tokenizer,
        "prompt_config": prompt_contract,
        "model_config": model_config,
        # Do not retain the large CoT/answer raw while eight GPU workers run.
        # Prompt specs and fingerprints are the only generation-time state.
        "record_count": len(records),
        "specs": specs,
        "prompt_fingerprint": behavior_prompt_fingerprint(prompt_contract, tokenizer),
        "source_fingerprint": fingerprint_behavior_source(
            records,
            show_progress=(
                not args.no_progress and int(os.environ.get("RANK", "0")) == 0
            ),
            progress_desc="Stage0 behavior: fingerprint source",
        ),
        "decode_contract": deterministic_decode_contract(
            direct_max_new_tokens=args.max_new_answer_tokens,
            native_max_new_tokens=args.max_new_native_tokens,
            native_label_source=args.native_label_source,
        ),
    }


def _check_reuse_valid(
    args: argparse.Namespace,
    context: Mapping[str, Any],
    *,
    backend_identity: Mapping[str, str],
    generation_layout: Mapping[str, Any] | None = None,
) -> bool:
    from think_bridge.data.behavior_manifest import (
        legacy_behavior_prompt_fingerprint,
        load_behavior_manifest,
    )
    from think_bridge.data.direct_raw import (
        direct_decode_contract,
        load_direct_raw,
    )

    output = Path(args.output)
    if not args.reuse_if_valid or not output.is_file():
        return False
    try:
        cached = load_behavior_manifest(
            output,
            prompt_fingerprint=context["prompt_fingerprint"],
            decode_contract=context["decode_contract"],
            source_fingerprint=context["source_fingerprint"],
            inference_backend=backend_identity,
            native_label_source=args.native_label_source,
            generation_layout=generation_layout,
            legacy_prompt_fingerprint_for_tokenizer_source=(
                lambda tokenizer_source: legacy_behavior_prompt_fingerprint(
                    context["prompt_config"],
                    context["tokenizer"],
                    tokenizer_name_or_path=tokenizer_source,
                )
            ),
        )
    except (RuntimeError, ValueError) as exc:
        print(
            "[behavior-manifest] existing artifact failed validation; "
            f"refusing reuse: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return False
    direct_raw_output = str(getattr(args, "direct_raw_output", "")).strip()
    if direct_raw_output:
        try:
            direct_rows = load_direct_raw(
                direct_raw_output,
                prompt_fingerprint=context["prompt_fingerprint"],
                source_fingerprint=context["source_fingerprint"],
                inference_backend=backend_identity,
                decode_contract=direct_decode_contract(
                    max_new_tokens=args.max_new_answer_tokens
                ),
                legacy_prompt_fingerprint_for_tokenizer_source=(
                    lambda tokenizer_source: legacy_behavior_prompt_fingerprint(
                        context["prompt_config"],
                        context["tokenizer"],
                        tokenizer_name_or_path=tokenizer_source,
                    )
                ),
            )
        except (RuntimeError, ValueError) as exc:
            print(
                "[direct-raw] existing artifact failed validation; "
                f"refusing reuse with behavior: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return False
        expected_keys = [spec["prompt_group_key"] for spec in context["specs"]]
        if [row["prompt_group_key"] for row in direct_rows] != expected_keys:
            print(
                "[direct-raw] existing artifact prompt order/count mismatch; "
                "refusing reuse with behavior",
                flush=True,
            )
            return False
    if args.native_label_source == "generate" and any(
        row.get("native_correct") is True
        and row.get("native_think_complete") is not True
        for row in cached.values()
    ):
        print(
            "[behavior-manifest] existing generated-native artifact predates "
            "the think-complete correctness contract; refusing reuse",
            flush=True,
        )
        return False
    valid = len(cached) == len(context["specs"])
    if valid:
        print(
            f"[behavior-manifest] reuse backend={backend_identity['name']} "
            f"prompts={len(context['specs'])} output={output}",
            flush=True,
        )
    return valid


def _reuse_valid(args, context, *, backend_identity, generation_layout=None):
    """Never overwrite an immutable complete or incompatible Stage0 artifact."""
    paths = [Path(value) for value in (args.output, args.direct_raw_output) if value]
    if not any(path.exists() for path in paths):
        return False
    if not args.reuse_if_valid or not all(path.is_file() for path in paths):
        raise FileExistsError(
            "Stage0 outputs already exist or are partial; use a new output location"
        )
    from think_bridge.model.executor_identity import assert_model_source_identity

    for path in paths:
        metadata = json.loads(path.read_text())["metadata"]
        assert_model_source_identity(
            args.model,
            metadata.get("frozen_executor_identity"),
            local_files_only=_offline(),
            no_progress=args.no_progress,
        )
    valid = _check_reuse_valid(
        args,
        context,
        backend_identity=backend_identity,
        generation_layout=generation_layout,
    )
    if not valid:
        raise ValueError(
            "Existing Stage0 artifacts are incompatible; preserve them and use a new output location"
        )
    return True


def _rows_from_tokens(
    specs: Sequence[Mapping[str, Any]],
    direct_ids: Sequence[Sequence[int]],
    direct_hit_eos: Sequence[bool],
    native_ids: Sequence[Sequence[int]] | None,
    native_hit_eos: Sequence[bool] | None,
    *,
    tokenizer: Any,
    native_label_source: str,
) -> list[dict[str, Any]]:
    from think_bridge.eval.answer_match import judge_answer

    if native_label_source not in {"generate", "stage0"}:
        raise ValueError("native_label_source must be 'generate' or 'stage0'")
    if len(specs) != len(direct_ids):
        raise RuntimeError("behavior output count differs from prompt specifications")
    if len(specs) != len(direct_hit_eos) or any(
        not isinstance(value, bool) for value in direct_hit_eos
    ):
        raise RuntimeError(
            "direct finish-flag count differs from prompt specifications"
        )
    if native_label_source == "generate":
        if native_ids is None or len(specs) != len(native_ids):
            raise RuntimeError("native output count differs from prompt specifications")
        if native_hit_eos is None or len(specs) != len(native_hit_eos):
            raise RuntimeError(
                "native finish-flag count differs from prompt specifications"
            )
        native_rows: Sequence[Sequence[int] | None] = native_ids
        native_finished: Sequence[bool | None] = native_hit_eos
    else:
        if native_ids is not None or native_hit_eos is not None:
            raise RuntimeError(
                "Stage 0 native labels forbid newly generated native outputs"
            )
        native_rows = [None] * len(specs)
        native_finished = [None] * len(specs)
    rows: list[dict[str, Any]] = []
    for spec, direct_tokens, direct_stopped, native_tokens, native_stopped in zip(
        specs,
        direct_ids,
        direct_hit_eos,
        native_rows,
        native_finished,
    ):
        direct_text = tokenizer.decode(direct_tokens, skip_special_tokens=True)
        direct_correct = bool(
            judge_answer(direct_text, spec["gold_answer"], spec["task_type"])
        )
        n_correct = int(spec["n_correct"])
        if native_label_source == "stage0":
            native_correct = n_correct > 0
            native_think_complete: bool | None = None
            native_cot_count: int | None = None
            native_cot_ids: list[int] | None = None
            native_boundary_ids: list[int] | None = None
            native_answer_ids: list[int] | None = None
            native_hit_eos_value: bool | None = None
        else:
            if native_tokens is None:  # pragma: no cover - guarded above.
                raise RuntimeError("generated native tokens unexpectedly missing")
            if not isinstance(native_stopped, bool):
                raise RuntimeError("generated native finish flag unexpectedly missing")
            (
                native_think_complete,
                native_cot_ids,
                native_boundary_ids,
                native_answer_ids,
            ) = _split_exact_native_pair(
                native_tokens,
                tokenizer=tokenizer,
                hit_eos=native_stopped,
            )
            native_cot_count = (
                len(native_cot_ids) if native_cot_ids is not None else None
            )
            native_hit_eos_value = native_stopped
            native_answer_text = (
                tokenizer.decode(native_answer_ids, skip_special_tokens=True)
                if native_answer_ids is not None
                else ""
            )
            native_correct = bool(
                native_think_complete
                and judge_answer(
                    native_answer_text,
                    spec["gold_answer"],
                    spec["task_type"],
                )
            )
        # Observed reasoning is audit/donor-length evidence even when a run is
        # truncated or wrong. It is never promoted to a reusable D target.
        observed_cot_ids = None
        generated_ids = None
        if native_tokens is not None:
            generated_ids = [int(token) for token in native_tokens]
            close_start = _last_subsequence_start(
                generated_ids, tokenizer.encode("</think>", add_special_tokens=False)
            )
            if close_start is not None:
                observed_cot_ids = generated_ids[:close_start]
            else:
                observed_cot_ids = list(generated_ids)
                if observed_cot_ids and observed_cot_ids[-1] == tokenizer.eos_token_id:
                    observed_cot_ids.pop()
        rows.append(
            {
                "native_generated_ids": generated_ids,
                "native_observed_cot_ids": observed_cot_ids,
                "native_observed_cot_token_count": None
                if observed_cot_ids is None
                else len(observed_cot_ids),
                "prompt_group_key": spec["prompt_group_key"],
                "direct_prompt_key": spec["direct_prompt_key"],
                "prompt_token_count": spec["prompt_token_count"],
                "direct_prompt_token_count": spec["direct_prompt_token_count"],
                "gold_answer_key": spec["gold_answer_key"],
                "has_correct_view": n_correct > 0,
                "n_correct": n_correct,
                "direct_correct": direct_correct,
                "native_correct": native_correct,
                "need_z": native_correct and not direct_correct,
                "native_think_complete": native_think_complete,
                "native_cot_token_count": native_cot_count,
                "native_cot_ids": native_cot_ids,
                "native_boundary_ids": native_boundary_ids,
                "native_self_answer_ids": native_answer_ids,
                "native_hit_eos": native_hit_eos_value,
                # Rank shards carry the direct audit row beside its compact label.
                # ``_finalize`` splits it into a separate artifact and never writes
                # this private field into the behavior manifest.
                "_direct_raw": {
                    "source_id": spec["source_id"],
                    "question": spec["question"],
                    "gold_answer": spec["gold_answer"],
                    "task_type": spec["task_type"],
                    "prompt_group_key": spec["prompt_group_key"],
                    "direct_prompt_key": spec["direct_prompt_key"],
                    "direct_output": direct_text,
                    "direct_token_ids": [int(token) for token in direct_tokens],
                    "direct_hit_eos": bool(direct_stopped),
                    "direct_correct": direct_correct,
                },
            }
        )
    return rows


def _progress(iterable: Any, *, args: argparse.Namespace, rank: int, total: int) -> Any:
    if args.no_progress:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        print("[behavior-manifest][WARN] tqdm unavailable; continuing without progress")
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=f"behavior-{args.backend} rank{rank}",
        unit="block",
        dynamic_ncols=True,
        position=rank,
    )


def _generate_vllm_rows(
    args: argparse.Namespace,
    specs: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    rank: int,
) -> list[dict[str, Any]]:
    if args.native_label_source != "generate":
        raise ValueError("native_label_source=stage0 requires the HF direct backend")
    _configure_vllm_worker_multiprocessing()
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        dtype=args.torch_dtype,
        trust_remote_code=True,
        generation_config="vllm",
        seed=int(args.seed),
    )
    eos_id = tokenizer.eos_token_id
    direct_params = SamplingParams(
        **_vllm_sampling_kwargs(
            max_new_tokens=args.max_new_answer_tokens,
            eos_token_id=eos_id,
        )
    )
    native_params = SamplingParams(
        **_vllm_sampling_kwargs(
            max_new_tokens=args.max_new_native_tokens,
            eos_token_id=eos_id,
        )
    )
    chunk = int(args.generation_batch_size)
    starts: Any = range(0, len(specs), chunk)
    starts = _progress(
        starts,
        args=args,
        rank=rank,
        total=(len(specs) + chunk - 1) // chunk,
    )
    rows: list[dict[str, Any]] = []
    for offset in starts:
        batch = specs[offset : offset + chunk]
        direct_outputs = llm.generate(
            _vllm_token_prompts([spec["direct_prompt_ids"] for spec in batch]),
            direct_params,
            use_tqdm=False,
        )
        native_outputs = llm.generate(
            _vllm_token_prompts([spec["prompt_ids"] for spec in batch]),
            native_params,
            use_tqdm=False,
        )
        direct_token_ids, direct_hit_eos = _extract_vllm_token_ids(
            direct_outputs, expected=len(batch)
        )
        native_token_ids, native_finished = _extract_vllm_token_ids(
            native_outputs, expected=len(batch)
        )
        rows.extend(
            _rows_from_tokens(
                batch,
                direct_token_ids,
                direct_hit_eos,
                native_token_ids,
                native_finished,
                tokenizer=tokenizer,
                native_label_source=args.native_label_source,
            )
        )
    return rows


def _greedy_generate_hf(
    model: Any,
    prompt_ids: Sequence[Sequence[int]],
    *,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> tuple[list[list[int]], list[bool]]:
    """Decode by exact argmax with explicit left padding and per-row EOS."""

    import torch
    from think_bridge.model.cache_utils import new_cache

    rows = [list(map(int, ids)) for ids in prompt_ids]
    if not rows:
        return [], []
    if int(max_new_tokens) <= 0:
        raise ValueError("max_new_tokens must be positive")
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    embedding = model.get_input_embeddings()
    output_head = model.get_output_embeddings()
    base_model = getattr(model, "model", None)
    pad_id = int(eos_token_id) if eos_token_id is not None else 0
    width = max(len(row) for row in rows)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for index, row in enumerate(rows):
        input_ids[index, width - len(row) :] = torch.tensor(
            row, dtype=torch.long, device=device
        )
        attention_mask[index, width - len(row) :] = 1

    def forward_executor(
        *,
        attention_mask: Any,
        position_ids: Any,
        cache_position: Any,
        cache: Any,
        input_ids: Any = None,
        inputs_embeds: Any = None,
    ) -> tuple[Any, Any]:
        kwargs: dict[str, Any] = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "cache_position": cache_position,
            "past_key_values": cache,
            "use_cache": True,
        }
        if input_ids is not None:
            kwargs["input_ids"] = input_ids
        else:
            kwargs["inputs_embeds"] = inputs_embeds
        if base_model is not None:
            output = base_model(**kwargs)
            return output.last_hidden_state, output.past_key_values
        kwargs["output_hidden_states"] = True
        output = model(**kwargs)
        return output.hidden_states[-1], output.past_key_values

    lengths = [len(row) for row in rows]
    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp_min(0)
    cache = new_cache()
    with torch.no_grad():
        prompt_hidden, cache = forward_executor(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=torch.arange(width, device=device),
            cache=cache,
        )
        next_position = torch.tensor(lengths, dtype=torch.long, device=device)
        logits = output_head(prompt_hidden[:, -1, :])
        finished = torch.zeros(len(rows), dtype=torch.bool, device=device)
        capacity = int(max_new_tokens)
        tail = max(capacity - 1, 0)
        prefix_width = int(attention_mask.size(1))
        decode_attention = torch.ones(
            (len(rows), prefix_width + tail),
            dtype=attention_mask.dtype,
            device=device,
        )
        decode_attention[:, :prefix_width].copy_(attention_mask)
        decode_positions = next_position.view(len(rows), 1) + torch.arange(
            tail, device=device
        ).view(1, -1)
        cache_positions = torch.arange(prefix_width, prefix_width + tail, device=device)
        sampled_tokens = torch.empty(
            (len(rows), capacity), dtype=torch.long, device=device
        )
        sampled_valid = torch.zeros(
            (len(rows), capacity), dtype=torch.bool, device=device
        )
        steps_taken = 0
        for step_index in range(capacity):
            next_token = logits.argmax(dim=-1)
            active = ~finished
            sampled_tokens[:, step_index].copy_(next_token)
            sampled_valid[:, step_index].copy_(active)
            if eos_token_id is not None:
                finished.logical_or_(active & next_token.eq(int(eos_token_id)))
            steps_taken = step_index + 1
            if steps_taken >= capacity:
                break
            if steps_taken % 16 == 0 and bool(finished.all()):
                break
            feed = next_token.masked_fill(finished, pad_id)
            attention_end = prefix_width + step_index + 1
            hidden, cache = forward_executor(
                inputs_embeds=embedding(feed.view(len(rows), 1)).to(model_dtype),
                attention_mask=decode_attention[:, :attention_end],
                position_ids=decode_positions[:, step_index : step_index + 1],
                cache_position=cache_positions[step_index : step_index + 1],
                cache=cache,
            )
            logits = output_head(hidden[:, -1, :])

    token_rows = sampled_tokens[:, :steps_taken].detach().to("cpu").tolist()
    valid_rows = sampled_valid[:, :steps_taken].detach().to("cpu").tolist()
    ended_rows = finished.detach().to("cpu").tolist()
    return _finalize_greedy_rows(
        token_rows,
        valid_rows,
        ended_rows,
        eos_token_id=eos_token_id,
    )


def _finalize_greedy_rows(
    token_rows: Sequence[Sequence[int]],
    valid_rows: Sequence[Sequence[bool]],
    ended_rows: Sequence[bool],
    *,
    eos_token_id: int | None,
) -> tuple[list[list[int]], list[bool]]:
    """Restore answer rows from buffered actions and remove terminal EOS."""

    if not (len(token_rows) == len(valid_rows) == len(ended_rows)):
        raise ValueError("buffered greedy row counts differ")
    eos = None if eos_token_id is None else int(eos_token_id)
    decoded: list[list[int]] = []
    hit_eos: list[bool] = []
    for tokens, valid, ended in zip(token_rows, valid_rows, ended_rows):
        if len(tokens) != len(valid):
            raise ValueError("buffered greedy token and validity widths differ")
        policy_ids = [
            int(token) for token, is_valid in zip(tokens, valid) if bool(is_valid)
        ]
        if bool(ended):
            if eos is None or not policy_ids or policy_ids[-1] != eos:
                raise RuntimeError(
                    "finished greedy row must end with the configured EOS"
                )
            decoded.append(policy_ids[:-1])
            hit_eos.append(True)
        else:
            if eos is not None and eos in policy_ids:
                raise RuntimeError("unfinished greedy row unexpectedly contains EOS")
            decoded.append(policy_ids)
            hit_eos.append(False)
    return decoded, hit_eos


def _generate_hf_rows(
    args: argparse.Namespace,
    specs: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    model: Any,
    rank: int,
) -> list[dict[str, Any]]:
    chunk = int(args.generation_batch_size)
    starts: Any = range(0, len(specs), chunk)
    starts = _progress(
        starts,
        args=args,
        rank=rank,
        total=(len(specs) + chunk - 1) // chunk,
    )
    eos_id = tokenizer.eos_token_id
    rows: list[dict[str, Any]] = []
    for offset in starts:
        batch = specs[offset : offset + chunk]
        direct_ids, direct_hit_eos = _greedy_generate_hf(
            model,
            [spec["direct_prompt_ids"] for spec in batch],
            max_new_tokens=args.max_new_answer_tokens,
            eos_token_id=eos_id,
        )
        native_result = (
            _greedy_generate_hf(
                model,
                [spec["prompt_ids"] for spec in batch],
                max_new_tokens=args.max_new_native_tokens,
                eos_token_id=eos_id,
            )
            if args.native_label_source == "generate"
            else None
        )
        native_ids = None if native_result is None else native_result[0]
        native_finished = None if native_result is None else native_result[1]
        rows.extend(
            _rows_from_tokens(
                batch,
                direct_ids,
                direct_hit_eos,
                native_ids,
                native_finished,
                tokenizer=tokenizer,
                native_label_source=args.native_label_source,
            )
        )
    return rows


def _read_parts(output: Path, *, world_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank in range(world_size):
        part = _part_path(output, world_size=world_size, rank=rank)
        try:
            shard = json.loads(part.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"behavior manifest shard cannot be read: {part}"
            ) from exc
        if not isinstance(shard, list):
            raise ValueError(f"behavior manifest shard is not an array: {part}")
        rows.extend(shard)
    return rows


def _index_hf_rows(
    indices: Sequence[int],
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(indices) != len(rows):
        raise ValueError(
            "HF behavior shard index/row counts differ: "
            f"indices={len(indices)} rows={len(rows)}"
        )
    indexed: list[dict[str, Any]] = []
    for index, row in zip(indices, rows):
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError(f"invalid HF behavior shard global index: {index!r}")
        if not isinstance(row, Mapping):
            raise ValueError("HF behavior shard row must be an object")
        indexed.append({"global_index": int(index), "row": dict(row)})
    return indexed


def _restore_indexed_hf_rows(
    indexed_rows: Sequence[Mapping[str, Any]],
    *,
    expected_total: int,
) -> list[dict[str, Any]]:
    if expected_total < 0:
        raise ValueError("HF behavior expected_total must be nonnegative")
    restored: list[dict[str, Any] | None] = [None] * expected_total
    for item in indexed_rows:
        if not isinstance(item, Mapping) or set(item) != {"global_index", "row"}:
            raise ValueError("HF indexed behavior shard fields are incompatible")
        index = item.get("global_index")
        row = item.get("row")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < expected_total
        ):
            raise ValueError(f"HF behavior global index is out of range: {index!r}")
        if restored[index] is not None:
            raise ValueError(f"HF behavior duplicate global index: {index}")
        if not isinstance(row, Mapping):
            raise ValueError(f"HF indexed behavior row is not an object: index={index}")
        restored[index] = dict(row)
    missing = [index for index, row in enumerate(restored) if row is None]
    if missing:
        raise ValueError(
            "HF indexed behavior shards lack global indices: "
            f"count={len(missing)} first={missing[:8]}"
        )
    return [row for row in restored if row is not None]


def _remove_parts(output: Path, *, world_size: int) -> None:
    for rank in range(world_size):
        _part_path(output, world_size=world_size, rank=rank).unlink(missing_ok=True)


def _wait_for_parts(
    output: Path,
    *,
    world_size: int,
    timeout_seconds: float,
    no_progress: bool,
    poll_seconds: float = 1.0,
    expected_row_count: int | None = None,
) -> list[dict[str, Any]]:
    """Wait for independent HF ranks without a long-running NCCL collective."""

    if world_size <= 0 or timeout_seconds <= 0 or poll_seconds <= 0:
        raise ValueError("behavior shard wait parameters must be positive")
    started = time.monotonic()
    last_reported = -1
    last_report_time = started
    while True:
        ready = sum(
            int(_part_path(output, world_size=world_size, rank=rank).is_file())
            for rank in range(world_size)
        )
        now = time.monotonic()
        elapsed = now - started
        if ready == world_size:
            if not no_progress:
                print(
                    "[behavior-hf][DP] shards ready "
                    f"{ready}/{world_size} elapsed={elapsed:.1f}s",
                    flush=True,
                )
            rows = _read_parts(output, world_size=world_size)
            if expected_row_count is None:
                return rows
            return _restore_indexed_hf_rows(rows, expected_total=expected_row_count)
        if elapsed >= timeout_seconds:
            missing = [
                rank
                for rank in range(world_size)
                if not _part_path(output, world_size=world_size, rank=rank).is_file()
            ]
            raise TimeoutError(
                "behavior HF shard wait timed out: "
                f"ready={ready}/{world_size} elapsed={elapsed:.1f}s "
                f"timeout={timeout_seconds:.1f}s missing_ranks={missing}"
            )
        if not no_progress and (
            ready != last_reported or now - last_report_time >= 60.0
        ):
            print(
                "[behavior-hf][DP] waiting shards "
                f"ready={ready}/{world_size} elapsed={elapsed:.1f}s "
                f"timeout={timeout_seconds:.1f}s",
                flush=True,
            )
            last_reported = ready
            last_report_time = now
        time.sleep(min(poll_seconds, max(timeout_seconds - elapsed, 0.001)))


def _finalize(
    args: argparse.Namespace,
    context: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    backend_identity: Mapping[str, str],
    world_size: int,
    generation_layout: Mapping[str, Any] | None = None,
) -> None:
    from think_bridge.data.behavior_manifest import (
        BEHAVIOR_MANIFEST_SCHEMA,
        load_behavior_manifest,
    )
    from think_bridge.data.direct_raw import (
        DIRECT_RAW_SCHEMA,
        direct_decode_contract,
        load_direct_raw,
    )

    specs = context["specs"]
    expected_keys = [spec["prompt_group_key"] for spec in specs]
    actual_keys = [row.get("prompt_group_key") for row in rows]
    if actual_keys != expected_keys:
        raise ValueError("behavior shards did not preserve unique prompt order")
    direct_rows = [row.get("_direct_raw") for row in rows]
    if any(not isinstance(row, Mapping) for row in direct_rows):
        raise ValueError("behavior shard lacks an exact direct raw row")
    if [row.get("prompt_group_key") for row in direct_rows] != expected_keys:
        raise ValueError("direct raw shards did not preserve unique prompt order")
    for index, (behavior_row, direct_row) in enumerate(zip(rows, direct_rows)):
        if direct_row.get("direct_prompt_key") != behavior_row.get(
            "direct_prompt_key"
        ) or direct_row.get("direct_correct") is not behavior_row.get("direct_correct"):
            raise ValueError(
                f"direct raw and compact behavior labels/identity differ: index={index}"
            )
    compact_rows = [
        {name: value for name, value in row.items() if name != "_direct_raw"}
        for row in rows
    ]
    from think_bridge.model.executor_identity import model_source_identity

    frozen_identity = model_source_identity(
        args.model, local_files_only=True, no_progress=args.no_progress
    )
    output = Path(args.output)
    created_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "schema_version": BEHAVIOR_MANIFEST_SCHEMA,
        "status": "complete",
        "metadata": {
            "created_at": created_at,
            "frozen_executor_identity": frozen_identity,
            "model_source": args.model,
            "tokenizer_source": getattr(
                context["tokenizer"], "name_or_path", args.model
            ),
            "data_source": str(Path(args.input)),
            "prompt_fingerprint": context["prompt_fingerprint"],
            "source_fingerprint": context["source_fingerprint"],
            "source_record_count": int(context["record_count"]),
            "prompt_count": len(rows),
            "inference_backend": dict(backend_identity),
            "native_label_source": args.native_label_source,
            "decode_contract": context["decode_contract"],
            **(
                {"generation_layout": dict(generation_layout)}
                if generation_layout is not None
                else {}
            ),
        },
        "prompts": compact_rows,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    _write_json(temporary, payload, indent=2)
    load_behavior_manifest(
        temporary,
        prompt_fingerprint=context["prompt_fingerprint"],
        decode_contract=context["decode_contract"],
        source_fingerprint=context["source_fingerprint"],
        inference_backend=backend_identity,
        native_label_source=args.native_label_source,
        generation_layout=generation_layout,
    )
    direct_output_value = str(getattr(args, "direct_raw_output", "")).strip()
    direct_temporary: Path | None = None
    if direct_output_value:
        direct_output = Path(direct_output_value)
        direct_payload = {
            "schema_version": DIRECT_RAW_SCHEMA,
            "status": "complete",
            "metadata": {
                "created_at": created_at,
                "frozen_executor_identity": frozen_identity,
                "model_source": args.model,
                "tokenizer_source": getattr(
                    context["tokenizer"], "name_or_path", args.model
                ),
                "data_source": str(Path(args.input)),
                "prompt_fingerprint": context["prompt_fingerprint"],
                "source_fingerprint": context["source_fingerprint"],
                "source_record_count": int(context["record_count"]),
                "prompt_count": len(direct_rows),
                "inference_backend": dict(backend_identity),
                "decode_contract": direct_decode_contract(
                    max_new_tokens=args.max_new_answer_tokens
                ),
            },
            "records": direct_rows,
        }
        direct_temporary = direct_output.with_suffix(direct_output.suffix + ".tmp")
        _write_json(direct_temporary, direct_payload, indent=2)
        load_direct_raw(
            direct_temporary,
            prompt_fingerprint=context["prompt_fingerprint"],
            source_fingerprint=context["source_fingerprint"],
            inference_backend=backend_identity,
            decode_contract=direct_decode_contract(
                max_new_tokens=args.max_new_answer_tokens
            ),
        )
    # Install only after all requested temporary artifacts pass validation.
    if direct_temporary is not None:
        direct_temporary.replace(Path(direct_output_value))
    temporary.replace(output)
    _remove_parts(output, world_size=world_size)
    print(
        "[behavior-manifest] complete "
        f"backend={backend_identity['name']} records={context['record_count']} "
        f"prompts={len(rows)} "
        f"with_correct={sum(int(row['has_correct_view']) for row in rows)} "
        f"direct_correct={sum(int(row['direct_correct']) for row in rows)} "
        f"native_hit_eos={sum(int(row['native_hit_eos'] is True) for row in rows)} "
        f"native_complete={sum(int(row['native_think_complete'] is True) for row in rows)} "
        f"native_correct={sum(int(row['native_correct']) for row in rows)} "
        f"need_z={sum(int(row['need_z']) for row in rows)} output={output}",
        flush=True,
    )
    if direct_output_value:
        print(
            "[direct-raw] complete "
            f"backend={backend_identity['name']} prompts={len(direct_rows)} "
            f"output={direct_output_value}",
            flush=True,
        )


def _device_tokens(world_size: int) -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    if tokens:
        if len(tokens) < world_size:
            raise RuntimeError(
                "vLLM behavior data parallelism exceeds CUDA_VISIBLE_DEVICES: "
                f"world={world_size} visible={tokens}"
            )
        return tokens[:world_size]
    return [str(rank) for rank in range(world_size)]


def _vllm_child_command(
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "think_bridge.cli.precompute_behavior_manifest",
        "--model",
        args.model,
        "--input",
        args.input,
        "--output",
        args.output,
        "--task_type",
        args.task_type,
        "--torch_dtype",
        args.torch_dtype,
        "--attn_implementation",
        args.attn_implementation,
        "--generation_batch_size",
        str(args.generation_batch_size),
        "--max_new_answer_tokens",
        str(args.max_new_answer_tokens),
        "--max_new_native_tokens",
        str(args.max_new_native_tokens),
        "--backend",
        "vllm",
        "--native_label_source",
        args.native_label_source,
        "--dp_world_size",
        "1",
        "--gpu_memory_utilization",
        str(args.gpu_memory_utilization),
        "--seed",
        str(args.seed),
        "--spec_shard",
        args.spec_shard,
    ]
    if args.no_progress:
        command.append("--no_progress")
    return command


def _run_vllm_parent(args: argparse.Namespace) -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("vLLM behavior data parallelism cannot run inside torchrun")
    _configure_vllm_worker_multiprocessing()
    context = _prepare_context(args)
    backend = _backend_identity("vllm")
    if _reuse_valid(
        args,
        context,
        backend_identity=backend,
    ):
        return
    world_size = int(args.dp_world_size)
    if world_size == 1:
        rows = _generate_vllm_rows(
            args, context["specs"], tokenizer=context["tokenizer"], rank=0
        )
        _write_json(
            _part_path(Path(args.output), world_size=1, rank=0),
            rows,
            indent=None,
        )
        _finalize(
            args,
            context,
            rows,
            backend_identity=backend,
            world_size=1,
        )
        return

    devices = _device_tokens(world_size)
    print(
        f"[behavior-vllm][DP] world_size={world_size} tp=1 "
        f"prompts={len(context['specs'])} chunk={args.generation_batch_size}",
        flush=True,
    )
    processes: list[subprocess.Popen[Any]] = []
    spec_paths: list[Path] = []
    try:
        for rank, device in enumerate(devices):
            start, end = _contiguous_shard_bounds(
                len(context["specs"]), world_size=world_size, rank=rank
            )
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = device
            env[_VLLM_CHILD_ENV] = "1"
            env[_VLLM_RANK_ENV] = str(rank)
            env[_VLLM_WORLD_ENV] = str(world_size)
            spec_path = _spec_path(Path(args.output), world_size=world_size, rank=rank)
            _write_json(spec_path, context["specs"][start:end], indent=None)
            spec_paths.append(spec_path)
            child_args = argparse.Namespace(**vars(args))
            child_args.spec_shard = str(spec_path)
            print(
                f"[behavior-vllm][DP] spawn rank{rank} gpu={device} "
                f"prompts=[{start},{end})",
                flush=True,
            )
            processes.append(
                subprocess.Popen(
                    _vllm_child_command(child_args),
                    env=env,
                )
            )
        exit_codes = [process.wait() for process in processes]
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for rank in range(world_size):
            _part_path(Path(args.output), world_size=world_size, rank=rank).unlink(
                missing_ok=True
            )
        raise
    finally:
        for spec_path in spec_paths:
            spec_path.unlink(missing_ok=True)
    failures = [(rank, code) for rank, code in enumerate(exit_codes) if code != 0]
    if failures:
        for rank, code in failures:
            print(
                f"[behavior-vllm][DP][ERROR] rank{rank} exit={code}",
                flush=True,
            )
        for rank in range(world_size):
            _part_path(Path(args.output), world_size=world_size, rank=rank).unlink(
                missing_ok=True
            )
        raise SystemExit(1)
    rows = _read_parts(Path(args.output), world_size=world_size)
    _finalize(
        args,
        context,
        rows,
        backend_identity=backend,
        world_size=world_size,
    )


def _run_vllm_child(args: argparse.Namespace) -> None:
    rank = int(os.environ[_VLLM_RANK_ENV])
    world_size = int(os.environ[_VLLM_WORLD_ENV])
    spec_path = Path(args.spec_shard)
    try:
        specs = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"vLLM behavior specification shard cannot be read: {spec_path}"
        ) from exc
    if not isinstance(specs, list) or any(not isinstance(row, dict) for row in specs):
        raise ValueError(
            f"vLLM behavior specification is not an object array: {spec_path}"
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=_offline(),
    )
    rows = _generate_vllm_rows(
        args,
        specs,
        tokenizer=tokenizer,
        rank=rank,
    )
    _write_json(
        _part_path(Path(args.output), world_size=world_size, rank=rank),
        rows,
        indent=None,
    )


def _run_hf(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("HF behavior precomputation requires a CUDA GPU")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid distributed HF behavior context")
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")
    context = _prepare_context(args)
    backend = _backend_identity("hf")
    generation_layout = _generation_layout(args, world_size=world_size)
    reuse = False
    if rank == 0:
        reuse = _reuse_valid(
            args,
            context,
            backend_identity=backend,
            generation_layout=generation_layout,
        )
    if distributed:
        flag = torch.tensor([int(reuse)], dtype=torch.int32, device=device)
        dist.broadcast(flag, src=0)
        reuse = bool(flag.item())
    if reuse:
        if distributed:
            dist.barrier()
            dist.destroy_process_group()
        return
    output = Path(args.output)
    if rank == 0:
        # A previous interrupted launch may have left complete-looking shards.
        # Remove them while all ranks are still synchronized so rank0 can only
        # observe files produced by this launch.
        _remove_parts(output, world_size=world_size)
    if distributed:
        # NCCL is needed only for the small identity/reuse broadcasts above.
        # Generation time is highly uneven across prompts, so a collective
        # after generation would turn the default NCCL timeout into a false
        # job failure.  From here each rank writes one atomic shard instead.
        dist.barrier()
        dist.destroy_process_group()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=getattr(torch, args.torch_dtype),
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
        local_files_only=_offline(),
    ).to(device)
    model.eval()
    global_indices = _hf_shard_indices(
        len(context["specs"]),
        world_size=world_size,
        rank=rank,
        native_label_source=args.native_label_source,
    )
    rows = _generate_hf_rows(
        args,
        [context["specs"][index] for index in global_indices],
        tokenizer=context["tokenizer"],
        model=model,
        rank=rank,
    )
    _write_json(
        _part_path(output, world_size=world_size, rank=rank),
        _index_hf_rows(global_indices, rows),
        indent=None,
    )
    if rank == 0:
        merged = _wait_for_parts(
            output,
            world_size=world_size,
            timeout_seconds=float(args.shard_wait_timeout_seconds),
            no_progress=bool(args.no_progress),
            expected_row_count=len(context["specs"]),
        )
        _finalize(
            args,
            context,
            merged,
            backend_identity=backend,
            world_size=world_size,
            generation_layout=generation_layout,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build deterministic frozen-executor behavior labels"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--direct_raw_output",
        default="",
        help="Optional raw greedy-direct artifact written by the same HF run",
    )
    parser.add_argument("--task_type", default="math", choices=("math", "native_qa"))
    parser.add_argument(
        "--torch_dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
    )
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--generation_batch_size", type=int, default=256)
    parser.add_argument("--max_new_answer_tokens", type=int, default=2048)
    parser.add_argument("--max_new_native_tokens", type=int, default=8192)
    parser.add_argument("--backend", choices=("vllm", "hf"), default="vllm")
    parser.add_argument(
        "--native_label_source",
        choices=("generate", "stage0"),
        default="generate",
        help="Generate both labels, or reuse native correctness from Stage 0",
    )
    parser.add_argument("--dp_world_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shard_wait_timeout_seconds",
        type=float,
        default=14400.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--reuse_if_valid", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--spec_shard", default="", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if (
        min(
            args.generation_batch_size,
            args.max_new_answer_tokens,
            args.max_new_native_tokens,
            args.dp_world_size,
        )
        <= 0
    ):
        raise ValueError(
            "behavior batch, generation, and parallelism values must be positive"
        )
    if not 0.0 < float(args.gpu_memory_utilization) <= 1.0:
        raise ValueError("gpu_memory_utilization must be in (0,1]")
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    if args.shard_wait_timeout_seconds <= 0:
        raise ValueError("shard_wait_timeout_seconds must be positive")
    if args.backend == "hf" and args.dp_world_size != 1:
        raise ValueError(
            "multi-GPU HF behavior uses torchrun WORLD_SIZE, not dp_world_size"
        )
    if args.native_label_source == "stage0" and args.backend != "hf":
        raise ValueError("native_label_source=stage0 requires backend=hf")
    if args.direct_raw_output and (
        args.backend != "hf" or args.native_label_source != "stage0"
    ):
        raise ValueError(
            "direct_raw_output requires exact HF with Stage 0 native labels"
        )
    if args.direct_raw_output and Path(args.direct_raw_output) == Path(args.output):
        raise ValueError("direct raw and compact behavior need distinct output files")
    output_paths = [
        Path(value)
        for value in (args.output, args.direct_raw_output)
        if str(value).strip()
    ]
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("behavior/direct outputs must be distinct")
    if args.backend == "vllm":
        if os.environ.get(_VLLM_CHILD_ENV) == "1":
            _run_vllm_child(args)
        else:
            _run_vllm_parent(args)
    else:
        _run_hf(args)


if __name__ == "__main__":
    main()
