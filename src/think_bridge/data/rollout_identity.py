"""Pure Stage0 rollout-identity contract shared by every consumer."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any


STAGE0_SCHEMA_VERSION = 1
_LEGACY_GENERATION_CONTRACT_VERSION = 2

_COMPLETE_PROVENANCE_FIELDS = frozenset({
    "generation_backend",
    "generation_model",
    "generation_seed",
    "generation_temperature",
    "generation_top_p",
    "generation_max_new_tokens",
    "finish_reason",
    "generated_token_count",
    "think_closed",
    "generation_complete",
})


def _progress(
    records: Sequence[Mapping[str, Any]],
    *,
    enabled: bool,
    desc: str,
) -> Any:
    if not enabled:
        return records
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return records
    return tqdm(
        records,
        total=len(records),
        desc=desc,
        unit="record",
        dynamic_ncols=True,
    )


def _validate_complete_rollout_identities(
    records: Sequence[Mapping[str, Any]],
    *,
    rendered_prompt_key_fn: Callable[[Mapping[str, Any]], str],
    source_name: str = "Stage0 source",
    show_progress: bool = False,
) -> tuple[str, ...]:
    """Validate the complete generation and provenance contract strictly."""

    if not records:
        raise ValueError(f"{source_name} must not be empty")

    seen_rollouts: set[tuple[str, int]] = set()
    id_to_prompt: dict[str, str] = {}
    id_to_rows: dict[str, list[Mapping[str, Any]]] = {}
    prompt_keys: list[str] = []
    global_generation_signature: tuple[Any, ...] | None = None
    global_n_sampled: int | None = None
    record_iter = _progress(
        records,
        enabled=show_progress,
        desc=f"{source_name}: rollout identity",
    )
    for index, record in enumerate(record_iter):
        schema_version = record.get("schema_version")
        legacy_version = record.get("stage0_generation_contract_version")
        current_schema = schema_version == STAGE0_SCHEMA_VERSION and legacy_version is None
        legacy_schema = (
            schema_version is None
            and legacy_version == _LEGACY_GENERATION_CONTRACT_VERSION
            and not isinstance(legacy_version, bool)
        )
        if not (current_schema or legacy_schema):
            raise ValueError(
                f"{source_name} has an unsupported Stage 0 schema: "
                f"index={index} schema_version={schema_version!r}"
            )
        correct = record.get("correct")
        answer_correct = record.get("answer_correct")
        if not isinstance(correct, bool) or not isinstance(answer_correct, bool):
            raise ValueError(
                f"{source_name} correct/answer_correct must be JSON Booleans: "
                f"index={index}"
            )
        if not str(record.get("answer") or "").strip():
            raise ValueError(f"{source_name} gold answer is empty: index={index}")
        n_sampled = record.get("n_sampled")
        n_correct = record.get("n_correct")
        if (
            isinstance(n_sampled, bool)
            or not isinstance(n_sampled, int)
            or n_sampled <= 0
            or isinstance(n_correct, bool)
            or not isinstance(n_correct, int)
            or not 0 <= n_correct <= n_sampled
        ):
            raise ValueError(
                f"{source_name} has invalid n_sampled/n_correct: index={index} "
                f"n_sampled={n_sampled!r} n_correct={n_correct!r}"
            )
        if global_n_sampled is None:
            global_n_sampled = int(n_sampled)
        elif n_sampled != global_n_sampled:
            raise ValueError(
                f"{source_name} requires one n_sampled value across prompts: "
                f"expected={global_n_sampled} index={index} actual={n_sampled}"
            )
        expected_bucket = (
            "none_correct"
            if n_correct == 0
            else "all_correct"
            if n_correct == n_sampled
            else "mixed"
        )
        if record.get("bucket") != expected_bucket:
            raise ValueError(
                f"{source_name} bucket differs from n_correct/n_sampled: "
                f"index={index} expected={expected_bucket!r} "
                f"actual={record.get('bucket')!r}"
            )

        backend = str(record.get("generation_backend") or "").strip().lower()
        model = str(record.get("generation_model") or "").strip()
        generation_seed = record.get("generation_seed")
        temperature = record.get("generation_temperature")
        top_p = record.get("generation_top_p")
        max_new_tokens = record.get("generation_max_new_tokens")
        if backend not in {"hf", "vllm"} or not model:
            raise ValueError(
                f"{source_name} has an invalid generation backend/model: index={index}"
            )
        if (
            isinstance(generation_seed, bool)
            or not isinstance(generation_seed, int)
            or generation_seed < 0
        ):
            raise ValueError(
                f"{source_name} generation_seed must be nonnegative: index={index}"
            )
        numeric_generation = {
            "generation_temperature": temperature,
            "generation_top_p": top_p,
        }
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in numeric_generation.values()
        ):
            raise ValueError(
                f"{source_name} generation temperature/top_p must be finite: "
                f"index={index}"
            )
        if float(temperature) < 0.0 or not 0.0 < float(top_p) <= 1.0:
            raise ValueError(
                f"{source_name} generation temperature/top_p is out of range: index={index}"
            )
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens <= 0
        ):
            raise ValueError(
                f"{source_name} generation_max_new_tokens must be positive: "
                f"index={index}"
            )
        generation_signature = (
            backend,
            model,
            float(temperature),
            float(top_p),
            int(max_new_tokens),
        )
        if global_generation_signature is None:
            global_generation_signature = generation_signature
        elif generation_signature != global_generation_signature:
            raise ValueError(
                f"{source_name} generation parameters differ within the artifact: index={index}"
            )

        finish_reason = str(record.get("finish_reason") or "").strip().lower()
        generated_count = record.get("generated_token_count")
        think_closed = record.get("think_closed")
        generation_complete = record.get("generation_complete")
        if finish_reason not in {"stop", "length", "unknown"}:
            raise ValueError(
                f"{source_name} has an invalid finish_reason: index={index} "
                f"value={finish_reason!r}"
            )
        if (
            isinstance(generated_count, bool)
            or not isinstance(generated_count, int)
            or generated_count < 0
        ):
            raise ValueError(
                f"{source_name} generated_token_count must be nonnegative: "
                f"index={index}"
            )
        if not isinstance(think_closed, bool) or not isinstance(
            generation_complete, bool
        ):
            raise ValueError(
                f"{source_name} think_closed/generation_complete must be Boolean: "
                f"index={index}"
            )
        cot = str(record.get("cot") or "").strip()
        self_answer = str(record.get("self_answer") or "").strip()
        expected_complete = bool(
            finish_reason == "stop" and think_closed and cot and self_answer
        )
        if generation_complete is not expected_complete:
            raise ValueError(
                f"{source_name} generation_complete differs from content/termination: "
                f"index={index}"
            )
        if correct is not bool(answer_correct and generation_complete):
            raise ValueError(
                f"{source_name} correct must equal answer_correct && "
                f"generation_complete: index={index}"
            )
        record_id_value = record.get("id")
        record_id = (
            str(record_id_value).strip()
            if record_id_value is not None
            else ""
        )
        if not record_id:
            raise ValueError(
                f"{source_name} lacks a non-empty global id: index={index}"
            )
        rollout_idx = record.get("rollout_idx")
        if (
            isinstance(rollout_idx, bool)
            or not isinstance(rollout_idx, int)
            or rollout_idx < 0
        ):
            raise ValueError(
                f"{source_name} rollout_idx must be nonnegative: "
                f"index={index} id={record_id!r} value={rollout_idx!r}"
            )
        rollout_key = (record_id, int(rollout_idx))
        if rollout_key in seen_rollouts:
            raise ValueError(
                f"{source_name} duplicates (id, rollout_idx): {rollout_key!r}"
            )
        seen_rollouts.add(rollout_key)

        prompt_key = rendered_prompt_key_fn(record)
        if not isinstance(prompt_key, str) or not prompt_key:
            raise ValueError(
                f"{source_name} has an invalid rendered prompt key: index={index}"
            )
        previous_prompt = id_to_prompt.setdefault(record_id, prompt_key)
        if previous_prompt != prompt_key:
            raise ValueError(
                f"{source_name} maps one global id to multiple rendered prompts: "
                f"id={record_id!r}"
            )
        id_to_rows.setdefault(record_id, []).append(record)
        prompt_keys.append(prompt_key)

    assert global_n_sampled is not None
    for record_id, grouped_rows in id_to_rows.items():
        first = grouped_rows[0]
        expected_meta = (
            int(first["n_sampled"]),
            int(first["n_correct"]),
            str(first["bucket"]),
            int(first["generation_seed"]),
            str(first.get("answer") or "").strip(),
            str(first.get("type") or "").strip().lower(),
        )
        for row in grouped_rows[1:]:
            actual_meta = (
                int(row["n_sampled"]),
                int(row["n_correct"]),
                str(row["bucket"]),
                int(row["generation_seed"]),
                str(row.get("answer") or "").strip(),
                str(row.get("type") or "").strip().lower(),
            )
            if actual_meta != expected_meta:
                raise ValueError(
                    f"{source_name} prompt/generation metadata differs for one id: "
                    f"id={record_id!r}"
                )
        rollout_indices = sorted(int(row["rollout_idx"]) for row in grouped_rows)
        expected_indices = list(range(int(first["n_sampled"])))
        if rollout_indices != expected_indices:
            raise ValueError(
                f"{source_name} must cover rollout_idx=0..n_sampled-1 per prompt: "
                f"id={record_id!r} expected={expected_indices} "
                f"actual={rollout_indices}"
            )
        actual_correct = sum(row["correct"] is True for row in grouped_rows)
        if actual_correct != int(first["n_correct"]):
            raise ValueError(
                f"{source_name} actual correct count differs from n_correct: "
                f"id={record_id!r} actual={actual_correct} "
                f"declared={first['n_correct']}"
            )
    return tuple(prompt_keys)


def _validate_base_paired_rollout_identities(
    records: Sequence[Mapping[str, Any]],
    *,
    rendered_prompt_key_fn: Callable[[Mapping[str, Any]], str],
    source_name: str,
    show_progress: bool = False,
) -> tuple[str, ...]:
    """Validate only facts present in a base paired Stage 0 artifact.

    This path deliberately does not infer or add generation provenance,
    finish reasons, token counts, or executor hashes.  It proves paired-row
    lineage and internal bucket consistency only.
    """

    seen_rollouts: set[tuple[str, int]] = set()
    id_to_prompt: dict[str, str] = {}
    id_to_rows: dict[str, list[Mapping[str, Any]]] = {}
    prompt_keys: list[str] = []
    record_iter = _progress(
        records,
        enabled=show_progress,
        desc=f"{source_name}: legacy rollout identity",
    )
    for index, record in enumerate(record_iter):
        correct = record.get("correct")
        if not isinstance(correct, bool):
            raise ValueError(
                f"{source_name} legacy correct must be a JSON Boolean: index={index}"
            )
        if not str(record.get("answer") or "").strip():
            raise ValueError(f"{source_name} legacy gold answer is empty: index={index}")
        if correct:
            cot = str(record.get("cot") or "").strip()
            self_answer = str(record.get("self_answer") or "").strip()
            if not cot or not self_answer or cot == self_answer:
                raise ValueError(
                    f"{source_name} legacy correct row requires distinct non-empty "
                    f"cot/self_answer paired fields: index={index}"
                )

        record_id_value = record.get("id")
        record_id = (
            str(record_id_value).strip() if record_id_value is not None else ""
        )
        if not record_id:
            raise ValueError(f"{source_name} lacks a non-empty global id: index={index}")
        rollout_idx = record.get("rollout_idx")
        if (
            isinstance(rollout_idx, bool)
            or not isinstance(rollout_idx, int)
            or rollout_idx < 0
        ):
            raise ValueError(
                f"{source_name} rollout_idx must be nonnegative: "
                f"index={index} id={record_id!r} value={rollout_idx!r}"
            )
        rollout_key = (record_id, int(rollout_idx))
        if rollout_key in seen_rollouts:
            raise ValueError(
                f"{source_name} duplicates (id, rollout_idx): {rollout_key!r}"
            )
        seen_rollouts.add(rollout_key)

        prompt_key = rendered_prompt_key_fn(record)
        if not isinstance(prompt_key, str) or not prompt_key:
            raise ValueError(
                f"{source_name} has an invalid rendered prompt key: index={index}"
            )
        previous_prompt = id_to_prompt.setdefault(record_id, prompt_key)
        if previous_prompt != prompt_key:
            raise ValueError(
                f"{source_name} maps one global id to multiple rendered prompts: "
                f"id={record_id!r}"
            )
        id_to_rows.setdefault(record_id, []).append(record)
        prompt_keys.append(prompt_key)

    rollout_count: int | None = None
    for record_id, grouped_rows in id_to_rows.items():
        group_size = len(grouped_rows)
        if rollout_count is None:
            rollout_count = group_size
        elif group_size != rollout_count:
            raise ValueError(
                f"{source_name} legacy rollout count must match across prompts: "
                f"expected={rollout_count} id={record_id!r} actual={group_size}"
            )
        rollout_indices = sorted(int(row["rollout_idx"]) for row in grouped_rows)
        expected_indices = list(range(group_size))
        if rollout_indices != expected_indices:
            raise ValueError(
                f"{source_name} legacy prompts must cover rollout_idx=0..N-1: "
                f"id={record_id!r} expected={expected_indices} "
                f"actual={rollout_indices}"
            )
        first = grouped_rows[0]
        answer = str(first.get("answer") or "").strip()
        task_type = str(first.get("type") or "").strip().lower()
        actual_correct = sum(row["correct"] is True for row in grouped_rows)
        expected_bucket = (
            "none_correct"
            if actual_correct == 0
            else "all_correct"
            if actual_correct == group_size
            else "mixed"
        )
        for row in grouped_rows:
            if (
                str(row.get("answer") or "").strip() != answer
                or str(row.get("type") or "").strip().lower() != task_type
            ):
                raise ValueError(
                    f"{source_name} legacy gold/task metadata differs for one id: "
                    f"id={record_id!r}"
                )
            if row.get("bucket") != expected_bucket:
                raise ValueError(
                    f"{source_name} legacy bucket differs from actual correctness: "
                    f"id={record_id!r} expected={expected_bucket!r} "
                    f"actual={row.get('bucket')!r}"
                )
            optional_n_sampled = row.get("n_sampled")
            if optional_n_sampled is not None and (
                isinstance(optional_n_sampled, bool)
                or not isinstance(optional_n_sampled, int)
                or optional_n_sampled != group_size
            ):
                raise ValueError(
                    f"{source_name} legacy n_sampled differs from actual rollouts: "
                    f"id={record_id!r} value={optional_n_sampled!r}"
                )
            optional_n_correct = row.get("n_correct")
            if optional_n_correct is not None and (
                isinstance(optional_n_correct, bool)
                or not isinstance(optional_n_correct, int)
                or optional_n_correct != actual_correct
            ):
                raise ValueError(
                    f"{source_name} legacy n_correct differs from actual correctness: "
                    f"id={record_id!r} value={optional_n_correct!r}"
                )

    warnings.warn(
        f"{source_name}: legacy-unverified Stage0 base paired contract accepted; "
        "generation finish_reason/token counts/executor artifact provenance "
        "are unavailable and were not fabricated",
        RuntimeWarning,
        stacklevel=2,
    )
    return tuple(prompt_keys)


def validate_stage0_rollout_identities(
    records: Sequence[Mapping[str, Any]],
    *,
    rendered_prompt_key_fn: Callable[[Mapping[str, Any]], str],
    source_name: str = "Stage0 source",
    show_progress: bool = False,
) -> tuple[str, ...]:
    """Validate complete or base-paired Stage 0 lineage.

    The returned keys correspond one-to-one with ``records`` and are derived
    only from rendered prompts.  Legacy acceptance is explicit and warned;
    unavailable provenance is never synthesized.
    """

    if not records:
        raise ValueError(f"{source_name} must not be empty")

    schemas = [record.get("schema_version") for record in records]
    legacy_versions = [
        record.get("stage0_generation_contract_version") for record in records
    ]
    current_complete = all(
        value == STAGE0_SCHEMA_VERSION and not isinstance(value, bool)
        for value in schemas
    ) and all(value is None for value in legacy_versions)
    legacy_complete = all(value is None for value in schemas) and all(
        value == _LEGACY_GENERATION_CONTRACT_VERSION
        and not isinstance(value, bool)
        for value in legacy_versions
    )
    if current_complete or legacy_complete:
        return _validate_complete_rollout_identities(
            records,
            rendered_prompt_key_fn=rendered_prompt_key_fn,
            source_name=source_name,
            show_progress=show_progress,
        )
    if all(value is None for value in schemas) and all(
        value is None for value in legacy_versions
    ):
        partial_fields = sorted({
            name
            for record in records
            for name in _COMPLETE_PROVENANCE_FIELDS
            if name in record
        })
        if partial_fields:
            raise ValueError(
                f"{source_name} mixes or partially supplies enriched Stage 0 metadata; "
                f"legacy interpretation is rejected: fields={partial_fields}"
            )
        return _validate_base_paired_rollout_identities(
            records,
            rendered_prompt_key_fn=rendered_prompt_key_fn,
            source_name=source_name,
            show_progress=show_progress,
        )
    raise ValueError(
        f"{source_name} mixes or uses unsupported Stage 0 schemas"
    )


__all__ = [
    "STAGE0_SCHEMA_VERSION",
    "validate_stage0_rollout_identities",
]
