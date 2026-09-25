"""Build/reuse model-level ThinkBridge targets without training-run coupling."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
from pathlib import Path
import shutil
import time
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

from think_bridge.data.behavior_manifest import (
    gold_answer_key,
    prompt_token_ids_key,
)
from think_bridge.data.templates import (
    MATH_NOTHINK_INSTRUCTION,
    MATH_THINK_INSTRUCTION,
    THINK_BOUNDARY_TEXT,
    THINK_OPEN_TEXT,
    build_prompt,
    build_thinking_prompt,
)
from think_bridge.model.checkpoint_policy import validate_runtime_sidecars
from think_bridge.model.artifact_schema import (
    STAGE1_SHARED_MANIFEST,
    STAGE1_TARGET_INDEX,
    artifact_header,
)
from think_bridge.model.training_config import TrainingConfig
from think_bridge.model.contract import (
    ANSWER_CAPACITY,
    COT_CONTENT_CAPACITY,
    D_ANSWER_EOS_RULE,
    D_ANSWER_PROVENANCE_SCHEMA,
    NEED_Z_COHORT_DEFINITION,
    TARGET_FIELDS,
    build_matched_hard_donor_manifest,
    canonical_json_sha256,
    require_manifest_fields,
    resolve_boundary_token_ids,
    resolve_bridge_semantic_group,
    validate_bridge_behavior_count,
    validate_matched_hard_donor_manifest,
    write_atomic_json,
    write_atomic_jsonl,
)
from think_bridge.training.manifest_identity import (
    resolve_target_index_artifact_path,
    validate_target_index_payload,
)
from think_bridge.training.input_identity import (
    file_reference,
    input_contract,
    verify_stage0_executor_proofs,
    load_bound_behavior,
    load_bound_direct,
    BEHAVIOR_BINDING,
)
from think_bridge.training.progress import iter_progress
from think_bridge.training.runtime_sidecars import (
    publish_runtime_sidecars,
)


@dataclass(frozen=True)
class _TargetCacheProbe:
    index: dict[str, Any] | None
    counts: dict[str, int] | None
    reason: str
    detail: str | None = None

    @property
    def reusable(self) -> bool:
        return self.index is not None and self.counts is not None


def _stage(action: str, detail: str) -> float:
    print(f"bridge target action={action} {detail}", flush=True)
    return time.monotonic()


def _done(action: str, started: float, detail: str) -> None:
    print(
        f"bridge target action={action} status=done "
        f"elapsed={time.monotonic() - started:.1f}s {detail}",
        flush=True,
    )


def _ensure_runtime_sidecars(
    config: TrainingConfig,
    arguments: Any,
    *,
    expected_tokenizer_sha256: str,
    expected_template_sha256: str,
    expected_boundary_ids_sha256: str,
    tokenizer: Any | None = None,
    executor: Any | None = None,
    run_dir: Path | None = None,
) -> None:
    """Publish or validate this run's tokenizer/lexical deployment sidecars."""

    destination = Path(arguments.run_dir if run_dir is None else run_dir).resolve(
        strict=True
    )
    sidecar_index = destination / "runtime_sidecars.json"
    expected = {
        "tokenizer_sha256": expected_tokenizer_sha256,
        "template_sha256": expected_template_sha256,
        "boundary_ids_sha256": expected_boundary_ids_sha256,
        "attn_implementation": config.attn_implementation,
        "boundary_text": config.boundary_text,
    }
    if sidecar_index.is_file():
        sidecars = validate_runtime_sidecars(destination)
        identity = sidecars["identity"]
        mismatched = [
            name for name, value in expected.items() if identity.get(name) != value
        ]
        if mismatched:
            raise ValueError(
                "runtime sidecar provenance differs from prepared targets: "
                + ", ".join(sorted(mismatched))
            )
        from think_bridge.model.executor_identity import assert_model_source_identity

        if "frozen_executor_identity" not in identity:
            raise ValueError(
                "Runtime sidecar lacks frozen F identity; prepare a new R run"
            )
        assert_model_source_identity(
            config.model_name_or_path,
            identity["frozen_executor_identity"],
            local_files_only=bool(arguments.local_files_only),
            no_progress=bool(arguments.no_progress),
        )
        return
    if tokenizer is None or executor is None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required to publish Bridge runtime sidecars"
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name_or_path,
            local_files_only=bool(arguments.local_files_only),
        )
        executor = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            local_files_only=bool(arguments.local_files_only),
            torch_dtype="auto",
            attn_implementation=config.attn_implementation,
        )
    identity = publish_runtime_sidecars(
        destination,
        tokenizer=tokenizer,
        executor=executor,
        attn_implementation=config.attn_implementation,
        boundary_text=config.boundary_text,
        no_progress=bool(arguments.no_progress),
    )
    mismatched = [
        name for name, value in expected.items() if identity.get(name) != value
    ]
    if mismatched:
        raise ValueError(
            "published runtime sidecar provenance differs from prepared targets: "
            + ", ".join(sorted(mismatched))
        )


def _iter_progress(
    rows: Sequence[Mapping[str, Any]], *, split: str, disabled: bool
) -> Iterable[tuple[int, Mapping[str, Any]]]:
    for index, row in enumerate(
        iter_progress(
            rows,
            total=len(rows),
            desc=f"compile-{split}",
            unit="record",
            disabled=disabled,
        )
    ):
        yield index, row


def _row_cot(row: Mapping[str, Any]) -> str:
    cot = str(row.get("cot") or "").strip()
    if cot:
        return cot
    steps = row.get("steps")
    if isinstance(steps, list) and steps and all(str(step).strip() for step in steps):
        return "\n".join(str(step).strip() for step in steps)
    raise ValueError("source record lacks an immutable CoT/steps target")


def _token_ids(tokenizer: Any, text: str, *, append_eos: bool) -> list[int]:
    ids = [int(token) for token in tokenizer.encode(text, add_special_tokens=False)]
    if not ids:
        raise ValueError("target tokenization produced an empty sequence")
    if append_eos:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has no EOS id for the Route2 target")
        eos = int(tokenizer.eos_token_id)
        if eos in ids:
            raise ValueError(
                "source target contains EOS before its forced terminal position"
            )
        ids.append(eos)
    return ids


def _training_pair_valid(row: Mapping[str, Any]) -> bool:
    return (
        row.get("correct") is True
        and str(row.get("cot") or "").strip() != ""
        and str(row.get("self_answer") or "").strip() != ""
        and (
            int(row.get("schema_version", -1)) == 1
            or int(row.get("stage0_generation_contract_version", -1)) == 2
        )
        and str(row.get("finish_reason") or "").lower() == "stop"
        and row.get("think_closed") is True
        and row.get("generation_complete") is True
    )


def _select_direct_answer_trajectories(
    labels: Mapping[str, Mapping[str, Any]],
    direct_records: Sequence[Mapping[str, Any]],
    *,
    eos_token_id: int,
    answer_capacity: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Select one complete successful direct-answer fallback per D prompt."""

    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
        raise ValueError("D-answer EOS token id is invalid")
    if isinstance(answer_capacity, bool) or int(answer_capacity) <= 0:
        raise ValueError("D-answer capacity is invalid")
    by_prompt: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(direct_records):
        prompt_key = str(record.get("prompt_group_key") or "")
        if not prompt_key or prompt_key in by_prompt:
            raise ValueError(
                f"direct raw prompt identity is empty or duplicated: index={index}"
            )
        by_prompt[prompt_key] = record
    selected: dict[str, dict[str, Any]] = {}
    candidate_count = 0
    excluded_non_eos_count = 0
    excluded_answer_capacity_count = 0
    for prompt_key, label in labels.items():
        n_correct = label.get("n_correct")
        direct_correct = label.get("direct_correct")
        if (
            isinstance(n_correct, bool)
            or not isinstance(n_correct, int)
            or n_correct < 0
            or not isinstance(direct_correct, bool)
        ):
            raise ValueError("sealed behavior lacks D-answer fallback labels")
        if not direct_correct or n_correct != 0:
            continue
        candidate_count += 1
        record = by_prompt.get(str(prompt_key))
        if record is None:
            raise ValueError("direct raw lacks a D-population prompt identity")
        if record.get("direct_correct") is not True:
            raise ValueError(
                "behavior/direct-raw correctness labels differ for D prompt"
            )
        if record.get("direct_hit_eos") is not True:
            excluded_non_eos_count += 1
            continue
        raw_tokens = record.get("direct_token_ids")
        if not isinstance(raw_tokens, list) or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in raw_tokens
        ):
            raise ValueError("direct raw D trajectory token ids are invalid")
        # The sealed greedy decoder contract stores answer tokens without EOS.
        # `direct_hit_eos=True` proves that the immediately following sampled
        # token was the tokenizer's terminal EOS; no incomplete row is repaired.
        answer_ids = [int(token) for token in raw_tokens] + [int(eos_token_id)]
        if len(answer_ids) > int(answer_capacity):
            excluded_answer_capacity_count += 1
            continue
        selected[str(prompt_key)] = {
            "record": dict(record),
            "answer_ids": answer_ids,
        }
    return selected, {
        "candidate_count": candidate_count,
        "eligible_count": len(selected),
        "excluded_non_eos_count": excluded_non_eos_count,
        "excluded_answer_capacity_count": excluded_answer_capacity_count,
    }


def _compile_split(
    *,
    tokenizer: Any,
    source: Path,
    split: str,
    route1_population: str,
    answer_only_target_source: str,
    no_progress: bool,
    route2_content_capacity: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    from think_bridge.data.dataset import load_data_file

    raw_rows = load_data_file(source)
    compiled: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    formal_candidate_count = 0
    excluded_cot_count = 0
    excluded_answer_count = 0
    for source_index, raw in _iter_progress(
        raw_rows, split=split, disabled=no_progress
    ):
        answer_only = route1_population == "answer-only"
        gold_reference = split == "train" and route1_population == "gold-reference"
        if (
            split == "train"
            and not answer_only
            and not gold_reference
            and not _training_pair_valid(raw)
        ):
            continue
        question = str(raw.get("question") or "").strip()
        answer = str(raw.get("answer") or "").strip()
        if not question or not answer:
            if split == "train":
                continue
            raise ValueError(f"{split} source row {source_index} lacks question/answer")
        cot_text = _row_cot(raw) if split == "train" else None
        prompt_text = build_thinking_prompt(tokenizer, question, task_type="math")
        prompt_ids = _token_ids(tokenizer, prompt_text, append_eos=False)
        direct_prompt_text = (
            build_prompt(tokenizer, question, task_type="math", think=False)
            + THINK_OPEN_TEXT
            + THINK_BOUNDARY_TEXT
        )
        direct_prompt_ids = _token_ids(tokenizer, direct_prompt_text, append_eos=False)
        cot_ids = (
            _token_ids(tokenizer, cot_text, append_eos=False)
            if cot_text is not None
            else []
        )
        readout_cot_ids = (
            _token_ids(tokenizer, cot_text, append_eos=True)
            if cot_text is not None
            else []
        )
        native_self_answer = str(raw.get("self_answer") or "").strip()
        if split == "train":
            if gold_reference:
                if not cot_text:
                    raise ValueError(f"gold reference row {source_index} lacks cot")
                answer_text = f"\\boxed{{{answer}}}"
                answer_source = "dataset-reference-cot-and-answer"
            elif answer_only and answer_only_target_source == "boxed":
                answer_text = f"\\boxed{{{answer}}}"
                answer_source = "explicit-answer-only-boxed-reference"
            else:
                if not native_self_answer:
                    continue
                answer_text = native_self_answer
                answer_source = (
                    "explicit-answer-only-selfgen"
                    if answer_only
                    else "native-self-answer"
                )
        else:
            answer_text = native_self_answer or f"\\boxed{{{answer}}}"
            answer_source = (
                "native-self-answer"
                if native_self_answer
                else "heldout-reference-diagnostic"
            )
        answer_ids = _token_ids(tokenizer, answer_text, append_eos=True)
        if split == "train":
            formal_candidate_count += 1
            cot_over = len(cot_ids) > COT_CONTENT_CAPACITY
            answer_over = len(answer_ids) > ANSWER_CAPACITY
            excluded_cot_count += int(cot_over)
            excluded_answer_count += int(answer_over)
            if answer_over:
                continue
        prompt_hash = canonical_json_sha256(prompt_ids)
        raw_id = str(raw.get("id") or canonical_json_sha256({"question": question}))
        rollout = raw.get("rollout_idx", 0)
        if isinstance(rollout, bool) or not isinstance(rollout, int) or rollout < 0:
            raise ValueError(
                f"{split} source row {source_index} has invalid rollout_idx"
            )
        problem_id = str(raw.get("problem_id") or raw_id)
        semantic_group_id = resolve_bridge_semantic_group(
            raw, split=split, problem_id=problem_id
        )
        source_rollout_id = (
            f"reference:{raw_id}" if gold_reference else f"{raw_id}:{rollout}"
        )
        record_id = f"{split}:{source_index}:{source_rollout_id}"
        cot_id = f"{prompt_hash}:{source_rollout_id}"
        row = {
            "record_id": record_id,
            "prompt_group_id": prompt_hash,
            "problem_id": problem_id,
            "semantic_group_id": semantic_group_id,
            "self_cot_id": cot_id,
            "self_cot_source_rollout_id": source_rollout_id,
            "provenance_valid": True,
            "prompt_ids": prompt_ids,
            "direct_prompt_ids": direct_prompt_ids,
            "cot_ids": cot_ids,
            "readout_cot_ids": readout_cot_ids,
            "route2_eligible": bool(readout_cot_ids)
            and len(readout_cot_ids) <= route2_content_capacity + 1,
            "answer_ids": answer_ids,
            "answer_source": answer_source,
            "split_id": split,
            "population": "reference" if gold_reference else "native",
            "n_correct": None,
            "reference_answer": answer,
            "task_type": "math",
            "need_z": None,
            "direct_correct": None,
            "locked_full_correct": None,
            "need_z_definition": NEED_Z_COHORT_DEFINITION,
            "_question": question,
        }
        compiled.append(row)
        provenance.append(
            {
                "record_id": record_id,
                "population": "reference" if gold_reference else "native",
                "source_rollout_id": source_rollout_id,
                "answer_source": answer_source,
            }
        )
    if not compiled:
        raise ValueError(f"{split} source produced no provenance-valid targets")
    return (
        compiled,
        provenance,
        {
            "formal_candidate_count": formal_candidate_count,
            "retained_count": len(compiled),
            "excluded_cot_count": excluded_cot_count,
            "excluded_answer_count": excluded_answer_count,
        },
    )


def _apply_sealed_behavior(
    rows: Sequence[dict[str, Any]],
    *,
    path: Path,
    source: Path,
    split: str,
    tokenizer: Any,
    route2_content_capacity: int = COT_CONTENT_CAPACITY,
) -> None:
    """Attach behavior labels by prompt key after structural validation."""

    if split not in {"train", "validation"}:
        raise ValueError(f"unsupported Bridge behavior split: {split}")
    labels, _metadata = load_bound_behavior(
        path, source, split=split, tokenizer=tokenizer
    )
    observed_prompt_keys: set[str] = set()
    for index, row in enumerate(rows):
        prompt_key = prompt_token_ids_key(row["prompt_ids"])
        label = labels.get(prompt_key)
        if label is None:
            raise ValueError(f"{split} behavior lacks prompt identity at row {index}")
        # Match HybridCollator(mode="direct") exactly.  The behavior-manifest
        # behavior manifest evaluates a no-think prompt followed by an empty
        # thinking block, not the bare chat generation prefix.
        direct_prompt = (
            build_prompt(
                tokenizer, str(row["_question"]), task_type="math", think=False
            )
            + THINK_OPEN_TEXT
            + THINK_BOUNDARY_TEXT
        )
        direct_ids = _token_ids(tokenizer, direct_prompt, append_eos=False)
        observed_binding = {
            "prompt_group_key": prompt_key,
            "prompt_token_count": len(row["prompt_ids"]),
            "direct_prompt_key": prompt_token_ids_key(direct_ids),
            "direct_prompt_token_count": len(direct_ids),
            "gold_answer_key": gold_answer_key(str(row["reference_answer"]), "math"),
        }
        mismatched_fields = [
            name
            for name, observed in observed_binding.items()
            if label[name] != observed
        ]
        if mismatched_fields:
            detail = ", ".join(
                f"{name}:manifest={label[name]!r}:prepared={observed_binding[name]!r}"
                for name in mismatched_fields
            )
            raise ValueError(
                f"{split} behavior prompt/direct/answer binding differs at row "
                f"{index}: {detail}"
            )
        direct = label.get("direct_correct")
        full = label.get("native_correct")
        need_z_label = label.get("need_z")
        n_correct = label.get("n_correct")
        if not all(isinstance(value, bool) for value in (direct, full, need_z_label)):
            raise ValueError(f"{split} behavior row {index} lacks exact boolean labels")
        try:
            validate_bridge_behavior_count(
                n_correct,
                split=split,
                locked_full_correct=full,
            )
        except ValueError as exc:
            raise ValueError(
                f"{split} behavior row {index} has invalid n_correct"
            ) from exc
        if bool(need_z_label) != (bool(full) and not bool(direct)):
            raise ValueError(
                f"{split} behavior need-z arithmetic differs at row {index}"
            )
        if split == "validation":
            from think_bridge.data.native_observation import observed_native_cot

            ids = observed_native_cot(
                label,
                eos_token_id=int(tokenizer.eos_token_id),
                close_token_ids=tokenizer.encode("</think>", add_special_tokens=False),
            )
            row["cot_length_known"] = ids is not None
            row["cot_ids"] = [] if ids is None else ids
            row["readout_cot_ids"] = (
                [] if ids is None else [*ids, int(tokenizer.eos_token_id)]
            )
            row["route2_eligible"] = (
                ids is not None and len(ids) <= route2_content_capacity
            )
            row["self_cot_source_rollout_id"] = f"heldout-native:{prompt_key}"
            row["self_cot_id"] = f"{prompt_key}:heldout-native"
        row["direct_correct"] = bool(direct)
        row["locked_full_correct"] = bool(full)
        row["need_z"] = bool(full) and not bool(direct)
        row["n_correct"] = int(n_correct)
        row["direct_prompt_ids"] = direct_ids
        row["need_z_definition"] = NEED_Z_COHORT_DEFINITION
        observed_prompt_keys.add(prompt_key)
    if not observed_prompt_keys or not observed_prompt_keys.issubset(set(labels)):
        raise ValueError(f"{split} behavior does not cover every retained prompt")
    return None


def _compile_d_answer_fallback_split(
    *,
    tokenizer: Any,
    source: Path,
    behavior_path: Path,
    direct_raw_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Compile prompt-uniform D rows from structurally valid paired artifacts."""

    from think_bridge.data.dataset import load_data_file

    raw_rows = load_data_file(source)
    labels, metadata = load_bound_behavior(
        behavior_path, source, split="train", tokenizer=tokenizer
    )
    direct_records = load_bound_direct(
        direct_raw_path, source, tokenizer=tokenizer, behavior_metadata=metadata
    )
    if {row["prompt_group_key"] for row in direct_records} != set(labels):
        raise ValueError(
            "Direct outputs and train behavior cover different prompt populations"
        )
    if tokenizer.eos_token_id is None:
        raise ValueError("sealed D-answer fallback requires tokenizer EOS")
    selected, statistics = _select_direct_answer_trajectories(
        labels,
        direct_records,
        eos_token_id=int(tokenizer.eos_token_id),
        answer_capacity=ANSWER_CAPACITY,
    )
    source_groups: dict[str, dict[str, Any]] = {}
    for source_index, raw in enumerate(raw_rows):
        question = str(raw.get("question") or "").strip()
        answer = str(raw.get("answer") or "").strip()
        if not question or not answer:
            continue
        prompt_text = build_thinking_prompt(tokenizer, question, task_type="math")
        prompt_ids = _token_ids(tokenizer, prompt_text, append_eos=False)
        prompt_key = prompt_token_ids_key(prompt_ids)
        raw_id = str(raw.get("id") or canonical_json_sha256({"question": question}))
        problem_id = str(raw.get("problem_id") or raw_id)
        semantic_group_id = resolve_bridge_semantic_group(
            raw, split="train", problem_id=problem_id
        )
        signature = {
            "question": question,
            "answer": answer,
            "problem_id": problem_id,
            "semantic_group_id": semantic_group_id,
            "prompt_ids": prompt_ids,
        }
        group = source_groups.setdefault(prompt_key, {"signature": signature})
        if group["signature"] != signature:
            raise ValueError("Stage0 prompt group has inconsistent source identity")
    direct_by_prompt = {
        str(record["prompt_group_key"]): record for record in direct_records
    }
    compiled: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for selected_index, prompt_key in enumerate(sorted(selected)):
        group = source_groups.get(prompt_key)
        if group is None:
            raise ValueError("D prompt lacks exact immutable Stage0 source identity")
        signature = group["signature"]
        direct = direct_by_prompt[prompt_key]
        direct_prompt_text = (
            build_prompt(
                tokenizer, signature["question"], task_type="math", think=False
            )
            + THINK_OPEN_TEXT
            + THINK_BOUNDARY_TEXT
        )
        direct_prompt_ids = _token_ids(tokenizer, direct_prompt_text, append_eos=False)
        expected_binding = {
            "question": signature["question"],
            "gold_answer": signature["answer"],
            "task_type": "math",
            "prompt_group_key": prompt_key,
            "direct_prompt_key": prompt_token_ids_key(direct_prompt_ids),
        }
        conflicts = [
            name
            for name, value in expected_binding.items()
            if direct.get(name) != value
        ]
        if conflicts:
            raise ValueError(
                f"direct raw D prompt/source identity differs: fields={conflicts}"
            )
        answer_ids = list(selected[prompt_key]["answer_ids"])
        record_id = f"train:d-answer:{selected_index}:{prompt_key}"
        row = {
            "record_id": record_id,
            "prompt_group_id": prompt_key,
            "problem_id": signature["problem_id"],
            "semantic_group_id": signature["semantic_group_id"],
            "self_cot_id": None,
            "self_cot_source_rollout_id": None,
            "provenance_valid": True,
            "prompt_ids": list(signature["prompt_ids"]),
            "direct_prompt_ids": direct_prompt_ids,
            "cot_ids": [],
            "readout_cot_ids": [],
            "route2_eligible": False,
            "answer_ids": answer_ids,
            "answer_source": "sealed-direct-raw-successful-no-think-trajectory",
            "split_id": "train",
            "population": "d_answer_fallback",
            "n_correct": 0,
            "reference_answer": signature["answer"],
            "task_type": "math",
            "need_z": False,
            "direct_correct": True,
            "locked_full_correct": False,
            "need_z_definition": NEED_Z_COHORT_DEFINITION,
            "_question": signature["question"],
        }
        compiled.append(row)
        provenance.append(
            {
                "record_id": record_id,
                "population": "d_answer_fallback",
                "source_rollout_id": None,
                "answer_source": row["answer_source"],
            }
        )
    return compiled, provenance, statistics


def _behavior_binding_error(
    *,
    split: str,
    source: Path,
    behavior: Path,
    cause: BaseException,
) -> ValueError:
    return ValueError(
        f"{split} source/behavior structural mismatch; source={source} "
        f"behavior={behavior}. Generate a complete Stage0 behavior manifest "
        "with matching prompt/direct/answer fields; Bridge will not synthesize "
        f"missing labels. detail={cause}"
    )


def _compile_target_publication_inputs(
    *,
    config: TrainingConfig,
    tokenizer: Any,
    arguments: Any,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, int]],
]:
    """Compile and behavior-bind targets without requiring executor weights."""

    split_paths = {
        "train": Path(arguments.train_source),
        "validation": Path(arguments.validation_source),
    }
    compiled: dict[str, list[dict[str, Any]]] = {}
    provenance: dict[str, list[dict[str, Any]]] = {}
    statistics: dict[str, dict[str, int]] = {}
    for split, source in split_paths.items():
        compiled[split], provenance[split], statistics[split] = _compile_split(
            tokenizer=tokenizer,
            source=source,
            split=split,
            route1_population=str(arguments.route1_population),
            answer_only_target_source=str(arguments.answer_only_target_source),
            no_progress=bool(arguments.no_progress),
            route2_content_capacity=int(config.cot_content_capacity),
        )
    for split in ("train", "validation"):
        if split == "train" and arguments.train_behavior is None:
            if str(arguments.route1_population) not in {
                "answer-only",
                "gold-reference",
            }:
                raise ValueError("staged Route1 requires a train behavior manifest")
            for row in compiled[split]:
                # Answer-only is a separate course-only population.  Its
                # missing Stage0 behavior labels remain explicitly unknown;
                # execution must not manufacture a B/C/D quadrant from them.
                row["direct_correct"] = None
                row["locked_full_correct"] = None
                row["need_z"] = None
                row["n_correct"] = None
                row["need_z_definition"] = NEED_Z_COHORT_DEFINITION
            continue
        behavior_path = Path(getattr(arguments, f"{split}_behavior"))
        try:
            _apply_sealed_behavior(
                compiled[split],
                path=behavior_path,
                source=split_paths[split],
                split=split,
                tokenizer=tokenizer,
                route2_content_capacity=int(config.cot_content_capacity),
            )
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
            raise _behavior_binding_error(
                split=split,
                source=split_paths[split],
                behavior=behavior_path,
                cause=exc,
            ) from exc
    direct_answer_source = getattr(arguments, "train_direct_answer_source", None)
    if str(arguments.route1_population) == "staged":
        if direct_answer_source is None:
            raise ValueError("staged Route1 requires --train-direct-answer-source")
        direct_rows, direct_provenance, direct_statistics = (
            _compile_d_answer_fallback_split(
                tokenizer=tokenizer,
                source=split_paths["train"],
                behavior_path=Path(arguments.train_behavior),
                direct_raw_path=Path(direct_answer_source),
            )
        )
        compiled["train"].extend(direct_rows)
        provenance["train"].extend(direct_provenance)
    else:
        if direct_answer_source is not None:
            raise ValueError("answer-only population must omit D-answer source")
        direct_statistics = {
            "candidate_count": 0,
            "eligible_count": 0,
            "excluded_non_eos_count": 0,
            "excluded_answer_capacity_count": 0,
        }
    statistics["train"].update(
        {
            "c_sample_count": sum(
                int(row["population"] == "native" and bool(row["need_z"]))
                for row in compiled["train"]
            ),
            "native_occurrence_count": sum(
                int(row["population"] == "native") for row in compiled["train"]
            ),
            "d_candidate_count": int(direct_statistics["candidate_count"]),
            "d_sample_count": int(direct_statistics["eligible_count"]),
            "d_excluded_non_eos_count": int(
                direct_statistics["excluded_non_eos_count"]
            ),
            "d_excluded_answer_capacity_count": int(
                direct_statistics["excluded_answer_capacity_count"]
            ),
        }
    )
    return compiled, provenance, statistics


def _write_stage1_cache_json(
    path: Path, value: Mapping[str, Any], *, label: str
) -> None:
    """Atomically refresh one reproducible Stage1 JSON cache in place."""

    target = Path(path)
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise ValueError(f"{label} cache path is not a regular file: {target}")
    result = write_atomic_json(target, value, replace_mismatch=True)
    reason = "exact-content" if result == "reuse" else "stage1-cache-refresh"
    print(f"bridge prepare stage={label} action={result} reason={reason} path={target}")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"target JSONL cache path is not a regular file: {path}")
    write_atomic_jsonl(path, rows, replace_mismatch=True)


def _hard_donor_payload(
    split_rows: Mapping[str, Sequence[Mapping[str, Any]]], *, content_capacity: int
) -> dict[str, Any]:
    return build_matched_hard_donor_manifest(
        split_rows, content_capacity=int(content_capacity)
    )


def _manifest_payload(
    config: TrainingConfig,
    *,
    index: Mapping[str, Any],
    compiled: Mapping[str, Sequence[Mapping[str, Any]]],
    tokenizer_sha256: str,
    template_sha256: str,
    boundary_ids: Sequence[int],
    z_width: int,
    world_size: int,
    route1_local_samples: int,
    route1_gradient_accumulation_steps: int,
) -> dict[str, Any]:
    del world_size, route1_local_samples, route1_gradient_accumulation_steps
    manifest: dict[str, Any] = {
        **artifact_header(STAGE1_SHARED_MANIFEST),
        "method": config.method,
        "route": "route1",
        "model_family": config.model_family,
        "tokenizer_sha256": tokenizer_sha256,
        "template_sha256": template_sha256,
        "boundary_ids_sha256": canonical_json_sha256(list(boundary_ids)),
        "z_width": int(z_width),
        "route2_content_capacity": int(config.cot_content_capacity),
        "need_z_cohort_definition": NEED_Z_COHORT_DEFINITION,
        "d_answer_provenance_schema": D_ANSWER_PROVENANCE_SCHEMA,
        "d_answer_eos_rule": D_ANSWER_EOS_RULE,
        "train_c_sample_count": sum(
            int(row["population"] == "native" and bool(row["need_z"]))
            for row in compiled["train"]
        ),
        "train_d_sample_count": sum(
            int(row["population"] == "d_answer_fallback") for row in compiled["train"]
        ),
        "train_d_excluded_non_eos_count": int(
            index["statistics"].get("train_d_excluded_non_eos_count", 0)
        ),
    }
    require_manifest_fields(manifest)
    return manifest


def _read_index(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("target index must be a JSON object")
    return value


def _locator_basename(value: str) -> str:
    return value.rstrip("/").rsplit("/", 1)[-1]


def _normalized_model_locator(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must be non-empty")
    return _locator_basename(text)


def _changed_compile_contract_fields(cached: Any, expected: Any) -> list[str]:
    """Return compact field paths for a compile-contract mismatch."""

    if isinstance(cached, Mapping) and isinstance(expected, Mapping):
        changed: list[str] = []
        for field in sorted(set(cached) | set(expected)):
            if field not in cached or field not in expected:
                changed.append(str(field))
                continue
            nested = _changed_compile_contract_fields(cached[field], expected[field])
            changed.extend(f"{field}.{name}" if name else str(field) for name in nested)
        return changed
    return [""] if cached != expected else []


def _compile_input_contract(arguments: Any) -> dict[str, Any]:
    paths = {
        "train_dataset": arguments.train_source,
        "eval_dataset": arguments.validation_source,
        "train_behavior": arguments.train_behavior,
        "eval_behavior": arguments.validation_behavior,
        "train_direct_answer_source": getattr(
            arguments, "train_direct_answer_source", None
        ),
    }
    roles = {
        "train_behavior": "explicit-answer-only",
        "train_direct_answer_source": "not-applicable-to-answer-only",
    }
    references = {
        name: file_reference(path, name)
        if path is not None
        else {"path": None, "role": roles[name]}
        for name, path in paths.items()
    }
    current = input_contract(references)
    expected = getattr(arguments, "input_references", None)
    if expected is not None and input_contract(expected) != current:
        raise ValueError(
            "Stage0 inputs changed after stage resolution; refusing stale targets"
        )
    return current


def _target_compile_contract(config: TrainingConfig, arguments: Any) -> dict[str, Any]:
    """Return only readable inputs that determine compiled target contents."""

    return {
        "model_family": str(config.model_family),
        "model": _normalized_model_locator(
            config.model_name_or_path, label="model_name_or_path"
        ),
        "tokenizer": _normalized_model_locator(
            config.tokenizer_name_or_path, label="tokenizer_name_or_path"
        ),
        "inputs": _compile_input_contract(arguments),
        "validation_cot_source": "native-observation-with-explicit-unknown",
        "behavior_binding": BEHAVIOR_BINDING,
        "prompt_template": {
            "task_type": "math",
            "renderer": (
                "tokenizer-chat-template-add-generation-prompt-"
                "enable-thinking-compatible-fallback"
            ),
            "think_instruction": MATH_THINK_INSTRUCTION,
            "nothink_instruction": MATH_NOTHINK_INSTRUCTION,
            "think_open_text": THINK_OPEN_TEXT,
            "compiled_think_boundary_text": THINK_BOUNDARY_TEXT,
        },
        "boundary_text": str(config.boundary_text),
        "population": str(arguments.route1_population),
        "answer_target_source": str(arguments.answer_only_target_source),
        "answer_sources": {
            "native": "native-self-answer",
            "answer_only_boxed": "explicit-answer-only-boxed-reference",
            "answer_only_selfgen": "explicit-answer-only-selfgen",
            "gold_reference": "dataset-reference-cot-and-answer",
            "validation_fallback": "heldout-reference-diagnostic",
            "direct_fallback": "sealed-direct-raw-successful-no-think-trajectory",
        },
        "capacities": {
            "cot_content": COT_CONTENT_CAPACITY,
            "answer": ANSWER_CAPACITY,
            "route2_content": int(config.cot_content_capacity),
        },
        "target_fields": list(TARGET_FIELDS),
        "need_z_cohort_definition": NEED_Z_COHORT_DEFINITION,
        "d_answer_provenance_schema": D_ANSWER_PROVENANCE_SCHEMA,
        "d_answer_eos_rule": D_ANSWER_EOS_RULE,
    }


def _read_cached_target_rows(
    index: Mapping[str, Any], *, index_path: Path
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Stream structural validation without retaining compiled token arrays."""

    counts: dict[str, int] = {}
    record_ids: set[str] = set()
    validation_rows: list[dict[str, Any]] = []
    for split in ("train", "validation"):
        path = resolve_target_index_artifact_path(
            str(index["artifacts"][split]["path"]), index_path=index_path
        )
        count = 0
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"cached {split} target has invalid JSON at line {line_number}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(
                        f"cached {split} target row {line_number} is not an object"
                    )
                _validate_completed_target_row(row, split=split, record_ids=record_ids)
                if split == "validation":
                    validation_rows.append(row)
                count += 1
        counts[split] = count
    return counts, validation_rows


@contextmanager
def _target_cache_lock(output_index: Path) -> Iterable[None]:
    """Serialize builders while immutable generations protect target readers."""

    output_index.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_index.parent / ".prepare.lock"
    with lock_path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _index_is_complete(
    index: Mapping[str, Any],
    *,
    index_path: Path | None = None,
    expected_model_family: str | None = None,
    **_unused: Any,
) -> bool:
    if expected_model_family is None:
        expected_model_family = str(index.get("model_family") or "")
    return (
        validate_target_index_payload(
            index,
            index_path=index_path,
            model_family=expected_model_family,
            allow_stale=True,
        )
        is not None
    )


def _validate_completed_target_row(
    row: Mapping[str, Any], *, split: str, record_ids: set[str]
) -> None:
    if not set(TARGET_FIELDS).issubset(row) or row.get("split_id") != split:
        raise ValueError(f"sealed {split} target row schema differs")
    record_id = str(row["record_id"])
    if not record_id or record_id in record_ids:
        raise ValueError("sealed target record ids are empty or duplicated")
    record_ids.add(record_id)
    if row.get("need_z_definition") != NEED_Z_COHORT_DEFINITION:
        raise ValueError(f"sealed {split} direct cohort binding differs")
    population = row.get("population")
    n_correct = row.get("n_correct")
    if population not in {"native", "d_answer_fallback", "reference"}:
        raise ValueError(f"sealed {split} population is invalid")
    answer_only = row.get("answer_source") in {
        "explicit-answer-only-boxed-reference",
        "explicit-answer-only-selfgen",
        "dataset-reference-cot-and-answer",
    }
    if answer_only:
        if (
            population not in {"native", "reference"}
            or n_correct is not None
            or any(
                row.get(field) is not None
                for field in (
                    "need_z",
                    "direct_correct",
                    "locked_full_correct",
                )
            )
            or row.get("answer_source")
            not in {
                "explicit-answer-only-boxed-reference",
                "explicit-answer-only-selfgen",
                "dataset-reference-cot-and-answer",
            }
        ):
            raise ValueError("sealed answer-only row forged Stage0 behavior labels")
    else:
        if not all(
            isinstance(row.get(field), bool)
            for field in (
                "need_z",
                "direct_correct",
                "locked_full_correct",
            )
        ):
            raise ValueError(f"sealed {split} behavior labels are invalid")
        try:
            validate_bridge_behavior_count(
                n_correct,
                split=split,
                locked_full_correct=row["locked_full_correct"],
            )
        except ValueError as exc:
            raise ValueError(f"sealed {split} n_correct is invalid") from exc
        if bool(row["need_z"]) != (
            bool(row["locked_full_correct"]) and not bool(row["direct_correct"])
        ):
            raise ValueError(f"sealed {split} direct need-z arithmetic differs")
    if population == "d_answer_fallback":
        if (
            split != "train"
            or n_correct != 0
            or row["direct_correct"] is not True
            or row["need_z"] is not False
            or row["cot_ids"] != []
            or row["readout_cot_ids"] != []
            or row["route2_eligible"] is not False
        ):
            raise ValueError("sealed direct-only population contract differs")


def _try_reuse_completed_targets(
    config: TrainingConfig,
    arguments: Any,
    *,
    output_index: Path,
) -> _TargetCacheProbe:
    """Admit one complete cache without loading a tokenizer or executor."""

    if not output_index.is_file():
        return _TargetCacheProbe(None, None, "cache-missing")
    try:
        index = _read_index(output_index)
        validate_target_index_payload(
            index,
            index_path=output_index,
            model_family=config.model_family,
        )
        if "compile_contract" not in index:
            return _TargetCacheProbe(None, None, "unsupported-cache-contract")
        expected_contract = _target_compile_contract(config, arguments)
        cached_contract = index["compile_contract"]
        expected_contract_view = expected_contract
        if cached_contract != expected_contract_view:
            changed = _changed_compile_contract_fields(
                cached_contract, expected_contract_view
            )
            detail = "changed_fields=" + ",".join(changed)
            return _TargetCacheProbe(None, None, "compile-contract-mismatch", detail)
        counts, validation_rows = _read_cached_target_rows(
            index, index_path=output_index
        )
        manifest_raw = json.loads(Path(arguments.manifest).read_text(encoding="utf-8"))
        manifest = require_manifest_fields(manifest_raw)
        if manifest["model_family"] != config.model_family:
            raise ValueError("shared manifest model family mismatch")
        statistics = index["statistics"]
        manifest_statistics = {
            "train_c_sample_count": "train_c_sample_count",
            "train_d_sample_count": "train_d_sample_count",
            "train_d_excluded_non_eos_count": ("train_d_excluded_non_eos_count"),
        }
        mismatched_statistics = [
            manifest_field
            for manifest_field, index_field in manifest_statistics.items()
            if (
                isinstance(statistics.get(index_field), bool)
                or not isinstance(statistics.get(index_field), int)
                or statistics[index_field] < 0
                or manifest[manifest_field] != statistics[index_field]
            )
        ]
        if mismatched_statistics:
            raise ValueError(
                "shared manifest target statistics differ: "
                + ", ".join(sorted(mismatched_statistics))
            )
        donor_payload = json.loads(Path(arguments.donors).read_text(encoding="utf-8"))
        if not isinstance(donor_payload, Mapping):
            raise ValueError("hard-donor cache must be a JSON object")
        validate_matched_hard_donor_manifest(
            donor_payload,
            validation_rows,
            split="validation",
            content_capacity=int(config.cot_content_capacity),
        )
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        return _TargetCacheProbe(None, None, "invalid-cache", str(exc))
    return _TargetCacheProbe(index, counts, "compatible-complete-cache")


def prepare_targets(config: TrainingConfig, arguments: Any) -> int:
    """Reuse or build train/validation targets in the path-derived cache."""

    output_index = Path(arguments.output_index)
    if output_index.parent.name != "targets" or output_index.name != "index.json":
        raise ValueError("target index must be the explicit targets/index.json path")
    if output_index.is_symlink() or (
        output_index.exists() and not output_index.is_file()
    ):
        raise ValueError("target index path must be a regular file")
    with _target_cache_lock(output_index):
        return _prepare_targets_locked(config, arguments, output_index=output_index)


def _prepare_targets_locked(
    config: TrainingConfig, arguments: Any, *, output_index: Path
) -> int:
    initial_contract = _target_compile_contract(config, arguments)
    overwrite = bool(getattr(arguments, "overwrite_cache", False))
    if overwrite:
        probe = _TargetCacheProbe(None, None, "explicit-overwrite")
    else:
        probe = _try_reuse_completed_targets(
            config, arguments, output_index=output_index
        )
    if probe.reusable:
        assert probe.index is not None
        assert probe.counts is not None
        manifest = require_manifest_fields(
            json.loads(Path(arguments.manifest).read_text(encoding="utf-8"))
        )
        _ensure_runtime_sidecars(
            config,
            arguments,
            expected_tokenizer_sha256=str(manifest["tokenizer_sha256"]),
            expected_template_sha256=str(manifest["template_sha256"]),
            expected_boundary_ids_sha256=str(manifest["boundary_ids_sha256"]),
        )
        verify_stage0_executor_proofs(config, arguments)
        started = _stage(
            "reuse",
            f"reason={probe.reason} model={config.model_family} path={output_index}",
        )
        _done(
            "reuse",
            started,
            f"train={probe.counts['train']} "
            f"validation={probe.counts['validation']} "
            f"manifest={arguments.manifest}",
        )
        return 0
    detail = f" detail={probe.detail}" if probe.detail else ""
    print(
        "bridge prepare stage=target-index action=rebuild "
        f"reason={probe.reason}{detail} path={output_index}",
        flush=True,
    )
    started = _stage(
        "rebuild",
        f"reason={probe.reason} model={config.model_family} method={config.method}",
    )
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required to prepare Bridge targets"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name_or_path,
        local_files_only=bool(arguments.local_files_only),
    )
    executor = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        local_files_only=bool(arguments.local_files_only),
        torch_dtype="auto",
        attn_implementation=config.attn_implementation,
    )
    verify_stage0_executor_proofs(config, arguments)
    from think_bridge.training.train import _tokenizer_sha256

    tokenizer_sha256 = _tokenizer_sha256(tokenizer)
    boundary_ids = resolve_boundary_token_ids(tokenizer, config.boundary_text)
    compiled, _provenance, statistics = _compile_target_publication_inputs(
        config=config,
        tokenizer=tokenizer,
        arguments=arguments,
    )
    formal_count = statistics["train"]["formal_candidate_count"]
    if formal_count <= 0:
        raise ValueError("training source has no formal provenance-valid candidates")
    for rows in compiled.values():
        for row in rows:
            row.pop("_question", None)
    compile_contract = _target_compile_contract(config, arguments)
    if compile_contract != initial_contract:
        raise ValueError(
            "Stage0 inputs changed during target compilation; no targets published"
        )
    generation_root = output_index.parent / "generations"
    generation_root.mkdir(parents=True, exist_ok=True)
    generation_name = uuid4().hex
    building = generation_root / f".{generation_name}.building"
    generation = generation_root / generation_name
    building.mkdir()
    artifacts: dict[str, dict[str, Any]] = {}
    template_sha256 = canonical_json_sha256(tokenizer.chat_template or "")
    boundary_ids_sha256 = canonical_json_sha256(boundary_ids)
    try:
        for split, rows in compiled.items():
            path = building / f"{split}.jsonl"
            _write_jsonl(path, rows)
            artifacts[split] = {
                "path": str((generation / path.name).relative_to(output_index.parent)),
                "count": len(rows),
            }
        building.replace(generation)
    finally:
        if building.exists():
            shutil.rmtree(building)
    index = {
        **artifact_header(STAGE1_TARGET_INDEX),
        "model_family": config.model_family,
        "compile_contract": compile_contract,
        "artifacts": artifacts,
        "statistics": {
            "train_horizon_coverage": (
                statistics["train"]["retained_count"] / formal_count
            ),
            "train_retained_count": statistics["train"]["retained_count"],
            "train_excluded_cot_count": statistics["train"]["excluded_cot_count"],
            "train_excluded_answer_count": statistics["train"]["excluded_answer_count"],
            "train_native_occurrence_count": statistics["train"][
                "native_occurrence_count"
            ],
            "train_c_sample_count": statistics["train"]["c_sample_count"],
            "train_d_candidate_count": statistics["train"]["d_candidate_count"],
            "train_d_sample_count": statistics["train"]["d_sample_count"],
            "train_d_excluded_non_eos_count": statistics["train"][
                "d_excluded_non_eos_count"
            ],
            "train_d_excluded_answer_capacity_count": statistics["train"][
                "d_excluded_answer_capacity_count"
            ],
        },
    }
    validate_target_index_payload(
        index,
        index_path=output_index,
        model_family=config.model_family,
    )
    donor_payload = _hard_donor_payload(
        {"validation": compiled["validation"]},
        content_capacity=int(config.cot_content_capacity),
    )
    _write_stage1_cache_json(Path(arguments.donors), donor_payload, label="hard-donors")
    manifest = _manifest_payload(
        config,
        index=index,
        compiled=compiled,
        tokenizer_sha256=tokenizer_sha256,
        template_sha256=template_sha256,
        boundary_ids=boundary_ids,
        z_width=int(executor.config.hidden_size),
        world_size=int(arguments.world_size),
        route1_local_samples=int(arguments.route1_local_samples),
        route1_gradient_accumulation_steps=int(
            arguments.route1_gradient_accumulation_steps
        ),
    )
    _write_stage1_cache_json(
        Path(arguments.manifest), manifest, label="route1-manifest"
    )
    # The shared publication is complete once its immutable target rows,
    # donor manifest, and Stage1 manifest have all been written and validated.
    # Commit that cache before creating run-local runtime sidecars: a failure in
    # one training run must not discard an otherwise reusable 100k-row compile.
    _write_stage1_cache_json(output_index, index, label="target-index")
    _ensure_runtime_sidecars(
        config,
        arguments,
        expected_tokenizer_sha256=tokenizer_sha256,
        expected_template_sha256=template_sha256,
        expected_boundary_ids_sha256=boundary_ids_sha256,
        tokenizer=tokenizer,
        executor=executor,
    )
    _done(
        "rebuild",
        started,
        f"train={len(compiled['train'])} validation={len(compiled['validation'])} "
        f"manifest={arguments.manifest}",
    )
    return 0


def resolve_target_artifact(
    index_path: Path,
    split: str,
    *,
    model_family: str | None = None,
    **_unused: Any,
) -> Path:
    index = _read_index(index_path)
    if not _index_is_complete(
        index,
        index_path=index_path,
        expected_model_family=model_family,
    ):
        raise ValueError("target index is incomplete or stale")
    if split not in {"train", "validation"}:
        raise ValueError("unknown target split")
    return resolve_target_index_artifact_path(
        str(index["artifacts"][split]["path"]),
        index_path=index_path,
    )
