"""Full-dataset free evaluation; independent of training checkpoint selection.

Controls are interventions on the same fixed latent slots. D never feeds R/F.
Optional behavior labels and native D targets must bind to exact rendered prompts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import unicodedata
from typing import Any


def _sha(path: Path) -> str:
    from think_bridge.model.contract import file_sha256

    return file_sha256(path)


def _normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _source_id(item, field):
    value = item.get(field)
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (str, int))
        or not str(value).strip()
    ):
        raise ValueError(f"source {field} must be a nonempty string or integer")
    return str(value).strip()


def load_rows(path: Path, tokenizer: Any) -> list[dict[str, Any]]:
    """Accept QA JSON arrays/{data:[...]} or JSONL, rejecting silent row loss."""
    from think_bridge.data.templates import build_thinking_prompt
    from think_bridge.data.behavior_manifest import (
        prompt_token_ids_key,
        gold_answer_key,
    )
    from think_bridge.data.dataset import load_data_file

    raw = load_data_file(path)
    if not isinstance(raw, list) or not raw:
        raise ValueError("dataset must contain a nonempty array of QA objects")
    rows, ids, prompts, questions = [], set(), {}, {}
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"dataset row {index} is not an object")
        question = item.get("question")
        answer = item.get("answer", item.get("reference_answer"))
        if (
            not isinstance(question, str)
            or not question.strip()
            or not isinstance(answer, (str, int, float))
            or isinstance(answer, bool)
            or not str(answer).strip()
        ):
            raise ValueError(f"dataset row {index} lacks question/answer")
        task = item.get("task_type", "math")
        if task != "math":
            raise ValueError("public evaluation currently supports math QA only")
        problem_id = _source_id(item, "problem_id")
        semantic_group = _source_id(item, "semantic_group_id")
        semantic_alias = _source_id(item, "semantic_id")
        if (
            semantic_group is not None
            and semantic_alias is not None
            and semantic_group != semantic_alias
        ):
            raise ValueError(
                f"conflicting semantic_group_id/semantic_id at row {index}"
            )
        semantic_id = semantic_group if semantic_group is not None else semantic_alias
        record_id = _source_id(item, "id") or problem_id or str(index)
        prompt = tokenizer.encode(
            build_thinking_prompt(tokenizer, question, task_type=task),
            add_special_tokens=False,
        )
        key = prompt_token_ids_key(prompt)
        normalized = _normalized(question)
        if record_id in ids:
            raise ValueError(f"dataset contains duplicate record ID at row {index}")
        answer_key = gold_answer_key(str(answer), task)
        if (key in prompts and prompts[key] != answer_key) or (
            normalized in questions and questions[normalized] != answer_key
        ):
            raise ValueError(
                f"dataset has conflicting answers for the same question/prompt at row {index}"
            )
        # Preserve repeated benchmark rows and their original metric weight.
        ids.add(record_id)
        prompts[key] = answer_key
        questions[normalized] = answer_key
        rows.append(
            dict(
                record_id=record_id,
                _question=question,
                reference_answer=str(answer),
                prompt_ids=prompt,
                prompt_group_key=key,
                normalized_question=normalized,
                problem_id=problem_id if problem_id is not None else record_id,
                semantic_id=semantic_id if semantic_id is not None else normalized,
                source_problem_id=problem_id,
                source_semantic_group_id=semantic_id,
                task_type=task,
            )
        )
    return rows


def bind_behavior(rows, path: Path, tokenizer, *, source: Path):
    # This structural contract binds native/direct labels to exact prompt IDs,
    # gold answers and the same no-thinking direct prompt used in training.
    from think_bridge.training.prepare import _apply_sealed_behavior

    _apply_sealed_behavior(
        rows, path=path, source=source, split="validation", tokenizer=tokenizer
    )
    for row in rows:
        row["group"] = {
            (True, True): "B",
            (True, False): "C",
            (False, True): "D",
            (False, False): "E",
        }[row["locked_full_correct"], row["direct_correct"]]
    return rows


def select_donors(rows, k: int, seed: int, *, protocol="standard"):
    """Nearest prompt length, then native CoT length if both known, hash ties.

    Exclude identical question/prompt and any shared supplied problem/semantic
    identity. Return up to k available donors; empty results preserve the anchor.
    """
    if protocol not in {"standard", "strict-identities", "text-only"}:
        raise ValueError("unsupported control protocol")
    problem_to_semantic = {}
    for row in rows:
        problem, semantic = (
            row.get("source_problem_id"),
            row.get("source_semantic_group_id"),
        )
        if protocol == "strict-identities" and (problem is None or semantic is None):
            raise ValueError(
                f"wrong-z strict-identities requires explicit source problem_id and "
                f"semantic_group_id (or semantic_id) for record {row['record_id']}; "
                "use standard to evaluate without these IDs"
            )
        if (
            protocol == "strict-identities"
            and problem is not None
            and semantic is not None
        ):
            if (
                problem in problem_to_semantic
                and problem_to_semantic[problem] != semantic
            ):
                raise ValueError(
                    f"source problem_id {problem!r} maps to conflicting semantic groups"
                )
            problem_to_semantic[problem] = semantic
    result = []
    for i, anchor in enumerate(rows):
        candidates = []
        for j, donor in enumerate(rows):
            same_prompt = any(
                anchor[key] == donor[key]
                for key in ("prompt_group_key", "normalized_question")
            )
            same_source_id = any(
                anchor.get(key) is not None and anchor[key] == donor.get(key)
                for key in ("source_problem_id", "source_semantic_group_id")
            )
            if i == j or same_prompt or same_source_id:
                continue
            distance = abs(len(anchor["prompt_ids"]) - len(donor["prompt_ids"]))
            # Observed native lengths include failures; success-only D targets
            # must not determine whether an anchor/donor participates.
            if anchor.get("cot_length_known", "cot_ids" in anchor) and donor.get(
                "cot_length_known", "cot_ids" in donor
            ):
                distance += abs(len(anchor["cot_ids"]) - len(donor["cot_ids"]))
            tie = hashlib.sha256(
                f"{seed}:{anchor['prompt_group_key']}:{donor['prompt_group_key']}".encode()
            ).hexdigest()
            candidates.append((distance, tie, j))
        selected, selected_prompts = [], set()
        for _, _, j in sorted(candidates):
            prompt = rows[j]["prompt_group_key"]
            if prompt in selected_prompts:
                continue
            selected.append(j)
            selected_prompts.add(prompt)
            if len(selected) == k:
                break
        result.append(selected)
    return result


def _mean(values):
    return sum(values) / len(values) if values else None


def summarize(records, *, conditions=None, include_groups=True):
    output = {}
    conditions = sorted(
        conditions
        if conditions is not None
        else {entry["condition"] for row in records for entry in row["generations"]}
    )
    for condition in conditions:
        grouped = [
            [g for g in row["generations"] if g["condition"] == condition]
            for row in records
        ]
        if condition != "wrong" and any((not group for group in grouped)):
            raise ValueError("incomplete question/condition coverage")
        all_grouped = grouped
        grouped = [group for group in grouped if group]
        metrics = dict(
            question_count=len(grouped),
            dataset_question_count=len(records),
            generation_count=sum(map(len, grouped)),
            answer_accuracy=_mean(
                [_mean([g["answer_correct"] for g in group]) for group in grouped]
            ),
            answer_cap_rate=_mean(
                [_mean([g["answer_cap_hit"] for g in group]) for group in grouped]
            ),
        )
        if condition == "wrong":
            histogram = {}
            for group in all_grouped:
                key = str(len(group))
                histogram[key] = histogram.get(key, 0) + 1
            metrics.update(
                available_question_count=len(grouped),
                unavailable_question_count=len(records) - len(grouped),
                requested_k=records[0].get("wrong_requested_k") if records else None,
                actual_k_histogram=histogram,
            )
        output[condition] = metrics
    groups = (
        sorted({row["group"] for row in records if "group" in row})
        if include_groups
        else []
    )
    return dict(
        conditions=output,
        groups={
            group: summarize(
                [row for row in records if row.get("group") == group],
                conditions=conditions,
                include_groups=False,
            )["conditions"]
            for group in groups
        },
    )


def evaluate(args):
    import torch
    from think_bridge.model.inference import _build_runtime
    from think_bridge.model.reasoner_inference import (
        reason_eval_prompts,
        reasoner_inference_metadata,
    )
    from think_bridge.model.evaluation_groups import FixedReasonerGroups
    from think_bridge.model.trajectory import generate_true_z_prefix
    from think_bridge.eval.answer_match import judge_answer
    from think_bridge.training.progress import bridge_progress
    from think_bridge.training.runtime_sidecars import load_route_runtime_identity
    from think_bridge.model.checkpoint_policy import checkpoint_owner_run_directory
    from think_bridge.model.contract import write_atomic_json

    runtime = _build_runtime(args)
    (model, tokenizer, device) = (runtime.model, runtime.tokenizer, runtime.device)
    model.eval()
    rows = load_rows(args.dataset, tokenizer)
    if args.behavior:
        rows = bind_behavior(rows, args.behavior, tokenizer, source=args.dataset)
    donors = (
        select_donors(rows, args.wrong_k, args.seed, protocol=args.control_protocol)
        if "wrong" in args.controls
        else [[] for _ in rows]
    )
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    identity = load_route_runtime_identity(
        checkpoint_owner_run_directory(runtime.reasoner_source.checkpoint),
        route="route1",
    )
    manifest = dict(
        status="running",
        dataset_sha256=_sha(args.dataset),
        behavior_sha256=_sha(args.behavior) if args.behavior else None,
        dataset_path=str(args.dataset.resolve()),
        question_count=len(rows),
        controls=list(args.controls),
        wrong_k=args.wrong_k if "wrong" in args.controls else 0,
        seed=args.seed,
        control_protocol="strict-identities"
        if args.control_protocol == "strict-identities"
        else "standard",
        control_identity_scope="matching prompt/text excluded; supplied problem/semantic IDs add exclusions",
        wrong_k_by_question=[
            dict(record_id=row["record_id"], actual_k=len(donors[i]))
            for (i, row) in enumerate(rows)
        ]
        if "wrong" in args.controls
        else [],
        max_new_tokens=args.max_new_tokens,
        runtime_identity=identity,
        reasoner_checkpoint=str(runtime.reasoner_source.checkpoint),
        reasoner_checkpoint_seal_sha256=_sha(
            runtime.reasoner_source.checkpoint / "checkpoint.json"
        ),
        dataset_split="user-supplied",
        protocol="independent-math-qa-greedy-batched-r-v2",
        reasoner_inference=reasoner_inference_metadata(model),
        answer_batch_size=1,
        judge="think_bridge.eval.answer_match.judge_answer",
        controls_scope="true/zero cover every question; wrong uses up to requested K available donors per question; same slot count",
        donor_rule="exclude same prompt/normalized question/problem/declared semantic ID; nearest prompt length plus observed native CoT length including failures when both known; SHA256 seed tie break",
        semantic_exclusion_limit="declared IDs and normalized text only; paraphrase equivalence is not inferred",
        group_source="bound behavior"
        if args.behavior
        else "unavailable; no groups inferred",
    )
    write_atomic_json(output / "manifest.json", manifest, replace_mismatch=True)
    started = time.monotonic()
    cache = FixedReasonerGroups(
        [row["prompt_ids"] for row in rows],
        lambda prompts: reason_eval_prompts(model, prompts),
        group_size=model._feedback_eval_batch_size,
    )
    records = []
    total = len(rows) * (len(args.controls) - int("wrong" in args.controls)) + sum(
        map(len, donors)
    )
    progress = bridge_progress(
        total=total,
        desc="Independent eval",
        unit="generation",
        disabled=args.no_progress,
    )

    def tensors(row):
        ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
        return (ids, torch.ones_like(ids, dtype=torch.bool))

    def latent(index):
        return (
            cache.select([index])[0].unsqueeze(0).to(device=device, dtype=torch.float32)
        )

    try:
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ),
            (output / "predictions.jsonl").open("x", encoding="utf-8") as stream,
        ):
            for index, row in enumerate(rows):
                (ids, mask) = tensors(row)
                own = latent(index)
                conditions = []
                for condition in args.controls:
                    conditions.extend(
                        [(condition, j) for j in donors[index]]
                        if condition == "wrong"
                        else [(condition, index)]
                    )
                record = {
                    key: value
                    for (key, value) in row.items()
                    if key
                    in {
                        "record_id",
                        "reference_answer",
                        "prompt_ids",
                        "prompt_group_key",
                        "problem_id",
                        "semantic_id",
                        "source_problem_id",
                        "source_semantic_group_id",
                        "group",
                    }
                }
                record.update(question=row["_question"], generations=[])
                if "wrong" in args.controls:
                    record.update(
                        wrong_requested_k=args.wrong_k,
                        wrong_actual_k=len(donors[index]),
                        wrong_available=bool(donors[index]),
                    )
                for condition, donor_index in conditions:
                    z = (
                        own
                        if condition == "true"
                        else torch.zeros_like(own)
                        if condition == "zero"
                        else latent(donor_index)
                    )
                    generated = generate_true_z_prefix(
                        model.executor,
                        embedding=model.executor.get_input_embeddings(),
                        prompt_ids=ids,
                        prompt_mask=mask,
                        z=z,
                        boundary_ids=model.boundary_ids,
                        eos_token_id=tokenizer.eos_token_id,
                        max_steps=args.max_new_tokens,
                        temperature=0.0,
                        seed=args.seed,
                    )
                    answer_ids = generated.token_ids[0][
                        generated.token_mask[0].bool()
                    ].tolist()
                    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
                    entry = dict(
                        condition=condition,
                        donor_record_id=rows[donor_index]["record_id"]
                        if condition == "wrong"
                        else None,
                        answer=answer,
                        answer_token_ids=answer_ids,
                        answer_correct=bool(
                            judge_answer(answer, row["reference_answer"], "math")
                        ),
                        answer_cap_hit=not answer_ids
                        or answer_ids[-1] != tokenizer.eos_token_id,
                    )
                    record["generations"].append(entry)
                    progress.update(1)
                records.append(record)
                stream.write(
                    json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                )
                stream.flush()
        if len(records) != len(rows):
            raise RuntimeError("evaluation did not cover every input question")
        summary = summarize(records, conditions=args.controls)
        summary.update(
            status="complete",
            question_count=len(records),
            elapsed_seconds=time.monotonic() - started,
            note="Control metrics are diagnostics; they do not alter R true-accuracy checkpoint selection.",
        )
        write_atomic_json(output / "summary.json", summary)
        manifest.update(
            status="complete",
            predictions_sha256=_sha(output / "predictions.jsonl"),
            summary_sha256=_sha(output / "summary.json"),
        )
        write_atomic_json(output / "manifest.json", manifest, replace_mismatch=True)
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
        return summary
    except BaseException as exc:
        manifest.update(
            status="failed",
            completed_questions=len(records),
            error=f"{type(exc).__name__}: {exc}",
        )
        write_atomic_json(output / "manifest.json", manifest, replace_mismatch=True)
        raise
    finally:
        progress.close()
