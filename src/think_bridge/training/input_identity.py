"""Content-bound Stage0 roles for native supervision and R training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from think_bridge.model.contract import canonical_json_sha256, require_sha256

INPUT_ROLES = (
    "train_dataset",
    "eval_dataset",
    "train_behavior",
    "eval_behavior",
    "train_direct_answer_source",
)
BEHAVIOR_BINDING = "raw-source-rendered-prompt-greedy-decode-producer-metadata"


def file_reference(path: str | Path, label: str) -> dict[str, Any]:
    candidate = Path(path).expanduser().resolve(strict=True)
    if not candidate.is_file():
        raise ValueError(f"{label} must be a regular file: {candidate}")
    before = candidate.stat()
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    after = candidate.stat()
    signature = lambda stat: (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )
    if signature(before) != signature(after):
        raise ValueError(f"{label} changed while hashing: {candidate}")
    return {
        "path": str(candidate),
        "sha256": digest.hexdigest(),
        "bytes": after.st_size,
    }


def content_identity(reference: Mapping[str, Any], label: str) -> dict[str, Any]:
    if reference.get("path") is None:
        role = reference.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError(f"{label} lacks an explicit absent-input role")
        return {"absent": role}
    try:
        digest = require_sha256(reference.get("sha256"), f"{label} sha256")
        size = reference["bytes"]
        if type(size) is not int or size < 0:
            raise ValueError("invalid file size")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} lacks a sealed content identity. Prepare and train a new R "
            "with content-bound Stage0 inputs; do not add hashes to an old checkpoint."
        ) from exc
    return {"sha256": digest, "bytes": size}


def input_contract(inputs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    missing = set(INPUT_ROLES).difference(inputs)
    if missing:
        raise ValueError(f"Stage0 input binding lacks roles: {sorted(missing)}")
    return {
        "artifact_type": "think-bridge.stage1.input-content-contract",
        "schema_version": 1,
        "roles": {role: content_identity(inputs[role], role) for role in INPUT_ROLES},
    }


def dataset_identity(inputs: Mapping[str, Mapping[str, Any]]) -> str:
    return canonical_json_sha256(input_contract(inputs))


def verify_reference(reference: Mapping[str, Any], label: str) -> Path | None:
    expected = content_identity(reference, label)
    if reference.get("path") is None:
        return None
    actual = file_reference(reference["path"], label)
    if content_identity(actual, label) != expected:
        raise ValueError(
            f"{label} content changed from its sealed Stage0 binding: {actual['path']}"
        )
    return Path(actual["path"])


def verify_inputs(inputs: Mapping[str, Mapping[str, Any]]) -> None:
    input_contract(inputs)
    for role in INPUT_ROLES:
        verify_reference(inputs[role], role)


def assert_target_input_binding(index_path: Path, inputs: Mapping[str, Any]) -> None:
    verify_inputs(inputs)
    try:
        index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Missing compiled targets: {index_path}. Copy the complete content-addressed "
            "dataset cache (targets/generations, targets/index.json, manifests/stage1.json "
            "and validation-donors.json) from the R training host, or rebuild it with "
            "the prepare-targets worker using this run's input data."
        ) from exc
    contract = index.get("compile_contract", {})
    if (
        contract.get("inputs") != input_contract(inputs)
        or contract.get("behavior_binding") != BEHAVIOR_BINDING
        or contract.get("validation_cot_source")
        != "native-observation-with-explicit-unknown"
    ):
        raise ValueError(
            "Prepared targets do not match the stage's five bound Stage0 inputs; "
            "copy or rebuild the selected R dataset cache before training D."
        )


def verify_stage0_executor_proofs(config: Any, arguments: Any) -> None:
    """Check available producer proofs; never manufacture proofs for legacy data."""
    from think_bridge.model.executor_identity import (
        model_source_identity,
        validate_executor_identity,
    )
    from think_bridge.data.dataset import load_data_file

    expected = None

    def current():
        nonlocal expected
        if expected is None:
            expected = model_source_identity(
                config.model_name_or_path,
                local_files_only=bool(arguments.local_files_only),
                no_progress=bool(arguments.no_progress),
            )
        return expected

    for name in ("train_behavior", "validation_behavior", "train_direct_answer_source"):
        path = getattr(arguments, name, None)
        if path is None:
            continue
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        if "frozen_executor_identity" in metadata:
            proof = validate_executor_identity(metadata["frozen_executor_identity"])
            if proof != current():
                raise ValueError(f"Stage0 {name} was generated by a different frozen F")
    for row in load_data_file(Path(arguments.train_source)):
        algorithm = row.get("generation_executor_identity_algorithm")
        if algorithm is not None and algorithm != "hf-pretrained-files-sha256":
            raise ValueError(
                f"Unsupported Stage0 native executor identity algorithm: {algorithm}"
            )
        if "generation_executor_artifact_sha256" in row:
            proof = require_sha256(
                row["generation_executor_artifact_sha256"], "native executor digest"
            )
            if algorithm is not None and proof != current()["sha256"]:
                raise ValueError(
                    "Stage0 native rollout was generated by a different frozen F"
                )
        elif algorithm is not None:
            raise ValueError(
                "Stage0 native executor proof declares an algorithm without a digest"
            )


def load_bound_behavior(path: Path, source: Path, *, split: str, tokenizer: Any):
    """Bind validated label rows to actual source prompts and gold answers."""
    from think_bridge.data.dataset import load_data_file
    from think_bridge.data.behavior_manifest import (
        validate_behavior_rows,
        prompt_token_ids_key,
        gold_answer_key,
    )
    from think_bridge.data.templates import build_thinking_prompt

    raw = load_data_file(source)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status", "complete") != "complete":
        raise ValueError("behavior data must be a complete JSON object")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("behavior metadata must be an object when provided")
    labels = validate_behavior_rows(
        payload.get("prompts"),
        native_label_source="stage0" if split == "train" else "generate",
        decode_contract=metadata.get("decode_contract"),
    )
    prompt_keys = {}

    def key(row):
        question = str(row.get("question") or "").strip()
        if not question:
            raise ValueError("Behavior source has an empty question")
        if question not in prompt_keys:
            ids = tokenizer.encode(
                build_thinking_prompt(tokenizer, question), add_special_tokens=False
            )
            prompt_keys[question] = prompt_token_ids_key(ids)
        return prompt_keys[question]

    if split == "train":
        from think_bridge.data.rollout_identity import (
            validate_stage0_rollout_identities,
        )

        validate_stage0_rollout_identities(raw, rendered_prompt_key_fn=key)
    observed = set()
    correct_counts = {}
    for row in raw:
        prompt = key(row)
        label = labels.get(prompt)
        if label is None or label["gold_answer_key"] != gold_answer_key(
            str(row.get("answer", "")).strip(), "math"
        ):
            raise ValueError("Behavior does not cover the raw source question/answer")
        observed.add(prompt)
        if split == "train":
            correct_counts[prompt] = correct_counts.get(prompt, 0) + int(row["correct"])
    if observed != set(labels):
        raise ValueError("Behavior prompt population differs from the raw source")
    if any(
        labels[prompt]["n_correct"] != count for prompt, count in correct_counts.items()
    ):
        raise ValueError(
            "Behavior native success counts differ from the source rollouts"
        )
    return labels, metadata


def load_bound_direct(
    path: Path, source: Path, *, tokenizer: Any, behavior_metadata: Mapping[str, Any]
):
    from think_bridge.data.direct_raw import validate_direct_records

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status", "complete") != "complete":
        raise ValueError("direct data must be a complete JSON object")
    return validate_direct_records(payload.get("records"))
