"""Inspectable frozen-executor greedy-direct outputs prepared by Stage0.

This artifact is deliberately separate from the compact behavior manifest.
It preserves exact output token ids and human-readable text for audits, while
Stage1 consumes only the derived prompt-level labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping


DIRECT_RAW_SCHEMA = 1
_LEGACY_DIRECT_RAW_SCHEMA = "thinkbridge-direct-raw-v1"
_SHA256_HEX_LENGTH = 64
_RECORD_FIELDS = (
    "source_id",
    "question",
    "gold_answer",
    "task_type",
    "prompt_group_key",
    "direct_prompt_key",
    "direct_output",
    "direct_token_ids",
    "direct_hit_eos",
    "direct_correct",
)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_backend(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"name", "version"}:
        raise ValueError("direct raw inference_backend fields are incompatible")
    name = value.get("name")
    version = value.get("version")
    if name != "hf" or not isinstance(version, str) or not version.strip():
        raise ValueError("formal direct raw requires a versioned exact-HF backend")
    return {"name": "hf", "version": version.strip()}


def _validate_decode_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("direct raw decode_contract must be an object")
    expected_fields = {
        "do_sample",
        "temperature",
        "top_p",
        "max_new_tokens",
        "decoder",
        "judge",
    }
    if set(value) != expected_fields:
        raise ValueError("direct raw decode_contract fields are incompatible")
    temperature = value.get("temperature")
    top_p = value.get("top_p")
    if (
        value.get("do_sample") is not False
        or isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or float(temperature) != 0.0
        or isinstance(top_p, bool)
        or not isinstance(top_p, (int, float))
        or float(top_p) != 1.0
        or value.get("decoder") != "thinkbridge.greedy-token-decode-v1"
        or value.get("judge")
        != "think_bridge.eval.answer_match.judge_answer"
    ):
        raise ValueError("direct raw requires the formal deterministic greedy contract")
    max_new_tokens = value.get("max_new_tokens")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
    ):
        raise ValueError("direct raw max_new_tokens must be a positive integer")
    return dict(value)


def direct_decode_contract(*, max_new_tokens: int) -> dict[str, Any]:
    value = {
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": int(max_new_tokens),
        "decoder": "thinkbridge.greedy-token-decode-v1",
        "judge": "think_bridge.eval.answer_match.judge_answer",
    }
    return _validate_decode_contract(value)


def load_direct_raw(
    path: str | Path,
    *,
    prompt_fingerprint: str,
    source_fingerprint: str,
    inference_backend: Mapping[str, Any],
    decode_contract: Mapping[str, Any],
    legacy_prompt_fingerprint_for_tokenizer_source: (
        Callable[[str], str] | None
    ) = None,
) -> list[dict[str, Any]]:
    """Load and fail-closed validate a complete direct-output artifact."""

    artifact_path = Path(path)
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"direct raw artifact cannot be read: {artifact_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("direct raw artifact must be a JSON object")
    if payload.get("schema_version") not in {
        DIRECT_RAW_SCHEMA,
        _LEGACY_DIRECT_RAW_SCHEMA,
    }:
        raise ValueError("direct raw schema version is incompatible")
    if payload.get("status") != "complete":
        raise ValueError("direct raw status must be 'complete'")
    metadata = payload.get("metadata")
    records = payload.get("records")
    if not isinstance(metadata, dict) or not isinstance(records, list):
        raise ValueError("direct raw requires metadata and a records array")

    expected_hashes = {"source_fingerprint": source_fingerprint}
    for name, expected in expected_hashes.items():
        if not _is_sha256(expected):
            raise ValueError(f"expected direct raw {name} must be SHA-256")
        if metadata.get(name) != expected:
            raise ValueError(f"direct raw {name} differs from this request")
    if not _is_sha256(prompt_fingerprint):
        raise ValueError("expected prompt_fingerprint must be SHA-256")
    stored_prompt_fingerprint = metadata.get("prompt_fingerprint")
    if not _is_sha256(stored_prompt_fingerprint):
        raise ValueError("direct raw metadata prompt_fingerprint must be SHA-256")
    if stored_prompt_fingerprint != prompt_fingerprint:
        legacy_fingerprint = None
        if legacy_prompt_fingerprint_for_tokenizer_source is not None:
            tokenizer_source = metadata.get("tokenizer_source")
            if isinstance(tokenizer_source, str) and tokenizer_source.strip():
                legacy_fingerprint = (
                    legacy_prompt_fingerprint_for_tokenizer_source(
                        tokenizer_source
                    )
                )
                if not _is_sha256(legacy_fingerprint):
                    raise ValueError(
                        "direct raw legacy prompt fingerprint resolver "
                        "did not return a SHA-256 digest"
                    )
        if stored_prompt_fingerprint != legacy_fingerprint:
            raise ValueError("direct raw prompt fingerprint differs from this request")
    if _validate_backend(metadata.get("inference_backend")) != _validate_backend(
        inference_backend
    ):
        raise ValueError("direct raw backend differs from this request")
    if _validate_decode_contract(metadata.get("decode_contract")) != (
        _validate_decode_contract(decode_contract)
    ):
        raise ValueError("direct raw decode contract differs from this request")
    prompt_count = metadata.get("prompt_count")
    if (
        isinstance(prompt_count, bool)
        or not isinstance(prompt_count, int)
        or prompt_count <= 0
        or prompt_count != len(records)
    ):
        raise ValueError("direct raw prompt_count differs from records")

    return validate_direct_records(records)


def validate_direct_records(records):
    """Validate direct targets without requiring a particular producer's metadata."""
    if not isinstance(records, list) or not records:
        raise ValueError("direct records must be a nonempty array")
    normalized: list[dict[str, Any]] = []
    seen_prompt_keys: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"direct raw records[{index}] must be an object")
        if not set(_RECORD_FIELDS).issubset(record):
            raise ValueError(f"direct raw records[{index}] fields are incompatible")
        for name in ("prompt_group_key", "direct_prompt_key"):
            if not _is_sha256(record.get(name)):
                raise ValueError(f"invalid direct raw {name}: index={index}")
        prompt_key = str(record["prompt_group_key"])
        if prompt_key in seen_prompt_keys:
            raise ValueError(f"direct raw duplicate prompt_group_key: {prompt_key}")
        seen_prompt_keys.add(prompt_key)
        for name in (
            "source_id",
            "question",
            "gold_answer",
            "task_type",
            "direct_output",
        ):
            if not isinstance(record.get(name), str):
                raise ValueError(f"direct raw {name} must be a string: index={index}")
        if not record["question"].strip() or not record["gold_answer"].strip():
            raise ValueError(f"direct raw question/gold must not be empty: index={index}")
        token_ids = record.get("direct_token_ids")
        if not isinstance(token_ids, list) or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in token_ids
        ):
            raise ValueError(f"invalid direct raw token IDs: index={index}")
        if not isinstance(record.get("direct_hit_eos"), bool) or not isinstance(
            record.get("direct_correct"), bool
        ):
            raise ValueError(f"direct raw EOS/correct must be Boolean: index={index}")
        normalized.append({name: record[name] for name in _RECORD_FIELDS})
    return normalized


__all__ = [
    "DIRECT_RAW_SCHEMA",
    "direct_decode_contract",
    "load_direct_raw",
]
