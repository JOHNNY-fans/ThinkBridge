"""Immutable run-level tokenizer and frozen lexical tensors for ThinkBridge."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from think_bridge.model.artifact_schema import (
    TOKENIZER_RUNTIME_IDENTITY,
    TOKENIZER_RUNTIME_SIDECAR,
    artifact_header,
)

from think_bridge.model.contract import (
    canonical_json_sha256,
    file_sha256,
    require_sha256,
    resolve_boundary_token_ids,
    write_atomic_json,
)


TOKENIZER_RELATIVE_PATH = Path("runtime_sidecars/tokenizer")
_BASE_RUNTIME_IDENTITY_FIELDS = (
    "artifact_type",
    "schema_version",
    "tokenizer_sha256",
    "template_sha256",
    "attn_implementation",
    "boundary_text",
    "boundary_token_ids",
    "boundary_token_count",
    "boundary_ids_sha256",
    "hidden_size",
    "vocab_size",
    "eos_token_id",
    "pad_token_id",
    "frozen_executor_identity",
)


def tokenizer_identity_sha256(tokenizer: Any) -> str:
    """Hash the tokenizer behavior consumed by compiled Bridge tensors."""

    return canonical_json_sha256(
        {
            "class": type(tokenizer).__qualname__,
            "vocab": sorted(
                (str(token), int(index))
                for token, index in tokenizer.get_vocab().items()
            ),
            "all_special_tokens": [
                str(token) for token in tokenizer.all_special_tokens
            ],
            "all_special_ids": [int(token) for token in tokenizer.all_special_ids],
            "model_max_length": int(tokenizer.model_max_length),
            "padding_side": str(tokenizer.padding_side),
            "truncation_side": str(tokenizer.truncation_side),
        }
    )


def _tokenizer_file_ledger(root: Path) -> dict[str, str]:
    ledger: dict[str, str] = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_symlink():
            raise ValueError("immutable tokenizer sidecar cannot contain symlinks")
        if path.is_file():
            ledger[path.relative_to(root).as_posix()] = file_sha256(path)
    if not ledger:
        raise ValueError("immutable tokenizer sidecar is empty")
    return ledger


def _runtime_identity(
    *,
    tokenizer: Any,
    executor: Any,
    attn_implementation: str,
    boundary_text: str,
    no_progress: bool = False,
) -> dict[str, Any]:
    from think_bridge.model.executor_identity import executor_identity

    boundary_ids = resolve_boundary_token_ids(tokenizer, boundary_text)
    eos_token_id = int(tokenizer.eos_token_id)
    pad_token_id = int(
        eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    )
    hidden_size = int(executor.config.hidden_size)
    vocab_size = int(executor.config.vocab_size)
    if hidden_size <= 0 or vocab_size <= 0:
        raise ValueError("frozen executor config has invalid lexical dimensions")
    return {
        **artifact_header(TOKENIZER_RUNTIME_IDENTITY),
        "tokenizer_sha256": tokenizer_identity_sha256(tokenizer),
        "template_sha256": canonical_json_sha256(tokenizer.chat_template or ""),
        "attn_implementation": str(attn_implementation),
        "boundary_text": str(boundary_text),
        "boundary_token_ids": [int(value) for value in boundary_ids],
        "boundary_token_count": len(boundary_ids),
        "boundary_ids_sha256": canonical_json_sha256(boundary_ids),
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "frozen_executor_identity": executor_identity(
            executor, no_progress=no_progress
        ),
    }


def publish_runtime_sidecars(
    run_dir: Path,
    *,
    tokenizer: Any,
    executor: Any,
    attn_implementation: str,
    boundary_text: str,
    no_progress: bool = False,
) -> dict[str, Any]:
    """Publish immutable tokenizer assets and frozen-model identity."""

    run = Path(run_dir).resolve(strict=True)
    sidecar_root = run / "runtime_sidecars"
    sidecar_root.mkdir(exist_ok=True)
    tokenizer_root = run / TOKENIZER_RELATIVE_PATH
    tokenizer_building = tokenizer_root.with_name("tokenizer.building")
    if not tokenizer_root.exists():
        if tokenizer_building.exists():
            raise FileExistsError(
                "incomplete tokenizer sidecar transaction already exists"
            )
        tokenizer_building.mkdir()
        tokenizer.save_pretrained(tokenizer_building)
        tokenizer_building.replace(tokenizer_root)
    tokenizer_ledger = _tokenizer_file_ledger(tokenizer_root)
    identity = _runtime_identity(
        tokenizer=tokenizer,
        executor=executor,
        attn_implementation=attn_implementation,
        boundary_text=boundary_text,
        no_progress=no_progress,
    )
    index = {
        **artifact_header(TOKENIZER_RUNTIME_SIDECAR),
        "tokenizer_path": TOKENIZER_RELATIVE_PATH.as_posix(),
        "tokenizer_files": tokenizer_ledger,
        "tokenizer_files_sha256": canonical_json_sha256(tokenizer_ledger),
        "identity": identity,
        "identity_sha256": canonical_json_sha256(identity),
    }
    write_atomic_json(run / "runtime_sidecars.json", index, replace_mismatch=False)
    return identity


def load_runtime_tokenizer(run_dir: Path) -> Any:
    from transformers import AutoTokenizer

    from think_bridge.model.checkpoint_policy import (
        validate_runtime_sidecars,
    )

    run = Path(run_dir).resolve(strict=True)
    index = validate_runtime_sidecars(run)
    tokenizer = AutoTokenizer.from_pretrained(
        run / str(index["tokenizer_path"]),
        local_files_only=True,
    )
    return tokenizer


def _validated_base_runtime_identity(
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the tokenizer and frozen-model identity."""

    if not isinstance(identity, Mapping):
        raise ValueError("immutable runtime identity must be a mapping")
    if (
        identity.get("artifact_type")
        not in {
            TOKENIZER_RUNTIME_IDENTITY,
        }
        or identity.get("schema_version") != 1
    ):
        raise ValueError("immutable runtime identity schema mismatch")
    missing = [name for name in _BASE_RUNTIME_IDENTITY_FIELDS if name not in identity]
    if missing:
        raise ValueError(
            "immutable runtime identity fields are missing: " + ", ".join(missing)
        )
    base = {name: identity[name] for name in _BASE_RUNTIME_IDENTITY_FIELDS}
    from think_bridge.model.executor_identity import validate_executor_identity

    validate_executor_identity(base["frozen_executor_identity"])
    base.update(artifact_header(TOKENIZER_RUNTIME_IDENTITY))
    for name in (
        "tokenizer_sha256",
        "template_sha256",
        "boundary_ids_sha256",
    ):
        require_sha256(base[name], name)
    for name in ("attn_implementation", "boundary_text"):
        if not isinstance(base[name], str):
            raise ValueError(f"immutable runtime identity {name} must be text")
    boundary_ids = base["boundary_token_ids"]
    if (
        not isinstance(boundary_ids, list)
        or not boundary_ids
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in boundary_ids
        )
        or base["boundary_token_count"] != len(boundary_ids)
        or base["boundary_ids_sha256"] != canonical_json_sha256(boundary_ids)
    ):
        raise ValueError("immutable runtime boundary-token identity mismatch")
    for name in ("hidden_size", "vocab_size"):
        value = base[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"immutable runtime identity {name} is invalid")
    for name in ("eos_token_id", "pad_token_id"):
        value = base[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"immutable runtime identity {name} is invalid")
    return base


def load_route_runtime_identity(
    run_dir: Path, *, route: str = "route1"
) -> Mapping[str, Any]:
    from think_bridge.model.checkpoint_policy import validate_runtime_sidecars

    if route != "route1":
        raise ValueError("runtime identity route must be route1")
    return _validated_base_runtime_identity(
        validate_runtime_sidecars(run_dir)["identity"]
    )
