"""Content identity of the immutable files loaded by the frozen executor.

Paths are locators, not identities. Each host hashes a file once per filesystem
version; local workers share a locked cache. This attests standard pretrained
loading from those files, not arbitrary changes to model tensors after loading.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from think_bridge.model.contract import canonical_json_sha256, require_sha256


def _cached_sha256(path: Path, *, no_progress: bool) -> str:
    import fcntl

    # Private, per-host cache: ctime also invalidates replacements with preserved
    # size/mtime. Never reuse a digest based only on a model name or HF revision.
    cache = Path(tempfile.gettempdir()) / f"thinkbridge-executor-hashes-{os.getuid()}"
    cache.mkdir(mode=0o700, exist_ok=True)
    if cache.is_symlink() or cache.stat().st_uid != os.getuid():
        raise ValueError("executor hash cache must be a private local directory")
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    signature = [
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    ]
    key = canonical_json_sha256([str(resolved), signature])
    with (cache / f"{key}.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record = cache / f"{key}.json"
        if record.is_file():
            payload = json.loads(record.read_text())
            if payload.get("signature") == signature:
                return require_sha256(payload.get("sha256"), "frozen file digest")
        from tqdm.auto import tqdm

        digest = hashlib.sha256()
        with (
            resolved.open("rb") as stream,
            tqdm(
                total=stat.st_size,
                unit="B",
                unit_scale=True,
                desc=f"Verify F {path.name}",
                disable=no_progress or stat.st_size < 1024**2,
            ) as progress,
        ):
            while chunk := stream.read(8 * 1024**2):
                digest.update(chunk)
                progress.update(len(chunk))
        after = resolved.stat()
        if signature != [
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ]:
            raise ValueError("frozen model files changed while hashing")
        value = digest.hexdigest()
        temporary = record.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps({"signature": signature, "sha256": value}))
        temporary.replace(record)
        return value


def model_source_identity(
    model_name_or_path: str,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
    no_progress: bool = False,
) -> dict[str, Any]:
    """Hash config and the exact standard HF weight set, independently of path."""
    root = Path(model_name_or_path).expanduser()
    if not root.is_dir():
        from transformers.utils.hub import cached_file, extract_commit_hash
        from huggingface_hub import snapshot_download

        config = cached_file(
            model_name_or_path,
            "config.json",
            revision=revision,
            local_files_only=local_files_only,
        )
        if not config:
            raise ValueError("frozen executor source configuration is unavailable")
        commit = extract_commit_hash(str(config), None)
        if not commit:
            raise ValueError("frozen executor Hub source did not resolve to an immutable commit")
        # Resolve one standard weight set from the same immutable revision as
        # config. This also works when only config was previously cached.
        weight_name = None
        shards = []
        for candidate in ("model.safetensors.index.json", "model.safetensors",
                          "pytorch_model.bin.index.json", "pytorch_model.bin"):
            resolved = cached_file(
                model_name_or_path, candidate, revision=commit,
                local_files_only=local_files_only,
                _raise_exceptions_for_missing_entries=False,
            )
            if resolved:
                weight_name = candidate
                if candidate.endswith(".index.json"):
                    mapping = json.loads(Path(resolved).read_text()).get("weight_map")
                    if not isinstance(mapping, dict) or not mapping:
                        raise ValueError("frozen executor weight index is invalid")
                    shards = list(set(mapping.values()))
                    if any(not isinstance(name, str) or Path(name).is_absolute()
                           or ".." in Path(name).parts for name in shards):
                        raise ValueError("frozen executor shard path is invalid")
                break
        if weight_name is None:
            raise ValueError("frozen executor pretrained weights are unavailable")
        from tqdm.auto import tqdm

        class DownloadProgress(tqdm):
            def __init__(self, *args, **kwargs):
                if no_progress:
                    kwargs["disable"] = True
                super().__init__(*args, **kwargs)

        root = Path(snapshot_download(
            model_name_or_path, revision=commit, local_files_only=local_files_only,
            allow_patterns=["config.json", "generation_config.json", weight_name, *shards],
            tqdm_class=DownloadProgress,
        ))
    files = ["config.json"]
    if (root / "generation_config.json").is_file():
        files.append("generation_config.json")
    candidates = [
        name
        for name in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
        if (root / name).is_file()
    ]
    if len(candidates) != 1:
        raise ValueError(
            "frozen executor requires exactly one unambiguous pretrained weight set"
        )
    for single, index in (
        ("model.safetensors", "model.safetensors.index.json"),
        ("pytorch_model.bin", "pytorch_model.bin.index.json"),
    ):
        if (root / single).is_file():
            files.append(single)
            break
        if (root / index).is_file():
            mapping = json.loads((root / index).read_text()).get("weight_map")
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("frozen executor weight index is invalid")
            shards = sorted(set(mapping.values()))
            if any(
                not isinstance(name, str)
                or Path(name).is_absolute()
                or ".." in Path(name).parts
                for name in shards
            ):
                raise ValueError("frozen executor shard path is invalid")
            files.extend([index, *shards])
            break
    else:
        raise ValueError(
            "frozen executor identity requires local pretrained weight files"
        )
    ledger = {
        name: _cached_sha256(root / name, no_progress=no_progress)
        for name in sorted(files)
    }
    return {
        "schema_version": 1,
        "algorithm": "hf-pretrained-files-sha256",
        "files": ledger,
        "sha256": canonical_json_sha256(ledger),
    }


def validate_executor_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    if (
        not isinstance(identity, Mapping)
        or set(identity) != {"schema_version", "algorithm", "files", "sha256"}
        or identity["schema_version"] != 1
        or identity["algorithm"] != "hf-pretrained-files-sha256"
        or not isinstance(identity["files"], dict)
        or "config.json" not in identity["files"]
        or not any(
            name.endswith((".safetensors", ".bin")) for name in identity["files"]
        )
        or identity["sha256"] != canonical_json_sha256(identity["files"])
    ):
        raise ValueError("missing or invalid frozen executor content identity")
    for name, digest in identity["files"].items():
        if (
            not isinstance(name, str)
            or Path(name).is_absolute()
            or ".." in Path(name).parts
        ):
            raise ValueError("invalid frozen executor file identity")
        require_sha256(digest, name)
    return dict(identity)


def executor_identity(executor: Any, *, no_progress: bool = False) -> dict[str, Any]:
    source = getattr(executor.config, "_name_or_path", None)
    if not source:
        raise ValueError("frozen executor has no verifiable pretrained source")
    return model_source_identity(
        str(source),
        revision=getattr(executor.config, "_commit_hash", None),
        local_files_only=True,
        no_progress=no_progress,
    )


def assert_model_source_identity(
    model_name_or_path: str, expected_identity: Mapping[str, Any], **kwargs: Any
) -> dict[str, Any]:
    expected = validate_executor_identity(expected_identity)
    observed = model_source_identity(model_name_or_path, **kwargs)
    if observed != expected:
        raise ValueError("frozen executor content identity mismatch")
    return observed


def assert_executor_identity(
    executor: Any, expected_identity: Mapping[str, Any], *, no_progress: bool = False
) -> dict[str, Any]:
    expected = validate_executor_identity(expected_identity)
    observed = executor_identity(executor, no_progress=no_progress)
    if observed != expected:
        raise ValueError("frozen executor content identity mismatch")
    return observed
