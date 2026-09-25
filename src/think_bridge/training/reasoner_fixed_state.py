"""Save and validate persistent R buffers independently of optimizer ownership."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

FILENAME = "reasoner_fixed_state.safetensors"
_METADATA = {"artifact_type": "reasoner_fixed_function_state", "format_version": "1"}
_NAMES = {"step_embed", "tau_bound"}


def reasoner_module(module: Any) -> Any | None:
    if hasattr(module, "tau_bound") and hasattr(module, "step_embed"):
        return module
    return getattr(module, "reasoner", None)


def fixed_state(module: Any) -> dict[str, torch.Tensor]:
    reasoner = reasoner_module(module)
    if reasoner is None:
        return {}
    persistent = set(reasoner.state_dict()).difference(
        dict(reasoner.named_parameters())
    )
    if persistent != _NAMES:
        raise ValueError("R persistent fixed-state schema differs")
    buffers = dict(reasoner.named_buffers())
    result = {
        name: buffers[name].detach().cpu().contiguous().clone()
        for name in sorted(_NAMES)
    }
    _validate(result)
    return result


def _validate(state: Mapping[str, Any]) -> None:
    if set(state) != _NAMES:
        raise ValueError("R requires complete step_embed and tau_bound")
    for name, value in state.items():
        if (
            not torch.is_tensor(value)
            or value.dtype != torch.float32
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"R fixed-state tensor must be finite FP32: {name}")
    if state["tau_bound"].shape != torch.Size([]) or state["tau_bound"].item() <= 0:
        raise ValueError("R fixed-state tau_bound must be a positive scalar")
    if state["step_embed"].ndim != 3:
        raise ValueError("R fixed-state step_embed must have rank three")


def save_fixed_state(root: Path, module: Any) -> None:
    state = fixed_state(module)
    if state:
        from think_bridge.training.train import _tensor_tree_sha256

        parameters = dict(reasoner_module(module).named_parameters())
        metadata = {
            **_METADATA,
            "reasoner_parameters_sha256": _tensor_tree_sha256(parameters),
            "fixed_state_sha256": _tensor_tree_sha256(state),
        }
        save_file(state, str(Path(root) / FILENAME), metadata=metadata)


def load_fixed_state(
    root: Path, parameters: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    root = Path(root)
    path = root / FILENAME
    # Benchmark does not validate the full training seal. Still enforce this
    # component's ledger entry when reading a published checkpoint/transition.
    from think_bridge.model.contract import file_sha256

    ledger = None
    for metadata_name in ("checkpoint.json",):
        metadata_path = root / metadata_name
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            ledger = (
                metadata.get("file_ledger") if isinstance(metadata, Mapping) else None
            )
            if not isinstance(ledger, Mapping):
                raise ValueError("R fixed-state checkpoint file ledger is missing")
            if FILENAME in ledger:
                if not path.is_file() or file_sha256(path) != ledger[FILENAME]:
                    raise ValueError(
                        "R fixed state differs from checkpoint file ledger"
                    )
            elif path.exists():
                raise ValueError(
                    "R fixed state is not registered in checkpoint file ledger"
                )
            break
    if path.exists():
        with safe_open(str(path), framework="pt", device="cpu") as stream:
            metadata = stream.metadata() or {}
            required = set(_METADATA) | {
                "reasoner_parameters_sha256",
                "fixed_state_sha256",
            }
            if (
                not required.issubset(metadata)
                or metadata.get("artifact_type") != _METADATA["artifact_type"]
                or metadata.get("format_version") != _METADATA["format_version"]
            ):
                raise ValueError("R fixed-state version/metadata differs")
        from think_bridge.training.train import _tensor_tree_sha256

        if metadata["reasoner_parameters_sha256"] != _tensor_tree_sha256(parameters):
            raise ValueError("R fixed-state parameter binding differs")
        state = dict(load_file(str(path), device="cpu"))
        _validate(state)
        if metadata["fixed_state_sha256"] != _tensor_tree_sha256(state):
            raise ValueError("R fixed-state content hash differs")
        return state

    raise FileNotFoundError(
        "checkpoint lacks required reasoner_fixed_state.safetensors"
    )


def restore_fixed_state(module: Any, state: Mapping[str, Any]) -> None:
    reasoner = reasoner_module(module)
    if reasoner is None:
        if state:
            raise ValueError("fixed R state supplied to a non-R owner")
        return
    _validate(state)
    expected = fixed_state(reasoner)
    if set(state) != set(expected):
        raise ValueError("R fixed-state architecture schema differs")
    for name, value in state.items():
        if value.shape != expected[name].shape:
            raise ValueError(f"R fixed-state shape differs: {name}")
    with torch.no_grad():
        for name, value in state.items():
            buffer = dict(reasoner.named_buffers())[name]
            expected_dtype = torch.float32
            if buffer.dtype != expected_dtype or buffer.requires_grad:
                raise ValueError(
                    f"runtime R fixed-state dtype/ownership differs: {name}"
                )
            buffer.copy_(value.to(device=buffer.device))


def load_owner_fixed_state(
    root: Path, owner: str, parameters: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    if owner == "D":
        return {}
    if owner != "R":
        raise ValueError("fixed-state owner must be R or D")
    if not parameters:
        raise ValueError("R fixed-state restore lacks portable R parameters")
    return load_fixed_state(root, parameters)
