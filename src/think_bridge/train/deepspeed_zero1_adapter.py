"""Capability-bound access to the DeepSpeed ZeRO-1 owner state we consume."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any


_CONSUMED_OWNER_CAPABILITIES = (
    "optimizer",
    "averaged_gradients",
    "params_in_partition",
    "single_partition_of_fp32_groups",
    "get_grad_norm_direct",
    "loss_scale",
    "check_grad_overflow",
    "overflow",
    "cpu_offload",
)


@dataclass(frozen=True)
class DeepSpeedZero1OwnerAdapter:
    """Checked minimal owner protocol; callers never touch private fields."""

    owner: Any
    group_count: int

    @classmethod
    def supports_partition_owner(cls, owner: Any, *, optimizer: Any) -> bool:
        """Identify only owners exposing the exact protocol consumed here.

        This structural discovery check deliberately does not bind, inspect
        runtime values, or mutate the owner. ``bind`` remains the fail-closed
        authority for capability values and group identity.
        """

        return getattr(owner, "optimizer", None) is optimizer and all(
            hasattr(owner, name) for name in _CONSUMED_OWNER_CAPABILITIES
        )

    @classmethod
    def bind(cls, owner: Any, *, optimizer: Any, group_count: int):
        count = int(group_count)
        if isinstance(group_count, bool) or count <= 0 or count != group_count:
            raise RuntimeError("ZeRO group count must be a positive integer")
        missing = tuple(
            name for name in _CONSUMED_OWNER_CAPABILITIES if not hasattr(owner, name)
        )
        if missing:
            raise RuntimeError(f"ZeRO-1 missing required owner capabilities: {missing}")
        if owner.optimizer is not optimizer:
            raise RuntimeError("ZeRO owner/optimizer identity mismatch")

        optimizer_groups = getattr(optimizer, "param_groups", None)
        if not isinstance(optimizer_groups, Sequence) or isinstance(
            optimizer_groups, (str, bytes, bytearray)
        ):
            raise RuntimeError(
                "ZeRO optimizer param_groups must be an indexable sequence"
            )
        if len(optimizer_groups) != count:
            raise RuntimeError("ZeRO optimizer param_groups group-cardinality drift")

        if not isinstance(owner.averaged_gradients, Mapping):
            raise RuntimeError("ZeRO averaged gradients must be a mapping")
        if owner.averaged_gradients:
            raise RuntimeError(
                "ZeRO owner must bind before backward publishes gradients"
            )

        for name in (
            "params_in_partition",
            "single_partition_of_fp32_groups",
        ):
            groups = getattr(owner, name)
            if not isinstance(groups, Sequence) or isinstance(
                groups, (str, bytes, bytearray)
            ):
                raise RuntimeError(f"ZeRO {name} must be an indexable group sequence")
            if len(groups) != count:
                raise RuntimeError(f"ZeRO {name} group-cardinality drift")
        for group in owner.params_in_partition:
            if not isinstance(group, Sequence) or isinstance(
                group, (str, bytes, bytearray)
            ):
                raise RuntimeError("ZeRO params_in_partition groups must be sequences")

        if not callable(owner.get_grad_norm_direct):
            raise RuntimeError("ZeRO get_grad_norm_direct must be callable")
        try:
            loss_scale = float(owner.loss_scale)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("ZeRO loss_scale must be finite and positive") from exc
        if not math.isfinite(loss_scale) or loss_scale <= 0.0:
            raise RuntimeError("ZeRO loss_scale must be finite and positive")
        for name in ("check_grad_overflow", "overflow", "cpu_offload"):
            if not isinstance(getattr(owner, name), bool):
                raise RuntimeError(f"ZeRO {name} must be boolean")
        if owner.cpu_offload:
            raise RuntimeError("ZeRO cpu_offload must be disabled")
        return cls(owner=owner, group_count=count)

    @property
    def signature(self) -> str:
        owner_type = type(self.owner)
        return (
            f"{owner_type.__module__}.{owner_type.__name__}:"
            f"zero1-fp32-groups={self.group_count}"
        )

    @property
    def averaged_gradients(self):
        return self.owner.averaged_gradients

    @property
    def optimizer(self):
        """Return the capability-bound base optimizer that owns Adam state."""

        return self.owner.optimizer

    def partition_parameters(self, index: int):
        return tuple(self.owner.params_in_partition[index])

    def master_parameter(self, index: int):
        return self.owner.single_partition_of_fp32_groups[index]

    def grad_norm(self, gradients, parameters):
        return self.owner.get_grad_norm_direct(gradients, parameters)

    def scaled_global_norm(self) -> float:
        """Return the backend-computed global norm used by Bridge clipping."""

        return float(self.owner.scaled_global_norm())

    @property
    def loss_scale(self) -> float:
        return float(self.owner.loss_scale)

    @property
    def check_grad_overflow(self) -> bool:
        return bool(self.owner.check_grad_overflow)

    @property
    def overflow(self) -> bool:
        return bool(self.owner.overflow)
