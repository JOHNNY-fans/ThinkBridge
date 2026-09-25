"""Frozen scientific recipe for Bridge staged feedback distillation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import hashlib


METHOD_VERSION = "bridge"
OBJECTIVE_VERSION = "answer-ce-bc-forward-kl-same-prompt-specificity"


@dataclass(frozen=True)
class Phase:
    name: str
    owner: str
    epochs: int


@dataclass(frozen=True)
class BridgeRecipe:
    method_version: str
    objective_version: str
    phases: tuple[Phase, ...]


BRIDGE_RECIPE = BridgeRecipe(
    method_version=METHOD_VERSION,
    objective_version=OBJECTIVE_VERSION,
    phases=(Phase("route1", "R", 2),),
)


def phase_for_epoch(epoch: int, *, route1_epochs: int = 2) -> Phase:
    """Resolve the supervised R course."""
    if any(
        isinstance(v, bool) or not isinstance(v, int) for v in (epoch, route1_epochs)
    ):
        raise TypeError("Bridge epoch counts must be integers")
    if route1_epochs <= 0 or not 0 <= epoch < route1_epochs:
        raise ValueError("Bridge epoch lies outside the configured R course")
    return Phase("route1", "R", route1_epochs)


def answer_course_geometry(epoch: int) -> str:
    """Legacy epoch helper; step-based training resolves suffix activity from its update clock."""

    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("Route1 answer course epoch must be non-negative")
    return "deployed_z_cot_suffix" if epoch == 0 else "deployed_z"


def route1_course_updates(
    optimizer_steps_per_epoch: int,
    course_epochs: float | None = 0.0,
    *,
    course_steps: int | None = None,
) -> int:
    """Positive steps win; otherwise positive epochs, otherwise no curriculum."""
    if isinstance(optimizer_steps_per_epoch, bool) or not isinstance(
        optimizer_steps_per_epoch, int
    ):
        raise TypeError("Route1 curriculum coordinates must be integers")
    if optimizer_steps_per_epoch <= 0:
        raise ValueError("Route1 curriculum epoch must contain optimizer updates")
    if course_steps is not None:
        if isinstance(course_steps, bool) or not isinstance(course_steps, int):
            raise ValueError("course_steps must be an integer or unset")
        if course_steps > 0:
            return course_steps
    if course_epochs is None:
        return 0
    if (
        isinstance(course_epochs, bool)
        or not isinstance(course_epochs, (int, float))
        or not math.isfinite(course_epochs)
    ):
        raise ValueError("course_epochs must be finite or unset")
    duration = optimizer_steps_per_epoch * max(0.0, course_epochs)
    if not math.isfinite(duration):
        raise ValueError("course_epochs produces a non-finite update budget")
    return math.ceil(duration)


def route1_curriculum_value(
    optimizer_step: int,
    optimizer_steps_per_epoch: int,
    *,
    course_epochs: float | None = 0.0,
    course_steps: int | None = None,
) -> float:
    """Head-removal fraction at completed update u; first=0, last course update=1."""
    if (
        isinstance(optimizer_step, bool)
        or not isinstance(optimizer_step, int)
        or optimizer_step < 0
    ):
        raise ValueError("Route1 curriculum update must be a nonnegative integer")
    duration = route1_course_updates(
        optimizer_steps_per_epoch, course_epochs, course_steps=course_steps
    )
    if duration <= 1:
        return 1.0
    return min(1.0, optimizer_step / (duration - 1))


def route1_specificity_scale(
    optimizer_step: int,
    optimizer_steps_per_epoch: int,
    *,
    course_epochs: float | None = 0.0,
    course_steps: int | None = None,
) -> float:
    """Specificity is independent of the CE curriculum, active from update one."""
    route1_curriculum_value(
        optimizer_step,
        optimizer_steps_per_epoch,
        course_epochs=course_epochs,
        course_steps=course_steps,
    )
    return 1.0


def route1_curriculum_cut(
    cot_token_count: int,
    *,
    curriculum: float,
    optimizer_step: int,
    prompt_group_key: str | None = None,
    prompt_ids: tuple[int, ...] | list[int] = (),
    seed: int = 42,
) -> int:
    """Apply deterministic stochastic rounding shared by one prompt's views."""

    if (
        isinstance(cot_token_count, bool)
        or not isinstance(cot_token_count, int)
        or cot_token_count < 0
        or isinstance(optimizer_step, bool)
        or not isinstance(optimizer_step, int)
        or optimizer_step < 0
    ):
        raise ValueError("Route1 curriculum cut coordinates are invalid")
    c = float(curriculum)
    identity = str(prompt_group_key or tuple(int(token) for token in prompt_ids))
    if not math.isfinite(c) or not 0.0 <= c <= 1.0 or not identity:
        raise ValueError("Route1 curriculum value/prompt identity is invalid")
    expected = c * cot_token_count
    lower = math.floor(expected)
    fraction = expected - lower
    if fraction == 0.0:
        return int(lower)
    # Preserve the established stochastic-rounding domain byte-for-byte so
    # release naming does not change any curriculum cut.
    domain = bytes.fromhex("7631372d636f757273657c")
    digest = hashlib.sha256(
        domain + f"{identity}|{optimizer_step}".encode("utf-8")
    ).digest()
    uniform = int.from_bytes(digest[:8], "big") / float(2**64)
    return int(lower + (uniform < fraction))
