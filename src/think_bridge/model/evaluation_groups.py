"""Stable R evaluation groups, formed before rank or answer-request splitting."""

from __future__ import annotations
from collections.abc import Callable, Sequence

GROUP_SIZE = 8


def evaluation_group_indices(
    count: int, *, rank: int = 0, world_size: int = 1, group_size: int = GROUP_SIZE
) -> list[list[int]]:
    if any(
        isinstance(v, bool) or not isinstance(v, int)
        for v in (count, rank, world_size, group_size)
    ):
        raise ValueError("Evaluation group geometry must be integer")
    if count < 0 or world_size < 1 or not 0 <= rank < world_size or group_size < 1:
        raise ValueError("Invalid evaluation group geometry")
    return [
        list(range(start, min(start + group_size, count)))
        for group, start in enumerate(range(0, count, group_size))
        if group % world_size == rank
    ]


class FixedReasonerGroups:
    """CPU z cache scoped to one model snapshot and one ordered prompt universe.

    Duplicate token-identical prompts use their first occurrence's group. Extra
    control donors must use a separate universe; they never change anchor groups.
    """

    def __init__(
        self,
        prompts: Sequence[Sequence[int]],
        reason: Callable,
        *,
        group_size=GROUP_SIZE,
        observe=None,
    ):
        self.prompts = [tuple(int(t) for t in p) for p in prompts]
        if any(not p for p in self.prompts):
            raise ValueError("Fixed R groups require nonempty prompts")
        self.groups = evaluation_group_indices(len(self.prompts), group_size=group_size)
        self.size = group_size
        self.reason = reason
        self.observe = observe
        self.cache = {}
        self.first = {}
        for i, p in enumerate(self.prompts):
            self.first.setdefault(p, i)

    def select(self, indices):
        requested = list(indices)
        if any(
            isinstance(i, bool)
            or not isinstance(i, int)
            or not 0 <= i < len(self.prompts)
            for i in requested
        ):
            raise ValueError("Evaluation index outside fixed prompt universe")
        canonical = [self.first[self.prompts[i]] for i in requested]
        for group_id in sorted(
            {i // self.size for i in canonical if i not in self.cache}
        ):
            group = self.groups[group_id]
            prompts = [self.prompts[i] for i in group]
            outputs = (
                self.reason(prompts)
                if self.observe is None
                else self.observe(group, prompts, self.reason)
            )
            if len(outputs) != len(group):
                raise RuntimeError("R lost rows in a fixed evaluation group")
            for i, z in zip(group, outputs):
                self.cache[i] = z.detach().cpu()
        return [self.cache[i] for i in canonical]

    def lookup(self, prompts):
        return self.select([self.first[tuple(p)] for p in prompts])
