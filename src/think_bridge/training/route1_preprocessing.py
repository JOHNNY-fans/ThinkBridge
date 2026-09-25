"""One-process immutable-input reuse for Bridge Route1 preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any, Mapping, Sequence

from think_bridge.model.contract import normalize_route1_sample_rows
from think_bridge.training.objective_window_plans import (
    complete_optimizer_window_count,
    route1_active_epoch_rows,
)


@dataclass(frozen=True)
class Route1NormalizedRowCache:
    """Bind raw target identities to one normalized per-process row set.

    The cache owns the normalized dictionaries for the lifetime of one worker.
    Training consumers may read them or copy individual rows, but must not
    mutate them in place.  ``source_rows`` is retained because Joint-SFT must
    validate the original native/direct target population after Route1 replaces
    that field with its B/C/D course population.
    """

    population: str
    source_rows: tuple[Mapping[str, Any], ...]
    rows: tuple[Mapping[str, Any], ...]
    _active_rows_by_identity: dict[
        tuple[int, bool, bool, bool], tuple[Mapping[str, Any], ...]
    ] = field(default_factory=dict, init=False, repr=False, compare=False)
    _active_rows_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
        compare=False,
    )

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Mapping[str, Any]],
        *,
        population: str,
    ) -> "Route1NormalizedRowCache":
        source_rows = tuple(rows)
        normalized_rows = tuple(
            normalize_route1_sample_rows(source_rows, population=population)
        )
        if len(source_rows) != len(normalized_rows):
            raise RuntimeError("Route1 normalization changed occurrence cardinality")
        source_ids = tuple(str(row.get("record_id", "")) for row in source_rows)
        normalized_ids = tuple(str(row.get("record_id", "")) for row in normalized_rows)
        if (
            not source_ids
            or source_ids != normalized_ids
            or "" in source_ids
            or len(set(source_ids)) != len(source_ids)
        ):
            raise RuntimeError("Route1 normalized/raw occurrence identity drifted")
        return cls(
            population=str(population),
            source_rows=source_rows,
            rows=normalized_rows,
        )

    def active_epoch_rows(
        self,
        *,
        epoch: int,
        compute_course: bool,
        compute_match: bool,
        compute_specific: bool,
    ) -> tuple[Mapping[str, Any], ...]:
        """Reuse one exact active-domain projection without copying rows."""

        identity = (
            int(epoch),
            bool(compute_course),
            bool(compute_match),
            bool(compute_specific),
        )
        with self._active_rows_lock:
            cached = self._active_rows_by_identity.get(identity)
            if cached is None:
                cached = tuple(
                    route1_active_epoch_rows(
                        self.rows,
                        population=self.population,
                        epoch=int(epoch),
                        compute_course=compute_course,
                        compute_match=compute_match,
                        compute_specific=compute_specific,
                    )
                )
                self._active_rows_by_identity[identity] = cached
            return cached


def resolve_route1_normalized_rows(
    rows: Sequence[Mapping[str, Any]] | Route1NormalizedRowCache,
    *,
    population: str,
) -> Route1NormalizedRowCache:
    """Return an existing compatible cache or normalize raw rows once."""

    if isinstance(rows, Route1NormalizedRowCache):
        if rows.population != str(population):
            raise ValueError(
                "Route1 normalized-row cache population differs from request"
            )
        return rows
    return Route1NormalizedRowCache.from_rows(rows, population=population)


def exact_route1_optimizer_steps(
    active_occurrence_count: int,
    *,
    local_samples: int,
    world_size: int,
    gradient_accumulation_steps: int,
) -> int:
    """Count retained optimizer windows without constructing an epoch plan."""

    return complete_optimizer_window_count(
        int(active_occurrence_count),
        local_microbatch=int(local_samples),
        world_size=int(world_size),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
    )
