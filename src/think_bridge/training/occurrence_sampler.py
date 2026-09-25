"""Deterministic occurrence-uniform epoch planning for ThinkBridge.

Route1 shuffles record identities without length ordering or prompt uniqueness.
Occurrence planning retains records first, then packs unique prompt groups
and orders physical batches by length.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Sequence, Tuple

from think_bridge.model.artifact_schema import (
    OCCURRENCE_SAMPLER_STATE,
    artifact_header,
)


LengthKey = Tuple[int, ...]


@dataclass(frozen=True)
class Route1OccurrenceCost:
    """Deterministic estimate of one Route1 occurrence's physical work."""

    answer_course_tokens: int
    stopped_rollout_tokens: int

    def __post_init__(self) -> None:
        values = (
            self.answer_course_tokens,
            self.stopped_rollout_tokens,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise ValueError("Route1 occurrence costs must be non-negative integers")

    @property
    def base_tokens(self) -> int:
        return int(self.answer_course_tokens + self.stopped_rollout_tokens)


@dataclass(frozen=True)
class CostBalancedRankAssignment:
    rank_microsteps: tuple[tuple[tuple[str, ...], ...], ...]
    rank_costs: tuple[int, ...]
    assignment_sha256: str


@dataclass(frozen=True)
class OccurrenceBatch:
    record_ids: tuple[str, ...]
    prompt_group_ids: tuple[str, ...]
    length_keys: tuple[LengthKey, ...]


@dataclass(frozen=True)
class OccurrenceEpochPlan:
    batches: tuple[OccurrenceBatch, ...]
    retained_record_ids: tuple[str, ...]
    dropped_record_ids: tuple[str, ...]
    retained_record_ids_sha256: str
    dropped_record_ids_sha256: str
    batch_order_sha256: str


class OccurrenceBatchSampler:
    """Standard BatchSampler-style rank shard with exact cursor replay.

    With shuffle_records, planning shuffles records and allows repeated prompts.
    The generic planner packs unique prompt groups before physical length ordering.
    """

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        seed: int,
        global_batch_size: int,
        rank: int,
        world_size: int,
        length_key: Callable[[Mapping[str, Any]], Sequence[int]],
        rank_assignment: Callable[
            [int, OccurrenceBatch, Mapping[str, Mapping[str, Any]]],
            CostBalancedRankAssignment,
        ]
        | None = None,
        shuffle_records: bool = False,
        drop_last: bool = True,
        complete_batch_multiple: int = 1,
    ) -> None:
        if not rows:
            raise ValueError("occurrence BatchSampler requires non-empty rows")
        if (
            isinstance(rank, bool)
            or isinstance(world_size, bool)
            or not isinstance(rank, int)
            or not isinstance(world_size, int)
            or world_size <= 0
            or rank < 0
            or rank >= world_size
        ):
            raise ValueError("occurrence BatchSampler rank/world_size is invalid")
        if int(global_batch_size) % int(world_size) != 0:
            raise ValueError("global occurrence batch must shard evenly across ranks")
        if not isinstance(drop_last, bool):
            raise ValueError("occurrence BatchSampler drop_last must be boolean")
        if (
            isinstance(complete_batch_multiple, bool)
            or not isinstance(complete_batch_multiple, int)
            or complete_batch_multiple <= 0
        ):
            raise ValueError("complete_batch_multiple must be a positive integer")
        self.rows = tuple(rows)
        self.seed = int(seed)
        self.shuffle_records = bool(shuffle_records)
        self.global_batch_size = int(global_batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.length_key = length_key
        self.rank_assignment = rank_assignment
        self.drop_last = drop_last
        self.complete_batch_multiple = int(complete_batch_multiple)
        self.epoch = 0
        self.cursor = 0
        self._record_index = {
            str(row.get("record_id", "")): index for index, row in enumerate(self.rows)
        }
        if len(self._record_index) != len(self.rows) or "" in self._record_index:
            raise ValueError("occurrence BatchSampler requires unique record ids")
        self._plan: OccurrenceEpochPlan | None = None

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("occurrence BatchSampler epoch must be non-negative")
        self.epoch = int(epoch)
        self.cursor = 0
        self._plan = None

    def set_cursor(self, cursor: int) -> None:
        """Position iteration within the current epoch plan without serialization."""

        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise ValueError(
                "occurrence BatchSampler cursor must be a non-negative integer"
            )
        if cursor > len(self.plan.batches):
            raise ValueError("occurrence BatchSampler cursor is outside the epoch")
        self.cursor = cursor

    @property
    def plan(self) -> OccurrenceEpochPlan:
        if self._plan is None:
            self._plan = plan_occurrence_epoch(
                self.rows,
                epoch=self.epoch,
                seed=self.seed,
                global_batch_size=self.global_batch_size,
                length_key=self.length_key,
                drop_last=self.drop_last,
                complete_batch_multiple=self.complete_batch_multiple,
                shuffle_records=self.shuffle_records,
            )
            if any(
                len(batch.record_ids) < self.world_size for batch in self._plan.batches
            ):
                raise ValueError(
                    "occurrence partial batch cannot provide one row to every rank"
                )
        return self._plan

    def __len__(self) -> int:
        return len(self.plan.batches)

    def __iter__(self) -> Iterator[list[int]]:
        for batch_index in range(self.cursor, len(self.plan.batches)):
            batch = self.plan.batches[batch_index]
            if self.rank_assignment is None:
                batch_size = len(batch.record_ids)
                start = batch_size * self.rank // self.world_size
                stop = batch_size * (self.rank + 1) // self.world_size
                selected = batch.record_ids[start:stop]
            else:
                rows_by_id = {
                    record_id: self.rows[self._record_index[record_id]]
                    for record_id in batch.record_ids
                }
                assignment = self.rank_assignment(batch_index, batch, rows_by_id)
                selected = tuple(
                    record_id
                    for microstep in assignment.rank_microsteps[self.rank]
                    for record_id in microstep
                )
                flattened = tuple(
                    record_id
                    for rank_microsteps in assignment.rank_microsteps
                    for microstep in rank_microsteps
                    for record_id in microstep
                )
                if len(flattened) != len(set(flattened)) or set(flattened) != set(
                    batch.record_ids
                ):
                    raise RuntimeError(
                        "cost-balanced occurrence shard changed batch membership"
                    )
            if not selected:
                raise RuntimeError(
                    "occurrence batch produced an empty distributed rank shard"
                )
            self.cursor = batch_index + 1
            yield [self._record_index[record_id] for record_id in selected]

    def state_dict(self, *, cursor: int | None = None) -> dict[str, Any]:
        resolved_cursor = self.cursor if cursor is None else int(cursor)
        if resolved_cursor < 0 or resolved_cursor > len(self):
            raise ValueError("occurrence BatchSampler cursor is outside the epoch")
        return {
            **artifact_header(OCCURRENCE_SAMPLER_STATE),
            "seed": self.seed,
            "epoch": self.epoch,
            "cursor": resolved_cursor,
            "rank": self.rank,
            "world_size": self.world_size,
            "global_batch_size": self.global_batch_size,
            "drop_last": self.drop_last,
            "complete_batch_multiple": self.complete_batch_multiple,
            "retained_record_ids_sha256": self.plan.retained_record_ids_sha256,
            "dropped_record_ids_sha256": self.plan.dropped_record_ids_sha256,
            "batch_order_sha256": self.plan.batch_order_sha256,
            "rank_assignment_sha256": self._rank_assignment_sha256(),
        }

    def _rank_assignment_sha256(self) -> str:
        if self.rank_assignment is None:
            return _identity_sha256(())
        values = []
        for batch_index, batch in enumerate(self.plan.batches):
            rows_by_id = {
                record_id: self.rows[self._record_index[record_id]]
                for record_id in batch.record_ids
            }
            values.append(
                self.rank_assignment(batch_index, batch, rows_by_id).assignment_sha256
            )
        return _identity_sha256(values)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            **artifact_header(OCCURRENCE_SAMPLER_STATE),
            "seed": self.seed,
            "rank": self.rank,
            "world_size": self.world_size,
            "global_batch_size": self.global_batch_size,
            "drop_last": self.drop_last,
            "complete_batch_multiple": self.complete_batch_multiple,
        }
        mismatch = [key for key, value in expected.items() if state.get(key) != value]
        if mismatch:
            raise ValueError(f"occurrence BatchSampler identity mismatch: {mismatch}")
        self.set_epoch(int(state["epoch"]))
        observed = self.state_dict(cursor=int(state["cursor"]))
        hashes = (
            "retained_record_ids_sha256",
            "dropped_record_ids_sha256",
            "batch_order_sha256",
            "rank_assignment_sha256",
        )
        if any(observed[key] != state.get(key) for key in hashes):
            raise ValueError("occurrence BatchSampler plan differs on resume")
        self.cursor = int(state["cursor"])


def _identity_sha256(values: Sequence[str]) -> str:
    payload = json.dumps(
        list(values),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stable_digest(*, domain: str, seed: int, epoch: int, identity: str) -> str:
    payload = f"{domain}\x1f{seed}\x1f{epoch}\x1f{identity}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def occurrence_steps_per_epoch(
    occurrence_count: int,
    *,
    global_batch_size: int,
    drop_last: bool = True,
) -> int:
    if isinstance(occurrence_count, bool) or int(occurrence_count) < 0:
        raise ValueError("occurrence_count must be a non-negative integer")
    if isinstance(global_batch_size, bool) or int(global_batch_size) <= 0:
        raise ValueError("global_batch_size must be a positive integer")
    if not isinstance(drop_last, bool):
        raise ValueError("drop_last must be boolean")
    if drop_last:
        return int(occurrence_count) // int(global_batch_size)
    return math.ceil(int(occurrence_count) / int(global_batch_size))


def _normalize_length_key(value: Sequence[int]) -> LengthKey:
    key = tuple(value)
    if not key or any(
        isinstance(item, bool) or not isinstance(item, int) for item in key
    ):
        raise ValueError("length_key must return a non-empty integer sequence")
    return key


def plan_occurrence_epoch(
    rows: Sequence[Mapping[str, Any]],
    *,
    epoch: int,
    seed: int,
    global_batch_size: int,
    length_key: Callable[[Mapping[str, Any]], Sequence[int]],
    drop_last: bool = True,
    complete_batch_multiple: int = 1,
    shuffle_records: bool = False,
) -> OccurrenceEpochPlan:
    """Plan deterministic record batches.

    Route1 uses ``shuffle_records``: shuffle canonical record ids by seed+epoch,
    then slice consecutive batches. Length keys are telemetry only. The default
    supports unique-group/length planning.
    """

    if isinstance(epoch, bool) or int(epoch) < 0:
        raise ValueError("epoch must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if (
        isinstance(complete_batch_multiple, bool)
        or not isinstance(complete_batch_multiple, int)
        or complete_batch_multiple <= 0
    ):
        raise ValueError("complete_batch_multiple must be a positive integer")
    steps = occurrence_steps_per_epoch(
        len(rows),
        global_batch_size=global_batch_size,
        drop_last=drop_last,
    )
    if drop_last:
        steps -= steps % int(complete_batch_multiple)
    if steps <= 0:
        raise ValueError(
            "occurrence epoch must contain at least one complete global batch"
        )

    rows_by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        record_id = str(row.get("record_id", "")).strip()
        group_id = str(row.get("prompt_group_id", "")).strip()
        if not record_id or not group_id:
            raise ValueError(
                "occurrence rows require non-empty record_id and prompt_group_id"
            )
        if record_id in rows_by_id:
            raise ValueError(f"duplicate occurrence record_id: {record_id}")
        rows_by_id[record_id] = row

    if shuffle_records:
        # Canonical record order makes the permutation independent of JSON order.
        ids = sorted(rows_by_id)
        random.Random(seed + epoch).shuffle(ids)
        count = steps * int(global_batch_size) if drop_last else len(ids)
        retained, dropped = ids[:count], ids[count:]
        batches = tuple(
            OccurrenceBatch(
                record_ids=tuple(part),
                prompt_group_ids=tuple(
                    str(rows_by_id[k]["prompt_group_id"]) for k in part
                ),
                length_keys=tuple(
                    _normalize_length_key(length_key(rows_by_id[k])) for k in part
                ),
            )
            for part in (
                retained[i : i + global_batch_size]
                for i in range(0, count, global_batch_size)
            )
        )
        return OccurrenceEpochPlan(
            batches=batches,
            retained_record_ids=tuple(sorted(retained)),
            dropped_record_ids=tuple(sorted(dropped)),
            retained_record_ids_sha256=_identity_sha256(sorted(retained)),
            dropped_record_ids_sha256=_identity_sha256(sorted(dropped)),
            batch_order_sha256=_identity_sha256(
                f"{i}:{k}" for i, b in enumerate(batches) for k in b.record_ids
            ),
        )

    ranked_ids = sorted(
        rows_by_id,
        key=lambda record_id: (
            _stable_digest(
                domain="bridge-occurrence-tail-v1",
                seed=seed,
                epoch=epoch,
                identity=record_id,
            ),
            record_id,
        ),
    )
    retained_count = steps * int(global_batch_size) if drop_last else len(ranked_ids)
    retained_ids = ranked_ids[:retained_count]
    dropped_ids = ranked_ids[retained_count:]

    grouped_ids: dict[str, list[str]] = {}
    normalized_lengths: dict[str, LengthKey] = {}
    for record_id in retained_ids:
        row = rows_by_id[record_id]
        group_id = str(row["prompt_group_id"])
        grouped_ids.setdefault(group_id, []).append(record_id)
        normalized_lengths[record_id] = _normalize_length_key(length_key(row))

    impossible = sorted(
        group_id
        for group_id, record_ids in grouped_ids.items()
        if len(record_ids) > steps
    )
    if impossible:
        raise ValueError(
            "unique-group packing is impossible: retained occurrences for a prompt "
            f"group exceed global batches: {impossible[:3]}"
        )

    bins: list[list[str]] = [[] for _ in range(steps)]
    capacities = [int(global_batch_size)] * steps
    if not drop_last and retained_count % int(global_batch_size):
        capacities[-1] = retained_count % int(global_batch_size)
    group_order = sorted(
        grouped_ids,
        key=lambda group_id: (
            -len(grouped_ids[group_id]),
            _stable_digest(
                domain="bridge-occurrence-group-order-v1",
                seed=seed,
                epoch=epoch,
                identity=group_id,
            ),
            group_id,
        ),
    )
    for group_id in group_order:
        record_ids = sorted(
            grouped_ids[group_id],
            key=lambda record_id: (
                normalized_lengths[record_id],
                _stable_digest(
                    domain="bridge-occurrence-record-order-v1",
                    seed=seed,
                    epoch=epoch,
                    identity=record_id,
                ),
                record_id,
            ),
        )
        if drop_last:
            candidate_bins = sorted(
                range(steps),
                key=lambda batch_index: (
                    len(bins[batch_index]),
                    _stable_digest(
                        domain="bridge-occurrence-bin-choice-v1",
                        seed=seed,
                        epoch=epoch,
                        identity=f"{group_id}\x1f{batch_index}",
                    ),
                    batch_index,
                ),
            )[: len(record_ids)]
        else:
            candidate_bins = sorted(
                (
                    batch_index
                    for batch_index in range(steps)
                    if len(bins[batch_index]) < capacities[batch_index]
                ),
                key=lambda batch_index: (
                    -(capacities[batch_index] - len(bins[batch_index])),
                    _stable_digest(
                        domain="bridge-occurrence-partial-bin-choice-v1",
                        seed=seed,
                        epoch=epoch,
                        identity=f"{group_id}\x1f{batch_index}",
                    ),
                    batch_index,
                ),
            )[: len(record_ids)]
            if len(candidate_bins) != len(record_ids):
                raise ValueError(
                    "unique-group packing cannot fill the partial-batch capacities"
                )
        candidate_bins.sort(
            key=lambda batch_index: (
                tuple(normalized_lengths[item] for item in bins[batch_index]),
                batch_index,
            )
        )
        for record_id, batch_index in zip(record_ids, candidate_bins):
            bins[batch_index].append(record_id)

    if any(len(items) != capacity for items, capacity in zip(bins, capacities)):
        raise AssertionError(
            "unique-group packing did not produce complete global batches"
        )

    ordered_bins: list[tuple[tuple[Any, ...], list[str]]] = []
    for batch_index, record_ids in enumerate(bins):
        ordered_record_ids = sorted(
            record_ids,
            key=lambda record_id: (
                normalized_lengths[record_id],
                _stable_digest(
                    domain="bridge-occurrence-physical-order-v1",
                    seed=seed,
                    epoch=epoch,
                    identity=record_id,
                ),
                record_id,
            ),
        )
        group_ids = [
            str(rows_by_id[record_id]["prompt_group_id"])
            for record_id in ordered_record_ids
        ]
        if len(set(group_ids)) != len(group_ids):
            raise AssertionError(
                "global occurrence batch contains duplicate prompt groups"
            )
        length_signature = tuple(
            normalized_lengths[record_id] for record_id in ordered_record_ids
        )
        ordinary_order_key: tuple[Any, ...] = (
            length_signature,
            _stable_digest(
                domain="bridge-occurrence-batch-order-v1",
                seed=seed,
                epoch=epoch,
                identity=str(batch_index),
            ),
            batch_index,
        )
        order_key = (
            ordinary_order_key
            if drop_last
            else (
                len(record_ids) != int(global_batch_size),
                *ordinary_order_key,
            )
        )
        ordered_bins.append((order_key, ordered_record_ids))
    ordered_bins.sort(key=lambda item: item[0])

    batches = tuple(
        OccurrenceBatch(
            record_ids=tuple(record_ids),
            prompt_group_ids=tuple(
                str(rows_by_id[record_id]["prompt_group_id"])
                for record_id in record_ids
            ),
            length_keys=tuple(
                normalized_lengths[record_id] for record_id in record_ids
            ),
        )
        for _order_key, record_ids in ordered_bins
    )
    flattened = [record_id for batch in batches for record_id in batch.record_ids]
    if len(flattened) != retained_count or len(set(flattened)) != retained_count:
        raise AssertionError("retained occurrence coverage is not exact")
    if set(flattened) != set(retained_ids):
        raise AssertionError("batch packing changed the retained occurrence set")

    retained_identity = tuple(sorted(retained_ids))
    dropped_identity = tuple(sorted(dropped_ids))
    batch_identity = tuple(
        f"{batch_index}:{record_id}"
        for batch_index, batch in enumerate(batches)
        for record_id in batch.record_ids
    )
    return OccurrenceEpochPlan(
        batches=batches,
        retained_record_ids=retained_identity,
        dropped_record_ids=dropped_identity,
        retained_record_ids_sha256=_identity_sha256(retained_identity),
        dropped_record_ids_sha256=_identity_sha256(dropped_identity),
        batch_order_sha256=_identity_sha256(batch_identity),
    )


from think_bridge.model.contract import (
    OCCURRENCE_WEIGHTING_UNIT as OCCURRENCE_WEIGHTING_UNIT,
)


from think_bridge.model.contract import (
    ROUTE1_OCCURRENCE_SAMPLER_SCHEMA as ROUTE1_OCCURRENCE_SAMPLER_SCHEMA,
)

from think_bridge.model.contract import OCCURRENCE_BATCH_UNIT as OCCURRENCE_BATCH_UNIT
