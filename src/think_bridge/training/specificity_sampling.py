"""Uniform optimizer-window Specificity donors, independent of training RNG.

One local RNG is initialized with the passed seed at each epoch. All ranks
visit the same global batches and canonical record order. Resuming replays
preceding batches of that epoch; no seed is derived from rank/step/question.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Mapping, Sequence

from think_bridge.model.contract import legal_route1_wrong_donor

SPECIFICITY_SAMPLING_POLICY = "uniform-in-optimizer-window-without-replacement"


def validate_donor_limit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            "specificity_donors_per_owner must be an integer >= 0 (0 means all)"
        )
    return value


@dataclass(frozen=True)
class SpecificityDonorSelection:
    donors: dict[str, tuple[str, ...]]
    eligible_pair_count: int
    selected_pair_count: int


class SpecificityDonorSampler:
    def __init__(self, *, seed: int, donors_per_owner: int):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("specificity sampler seed must be an integer")
        self.limit = validate_donor_limit(donors_per_owner)
        self.rng = random.Random(seed)

    def sample(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        enabled: bool = True,
        active_populations: Sequence[str] = ("C",),
        owner_populations: Sequence[str] | None = None,
    ) -> SpecificityDonorSelection:
        by_id = {str(row["record_id"]): row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError("specificity donor batch contains duplicate records")
        ordered = sorted(by_id)
        selected: dict[str, tuple[str, ...]] = {}
        eligible_count = 0
        for owner_id in ordered:
            eligible = [
                donor_id
                for donor_id in ordered
                if (
                    owner_populations is None
                    or by_id[owner_id].get("quadrant") in owner_populations
                )
                and legal_route1_wrong_donor(
                    by_id[owner_id],
                    by_id[donor_id],
                    enabled=enabled,
                    active_populations=active_populations,
                )
            ]
            eligible_count += len(eligible)
            if self.limit and len(eligible) > self.limit:
                chosen = self.rng.sample(eligible, self.limit)
            else:
                chosen = eligible
            # Stable execution order; membership is sampled uniformly.
            selected[owner_id] = tuple(sorted(chosen))
        return SpecificityDonorSelection(
            selected, eligible_count, sum(map(len, selected.values()))
        )
