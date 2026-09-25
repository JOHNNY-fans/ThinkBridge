"""Fail-closed, dependency-free contracts for the isolated ThinkBridge pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping, Sequence

from think_bridge.model.artifact_schema import (
    STAGE1_HARD_DONOR_MANIFEST,
    STAGE1_SHARED_MANIFEST,
    artifact_header,
    require_artifact_header,
)


OBJECTIVE_VERSION = "answer-ce-bc-forward-kl-same-prompt-specificity"
SPECIFICITY_OBJECTIVE_VERSION = "same-prompt-capped-soft-infonce-forward-kl"
CHECKPOINT_SCHEMA_VERSION = 1
COURSE_SCHEMA_VERSION = "answer-eos-ce-no-curriculum"
CAUSAL_OBJECTIVE_SCHEMA_VERSION = (
    "answer-ce-bc-fkl-live-same-prompt-specificity-no-direct"
)
GEOMETRY_SCHEMA_VERSION = "recursive-t1-b64-k64-answer-only"
ROUTE1_VALIDATION_REPORT_SCHEMA_VERSION = "think-bridge.evaluation.route1-report"
KZ = 64
ROUTE1_NULL_MODES = ("direct",)
ROUTE1_STANDALONE_NULL_MODES = ("direct", "zero64")
ALIGNMENT_CAPACITY = 6144
COT_CONTENT_CAPACITY = 6144
ANSWER_CAPACITY = 2048
NEED_Z_COHORT_DEFINITION = (
    "sealed-native-correct-and-not-sealed-direct-no-think-correct-v1"
)
D_ANSWER_PROVENANCE_SCHEMA = "sealed-d-answer-fallback-provenance"
D_ANSWER_EOS_RULE = "accept-hit-eos-only-and-append-observed-terminal-eos"
TARGET_FIELDS = (
    "record_id",
    "prompt_group_id",
    "problem_id",
    "semantic_group_id",
    "self_cot_id",
    "self_cot_source_rollout_id",
    "provenance_valid",
    "prompt_ids",
    "direct_prompt_ids",
    "cot_ids",
    "readout_cot_ids",
    "route2_eligible",
    "answer_ids",
    "answer_source",
    "split_id",
    "population",
    "n_correct",
    "reference_answer",
    "task_type",
    "need_z",
    "direct_correct",
    "locked_full_correct",
    "need_z_definition",
)
ROUTE1_OCCURRENCE_SAMPLER_SCHEMA = "think-bridge.training.route1-sampler"
OCCURRENCE_WEIGHTING_UNIT = "one-stage0-trajectory-sample-one-answer-view"
OCCURRENCE_BATCH_UNIT = "seeded-shuffled-records-per-distributed-microbatch"


def normalize_route1_null_mode(value: Any) -> str:
    """Return one closed Route1 null geometry without accepting aliases."""

    if not isinstance(value, str) or value not in ROUTE1_NULL_MODES:
        raise ValueError(
            "Route1 null mode must be exactly one of: " + ", ".join(ROUTE1_NULL_MODES)
        )
    return value


def normalize_route1_standalone_null_mode(value: Any) -> str:
    """Allow zero64 only for an explicitly selected standalone diagnostic."""

    if not isinstance(value, str) or value not in ROUTE1_STANDALONE_NULL_MODES:
        raise ValueError(
            "standalone Route1 null mode must be exactly one of: "
            + ", ".join(ROUTE1_STANDALONE_NULL_MODES)
        )
    return value


ROUTE1_POPULATIONS = ("B", "C", "D", "E")
ROUTE1_EVALUATION_SUBSET_PROTOCOL = (
    "bridge-route1-validation-population-stratified-stable-hash-v1"
)
ROUTE1_DIRECT_BASELINE_SCHEMA = (
    "bridge-route1-sealed-validation-behavior-direct-baseline-v1"
)
ROUTE1_DIRECT_BASELINE_MAX_TOKENS = 2048
ROUTE1_SELECTOR_WRONG_CONDITIONS = tuple(f"wrong_z_{index}" for index in range(1, 9))
ROUTE1_COURSE_REDUCTIONS = ("post_epoch0_c_vs_b_d_balanced",)


def route1_population(native_correct: Any, direct_correct: Any) -> str:
    """Map the two sealed correctness labels to exactly one B/C/D/E quadrant."""

    if not isinstance(native_correct, bool) or not isinstance(direct_correct, bool):
        raise TypeError("Route1 population labels must be booleans")
    return {
        (True, True): "B",
        (True, False): "C",
        (False, True): "D",
        (False, False): "E",
    }[(native_correct, direct_correct)]


def route1_selector_generation_conditions(
    native_correct: Any,
    direct_correct: Any,
    *,
    include_controls: bool = True,
    donor_count: int = 8,
) -> tuple[str, ...]:
    """Return only checkpoint-dependent Route1 selector decode conditions."""

    population = route1_population(native_correct, direct_correct)
    if population == "C" and include_controls:
        return ("true_z", *ROUTE1_SELECTOR_WRONG_CONDITIONS[:donor_count])
    return ("true_z",)


def route1_selector_request_counts(
    rows: Sequence[Mapping[str, Any]],
    *,
    include_controls: bool = True,
    donors_by_anchor: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Count the exact checkpoint-dependent sparse selector decodes."""

    true_z = len(rows)
    c_prompt_count = sum(
        route1_population(row.get("locked_full_correct"), row.get("direct_correct"))
        == "C"
        for row in rows
    )
    donors = len(ROUTE1_SELECTOR_WRONG_CONDITIONS) if include_controls else 0
    counts = [
        (
            len(donors_by_anchor.get(str(row["record_id"]), ()))
            if donors_by_anchor is not None
            else donors
        )
        if include_controls
        else 0
        for row in rows
        if route1_population(row.get("locked_full_correct"), row.get("direct_correct"))
        == "C"
    ]
    wrong_z = sum(counts)
    return {
        "protocol": "bridge-route1-sparse-selector-request-count",
        "true_z": true_z,
        "direct": 0,
        "wrong_z": wrong_z,
        "c_prompt_count": c_prompt_count,
        "requested_wrong_z_per_c_prompt": donors,
        "wrong_z_per_c_prompt": counts[0] if counts and len(set(counts)) == 1 else None,
        "wrong_z_per_c_prompt_counts": counts,
        "total": true_z + wrong_z,
    }


def select_route1_evaluation_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_eval_samples: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select one deterministic population-stratified validation domain.

    The full sealed order is preserved when the limit is unset.  A positive
    limit ranks rows within population by a stable record-id hash, reserves the
    up to two available C prompts for the paired bootstrap, then covers B/D/E when the
    budget permits and fills the remainder by deterministic population rounds.
    Selected rows are returned in their original sealed order before sharding.
    """

    normalized = [dict(row) for row in rows]
    if not normalized:
        raise ValueError("Route1 evaluation rows are empty")
    by_population: dict[str, list[tuple[int, dict[str, Any]]]] = {
        population: [] for population in ROUTE1_POPULATIONS
    }
    record_ids: set[str] = set()
    for index, row in enumerate(normalized):
        record_id = str(row.get("record_id", ""))
        if not record_id or record_id in record_ids:
            raise ValueError("Route1 evaluation record ids are empty or duplicated")
        record_ids.add(record_id)
        population = route1_population(
            row.get("locked_full_correct"), row.get("direct_correct")
        )
        by_population[population].append((index, row))
    if max_eval_samples is not None and (
        isinstance(max_eval_samples, bool)
        or not isinstance(max_eval_samples, int)
        or max_eval_samples <= 0
    ):
        raise ValueError("max_eval_samples must be a positive integer or None")
    source_count = len(normalized)
    limit = (
        source_count
        if max_eval_samples is None
        else min(int(max_eval_samples), source_count)
    )

    def stable_rank(item: tuple[int, dict[str, Any]]) -> tuple[str, int]:
        index, row = item
        digest = hashlib.sha256(
            (ROUTE1_EVALUATION_SUBSET_PROTOCOL + "\x1f" + str(row["record_id"])).encode(
                "utf-8"
            )
        ).hexdigest()
        return digest, index

    ranked = {
        population: sorted(values, key=stable_rank)
        for population, values in by_population.items()
    }
    selected_indices: set[int] = set()

    def take(population: str, count: int) -> None:
        for index, _row in ranked[population]:
            if len(selected_indices) >= limit or count <= 0:
                return
            if index not in selected_indices:
                selected_indices.add(index)
                count -= 1

    take("C", 2)
    for population in ("B", "D", "E"):
        take(population, 1)
    cursors = {population: 0 for population in ROUTE1_POPULATIONS}
    while len(selected_indices) < limit:
        before = len(selected_indices)
        for population in ROUTE1_POPULATIONS:
            values = ranked[population]
            while (
                cursors[population] < len(values)
                and values[cursors[population]][0] in selected_indices
            ):
                cursors[population] += 1
            if cursors[population] < len(values):
                selected_indices.add(values[cursors[population]][0])
                cursors[population] += 1
                if len(selected_indices) == limit:
                    break
        if len(selected_indices) == before:
            raise RuntimeError("Route1 evaluation subset allocator stalled")
    selected = [
        row for index, row in enumerate(normalized) if index in selected_indices
    ]
    selected_domain = [
        {
            "record_id": str(row["record_id"]),
            "prompt_group_id": str(row["prompt_group_id"]),
            "population": route1_population(
                row["locked_full_correct"], row["direct_correct"]
            ),
        }
        for row in selected
    ]
    denominators = {
        population: sum(item["population"] == population for item in selected_domain)
        for population in ROUTE1_POPULATIONS
    }
    identity = {
        "protocol": ROUTE1_EVALUATION_SUBSET_PROTOCOL,
        "max_eval_samples": max_eval_samples,
        "source_sample_count": source_count,
        "actual_sample_count": len(selected),
        "population_denominators": denominators,
        "evaluation_subset_sha256": canonical_json_sha256(selected_domain),
    }
    return selected, identity


def route1_direct_baseline_identity(
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    answer_max_tokens: int,
) -> dict[str, Any]:
    """Describe direct-label evaluation without hashing prepared data content."""

    if (
        isinstance(answer_max_tokens, bool)
        or not isinstance(answer_max_tokens, int)
        or answer_max_tokens != ROUTE1_DIRECT_BASELINE_MAX_TOKENS
    ):
        raise ValueError(
            "Route1 direct baseline answer horizon differs from sealed behavior"
        )
    required = ("tokenizer_sha256", "template_sha256")
    references = {
        name: require_sha256(str(manifest.get(name, "")), name) for name in required
    }
    for row in rows:
        if not str(row.get("record_id", "")) or not isinstance(
            row.get("direct_correct"), bool
        ):
            raise ValueError("Route1 direct baseline row binding is invalid")
    if not rows:
        raise ValueError("Route1 direct baseline row binding is invalid")
    payload = {
        "protocol": ROUTE1_DIRECT_BASELINE_SCHEMA,
        "source": "sealed-validation-behavior-direct-correct",
        "direct_prompt": "no-think-plus-empty-think-block",
        "answer_max_tokens": int(answer_max_tokens),
        "decoder": "thinkbridge.greedy-token-decode-v1",
        "temperature": 0.0,
        "top_p": 1.0,
        "judge": "think_bridge.eval.answer_match.judge_answer",
        **references,
        "record_count": len(rows),
    }
    return {**payload, "identity_sha256": canonical_json_sha256(payload)}


def validate_bridge_behavior_count(
    n_correct: Any,
    *,
    split: str,
    locked_full_correct: Any,
) -> int:
    """Validate source-view counts without redefining generated native labels.

    Train manifests use Stage0 multi-view correctness, so the count and locked
    prompt label must agree. Validation manifests use one independent frozen-F
    generation while ``n_correct`` still counts nonexistent source views; its
    locked native label is therefore intentionally independent of this count.
    """

    if split not in {"train", "validation"}:
        raise ValueError(f"unsupported Bridge behavior split: {split}")
    if not isinstance(locked_full_correct, bool):
        raise ValueError(f"{split} locked native correctness must be boolean")
    if isinstance(n_correct, bool) or not isinstance(n_correct, int) or n_correct < 0:
        raise ValueError(f"{split} n_correct must be a nonnegative integer")
    if split == "train" and locked_full_correct is not (n_correct > 0):
        raise ValueError("train n_correct differs from locked native correctness")
    return n_correct


def route1_course_population_side(population: Any, epoch: Any) -> str | None:
    """Return the fixed course side for one B/C/D sample in every epoch.

    B retains its paired native target, D is the authorized suffix-free direct
    fallback, and both share the direct-correct population side.  C owns the
    other side and E never enters Route1 training.  Epoch changes only the
    native-prefix curriculum, never course eligibility or reduction.
    """

    if population not in ROUTE1_POPULATIONS:
        raise ValueError("Route1 course population must be B, C, D, or E")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("Route1 course epoch must be a nonnegative integer")
    if population == "C":
        return "c"
    if population in {"B", "D"}:
        return "direct_correct_side"
    return None


def route1_course_reduction(epoch: Any) -> str:
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("Route1 course epoch must be a nonnegative integer")
    return ROUTE1_COURSE_REDUCTIONS[0]


def normalize_route1_sample_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    population: str = "staged",
) -> list[dict[str, Any]]:
    """Normalize every immutable target record into one Route1 sample/view.

    Prompt identity is consulted only to validate sealed metadata and the
    single D fallback invariant.  It never combines native trajectories.
    """

    if population not in {"staged", "answer-only", "gold-reference"}:
        raise ValueError("Route1 population must be staged or answer-only")
    if population in {"answer-only", "gold-reference"}:
        gold_reference = population == "gold-reference"
        result: list[dict[str, Any]] = []
        for row in rows:
            if (
                row.get("population") != ("reference" if gold_reference else "native")
                or row.get("provenance_valid") is not True
                or any(
                    row.get(field) is not None
                    for field in (
                        "n_correct",
                        "need_z",
                        "direct_correct",
                        "locked_full_correct",
                    )
                )
                or row.get("answer_source")
                not in {
                    "explicit-answer-only-boxed-reference",
                    "explicit-answer-only-selfgen",
                    "dataset-reference-cot-and-answer",
                }
            ):
                raise ValueError(
                    "answer-only Route1 row lacks its explicit label-free contract"
                )
            prompt_group_id = str(row.get("prompt_group_id", ""))
            answer_ids = list(row.get("answer_ids", ()))
            if gold_reference and (
                row.get("answer_source") != "dataset-reference-cot-and-answer"
                or not row.get("cot_ids")
            ):
                raise ValueError(
                    "gold-reference requires paired dataset cot and answer"
                )
            if not prompt_group_id or not answer_ids:
                raise ValueError("answer-only Route1 row lacks prompt/answer tokens")
            result.append(
                {
                    **dict(row),
                    # A is an execution-only course population.  It is not a
                    # fabricated B/C/D label and never activates need-z leaves.
                    "quadrant": "G" if gold_reference else "A",
                    "population": "G" if gold_reference else "A",
                    "answer_views": [
                        {
                            "kind": "reference" if gold_reference else "answer_only",
                            "record_id": str(row["record_id"]),
                            "cot_ids": list(row["cot_ids"]) if gold_reference else [],
                            "answer_ids": answer_ids,
                        }
                    ],
                    "native_view_count": 1 if gold_reference else 0,
                    "direct_fallback": False,
                    "answer_only": not gold_reference,
                }
            )
        if not result:
            raise ValueError("answer-only Route1 population is empty")
        return result

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    order: list[str] = []
    for row in rows:
        prompt_group_id = str(row.get("prompt_group_id", ""))
        if not prompt_group_id:
            raise ValueError("Route1 target row lacks prompt identity")
        if prompt_group_id not in grouped:
            grouped[prompt_group_id] = []
            order.append(prompt_group_id)
        grouped[prompt_group_id].append(row)

    invariant_fields = (
        "problem_id",
        "semantic_group_id",
        "prompt_ids",
        "direct_prompt_ids",
        "direct_correct",
        "locked_full_correct",
        "n_correct",
    )
    for prompt_group_id in order:
        prompt_rows = grouped[prompt_group_id]
        first = prompt_rows[0]
        for row in prompt_rows:
            if any(row.get(field) != first.get(field) for field in invariant_fields):
                raise ValueError(
                    f"Route1 prompt rows disagree on sealed identity: {prompt_group_id}"
                )
            if row.get("provenance_valid") is not True:
                raise ValueError("Route1 prompt row lacks complete provenance")
        native_rows = [row for row in prompt_rows if row.get("population") == "native"]
        direct_rows = [
            row for row in prompt_rows if row.get("population") == "d_answer_fallback"
        ]
        if len(native_rows) + len(direct_rows) != len(prompt_rows):
            raise ValueError("Route1 prompt contains an unknown target population")
        if native_rows and direct_rows:
            raise ValueError("B/C native success must mask every direct answer target")
        if native_rows:
            if (
                not bool(first.get("locked_full_correct"))
                or int(first.get("n_correct", 0)) <= 0
            ):
                raise ValueError("B/C native samples disagree with sealed prompt label")
        elif len(direct_rows) != 1:
            raise ValueError(
                "D requires exactly one sealed direct-answer fallback sample"
            )

    result: list[dict[str, Any]] = []
    for row in rows:
        native = row.get("population") == "native"
        direct_correct = bool(row.get("direct_correct"))
        quadrant = route1_population(native, direct_correct)
        if quadrant == "E":
            raise ValueError("E records must not enter Route1 training")
        cot_ids = list(row.get("cot_ids", ()))
        answer_ids = list(row.get("answer_ids", ()))
        if not answer_ids:
            raise ValueError("Route1 sample lacks an answer")
        if native:
            if quadrant not in {"B", "C"} or not cot_ids:
                raise ValueError("native sample lacks its paired CoT/answer provenance")
            kind = "native"
        else:
            if quadrant != "D" or cot_ids:
                raise ValueError("D fallback sample must be suffix-free")
            kind = "direct"
        result.append(
            {
                **dict(row),
                "quadrant": quadrant,
                "population": quadrant,
                "need_z": quadrant == "C",
                "answer_views": [
                    {
                        "kind": kind,
                        "record_id": str(row["record_id"]),
                        "cot_ids": cot_ids,
                        "answer_ids": answer_ids,
                    }
                ],
                "native_view_count": int(native),
                "direct_fallback": quadrant == "D",
            }
        )
    return result


def legal_route1_wrong_donor(
    owner: Mapping[str, Any],
    donor: Mapping[str, Any],
    *,
    enabled: bool,
    active_populations: Sequence[str] = ("B", "C"),
    owner_populations: Sequence[str] | None = ("C",),
) -> bool:
    """Require eligible C owners and cross-question B/C donors."""

    if not isinstance(enabled, bool):
        raise TypeError("Route1 specificity switch must be boolean")
    if not enabled:
        return False
    populations = set(ROUTE1_POPULATIONS) | {"G"}
    active = frozenset(str(value) for value in active_populations)
    if not active or not active.issubset(populations):
        raise ValueError("Route1 wrong-z active populations must be a non-empty subset")
    if (
        owner.get("population") not in populations
        or donor.get("population") not in populations
    ):
        raise ValueError("Route1 wrong-z bank contains an unknown population")
    owner_active = owner.get("population") in active
    donor_active = donor.get("population") in active
    return (
        owner_active
        and donor_active
        and (owner_populations is None or owner.get("population") in owner_populations)
        and all(
            str(donor.get(field, "")) != str(owner.get(field, ""))
            and bool(str(donor.get(field, "")))
            and bool(str(owner.get(field, "")))
            for field in ("prompt_group_id", "problem_id", "semantic_group_id")
        )
    )


def stable_order_key(*parts: Any) -> int:
    """Return a stable row-order key; never pass this value to an RNG."""

    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def validation_randomness_identity(*, route: str, base_seed: int) -> dict[str, Any]:
    """Seal the common-random-numbers protocol shared by every checkpoint."""

    if route not in {"route1"}:
        raise ValueError("validation randomness route is invalid")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("validation randomness base seed is invalid")
    return {
        "protocol": "shared-root-seed",
        "route": route,
        "base_seed": int(base_seed),
        "checkpoint_step_participates": False,
    }


def route1_confidence_metrics(
    paired_rows: Sequence[Mapping[str, Any]],
    *,
    generation_seed: int,
    run_seed: int,
) -> dict[str, Any]:
    """Compute C-population robust G1 and a prompt-row bootstrap CI.

    Each bootstrap sample recomputes ``mean(true)-max(mean(direct),mean(wrong))``;
    wrong accuracy averages donors within each question first. Missing controls
    are excluded from this diagnostic, never from full true accuracy.
    """

    all_c = [row for row in paired_rows if row.get("population") == "C"]
    c_rows = [
        row
        for row in all_c
        if any(key.startswith("wrong_z_") for key in row["correct"])
    ]
    if len({str(row.get("prompt_group_id", "")) for row in all_c}) != len(all_c):
        raise ValueError("Route1 confidence rows must be unique prompt rows")
    donor_counts = [
        sum(key.startswith("wrong_z_") for key in row["correct"]) for row in c_rows
    ]
    base = {
        "c_prompt_count": len(c_rows),
        "control_available_count": len(c_rows),
        "control_unavailable_count": len(all_c) - len(c_rows),
        "wrong_donor_count": donor_counts[0]
        if donor_counts and len(set(donor_counts)) == 1
        else None,
        "wrong_donor_counts": donor_counts,
    }
    if not c_rows:
        return {
            **base,
            "a_true": None,
            "a_direct": None,
            "a_wrong": None,
            "robust_g1": None,
            "robust_g1_ci_low": None,
        }

    def statistic(rows):
        a_true = sum(float(row["correct"]["true_z"]) for row in rows) / len(rows)
        a_direct = sum(float(row["correct"]["direct"]) for row in rows) / len(rows)
        means = []
        for row in rows:
            values = [
                float(v) for k, v in row["correct"].items() if k.startswith("wrong_z_")
            ]
            means.append(sum(values) / len(values))
        a_wrong = sum(means) / len(means)
        return a_true, a_direct, a_wrong, a_true - max(a_direct, a_wrong)

    a_true, a_direct, a_wrong, robust_g1 = statistic(c_rows)
    ci_low = None
    if len(c_rows) >= 2:
        generator = random.Random(int(generation_seed))
        samples = []
        for _ in range(10_000):
            sample = [c_rows[generator.randrange(len(c_rows))] for _ in c_rows]
            samples.append(statistic(sample)[3])
        samples.sort()
        ci_low = float(samples[max(0, math.ceil(0.025 * len(samples)) - 1)])
    return {
        **base,
        "a_true": a_true,
        "a_direct": a_direct,
        "a_wrong": a_wrong,
        "robust_g1": robust_g1,
        "robust_g1_ci_low": ci_low,
    }


def route1_selector_metrics(
    paired_rows: Sequence[Mapping[str, Any]],
    *,
    generation_seed: int,
    run_seed: int,
    include_controls: bool = True,
) -> dict[str, Any]:
    """Recompute the complete Route1 selector metric set from sparse rows."""

    if not paired_rows:
        raise ValueError("Route1 selector metrics require evaluation rows")
    full_conditions = frozenset(("true_z", "direct", *ROUTE1_SELECTOR_WRONG_CONDITIONS))
    sparse_true = frozenset(("true_z",))
    clean_rows: list[Mapping[str, Any]] = []
    for row in paired_rows:
        population = row.get("population")
        correct = row.get("correct")
        if population not in ROUTE1_POPULATIONS or not isinstance(correct, Mapping):
            raise ValueError("Route1 selector paired-row schema is invalid")
        conditions = frozenset(correct)
        if population == "C":
            wrong = conditions - {"true_z", "direct"}
            expected_wrong = frozenset(ROUTE1_SELECTOR_WRONG_CONDITIONS[: len(wrong)])
            if (
                not {"true_z", "direct"}.issubset(conditions)
                or wrong != expected_wrong
                or (wrong and not include_controls)
            ):
                raise ValueError("Route1 C row has invalid true/direct/wrong controls")
        elif conditions not in {sparse_true, full_conditions}:
            raise ValueError("Route1 non-C row has a partial control schema")
        if any(not isinstance(value, bool) for value in correct.values()):
            raise ValueError("Route1 selector correctness values must be booleans")
        clean_rows.append(row)
    confidence = (
        route1_confidence_metrics(
            clean_rows,
            generation_seed=int(generation_seed),
            run_seed=int(run_seed),
        )
        if include_controls
        else None
    )
    true_values = [bool(row["correct"]["true_z"]) for row in clean_rows]
    quadrant_rows = {
        population: [row for row in clean_rows if row["population"] == population]
        for population in ROUTE1_POPULATIONS
    }
    quadrant_accuracy = {
        population: (
            sum(bool(row["correct"]["true_z"]) for row in values) / len(values)
            if values
            else None
        )
        for population, values in quadrant_rows.items()
    }
    c_rows = quadrant_rows["C"]
    a_true = quadrant_accuracy["C"]
    a_direct = (
        sum(bool(row["correct"]["direct"]) for row in c_rows) / len(c_rows)
        if c_rows
        else None
    )
    return {
        "true_z_full_accuracy": sum(true_values) / len(true_values),
        "a_true": a_true,
        "a_direct": a_direct,
        "raw_g1": a_true - a_direct if c_rows else None,
        **(
            {
                name: confidence[name]
                for name in (
                    "a_wrong",
                    "robust_g1",
                    "robust_g1_ci_low",
                    "control_available_count",
                    "control_unavailable_count",
                    "wrong_donor_counts",
                )
            }
            if confidence is not None
            else {}
        ),
        "b_retention": quadrant_accuracy["B"],
        "d_retention": quadrant_accuracy["D"],
        "true_full_correct": sum(true_values),
        "true_full_total": len(clean_rows),
        **{
            f"{population.lower()}_denominator": len(quadrant_rows[population])
            for population in ROUTE1_POPULATIONS
        },
        **{
            f"{population.lower()}_accuracy": quadrant_accuracy[population]
            for population in ROUTE1_POPULATIONS
        },
        "c_paired_count": len(quadrant_rows["C"]),
        "wrong_pair_count": sum(
            sum(key.startswith("wrong_z_") for key in row["correct"]) for row in c_rows
        ),
    }


def write_atomic_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    replace_mismatch: bool = False,
) -> str:
    """Publish JSON atomically, reusing only an exact completed artifact.

    ``replace_mismatch=False`` is for immutable seals: a completed but different
    payload is an identity error.  ``replace_mismatch=True`` is for reproducible
    Stage1 caches such as target indexes, manifests, and donors.  Every writer
    owns a unique same-directory build file; immutable publication uses an
    atomic hard-link create so concurrent writers cannot overwrite one another.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    expected = json.loads(serialized)
    existed = target.exists()
    if target.is_symlink():
        raise ValueError(f"JSON artifact path cannot be a symlink: {target}")
    if target.is_file():
        try:
            observed = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            if not replace_mismatch:
                raise
            observed = None
        if observed == expected:
            return "reuse"
        if not replace_mismatch:
            raise ValueError(
                f"existing sealed JSON differs from exact payload: {target}"
            )
    elif existed:
        raise FileExistsError(f"JSON artifact path is not a file: {target}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.building.",
        dir=str(target.parent),
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        observed = json.loads(temporary.read_text(encoding="utf-8"))
        if observed != expected:
            raise ValueError("JSON building artifact failed exact validation")
        if replace_mismatch:
            os.replace(temporary, target)
            return "replace" if existed else "build"
        try:
            os.link(temporary, target)
        except FileExistsError:
            if not target.is_file():
                raise FileExistsError(f"JSON artifact path is not a file: {target}")
            completed = json.loads(target.read_text(encoding="utf-8"))
            if completed != expected:
                raise ValueError(
                    f"concurrent sealed JSON differs from exact payload: {target}"
                )
            return "reuse"
        return "build"
    finally:
        temporary.unlink(missing_ok=True)


SEMANTIC_GROUPING_SCHEMA = {
    "train": "explicit-or-ops-v1-fail-closed",
    "validation": "explicit-or-problem-fallback-v1",
}
HARD_DONOR_MATCHING_CONTRACT = {
    "protocol": "bridge-stable-wrong-donor-up-to-k8",
    "donor_count": 8,
    "count_policy": "min(requested, eligible); zero means unavailable",
    "stratum_fields": [],
    "telemetry_fields": ["direct_correct"],
    "exclusion_fields": [
        "prompt_group_id",
        "problem_id",
        "semantic_group_id",
    ],
    "eligible_donor_domain": "known CoT length within capacity, or unobserved length",
    "distance_rule": (
        "abs(prompt_token_count-anchor_prompt_token_count)+"
        "abs(cot_token_count-anchor_cot_token_count) when both observed; otherwise prompt-only"
    ),
    "sort_order": [
        "total_token_length_delta",
        "anchor_donor_record_id_sha256",
    ],
    "tie_break_rule": "sha256(anchor_record_id\\x1fdonor_record_id)",
}


def write_atomic_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    replace_mismatch: bool = False,
) -> str:
    """Validate and atomically publish an immutable seal or mutable JSONL cache."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    expected: list[dict[str, Any]] = []
    serialized_rows: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("JSONL rows must be mappings")
        normalized = dict(row)
        serialized_rows.append(
            json.dumps(normalized, sort_keys=True, ensure_ascii=False) + "\n"
        )
        expected.append(normalized)
    existed = target.exists()
    if target.is_symlink():
        raise ValueError(f"JSONL artifact path cannot be a symlink: {target}")
    if target.is_file():
        try:
            observed = [
                json.loads(line)
                for line in target.read_text(encoding="utf-8").splitlines()
                if line
            ]
        except (OSError, TypeError, ValueError):
            if not replace_mismatch:
                raise
            observed = None
        if observed == expected:
            return "reuse"
        if not replace_mismatch:
            raise FileExistsError(f"refusing to overwrite sealed JSONL: {target}")
    elif existed:
        raise FileExistsError(f"JSONL artifact path is not a file: {target}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.building.", dir=str(target.parent), text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.writelines(serialized_rows)
            stream.flush()
            os.fsync(stream.fileno())
        observed = [
            json.loads(line)
            for line in temporary.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if observed != expected or any(
            not isinstance(row, Mapping) for row in observed
        ):
            raise ValueError("JSONL building artifact failed exact validation")
        if replace_mismatch:
            os.replace(temporary, target)
            return "replace" if existed else "build"
        try:
            os.link(temporary, target)
        except FileExistsError:
            completed = [
                json.loads(line)
                for line in target.read_text(encoding="utf-8").splitlines()
                if line
            ]
            if completed != expected:
                raise ValueError(
                    f"concurrent sealed JSONL differs from exact rows: {target}"
                )
            return "reuse"
        return "build"
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class CoursePhase:
    name: str
    active_owner: str
    epoch: int


@dataclass(frozen=True)
class Route1BatchGeometry:
    local_samples: int
    world_size: int
    gradient_accumulation_steps: int
    microstep_global_samples: int
    optimizer_global_samples: int


@dataclass(frozen=True)
class StartupProbeTokenIds:
    prompt: tuple[int, ...]
    cot: tuple[int, ...]
    answer: tuple[int, ...]


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_BRIDGE_STAGE1_CACHE_ROOT = Path("data/stage1")
_BRIDGE_STAGE1_MODEL_DIRECTORIES = {
    "qwen3-0.6b": "Qwen3-0.6B",
    "qwen3-4b": "Qwen3-4B",
}


def is_bridge_isolated_path(path: str | Path) -> bool:
    """Accept one concrete non-root artifact path without name policing."""

    candidate = Path(path).expanduser()
    if str(candidate).strip() in {"", ".", "/"}:
        return False
    try:
        normalized = candidate.resolve(strict=False)
    except (OSError, RuntimeError):
        return False
    return normalized != Path(normalized.anchor)


def bridge_stage1_model_root(
    model_family: str,
    *,
    cache_root: str | Path = _BRIDGE_STAGE1_CACHE_ROOT,
) -> Path:
    """Return one explicit non-root Stage1 cache namespace."""

    raw_root = str(cache_root).strip()
    root = Path(raw_root).expanduser()
    if (
        raw_root in {"", ".", "..", "/"}
        or ".." in root.parts
        or root == Path(root.anchor)
        or (root.exists() and not root.is_dir())
    ):
        raise ValueError(
            "ThinkBridge cache root must be explicit, non-root, and traversal-free"
        )
    cursor = Path(root.anchor) if root.is_absolute() else Path()
    parts = root.parts[1:] if root.is_absolute() else root.parts
    for component in parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ValueError("ThinkBridge cache root cannot traverse a symlink alias")
    try:
        model_directory = _BRIDGE_STAGE1_MODEL_DIRECTORIES[str(model_family)]
    except KeyError as exc:
        raise ValueError(
            "ThinkBridge Stage1 artifacts require a maintained model family"
        ) from exc
    return root / model_directory


def resolve_boundary_token_ids(tokenizer: Any, boundary_text: str) -> tuple[int, ...]:
    """Resolve and validate the canonical boundary from one runtime tokenizer."""

    if not isinstance(boundary_text, str) or not boundary_text:
        raise ValueError("boundary_text must be a non-empty string")
    raw_ids = tokenizer.encode(boundary_text, add_special_tokens=False)
    if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes)):
        raise ValueError("runtime tokenizer boundary encoding must be a sequence")
    if not raw_ids:
        raise ValueError("runtime tokenizer boundary encoding is empty")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw_ids):
        raise ValueError("runtime tokenizer boundary ids must be integers")
    boundary_ids = tuple(int(value) for value in raw_ids)
    try:
        vocabulary_size = int(len(tokenizer))
    except (TypeError, AttributeError):
        vocabulary_size = int(getattr(tokenizer, "vocab_size", 0))
    if vocabulary_size <= 0:
        raise ValueError("runtime tokenizer has no valid vocabulary size")
    if any(value < 0 or value >= vocabulary_size for value in boundary_ids):
        raise ValueError("runtime tokenizer boundary id is outside its vocabulary")

    # Enforce encode/decode/encode only when the tokenizer declares the exact
    # canonical text round-trip. Some official special-token decoders expose a
    # normalized spelling, so a non-identical decode is not used as a false gate.
    try:
        decoded = tokenizer.decode(
            list(boundary_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        try:
            decoded = tokenizer.decode(list(boundary_ids))
        except (AttributeError, TypeError):
            decoded = None
    except AttributeError:
        decoded = None
    if decoded == boundary_text:
        repeated = tokenizer.encode(decoded, add_special_tokens=False)
        if list(repeated) != list(boundary_ids):
            raise ValueError("runtime tokenizer boundary round-trip is not exact")
    return boundary_ids


def validate_boundary_token_identity(
    boundary_token_ids: Any,
    *,
    boundary_token_count: Any,
    boundary_ids_sha256: Any,
) -> tuple[int, ...]:
    if (
        not isinstance(boundary_token_ids, (list, tuple))
        or not boundary_token_ids
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in boundary_token_ids
        )
    ):
        raise ValueError("sealed boundary_token_ids must be non-empty integer ids")
    normalized = tuple(int(value) for value in boundary_token_ids)
    if (
        isinstance(boundary_token_count, bool)
        or not isinstance(boundary_token_count, int)
        or boundary_token_count != len(normalized)
    ):
        raise ValueError("sealed boundary_token_count differs from boundary ids")
    require_sha256(boundary_ids_sha256, "boundary_ids_sha256")
    if boundary_ids_sha256 != canonical_json_sha256(normalized):
        raise ValueError("sealed boundary id hash differs from boundary ids")
    return normalized


def resolve_bridge_semantic_group(
    row: Mapping[str, Any], *, split: str, problem_id: str
) -> str:
    """Resolve only sealed source semantics; never infer from prompt text/model."""

    if split not in SEMANTIC_GROUPING_SCHEMA:
        raise ValueError(f"unknown ThinkBridge split for semantic grouping: {split}")
    explicit = row.get("semantic_group_id")
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit.strip():
            raise ValueError("explicit semantic_group_id must be a non-empty string")
        return explicit.strip()
    if split == "train":
        ops = row.get("ops")
        if not isinstance(ops, str) or not ops.strip():
            raise ValueError(
                "train row needs explicit semantic_group_id or immutable non-empty ops"
            )
        return f"ops-v1:{ops.strip()}"
    fallback = str(problem_id).strip()
    if not fallback:
        raise ValueError("held-out problem fallback requires a non-empty problem_id")
    return fallback


def validate_compiled_cot_capacity(
    *,
    content_token_count: int,
    readout_token_count: int,
    route2_eligible: bool,
    content_capacity: int,
) -> None:
    """Validate legacy compiled CoT metadata retained for data compatibility."""
    counts = content_token_count, readout_token_count
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in counts
    ):
        raise ValueError("prepared Route2 token counts must be nonnegative integers")
    if (
        isinstance(content_capacity, bool)
        or not isinstance(content_capacity, int)
        or content_capacity <= 0
    ):
        raise ValueError("Route2 content capacity must be a positive integer")
    if not isinstance(route2_eligible, bool):
        raise TypeError("prepared Route2 eligibility switches must be booleans")
    if readout_token_count != content_token_count + 1:
        raise ValueError("Route2 readout length must equal content plus terminal EOS")
    fits = content_token_count <= content_capacity
    if route2_eligible != fits:
        raise ValueError("route2 eligibility differs from configured content capacity")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a 64-character SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be hexadecimal") from exc
    return value.lower()


def phase_for_epoch(epoch: int, *, route1_epochs: int = 2) -> CoursePhase:
    if any(
        isinstance(v, bool) or not isinstance(v, int) for v in (epoch, route1_epochs)
    ):
        raise TypeError("ThinkBridge epoch counts must be integers")
    if route1_epochs <= 0 or not 0 <= epoch < route1_epochs:
        raise ValueError("ThinkBridge epoch lies outside the configured R course")
    return CoursePhase("route1", "R", epoch)


def validate_hard_donors(
    anchor: Mapping[str, Any], donors: Sequence[Mapping[str, Any]]
) -> None:
    required = (
        "record_id",
        "prompt_group_id",
        "problem_id",
        "semantic_group_id",
        "need_z",
        "direct_correct",
    )
    if len(donors) > 8:
        raise ValueError("at most eight cross-prompt wrong-z donors are supported")
    if any(key not in anchor for key in required):
        raise ValueError("anchor donor-domain metadata is incomplete")
    if not isinstance(anchor["need_z"], bool) or not isinstance(
        anchor["direct_correct"], bool
    ):
        raise ValueError("anchor need_z/direct_correct behavior labels are invalid")
    donor_ids: set[str] = set()
    for donor in donors:
        if any(key not in donor for key in required):
            raise ValueError("donor metadata is incomplete")
        donor_id = str(donor["record_id"])
        if not donor_id or donor_id in donor_ids:
            raise ValueError("hard donors must be distinct")
        donor_ids.add(donor_id)
        if not isinstance(donor["need_z"], bool) or not isinstance(
            donor["direct_correct"], bool
        ):
            raise ValueError("hard donor behavior labels are invalid")
        if any(
            donor[key] == anchor[key]
            for key in ("prompt_group_id", "problem_id", "semantic_group_id")
        ):
            raise ValueError(
                "hard donor overlaps anchor prompt/problem/semantic domain"
            )


def _hard_donor_row_metadata(row: Mapping[str, Any], *, split: str) -> dict[str, Any]:
    required = (
        "record_id",
        "prompt_group_id",
        "problem_id",
        "semantic_group_id",
        "prompt_ids",
        "cot_ids",
        "need_z",
        "direct_correct",
    )
    missing = [field for field in required if field not in row]
    if missing:
        if "need_z" in missing or "direct_correct" in missing:
            raise ValueError(
                f"{split} hard-donor row lacks need_z/direct_correct behavior labels"
            )
        raise ValueError(f"{split} hard-donor row metadata is incomplete: {missing}")
    record_id = str(row["record_id"])
    prompt_group_id = str(row["prompt_group_id"])
    problem_id = str(row["problem_id"])
    semantic_group_id = str(row["semantic_group_id"])
    if not all((record_id, prompt_group_id, problem_id, semantic_group_id)):
        raise ValueError(f"{split} hard-donor identity fields must be non-empty")
    if not isinstance(row["need_z"], bool) or not isinstance(
        row["direct_correct"], bool
    ):
        raise ValueError(
            f"{split} hard-donor need_z/direct_correct behavior labels are invalid"
        )
    prompt_ids = row["prompt_ids"]
    cot_ids = row["cot_ids"]
    if (
        not isinstance(prompt_ids, list)
        or not prompt_ids
        or any(type(token) is not int or token < 0 for token in prompt_ids)
        or not isinstance(cot_ids, list)
        or any(type(token) is not int or token < 0 for token in cot_ids)
    ):
        raise ValueError(f"{split} hard-donor prompt/CoT token ids are invalid")
    return {
        "record_id": record_id,
        "prompt_group_id": prompt_group_id,
        "problem_id": problem_id,
        "semantic_group_id": semantic_group_id,
        "prompt_ids": [int(token) for token in prompt_ids],
        "cot_ids": [int(token) for token in cot_ids],
        "cot_length_known": row.get("cot_length_known", True),
        "need_z": bool(row["need_z"]),
        "direct_correct": bool(row["direct_correct"]),
    }


def build_matched_hard_donor_bindings(
    rows: Sequence[Mapping[str, Any]], *, split: str, content_capacity: int
) -> list[dict[str, Any]]:
    """Build up to eight distinct deterministic held-out wrong-z donors."""

    if split != "validation":
        raise ValueError("hard-donor bindings are validation-only")
    if int(content_capacity) <= 0:
        raise ValueError("hard-donor Route2 content capacity must be positive")
    normalized = [_hard_donor_row_metadata(row, split=split) for row in rows]
    by_id = {row["record_id"]: row for row in normalized}
    if not normalized or len(by_id) != len(normalized):
        raise ValueError(
            f"{split} hard-donor source record ids are empty or duplicated"
        )
    bindings: list[dict[str, Any]] = []
    for anchor in sorted(normalized, key=lambda row: row["record_id"]):
        scored: list[tuple[int, str, dict[str, Any]]] = []
        for donor in normalized:
            if donor["cot_length_known"] and len(donor["cot_ids"]) > int(
                content_capacity
            ):
                continue
            if any(
                donor[key] == anchor[key]
                for key in ("prompt_group_id", "problem_id", "semantic_group_id")
            ):
                continue
            prompt_delta = abs(len(donor["prompt_ids"]) - len(anchor["prompt_ids"]))
            cot_delta = (
                abs(len(donor["cot_ids"]) - len(anchor["cot_ids"]))
                if donor["cot_length_known"] and anchor["cot_length_known"]
                else None
            )
            total_delta = prompt_delta + (cot_delta if cot_delta is not None else 0)
            tie_break = hashlib.sha256(
                f"{anchor['record_id']}\x1f{donor['record_id']}".encode("utf-8")
            ).hexdigest()
            scored.append(
                (
                    total_delta,
                    tie_break,
                    {
                        "record_id": donor["record_id"],
                        "prompt_group_id": donor["prompt_group_id"],
                        "problem_id": donor["problem_id"],
                        "semantic_group_id": donor["semantic_group_id"],
                        "prompt_ids": donor["prompt_ids"],
                        "need_z": donor["need_z"],
                        "direct_correct": donor["direct_correct"],
                        "prompt_token_count": len(donor["prompt_ids"]),
                        "cot_token_count": len(donor["cot_ids"])
                        if donor["cot_length_known"]
                        else None,
                        "prompt_token_length_delta": prompt_delta,
                        "cot_token_length_delta": cot_delta,
                        "total_token_length_delta": total_delta,
                        "anchor_donor_record_id_sha256": tie_break,
                    },
                )
            )
        scored.sort(key=lambda item: (item[0], item[1]))
        donors = []
        for rank, (_, _, donor) in enumerate(
            scored[: int(HARD_DONOR_MATCHING_CONTRACT["donor_count"])], start=1
        ):
            selected = dict(donor)
            selected["selection_rank"] = rank
            donors.append(selected)
        validate_hard_donors(anchor, donors)
        bindings.append(
            {
                "anchor_record_id": anchor["record_id"],
                "split_id": split,
                "anchor_stratum": {
                    "need_z": anchor["need_z"],
                },
                "anchor_telemetry": {
                    "direct_correct": anchor["direct_correct"],
                },
                "anchor_prompt_token_count": len(anchor["prompt_ids"]),
                "anchor_cot_token_count": len(anchor["cot_ids"])
                if anchor["cot_length_known"]
                else None,
                "donors": donors,
            }
        )
    return bindings


def build_matched_hard_donor_manifest(
    split_rows: Mapping[str, Sequence[Mapping[str, Any]]], *, content_capacity: int
) -> dict[str, Any]:
    if set(split_rows) != {"validation"}:
        raise ValueError("course hard-donor manifest requires validation only")
    records = build_matched_hard_donor_bindings(
        split_rows["validation"],
        split="validation",
        content_capacity=content_capacity,
    )
    return {
        **artifact_header(STAGE1_HARD_DONOR_MANIFEST),
        "matching_contract": dict(HARD_DONOR_MATCHING_CONTRACT),
        "records": records,
    }


def validate_matched_hard_donor_manifest(
    payload: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    content_capacity: int,
) -> list[dict[str, Any]]:
    """Recompute and validate every sealed donor binding for one held-out split."""

    if set(payload) != {
        "artifact_type",
        "schema_version",
        "matching_contract",
        "records",
    }:
        raise ValueError("hard-donor manifest has missing or unknown top-level fields")
    require_artifact_header(
        payload, STAGE1_HARD_DONOR_MANIFEST, label="hard-donor manifest"
    )
    if payload["matching_contract"] != HARD_DONOR_MATCHING_CONTRACT:
        raise ValueError("hard-donor matching contract differs from the sealed rule")
    records = payload["records"]
    if not isinstance(records, list) or any(
        not isinstance(row, dict) for row in records
    ):
        raise ValueError("hard-donor records must be a list of objects")
    actual = [dict(row) for row in records if row.get("split_id") == split]
    expected = build_matched_hard_donor_bindings(
        rows, split=split, content_capacity=content_capacity
    )
    if actual != expected:
        raise ValueError(
            "hard-donor ids, exact need cohort, distance ordering, or tie-break seal mismatch"
        )
    if split != "validation" or len(actual) != len(records):
        raise ValueError("course hard-donor manifest is validation-only")
    return actual


def validate_route1_batch_geometry(
    *, local_samples: int, world_size: int, gradient_accumulation_steps: int
) -> Route1BatchGeometry:
    """Derive one legal Route1 sample geometry from runtime parameters."""

    local = int(local_samples)
    world = int(world_size)
    accumulation = int(gradient_accumulation_steps)
    if local <= 0 or world <= 0 or accumulation <= 0:
        raise ValueError("Route1 sample batch geometry must be positive")
    microstep = local * world
    optimizer = microstep * accumulation
    return Route1BatchGeometry(local, world, accumulation, microstep, optimizer)


def manifest_field_names(route: str) -> frozenset[str]:
    """Return the exact fields for the current shared Stage1 manifest."""

    common = {
        "artifact_type",
        "schema_version",
        "method",
        "route",
        "model_family",
        "tokenizer_sha256",
        "template_sha256",
        "boundary_ids_sha256",
        "z_width",
        "route2_content_capacity",
        "need_z_cohort_definition",
        "d_answer_provenance_schema",
        "d_answer_eos_rule",
        "train_c_sample_count",
        "train_d_sample_count",
        "train_d_excluded_non_eos_count",
    }
    if route != "route1":
        raise ValueError("shared training manifest route must be route1")
    return frozenset(common)


def require_manifest_fields(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return the reference-only shared artifact identity."""
    if not isinstance(manifest, Mapping):
        raise ValueError("Bridge shared manifest must be a mapping")
    route = str(manifest.get("route", "")).lower()
    require_artifact_header(
        manifest, STAGE1_SHARED_MANIFEST, label="Bridge shared manifest"
    )
    required = manifest_field_names(route)
    missing = sorted(required.difference(manifest))
    unknown = sorted(set(manifest).difference(required))
    if missing or unknown:
        raise ValueError(
            f"Bridge manifest exact field mismatch; missing={missing}, unknown={unknown}"
        )
    manifest = {name: manifest[name] for name in required}
    if manifest["method"] != "bridge" or route != "route1":
        raise ValueError("manifest method/route mismatch")
    if manifest["model_family"] not in _BRIDGE_STAGE1_MODEL_DIRECTORIES:
        raise ValueError("manifest model family mismatch")
    for field in ("tokenizer_sha256", "template_sha256", "boundary_ids_sha256"):
        require_sha256(manifest[field], field)
    if isinstance(manifest["z_width"], bool) or int(manifest["z_width"]) <= 0:
        raise ValueError("manifest z_width is invalid")
    if (
        isinstance(manifest["route2_content_capacity"], bool)
        or not isinstance(manifest["route2_content_capacity"], int)
        or manifest["route2_content_capacity"] <= 0
    ):
        raise ValueError("manifest Route2 content capacity is invalid")
    if manifest["need_z_cohort_definition"] != NEED_Z_COHORT_DEFINITION:
        raise ValueError("manifest need-z cohort definition mismatch")
    if manifest["d_answer_provenance_schema"] != D_ANSWER_PROVENANCE_SCHEMA:
        raise ValueError("manifest D-answer provenance schema mismatch")
    if manifest["d_answer_eos_rule"] != D_ANSWER_EOS_RULE:
        raise ValueError("manifest D-answer EOS rule mismatch")
    for field in (
        "train_c_sample_count",
        "train_d_sample_count",
        "train_d_excluded_non_eos_count",
    ):
        value = manifest[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"manifest {field} is invalid")
    return dict(manifest)


def assert_shared_manifest_semantically_equal(
    observed: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    """Require exact current shared-manifest semantics."""

    try:
        observed_semantics = require_manifest_fields(observed)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"existing shared manifest semantic identity is invalid: {error}"
        ) from error
    try:
        expected_semantics = require_manifest_fields(expected)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"expected shared manifest semantic identity is invalid: {error}"
        ) from error
    if observed_semantics != expected_semantics:
        fields = sorted(
            field
            for field in set(observed_semantics) | set(expected_semantics)
            if observed_semantics.get(field) != expected_semantics.get(field)
        )
        raise ValueError(
            "existing shared manifest semantic identity differs from exact "
            f"payload: fields={fields}"
        )
