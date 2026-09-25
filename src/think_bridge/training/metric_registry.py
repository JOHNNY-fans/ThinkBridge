"""Canonical Bridge validation metrics and auditable multi-metric selection.

HF and ms-swift expose one ``metric_for_best_model``.  Bridge keeps that clean
public name while deliberately accepting a controlled comma-separated list.
Every value is extracted through this registry, normalized so larger is
better, aggregated by an explicit policy, and sealed with its tie-break trace.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import warnings
from typing import Any, Mapping, Sequence


_ENTRANCE_EVALUATOR = {"reasoner-sft": "stage1.route1"}
_REPORT_ROUTE_ENTRANCE = {"route1": "reasoner-sft"}

_DEFAULT_PRIMARY = {"reasoner-sft": ("route1.true_z_full_accuracy",)}

_DEFAULT_SECONDARY = {"reasoner-sft": ()}

# Read old sealed reports without silently changing how their ties were ranked.
_RECORDED_SECONDARY = {
    "reasoner-sft": (
        "route1.robust_g1",
        "route1.b_retention",
        "route1.d_retention",
        "route1.robust_g1_ci_low",
    )
}


@dataclass(frozen=True)
class MetricSpec:
    canonical_name: str
    report_sources: tuple[tuple[str, tuple[str, ...]], ...]
    direction: str
    normalization: str
    minimum: float
    maximum: float | None
    applicable_evaluators: tuple[str, ...]
    default_primary_for: tuple[str, ...]
    secondary_priority_for: tuple[str, ...]

    def source_for(self, evaluator: str) -> tuple[str, ...]:
        sources = dict(self.report_sources)
        if evaluator not in sources:
            raise ValueError(
                f"canonical metric {self.canonical_name} is not applicable to "
                f"evaluator {evaluator}"
            )
        return sources[evaluator]

    def normalized(self, value: float) -> float:
        observed = float(value)
        if not math.isfinite(observed):
            raise ValueError(f"canonical metric {self.canonical_name} must be finite")
        if observed < self.minimum or (
            self.maximum is not None and observed > self.maximum
        ):
            upper = "+inf" if self.maximum is None else str(self.maximum)
            raise ValueError(
                f"canonical metric {self.canonical_name} is outside range "
                f"[{self.minimum}, {upper}]: {observed}"
            )
        if self.normalization == "bounded_linear":
            if self.maximum is None or self.maximum <= self.minimum:
                raise RuntimeError(
                    f"canonical metric {self.canonical_name} has invalid bounds"
                )
            scaled = (observed - self.minimum) / (self.maximum - self.minimum)
            return scaled if self.direction == "maximize" else 1.0 - scaled
        if self.normalization == "inverse_one_plus":
            if self.direction != "minimize" or self.minimum != 0.0:
                raise RuntimeError(
                    f"canonical metric {self.canonical_name} has invalid inverse normalization"
                )
            return 1.0 / (1.0 + observed)
        raise RuntimeError(
            f"canonical metric {self.canonical_name} normalization is unknown"
        )


def _spec(
    name: str,
    *,
    sources: Mapping[str, Sequence[str]],
    direction: str = "maximize",
    normalization: str = "bounded_linear",
    minimum: float = 0.0,
    maximum: float | None = 1.0,
) -> MetricSpec:
    if direction not in {"maximize", "minimize"}:
        raise ValueError(f"invalid metric direction: {direction}")
    return MetricSpec(
        canonical_name=name,
        report_sources=tuple(
            (evaluator, tuple(path)) for evaluator, path in sorted(sources.items())
        ),
        direction=direction,
        normalization=normalization,
        minimum=float(minimum),
        maximum=None if maximum is None else float(maximum),
        applicable_evaluators=tuple(sorted(sources)),
        default_primary_for=tuple(
            entrance
            for entrance, metrics in _DEFAULT_PRIMARY.items()
            if name in metrics
        ),
        secondary_priority_for=tuple(
            entrance
            for entrance, metrics in _DEFAULT_SECONDARY.items()
            if name in metrics
        ),
    )


_REGISTRY = {
    "route1.true_z_full_accuracy": _spec(
        "route1.true_z_full_accuracy",
        sources={"stage1.route1": ("gate_metrics", "true_z_full_accuracy")},
    ),
    "route1.robust_g1": _spec(
        "route1.robust_g1",
        sources={"stage1.route1": ("gate_metrics", "robust_g1")},
        minimum=-1.0,
    ),
    "route1.b_retention": _spec(
        "route1.b_retention", sources={"stage1.route1": ("gate_metrics", "b_retention")}
    ),
    "route1.d_retention": _spec(
        "route1.d_retention", sources={"stage1.route1": ("gate_metrics", "d_retention")}
    ),
    "route1.robust_g1_ci_low": _spec(
        "route1.robust_g1_ci_low",
        sources={"stage1.route1": ("gate_metrics", "robust_g1_ci_low")},
        minimum=-1.0,
    ),
}


@dataclass(frozen=True)
class MetricPolicy:
    entrance: str
    evaluator: str
    primary_metrics: tuple[str, ...]
    secondary_metrics: tuple[str, ...]
    aggregation: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "entrance": self.entrance,
            "evaluator": self.evaluator,
            "primary_metrics": list(self.primary_metrics),
            "secondary_metrics": list(self.secondary_metrics),
            "aggregation": self.aggregation,
        }


@dataclass(frozen=True)
class ReportScore:
    step: int
    raw_metrics: dict[str, float]
    normalized_metrics: dict[str, float]
    aggregate_score: float
    secondary_values: tuple[float, ...]


@dataclass(frozen=True)
class SelectionDecision:
    step: int
    candidate: Mapping[str, Any]
    raw_metrics: dict[str, float]
    normalized_metrics: dict[str, float]
    aggregate_score: float
    tie_break: dict[str, str]


def resolve_metric_policy(
    *,
    entrance: str,
    metric_for_best_model: str | None,
    metric_aggregation: str,
) -> MetricPolicy:
    if entrance not in _ENTRANCE_EVALUATOR:
        raise ValueError(f"unknown maintained metric entrance: {entrance}")
    if metric_aggregation != "mean":
        raise ValueError("metric_aggregation must be mean")
    if metric_for_best_model is None:
        primary = _DEFAULT_PRIMARY[entrance]
    else:
        primary = tuple(item.strip() for item in str(metric_for_best_model).split(","))
        if not primary or any(not item for item in primary):
            raise ValueError("metric_for_best_model must contain canonical metrics")
        if len(set(primary)) != len(primary):
            raise ValueError("metric_for_best_model contains a duplicate metric")
    evaluator = _ENTRANCE_EVALUATOR[entrance]
    for name in primary:
        spec = _REGISTRY.get(name)
        if spec is None:
            raise ValueError(f"unknown canonical metric: {name}")
        if evaluator not in spec.applicable_evaluators:
            raise ValueError(f"canonical metric {name} is not applicable to {entrance}")
    secondary = tuple(
        name for name in _DEFAULT_SECONDARY[entrance] if name not in primary
    )
    return MetricPolicy(
        entrance=entrance,
        evaluator=evaluator,
        primary_metrics=tuple(primary),
        secondary_metrics=secondary,
        aggregation=metric_aggregation,
    )


def metric_policy_from_mapping(value: Mapping[str, Any]) -> MetricPolicy:
    if not isinstance(value, Mapping):
        raise ValueError("metric policy must be a mapping")
    primary = value.get("primary_metrics")
    if (
        not isinstance(primary, list)
        or not primary
        or any(not isinstance(name, str) or not name for name in primary)
    ):
        raise ValueError("metric policy primary metrics are invalid")
    policy = resolve_metric_policy(
        entrance=str(value.get("entrance", "")),
        metric_for_best_model=",".join(primary),
        metric_aggregation=str(value.get("aggregation", "")),
    )
    if policy.as_dict() != dict(value):
        raise ValueError("metric policy differs from the canonical registry")
    return policy


def report_metric_policy(
    *, route: str, value: Mapping[str, Any] | None
) -> MetricPolicy:
    """Resolve and route-check the policy persisted by an evaluation report."""

    entrance = _REPORT_ROUTE_ENTRANCE.get(route)
    if entrance is None:
        raise ValueError("metric report route must be route1")
    policy = (
        resolve_metric_policy(
            entrance=entrance,
            metric_for_best_model=None,
            metric_aggregation="mean",
        )
        if value is None
        else metric_policy_from_mapping(value)
    )
    expected_evaluator = _ENTRANCE_EVALUATOR[entrance]
    if policy.evaluator != expected_evaluator:
        raise ValueError("metric report policy evaluator differs from its route")
    return policy


def require_report_metric_policy(
    report: Mapping[str, Any], *, route: str
) -> MetricPolicy:
    """Read one explicit canonical metric policy from a persisted report."""

    value = report.get("metric_policy")
    if not isinstance(value, Mapping):
        raise ValueError("evaluation report lacks its sealed metric policy")
    return report_metric_policy(route=route, value=value)


def _metric_value(report: Mapping[str, Any], *, evaluator: str, name: str) -> float:
    spec = _REGISTRY[name]
    cursor: Any = report
    path = spec.source_for(evaluator)
    for component in path:
        if not isinstance(cursor, Mapping) or component not in cursor:
            raise ValueError(f"missing canonical metric {name} at " + ".".join(path))
        cursor = cursor[component]
    if isinstance(cursor, bool) or not isinstance(cursor, (int, float)):
        raise ValueError(f"canonical metric {name} must be numeric")
    value = float(cursor)
    spec.normalized(value)
    return value


def _report_step(report: Mapping[str, Any], *, evaluator: str) -> int:
    field = "course_step" if evaluator == "stage2.dual" else "step"
    value = report.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"metric report {field} must be a nonnegative integer")
    return int(value)


def score_report(
    report: Mapping[str, Any], *, evaluator: str, policy: MetricPolicy
) -> ReportScore:
    if evaluator != policy.evaluator:
        raise ValueError(
            f"metric evaluator {evaluator} differs from policy {policy.evaluator}"
        )
    raw = {
        name: _metric_value(report, evaluator=evaluator, name=name)
        for name in policy.primary_metrics
    }
    normalized = {
        name: _REGISTRY[name].normalized(value) for name, value in raw.items()
    }
    if policy.aggregation != "mean":
        raise ValueError("metric aggregation protocol is unsupported")
    aggregate = math.fsum(normalized.values()) / len(normalized)
    if not math.isfinite(aggregate):
        raise ValueError("metric aggregate score must be finite")
    secondary = tuple(
        _REGISTRY[name].normalized(
            _metric_value(report, evaluator=evaluator, name=name)
        )
        for name in policy.secondary_metrics
    )
    return ReportScore(
        step=_report_step(report, evaluator=evaluator),
        raw_metrics=raw,
        normalized_metrics=normalized,
        aggregate_score=aggregate,
        secondary_values=secondary,
    )


def select_best_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    evaluator: str,
    policy: MetricPolicy,
) -> SelectionDecision:
    if not candidates:
        raise ValueError("metric selector received no candidates")
    if evaluator == "stage1.route1":
        # Execution settings are recorded evidence, not resume admission gates.
        # Current training allows ordinary batches and framework TF32 defaults.
        reference = candidates[0].get("reasoner_inference")
        if any(row.get("reasoner_inference") != reference for row in candidates[1:]):
            warnings.warn(
                "R evaluation execution metadata differs across candidates; selecting "
                "by recorded validation scores. This does not establish numerical "
                "equivalence across batching or precision settings.",
                RuntimeWarning,
                stacklevel=2,
            )
    scored = [
        (candidate, score_report(candidate, evaluator=evaluator, policy=policy))
        for candidate in candidates
    ]
    steps = [score.step for _candidate, score in scored]
    if len(set(steps)) != len(steps):
        raise ValueError("metric selector candidate steps must be unique")
    winner_candidate, winner = max(
        scored,
        key=lambda item: (
            item[1].aggregate_score,
            item[1].secondary_values,
            item[1].step,
        ),
    )
    aggregate_ties = [
        score
        for _candidate, score in scored
        if score.aggregate_score == winner.aggregate_score
    ]
    if len(aggregate_ties) == 1:
        tie_break = {"kind": "primary"}
    else:
        differing_secondary: str | None = None
        for index, name in enumerate(policy.secondary_metrics):
            values = {score.secondary_values[index] for score in aggregate_ties}
            if len(values) > 1:
                differing_secondary = name
                break
        tie_break = (
            {"kind": "secondary", "metric": differing_secondary}
            if differing_secondary is not None
            else {"kind": "step"}
        )
    return SelectionDecision(
        step=winner.step,
        candidate=winner_candidate,
        raw_metrics=dict(winner.raw_metrics),
        normalized_metrics=dict(winner.normalized_metrics),
        aggregate_score=winner.aggregate_score,
        tie_break=tie_break,
    )
