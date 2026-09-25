"""Canonical Bridge Stage1 validation-report path semantics."""

from __future__ import annotations

from pathlib import Path
import re


_VALIDATION_REPORT_ROUTES = frozenset({"route1"})


def _canonical_validation_report_pattern(route: str) -> re.Pattern[str]:
    if route not in _VALIDATION_REPORT_ROUTES:
        raise ValueError("canonical validation report route is invalid")
    return re.compile(
        rf"bridge-{re.escape(route)}-validation-step-(0|[1-9][0-9]*)\.json"
    )


def canonical_validation_report_paths(
    report_dir: str | Path,
    *,
    route: str,
) -> tuple[Path, ...]:
    """Enumerate only canonical reports, ordered by their numeric step."""

    root = Path(report_dir)
    pattern = _canonical_validation_report_pattern(route)
    reports: list[tuple[int, Path]] = []
    for path in root.glob("*.json"):
        match = pattern.fullmatch(path.name)
        if match is not None:
            reports.append((int(match.group(1)), path))
    reports.sort(key=lambda item: item[0])
    return tuple(path for _step, path in reports)
