"""Fail-closed CLI for passing one selected Bridge checkpoint between stages."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from think_bridge.model.checkpoint_policy import (
    parse_checkpoint_path,
    read_checkpoint_pointer,
)


_CHECKPOINT_ROUTES = frozenset({"route1"})


def resolve_stage_checkpoint_pointer(
    pointer: Path, *, run_dir: Path, expected_route: str
) -> Path:
    """Return one fully validated checkpoint from the required stage route."""

    if expected_route not in _CHECKPOINT_ROUTES:
        raise ValueError("expected checkpoint route is invalid")
    checkpoint = read_checkpoint_pointer(Path(pointer), run_dir=Path(run_dir))
    route, _ = parse_checkpoint_path(checkpoint)
    if route != expected_route:
        raise ValueError(
            f"wrong checkpoint route: expected {expected_route}, observed {route}"
        )
    return checkpoint.resolve(strict=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m think_bridge.training.checkpoint_pointer_cli"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pointer", type=Path, required=True)
    parser.add_argument(
        "--expected-route", choices=sorted(_CHECKPOINT_ROUTES), required=True
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        checkpoint = resolve_stage_checkpoint_pointer(
            arguments.pointer,
            run_dir=arguments.run_dir,
            expected_route=arguments.expected_route,
        )
    except (KeyError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
