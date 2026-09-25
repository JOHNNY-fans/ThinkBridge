"""Resolve one invocation-bound Bridge split-stage run without directory scans."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from think_bridge.utils.run_dir import resolve_stage_run_locator


_STAGES = ("reasoner-sft",)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m think_bridge.training.stage_run_locator_cli"
    )
    parser.add_argument("--locator", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=_STAGES, required=True)
    parser.add_argument("--require-completed", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        run = resolve_stage_run_locator(
            arguments.locator,
            project_dir=arguments.project_dir,
            stage=arguments.stage,
            require_completed=bool(arguments.require_completed),
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
