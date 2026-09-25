"""Public ``think-bridge reasoner-sft`` driver."""

from __future__ import annotations


def main() -> int:
    from think_bridge.training.stage_driver import run_reasoner_driver

    return run_reasoner_driver()


if __name__ == "__main__":
    raise SystemExit(main())
