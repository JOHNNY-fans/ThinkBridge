"""Bounded, observable Math-Verify judging for saved benchmark answers.

Keep verify's candidate search and equivalence rules intact. In particular,
raise_on_error would abort at the first timed-out candidate and can miss a
later successful comparison, so detect its timeout log records instead.
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
import threading

JUDGE_PROTOCOL = "math-verify-timeout-10s-retry15s-v1"
TIMEOUT_SECONDS = (10, 15)


class _TimeoutCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.thread = threading.get_ident()
        self.count = 0

    def emit(self, record):
        if record.thread == self.thread and record.getMessage().startswith(
            "Timeout during "
        ):
            self.count += 1


@contextmanager
def _capture_timeouts():
    logger = logging.getLogger("math_verify")
    capture = _TimeoutCapture()
    logger.addHandler(capture)
    try:
        yield capture
    finally:
        logger.removeHandler(capture)


def verify_with_diagnostics(gold, pred, *, parse, verify, diagnostics):
    from math_verify.errors import TimeoutException

    diagnostics.update(
        protocol=JUDGE_PROTOCOL,
        status="ok",
        attempts=0,
        timeout_events=0,
        timeout_seconds=list(TIMEOUT_SECONDS),
    )
    # Budgets apply to each parse and each candidate comparison, as specified
    # by Math-Verify, not to generation or the total wall time of the row.
    for attempt, seconds in enumerate(TIMEOUT_SECONDS, 1):
        with _capture_timeouts() as captured:
            try:
                gp = parse(f"\\boxed{{{gold}}}", parsing_timeout=seconds)
                pp = parse(f"\\boxed{{{pred}}}", parsing_timeout=seconds)
                correct = bool(verify(gp, pp, timeout_seconds=seconds))
            except TimeoutException:
                captured.count += 1
                correct = False
            except Exception:
                correct = False
        diagnostics["attempts"] = attempt
        diagnostics["timeout_events"] += captured.count
        if correct or captured.count == 0:
            return correct
    diagnostics["status"] = "timeout"
    return False
