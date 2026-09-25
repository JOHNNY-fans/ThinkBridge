"""Incremental response-boundary timing; never infer a timestamp from final text."""
from __future__ import annotations

import math
import time
from typing import Any, Callable

RESPONSE_TTFT_CONTRACT = "request-to-first-response-token-full-generation-v1"
SINGLE_TURN_BATCH_TTFT_CONTRACT = "batched-request-to-first-response-token-stop-per-row-v1"
SINGLE_TURN_TTFT_CONTRACT = "request-to-first-response-token-stop-generation-v1"


class ResponseTokenTimer:
    """Observe actual emitted tokens, excluding thinking and control tokens.

    Route1 explicitly closes thinking in its physical prefix, so callers use
    response_started=True. A native thinking model must instead pass False;
    the complete closing marker is consumed before response tokens qualify.
    """

    def __init__(self, tokenizer: Any, *, started: float,
                 synchronize: Callable[[], None],
                 clock: Callable[[], float] = time.perf_counter,
                 response_started: bool = True) -> None:
        self.started = started
        self.synchronize = synchronize
        self.clock = clock
        self.in_response = response_started
        self.special_ids = set(int(x) for x in getattr(tokenizer, "all_special_ids", ()))
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None:
            self.special_ids.add(int(eos))
        self.open_ids = tuple(int(x) for x in tokenizer.encode("<think>", add_special_tokens=False))
        self.open_prefix: list[int] = []
        self.pending_response: tuple[int, float] | None = None
        self.close_ids = tuple(int(x) for x in tokenizer.encode("</think>", add_special_tokens=False))
        if not response_started and not self.close_ids:
            raise ValueError("thinking TTFT requires a nonempty closing marker")
        self.tail: list[int] = []
        self.first_response_token_ids: tuple[int, ...] = ()
        self.seconds: float | None = None
        self.observed_tokens = 0

    def __call__(self, row_index: int, token_id: int) -> None:
        if row_index != 0:
            raise RuntimeError("response TTFT requires batch size one")
        self.observed_tokens += 1
        if self.seconds is not None:
            return
        token_id = int(token_id)
        if not self.in_response:
            self.tail.append(token_id)
            self.tail = self.tail[-len(self.close_ids):]
            if tuple(self.tail) == self.close_ids:
                self.in_response = True
            return
        # A model may emit another thinking block despite the physical prefix.
        # Only a matching opening-marker prefix delays classification; ordinary
        # response tokens are timestamped immediately.
        if self.open_prefix:
            if token_id == self.open_ids[len(self.open_prefix)]:
                self._opening_token(token_id)
                return
            self.open_prefix.clear()
            if self.pending_response is not None:
                self.finish()
                return
        if self.open_ids and token_id == self.open_ids[0]:
            self._opening_token(token_id)
            return
        if token_id not in self.special_ids:
            self.seconds = self._elapsed()
            self.first_response_token_ids = (token_id,)

    def _elapsed(self) -> float:
        self.synchronize()
        elapsed = self.clock() - self.started
        if elapsed < 0.0 or not math.isfinite(elapsed):
            raise RuntimeError("TTFT clock returned an invalid latency")
        return elapsed

    def _opening_token(self, token_id: int) -> None:
        self.open_prefix.append(token_id)
        if tuple(self.open_prefix) == self.open_ids:
            if not self.close_ids:
                raise ValueError("thinking TTFT requires a nonempty closing marker")
            self.in_response = False
            self.tail.clear()
            self.open_prefix.clear()
            self.pending_response = None
        elif self.pending_response is None and token_id not in self.special_ids:
            # A partial text marker could prove to be ordinary response text.
            # Preserve its actual arrival time instead of assigning the later
            # disambiguation time or reading a timestamp from the final string.
            self.pending_response = (token_id, self._elapsed())

    def finish(self) -> None:
        """Resolve an incomplete textual marker at EOS/cap without new timing."""
        if self.seconds is None and self.pending_response is not None:
            token_id, elapsed = self.pending_response
            self.seconds = elapsed
            self.first_response_token_ids = (token_id,)
        self.pending_response = None
