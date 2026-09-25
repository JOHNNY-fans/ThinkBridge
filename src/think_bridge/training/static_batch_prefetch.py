"""Bounded one-ahead preparation for deterministic CPU training batches."""

from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time
from typing import Callable, Generic, Iterable, Iterator, TypeVar


_SourceT = TypeVar("_SourceT")
_PreparedT = TypeVar("_PreparedT")
_END = object()


@dataclass(frozen=True)
class PreparedBatch(Generic[_PreparedT]):
    value: _PreparedT
    wait_seconds: float
    source_seconds: float = 0.0
    prepare_seconds: float = 0.0


@dataclass(frozen=True)
class _PreparedOutcome(Generic[_PreparedT]):
    value: _PreparedT
    source_seconds: float
    prepare_seconds: float


@dataclass(frozen=True)
class _Failure:
    error: BaseException


class DepthOneBatchPrefetch(Iterator[PreparedBatch[_PreparedT]]):
    """Prepare exactly one next source item on a single bounded worker.

    The worker advances the source only after the consumer takes the previous
    result.  Consequently sampler order, RNG order, and resume cursors are the
    same as a synchronous iterator, while at most one prepared item exists.
    The supplied ``prepare`` function must be finite CPU work; model/CUDA and
    service operations are intentionally outside this abstraction.
    """

    def __init__(
        self,
        source: Iterable[_SourceT],
        *,
        prepare: Callable[[_SourceT], _PreparedT],
        close_timeout_seconds: float,
        wait_reporter: Callable[[str, float, float], None] | None = None,
        wait_report_interval_seconds: float = 1.0,
    ) -> None:
        timeout = float(close_timeout_seconds)
        if timeout <= 0.0:
            raise ValueError("static batch prefetch close timeout must be positive")
        self._source = iter(source)
        self._prepare = prepare
        self._close_timeout_seconds = timeout
        if wait_reporter is not None and not callable(wait_reporter):
            raise TypeError("static batch prefetch wait reporter must be callable")
        report_interval = float(wait_report_interval_seconds)
        if report_interval <= 0.0:
            raise ValueError("static batch prefetch report interval must be positive")
        self._wait_reporter = wait_reporter
        self._wait_report_interval_seconds = report_interval
        self._results: "queue.Queue[object]" = queue.Queue(maxsize=1)
        self._request = threading.Event()
        self._stop = threading.Event()
        self._closed = False
        self._phase_lock = threading.Lock()
        self._phase = "source"
        self._phase_started = time.perf_counter()
        self._thread = threading.Thread(
            target=self._worker,
            name="bridge-static-batch-prefetch",
            daemon=False,
        )
        self._thread.start()
        self._request.set()

    def _set_phase(self, phase: str) -> None:
        with self._phase_lock:
            self._phase = str(phase)
            self._phase_started = time.perf_counter()

    def _phase_snapshot(self) -> tuple[str, float]:
        with self._phase_lock:
            return self._phase, self._phase_started

    def _worker(self) -> None:
        while True:
            self._request.wait()
            self._request.clear()
            if self._stop.is_set():
                return
            self._set_phase("source")
            source_started = time.perf_counter()
            try:
                source_value = next(self._source)
            except StopIteration:
                self._results.put(_END)
                return
            except BaseException as error:
                self._results.put(_Failure(error))
                return
            source_seconds = time.perf_counter() - source_started
            self._set_phase("prepare")
            prepare_started = time.perf_counter()
            try:
                prepared = self._prepare(source_value)
            except BaseException as error:
                self._results.put(_Failure(error))
                return
            prepare_seconds = time.perf_counter() - prepare_started
            self._results.put(
                _PreparedOutcome(
                    value=prepared,
                    source_seconds=source_seconds,
                    prepare_seconds=prepare_seconds,
                )
            )
            self._set_phase("ready")

    def __iter__(self) -> "DepthOneBatchPrefetch[_SourceT, _PreparedT]":
        return self

    def __next__(self) -> PreparedBatch[_PreparedT]:
        if self._closed:
            raise StopIteration
        started = time.perf_counter()
        if self._wait_reporter is None:
            outcome = self._results.get()
        else:
            while True:
                phase, phase_started = self._phase_snapshot()
                now = time.perf_counter()
                self._wait_reporter(
                    phase,
                    max(now - phase_started, 0.0),
                    max(now - started, 0.0),
                )
                try:
                    outcome = self._results.get(
                        timeout=self._wait_report_interval_seconds
                    )
                    break
                except queue.Empty:
                    continue
        wait_seconds = time.perf_counter() - started
        if outcome is _END:
            self.close()
            raise StopIteration
        if isinstance(outcome, _Failure):
            error = outcome.error
            self.close()
            raise error
        if not isinstance(outcome, _PreparedOutcome):
            self.close()
            raise RuntimeError("static batch prefetch produced an invalid outcome")
        self._request.set()
        return PreparedBatch(
            value=outcome.value,
            wait_seconds=wait_seconds,
            source_seconds=outcome.source_seconds,
            prepare_seconds=outcome.prepare_seconds,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._request.set()
        self._thread.join(timeout=self._close_timeout_seconds)
        if self._thread.is_alive():
            raise RuntimeError(
                "static batch prefetch worker exceeded its bounded close timeout"
            )

    def __enter__(self) -> "DepthOneBatchPrefetch[_SourceT, _PreparedT]":
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> bool:
        self.close()
        return False
