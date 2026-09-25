"""Bounded dependency-light client for the private Bridge Route1 service."""

from __future__ import annotations

from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    TimeoutError as FutureTimeout,
    wait,
)
from dataclasses import dataclass, replace
import hashlib
import http.client
import json
import math
import socket
import threading
import time
from typing import Any, Callable, Iterable, Iterator, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request

from think_bridge.training.vllm_runtime import (
    ROUTE1_GENERATION_PROBE_STATUS,
    Route1ServiceRequest,
    Route1ServiceResponse,
    TensorPayload,
)


class BridgeServiceHTTPError(RuntimeError):
    """A bounded HTTP error returned by the private generation service."""


def route1_service_failure_event(
    error: BaseException,
    *,
    rank: int,
    detail_limit: int = 2048,
) -> dict[str, Any]:
    """Describe one local trainer failure without logging a large traceback."""

    limit = max(1, int(detail_limit))
    root = error
    seen: set[int] = set()
    for _ in range(8):
        if id(root) in seen:
            break
        seen.add(id(root))
        nested = root.__cause__ if root.__cause__ is not None else root.__context__
        if nested is None:
            break
        root = nested
    return {
        "event": "bridge-route1-trainer-local-service-error",
        "rank": int(rank),
        "exception": type(error).__name__,
        "detail": str(error)[:limit],
        "root_cause_exception": type(root).__name__,
        "root_cause_detail": str(root)[:limit],
    }


class _HTTPConnectionResponse:
    """Context-manage one response and unregister its private connection."""

    def __init__(
        self,
        connection: http.client.HTTPConnection,
        response: http.client.HTTPResponse,
        release: Callable[[http.client.HTTPConnection], None],
    ) -> None:
        self._connection = connection
        self._response = response
        self._release = release
        self.status = int(response.status)
        self.headers = response.headers

    def read(self, size: int = -1) -> bytes:
        return self._response.read(size)

    def __enter__(self) -> "_HTTPConnectionResponse":
        return self

    def __exit__(self, _type, _value, _traceback) -> bool:
        try:
            self._response.close()
        finally:
            self._connection.close()
            self._release(self._connection)
        return False


class _LoopbackHTTPTransport:
    """Abortable one-connection-per-request loopback HTTP transport."""

    def __init__(self, host: str, port: int) -> None:
        self.address = (str(host), int(port))
        self._lock = threading.Lock()
        self._active: set[http.client.HTTPConnection] = set()

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    def _release(self, connection: http.client.HTTPConnection) -> None:
        with self._lock:
            self._active.discard(connection)

    def open(self, request: Request, *, timeout: float) -> _HTTPConnectionResponse:
        connection = http.client.HTTPConnection(
            self.address[0], self.address[1], timeout=float(timeout)
        )
        with self._lock:
            self._active.add(connection)
        try:
            connection.request(
                request.get_method(),
                request.selector,
                body=request.data,
                headers=dict(request.header_items()),
            )
            response = connection.getresponse()
        except BaseException:
            connection.close()
            self._release(connection)
            raise
        return _HTTPConnectionResponse(connection, response, self._release)

    def abort_pending(self) -> None:
        with self._lock:
            connections = tuple(self._active)
        for connection in connections:
            active_socket = connection.sock
            if active_socket is not None:
                try:
                    active_socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            connection.close()


class BridgeRoute1ServiceClient:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        timeout_seconds: float,
        teardown_timeout_seconds: float | None = None,
        max_response_bytes: int,
        max_in_flight: int,
        opener: Callable[..., Any] | None = None,
        abort_in_flight: Callable[[], None] | None = None,
        wait_reporter: Callable[[str], None] | None = None,
        watchdog_interval_seconds: float = 60.0,
    ) -> None:
        if host != "127.0.0.1":
            raise ValueError("Bridge Route1 service client requires 127.0.0.1")
        if isinstance(port, bool) or int(port) <= 0:
            raise ValueError("Bridge Route1 service port is invalid")
        if not math.isfinite(float(timeout_seconds)) or timeout_seconds <= 0:
            raise ValueError("Bridge Route1 request timeout is invalid")
        if int(max_response_bytes) <= 0 or int(max_in_flight) <= 0:
            raise ValueError("Bridge Route1 client bounds must be positive")
        watchdog_interval = float(watchdog_interval_seconds)
        if not math.isfinite(watchdog_interval) or watchdog_interval <= 0.0:
            raise ValueError("Bridge Route1 watchdog interval is invalid")
        teardown_timeout = (
            min(float(timeout_seconds), 5.0)
            if teardown_timeout_seconds is None
            else float(teardown_timeout_seconds)
        )
        if not math.isfinite(teardown_timeout) or teardown_timeout <= 0.0:
            raise ValueError("Bridge Route1 teardown timeout is invalid")
        self._base_url = f"http://{host}:{int(port)}"
        self._timeout = float(timeout_seconds)
        self._teardown_timeout = teardown_timeout
        self._max_response_bytes = int(max_response_bytes)
        self._max_in_flight = int(max_in_flight)
        self._wait_reporter = wait_reporter
        self._watchdog_interval = watchdog_interval
        self._last_stream_peak_pending = 0
        self._last_stream_wall_seconds = 0.0
        self._last_stream_overlap_seconds = 0.0
        # One registered HTTPConnection per request makes loopback traffic
        # proxy-independent and lets a first peer failure close every blocked
        # socket before joining the bounded worker pool.
        self._transport = _LoopbackHTTPTransport(host, int(port))
        self._opener = self._transport.open if opener is None else opener
        self._abort_in_flight = (
            self._transport.abort_pending
            if abort_in_flight is None
            else abort_in_flight
        )

    @property
    def max_queued_requests(self) -> int:
        return 2 * self._max_in_flight

    @property
    def last_stream_peak_pending(self) -> int:
        return int(self._last_stream_peak_pending)

    @property
    def last_stream_wall_seconds(self) -> float:
        return float(self._last_stream_wall_seconds)

    @property
    def last_stream_overlap_seconds(self) -> float:
        return float(self._last_stream_overlap_seconds)

    def _read_bounded(self, response) -> bytes:
        content_length = response.headers.get("Content-Length")
        if (
            content_length is not None
            and int(content_length) > self._max_response_bytes
        ):
            raise RuntimeError(
                "Bridge Route1 service response exceeds configured bound"
            )
        payload = response.read(self._max_response_bytes + 1)
        if len(payload) > self._max_response_bytes:
            raise RuntimeError(
                "Bridge Route1 service response exceeds configured bound"
            )
        return payload

    @staticmethod
    def _read_error_detail(response, *, limit: int = 8192) -> str:
        """Read one bounded service error body from either HTTP transport."""

        payload = response.read(int(limit) + 1)
        truncated = len(payload) > int(limit)
        payload = payload[: int(limit)]
        text = payload.decode("utf-8", errors="replace").strip()
        try:
            decoded = json.loads(text)
        except (UnicodeError, json.JSONDecodeError):
            decoded = None
        if isinstance(decoded, dict):
            error = str(decoded.get("error", "service-error"))
            detail = str(decoded.get("detail", ""))
            text = f"{error}: {detail}" if detail else error
        if not text:
            text = "empty error response"
        if truncated:
            text += " [truncated]"
        return text

    def health(self, expected_instance_token: str | None = None) -> bool:
        request = Request(self._base_url + "/health", method="GET")
        try:
            with self._opener(request, timeout=self._timeout) as response:
                if response.status != 200:
                    return False
                payload = self._read_bounded(response)
        except (
            HTTPError,
            URLError,
            TimeoutError,
            OSError,
            ValueError,
            http.client.HTTPException,
        ):
            return False
        try:
            status = json.loads(payload.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            return False
        if not isinstance(status, dict):
            return False
        instance_token = status.get("instance_token")
        if (
            status.get("status") != "ready"
            or status.get("probe_status") != ROUTE1_GENERATION_PROBE_STATUS
            or not isinstance(instance_token, str)
            or not instance_token
        ):
            return False
        return (
            expected_instance_token is None or instance_token == expected_instance_token
        )

    def submit(self, request: Route1ServiceRequest) -> Route1ServiceResponse:
        http_request = Request(
            self._base_url + "/route1",
            data=request.to_wire(),
            method="POST",
            headers={"Content-Type": "application/x-think-bridge-route1"},
        )
        try:
            with self._opener(http_request, timeout=self._timeout) as response:
                if response.status != 200:
                    detail = self._read_error_detail(response)
                    raise BridgeServiceHTTPError(
                        "Bridge Route1 service HTTP failure "
                        f"status={response.status}: {detail}"
                    )
                payload = self._read_bounded(response)
        except HTTPError as exc:
            detail = self._read_error_detail(exc)
            raise BridgeServiceHTTPError(
                f"Bridge Route1 service HTTP failure status={exc.code}: {detail}"
            ) from exc
        except (URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
            raise RuntimeError(
                "Bridge Route1 service request failed or timed out"
            ) from exc
        response = Route1ServiceResponse.from_wire(payload)
        response.assert_matches(request)
        return response

    def abort_pending(self) -> None:
        """Abort every live loopback socket owned by this client."""

        self._abort_in_flight()

    def stream(
        self, requests: Iterable[Route1ServiceRequest]
    ) -> Iterator[Route1ServiceResponse]:
        request_iterator = iter(requests)
        executor = ThreadPoolExecutor(
            max_workers=self._max_in_flight,
            thread_name_prefix="bridge-service",
        )
        futures: dict[Future[Route1ServiceResponse], Route1ServiceRequest] = {}
        completion_times: dict[Future[Route1ServiceResponse], float] = {}
        completion_lock = threading.Lock()
        queue_bound = 2 * self._max_in_flight
        exhausted = False
        primary_error: BaseException | None = None

        def record_completion(future: Future[Route1ServiceResponse]) -> None:
            with completion_lock:
                completion_times[future] = time.monotonic()

        def refill() -> None:
            nonlocal exhausted
            while not exhausted and len(futures) < queue_bound:
                try:
                    request = next(request_iterator)
                except StopIteration:
                    exhausted = True
                    break
                future = executor.submit(self.submit, request)
                future.add_done_callback(record_completion)
                futures[future] = request
                self._last_stream_peak_pending = max(
                    self._last_stream_peak_pending, len(futures)
                )

        try:
            self._last_stream_peak_pending = 0
            self._last_stream_wall_seconds = 0.0
            self._last_stream_overlap_seconds = 0.0
            stream_started = time.monotonic()
            refill()
            if not futures:
                raise ValueError("Bridge Route1 stream has no requests")
            wait_started = time.monotonic()
            while futures:
                completed, _pending = wait(
                    tuple(futures),
                    timeout=self._watchdog_interval,
                    return_when=FIRST_COMPLETED,
                )
                if not completed:
                    if self._wait_reporter is not None:
                        pending = sorted(
                            (
                                int(
                                    getattr(
                                        request, "step", getattr(request, "cycle", 0)
                                    )
                                ),
                                int(
                                    getattr(
                                        request,
                                        "micro_step",
                                        getattr(request, "update", 0),
                                    )
                                ),
                                int(request.chunk_id),
                            )
                            for request in futures.values()
                        )
                        self._wait_reporter(
                            "Bridge Route1 watchdog "
                            f"elapsed={time.monotonic() - wait_started:.1f}s "
                            "stage=service-wait "
                            f"requests={pending}"
                        )
                    continue
                failures = []
                for future in completed:
                    failure = future.exception()
                    if failure is not None:
                        failures.append((future, failure))
                if failures:
                    # Prefer a service-authored HTTP response over peer socket
                    # errors caused by aborting the rest of the same burst.
                    _future, failure = min(
                        failures,
                        key=lambda item: (
                            not isinstance(item[1], BridgeServiceHTTPError),
                            completion_times.get(item[0], time.monotonic()),
                        ),
                    )
                    raise failure
                for future in completed:
                    futures.pop(future)
                    response = future.result()
                    # Refill before yielding: with one trainer-side worker, the
                    # next same-step request starts while live-F consumes this
                    # response.  The queue remains strictly bounded and never
                    # crosses optimizer-step ownership.
                    refill()
                    active_at_yield = tuple(
                        pending_future
                        for pending_future in futures
                        if not pending_future.done()
                    )
                    yielded_at = time.monotonic()
                    yield response
                    resumed_at = time.monotonic()
                    if active_at_yield:
                        with completion_lock:
                            finished = tuple(
                                completion_times.get(pending_future, resumed_at)
                                for pending_future in active_at_yield
                            )
                        active_until = min(resumed_at, max(finished))
                        self._last_stream_overlap_seconds += max(
                            0.0, active_until - yielded_at
                        )
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            self._last_stream_wall_seconds = max(0.0, time.monotonic() - stream_started)
            for future in futures:
                future.cancel()
            # Prefetched microsteps share this client/transport. Exhausting one
            # stream normally must not close another stream's active sockets.
            # Unfinished work means cancellation/failure; those remain fatal to
            # the update and must unblock outstanding HTTP calls before join.
            if futures:
                self._abort_in_flight()
            _done, pending = wait(tuple(futures), timeout=self._teardown_timeout)
            if pending:
                executor.shutdown(wait=False, cancel_futures=True)
                if primary_error is None:
                    raise RuntimeError(
                        "Bridge Route1 client teardown exceeded its bounded timeout "
                        f"with {len(pending)} request(s) still active"
                    )
            else:
                executor.shutdown(wait=True, cancel_futures=True)


@dataclass(frozen=True)
class Route1AsyncTelemetry:
    submit_seconds: float
    service_seconds: float
    queue_seconds: float
    generation_seconds: float
    overlap_seconds: float
    wait_seconds: float
    token_count: int
    request_count: int
    service_batch_rows: int
    service_real_rows: int
    service_active_requests: int

    @property
    def overlap_efficiency(self) -> float:
        if self.service_seconds <= 0.0:
            return 0.0
        return min(1.0, max(0.0, self.overlap_seconds / self.service_seconds))

    @property
    def service_occupancy(self) -> float:
        if self.service_batch_rows <= 0:
            return 0.0
        return float(self.service_real_rows) / float(self.service_batch_rows)


@dataclass(frozen=True)
class _AsyncWorkResult:
    responses: tuple[Route1ServiceResponse, ...]
    started_at: float
    finished_at: float


class Route1AsyncTicket:
    """One current-update microstep request set with bounded ownership."""

    def __init__(
        self,
        *,
        pool: "Route1AsyncRequestPool",
        requests: tuple[Route1ServiceRequest, ...],
        future: Future[_AsyncWorkResult],
        submitted_at: float,
        submit_seconds: float,
    ) -> None:
        self._pool = pool
        self.requests = requests
        self.future = future
        self.submitted_at = float(submitted_at)
        self.submit_seconds = float(submit_seconds)
        self._finished = False

    @property
    def update(self) -> int:
        return int(self.requests[0].step)

    @property
    def micro_step(self) -> int:
        return int(self.requests[0].micro_step)

    def cancel(self) -> None:
        self._pool.cancel(self)


class Route1AsyncRequestPool:
    """Bounded submit/overlap/await pool for one trainer rank.

    The pool never crosses an optimizer update: callers must resolve/cancel all
    tickets and call ``finish_update`` before clipping or stepping.
    """

    def __init__(
        self,
        *,
        client: Any,
        max_pending_microsteps: int,
        request_timeout_seconds: float,
        backpressure_timeout_seconds: float,
        teardown_timeout_seconds: float = 5.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        bounds = (
            int(max_pending_microsteps),
            float(request_timeout_seconds),
            float(backpressure_timeout_seconds),
            float(teardown_timeout_seconds),
        )
        if bounds[0] <= 0 or any(
            not math.isfinite(value) or value <= 0.0 for value in bounds[1:]
        ):
            raise ValueError("Route1 async request-pool bounds are invalid")
        self._client = client
        self._max_pending = bounds[0]
        self._request_timeout = bounds[1]
        self._backpressure_timeout = bounds[2]
        self._teardown_timeout = bounds[3]
        self._monotonic = monotonic
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_pending,
            thread_name_prefix="bridge-async-rollout",
        )
        self._slots = threading.BoundedSemaphore(self._max_pending)
        self._lock = threading.Lock()
        self._active: dict[tuple[int, int], Route1AsyncTicket] = {}
        self._seen: set[tuple[Any, ...]] = set()
        self._completed_update = -1
        self._open_update: int | None = None
        self._closed = False

    @staticmethod
    def _validate_request_set(
        requests: Sequence[Route1ServiceRequest],
    ) -> tuple[Route1ServiceRequest, ...]:
        values = tuple(requests)
        if not values:
            raise ValueError("Route1 async submission has no service requests")
        owner = (
            values[0].run_id,
            values[0].rank,
            values[0].transaction_id,
            values[0].step,
            values[0].micro_step,
        )
        if any(
            (
                value.run_id,
                value.rank,
                value.transaction_id,
                value.step,
                value.micro_step,
            )
            != owner
            for value in values
        ):
            raise ValueError("Route1 async request set mixes microstep owners")
        chunk_ids = tuple(int(value.chunk_id) for value in values)
        if chunk_ids != tuple(range(len(values))):
            raise ValueError(
                "Route1 async request chunks are not contiguous and ordered"
            )
        if len({value.identity for value in values}) != len(values):
            raise ValueError("Route1 async request set contains duplicate identities")
        return values

    def _execute(self, requests: tuple[Route1ServiceRequest, ...]) -> _AsyncWorkResult:
        started = self._monotonic()
        responses = tuple(self._client.stream(requests))
        return _AsyncWorkResult(
            responses=responses,
            started_at=started,
            finished_at=self._monotonic(),
        )

    def submit(self, requests: Sequence[Route1ServiceRequest]) -> Route1AsyncTicket:
        values = self._validate_request_set(requests)
        submitted_at = self._monotonic()
        key = (int(values[0].step), int(values[0].micro_step))
        identities = {value.identity for value in values}
        with self._lock:
            if self._closed:
                raise RuntimeError("Route1 async request pool is closed")
            if int(values[0].step) <= self._completed_update:
                raise RuntimeError("Route1 async request is stale")
            if (
                self._open_update is not None
                and int(values[0].step) != self._open_update
            ):
                raise RuntimeError("Route1 async request crosses an optimizer update")
            if identities.intersection(self._seen) or key in self._active:
                raise RuntimeError("Route1 async request is duplicate")
            if self._open_update is None:
                self._open_update = int(values[0].step)
        if not self._slots.acquire(timeout=self._backpressure_timeout):
            raise TimeoutError("Route1 async request backpressure timed out")
        try:
            with self._lock:
                if self._closed:
                    raise RuntimeError("Route1 async request pool is closed")
                self._seen.update(identities)
                future = self._executor.submit(self._execute, values)
                ticket = Route1AsyncTicket(
                    pool=self,
                    requests=values,
                    future=future,
                    submitted_at=submitted_at,
                    submit_seconds=self._monotonic() - submitted_at,
                )
                self._active[key] = ticket
                return ticket
        except BaseException:
            self._slots.release()
            raise

    def _finish_ticket(self, ticket: Route1AsyncTicket) -> None:
        with self._lock:
            self._finish_ticket_locked(ticket)

    def _finish_ticket_locked(self, ticket: Route1AsyncTicket) -> None:
        if ticket._finished:
            return
        key = (ticket.update, ticket.micro_step)
        if self._active.get(key) is ticket:
            self._active.pop(key)
        ticket._finished = True
        self._slots.release()

    def resolve(
        self, ticket: Route1AsyncTicket
    ) -> tuple[tuple[Route1ServiceResponse, ...], Route1AsyncTelemetry]:
        if ticket._pool is not self:
            raise ValueError("Route1 async ticket belongs to another pool")
        wait_started = self._monotonic()
        try:
            try:
                work = ticket.future.result(timeout=self._request_timeout)
            except FutureTimeout as exc:
                self._client.abort_pending()
                ticket.future.cancel()
                raise TimeoutError("Route1 async service request timed out") from exc
            by_identity = {response.identity: response for response in work.responses}
            expected = {request.identity for request in ticket.requests}
            if set(by_identity) != expected or len(by_identity) != len(work.responses):
                raise RuntimeError(
                    "Route1 async response set is incomplete or duplicate"
                )
            ordered = tuple(
                by_identity[request.identity] for request in ticket.requests
            )
            for request, response in zip(ticket.requests, ordered):
                response.assert_matches(request)
            resolved_at = self._monotonic()
            wait_seconds = resolved_at - wait_started
            service_seconds = max(0.0, work.finished_at - work.started_at)
            overlap_seconds = max(0.0, service_seconds - wait_seconds)
            telemetry = Route1AsyncTelemetry(
                submit_seconds=float(ticket.submit_seconds),
                service_seconds=service_seconds,
                queue_seconds=sum(
                    float(response.timings["service_queue_seconds"])
                    for response in ordered
                ),
                generation_seconds=sum(
                    float(response.timings["rollout_generation_seconds"])
                    for response in ordered
                ),
                overlap_seconds=overlap_seconds,
                wait_seconds=wait_seconds,
                token_count=sum(
                    len(token_ids)
                    for response in ordered
                    for token_ids, padding in zip(
                        response.prefix_token_ids,
                        response.sample_is_padding,
                    )
                    if not padding
                ),
                request_count=len(ordered),
                service_batch_rows=sum(
                    int(response.counts["service_batch_rows"]) for response in ordered
                ),
                service_real_rows=sum(
                    int(response.counts["service_real_rows"]) for response in ordered
                ),
                service_active_requests=max(
                    int(response.counts["service_active_requests"])
                    for response in ordered
                ),
            )
            return ordered, telemetry
        except BaseException:
            self._client.abort_pending()
            raise
        finally:
            self._finish_ticket(ticket)

    def cancel(self, ticket: Route1AsyncTicket) -> None:
        if ticket._pool is not self:
            raise ValueError("Route1 async ticket belongs to another pool")
        with self._lock:
            # CUDA DDP copies nested keyword dicts. A caller's `resolved`
            # marker can remain false after the model consumed this ticket.
            # The pool owns completion; disposing a completed ticket must not
            # abort the shared transport and disconnect other microsteps.
            if ticket._finished:
                return
            self._client.abort_pending()
            ticket.future.cancel()
            self._finish_ticket_locked(ticket)

    def finish_update(self, update: int) -> None:
        value = int(update)
        with self._lock:
            if value <= self._completed_update:
                raise RuntimeError("Route1 async update completion is stale")
            if self._open_update is not None and value != self._open_update:
                raise RuntimeError(
                    "Route1 async completion differs from its open update"
                )
            pending = sorted(key for key in self._active if key[0] <= value)
            if pending:
                raise RuntimeError(
                    f"Route1 async update has pending requests: {pending}"
                )
            self._completed_update = value
            self._open_update = None
            self._seen = {
                identity for identity in self._seen if int(identity[3]) > value
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            tickets = tuple(self._active.values())
        self._client.abort_pending()
        for ticket in tickets:
            ticket.future.cancel()
        _done, pending = wait(
            tuple(ticket.future for ticket in tickets),
            timeout=self._teardown_timeout,
        )
        for ticket in tickets:
            self._finish_ticket(ticket)
        self._executor.shutdown(wait=not pending, cancel_futures=True)
        if pending:
            raise RuntimeError(
                "Route1 async request-pool teardown exceeded its bounded timeout"
            )


@dataclass(frozen=True)
class Route1PrefixTicket:
    pool_ticket: Route1AsyncTicket | None
    real_rows_per_request: tuple[int, ...]
    view_count: int
    submit_seconds: float


@dataclass(frozen=True)
class Route1GeneratedPrefixBatch:
    token_ids: Any
    token_mask: Any
    telemetry: Route1AsyncTelemetry


class BridgeRoute1AsyncPrefixProvider:
    """Torch adapter that transports only detached current-microstep z."""

    def __init__(
        self,
        *,
        pool: Route1AsyncRequestPool,
        run_id: str,
        rank: int,
        physical_chunk_size: int,
        generation_seed: int,
        max_prefix_tokens: int,
        boundary_ids: Sequence[int],
        eos_token_id: int,
        synchronize_failure: Callable[[BaseException | None], None],
        temperature: float = 0.0,
    ) -> None:
        if int(physical_chunk_size) <= 0 or int(max_prefix_tokens) <= 0:
            raise ValueError("Route1 async prefix geometry is invalid")
        self._pool = pool
        self._run_id = str(run_id)
        self._rank = int(rank)
        self._physical = int(physical_chunk_size)
        self._generation_seed = int(generation_seed)
        self._temperature = float(temperature)
        self._max_prefix_tokens = int(max_prefix_tokens)
        self._boundary_ids = tuple(int(value) for value in boundary_ids)
        self._eos_token_id = int(eos_token_id)
        self._synchronize_failure = synchronize_failure

    def submit(
        self,
        *,
        update: int,
        micro_step: int,
        sample_keys: Sequence[str],
        prompt_ids: Any,
        prompt_mask: Any,
        true_z: Any,
        view_to_sample: Any,
        cot_ids: Any = None,
        cot_mask: Any = None,
    ) -> Route1PrefixTicket:
        import torch

        started = time.monotonic()
        view_indices = tuple(
            int(value) for value in view_to_sample.detach().cpu().tolist()
        )
        if not view_indices:
            return Route1PrefixTicket(None, (), 0, time.monotonic() - started)
        if len(sample_keys) != int(prompt_ids.size(0)):
            raise ValueError("Route1 async sample identities differ from prompt rows")
        if (cot_ids is None) != (cot_mask is None):
            raise ValueError("Route1 async course hint requires both ids and mask")
        if cot_ids is not None and (
            cot_ids.ndim != 2
            or cot_ids.shape != cot_mask.shape
            or int(cot_ids.size(0)) != len(view_indices)
        ):
            raise ValueError(
                "Route1 async course hint must align with distillation views"
            )
        owner = torch.as_tensor(view_indices, dtype=torch.long, device=true_z.device)
        # Transport one detached microstep snapshot, then split/pad on CPU.
        # Per-chunk GPU indexing and TensorPayload's finite check otherwise
        # introduce another device/host boundary for every physical request.
        detached_view_z = true_z.index_select(0, owner).detach().cpu()
        if detached_view_z.requires_grad:
            raise RuntimeError("Route1 async transport z still owns a gradient")
        # These are immutable token/mask metadata, already prepared on CPU by
        # the trainer. Copy each tensor back once instead of synchronizing a
        # masked_select().cpu() for every owner and every visible course hint.
        prompt_ids_cpu = prompt_ids.detach().cpu()
        prompt_mask_cpu = prompt_mask.detach().cpu().bool()
        cot_ids_cpu = None if cot_ids is None else cot_ids.detach().cpu()
        cot_mask_cpu = None if cot_mask is None else cot_mask.detach().cpu().bool()
        prompt_rows: list[tuple[int, ...]] = []
        view_keys: list[str] = []
        for view_index, sample_index in enumerate(view_indices):
            valid = prompt_mask_cpu[sample_index]
            row = tuple(
                int(value)
                for value in prompt_ids_cpu[sample_index].masked_select(valid).tolist()
            )
            if not row:
                raise ValueError("Route1 async prompt row is empty")
            # The service appends z and the answer boundary. Concatenate the
            # owner's visible hint here so online sampling and HF scoring use
            # exactly q + hint + z + boundary, without changing R's q-only input.
            if cot_ids is not None:
                hint = tuple(
                    int(value)
                    for value in cot_ids_cpu[view_index]
                    .masked_select(cot_mask_cpu[view_index])
                    .tolist()
                )
                row += hint
            prompt_rows.append(row)
            view_keys.append(str(sample_keys[sample_index]))
        transaction_digest = hashlib.sha256(
            (
                f"{self._run_id}\x1f{self._rank}\x1f{int(update)}\x1f{int(micro_step)}"
            ).encode("utf-8")
        ).hexdigest()
        transaction = f"tx-{transaction_digest[:40]}"
        requests: list[Route1ServiceRequest] = []
        real_rows: list[int] = []
        for chunk_id, start in enumerate(range(0, len(view_indices), self._physical)):
            stop = min(start + self._physical, len(view_indices))
            indices = list(range(start, stop))
            real_count = len(indices)
            while len(indices) < self._physical:
                indices.append(indices[-1])
            selected = torch.as_tensor(
                indices, dtype=torch.long, device=detached_view_z.device
            )
            chunk_z = detached_view_z.index_select(0, selected).detach()
            chunk_prompts = tuple(prompt_rows[index] for index in indices)
            chunk_keys = tuple(
                (
                    view_keys[index]
                    if offset < real_count
                    else f"{view_keys[index]}-padding-{chunk_id}-{offset}"
                )
                for offset, index in enumerate(indices)
            )
            requests.append(
                Route1ServiceRequest(
                    run_id=self._run_id,
                    transaction_id=transaction,
                    rank=self._rank,
                    step=int(update),
                    micro_step=int(micro_step),
                    chunk_id=int(chunk_id),
                    sample_ids=tuple(
                        self._rank * self._physical + start + offset
                        for offset in range(self._physical)
                    ),
                    sample_keys=chunk_keys,
                    sample_is_padding=tuple(
                        offset >= real_count for offset in range(self._physical)
                    ),
                    prompt_ids=chunk_prompts,
                    prompt_lengths=tuple(len(row) for row in chunk_prompts),
                    detached_z=TensorPayload.from_torch(chunk_z, wire_dtype="float32"),
                    boundary_ids=self._boundary_ids,
                    generation_seed=self._generation_seed,
                    temperature=self._temperature,
                    max_prefix_tokens=self._max_prefix_tokens,
                    z_present=(True,) * self._physical,
                    append_boundary=(True,) * self._physical,
                    eos_token_id=self._eos_token_id,
                )
            )
            real_rows.append(real_count)
        pool_ticket = self._pool.submit(tuple(requests))
        return Route1PrefixTicket(
            pool_ticket=pool_ticket,
            real_rows_per_request=tuple(real_rows),
            view_count=len(view_indices),
            submit_seconds=time.monotonic() - started,
        )

    def resolve(
        self,
        ticket: Route1PrefixTicket | None,
        *,
        device: Any,
        local_error: BaseException | None = None,
        view_count: int = 0,
    ) -> Route1GeneratedPrefixBatch:
        import torch

        responses: tuple[Route1ServiceResponse, ...] = ()
        if ticket is None and local_error is None:
            raise ValueError("Route1 async resolve lacks a ticket or local error")
        resolved_view_count = int(
            ticket.view_count if ticket is not None else view_count
        )
        telemetry = Route1AsyncTelemetry(
            submit_seconds=float(ticket.submit_seconds if ticket is not None else 0.0),
            service_seconds=0.0,
            queue_seconds=0.0,
            generation_seconds=0.0,
            overlap_seconds=0.0,
            wait_seconds=0.0,
            token_count=0,
            request_count=0,
            service_batch_rows=0,
            service_real_rows=0,
            service_active_requests=0,
        )
        if (
            ticket is not None
            and ticket.pool_ticket is not None
            and local_error is None
        ):
            try:
                responses, telemetry = self._pool.resolve(ticket.pool_ticket)
                telemetry = replace(
                    telemetry,
                    submit_seconds=float(ticket.submit_seconds),
                )
            except BaseException as exc:
                local_error = exc
        self._synchronize_failure(local_error)
        if local_error is not None:
            raise RuntimeError(
                "Route1 async generation failed locally"
            ) from local_error
        materialization_error: BaseException | None = None
        token_ids = None
        token_mask = None
        try:
            token_ids = torch.full(
                (resolved_view_count, self._max_prefix_tokens),
                self._eos_token_id,
                dtype=torch.long,
                device=device,
            )
            token_mask = torch.zeros_like(token_ids, dtype=torch.bool)
            destination = 0
            for response, real_count in zip(
                responses, (() if ticket is None else ticket.real_rows_per_request)
            ):
                for row in response.prefix_token_ids[:real_count]:
                    width = len(row)
                    token_ids[destination, :width] = torch.as_tensor(
                        row, dtype=torch.long, device=device
                    )
                    token_mask[destination, :width] = True
                    destination += 1
            if destination != resolved_view_count:
                raise RuntimeError("Route1 async prefix response order is incomplete")
        except BaseException as exc:
            materialization_error = exc
        self._synchronize_failure(materialization_error)
        if materialization_error is not None:
            raise RuntimeError(
                "Route1 async response materialization failed locally"
            ) from materialization_error
        if token_ids is None or token_mask is None:
            raise AssertionError("Route1 response materialization produced no tensors")
        return Route1GeneratedPrefixBatch(token_ids, token_mask, telemetry)

    def discard(self, ticket: Route1PrefixTicket | None) -> None:
        if ticket is not None and ticket.pool_ticket is not None:
            self._pool.cancel(ticket.pool_ticket)

    def finish_update(self, update: int) -> None:
        self._pool.finish_update(int(update))

    def close(self) -> None:
        self._pool.close()
