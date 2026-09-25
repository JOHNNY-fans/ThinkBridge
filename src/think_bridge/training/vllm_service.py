"""Private localhost HTTP data plane for Bridge Route1 stop-gradient F work."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import signal
import sys
import threading
import time
import traceback
from typing import Any, Sequence

from think_bridge.training.vllm_runtime import (
    ROUTE1_GENERATION_PROBE_STATUS,
    ROUTE1_MAX_HTTP_REQUEST_BYTES,
    ROUTE1_MAX_HTTP_RESPONSE_BYTES,
    Route1ServiceRequest,
    parse_strict_bool,
    validate_private_service_instance_token,
)


@dataclass
class _ServiceLease:
    admission: "_ServiceAdmission"
    queue_seconds: float
    active_requests: int
    released: bool = False

    def release(self) -> None:
        if not self.released:
            self.released = True
            self.admission.release()


class _ServiceAdmission:
    """Bound concurrent generation and queued HTTP handlers explicitly."""

    def __init__(self, *, max_active: int, max_queued: int) -> None:
        if int(max_active) <= 0 or int(max_queued) <= 0:
            raise ValueError("Route1 service admission bounds must be positive")
        self._max_active = int(max_active)
        self._max_queued = int(max_queued)
        self._active = 0
        self._queued = 0
        self._condition = threading.Condition()

    @property
    def queued_requests(self) -> int:
        with self._condition:
            return int(self._queued)

    def acquire(self, *, timeout_seconds: float) -> _ServiceLease:
        timeout = float(timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("Route1 service queue timeout must be positive")
        started = time.monotonic()
        with self._condition:
            if self._active < self._max_active:
                self._active += 1
                return _ServiceLease(self, 0.0, self._active)
            if self._queued >= self._max_queued:
                raise RuntimeError("Route1 service queue is full")
            self._queued += 1
            try:
                while self._active >= self._max_active:
                    remaining = timeout - (time.monotonic() - started)
                    if remaining <= 0.0:
                        raise TimeoutError("Route1 service queue timed out")
                    self._condition.wait(timeout=remaining)
                self._active += 1
                return _ServiceLease(
                    self,
                    time.monotonic() - started,
                    self._active,
                )
            finally:
                self._queued -= 1

    def release(self) -> None:
        with self._condition:
            self._active -= 1
            if self._active < 0:
                raise RuntimeError("Route1 service active-request accounting underflow")
            self._condition.notify_all()


class _ServiceHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        *args: Any,
        instance_token: str,
        max_request_bytes: int,
        max_response_bytes: int,
        max_concurrent_requests: int,
        max_queued_requests: int,
        queue_timeout_seconds: float,
        **kwargs: Any,
    ):
        self.backend: Any | None = None
        self.instance_token = validate_private_service_instance_token(instance_token)
        self.max_request_bytes = int(max_request_bytes)
        self.max_response_bytes = int(max_response_bytes)
        self.admission = _ServiceAdmission(
            max_active=int(max_concurrent_requests),
            max_queued=int(max_queued_requests),
        )
        self.queue_timeout_seconds = float(queue_timeout_seconds)
        self._backend_closed = True
        self._ready = False
        super().__init__(*args, **kwargs)

    @property
    def ready(self) -> bool:
        return bool(self._ready and self.backend is not None)

    def attach_backend(self, backend: Any) -> None:
        if backend is None or self.backend is not None or self._ready:
            raise RuntimeError("Route1 service backend attachment is invalid")
        self.backend = backend
        self._backend_closed = False
        self.server_activate()
        self._ready = True

    def server_close(self) -> None:
        try:
            self._ready = False
            if not self._backend_closed and self.backend is not None:
                close = getattr(self.backend, "close", None)
                if callable(close):
                    close()
        finally:
            self._backend_closed = True
            self.backend = None
            super().server_close()


class _Handler(BaseHTTPRequestHandler):
    server_version = "Think-Bridge-Route1/1"

    def log_message(self, format: str, *args: Any) -> None:
        return None

    def _send(self, status: int, payload: bytes, content_type: str) -> bool:
        try:
            self.send_response(int(status))
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            print(
                json.dumps(
                    {
                        "event": "bridge-route1-service-response-disconnected",
                        "status": int(status),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                file=sys.stderr,
                flush=True,
            )
            return False
        return True

    def do_GET(self) -> None:
        if self.path != "/health":
            self._send(404, b'{"error":"not-found"}', "application/json")
            return
        ready = bool(
            self.server.ready
            and getattr(self.server.backend, "probe_status", "")
            == ROUTE1_GENERATION_PROBE_STATUS
        )
        payload = json.dumps(
            {
                "status": "ready" if ready else "starting",
                "probe_status": str(getattr(self.server.backend, "probe_status", "")),
                "instance_token": self.server.instance_token,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        self._send(200 if ready else 503, payload, "application/json")

    def do_POST(self) -> None:
        if self.path != "/route1":
            self._send(404, b'{"error":"not-found"}', "application/json")
            return
        if not self.server.ready:
            self._send(
                503,
                b'{"error":"service-not-ready"}',
                "application/json",
            )
            return
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length) if raw_length is not None else -1
        except ValueError:
            length = -1
        if length <= 0 or length > self.server.max_request_bytes:
            self._send(
                413,
                b'{"error":"invalid-or-oversized-request"}',
                "application/json",
            )
            return
        payload = self.rfile.read(length)
        if len(payload) != length:
            self._send(400, b'{"error":"truncated-request"}', "application/json")
            return
        request: Route1ServiceRequest | None = None
        lease: _ServiceLease | None = None
        try:
            request = Route1ServiceRequest.from_wire(payload)
            lease = self.server.admission.acquire(
                timeout_seconds=self.server.queue_timeout_seconds
            )
            response = self.server.backend.execute(
                request,
                service_queue_seconds=lease.queue_seconds,
                service_active_requests=lease.active_requests,
            )
            response.assert_matches(request)
            encoded = response.to_wire()
            if len(encoded) > self.server.max_response_bytes:
                raise RuntimeError("Route1 response exceeds its HTTP byte bound")
        except Exception as exc:
            identity = None if request is None else list(request.identity)
            print(
                json.dumps(
                    {
                        "event": "bridge-route1-service-error",
                        "error": type(exc).__name__,
                        "detail": str(exc)[:2048],
                        "request_identity": identity,
                        "traceback": traceback.format_exc(limit=16)[-8192:],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                file=sys.stderr,
                flush=True,
            )
            detail = json.dumps(
                {"error": type(exc).__name__, "detail": str(exc)[:2048]},
                separators=(",", ":"),
            ).encode("utf-8")
            status = (
                429
                if "queue is full" in str(exc)
                else 504
                if isinstance(exc, TimeoutError)
                else 500
            )
            self._send(status, detail, "application/json")
            return
        finally:
            if lease is not None:
                lease.release()
        self._send(200, encoded, "application/x-think-bridge-route1")


def build_http_server(
    *,
    host: str,
    port: int,
    instance_token: str,
    backend: Any | None = None,
    max_request_bytes: int,
    max_response_bytes: int = ROUTE1_MAX_HTTP_RESPONSE_BYTES,
    max_concurrent_requests: int = 1,
    max_queued_requests: int = 1,
    queue_timeout_seconds: float = 60.0,
) -> _ServiceHTTPServer:
    if host != "127.0.0.1":
        raise ValueError("Bridge Route1 service must bind exactly 127.0.0.1")
    if int(port) < 0 or int(port) > 65535:
        raise ValueError("Bridge Route1 service port is invalid")
    if int(max_request_bytes) <= 0 or int(max_response_bytes) <= 0:
        raise ValueError("Bridge Route1 service byte bounds must be positive")
    server = _ServiceHTTPServer(
        (host, int(port)),
        _Handler,
        instance_token=instance_token,
        max_request_bytes=int(max_request_bytes),
        max_response_bytes=int(max_response_bytes),
        max_concurrent_requests=int(max_concurrent_requests),
        max_queued_requests=int(max_queued_requests),
        queue_timeout_seconds=float(queue_timeout_seconds),
        bind_and_activate=False,
    )
    try:
        server.server_bind()
        if backend is not None:
            server.attach_backend(backend)
    except BaseException:
        server.server_close()
        raise
    return server


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="think-bridge-route1-service")
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve")
    serve.add_argument("--model", required=True)
    serve.add_argument("--seed", type=int, default=42)
    serve.add_argument("--host", required=True)
    serve.add_argument("--port", type=int, required=True)
    serve.add_argument(
        "--instance-token",
        type=validate_private_service_instance_token,
        required=True,
    )
    serve.add_argument("--data-parallel-size", type=int, required=True)
    serve.add_argument("--tensor-parallel-size", type=int, required=True)
    serve.add_argument("--physical-chunk-size", type=int, required=True)
    serve.add_argument("--request-timeout-seconds", type=float, required=True)
    serve.add_argument("--worker-extension-cls", required=True)
    serve.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    serve.add_argument("--enforce-eager", type=parse_strict_bool, required=True)
    serve.add_argument("--max-num-seqs", type=int, default=64)
    serve.add_argument("--max-in-flight", type=int, default=1)
    serve.add_argument("--max-queued-requests", type=int, default=1)
    serve.add_argument("--queue-timeout-seconds", type=float, default=60.0)
    serve.add_argument(
        "--max-request-bytes", type=int, default=ROUTE1_MAX_HTTP_REQUEST_BYTES
    )
    serve.add_argument(
        "--max-response-bytes", type=int, default=ROUTE1_MAX_HTTP_RESPONSE_BYTES
    )
    return parser


def build_backend_from_arguments(arguments: argparse.Namespace) -> Any:
    """Construct the GPU backend from the exact parsed service identity."""

    from think_bridge.training.vllm_worker import BridgeRoute1VLLMBackend

    backend = BridgeRoute1VLLMBackend(
        model_name_or_path=arguments.model,
        seed=arguments.seed,
        data_parallel_size=arguments.data_parallel_size,
        tensor_parallel_size=arguments.tensor_parallel_size,
        worker_extension_cls=arguments.worker_extension_cls,
        gpu_memory_utilization=arguments.gpu_memory_utilization,
        max_num_seqs=arguments.max_num_seqs,
        request_timeout_seconds=arguments.request_timeout_seconds,
        physical_chunk_size=arguments.physical_chunk_size,
        enforce_eager=parse_strict_bool(arguments.enforce_eager),
    )
    return backend


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command != "serve":
        raise ValueError("unknown Bridge Route1 service command")
    startup = time.perf_counter()
    try:
        server = build_http_server(
            host=arguments.host,
            port=arguments.port,
            instance_token=arguments.instance_token,
            max_request_bytes=arguments.max_request_bytes,
            max_response_bytes=arguments.max_response_bytes,
            max_concurrent_requests=arguments.max_in_flight,
            max_queued_requests=arguments.max_queued_requests,
            queue_timeout_seconds=arguments.queue_timeout_seconds,
        )
    except OSError as exc:
        print(
            json.dumps(
                {
                    "event": "bridge-route1-service-startup-error",
                    "phase": "socket-bind",
                    "error": type(exc).__name__,
                    "detail": str(exc)[:2048],
                    "host": arguments.host,
                    "port": int(arguments.port),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )
        raise
    ready_announced = False
    try:
        backend = build_backend_from_arguments(arguments)
        if backend.probe_status != ROUTE1_GENERATION_PROBE_STATUS:
            try:
                backend.close()
            except Exception as close_error:
                raise RuntimeError(
                    "vLLM generation capability probe failed and backend "
                    f"cleanup failed: {close_error}"
                ) from close_error
            raise RuntimeError("vLLM generation capability probe did not complete")
        server.attach_backend(backend)
        print(
            json.dumps(
                {
                    "event": "bridge-route1-service-ready",
                    "instance_token": arguments.instance_token,
                    "model": str(arguments.model),
                    "data_parallel_size": int(arguments.data_parallel_size),
                    "tensor_parallel_size": int(arguments.tensor_parallel_size),
                    "probe_status": backend.probe_status,
                    "batch_invariant": getattr(backend, "batch_invariant", False),
                    "vllm_version": backend.vllm_version,
                    "embedding_identity": backend.embedding_identity,
                    "startup_phases_seconds": backend.startup_timings,
                    "service_batch": int(arguments.physical_chunk_size),
                    "enforce_eager": bool(arguments.enforce_eager),
                    "gpu_memory_utilization": float(arguments.gpu_memory_utilization),
                    "max_request_bytes": int(arguments.max_request_bytes),
                    "max_response_bytes": int(arguments.max_response_bytes),
                    "startup_seconds": round(time.perf_counter() - startup, 3),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        ready_announced = True
        stop_once = threading.Event()

        def stop(_signum: int, _frame: Any) -> None:
            if not stop_once.is_set():
                stop_once.set()
                threading.Thread(target=server.shutdown, daemon=True).start()

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        server.serve_forever(poll_interval=0.1)
    except BaseException as exc:
        if not ready_announced:
            print(
                json.dumps(
                    {
                        "event": "bridge-route1-service-startup-error",
                        "phase": "backend-init-or-activate",
                        "error": type(exc).__name__,
                        "detail": str(exc)[:2048],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                file=sys.stderr,
                flush=True,
            )
        raise
    finally:
        try:
            server.server_close()
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "event": "bridge-route1-service-shutdown-error",
                        "error": type(exc).__name__,
                        "detail": str(exc)[:2048],
                        "traceback": traceback.format_exc(limit=16)[-8192:],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                file=sys.stderr,
                flush=True,
            )
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
