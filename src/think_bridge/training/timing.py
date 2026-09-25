"""Accumulate training timings, including deferred CUDA events."""

from typing import Any, Mapping


def merge_timing_sink(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    """Accumulate seconds and retain deferred CUDA events without synchronizing."""
    for name, value in source.items():
        if name == "_training_cuda_events":
            target.setdefault(name, []).extend(value)
        else:
            target[name] = float(target.get(name, 0.0)) + float(value)


def finalize_timing_sink(sink: dict[str, Any]) -> None:
    events = list(sink.pop("_training_cuda_events", ()))
    if not events:
        return
    events[-1][2].synchronize()
    for name, start, stop in events:
        sink[name] = (
            float(sink.get(name, 0.0)) + float(start.elapsed_time(stop)) / 1000.0
        )
