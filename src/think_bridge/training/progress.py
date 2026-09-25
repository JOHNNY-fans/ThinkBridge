"""Shared CLI progress bars for long-running ThinkBridge work."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
import sys
import time
from typing import Any, TypeVar

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # Local dependency-light contract environments.
    tqdm = None


_T = TypeVar("_T")


_PROGRESS_STYLES = frozenset({"auto", "tqdm", "plain"})
_VALIDATION_PROGRESS_COLOUR = "#228B22"
_ROUTE1_LABEL_COLOUR = "#168AAD"
_ANSI_RESET = "\x1b[0m"
_TQDM_BAR_FORMAT = (
    "{desc} {percentage:3.0f}%|{bar:32}| {n_fmt}/{total_fmt} "
    "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
)


@dataclass(frozen=True)
class BridgeTrainingStage:
    """One user-facing stage in the configured Bridge course."""

    description: str
    colour: str


def _training_route_label(
    *, route: str, epoch: int, route1_epochs: int = 2
) -> tuple[str, str]:
    if any(
        isinstance(v, bool) or not isinstance(v, int) for v in (epoch, route1_epochs)
    ):
        raise TypeError("Bridge progress epoch values must be integers")
    if route != "route1" or not 0 <= epoch < route1_epochs:
        raise ValueError("unknown R training epoch")
    return "Route1/R", _ROUTE1_LABEL_COLOUR


def _ansi_foreground(text: str, colour: str) -> str:
    red = int(colour[1:3], 16)
    green = int(colour[3:5], 16)
    blue = int(colour[5:7], 16)
    return f"\x1b[38;2;{red};{green};{blue}m{text}{_ANSI_RESET}"


def bridge_training_label(
    *,
    route: str,
    epoch: int,
    route1_epochs: int = 2,
    update_budget: int | None = None,
    ansi: bool = False,
) -> str:
    """Format one course-global epoch or capped-update label."""
    (route_label, colour) = _training_route_label(
        route=route, epoch=epoch, route1_epochs=route1_epochs
    )
    if update_budget is None:
        local_epoch = int(epoch) + 1
        local_epochs = int(route1_epochs)
        suffix = f"Epoch {local_epoch}/{local_epochs}"
    else:
        if isinstance(update_budget, bool) or not isinstance(update_budget, int):
            raise TypeError("Bridge progress update budget must be an integer")
        if update_budget <= 0:
            raise ValueError("Bridge progress update budget must be positive")
        suffix = f"Step 0/{update_budget}"
    label = f"[Stage1 · {route_label} · {suffix}]"
    return _ansi_foreground(label, colour) if ansi else label


def bridge_training_stage(
    *,
    route: str,
    epoch: int,
    route1_epochs: int = 2,
    update_budget: int | None = None,
) -> BridgeTrainingStage:
    """Resolve an exact course-global label without exposing curriculum names."""

    _, colour = _training_route_label(
        route=route,
        epoch=epoch,
        route1_epochs=route1_epochs,
    )
    return BridgeTrainingStage(
        description=bridge_training_label(
            route=route,
            epoch=epoch,
            route1_epochs=route1_epochs,
            update_budget=update_budget,
        ),
        colour=colour,
    )


TRAINING_WINDOW_METRICS = (
    "loss_total",
    "loss_route1",
    "loss_ce",
    "loss_match",
    "loss_specific",
    "lr_R",
    "batch_sample_count",
    "batch_unique_group_count",
    "course_active_count",
    "course_c_active_count",
    "course_b_d_side_active_count",
    "distill_b_sample_count",
    "distill_c_sample_count",
    "distill_reference_sample_count",
    "specificity_eligible_pair_count",
    "specificity_selected_pair_count",
    "specificity_same_prompt_record_pair_count",
    "specificity_donors_per_owner",
    "match_active_count",
    "specific_active_count",
    "preclip_grad_R",
    "postclip_grad_R",
    "clip_coefficient",
    "throughput_samples_per_second",
    "throughput_valid_tokens_per_second",
    "throughput_latent_blocks_per_second",
    "update_seconds",
    "cuda_allocated_peak_bytes",
    "cuda_reserved_peak_bytes",
    "reasoner_forward_seconds",
    "live_f_forward_seconds",
    "lm_head_loss_seconds",
    "backward_seconds",
    "rank0_optimizer_step_seconds",
    "optimizer_seconds",
    "batch_prepare_wait_seconds",
    "batch_collate_padding_seconds",
    "collective_wait_seconds",
    "integrity_scan_seconds",
    "valid_token_count",
    "latent_block_count",
    "padding_ratio",
    "predicted_attention_cost_padding_ratio",
    "rank_compute_skew_ratio",
    "local_physical_chunk_count",
    "rank_max_live_z_seconds",
    "rank_max_answer_course_seconds",
    "rank_max_need_objectives_seconds",
    "rank_max_need_objectives_rollout_seconds",
    "rank_max_need_objectives_native_seconds",
    "rank_max_need_objectives_direct_seconds",
    "rank_max_need_objectives_true_seconds",
    "rank_max_need_objectives_wrong_detached_seconds",
    "rank_max_need_objectives_wrong_live_replay_seconds",
    "rank_max_backward_seconds",
    "rank_max_optimizer_seconds",
    "rank_max_peak_memory_bytes",
    "rank_max_c_view_count",
    "rank_max_legal_wrong_pair_count",
    "rank_max_wrong_physical_chunk_count",
)


def bridge_training_postfix(
    *,
    route: str,
    arm: str,
    metrics: Mapping[str, float | int | None],
    learning_rate: float,
) -> dict[str, str]:
    """Build the single compact tqdm postfix for the active training arm."""

    if route not in {"route1"}:
        raise ValueError("training postfix route must be route1")
    if arm != "bridge":
        raise ValueError("training postfix requires the registered Bridge method")

    def rendered(name: str) -> str:
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"training postfix metric is unavailable: {name}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"training postfix metric is non-finite: {name}")
        return f"{number:.4f}"

    del learning_rate
    return {"loss": rendered("loss_total")}


class BridgeLoggingWindow:
    """Accumulate independent finite means for the declared training metrics."""

    def __init__(self, *, metric_names: Iterable[str]) -> None:
        names = tuple(metric_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("logging-window metric names must be non-empty strings")
        if len(names) != len(set(names)):
            raise ValueError("logging-window metric names must be unique")
        self._names = names
        self.reset()

    def add(self, metrics: Mapping[str, float | int | None]) -> None:
        self._updates += 1
        for name in self._names:
            value = metrics.get(name)
            if value is None:
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"logging-window metric is non-finite: {name}")
            self._sums[name] += numeric
            self._counts[name] += 1
            if name in self._memory_peaks:
                self._memory_peaks[name] = max(self._memory_peaks[name] or 0.0, numeric)

    def means(self) -> dict[str, float | int | None]:
        values: dict[str, float | int | None] = {
            f"window_mean_{name}": (
                None
                if self._counts[name] == 0
                else self._sums[name] / self._counts[name]
            )
            for name in self._names
        }
        values["logging_window_updates"] = self._updates
        values.update(
            {f"window_max_{name}": value for name, value in self._memory_peaks.items()}
        )
        return values

    def snapshot(self, *, reset: bool) -> dict[str, float | int | None]:
        """Read one window and consume it only for a real train-log event."""

        if not isinstance(reset, bool):
            raise TypeError("logging-window reset flag must be boolean")
        values = self.means()
        if reset:
            self.reset()
        return values

    def reset(self) -> None:
        self._memory_peaks = {
            name: None
            for name in self._names
            if name in {"cuda_allocated_peak_bytes", "cuda_reserved_peak_bytes"}
        }
        self._sums = {name: 0.0 for name in self._names}
        self._counts = {name: 0 for name in self._names}
        self._updates = 0


def write_progress(message: str) -> None:
    """Write one complete console line without corrupting an active tqdm bar."""

    if not isinstance(message, str) or not message:
        raise ValueError("progress message must be a non-empty string")
    if "\n" in message or "\r" in message:
        raise ValueError("progress message must be exactly one complete line")
    if tqdm is None:
        print(message, flush=True)
    else:
        tqdm.write(message)


class _DisabledProgress:
    """Silent tqdm-compatible surface for dependency-light --no-progress runs."""

    def __init__(self, *, total: int, initial: int) -> None:
        self.total = total
        self.n = initial
        self.desc = ""

    def update(self, increment: int = 1) -> None:
        self.n += int(increment)

    def set_postfix(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def set_description(self, desc: str, refresh: bool = True) -> None:
        del refresh
        self.desc = str(desc)

    def close(self) -> None:
        return None


def _clock_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class _PlainProgress:
    """Throttled newline-only progress for redirected and non-interactive runs."""

    _REPORT_INTERVAL_SECONDS = 30.0

    def __init__(
        self,
        *,
        total: int,
        initial: int,
        desc: str,
        unit: str,
        stream: Any,
    ) -> None:
        self.total = int(total)
        self.n = int(initial)
        self.desc = str(desc)
        self.unit = str(unit)
        self.stream = stream
        self._started = time.monotonic()
        self._last_reported_at = self._started
        self._last_reported_n: int | None = None
        self._postfix: dict[str, str] = {}
        self._closed = False
        self._report()

    def _report(self) -> None:
        now = time.monotonic()
        elapsed = max(now - self._started, 0.0)
        completed = max(0, min(self.n, self.total))
        percentage = 100.0 if self.total == 0 else 100.0 * completed / self.total
        rate = (completed / elapsed) if elapsed > 0.0 else 0.0
        remaining = max(self.total - completed, 0)
        eta = (remaining / rate) if rate > 0.0 else None
        postfix = "".join(f" {name}={value}" for name, value in self._postfix.items())
        self.stream.write(
            f"{self.desc}: {completed}/{self.total} {percentage:.1f}% "
            f"rate={rate:.2f} {self.unit}/s elapsed={_clock_duration(elapsed)} "
            f"ETA={'--' if eta is None and remaining > 0 else _clock_duration(eta or 0.0)}"
            f"{postfix}\n"
        )
        self.stream.flush()
        self._last_reported_at = now
        self._last_reported_n = self.n

    def update(self, increment: int = 1) -> None:
        value = int(increment)
        if value < 0:
            raise ValueError("progress increment must be non-negative")
        self.n += value
        if self.n > self.total:
            raise ValueError("progress exceeded its declared total")
        now = time.monotonic()
        if (
            self.n == self.total
            or now - self._last_reported_at >= self._REPORT_INTERVAL_SECONDS
        ):
            self._report()

    def set_postfix(self, *args: Any, **kwargs: Any) -> None:
        refresh = kwargs.pop("refresh", False)
        if refresh:
            raise ValueError("plain progress does not permit forced refresh")
        values: dict[str, Any] = {}
        if args:
            if len(args) != 1 or not isinstance(args[0], Mapping):
                raise TypeError("progress postfix requires one mapping")
            values.update(args[0])
        values.update(kwargs)
        self._postfix = {str(name): str(value) for name, value in values.items()}

    def set_description(self, desc: str, refresh: bool = True) -> None:
        if refresh:
            raise ValueError("plain progress does not permit forced refresh")
        self.desc = str(desc)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._last_reported_n != self.n:
            self._report()


class BridgeStageReporter:
    """Publish real phase/elapsed state without advancing optimizer progress."""

    def __init__(
        self,
        progress_bar: Any,
        *,
        label: str,
        dynamic_report_interval_seconds: float = 1.0,
        plain_report_interval_seconds: float = 30.0,
    ) -> None:
        if not isinstance(label, str) or not label.strip():
            raise ValueError("stage reporter label must be non-empty")
        dynamic_interval = float(dynamic_report_interval_seconds)
        plain_interval = float(plain_report_interval_seconds)
        if dynamic_interval <= 0.0 or plain_interval <= 0.0:
            raise ValueError("stage reporter intervals must be positive")
        self._progress_bar = progress_bar
        self._label = label.strip()
        self._dynamic_interval = dynamic_interval
        self._plain_interval = plain_interval
        self._started = time.monotonic()
        self._last_stage: str | None = None
        self._last_emitted_at: float | None = None

    def report(
        self,
        stage: str,
        *,
        elapsed_seconds: float | None = None,
        important: bool = False,
    ) -> None:
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError("progress stage must be non-empty")
        if not isinstance(important, bool):
            raise TypeError("progress stage importance must be boolean")
        now = time.monotonic()
        elapsed = (
            now - self._started if elapsed_seconds is None else float(elapsed_seconds)
        )
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError(
                "progress stage elapsed time must be finite and nonnegative"
            )
        normalized = stage.strip()
        changed = normalized != self._last_stage
        plain = isinstance(self._progress_bar, (_PlainProgress, _DisabledProgress))
        interval = self._plain_interval if plain else self._dynamic_interval
        due = self._last_emitted_at is None or now - self._last_emitted_at >= interval
        if plain:
            should_emit = (important and changed) or due
            if should_emit:
                active_label = str(
                    getattr(self._progress_bar, "desc", "") or self._label
                )
                message = (
                    f"{active_label} stage={normalized} "
                    f"elapsed={_clock_duration(elapsed)}"
                )
                if isinstance(self._progress_bar, _PlainProgress):
                    self._progress_bar.stream.write(message + "\n")
                    self._progress_bar.stream.flush()
                else:
                    write_progress(message)
        else:
            should_emit = changed or due
            if should_emit:
                self._progress_bar.set_postfix(
                    {
                        "stage": normalized,
                        "work_elapsed": _clock_duration(elapsed),
                    },
                    refresh=True,
                )
        self._last_stage = normalized
        if should_emit:
            self._last_emitted_at = now


@contextmanager
def suspend_bridge_progress(progress_bar: Any) -> Iterator[None]:
    """Clear one dynamic outer bar while a nested progress phase owns its line."""

    dynamic = not isinstance(progress_bar, (_PlainProgress, _DisabledProgress))
    clear = getattr(progress_bar, "clear", None)
    refresh = getattr(progress_bar, "refresh", None)
    if dynamic and callable(clear):
        clear()
    try:
        yield
    finally:
        if dynamic and callable(refresh):
            refresh()


def _progress_colour(desc: str) -> str:
    label = desc.casefold()
    if "save" in label or "saving" in label:
        return "green"
    if "eval" in label or "valid" in label:
        return _VALIDATION_PROGRESS_COLOUR
    if "route1" in label:
        return _ROUTE1_LABEL_COLOUR
    return "blue"


def _training_label_colour(desc: str) -> str | None:
    if desc.startswith("[Stage1 · Route1/R · ") and desc.endswith("]"):
        return _ROUTE1_LABEL_COLOUR
    return None


def set_bridge_training_stage(
    progress_bar: Any,
    *,
    route: str,
    epoch: int,
    route1_epochs: int = 2,
    update_budget: int | None = None,
) -> BridgeTrainingStage:
    """Update the one persistent training bar at an epoch boundary."""

    stage = bridge_training_stage(
        route=route,
        epoch=epoch,
        route1_epochs=route1_epochs,
        update_budget=update_budget,
    )
    description = bridge_training_label(
        route=route,
        epoch=epoch,
        route1_epochs=route1_epochs,
        update_budget=update_budget,
        ansi=getattr(progress_bar, "_bridge_ansi_training_labels", False) is True,
    )
    progress_bar.set_description(
        description,
        refresh=not isinstance(progress_bar, (_PlainProgress, _DisabledProgress)),
    )
    return stage


def bridge_progress(
    *,
    total: int,
    desc: str,
    unit: str,
    initial: int = 0,
    disabled: bool = False,
    stream: Any | None = None,
) -> Any:
    """Create one dynamic bar or a throttled newline-only fallback.

    ``THINK_BRIDGE_PROGRESS_STYLE=tqdm`` overrides non-TTY detection for torchrun
    launchers whose inherited stderr is still watched interactively.
    """

    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ValueError("progress total must be a non-negative integer")
    if (
        isinstance(initial, bool)
        or not isinstance(initial, int)
        or initial < 0
        or initial > total
    ):
        raise ValueError("progress initial must be an integer within total")
    if not isinstance(desc, str) or not desc.strip():
        raise ValueError("progress description must be non-empty")
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("progress unit must be non-empty")
    if disabled:
        return _DisabledProgress(total=total, initial=initial)
    output = sys.stderr if stream is None else stream
    style = os.environ.get("THINK_BRIDGE_PROGRESS_STYLE", "auto").strip().casefold()
    if style not in _PROGRESS_STYLES:
        raise ValueError(
            "THINK_BRIDGE_PROGRESS_STYLE must be one of auto, tqdm, or plain"
        )
    is_tty = bool(callable(getattr(output, "isatty", None)) and output.isatty())
    term_is_dumb = os.environ.get("TERM", "").casefold() == "dumb"
    force_tqdm = style == "tqdm"
    training_label_colour = _training_label_colour(desc)
    if force_tqdm and tqdm is None:
        raise RuntimeError("THINK_BRIDGE_PROGRESS_STYLE=tqdm requires the tqdm package")
    if (
        style == "plain"
        or tqdm is None
        or (term_is_dumb and not force_tqdm)
        or (not force_tqdm and not is_tty)
    ):
        return _PlainProgress(
            total=total,
            initial=initial,
            desc=desc,
            unit=unit,
            stream=output,
        )
    colour_enabled = "NO_COLOR" not in os.environ
    ansi_training_label = bool(
        training_label_colour is not None and colour_enabled and (is_tty or force_tqdm)
    )
    arguments: dict[str, Any] = dict(
        total=total,
        initial=initial,
        desc=(
            _ansi_foreground(desc, training_label_colour)
            if ansi_training_label
            else desc
        ),
        unit=unit,
        dynamic_ncols=True,
        bar_format=_TQDM_BAR_FORMAT,
        smoothing=0.1,
        mininterval=0.5,
        leave=True,
        disable=False,
        file=output,
    )
    if colour_enabled:
        arguments["colour"] = (
            training_label_colour
            if training_label_colour is not None
            else _progress_colour(desc)
        )
    progress_bar = tqdm(**arguments)
    progress_bar._bridge_ansi_training_labels = ansi_training_label
    return progress_bar


def iter_progress(
    iterable: Iterable[_T],
    *,
    total: int,
    desc: str,
    unit: str,
    disabled: bool = False,
    update_size: Callable[[_T], int] | None = None,
) -> Iterator[_T]:
    """Yield inputs unchanged while advancing and safely closing one bar."""

    progress_bar = bridge_progress(
        total=total,
        desc=desc,
        unit=unit,
        disabled=disabled,
    )
    try:
        for item in iterable:
            yield item
            increment = 1 if update_size is None else update_size(item)
            if (
                isinstance(increment, bool)
                or not isinstance(increment, int)
                or increment <= 0
            ):
                raise ValueError("progress update size must be a positive integer")
            progress_bar.update(increment)
    finally:
        progress_bar.close()
