"""Public ``think-bridge`` command router.

The top-level CLI stays intentionally thin: it expands an optional YAML file,
selects the requested command module, and launches it either as one Python
process or through ``torch.distributed.run``. Training semantics live in the
subcommand modules and are not duplicated here.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from think_bridge import __version__


@dataclass(frozen=True)
class CommandSpec:
    """One stable public command and its launch policy."""

    module: str
    summary: str
    distributed: bool = True


COMMANDS: dict[str, CommandSpec] = {
    "prepare-stage0": CommandSpec(
        "think_bridge.cli.prepare_stage0",
        "Generate immutable native/direct/behavior training artifacts.",
        distributed=False,
    ),
    "sample-native-rollouts": CommandSpec(
        "think_bridge.cli.sample_fnative_vllm",
        "Collect complete paired frozen-model rollouts.",
        distributed=False,
    ),
    "precompute-behavior": CommandSpec(
        "think_bridge.cli.precompute_behavior_manifest",
        "Generate frozen-model behavior and exact direct outputs.",
        distributed=False,
    ),
    "benchmark": CommandSpec(
        "think_bridge.cli.benchmark",
        "Run single-turn, multi-turn or HF TTFT benchmarks.",
        distributed=False,
    ),
    "eval": CommandSpec(
        "think_bridge.cli.evaluate",
        "Evaluate held-out answers with optional latent controls.",
        distributed=False,
    ),
    "reasoner-sft": CommandSpec(
        "think_bridge.cli.reasoner_sft",
        "Train the latent reasoner R.",
        distributed=False,
    ),
    "infer": CommandSpec(
        "think_bridge.cli.infer",
        "Generate a latent-conditioned answer.",
        distributed=False,
    ),
}

ROUTE_MAPPING: dict[str, str] = {name: spec.module for name, spec in COMMANDS.items()}

_STAGE_OWNED_PUBLIC_COMMANDS = frozenset({"reasoner-sft"})
_ARGUMENT_SUMMARY_KEYS = (
    "model",
    "model_family",
    "train_dataset",
    "eval_dataset",
    "reasoner_train_dataset",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "max_steps",
    "max_eval_samples",
    "generation_backend",
    "output_dir",
    "resume_from_checkpoint",
)
_SENSITIVE_OPTIONS = frozenset(
    {
        "api_key",
        "api_token",
        "auth_token",
        "access_token",
        "hf_token",
        "password",
        "secret",
        "credential",
    }
)


def _normalized_option_name(token: str) -> str:
    return token.lstrip("-").replace("-", "_")


def _command_option_entries(
    command_args: list[str],
) -> list[tuple[str, str | None]]:
    entries: list[tuple[str, str | None]] = []
    index = 0
    while index < len(command_args):
        token = str(command_args[index])
        if not token.startswith("--") or token == "--":
            index += 1
            continue
        if "=" in token:
            option, value = token.split("=", 1)
            entries.append((_normalized_option_name(option), value))
            index += 1
            continue
        value: str | None = None
        if index + 1 < len(command_args):
            candidate = str(command_args[index + 1])
            if not candidate.startswith("--"):
                value = candidate
                index += 1
        entries.append((_normalized_option_name(token), value))
        index += 1
    return entries


def _compact_argument_value(name: str, value: str) -> str:
    if name in _SENSITIVE_OPTIONS:
        return "<redacted>"
    normalized = " ".join(str(value).split())
    if "/" in normalized or "\\" in normalized:
        leaf = normalized.rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1]
        normalized = f"…/{leaf or '?'}"
    if len(normalized) > 36:
        normalized = normalized[:35] + "…"
    return normalized


def _summarize_command_args(command_args: list[str]) -> str:
    """Return a deterministic diagnostic summary without full private paths."""

    entries = _command_option_entries(command_args)
    if not entries:
        return "options=0"
    latest = {name: value for name, value in entries}
    parts = [f"options={len(entries)}"]
    for name in _ARGUMENT_SUMMARY_KEYS:
        value = latest.get(name)
        if value is not None:
            parts.append(f"{name}={_compact_argument_value(name, value)}")
    flags = tuple(f"--{name}" for name, value in entries if value is None)
    if flags:
        parts.append("flags=" + ", ".join(dict.fromkeys(flags)))
    other_names = tuple(
        f"--{name}"
        for name, _value in entries
        if name not in _ARGUMENT_SUMMARY_KEYS and f"--{name}" not in flags
    )
    unique_other = tuple(dict.fromkeys(other_names))
    if unique_other:
        shown = unique_other[:8]
        suffix = (
            ""
            if len(unique_other) <= len(shown)
            else f",…(+{len(unique_other) - len(shown)})"
        )
        parts.append("other=" + ", ".join(shown) + suffix)
    return "; ".join(parts)


def _option_value(command_args: list[str], name: str) -> str | None:
    normalized = _normalized_option_name(name)
    return next(
        (
            value
            for entry_name, value in reversed(_command_option_entries(command_args))
            if entry_name == normalized
        ),
        None,
    )


def _visible_gpu_tokens(environ: dict[str, str]) -> tuple[str, ...] | None:
    if "CUDA_VISIBLE_DEVICES" not in environ:
        return None
    raw = str(environ["CUDA_VISIBLE_DEVICES"]).strip()
    if not raw:
        return ()
    return tuple(value.strip() for value in raw.split(",") if value.strip())


def _positive_environment_integer(environ: dict[str, str], name: str) -> int | None:
    raw = str(environ.get(name, "")).strip()
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _build_banner_content(
    *,
    command: str,
    spec: CommandSpec,
    command_args: list[str],
    num_gpus: int,
    child_distributed: bool,
    environ: dict[str, str],
) -> dict[str, str | int]:
    """Describe the process topology owned by the selected command."""

    content: dict[str, str | int] = {
        "command": command,
        "module": spec.module,
    }
    visible = _visible_gpu_tokens(environ)
    content["visible_gpus"] = "unknown" if visible is None else len(visible)
    managed_stage1 = command in _STAGE_OWNED_PUBLIC_COMMANDS
    if managed_stage1:
        content["driver_processes"] = 1
        trainer_world = _positive_environment_integer(environ, "NPROC_PER_NODE")
        content["trainer_world_size"] = (
            "unknown" if trainer_world is None else trainer_world
        )
        if (
            command in _STAGE_OWNED_PUBLIC_COMMANDS
            and visible is not None
            and trainer_world is not None
            and len(visible) >= trainer_world
        ):
            trainers = visible[-trainer_world:]
            backend = _option_value(command_args, "generation_backend") or "torch"
            service = (
                visible[:-trainer_world]
                if command in {"reasoner-sft"} and backend == "vllm"
                else ()
            )
            content["service_gpus"] = ",".join(service) if service else "none"
            content["trainer_gpus"] = ",".join(trainers)
    elif child_distributed:
        content["launcher_processes"] = int(num_gpus)
    else:
        content["driver_processes"] = 1
    content["args"] = _summarize_command_args(command_args)
    return content


def _config_items_to_argv(config: dict[str, Any]) -> list[str]:
    """Serialize one flat YAML mapping into command-line arguments."""

    result: list[str] = []
    for raw_key, value in config.items():
        key = str(raw_key).strip()
        if not key or key.startswith("-"):
            raise ValueError(f"invalid YAML option name: {raw_key!r}")
        if value is None:
            continue
        if isinstance(value, dict):
            raise ValueError(
                f"YAML option {key!r} must be scalar or list, not a mapping"
            )
        option = f"--{key}"
        if isinstance(value, bool) and key in {"local_files_only", "no_progress"}:
            if value:
                result.append(option)
            continue
        result.append(option)
        if isinstance(value, bool):
            result.append(str(value).lower())
        elif isinstance(value, list):
            result.append(",".join(str(item) for item in value))
        else:
            result.append(str(value))
    return result


def _parse_yaml_args(argv: list[str]) -> list[str]:
    """Expand each ``--config FILE.yaml`` in place.

    Arguments written after ``--config`` remain after the expanded values, so
    the downstream parser can apply its ordinary last-value-wins behavior.
    """

    expanded: list[str] = []
    index = 0
    while index < len(argv):
        if argv[index] != "--config":
            expanded.append(argv[index])
            index += 1
            continue
        if index + 1 >= len(argv):
            raise ValueError("--config requires a YAML file path")
        config_path = Path(argv[index + 1]).expanduser()
        with config_path.open("r", encoding="utf-8") as handle:
            try:
                import yaml
            except ImportError:
                import json

                try:
                    loaded = json.load(handle)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "PyYAML is unavailable; --config fallback accepts JSON mappings"
                    ) from exc
            else:
                loaded = yaml.safe_load(handle)
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ValueError(f"YAML config root must be a mapping: {config_path}")
        expanded.extend(_config_items_to_argv(loaded))
        index += 2
    return expanded


def _format_help() -> str:
    width = max(len(name) for name in COMMANDS)
    command_lines = [
        f"  {name:<{width}}  {spec.summary}" for name, spec in COMMANDS.items()
    ]
    return "\n".join(
        [
            "Usage: think-bridge <command> [options]",
            "",
            "Commands:",
            *command_lines,
            "",
            "Global options:",
            "  -h, --help       Show this help.",
            "  -V, --version    Show the installed package version.",
            "  --config FILE    Expand a flat YAML mapping in place.",
            "",
            "Stage 1 owns its trainer topology through NPROC_PER_NODE.",
        ]
    )


def _build_launch_args(
    *,
    script_path: str,
    command_args: list[str],
    num_gpus: int,
    master_port: str,
    distributed: bool,
) -> list[str]:
    """Build the child argv without invoking a shell."""

    if num_gpus <= 0:
        raise ValueError(f"NUM_GPUS must be positive, got {num_gpus}")
    if num_gpus == 1 or not distributed:
        return [sys.executable, script_path, *command_args]

    launch = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={num_gpus}",
    ]
    if master_port:
        launch.extend(
            [
                "--nnodes=1",
                "--rdzv-backend=c10d",
                f"--rdzv-endpoint=127.0.0.1:{master_port}",
                f"--rdzv-id=tb-{master_port}",
            ]
        )
    else:
        launch.append("--standalone")
    return [*launch, script_path, *command_args]


def cli_main() -> None:
    """Run the public command router."""

    argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help"}:
        print(_format_help())
        return
    if argv[0] in {"-V", "--version"}:
        print(f"think-bridge {__version__}")
        return

    from think_bridge.utils.logger import logger, print_box

    command = argv[0]
    spec = COMMANDS.get(command)
    if spec is None:
        logger.error(f"Unknown command: {command!r}. Available: {list(COMMANDS)}")
        raise SystemExit(2)

    try:
        command_args = _parse_yaml_args(argv[1:])
        num_gpus = int(os.environ.get("NUM_GPUS", "1"))
        child_distributed = spec.distributed
        child_args = _build_launch_args(
            script_path=_resolve_module_path(spec.module),
            command_args=command_args,
            num_gpus=num_gpus,
            master_port=os.environ.get("MASTER_PORT", "").strip(),
            distributed=child_distributed,
        )
    except (OSError, TypeError, ValueError) as exc:
        logger.error(str(exc))
        raise SystemExit(2) from exc

    print_box(
        "ThinkBridge CLI",
        _build_banner_content(
            command=command,
            spec=spec,
            command_args=command_args,
            num_gpus=num_gpus,
            child_distributed=child_distributed,
            environ=dict(os.environ),
        ),
    )
    if command in {"reasoner-sft", "eval", "benchmark"}:
        from think_bridge.training.process_cleanup import run_managed_process

        # The driver owns a separately spawned vLLM service. Let its finally
        # block stop that service before force-killing the driver group.
        result = run_managed_process(
            child_args, env=os.environ.copy(), shutdown_timeout_seconds=30.0
        )
    else:
        result = subprocess.run(child_args, env=os.environ.copy(), check=False)
    raise SystemExit(result.returncode)


def _resolve_module_path(module: str) -> str:
    module_spec = find_spec(module)
    if module_spec is None or module_spec.origin is None:
        raise ValueError(f"Cannot find module: {module}")
    return module_spec.origin


if __name__ == "__main__":
    cli_main()
