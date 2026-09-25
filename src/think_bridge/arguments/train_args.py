"""DeepSpeed configuration validation used by the training runtime."""

from __future__ import annotations

import json
import math
from pathlib import Path


def validate_stage1_deepspeed_config(path: str | None) -> int:
    """Validate FP32 owner state with DeepSpeed-managed BF16 autocast."""

    if not path:
        return 0
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ValueError(f"DeepSpeed config does not exist: {config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"DeepSpeed config cannot be read: {config_path}") from exc
    if not isinstance(config, dict):
        raise ValueError("DeepSpeed config root must be an object")

    zero = config.get("zero_optimization")
    stage = zero.get("stage") if isinstance(zero, dict) else None
    if stage != 1:
        raise ValueError("ThinkBridge supports DeepSpeed ZeRO stage 1 only")
    for name in ("offload_optimizer", "offload_param"):
        section = zero.get(name, {})
        if section and (
            not isinstance(section, dict)
            or str(section.get("device", "none")).strip().lower() != "none"
        ):
            raise ValueError(f"DeepSpeed {name} must be disabled")
    for precision in ("bf16", "fp16"):
        section = config.get(precision, {})
        if not isinstance(section, dict) or section.get("enabled") is not False:
            raise ValueError(
                f"DeepSpeed {precision}.enabled must be false; owner state stays FP32"
            )
    autocast = config.get("torch_autocast")
    if (
        not isinstance(autocast, dict)
        or autocast.get("enabled") is not True
        or str(autocast.get("dtype", "")).lower() != "bfloat16"
        or autocast.get("lower_precision_safe_modules") != []
    ):
        raise ValueError(
            "DeepSpeed torch_autocast must enable bfloat16 compute with "
            "an empty lower_precision_safe_modules list"
        )
    if str(config.get("communication_data_type", "")).lower() != "fp32":
        raise ValueError("DeepSpeed communication_data_type must be fp32")
    for name in (
        "gradient_accumulation_steps",
        "train_batch_size",
        "train_micro_batch_size_per_gpu",
    ):
        if config.get(name) != "auto":
            raise ValueError(f"DeepSpeed {name} must be auto")
    if float(config.get("gradient_clipping", math.nan)) != 0.0:
        raise ValueError("DeepSpeed gradient_clipping must be 0")
    if "optimizer" in config or "scheduler" in config:
        raise ValueError("DeepSpeed must not declare an optimizer or scheduler")
    if config.get("zero_allow_untested_optimizer") is not True:
        raise ValueError("DeepSpeed must allow the owner-provided optimizer")
    return 1
