"""ThinkBridge public package."""

from __future__ import annotations

try:
    from importlib.metadata import version as _pkg_version

    _discovered_version = _pkg_version("think-bridge")
    __version__ = str(_discovered_version) if _discovered_version else "1.0.0"
except Exception:
    __version__ = "1.0.0"

__all__ = [
    "TrainingConfig",
    "__version__",
]


def __getattr__(name: str):
    if name == "TrainingConfig":
        from think_bridge.model.training_config import TrainingConfig

        return TrainingConfig
    raise AttributeError(name)
