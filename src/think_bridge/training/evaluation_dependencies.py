"""Check lightweight evaluation imports before allocating training resources."""

from __future__ import annotations


def check_evaluation_imports() -> dict[str, str]:
    from importlib import import_module
    import sys

    dependencies = {
        "think_bridge.model.evaluation_groups": (
            "evaluation_group_indices",
            "FixedReasonerGroups",
        ),
        "think_bridge.model.reasoner_inference": ("reason_eval_prompts",),
        "think_bridge.eval.answer_protocol": ("ordered_true_z_outputs",),
    }
    origins = {}
    for name, symbols in dependencies.items():
        try:
            module = import_module(name)
            for symbol in symbols:
                if not callable(getattr(module, symbol, None)):
                    raise ImportError(f"{name} is missing callable {symbol}")
        except ImportError as exc:
            raise RuntimeError(
                f"Training evaluation dependency unavailable: {name}; "
                f"python={sys.executable}. Install the complete ThinkBridge package "
                "and check PYTHONPATH/package import locations before restarting."
            ) from exc
        origins[name] = str(module.__file__)
    return origins
