"""Shared compute precision policy; trainable parameters remain FP32."""

from contextlib import contextmanager

QUESTION_READER_INPUT_MODES = frozenset(
    {
        "last-query-reader-self-loop",
    }
)
REASONER_INPUT_MODES = QUESTION_READER_INPUT_MODES


def saved_reasoner_input(resolved):
    geometry = resolved.get("reasoner_geometry") or {}
    mode = geometry.get("input_mode")
    if mode not in REASONER_INPUT_MODES:
        raise ValueError("unknown saved reasoner input_mode")
    taps = 1 if mode in QUESTION_READER_INPUT_MODES else 6
    if geometry.get("tap_count", taps) != taps:
        raise ValueError("saved R tap_count disagrees with input_mode")
    return {"reasoner_input_mode": mode, "tap_count": taps}


def saved_reasoner_layers(resolved):
    """Read the explicit physical-layer count from the checkpoint."""
    layers = (resolved.get("reasoner_geometry") or {}).get("num_layers")
    if isinstance(layers, bool) or not isinstance(layers, int) or layers < 1:
        raise ValueError("saved R num_layers must be a positive integer")
    return layers


def saved_reasoner_loop_steps(resolved):
    """Read the explicit shared-layer loop count from the checkpoint."""
    loops = (resolved.get("reasoner_geometry") or {}).get("loop_steps")
    if isinstance(loops, bool) or not isinstance(loops, int) or loops < 1:
        raise ValueError("saved R loop_steps must be a positive integer")
    return loops


def saved_output_normalization(resolved):
    """Require the residual output-normalization contract."""
    mode = (resolved.get("reasoner_geometry") or {}).get("output_normalization")
    if mode != "residual":
        raise ValueError("saved R output_normalization must be residual or rmsnorm")
    return mode


def feedback_policy(source=None):
    """Serializable FP32-owner/BF16-compute policy with bounded R evaluation."""
    from collections.abc import Mapping

    source = source or {}
    if isinstance(source, Mapping):
        source = source.get("feedback_policy", source)
        get = source.get
    else:
        get = lambda name, default: getattr(source, name, default)
    dtype = get("reasoner_compute_dtype", "bfloat16")
    group = get("reasoner_eval_group_size", 1)
    if dtype not in ("bfloat16", "float32"):
        raise ValueError("reasoner_compute_dtype must be bfloat16 or float32")
    if isinstance(group, bool) or not isinstance(group, int) or group < 0:
        raise ValueError(
            "reasoner_eval_group_size must be 0 (caller batches) or a positive integer"
        )
    return {
        "reasoner_compute_dtype": dtype,
        "reasoner_eval_group_size": group,
        "executor_compute_dtype": "bfloat16",
        "owner_dtype": "float32",
        "tf32": "framework-default",
    }


def configure_feedback(model, source=None):
    """Configure before execution; never cast the shared F or detach gradients."""
    policy = feedback_policy(source)
    model._feedback_policy = policy
    model._feedback_dynamic_batches = policy["reasoner_eval_group_size"] == 0
    model._feedback_fixed_groups = policy["reasoner_eval_group_size"] > 1
    model._feedback_eval_batch_size = policy["reasoner_eval_group_size"]
    model.reasoner._compute_dtype = policy["reasoner_compute_dtype"]
    return policy


@contextmanager
def reasoner_compute_context(reasoner, device):
    """Optional FP32 R island. F and all differentiable casts stay unchanged."""
    if getattr(reasoner, "_compute_dtype", "bfloat16") == "float32":
        import torch

        with torch.autocast(device_type=device.type, enabled=False):
            yield
    else:
        yield


@contextmanager
def feedback_compute_context(model, device):
    if not getattr(model, "_feedback_fp32_trial", False):
        if hasattr(model, "_feedback_policy"):
            import torch

            mixed = (
                device.type == "cuda"
                or model.executor.get_input_embeddings().weight.dtype == torch.bfloat16
            )
            # Separate owner/donor calls must accumulate into FP32 owners
            # independently. Reusing autocast's BF16 parameter casts merges
            # their cast-backward nodes depending on the outer prefetch scope.
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=mixed,
                cache_enabled=False,
            ):
                yield
        else:
            yield
        return
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if model.executor.get_input_embeddings().weight.dtype != torch.float32:
        raise RuntimeError("FP32 feedback executor was not prepared")
    if torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("FP32 feedback trial requires TF32 disabled")
    # Does not disable gradients: prompt no_grad and live feedback BPTT remain
    # owned by materialize_feedback_latent, identically for train/eval.
    with (
        torch.autocast(device_type=device.type, enabled=False),
        sdpa_kernel(SDPBackend.MATH),
    ):
        yield
