"""BF16 R/F with independent reasoner and answer batch sizes."""

from contextlib import contextmanager

ANSWER_GROUP_SIZE = 8
REASONER_BATCH_SIZE = 64


def hf_evaluation_protocol(answer_group_size=ANSWER_GROUP_SIZE, reasoner_batch_size=REASONER_BATCH_SIZE):
    if type(answer_group_size) is not int or answer_group_size <= 0:
        raise ValueError("answer_group_size must be a positive integer")
    if type(reasoner_batch_size) is not int or reasoner_batch_size <= 0:
        raise ValueError("reasoner_batch_size must be a positive integer")
    return dict(
        schema=(f"hf-bf16-singleton-r-group{answer_group_size}-v1"
                if reasoner_batch_size == 1 else
                f"hf-bf16-r{reasoner_batch_size}-group{answer_group_size}-v2"),
        backend="hf",
        reasoner_compute_dtype="bfloat16",
        executor_compute_dtype="bfloat16",
        reasoner_batch_size=reasoner_batch_size,
        answer_group_size=answer_group_size,
        grouping="ordered-complete-groups-before-rank-sharding",
        controls="separate-from-true-z",
        temperature=0.0,
        top_p=1.0,
        tf32_matmul=False,
    )


def configure_hf_evaluation(model, *, answer_group_size=None, reasoner_batch_size=None):
    import torch
    from think_bridge.model.feedback_precision import configure_feedback

    if reasoner_batch_size is None:
        reasoner_batch_size = getattr(model, "_feedback_eval_batch_size", REASONER_BATCH_SIZE)
    # Legacy caller-sized mode gets the public bounded default in HF evaluation.
    reasoner_batch_size = reasoner_batch_size or REASONER_BATCH_SIZE
    configure_feedback(model, dict(reasoner_compute_dtype="bfloat16",
                                   reasoner_eval_group_size=reasoner_batch_size))
    torch.backends.cuda.matmul.allow_tf32 = False
    if answer_group_size is None:
        answer_group_size = getattr(model, "_answer_evaluation_protocol", {}).get(
            "answer_group_size", ANSWER_GROUP_SIZE
        )
    model._answer_evaluation_protocol = hf_evaluation_protocol(answer_group_size, reasoner_batch_size)


@contextmanager
def hf_evaluation_context(model, *, answer_group_size=None):
    import torch

    names = (
        "_feedback_policy",
        "_feedback_dynamic_batches",
        "_feedback_fixed_groups",
        "_feedback_eval_batch_size",
    )
    missing = object()
    prior = {name: getattr(model, name, missing) for name in names}
    dtype = getattr(model.reasoner, "_compute_dtype", missing)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        configure_hf_evaluation(model, answer_group_size=answer_group_size)
        yield
    finally:
        for name, value in prior.items():
            if value is missing:
                if hasattr(model, name):
                    delattr(model, name)
            else:
                setattr(model, name, value)
        if dtype is missing:
            if hasattr(model.reasoner, "_compute_dtype"):
                delattr(model.reasoner, "_compute_dtype")
        else:
            model.reasoner._compute_dtype = dtype
        torch.backends.cuda.matmul.allow_tf32 = tf32


def canonical_reason(model, prompts):
    """Apply the configured R batch independently of answer decoding."""
    from think_bridge.model.reasoner_inference import reason_eval_prompts

    with hf_evaluation_context(model):
        return reason_eval_prompts(model, prompts)


def decode_groups(model, tokenizer, requests, *, max_tokens, answer_group_size=None):
    from think_bridge.training.eval import _free_answers

    if answer_group_size is None:
        answer_group_size = getattr(model, "_answer_evaluation_protocol", {}).get(
            "answer_group_size", ANSWER_GROUP_SIZE
        )
    hf_evaluation_protocol(answer_group_size)  # Validate before grouping.
    result = []
    for start in range(0, len(requests), answer_group_size):
        result.extend(
            _free_answers(
                model,
                tokenizer,
                requests[start : start + answer_group_size],
                max_tokens=max_tokens,
                temperature=0.0,
                top_p=1.0,
            )
        )
    return result
