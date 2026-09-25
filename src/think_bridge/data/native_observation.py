"""Observed native rollout tokens used by R data preparation and donor matching."""


def observed_native_cot(label, *, eos_token_id, close_token_ids=None):
    """Use real generated tokens for donor lengths, even for failed rollouts."""
    if "native_observed_cot_ids" in label:
        ids = label["native_observed_cot_ids"]
        count = label.get("native_observed_cot_token_count")
    else:
        ids = label.get("native_cot_ids")
        count = label.get("native_cot_token_count")
    if ids is None and count is None:
        return None
    if (
        not isinstance(ids, list)
        or type(count) is not int
        or count != len(ids)
        or any(type(t) is not int or t < 0 or t == eos_token_id for t in ids)
    ):
        raise ValueError(
            "Validation behavior lacks exact observed native CoT tokens/length. "
            "Regenerate validation_behavior with the Stage0 producer that retains "
            "failed observations; do not replace missing CoT with an empty target."
        )
    if "native_observed_cot_ids" in label:
        generated = label.get("native_generated_ids")
        if not isinstance(generated, list) or any(
            type(t) is not int or t < 0 for t in generated
        ):
            raise ValueError(
                "Observed native CoT lacks its original generated token stream"
            )
        if generated[:count] != ids:
            raise ValueError(
                "Observed native CoT differs from the generated token prefix"
            )
        if label.get("native_cot_ids") is not None and label["native_cot_ids"] != ids:
            raise ValueError(
                "Observed native CoT differs from the complete paired target"
            )
        if close_token_ids:
            close = list(close_token_ids)
            positions = [
                i
                for i in range(len(generated) - len(close) + 1)
                if generated[i : i + len(close)] == close
            ]
            if positions:
                expected = generated[: positions[-1]]
            else:
                expected = (
                    generated[:-1]
                    if generated and generated[-1] == eos_token_id
                    else generated
                )
            if expected != ids:
                raise ValueError(
                    "Observed native CoT does not end at the actual thinking boundary"
                )
    return list(ids)
