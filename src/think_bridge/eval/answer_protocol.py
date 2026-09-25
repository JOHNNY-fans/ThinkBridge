"""Shared greedy true-z request construction and ordered validation execution."""

from __future__ import annotations

import hashlib

ANSWER_BATCH = 64


def answer_protocol():
    return {
        "request_batch_size": ANSWER_BATCH,
        "max_num_seqs": 64,
        "temperature": 0.0,
        "top_p": 1.0,
        "tensor_parallel_size": 1,
        "request_order": "global-dataset-order-one-request-at-a-time",
        "padding": "repeat-first-row-of-final-request",
        "dp_assignment": "global-row-index-modulo-dp",
        "controls": "after-all-true-z-requests",
    }


def input_signature(prompt_ids, z, boundary_ids, eos_token_id):
    """Observability only; never an admission gate or a training objective."""
    import torch

    ids = [int(t) for t in prompt_ids]
    value = z.detach().cpu().to(torch.bfloat16).contiguous()
    return {
        "prompt_ids": ids,
        "z_bf16_sha256": hashlib.sha256(
            value.view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
        "z_shape": list(value.shape),
        "boundary_ids": [int(t) for t in boundary_ids],
        "eos_token_id": int(eos_token_id),
    }


def true_z_request(
    rows,
    z,
    *,
    boundary_ids,
    eos_token_id,
    seed,
    max_tokens,
    offset,
    physical=ANSWER_BATCH,
    namespace="benchmark",
    request_index=0,
):
    import torch
    from think_bridge.training.vllm_runtime import Route1ServiceRequest, TensorPayload

    real = len(rows)
    if not 0 < real <= physical or len(z) != real:
        raise ValueError("True-z request rows and latents must fit the request batch")
    selected = list(range(real)) + [0] * (physical - real)
    prompt_ids = tuple(tuple(int(t) for t in rows[i]["prompt_ids"]) for i in selected)
    value = torch.stack([z[i].detach().cpu().float() for i in selected])
    # Names identify requests only, never modify sampling seeds or DP assignment.
    transaction = (
        "eval-"
        + hashlib.sha256(namespace.encode()).hexdigest()[:16]
        + f"-{request_index}"
    )
    return Route1ServiceRequest(
        run_id="greedy-evaluation",
        transaction_id=transaction,
        rank=0,
        step=request_index,
        micro_step=0,
        chunk_id=request_index,
        sample_ids=tuple(range(offset, offset + physical)),
        sample_keys=tuple(f"answer-{offset + i}" for i in range(physical)),
        sample_is_padding=tuple(i >= real for i in range(physical)),
        prompt_ids=prompt_ids,
        prompt_lengths=tuple(map(len, prompt_ids)),
        detached_z=TensorPayload.from_torch(value),
        boundary_ids=tuple(map(int, boundary_ids)),
        generation_seed=int(seed),
        max_prefix_tokens=int(max_tokens),
        z_present=(True,) * physical,
        append_boundary=(True,) * physical,
        eos_token_id=int(eos_token_id),
        temperature=0.0,
    )


def answer_outputs(response, request, tokenizer, real):
    response.assert_matches(request)
    eos, maximum = request.eos_token_id, request.max_prefix_tokens
    return [
        {
            "ids": list(ids),
            "text": tokenizer.decode(ids, skip_special_tokens=True),
            "terminated": bool(ids and ids[-1] == eos),
            "cap_hit": bool(ids and ids[-1] != eos and len(ids) >= maximum),
        }
        for ids in response.prefix_token_ids[:real]
    ]


def ordered_true_z_outputs(
    model,
    tokenizer,
    rows,
    *,
    execute,
    seed,
    max_tokens,
    namespace,
    rank=0,
    world_size=1,
    no_progress=False,
):
    """Parallel HF work; globally ordered, non-overlapping vLLM requests.

    All ranks must enter, including ranks with no rows. Failure rendezvous runs
    before generation and after EACH request; controls only start after return.
    No tensors are communicated, and answer requests retain global indices.
    """
    import torch.distributed as distributed
    from think_bridge.model.evaluation_groups import (
        FixedReasonerGroups,
        evaluation_group_indices,
    )
    from think_bridge.model.reasoner_inference import reason_eval_prompts
    from think_bridge.training.runtime_backend import collect_rank_failures
    from think_bridge.training.progress import bridge_progress

    def rendezvous(failure):
        failures = collect_rank_failures(
            distributed, world_size=world_size, local_failure=failure
        )
        if failures:
            raise RuntimeError("ordered true-z evaluation: " + "; ".join(failures))

    groups = evaluation_group_indices(len(rows), group_size=ANSWER_BATCH)
    owned = [
        i
        for group_index, group in enumerate(groups)
        if group_index % world_size == rank
        for i in group
    ]
    cache, outputs = {}, {}
    failure = None
    try:
        capture = getattr(model, "_live_feedback_capture", None)
        first = {}
        if capture is not None:
            for i, row in enumerate(rows):
                first.setdefault(tuple(row["prompt_ids"]), i)
        observe = (
            None
            if capture is None
            else lambda group, ps, reason: capture.reason(
                ps,
                start=group[0],
                total=len(rows),
                reason=reason,
                published_indices=[
                    i for i in group if first[tuple(rows[i]["prompt_ids"])] == i
                ],
            )
        )
        fixed = FixedReasonerGroups(
            [r["prompt_ids"] for r in rows],
            lambda ps: reason_eval_prompts(model, ps),
            group_size=getattr(model, "_feedback_eval_batch_size", 64) or 64,
            observe=observe,
        )
        # Independent GPUs may prepare z together, but never alter request order.
        progress = bridge_progress(
            total=len(owned),
            desc="Validation z",
            unit="row",
            disabled=no_progress or rank != 0,
        )
        try:
            for indices in evaluation_group_indices(len(owned)):
                selected = [owned[i] for i in indices]
                cache.update(
                    zip(
                        (tuple(rows[i]["prompt_ids"]) for i in selected),
                        fixed.select(selected),
                    )
                )
                progress.update(len(selected))
        finally:
            progress.close()
    except Exception as exc:
        failure = f"rank={rank} z preparation: {type(exc).__name__}: {exc}"
    rendezvous(failure)
    progress = bridge_progress(
        total=len(rows),
        desc="Validation true-z",
        unit="row",
        disabled=no_progress or rank != 0,
    )
    try:
        for group_index, indices in enumerate(groups):
            failure = None
            if group_index % world_size == rank:
                try:
                    batch = [rows[i] for i in indices]
                    z = [cache[tuple(r["prompt_ids"])] for r in batch]
                    request = true_z_request(
                        batch,
                        z,
                        boundary_ids=model.boundary_ids.tolist(),
                        eos_token_id=model.eos_token_id,
                        seed=seed,
                        max_tokens=max_tokens,
                        offset=indices[0],
                        namespace=namespace,
                        request_index=group_index,
                    )
                    values = answer_outputs(
                        execute(request), request, tokenizer, len(batch)
                    )
                    for i, row, latent, value in zip(indices, batch, z, values):
                        value["reasoner_input"] = input_signature(
                            row["prompt_ids"],
                            latent,
                            request.boundary_ids,
                            request.eos_token_id,
                        )
                        outputs[str(row["record_id"])] = value
                except Exception as exc:
                    failure = f"rank={rank} answer request {group_index}: {type(exc).__name__}: {exc}"
            rendezvous(failure)
            progress.update(len(indices))
    finally:
        progress.close()
    return owned, outputs, cache
