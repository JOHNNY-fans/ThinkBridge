"""Prompt-isolated latent reasoner evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def reasoner_inference_metadata(model: Any = None) -> dict[str, Any]:
    if getattr(model, "_feedback_fp32_trial", False):
        return {
            "physical_batch_size": model._feedback_eval_batch_size,
            "padding": "right",
            "cuda_compute_dtype": "float32",
            "tf32": False,
            "attention": "math",
            "scope": "feedback-trial",
            **(
                {
                    "grouping": "ordered-global-groups",
                    "group_size": model._feedback_eval_batch_size,
                    "duplicate_prompts": "first-occurrence",
                    "tail": "actual-size",
                    "rank_assignment": "requested-rows-with-complete-r-groups",
                }
                if getattr(model, "_feedback_fixed_groups", False)
                else {}
            ),
        }
    size = getattr(model, "_feedback_eval_batch_size", 1)
    result = {
        "physical_batch_size": size,
        "padding": "right" if size > 1 else "none",
        "cuda_compute_dtype": "bfloat16",
    }
    if hasattr(model, "_inference_reasoner_input"):
        result.update(model._inference_reasoner_input)
    if hasattr(model, "_feedback_policy"):
        import torch

        result.update(model._feedback_policy)
        result.update(
            tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
            tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
        )
    if getattr(model, "_feedback_fixed_groups", False):
        result.update(
            grouping="ordered-global-groups",
            group_size=size,
            duplicate_prompts="first-occurrence",
            tail="actual-size",
            rank_assignment="requested-rows-with-complete-r-groups",
        )
    if getattr(model, "_feedback_dynamic_batches", False):
        result.update(
            physical_batch_size=None, padding="right", grouping="caller-batches"
        )
    return result


def reason_eval_prompts(
    model: Any, prompts: Sequence[Sequence[int]], *, chunk_size: int | None = None
) -> list[torch.Tensor]:
    """Materialize reasoner outputs under the saved evaluation batch contract.

    BF16 HF/R kernels are not governed by vLLM's batch-invariant option.
    Changing padding or batching here can change z and greedy answer tokens.
    This controls eval execution only; training batches and gradients are intact.
    """
    import torch

    if not prompts or any(len(row) == 0 for row in prompts):
        raise ValueError("R evaluation requires nonempty prompt rows")
    device = model.executor.get_input_embeddings().weight.device
    result = []
    if (
        getattr(model, "_feedback_fp32_trial", False)
        or getattr(model, "_feedback_fixed_groups", False)
        or getattr(model, "_feedback_dynamic_batches", False)
    ):
        # Fixed groups cannot be split by an unrelated answer/request chunk.
        size = (
            (chunk_size or len(prompts))
            if getattr(model, "_feedback_dynamic_batches", False)
            else model._feedback_eval_batch_size
        )
        if chunk_size is not None and not getattr(
            model, "_feedback_fixed_groups", False
        ):
            size = min(chunk_size, size)
        if size <= 0:
            raise ValueError("R evaluation chunk size must be positive")
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda"
                and not getattr(model, "_feedback_fp32_trial", False),
            ),
        ):
            for start in range(0, len(prompts), size):
                batch = prompts[start : start + size]
                width = max(len(row) for row in batch)
                ids = torch.zeros((len(batch), width), dtype=torch.long, device=device)
                mask = torch.zeros_like(ids, dtype=torch.bool)
                for i, row in enumerate(batch):
                    ids[i, : len(row)] = torch.tensor(
                        row, dtype=torch.long, device=device
                    )
                    mask[i, : len(row)] = True
                z = model.reason(ids, mask).detach()
                if z.ndim != 3 or z.shape[0] != len(batch):
                    raise RuntimeError("R evaluation returned an invalid latent shape")
                result.extend(z.unbind(0))
        return result
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ),
    ):
        for row in prompts:
            ids = torch.tensor([list(row)], dtype=torch.long, device=device)
            z = model.reason(ids, torch.ones_like(ids, dtype=torch.bool)).detach()
            if z.ndim != 3 or z.shape[0] != 1:
                raise RuntimeError("R evaluation returned an invalid latent shape")
            result.append(z[0])
    return result


def reason_eval_padded(
    model: Any,
    ids: torch.Tensor,
    mask: torch.Tensor,
    *,
    chunk_size: int | None = None,
    reader_context_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Remove caller padding before applying the shared evaluation grouping."""
    import torch

    if ids.ndim != 2 or ids.shape != mask.shape or ids.shape[0] == 0:
        raise ValueError("R eval ids and masks must share nonempty [batch, sequence]")
    valid = mask.bool()
    lengths = valid.long().sum(-1)
    expected = torch.arange(ids.shape[1], device=ids.device)[None, :] < lengths[:, None]
    if bool((lengths <= 0).any()) or not torch.equal(valid, expected):
        raise ValueError("R eval requires nonempty right-padded prompt masks")
    if reader_context_mask is not None:
        if (
            reader_context_mask.shape != mask.shape
            or reader_context_mask.dtype != torch.bool
            or reader_context_mask.device != ids.device
            or bool((reader_context_mask & ~valid).any())
            or not bool(reader_context_mask.any(-1).all())
        ):
            raise ValueError(
                "Reader context mask must select nonempty valid prompt positions"
            )
        if any(
            getattr(model, key, False)
            for key in (
                "_feedback_fp32_trial",
                "_feedback_fixed_groups",
                "_feedback_dynamic_batches",
            )
        ):
            raise ValueError(
                "Reader visibility intervention requires maintained singleton BF16 R evaluation"
            )
        # Keep the full prompt and its positions. Only the Reader gets a restricted mask.
        outputs = []
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=ids.device.type,
                dtype=torch.bfloat16,
                enabled=ids.device.type == "cuda",
            ),
        ):
            for i, n in enumerate(lengths.cpu().tolist()):
                z = model.reason(
                    ids[i : i + 1, :n],
                    valid[i : i + 1, :n],
                    reader_context_mask=reader_context_mask[i : i + 1, :n],
                ).detach()
                if z.ndim != 3 or z.shape[0] != 1:
                    raise RuntimeError("R evaluation returned an invalid latent shape")
                outputs.append(z)
        return torch.cat(outputs, dim=0)
    cpu_ids, cpu_lengths = ids.detach().cpu().tolist(), lengths.cpu().tolist()
    return torch.stack(
        reason_eval_prompts(
            model,
            [row[:n] for row, n in zip(cpu_ids, cpu_lengths)],
            chunk_size=chunk_size,
        )
    )
