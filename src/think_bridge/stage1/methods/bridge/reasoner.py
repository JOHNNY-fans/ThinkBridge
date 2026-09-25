"""Materialize 64 latent states from one frozen prompt representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from think_bridge.model.cache_utils import new_cache
from think_bridge.model.feedback_precision import QUESTION_READER_INPUT_MODES
from think_bridge.model.reasoner import RecurrentReasoner


class FeedbackReasoner(RecurrentReasoner):
    def __init__(
        self,
        d_model: int,
        *,
        latent_steps: int,
        latents_per_step: int,
        loop_steps: int = 2,
        num_layers: int = 2,
        num_heads: int,
        dim_feedforward: int,
        bound_scale_init: float,
        zero_residual_init: bool = False,
        output_normalization: str = "residual",
        input_mode: str = "last-query-reader-self-loop",
        dropout_p: float = 0.0,
        dropout_views: int = 1,
    ) -> None:
        super().__init__(
            d_model,
            latent_steps=int(latent_steps),
            latents_per_step=int(latents_per_step),
            loop_steps=loop_steps,
            num_layers=num_layers,
            num_heads=int(num_heads),
            dim_feedforward=int(dim_feedforward),
            bound_scale_init=float(bound_scale_init),
            zero_residual_init=bool(zero_residual_init),
            output_normalization=str(output_normalization),
            input_mode=input_mode,
            dropout_p=float(dropout_p),
            dropout_views=int(dropout_views),
        )


@dataclass(frozen=True)
class FeedbackLatent:
    z: torch.Tensor
    raw_z: torch.Tensor
    tap_histories: tuple[torch.Tensor, ...]


def question_taps(output: Any, reasoner: RecurrentReasoner) -> tuple[torch.Tensor, ...]:
    last = getattr(output, "last_hidden_state", None)
    if last is None:
        raise RuntimeError("question reader requires F last_hidden_state")
    return (last,)


def materialize_feedback_latent(
    executor: torch.nn.Module,
    reasoner: FeedbackReasoner,
    *,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    embedding_rms: torch.Tensor,
    reader_context_mask: torch.Tensor | None = None,
) -> FeedbackLatent:
    """Read frozen prompt states once; preserve gradients through the latent path."""

    if prompt_ids.ndim != 2 or prompt_ids.shape != prompt_mask.shape:
        raise ValueError("feedback prompts must share [B,S] ids/mask")
    valid = prompt_mask.bool()
    if reader_context_mask is not None:
        if (
            reasoner.input_mode != "last-query-reader-self-loop"
            or reasoner.latent_steps != 1
        ):
            raise ValueError(
                "Reader visibility intervention requires one-step last-query-reader-self-loop R"
            )
        if reasoner.training:
            raise ValueError("Reader visibility intervention is evaluation-only")
        if (
            reader_context_mask.shape != valid.shape
            or reader_context_mask.device != valid.device
            or reader_context_mask.dtype != torch.bool
        ):
            raise ValueError(
                "Reader context mask must be boolean and match prompt geometry/device"
            )
        if bool((reader_context_mask & ~valid).any()) or not bool(
            reader_context_mask.any(-1).all()
        ):
            raise ValueError(
                "Reader context mask must select nonempty valid prompt positions"
            )
    lengths = valid.long().sum(dim=-1)
    if bool((lengths <= 0).any()):
        raise ValueError("feedback prompts cannot be empty")
    expected = torch.arange(prompt_ids.size(1), device=prompt_ids.device).unsqueeze(
        0
    ) < lengths.unsqueeze(1)
    if not torch.equal(valid, expected):
        raise ValueError("feedback prompt masks must be right-padded prefixes")
    base = getattr(executor, "model", None)
    if base is None:
        raise TypeError("feedback executor must expose its frozen base model")
    executor_dtype = executor.get_input_embeddings().weight.dtype
    width = int(prompt_ids.size(1))
    positions = (valid.long().cumsum(dim=-1) - 1).clamp_min(0)
    cache = new_cache()

    def forward_f(**kwargs: Any) -> Any:
        return base(
            **kwargs,
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=reasoner.input_mode not in QUESTION_READER_INPUT_MODES,
            return_dict=True,
        )

    with torch.no_grad():
        prompt_output = forward_f(
            input_ids=prompt_ids,
            attention_mask=valid,
            position_ids=positions,
            cache_position=torch.arange(width, device=prompt_ids.device),
        )
    cache = prompt_output.past_key_values
    histories = tuple(tap.detach() for tap in question_taps(prompt_output, reasoner))
    # Only the cache and the selected tap histories cross the prompt boundary.
    # Keeping the HF output object alive would retain every layer's hidden
    # state tuple for the entire recurrent feedback pass.
    del prompt_output
    from think_bridge.model.feedback_precision import reasoner_compute_context

    with reasoner_compute_context(reasoner, prompt_ids.device):
        raw_z = reasoner.emit_step(
            histories, valid, step_index=0, reader_context_mask=reader_context_mask
        )
        z = reasoner.bound(raw_z, embedding_rms=embedding_rms)
    if z.shape[1] != 64 or z.dtype != torch.float32:
        raise RuntimeError("reasoner must return 64 FP32 latent states")
    return FeedbackLatent(z=z, raw_z=raw_z, tap_histories=histories)
