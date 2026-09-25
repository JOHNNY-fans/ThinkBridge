"""Stopped current-policy trajectories and aligned answer-state builders for Bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as torch_functional
import torch.utils.checkpoint

from think_bridge.model.contract import ANSWER_CAPACITY, KZ


BranchGeometry = Literal[
    "natural_cot",
    "direct",
    "deployed_z",
    "deployed_z_cot_suffix",
    "deployed_null",
    "course_no_z",
]


@dataclass(frozen=True)
class AnswerBranchBatch:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    logit_indices: torch.Tensor
    target_ids: torch.Tensor
    target_mask: torch.Tensor


@dataclass(frozen=True)
class StoppedAnswerPrefix:
    token_ids: torch.Tensor
    token_mask: torch.Tensor


@dataclass(frozen=True)
class BatchedIncrementalExecutorState:
    past_key_values: Any
    physical_cache_length: int
    attention_mask: torch.Tensor
    next_logical_positions: torch.Tensor


def freeze_executor_parameters(executor: torch.nn.Module) -> None:
    for parameter in executor.parameters():
        parameter.requires_grad_(False)


def executor_forward_pre_lm(
    executor: torch.nn.Module,
    *,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    logit_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run frozen F and project only explicitly requested hidden positions."""

    core = getattr(executor, "model", None)
    head = getattr(executor, "lm_head", None)
    if core is None or head is None:
        raise TypeError("executor must expose HF-compatible .model and .lm_head")
    output = core(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        return_dict=True,
    )
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None or hidden.shape != inputs_embeds.shape:
        raise RuntimeError("frozen executor omitted aligned post-norm hidden states")
    if logit_indices is None:
        return hidden, None
    if (
        logit_indices.ndim != 2
        or logit_indices.size(0) != hidden.size(0)
        or logit_indices.dtype not in (torch.int32, torch.int64)
        or bool((logit_indices < 0).any())
        or bool((logit_indices >= hidden.size(1)).any())
    ):
        raise ValueError("executor logit indices must be valid [B,T] positions")
    selected = hidden.gather(
        1, logit_indices.long().unsqueeze(-1).expand(-1, -1, hidden.size(-1))
    )
    return hidden, head(selected).float()


def _prefix_lengths(
    mask: torch.Tensor, label: str, *, permit_empty: bool = False
) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError(f"{label} mask must be [B,T]")
    valid = mask.bool()
    lengths = valid.long().sum(dim=-1)
    if not permit_empty and bool((lengths <= 0).any()):
        raise ValueError(f"{label} contains an empty row")
    expected = torch.arange(mask.size(1), device=mask.device).unsqueeze(
        0
    ) < lengths.unsqueeze(1)
    if not torch.equal(valid, expected):
        raise ValueError(f"{label} mask must be a strict right-padded prefix")
    return lengths


def _pad_embeddings(rows: list[torch.Tensor]) -> torch.Tensor:
    maximum = max(int(row.size(0)) for row in rows)
    return torch.stack(
        [torch_functional.pad(row, (0, 0, 0, maximum - row.size(0))) for row in rows],
        dim=0,
    )


def _pad_long(rows: list[torch.Tensor], *, value: int = 0) -> torch.Tensor:
    maximum = max(int(row.numel()) for row in rows)
    return torch.stack(
        [
            torch_functional.pad(row, (0, maximum - row.numel()), value=int(value))
            for row in rows
        ],
        dim=0,
    )


def _pad_bool(rows: list[torch.Tensor]) -> torch.Tensor:
    maximum = max(int(row.numel()) for row in rows)
    return torch.stack(
        [
            torch_functional.pad(row, (0, maximum - row.numel()), value=False)
            for row in rows
        ],
        dim=0,
    )


def build_answer_branch(
    *,
    embedding: Any,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    boundary_ids: torch.Tensor,
    answer_ids: torch.Tensor,
    answer_mask: torch.Tensor,
    geometry: BranchGeometry,
    z: torch.Tensor | None = None,
    cot_ids: torch.Tensor | None = None,
    cot_mask: torch.Tensor | None = None,
) -> AnswerBranchBatch:
    """Build one teacher/true/null/donor branch on shared answer-step semantics."""

    if prompt_ids.ndim != 2 or prompt_ids.shape != prompt_mask.shape:
        raise ValueError("prompt ids/mask must share [B,S]")
    if answer_ids.ndim != 2 or answer_ids.shape != answer_mask.shape:
        raise ValueError("answer ids/mask must share [B,T]")
    if prompt_ids.size(0) != answer_ids.size(0):
        raise ValueError("prompt and answer batch sizes differ")
    prompt_lengths = _prefix_lengths(prompt_mask, "prompt")
    answer_lengths = _prefix_lengths(answer_mask, "answer")
    if bool((answer_lengths > ANSWER_CAPACITY).any()):
        raise ValueError("answer trajectory exceeds the configured answer horizon")
    if boundary_ids.ndim != 1 or boundary_ids.numel() <= 0:
        raise ValueError("boundary ids must be one non-empty vector")
    if boundary_ids.device != prompt_ids.device:
        boundary_ids = boundary_ids.to(prompt_ids.device)
    batch_size = int(prompt_ids.size(0))
    boundary_embeds = embedding(boundary_ids).detach()
    hidden_size = int(boundary_embeds.size(-1))

    cot_lengths = None
    if geometry == "natural_cot":
        if (
            cot_ids is None
            or cot_mask is None
            or cot_ids.shape != cot_mask.shape
            or cot_ids.size(0) != batch_size
        ):
            raise ValueError("natural teacher requires aligned CoT ids/mask")
        cot_lengths = _prefix_lengths(cot_mask, "natural cot")
        if z is not None:
            raise ValueError("natural teacher cannot carry z")
    elif geometry == "course_no_z":
        if (
            z is not None
            or cot_ids is None
            or cot_mask is None
            or cot_ids.shape != cot_mask.shape
            or cot_ids.size(0) != batch_size
        ):
            raise ValueError("course no-z control requires aligned CoT prefix and no z")
        cot_lengths = _prefix_lengths(
            cot_mask, "course no-z cot prefix", permit_empty=True
        )
    elif geometry == "direct":
        if z is not None or cot_ids is not None or cot_mask is not None:
            raise ValueError("direct teacher carries no CoT or latent tensor")
    elif geometry in {"deployed_z", "deployed_z_cot_suffix"}:
        if (
            z is None
            or (
                z.ndim != 3 or z.size(0) != batch_size or z.size(1) not in {32, 64, 128}
            )
            or z.size(-1) != hidden_size
        ):
            raise ValueError(f"deployed z must be [B,K,dF], K in 32/64/128")
        if geometry == "deployed_z" and (cot_ids is not None or cot_mask is not None):
            raise ValueError("deployed z branch cannot carry native CoT tokens")
        if geometry == "deployed_z_cot_suffix":
            if (
                cot_ids is None
                or cot_mask is None
                or cot_ids.shape != cot_mask.shape
                or cot_ids.size(0) != batch_size
            ):
                raise ValueError("answer course requires aligned CoT suffix ids/mask")
            cot_lengths = _prefix_lengths(
                cot_mask, "answer-course cot suffix", permit_empty=True
            )
    elif geometry == "deployed_null":
        if z is not None or cot_ids is not None or cot_mask is not None:
            raise ValueError("exact null branch has no condition tensor or tokens")
    else:
        raise ValueError(f"unknown answer branch geometry: {geometry}")

    embed_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    position_rows: list[torch.Tensor] = []
    logit_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    for row in range(batch_size):
        prompt_length = int(prompt_lengths[row])
        answer_length = int(answer_lengths[row])
        prompt_embed = embedding(prompt_ids[row, :prompt_length]).detach()
        answer_embed = embedding(answer_ids[row, :answer_length]).detach()
        prompt_positions = torch.arange(
            prompt_length, dtype=torch.long, device=prompt_ids.device
        )
        if geometry in {"natural_cot", "course_no_z"}:
            cot_length = int(cot_lengths[row])
            condition_embed = embedding(cot_ids[row, :cot_length]).detach()
            condition_positions = torch.arange(
                prompt_length,
                prompt_length + cot_length,
                dtype=torch.long,
                device=prompt_ids.device,
            )
            boundary_start = prompt_length + cot_length
        elif geometry in {"deployed_z", "deployed_z_cot_suffix"}:
            # R is intentionally FP32 while the frozen executor is BF16 on
            # CUDA.  torch.cat promotes mixed FP32/BF16 inputs to FP32, which
            # then fails in Qwen's BF16 projection layers (most visibly under
            # the 4B ZeRO-1 path).  Cast only at the frozen-F interface; this
            # remains differentiable and therefore preserves the z -> R
            # gradient while keeping R and its optimizer state FP32.
            z_embed = z[row].to(
                device=boundary_embeds.device,
                dtype=boundary_embeds.dtype,
            )
            if geometry == "deployed_z_cot_suffix":
                cot_length = int(cot_lengths[row])
                cot_embed = embedding(cot_ids[row, :cot_length]).detach()
                condition_embed = torch.cat((z_embed, cot_embed), dim=0)
            else:
                condition_embed = z_embed
            condition_positions = torch.arange(
                prompt_length,
                prompt_length + condition_embed.size(0),
                dtype=torch.long,
                device=prompt_ids.device,
            )
            boundary_start = prompt_length + condition_embed.size(0)
        elif geometry == "deployed_null":
            condition_embed = boundary_embeds.new_zeros((KZ, hidden_size))
            condition_positions = torch.arange(
                prompt_length,
                prompt_length + KZ,
                dtype=torch.long,
                device=prompt_ids.device,
            )
            boundary_start = prompt_length + KZ
        else:
            condition_embed = boundary_embeds.new_empty((0, hidden_size))
            condition_positions = torch.empty(
                0, dtype=torch.long, device=prompt_ids.device
            )
            boundary_start = prompt_length
        if geometry == "direct":
            # The direct prompt already contains the empty-think boundary.
            # Appending ``boundary_ids`` here would silently create a second

            answer_positions = torch.arange(
                prompt_length,
                prompt_length + answer_length,
                dtype=torch.long,
                device=prompt_ids.device,
            )
            inputs = torch.cat((prompt_embed, answer_embed), dim=0)
            positions = torch.cat((prompt_positions, answer_positions), dim=0)
            qstar_index = prompt_length - 1
        else:
            suffix_positions = torch.arange(
                boundary_start,
                boundary_start + boundary_ids.numel() + answer_length,
                dtype=torch.long,
                device=prompt_ids.device,
            )
            inputs = torch.cat(
                (prompt_embed, condition_embed, boundary_embeds, answer_embed), dim=0
            )
            positions = torch.cat(
                (prompt_positions, condition_positions, suffix_positions), dim=0
            )
            qstar_index = (
                prompt_length + condition_embed.size(0) + boundary_ids.numel() - 1
            )
        if geometry == "deployed_z_cot_suffix":
            cot_length = int(cot_lengths[row])
            z_last = prompt_length + z.size(1) - 1
            suffix_indices = z_last + torch.arange(
                cot_length, dtype=torch.long, device=prompt_ids.device
            )
            answer_start = (
                prompt_length + z.size(1) + cot_length + boundary_ids.numel() - 1
            )
            answer_indices = answer_start + torch.arange(
                answer_length, dtype=torch.long, device=prompt_ids.device
            )
            logit_indices = torch.cat((suffix_indices, answer_indices), dim=0)
            target = (
                torch.cat(
                    (cot_ids[row, :cot_length], answer_ids[row, :answer_length]),
                    dim=0,
                )
                .detach()
                .long()
            )
        else:
            logit_indices = qstar_index + torch.arange(
                answer_length, dtype=torch.long, device=prompt_ids.device
            )
            target = answer_ids[row, :answer_length].detach().long()
        embed_rows.append(inputs)
        mask_rows.append(
            torch.ones(inputs.size(0), dtype=torch.bool, device=prompt_ids.device)
        )
        position_rows.append(positions)
        logit_rows.append(logit_indices)
        target_rows.append(target)

    target_lengths = torch.tensor(
        [int(row.numel()) for row in target_rows],
        dtype=torch.long,
        device=answer_ids.device,
    )
    maximum_target = int(target_lengths.max())
    target_mask = torch.arange(maximum_target, device=answer_ids.device).unsqueeze(
        0
    ) < target_lengths.unsqueeze(1)
    return AnswerBranchBatch(
        inputs_embeds=_pad_embeddings(embed_rows),
        attention_mask=_pad_bool(mask_rows),
        position_ids=_pad_long(position_rows),
        logit_indices=torch.stack(
            [
                torch_functional.pad(row, (0, maximum_target - row.numel()), value=0)
                for row in logit_rows
            ],
            dim=0,
        ),
        target_ids=_pad_long(target_rows),
        target_mask=target_mask.detach(),
    )


def forward_answer_hidden(
    executor: torch.nn.Module,
    batch: AnswerBranchBatch,
    *,
    live: bool,
    gradient_checkpointing: bool = False,
) -> torch.Tensor:
    """Return answer prediction hidden states; F stays frozen while z may be live."""

    core = getattr(executor, "model", None)
    if core is None:
        raise TypeError("executor must expose its frozen base model as .model")

    def run_core(
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        output = core(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None or hidden.shape != inputs_embeds.shape:
            raise RuntimeError("frozen executor did not return aligned hidden states")
        return hidden

    if live:
        if not batch.inputs_embeds.requires_grad:
            raise RuntimeError("live answer branch lost its z gradient")
        hidden = (
            torch.utils.checkpoint.checkpoint(
                run_core,
                batch.inputs_embeds,
                batch.attention_mask,
                batch.position_ids,
                use_reentrant=False,
            )
            if gradient_checkpointing
            else run_core(batch.inputs_embeds, batch.attention_mask, batch.position_ids)
        )
    else:
        with torch.no_grad():
            hidden = run_core(
                batch.inputs_embeds, batch.attention_mask, batch.position_ids
            )
    indices = batch.logit_indices
    gathered = hidden.gather(1, indices.unsqueeze(-1).expand(-1, -1, hidden.size(-1)))
    return gathered if live else gathered.detach()


def _build_true_z_prefill(
    *,
    embedding: Any,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    z: torch.Tensor,
    boundary_ids: torch.Tensor,
    cot_ids: torch.Tensor | None = None,
    cot_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build q + optional current hint + z + boundary, with compact positions.

    This is a fresh scoring/generation prefill. R's q+z feedback cache cannot
    be reused after inserting a CoT prefix before z.
    """
    prompt_lengths = _prefix_lengths(prompt_mask, "prompt")
    if z.ndim != 3 or z.size(0) != prompt_ids.size(0) or z.size(1) not in {32, 64, 128}:
        raise ValueError(f"deployed z must be [B,K,dF], K in 32/64/128")
    if (cot_ids is None) != (cot_mask is None):
        raise ValueError("generation CoT ids/mask must be jointly supplied")
    cot_lengths = None
    if cot_ids is not None:
        if (
            cot_ids.ndim != 2
            or cot_ids.shape != cot_mask.shape
            or cot_ids.size(0) != prompt_ids.size(0)
        ):
            raise ValueError("generation CoT prefix must align with prompt rows")
        cot_lengths = _prefix_lengths(
            cot_mask, "generation cot prefix", permit_empty=True
        )
    boundary_ids = boundary_ids.to(prompt_ids.device)
    boundary_embed = embedding(boundary_ids).detach()
    if z.size(-1) != boundary_embed.size(-1):
        raise ValueError(f"deployed z must be [B,K,dF], K in 32/64/128")
    rows, masks, positions = [], [], []
    for row in range(prompt_ids.size(0)):
        prompt_length = int(prompt_lengths[row])
        parts = [embedding(prompt_ids[row, :prompt_length]).detach()]
        if cot_lengths is not None:
            parts.append(embedding(cot_ids[row, : int(cot_lengths[row])]).detach())
        parts.extend(
            (
                z[row]
                .detach()
                .to(device=boundary_embed.device, dtype=boundary_embed.dtype),
                boundary_embed,
            )
        )
        inputs = torch.cat(parts, dim=0)
        rows.append(inputs)
        masks.append(
            torch.ones(inputs.size(0), dtype=torch.bool, device=prompt_ids.device)
        )
        positions.append(
            torch.arange(inputs.size(0), dtype=torch.long, device=prompt_ids.device)
        )
    return _pad_embeddings(rows), _pad_bool(masks), _pad_long(positions)


def incremental_executor_prefill_batched(
    executor: torch.nn.Module,
    *,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[BatchedIncrementalExecutorState, torch.Tensor]:
    lengths = _prefix_lengths(attention_mask, "incremental prefill")
    core = getattr(executor, "model", None)
    head = getattr(executor, "lm_head", None)
    if core is None or head is None:
        raise TypeError("incremental executor requires .model and .lm_head")
    physical_length = int(inputs_embeds.size(1))
    output = core(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache_position=torch.arange(
            physical_length, dtype=torch.long, device=inputs_embeds.device
        ),
        use_cache=True,
        return_dict=True,
    )
    hidden = getattr(output, "last_hidden_state", None)
    past = getattr(output, "past_key_values", None)
    if hidden is None or hidden.shape != inputs_embeds.shape or past is None:
        raise RuntimeError("incremental executor prefill omitted hidden/cache state")
    rows = torch.arange(inputs_embeds.size(0), device=inputs_embeds.device)
    last = lengths - 1
    logits = head(hidden[rows, last].unsqueeze(1)).float()
    return (
        BatchedIncrementalExecutorState(
            past_key_values=past,
            physical_cache_length=physical_length,
            attention_mask=attention_mask.bool(),
            next_logical_positions=position_ids[rows, last] + 1,
        ),
        logits,
    )


def incremental_executor_step_batched(
    executor: torch.nn.Module,
    state: BatchedIncrementalExecutorState,
    *,
    token_embed: torch.Tensor,
    active_mask: torch.Tensor,
) -> tuple[BatchedIncrementalExecutorState, torch.Tensor]:
    core = executor.model
    head = executor.lm_head
    attention_mask = torch.cat((state.attention_mask, active_mask.unsqueeze(1)), dim=1)
    output = core(
        inputs_embeds=token_embed,
        attention_mask=attention_mask,
        position_ids=state.next_logical_positions.unsqueeze(1),
        cache_position=torch.tensor(
            [state.physical_cache_length],
            dtype=torch.long,
            device=token_embed.device,
        ),
        past_key_values=state.past_key_values,
        use_cache=True,
        return_dict=True,
    )
    hidden = getattr(output, "last_hidden_state", None)
    past = getattr(output, "past_key_values", None)
    if hidden is None or hidden.shape[:2] != token_embed.shape[:2] or past is None:
        raise RuntimeError("incremental executor step omitted hidden/cache state")
    return (
        BatchedIncrementalExecutorState(
            past_key_values=past,
            physical_cache_length=state.physical_cache_length + 1,
            attention_mask=attention_mask,
            next_logical_positions=(state.next_logical_positions + active_mask.long()),
        ),
        head(hidden).float(),
    )


@torch.no_grad()
def generate_true_z_prefix(
    executor: torch.nn.Module,
    *,
    embedding: Any,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    z: torch.Tensor,
    boundary_ids: torch.Tensor,
    eos_token_id: int,
    max_steps: int,
    temperature: float = 0.0,
    seed: int = 42,
    cot_ids: torch.Tensor | None = None,
    cot_mask: torch.Tensor | None = None,
) -> StoppedAnswerPrefix:
    """Generate one stopped deployed true-z prefix through first EOS or H."""

    if max_steps <= 0:
        raise ValueError("greedy trajectory horizon must be positive")
    import math

    if not math.isfinite(float(temperature)) or temperature < 0:
        raise ValueError("prefix temperature must be finite and nonnegative")
    inputs, mask, positions = _build_true_z_prefill(
        embedding=embedding,
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        z=z.detach(),
        boundary_ids=boundary_ids.detach(),
        cot_ids=cot_ids,
        cot_mask=cot_mask,
    )
    state, logits = incremental_executor_prefill_batched(
        executor,
        inputs_embeds=inputs,
        attention_mask=mask,
        position_ids=positions,
    )
    batch_size = int(prompt_ids.size(0))
    output_ids = torch.full(
        (batch_size, int(max_steps)),
        int(eos_token_id),
        dtype=torch.long,
        device=prompt_ids.device,
    )
    output_mask = torch.zeros_like(output_ids, dtype=torch.bool)
    active = torch.ones(batch_size, dtype=torch.bool, device=prompt_ids.device)
    # Independent requests use the caller seed, without row/rank offsets.
    # Per-row streams avoid changing samples when physical batches are split.
    generators = (
        [
            torch.Generator(device=prompt_ids.device).manual_seed(int(seed))
            for _ in range(batch_size)
        ]
        if temperature > 0
        else []
    )
    for step in range(int(max_steps)):
        if temperature == 0:
            predicted = logits[:, -1].argmax(dim=-1).detach().long()
        else:
            probabilities = torch.softmax(logits[:, -1].float() / temperature, dim=-1)
            predicted = (
                torch.stack(
                    [
                        torch.multinomial(row, 1, generator=generator).squeeze(0)
                        for row, generator in zip(probabilities, generators)
                    ]
                )
                .detach()
                .long()
            )
        output_ids[active, step] = predicted[active]
        output_mask[active, step] = True
        active = active & predicted.ne(int(eos_token_id))
        if not bool(active.any()) or step + 1 == int(max_steps):
            break
        safe_ids = torch.where(
            active, predicted, torch.full_like(predicted, int(eos_token_id))
        )
        token_embed = embedding(safe_ids).detach().unsqueeze(1)
        state, logits = incremental_executor_step_batched(
            executor,
            state,
            token_embed=token_embed,
            active_mask=active,
        )
    return StoppedAnswerPrefix(output_ids.detach(), output_mask.detach())
