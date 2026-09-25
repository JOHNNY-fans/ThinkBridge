"""Bridge frozen executor and latent reasoner for deployed answer generation."""

from __future__ import annotations


from dataclasses import dataclass
import math
import time
from typing import Mapping, Any

import torch
import torch.distributed as torch_distributed
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint


from think_bridge.model.objectives import (
    _per_occurrence_gold_ce_rows_from_hidden_validated,
    route1_course_reduction_scales,
)
from think_bridge.training.distributed_protocol import (
    route1_rendezvous_moments_aligned,
    route1_rendezvous_tag,
)
from think_bridge.stage1.methods.bridge.reasoner import (
    FeedbackReasoner,
    materialize_feedback_latent,
)
from think_bridge.model.trajectory import (
    build_answer_branch,
    forward_answer_hidden,
    freeze_executor_parameters,
)
from think_bridge.model.contract import ANSWER_CAPACITY, route1_course_reduction
from think_bridge.stage1.methods.bridge.recipe import phase_for_epoch


def _route1_detached_all_gather_cat(local: torch.Tensor) -> torch.Tensor:
    """Gather a frozen z snapshot without attaching a collective autograd edge."""

    snapshot = local.detach().contiguous()
    if not (torch_distributed.is_available() and torch_distributed.is_initialized()):
        return snapshot.clone().requires_grad_(True)
    world = int(torch_distributed.get_world_size())
    gathered = [torch.empty_like(snapshot) for _ in range(world)]
    torch_distributed.all_gather(gathered, snapshot)
    return torch.cat(gathered, dim=0).requires_grad_(True)


def _route1_reduce_staged_global_z_gradient(
    global_gradient: torch.Tensor, *, local_rows: int
) -> torch.Tensor:
    """Return the once-per-microstep donor gradient owned by this rank."""

    if local_rows <= 0 or int(global_gradient.size(0)) % int(local_rows) != 0:
        raise ValueError("Route1 staged donor-gradient geometry is invalid")
    if not (torch_distributed.is_available() and torch_distributed.is_initialized()):
        if int(global_gradient.size(0)) != int(local_rows):
            raise RuntimeError("single-rank Route1 donor bank is not local")
        return global_gradient.contiguous()
    world = int(torch_distributed.get_world_size())
    rank = int(torch_distributed.get_rank())
    if int(global_gradient.size(0)) != world * int(local_rows):
        raise RuntimeError("distributed Route1 donor bank geometry differs")
    contiguous = global_gradient.contiguous()
    backend = str(torch_distributed.get_backend()).lower()
    if ("nccl" in backend or "xccl" in backend) and hasattr(
        torch_distributed, "reduce_scatter_tensor"
    ):
        local = torch.empty(
            (int(local_rows), *contiguous.shape[1:]),
            dtype=contiguous.dtype,
            device=contiguous.device,
        )
        torch_distributed.reduce_scatter_tensor(
            local, contiguous, op=torch_distributed.ReduceOp.SUM
        )
        return local
    reduced = contiguous.clone()
    torch_distributed.all_reduce(reduced, op=torch_distributed.ReduceOp.SUM)
    return reduced.narrow(0, rank * int(local_rows), int(local_rows)).contiguous()


def _route1_staged_vjp_failure_rendezvous(
    local_error: BaseException | None,
    *,
    device: torch.device,
    label: str = "staged frozen-F replay",
) -> None:
    """Propagate one rank's pre-reduction failure with a scalar hot path.

    This rendezvous runs several times per Route1 microstep.  The ordinary
    no-error path therefore exchanges only one integer flag.  Error text is
    gathered only after every rank has observed that at least one peer failed.
    Keeping diagnostic payloads out of the healthy NCCL stream avoids a large
    number of auxiliary all-gathers interleaved with the optimizer's own
    collectives.
    """

    if not (torch_distributed.is_available() and torch_distributed.is_initialized()):
        if local_error is not None:
            raise local_error
        return
    world = int(torch_distributed.get_world_size())
    tag = route1_rendezvous_tag(label)
    summary = torch.tensor(
        [int(local_error is not None), tag, tag * tag],
        dtype=torch.int64,
        device=device,
    )
    torch_distributed.all_reduce(summary, op=torch_distributed.ReduceOp.SUM)
    failure_count, tag_sum, tag_square_sum = (int(value) for value in summary.tolist())
    phase_aligned = route1_rendezvous_moments_aligned(
        world_size=world,
        tag_sum=tag_sum,
        tag_square_sum=tag_square_sum,
    )
    if failure_count == 0 and phase_aligned:
        return

    message = (
        None
        if local_error is None
        else f"{type(local_error).__name__}: {local_error}"[:2048]
    )
    local_status = {"label": label, "error": message}
    gathered: list[dict[str, str | None] | None] = [None for _ in range(world)]
    torch_distributed.all_gather_object(gathered, local_status)
    statuses = [
        f"rank{rank}={value}"
        for rank, value in enumerate(gathered)
        if value is not None
    ]
    if not phase_aligned:
        raise RuntimeError(
            "Route1 ranks reached different streamed-VJP collective boundaries: "
            + "; ".join(statuses)
        )
    failures = [
        f"rank{rank}={value['error']}"
        for rank, value in enumerate(gathered)
        if value is not None and value["error"] is not None
    ]
    raised = RuntimeError(
        f"Route1 {label} failed before its next collective: " + "; ".join(failures)
    )
    if local_error is not None:
        raise raised from local_error
    raise raised


def _route1_symmetric_staged_vjp_call(
    function: Any,
    *,
    device: torch.device,
    label: str = "staged frozen-F replay",
) -> Any:
    """Run all local F replays, then rendezvous once before donor reduction."""

    result = None
    local_error: BaseException | None = None
    try:
        result = function()
    except BaseException as exc:
        local_error = exc
    _route1_staged_vjp_failure_rendezvous(local_error, device=device, label=label)
    return result


def _route1_accumulate_indexed_gradient(
    destination: torch.Tensor,
    indices: torch.Tensor,
    gradient: torch.Tensor,
) -> None:
    if indices.ndim != 1 or int(indices.numel()) != int(gradient.size(0)):
        raise ValueError("Route1 staged gradient indices do not align")
    destination.index_add_(0, indices.long(), gradient.detach().float())


def _route1_vector_vjp(
    rows: torch.Tensor,
    live_input: torch.Tensor,
    coefficients: torch.Tensor,
    *,
    retain_graph: bool = False,
) -> torch.Tensor:
    """Consume one physical frozen-F graph and return only its input VJP."""

    if rows.ndim != 1 or coefficients.shape != rows.shape:
        raise ValueError("Route1 row VJP coefficients do not align")
    gradient = torch.autograd.grad(
        rows,
        live_input,
        grad_outputs=coefficients.to(dtype=rows.dtype),
        retain_graph=bool(retain_graph),
        create_graph=False,
        allow_unused=False,
    )[0]
    return gradient.detach().float()


def _route1_component_surrogate(
    true_z: torch.Tensor,
    staged_gradient: torch.Tensor,
    detached_value: torch.Tensor,
    *,
    audit_terms: tuple[tuple[torch.Tensor, torch.Tensor], ...] = (),
) -> torch.Tensor:
    """Attach a staged z VJP to R while preserving the detached scalar value."""

    if staged_gradient.shape != true_z.shape:
        raise ValueError("Route1 staged local z gradient geometry differs")
    backward_value = (true_z.float() * staged_gradient.detach().float()).sum()
    for audit_proxy, audit_gradient in audit_terms:
        if audit_proxy.shape != audit_gradient.shape:
            raise ValueError("Route1 staged audit geometry differs")
        backward_value = (
            backward_value
            + (audit_proxy.float() * audit_gradient.detach().float()).sum()
        )
    scalar = detached_value.detach().float().reshape(())
    return backward_value + (scalar - backward_value.detach())


def _route1_rng_state(device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    cpu = torch.random.get_rng_state()
    cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return cpu, cuda


def _route1_restore_rng_state(
    state: tuple[torch.Tensor, torch.Tensor | None], *, device: torch.device
) -> None:
    cpu, cuda = state
    torch.random.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state(cuda, device)


def _route1_replay_with_rng(
    function: Any,
    *,
    replay_state: tuple[torch.Tensor, torch.Tensor | None],
    outer_state: tuple[torch.Tensor, torch.Tensor | None],
    device: torch.device,
) -> Any:
    """Replay one frozen-F chunk without consuming the training RNG twice."""

    _route1_restore_rng_state(replay_state, device=device)
    try:
        return function()
    finally:
        _route1_restore_rng_state(outer_state, device=device)


@dataclass(frozen=True)
class OwnershipState:
    phase: str
    active_owner: str
    trainable_parameter_names: tuple[str, ...]


@dataclass(frozen=True)
class BridgeRoute1Forward:
    loss: torch.Tensor
    loss_answer_course: torch.Tensor
    loss_match: torch.Tensor
    loss_specific: torch.Tensor
    course_active_count: int
    course_c_active_count: int
    course_b_d_side_active_count: int
    match_active_count: int
    specific_active_count: int
    valid_course_tokens: torch.Tensor
    physical_chunk_count: int
    c_view_count: int
    legal_wrong_pair_count: int
    wrong_physical_chunk_count: int
    z: torch.Tensor
    specificity_stats: "Route1SpecificityStats | None" = None
    rollout_request_count: int = 0
    rollout_token_count: int = 0
    rollout_service_batch_rows: int = 0
    rollout_service_real_rows: int = 0
    audit_graph: "Route1ComponentAuditGraph | None" = None

    def detached(self) -> "BridgeRoute1Forward":
        return BridgeRoute1Forward(
            loss=self.loss.detach(),
            loss_answer_course=self.loss_answer_course.detach(),
            loss_match=self.loss_match.detach(),
            loss_specific=self.loss_specific.detach(),
            course_active_count=int(self.course_active_count),
            course_c_active_count=int(self.course_c_active_count),
            course_b_d_side_active_count=int(self.course_b_d_side_active_count),
            match_active_count=int(self.match_active_count),
            specific_active_count=int(self.specific_active_count),
            valid_course_tokens=self.valid_course_tokens.detach(),
            physical_chunk_count=int(self.physical_chunk_count),
            c_view_count=int(self.c_view_count),
            legal_wrong_pair_count=int(self.legal_wrong_pair_count),
            wrong_physical_chunk_count=int(self.wrong_physical_chunk_count),
            specificity_stats=(
                None
                if self.specificity_stats is None
                else self.specificity_stats.detached()
            ),
            rollout_request_count=int(self.rollout_request_count),
            rollout_token_count=int(self.rollout_token_count),
            rollout_service_batch_rows=int(self.rollout_service_batch_rows),
            rollout_service_real_rows=int(self.rollout_service_real_rows),
            z=self.z.detach(),
            audit_graph=None,
        )


@dataclass(frozen=True)
class Route1ComponentAuditGraph:
    """Optional audit views and full-R surrogates; never retained in normal runs."""

    course_z: tuple[torch.Tensor, ...]
    match_owner_z: tuple[torch.Tensor, ...]
    specific_owner_z: tuple[torch.Tensor, ...]
    specific_donor_z: tuple[torch.Tensor, ...]
    reference_tensors_frozen: bool
    # With the optimizer-window bank these share the *actual* R graphs used
    # by the forward, including dropout. They are gradient surrogates, not two
    # separately meaningful loss values. The donor term already has its sign.
    specific_owner_loss: torch.Tensor | None = None
    specific_donor_loss: torch.Tensor | None = None


@dataclass(frozen=True)
class Route1SpecificityStats:
    """Coverage and eligible-owner means, normalized over the optimizer window."""

    owner_count: int
    covered_owner_count: int
    pair_count: int
    true_kl: torch.Tensor | None = None
    no_z_kl: torch.Tensor | None = None
    wrong_kl: torch.Tensor | None = None
    margin_active_fraction: torch.Tensor | None = None
    soft_weight_mean: torch.Tensor | None = None
    negative_cap_fraction: torch.Tensor | None = None
    effective_wrong_kl: torch.Tensor | None = None

    def detached(self) -> "Route1SpecificityStats":
        return self


class _Route1ForwardTimer:
    """Collect rank-local CUDA stage timings with one synchronization."""

    def __init__(
        self,
        device: torch.device,
        sink: dict[str, float] | None,
        progress_sink: Any | None,
    ) -> None:
        self.device = device
        self.sink = sink
        self.progress_sink = progress_sink
        self.events: list[tuple[str, Any, Any]] = []

    def call(self, field: str, function: Any, *, stage: str) -> Any:
        if self.progress_sink is not None:
            self.progress_sink(stage)
        if self.sink is None:
            return function()
        if self.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            value = function()
            stop.record()
            self.events.append((field, start, stop))
            return value
        started = time.perf_counter()
        value = function()
        self.sink[field] = self.sink.get(field, 0.0) + (time.perf_counter() - started)
        return value

    def finish(self) -> None:
        if self.sink is None or not self.events:
            return
        self.events[-1][2].synchronize()
        for field, start, stop in self.events:
            self.sink[field] = self.sink.get(field, 0.0) + (
                float(start.elapsed_time(stop)) / 1000.0
            )


def _set_module_owner(module: nn.Module, enabled: bool, owner: str) -> tuple[str, ...]:
    names: list[str] = []
    for name, parameter in module.named_parameters():
        if enabled and parameter.dtype != torch.float32:
            raise TypeError(
                f"{owner} trainable parameter is not FP32: {name}={parameter.dtype}"
            )
        parameter.requires_grad_(enabled)
        if enabled:
            names.append(f"{owner}.{name}")
    return tuple(names)


def set_bridge_phase_ownership(
    executor: nn.Module, reasoner: nn.Module, *, epoch: int, route1_epochs: int = 2
) -> OwnershipState:
    phase = phase_for_epoch(int(epoch), route1_epochs=int(route1_epochs))
    freeze_executor_parameters(executor)
    trainable = _set_module_owner(reasoner, True, "R")
    if not trainable:
        raise RuntimeError("R has no trainable parameters")
    return OwnershipState(phase.name, "R", trainable)


def build_owner_optimizer(
    reasoner: nn.Module,
    *,
    owner: str = "R",
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    if owner != "R":
        raise ValueError("optimizer owner must be R")
    parameters = [p for p in reasoner.parameters() if p.requires_grad]
    if not parameters or any(p.dtype != torch.float32 for p in parameters):
        raise RuntimeError("R must expose non-empty FP32 trainable parameters")
    return torch.optim.AdamW(
        parameters, lr=float(learning_rate), weight_decay=float(weight_decay)
    )


class BridgeParallelModel(nn.Module):
    """R owns Route1; F and its lexical parameters remain frozen."""

    def __init__(
        self,
        *,
        executor: nn.Module,
        reasoner: FeedbackReasoner,
        alignment_capacity: int,
        boundary_ids: torch.Tensor,
        eos_token_id: int,
        method: str = "bridge",
        vocab_chunk_size: int = 8192,
        trajectory_max_steps: int = ANSWER_CAPACITY,
        route1_gradient_checkpointing: bool = False,
        route1_specificity_wrong_gradient: str = "live",
        route1_specificity_include_direct: bool = False,
        route1_specificity_loss: str = "same-prompt-capped-soft-infonce",
        route1_specificity_margin: float = 0.1,
        route1_specificity_temperature: float = 0.1,
        route1_specificity_negative_kl_cap: float | None = 0.2,
        route1_generation_temperature: float = 0.0,
        generation_seed: int = 42,
    ) -> None:
        super().__init__()
        if method != "bridge":
            raise ValueError("unsupported staged-feedback method")
        if int(alignment_capacity) <= 0:
            raise ValueError("alignment capacity must be positive")
        self.executor = executor
        self.reasoner = reasoner.float()
        self.alignment_capacity = int(alignment_capacity)
        if isinstance(eos_token_id, bool) or int(eos_token_id) < 0:
            raise ValueError("EOS token id must be a non-negative integer")
        self.eos_token_id = int(eos_token_id)
        self.method = method
        if route1_specificity_wrong_gradient != "live":
            raise ValueError("specificity_wrong_gradient must be live")
        self.route1_specificity_wrong_gradient = route1_specificity_wrong_gradient
        if route1_specificity_include_direct is not False:
            raise ValueError("specificity_include_direct must be false")
        self.route1_specificity_include_direct = route1_specificity_include_direct
        if route1_specificity_loss != "same-prompt-capped-soft-infonce":
            raise ValueError("unsupported specificity loss")
        if (
            not math.isfinite(float(route1_specificity_margin))
            or route1_specificity_margin < 0
        ):
            raise ValueError("specificity_margin must be finite and nonnegative")
        self.route1_specificity_loss = route1_specificity_loss
        self.route1_specificity_margin = float(route1_specificity_margin)
        if (
            not math.isfinite(float(route1_specificity_temperature))
            or route1_specificity_temperature <= 0
        ):
            raise ValueError("specificity_temperature must be finite and positive")
        self.route1_specificity_temperature = float(route1_specificity_temperature)
        if route1_specificity_negative_kl_cap is not None:
            if (
                not math.isfinite(float(route1_specificity_negative_kl_cap))
                or route1_specificity_negative_kl_cap <= 0
            ):
                raise ValueError(
                    "specificity_negative_kl_cap must be finite and positive"
                )
        if (
            route1_specificity_loss
            in {
                "capped-soft-infonce",
                "normalized-capped-soft-infonce",
                "same-prompt-capped-soft-infonce",
            }
            and route1_specificity_negative_kl_cap is None
        ):
            raise ValueError("capped soft InfoNCE requires specificity_negative_kl_cap")
        self.route1_specificity_negative_kl_cap = route1_specificity_negative_kl_cap
        # The active hidden-state KL and answer-CE paths are bounded by

        # configuration field.  Keep one inert internal placeholder so old
        # call signatures remain reconstructable without turning it into an
        # admission or resume gate.
        self.vocab_chunk_size = 1
        if int(trajectory_max_steps) != ANSWER_CAPACITY:
            raise ValueError("Bridge native trajectory must use the 2048-token horizon")
        self.trajectory_max_steps = int(trajectory_max_steps)
        self.route1_generation_temperature = float(route1_generation_temperature)
        self.generation_seed = int(generation_seed)
        if (
            not math.isfinite(self.route1_generation_temperature)
            or self.route1_generation_temperature < 0
        ):
            raise ValueError(
                "Route1 generation temperature must be finite and nonnegative"
            )
        self.route1_gradient_checkpointing = bool(route1_gradient_checkpointing)
        self.register_buffer(
            "boundary_ids", boundary_ids.detach().long().clone(), persistent=True
        )
        embedding = self.executor.get_input_embeddings()
        embedding_rms = embedding.weight.detach().float().square().mean().sqrt()
        self.register_buffer(
            "embedding_rms", embedding_rms.reshape(()), persistent=True
        )
        freeze_executor_parameters(self.executor)
        self.executor.eval()
        self.reasoner._gradient_checkpointing = self.route1_gradient_checkpointing

    @classmethod
    def build_reasoner(
        cls,
        executor: nn.Module,
        *,
        latent_steps: int = 1,
        latents_per_step: int = 64,
        loop_steps: int = 2,
        num_layers: int = 2,
        num_heads: int = 16,
        dim_feedforward: int = 4096,
        bound_scale_init: float = 2.0,
        zero_residual_init: bool = False,
        output_normalization: str = "residual",
        input_mode: str = "last-query-reader-self-loop",
        dropout_p: float = 0.0,
        dropout_views: int = 1,
    ) -> FeedbackReasoner:
        width = int(executor.config.hidden_size)
        return FeedbackReasoner(
            width,
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
        ).float()

    def train(self, mode: bool = True) -> "BridgeParallelModel":
        super().train(mode)
        self.executor.eval()
        return self

    def reason(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        *,
        reader_context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from think_bridge.model.feedback_precision import feedback_compute_context

        with feedback_compute_context(self, prompt_ids.device):
            z = materialize_feedback_latent(
                self.executor,
                self.reasoner,
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask,
                embedding_rms=self.embedding_rms,
                reader_context_mask=reader_context_mask,
            ).z
        if z.dtype != torch.float32 or z.shape[1] != self.reasoner.num_slots:
            raise RuntimeError(
                "Bridge reasoner output must be FP32 z with the configured slot count"
            )
        return z

    def reason_views(
        self, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Materialize configured stochastic R views during training only."""
        requested = int(getattr(self.reasoner, "dropout_views", 1))
        count = requested if self.training and requested > 1 else 1
        return tuple(self.reason(prompt_ids, prompt_mask) for _ in range(count))

    def _branch_hidden(
        self,
        *,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        answer_ids: torch.Tensor,
        answer_mask: torch.Tensor,
        geometry: str,
        live: bool,
        checkpoint_backbone: bool = False,
        z: torch.Tensor | None = None,
        cot_ids: torch.Tensor | None = None,
        cot_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Any]:
        batch = build_answer_branch(
            embedding=self.executor.get_input_embeddings(),
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            boundary_ids=self.boundary_ids,
            answer_ids=answer_ids,
            answer_mask=answer_mask,
            geometry=geometry,
            z=z,
            cot_ids=cot_ids,
            cot_mask=cot_mask,
        )
        hidden = forward_answer_hidden(
            self.executor,
            batch,
            live=live,
            gradient_checkpointing=bool(checkpoint_backbone and live),
        )
        return hidden, batch

    def _route1_nll_rows(
        self,
        *,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        answer_ids: torch.Tensor,
        answer_mask: torch.Tensor,
        owner_indices: torch.Tensor,
        z: torch.Tensor | None,
        donor_indices: torch.Tensor | None,
        geometry: str,
        physical_chunk_size: int,
        cot_ids: torch.Tensor | None = None,
        cot_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        """Evaluate paired answer NLL rows in bounded frozen-F chunks.

        Under the 4B checkpoint policy the whole live chunk, including the
        large-vocabulary CE projection, is recomputed during backward.  The
        callable binds every chunk tensor at definition time so a later loop
        iteration cannot change the non-reentrant replay inputs.
        """

        if owner_indices.ndim != 1:
            raise ValueError("Route1 owner indices must be one vector")
        if donor_indices is not None and donor_indices.shape != owner_indices.shape:
            raise ValueError("Route1 owner/donor indices must align")
        if physical_chunk_size <= 0:
            raise ValueError("Route1 physical chunk size must be positive")
        if owner_indices.numel() == 0:
            owner = z if z is not None else next(self.reasoner.parameters())
            return (
                owner.sum().mul(0.0).expand((0,)),
                0,
                owner.new_zeros((), dtype=torch.long),
            )
        head = getattr(self.executor, "lm_head", None)
        if head is None or not hasattr(head, "weight"):
            raise TypeError("frozen executor lacks an LM head")
        rows: list[torch.Tensor] = []
        chunks = 0
        valid_tokens = torch.zeros((), dtype=torch.long, device=answer_mask.device)
        for start in range(0, int(owner_indices.numel()), int(physical_chunk_size)):
            stop = min(start + int(physical_chunk_size), int(owner_indices.numel()))
            owners = owner_indices[start:stop].long()
            condition = None
            if geometry in {"deployed_z", "deployed_z_cot_suffix"}:
                if z is None:
                    raise ValueError("deployed Route1 branch lacks z")
                sources = (
                    owners
                    if donor_indices is None
                    else donor_indices[start:stop].long()
                )
                condition = z.index_select(0, sources)

            selected_prompt_ids = prompt_ids.index_select(0, owners)
            selected_prompt_mask = prompt_mask.index_select(0, owners)
            selected_answer_ids = answer_ids.index_select(0, owners)
            selected_answer_mask = answer_mask.index_select(0, owners)
            selected_cot_ids = (
                None if cot_ids is None else cot_ids.index_select(0, owners)
            )
            selected_cot_mask = (
                None if cot_mask is None else cot_mask.index_select(0, owners)
            )

            def chunk_nll(
                selected_z: torch.Tensor | None,
                *,
                chunk_prompt_ids: torch.Tensor = selected_prompt_ids,
                chunk_prompt_mask: torch.Tensor = selected_prompt_mask,
                chunk_answer_ids: torch.Tensor = selected_answer_ids,
                chunk_answer_mask: torch.Tensor = selected_answer_mask,
                chunk_cot_ids: torch.Tensor | None = selected_cot_ids,
                chunk_cot_mask: torch.Tensor | None = selected_cot_mask,
            ) -> torch.Tensor:
                hidden, branch = self._branch_hidden(
                    prompt_ids=chunk_prompt_ids,
                    prompt_mask=chunk_prompt_mask,
                    answer_ids=chunk_answer_ids,
                    answer_mask=chunk_answer_mask,
                    geometry=geometry,
                    live=(geometry in {"deployed_z", "deployed_z_cot_suffix"}),
                    z=selected_z,
                    cot_ids=chunk_cot_ids,
                    cot_mask=chunk_cot_mask,
                )
                return _per_occurrence_gold_ce_rows_from_hidden_validated(
                    hidden,
                    branch.target_ids,
                    branch.target_mask,
                    branch.target_mask.long().sum(dim=1),
                    lm_head_weight=head.weight,
                    lm_head_bias=getattr(head, "bias", None),
                )

            if condition is not None and self.route1_gradient_checkpointing:
                chunk_rows = activation_checkpoint(
                    chunk_nll,
                    condition,
                    use_reentrant=False,
                )
            else:
                chunk_rows = chunk_nll(condition)
            rows.append(chunk_rows)
            chunks += 1
            with torch.no_grad():
                valid_tokens = valid_tokens + selected_answer_mask.sum()
                if (
                    geometry == "deployed_z_cot_suffix"
                    and selected_cot_mask is not None
                ):
                    valid_tokens = valid_tokens + selected_cot_mask.sum()
        return torch.cat(rows), chunks, valid_tokens.detach()

    def _route1_c_view_rows(
        self,
        *,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        direct_prompt_ids: torch.Tensor,
        direct_prompt_mask: torch.Tensor,
        native_cot_ids: torch.Tensor,
        native_cot_mask: torch.Tensor,
        view_to_sample: torch.Tensor,
        true_z: torch.Tensor,
        differentiable_global_z_bank: torch.Tensor,
        wrong_candidate_mask: torch.Tensor,
        global_match_sample_count: int,
        global_specific_sample_count: int,
        distill_row_weights: torch.Tensor,
        distill_cot_ids: torch.Tensor,
        distill_cot_mask: torch.Tensor,
        physical_chunk_size: int,
        wrong_control_chunk_size: int,
        specificity_tau: float,
        component_gradient_audit: bool,
        timer: _Route1ForwardTimer | None,
        specificity_step_audit: bool = False,
        generated_prefix: Any | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        int,
        tuple[
            tuple[torch.Tensor, ...],
            tuple[torch.Tensor, ...],
            tuple[torch.Tensor, ...],
            bool,
        ],
        dict[str, int],
        Route1SpecificityStats,
    ]:
        """Answer FKL plus live-donor soft multi-positive specificity."""
        from think_bridge.model.specificity import c_components

        return c_components(
            self,
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            direct_prompt_ids=direct_prompt_ids,
            direct_prompt_mask=direct_prompt_mask,
            specificity_tau=specificity_tau,
            native_cot_ids=native_cot_ids,
            native_cot_mask=native_cot_mask,
            view_to_sample=view_to_sample,
            true_z=true_z,
            global_z=differentiable_global_z_bank,
            wrong_candidate_mask=wrong_candidate_mask,
            global_match_sample_count=global_match_sample_count,
            global_specific_sample_count=global_specific_sample_count,
            distill_row_weights=distill_row_weights,
            distill_cot_ids=distill_cot_ids,
            distill_cot_mask=distill_cot_mask,
            physical_chunk_size=physical_chunk_size,
            wrong_control_chunk_size=wrong_control_chunk_size,
            generated_prefix=generated_prefix,
            timer=timer,
            staged=False,
            collect_diagnostics=specificity_step_audit,
        )

    def _route1_streamed_course_component(
        self,
        *,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        course_cot_ids: torch.Tensor,
        course_cot_mask: torch.Tensor,
        course_answer_ids: torch.Tensor,
        course_answer_mask: torch.Tensor,
        course_view_to_sample: torch.Tensor,
        course_view_to_course_sample: torch.Tensor,
        course_sample_side: torch.Tensor,
        true_z_proxy: torch.Tensor,
        answer_course_geometry: str,
        course_reduction: str,
        global_course_c_sample_count: int,
        global_course_direct_side_sample_count: int,
        physical_chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """Stage course z VJPs one frozen-F physical chunk at a time."""

        world = (
            int(torch_distributed.get_world_size())
            if torch_distributed.is_available() and torch_distributed.is_initialized()
            else 1
        )
        scales = route1_course_reduction_scales(
            reduction=course_reduction,
            global_c_sample_count=int(global_course_c_sample_count),
            global_direct_side_sample_count=int(global_course_direct_side_sample_count),
            world_size=world,
        )
        staged = torch.zeros_like(true_z_proxy, dtype=torch.float32)
        detached_value = true_z_proxy.new_zeros((), dtype=torch.float32)
        valid_tokens = true_z_proxy.new_zeros((), dtype=torch.long)
        chunks = 0
        view_count = int(course_view_to_sample.numel())
        for start in range(0, view_count, int(physical_chunk_size)):
            stop = min(start + int(physical_chunk_size), view_count)
            view_indices = torch.arange(
                start, stop, dtype=torch.long, device=prompt_ids.device
            )
            sample_indices = course_view_to_sample.index_select(0, view_indices).long()
            selected_z = true_z_proxy.index_select(0, sample_indices)
            rows, chunk_count, chunk_tokens = self._route1_nll_rows(
                prompt_ids=prompt_ids.index_select(0, sample_indices),
                prompt_mask=prompt_mask.index_select(0, sample_indices),
                answer_ids=course_answer_ids.index_select(0, view_indices),
                answer_mask=course_answer_mask.index_select(0, view_indices),
                owner_indices=torch.arange(
                    stop - start, dtype=torch.long, device=prompt_ids.device
                ),
                z=selected_z,
                donor_indices=None,
                geometry=answer_course_geometry,
                physical_chunk_size=stop - start,
                cot_ids=(
                    course_cot_ids.index_select(0, view_indices)
                    if answer_course_geometry == "deployed_z_cot_suffix"
                    else None
                ),
                cot_mask=(
                    course_cot_mask.index_select(0, view_indices)
                    if answer_course_geometry == "deployed_z_cot_suffix"
                    else None
                ),
            )
            course_indices = course_view_to_course_sample.index_select(
                0, view_indices
            ).long()
            sides = course_sample_side.index_select(0, course_indices).long()
            coefficients = torch.where(
                sides.eq(0),
                rows.new_full(rows.shape, float(scales.c_local_sum_scale)),
                rows.new_full(rows.shape, float(scales.direct_side_local_sum_scale)),
            )
            selected_gradient = _route1_vector_vjp(rows, selected_z, coefficients)
            _route1_accumulate_indexed_gradient(
                staged, sample_indices, selected_gradient
            )
            detached_value = (
                detached_value
                + (rows.detach().float() * coefficients.detach().float()).sum()
            )
            valid_tokens = valid_tokens + chunk_tokens.detach().long()
            chunks += int(chunk_count)
            del rows, selected_gradient, selected_z
        return staged, detached_value, chunks, valid_tokens.detach()

    def _route1_streamed_c_components(
        self,
        *,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        direct_prompt_ids: torch.Tensor,
        direct_prompt_mask: torch.Tensor,
        native_cot_ids: torch.Tensor,
        native_cot_mask: torch.Tensor,
        view_to_sample: torch.Tensor,
        true_z_proxy: torch.Tensor,
        global_z_proxy: torch.Tensor,
        wrong_candidate_mask: torch.Tensor,
        true_z_views: tuple[torch.Tensor, ...] | None = None,
        specificity_positive_mask: torch.Tensor | None = None,
        specificity_donor_prompt_ids: torch.Tensor | None = None,
        specificity_donor_prompt_mask: torch.Tensor | None = None,
        global_match_sample_count: int,
        global_specific_sample_count: int,
        distill_row_weights: torch.Tensor,
        distill_cot_ids: torch.Tensor,
        distill_cot_mask: torch.Tensor,
        physical_chunk_size: int,
        wrong_control_chunk_size: int,
        specificity_tau: float,
        specificity_step_audit: bool,
        timer: _Route1ForwardTimer,
        generated_prefix: Any | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        dict[str, int],
        Route1SpecificityStats,
        bool,
        torch.Tensor | None,
    ]:
        """Answer FKL plus live-donor soft multi-positive specificity."""
        from think_bridge.model.specificity import c_components

        return c_components(
            self,
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            direct_prompt_ids=direct_prompt_ids,
            direct_prompt_mask=direct_prompt_mask,
            specificity_tau=specificity_tau,
            native_cot_ids=native_cot_ids,
            native_cot_mask=native_cot_mask,
            view_to_sample=view_to_sample,
            true_z=true_z_proxy,
            global_z=global_z_proxy,
            true_z_views=true_z_views,
            wrong_candidate_mask=wrong_candidate_mask,
            specificity_positive_mask=specificity_positive_mask,
            specificity_donor_prompt_ids=specificity_donor_prompt_ids,
            specificity_donor_prompt_mask=specificity_donor_prompt_mask,
            global_match_sample_count=global_match_sample_count,
            global_specific_sample_count=global_specific_sample_count,
            distill_row_weights=distill_row_weights,
            distill_cot_ids=distill_cot_ids,
            distill_cot_mask=distill_cot_mask,
            physical_chunk_size=physical_chunk_size,
            wrong_control_chunk_size=wrong_control_chunk_size,
            generated_prefix=generated_prefix,
            timer=timer,
            staged=True,
            collect_diagnostics=specificity_step_audit,
        )

    def _forward_route1_bridge_streamed_from_z(
        self,
        *,
        true_z: torch.Tensor,
        timer: _Route1ForwardTimer,
        true_z_views: tuple[torch.Tensor, ...] | None,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        direct_prompt_ids: torch.Tensor,
        direct_prompt_mask: torch.Tensor,
        course_cot_ids: torch.Tensor,
        course_cot_mask: torch.Tensor,
        course_answer_ids: torch.Tensor,
        course_answer_mask: torch.Tensor,
        course_view_to_sample: torch.Tensor,
        course_view_to_course_sample: torch.Tensor,
        course_sample_side: torch.Tensor,
        c_native_cot_ids: torch.Tensor,
        c_native_cot_mask: torch.Tensor,
        c_view_to_sample: torch.Tensor,
        wrong_candidate_mask: torch.Tensor,
        specificity_positive_mask: torch.Tensor | None = None,
        specificity_donor_prompt_ids: torch.Tensor | None = None,
        specificity_donor_prompt_mask: torch.Tensor | None = None,
        answer_course_geometry: str,
        course_reduction: str,
        global_course_c_sample_count: int,
        global_course_direct_side_sample_count: int,
        global_match_sample_count: int,
        global_specific_sample_count: int,
        distill_row_weights: torch.Tensor,
        distill_cot_ids: torch.Tensor,
        distill_cot_mask: torch.Tensor,
        course_weight: float,
        match_weight: float,
        specific_weight: float,
        specificity_tau: float,
        physical_chunk_size: int,
        wrong_control_chunk_size: int,
        component_gradient_audit: bool,
        specificity_step_audit: bool,
        timing_sink: dict[str, float] | None,
        free_generation_provider: Any | None,
        rollout_update: int,
        rollout_micro_step: int,
        rollout_sample_keys: tuple[str, ...],
        prepared_state: dict[str, Any] | None = None,
    ) -> BridgeRoute1Forward:
        """Bound F activation lifetime and traverse the shared R graph once."""

        if specificity_donor_prompt_ids is None:
            specificity_donor_prompt_ids = prompt_ids.new_empty((0, 1))
            specificity_donor_prompt_mask = prompt_mask.new_empty((0, 1))
        elif specificity_donor_prompt_mask is None:
            raise ValueError(
                "specificity donor prompt ids/mask must be supplied together"
            )

        if true_z_views is None:
            true_z_views = (true_z,)
        true_z_views = tuple(true_z_views)
        if not true_z_views or true_z_views[0] is not true_z:
            raise ValueError("Route1 true_z_views must start with true_z")
        if any(view.shape != true_z.shape for view in true_z_views):
            raise ValueError("Route1 true_z views must share [B,K,D] geometry")
        true_z_proxies = tuple(
            view.detach().requires_grad_(True) for view in true_z_views
        )
        true_z_proxy = true_z_proxies[0]
        compute_course = float(course_weight) > 0.0
        compute_specific = int(global_specific_sample_count) > 0
        if compute_course != bool(
            int(global_course_c_sample_count)
            + int(global_course_direct_side_sample_count)
        ):
            raise ValueError("Route1 course weight/count activation differs")
        if not compute_course and int(course_view_to_sample.numel()) != 0:
            raise ValueError("zero-weight Route1 course received physical views")
        use_window_donor_bank = bool(
            compute_specific and specificity_donor_prompt_ids.numel() > 0
        )
        global_z_proxy = (
            _route1_detached_all_gather_cat(true_z_proxy)
            if compute_specific and not use_window_donor_bank
            else true_z_proxy.new_empty((0, *true_z_proxy.shape[1:])).requires_grad_(
                True
            )
        )
        donor_width = int(
            specificity_donor_prompt_ids.size(0)
            if use_window_donor_bank
            else global_z_proxy.size(0)
        )
        if compute_specific and wrong_candidate_mask.shape != (
            int(prompt_ids.size(0)),
            donor_width,
        ):
            raise ValueError("Route1 sealed global wrong candidate mask differs")

        generated_prefix = None
        rollout_ticket = None
        rollout_submit_error: BaseException | None = None
        rollout_active = bool(
            free_generation_provider is not None
            and (
                int(global_match_sample_count) > 0
                or int(global_specific_sample_count) > 0
            )
        )
        if prepared_state is not None:
            rollout_ticket = prepared_state["ticket"]
            rollout_submit_error = prepared_state["error"]
        elif rollout_active:
            try:
                rollout_ticket = free_generation_provider.submit(
                    update=int(rollout_update),
                    micro_step=int(rollout_micro_step),
                    sample_keys=tuple(rollout_sample_keys),
                    prompt_ids=prompt_ids,
                    prompt_mask=prompt_mask,
                    true_z=true_z.detach(),
                    view_to_sample=c_view_to_sample,
                    cot_ids=distill_cot_ids,
                    cot_mask=distill_cot_mask,
                )
            except BaseException as exc:
                rollout_submit_error = exc

        course_result = None
        course_error: BaseException | None = None
        if compute_course:
            try:
                course_results = []
                for view_proxy in true_z_proxies:
                    course_results.append(
                        timer.call(
                            "answer_course_seconds",
                            lambda view_proxy=view_proxy: (
                                self._route1_streamed_course_component(
                                    prompt_ids=prompt_ids,
                                    prompt_mask=prompt_mask,
                                    course_cot_ids=course_cot_ids,
                                    course_cot_mask=course_cot_mask,
                                    course_answer_ids=course_answer_ids,
                                    course_answer_mask=course_answer_mask,
                                    course_view_to_sample=course_view_to_sample,
                                    course_view_to_course_sample=course_view_to_course_sample,
                                    course_sample_side=course_sample_side,
                                    true_z_proxy=view_proxy,
                                    answer_course_geometry=answer_course_geometry,
                                    course_reduction=course_reduction,
                                    global_course_c_sample_count=int(
                                        global_course_c_sample_count
                                    ),
                                    global_course_direct_side_sample_count=int(
                                        global_course_direct_side_sample_count
                                    ),
                                    physical_chunk_size=int(physical_chunk_size),
                                )
                            ),
                            stage="ce/streamed-vjp",
                        )
                    )
                course_result = (
                    sum(
                        (item[0] for item in course_results[0:]),
                        true_z_proxy.new_zeros(true_z_proxy.shape),
                    )
                    / float(len(course_results)),
                    sum(
                        (item[1] for item in course_results), true_z_proxy.new_zeros(())
                    )
                    / float(len(course_results)),
                    sum(item[2] for item in course_results),
                    course_results[0][3],
                )
                course_gradient_views = tuple(item[0] for item in course_results)
            except BaseException as exc:
                course_error = exc
                course_gradient_views = (true_z_proxy.new_zeros(true_z_proxy.shape),)
        else:
            course_gradient_views = (true_z_proxy.new_zeros(true_z_proxy.shape),)
            course_result = (
                torch.zeros_like(true_z_proxy, dtype=torch.float32),
                true_z_proxy.new_zeros((), dtype=torch.float32),
                0,
                true_z_proxy.new_zeros((), dtype=torch.long),
            )

        generated_prefix = None
        if rollout_active:
            generated_prefix = free_generation_provider.resolve(
                rollout_ticket,
                device=prompt_ids.device,
                local_error=(
                    course_error if course_error is not None else rollout_submit_error
                ),
                view_count=int(c_view_to_sample.numel()),
            )
            if prepared_state is not None:
                prepared_state["resolved"] = True
            telemetry = generated_prefix.telemetry
            if timing_sink is not None:
                for field, value in {
                    "rollout_submit_seconds": telemetry.submit_seconds,
                    "rollout_queue_seconds": telemetry.queue_seconds,
                    "rollout_service_seconds": telemetry.service_seconds,
                    "rollout_generation_seconds": telemetry.generation_seconds,
                    "rollout_overlap_seconds": telemetry.overlap_seconds,
                    "rollout_wait_seconds": telemetry.wait_seconds,
                    "trainer_overlap_efficiency": telemetry.overlap_efficiency,
                    "service_batch_occupancy": telemetry.service_occupancy,
                    "need_objectives_rollout_seconds": telemetry.service_seconds,
                }.items():
                    timing_sink[field] = timing_sink.get(field, 0.0) + float(value)
        if compute_course:
            _route1_staged_vjp_failure_rendezvous(
                course_error,
                device=true_z_proxy.device,
                label=(
                    f"update={int(rollout_update)} "
                    f"microstep={int(rollout_micro_step)} "
                    "streamed answer-course replay"
                ),
            )
        if course_result is None:
            raise RuntimeError("Route1 course result is missing after rendezvous")
        (
            course_gradient,
            course_value,
            course_chunks,
            valid_course_tokens,
        ) = course_result

        c_result = _route1_symmetric_staged_vjp_call(
            lambda: timer.call(
                "need_objectives_seconds",
                lambda: self._route1_streamed_c_components(
                    prompt_ids=prompt_ids,
                    prompt_mask=prompt_mask,
                    direct_prompt_ids=direct_prompt_ids,
                    direct_prompt_mask=direct_prompt_mask,
                    native_cot_ids=c_native_cot_ids,
                    native_cot_mask=c_native_cot_mask,
                    view_to_sample=c_view_to_sample,
                    true_z_proxy=true_z_proxy,
                    global_z_proxy=global_z_proxy,
                    true_z_views=true_z_proxies,
                    wrong_candidate_mask=wrong_candidate_mask,
                    specificity_positive_mask=specificity_positive_mask,
                    specificity_donor_prompt_ids=specificity_donor_prompt_ids,
                    specificity_donor_prompt_mask=specificity_donor_prompt_mask,
                    global_match_sample_count=int(global_match_sample_count),
                    global_specific_sample_count=int(global_specific_sample_count),
                    distill_row_weights=distill_row_weights,
                    distill_cot_ids=distill_cot_ids,
                    distill_cot_mask=distill_cot_mask,
                    physical_chunk_size=int(physical_chunk_size),
                    wrong_control_chunk_size=int(wrong_control_chunk_size),
                    specificity_tau=float(specificity_tau),
                    specificity_step_audit=bool(specificity_step_audit),
                    timer=timer,
                    generated_prefix=generated_prefix,
                ),
                stage="need-objectives/streamed-vjp",
            ),
            device=true_z_proxy.device,
            label=(
                f"update={int(rollout_update)} "
                f"microstep={int(rollout_micro_step)} streamed B/C replay/VJP"
            ),
        )
        (
            match_gradient,
            specific_owner_gradient,
            specific_global_gradient,
            match_value,
            specific_value,
            c_chunks,
            objective_counts,
            specificity_stats,
            reference_tensors_frozen,
            specificity_donor_surrogate,
        ) = c_result
        specific_donor_gradient = (
            _route1_reduce_staged_global_z_gradient(
                specific_global_gradient,
                local_rows=int(true_z_proxy.size(0)),
            )
            if compute_specific and not use_window_donor_bank
            else torch.zeros_like(true_z_proxy, dtype=torch.float32)
        )
        if specific_owner_gradient.ndim == 4:
            specific_gradient = (
                specific_owner_gradient + specific_donor_gradient.unsqueeze(0)
            )
        else:
            specific_gradient = specific_owner_gradient + specific_donor_gradient

        owner_audit_proxy = (
            true_z.detach().requires_grad_(True) if component_gradient_audit else None
        )

        def _view_surrogate(gradients, value, *, audit_proxy=None):
            if gradients.ndim == 3:
                return _route1_component_surrogate(
                    true_z,
                    gradients,
                    value,
                    audit_terms=(
                        ((audit_proxy, gradients),) if audit_proxy is not None else ()
                    ),
                )
            views = tuple(true_z_views)
            if gradients.ndim != 4 or gradients.size(0) != len(views):
                raise RuntimeError("Route1 multi-view gradient geometry is invalid")
            result = true_z.new_zeros((), dtype=torch.float32)
            for index, view in enumerate(views):
                result = result + _route1_component_surrogate(
                    view,
                    gradients[index],
                    value / float(len(views)),
                )
            return result

        loss_answer_course = _view_surrogate(
            torch.stack(course_gradient_views, dim=0)
            / float(len(course_gradient_views))
            if len(course_gradient_views) > 1
            else course_gradient_views[0],
            course_value,
        )

        loss_match = _view_surrogate(
            match_gradient, match_value, audit_proxy=owner_audit_proxy
        )
        loss_specific = _view_surrogate(
            specific_gradient, specific_value, audit_proxy=owner_audit_proxy
        )
        if isinstance(specificity_donor_surrogate, tuple):
            positive_surrogate, specificity_donor_surrogate = (
                specificity_donor_surrogate
            )
            loss_specific = loss_specific + positive_surrogate
        specific_owner_loss = (
            loss_specific
            if component_gradient_audit and use_window_donor_bank
            else None
        )
        specific_donor_loss = (
            (
                specificity_donor_surrogate
                if specificity_donor_surrogate is not None
                else true_z.sum() * 0.0
            )
            if component_gradient_audit and use_window_donor_bank
            else None
        )
        if specificity_donor_surrogate is not None:
            loss_specific = loss_specific + specificity_donor_surrogate
        total_loss = (
            float(course_weight) * loss_answer_course
            + float(match_weight) * loss_match
            + float(specific_weight) * loss_specific
        )
        course_active_count = int(global_course_c_sample_count) + int(
            global_course_direct_side_sample_count
        )
        result = BridgeRoute1Forward(
            loss=total_loss,
            loss_answer_course=loss_answer_course,
            loss_match=loss_match,
            loss_specific=loss_specific,
            course_active_count=course_active_count,
            course_c_active_count=int(global_course_c_sample_count),
            course_b_d_side_active_count=int(global_course_direct_side_sample_count),
            match_active_count=int(global_match_sample_count),
            specific_active_count=int(global_specific_sample_count),
            valid_course_tokens=valid_course_tokens.detach(),
            physical_chunk_count=int(course_chunks + c_chunks),
            c_view_count=int(objective_counts["c_view_count"]),
            legal_wrong_pair_count=int(objective_counts["legal_wrong_pair_count"]),
            wrong_physical_chunk_count=int(
                objective_counts["wrong_physical_chunk_count"]
            ),
            specificity_stats=specificity_stats,
            rollout_request_count=int(
                0
                if generated_prefix is None
                else generated_prefix.telemetry.request_count
            ),
            rollout_token_count=int(
                0
                if generated_prefix is None
                else generated_prefix.telemetry.token_count
            ),
            rollout_service_batch_rows=int(
                0
                if generated_prefix is None
                else generated_prefix.telemetry.service_batch_rows
            ),
            rollout_service_real_rows=int(
                0
                if generated_prefix is None
                else generated_prefix.telemetry.service_real_rows
            ),
            z=true_z,
            audit_graph=(
                Route1ComponentAuditGraph(
                    course_z=(
                        (true_z,) if int(course_view_to_sample.numel()) > 0 else ()
                    ),
                    match_owner_z=(
                        (owner_audit_proxy,)
                        if owner_audit_proxy is not None
                        and int(c_view_to_sample.numel()) > 0
                        and int(global_match_sample_count) > 0
                        else ()
                    ),
                    specific_owner_z=(
                        (owner_audit_proxy,)
                        if owner_audit_proxy is not None
                        and int(c_view_to_sample.numel()) > 0
                        and int(global_specific_sample_count) > 0
                        else ()
                    ),
                    specific_donor_z=(
                        (global_z_proxy,)
                        if compute_specific
                        and int(objective_counts["legal_wrong_pair_count"]) > 0
                        and self.route1_specificity_wrong_gradient == "live"
                        else ()
                    ),
                    reference_tensors_frozen=bool(reference_tensors_frozen),
                    specific_owner_loss=specific_owner_loss,
                    specific_donor_loss=specific_donor_loss,
                )
                if component_gradient_audit
                else None
            ),
        )
        timer.finish()
        return result

    def prepare_route1_microstep(
        self,
        arguments: Mapping[str, Any],
        *,
        provider: Any,
        update: int,
        microstep: int,
        sample_keys: tuple[str, ...],
        timing_sink: Any = None,
        progress_sink: Any = None,
    ) -> dict[str, Any]:
        """Prepare current-parameter z and a detached request on the caller thread."""
        prompt_ids, prompt_mask = arguments["prompt_ids"], arguments["prompt_mask"]
        timer = _Route1ForwardTimer(prompt_ids.device, timing_sink, progress_sink)
        z = _route1_symmetric_staged_vjp_call(
            lambda: timer.call(
                "live_z_seconds",
                lambda: self.reason(prompt_ids, prompt_mask),
                stage="prefetch-live-z",
            ),
            device=prompt_ids.device,
            label=f"update={update} microstep={microstep} prepared reasoner",
        )
        state = {
            "z": z,
            "ticket": None,
            "error": None,
            "update": int(update),
            "microstep": int(microstep),
            "sample_keys": tuple(sample_keys),
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "distill_cot_ids": arguments["distill_cot_ids"],
            "distill_cot_mask": arguments["distill_cot_mask"],
            "provider": provider,
            "consumed": False,
            "resolved": False,
            # CUDA timings are settled by the existing consume-side
            # timer.finish(), never by an extra prepare synchronization.
            "timing_events": tuple(timer.events),
        }
        if provider is not None and (
            arguments["global_match_sample_count"] > 0
            or arguments["global_specific_sample_count"] > 0
        ):
            try:
                state["ticket"] = provider.submit(
                    update=update,
                    micro_step=microstep,
                    sample_keys=sample_keys,
                    prompt_ids=prompt_ids,
                    prompt_mask=prompt_mask,
                    true_z=z.detach(),
                    view_to_sample=arguments["c_view_to_sample"],
                    cot_ids=arguments["distill_cot_ids"],
                    cot_mask=arguments["distill_cot_mask"],
                )
            except BaseException as exc:
                state["error"] = exc
        return state

    def forward_route1_bridge(
        self,
        *,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        direct_prompt_ids: torch.Tensor,
        direct_prompt_mask: torch.Tensor,
        course_cot_ids: torch.Tensor,
        course_cot_mask: torch.Tensor,
        course_answer_ids: torch.Tensor,
        course_answer_mask: torch.Tensor,
        course_view_to_sample: torch.Tensor,
        course_view_to_course_sample: torch.Tensor,
        course_sample_side: torch.Tensor,
        c_native_cot_ids: torch.Tensor,
        c_native_cot_mask: torch.Tensor,
        c_view_to_sample: torch.Tensor,
        c_view_to_c_sample: torch.Tensor,
        c_sample_indices: torch.Tensor,
        wrong_candidate_mask: torch.Tensor,
        specificity_positive_mask: torch.Tensor | None = None,
        specificity_donor_prompt_ids: torch.Tensor | None = None,
        specificity_donor_prompt_mask: torch.Tensor | None = None,
        answer_course_geometry: str,
        course_reduction: str,
        global_course_c_sample_count: int,
        global_course_direct_side_sample_count: int,
        global_match_sample_count: int,
        global_specific_sample_count: int,
        distill_row_weights: torch.Tensor,
        distill_cot_ids: torch.Tensor,
        distill_cot_mask: torch.Tensor,
        course_weight: float,
        match_weight: float,
        specific_weight: float,
        specificity_tau: float,
        physical_chunk_size: int,
        wrong_control_chunk_size: int,
        component_gradient_audit: bool = False,
        specificity_step_audit: bool = False,
        timing_sink: dict[str, float] | None = None,
        progress_sink: Any | None = None,
        free_generation_provider: Any | None = None,
        rollout_update: int = 0,
        rollout_micro_step: int = 0,
        rollout_sample_keys: tuple[str, ...] = (),
        prepared_state: dict[str, Any] | None = None,
    ) -> BridgeRoute1Forward:
        """Run the fixed B/C/D course plus configured native-population Match/Specificity."""

        if specificity_donor_prompt_ids is None:
            specificity_donor_prompt_ids = prompt_ids.new_empty((0, 1))
            specificity_donor_prompt_mask = prompt_mask.new_empty((0, 1))
        elif specificity_donor_prompt_mask is None:
            raise ValueError(
                "specificity donor prompt ids/mask must be supplied together"
            )

        local_sample_count = int(prompt_ids.size(0))
        course_view_count = int(course_view_to_sample.numel())
        local_course_sample_count = int(course_sample_side.numel())
        c_view_count = int(c_view_to_sample.numel())
        if (
            prompt_mask.shape != prompt_ids.shape
            or direct_prompt_mask.shape != direct_prompt_ids.shape
            or direct_prompt_ids.size(0) != local_sample_count
            or course_cot_mask.shape != course_cot_ids.shape
            or course_cot_ids.size(0) != course_view_count
            or course_answer_mask.shape != course_answer_ids.shape
            or course_answer_ids.size(0) != course_view_count
            or course_view_to_sample.ndim != 1
            or course_view_to_course_sample.shape != course_view_to_sample.shape
            or course_sample_side.ndim != 1
            or c_native_cot_mask.shape != c_native_cot_ids.shape
            or c_native_cot_ids.size(0) != c_view_count
            or c_view_to_c_sample.shape != c_view_to_sample.shape
            or c_sample_indices.ndim != 1
            or distill_row_weights.shape != c_view_to_sample.shape
            or distill_cot_ids.ndim != 2
            or distill_cot_mask.shape != distill_cot_ids.shape
            or distill_cot_ids.size(0) != c_view_count
        ):
            raise ValueError("Route1 sample/view batch geometry is invalid")
        if local_sample_count <= 0:
            raise ValueError("Route1 requires at least one logical-batch sample")
        if bool(
            (course_view_to_sample < 0).any()
            or (course_view_to_sample >= local_sample_count).any()
            or (course_view_to_course_sample < 0).any()
            or (course_view_to_course_sample >= local_course_sample_count).any()
            or (c_view_to_sample < 0).any()
            or (c_view_to_sample >= local_sample_count).any()
        ):
            raise ValueError("Route1 view-to-sample mapping is outside the batch")
        if bool(((course_sample_side < 0) | (course_sample_side > 1)).any()):
            raise ValueError("Route1 course sample side must encode C=0 or B/D=1")
        if bool(course_view_count) != bool(local_course_sample_count):
            raise ValueError(
                "Route1 course samples and views must be jointly empty/nonempty"
            )
        expected_course_views = torch.arange(
            local_course_sample_count,
            dtype=course_view_to_course_sample.dtype,
            device=course_view_to_course_sample.device,
        )
        if not torch.equal(
            torch.sort(course_view_to_course_sample.long()).values,
            expected_course_views.long(),
        ):
            raise ValueError(
                "Route1 course population must carry exactly one view per course sample"
            )
        if c_view_count != int(c_sample_indices.numel()):
            raise ValueError(
                "Route1 B/C native population must carry exactly one view per C sample"
            )
        expected_c_views = torch.arange(
            int(c_sample_indices.numel()),
            dtype=c_view_to_c_sample.dtype,
            device=c_view_to_c_sample.device,
        )
        if not torch.equal(
            torch.sort(c_view_to_c_sample.long()).values,
            expected_c_views.long(),
        ) or not torch.equal(
            c_view_to_sample.long(),
            c_sample_indices.index_select(0, c_view_to_c_sample.long()).long(),
        ):
            raise ValueError(
                "Route1 B/C native view ownership is duplicated, missing, or misindexed"
            )
        if not isinstance(component_gradient_audit, bool):
            raise TypeError("Route1 component-gradient audit switch must be boolean")
        if answer_course_geometry not in {
            "deployed_z_cot_suffix",
            "deployed_z",
        }:
            raise ValueError("Route1 answer-course geometry is invalid")
        if answer_course_geometry == "deployed_z" and bool(course_cot_mask.any()):
            raise ValueError("completed Route1 course must be pure deployed z")
        expected_reduction = route1_course_reduction(
            0 if answer_course_geometry == "deployed_z_cot_suffix" else 1
        )
        if course_reduction != expected_reduction:
            raise ValueError("Route1 course reduction differs from its fixed contract")
        counts = {
            "course_c": global_course_c_sample_count,
            "course_direct_side": global_course_direct_side_sample_count,
            "match": global_match_sample_count,
            "specific": global_specific_sample_count,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts.values()
        ):
            raise ValueError(f"Route1 global counts are invalid: {counts}")
        timer = _Route1ForwardTimer(prompt_ids.device, timing_sink, progress_sink)
        if prepared_state is None:
            true_z = _route1_symmetric_staged_vjp_call(
                lambda: timer.call(
                    "live_z_seconds",
                    lambda: self.reason(prompt_ids, prompt_mask),
                    stage="live-z",
                ),
                device=prompt_ids.device,
                label=(
                    f"update={int(rollout_update)} "
                    f"microstep={int(rollout_micro_step)} reasoner forward"
                ),
            )
        else:
            if (
                prepared_state["consumed"]
                or prepared_state["update"] != int(rollout_update)
                or prepared_state["microstep"] != int(rollout_micro_step)
                or prepared_state["sample_keys"] != tuple(rollout_sample_keys)
                or prepared_state["provider"] is not free_generation_provider
                or prepared_state["prompt_ids"] is not prompt_ids
                or prepared_state["prompt_mask"] is not prompt_mask
                or prepared_state["distill_cot_ids"] is not distill_cot_ids
                or prepared_state["distill_cot_mask"] is not distill_cot_mask
            ):
                raise ValueError(
                    "prepared Route1 state is stale or belongs to a different request"
                )
            prepared_state["consumed"] = True
            true_z = prepared_state["z"]
            timer.events.extend(prepared_state.get("timing_events", ()))
        true_z_views: tuple[torch.Tensor, ...] = (true_z,)
        requested_views = int(getattr(self.reasoner, "dropout_views", 1))
        if (
            self.training
            and requested_views > 1
            and getattr(self.reasoner, "latent_view_dropout", 0.0) > 0.0
        ):
            extra_views = tuple(
                _route1_symmetric_staged_vjp_call(
                    lambda: self.reason(prompt_ids, prompt_mask),
                    device=prompt_ids.device,
                    label=(
                        f"update={int(rollout_update)} microstep={int(rollout_micro_step)} "
                        f"reasoner dropout view={index}"
                    ),
                )
                for index in range(1, requested_views)
            )
            true_z_views = (true_z, *extra_views)
        # `route1_bridge` has one maintained execution semantics.  The compatibility
        # argument is retained so sealed pre-split configs still parse, but both
        # values enter the same bounded streamed-VJP implementation.
        return self._forward_route1_bridge_streamed_from_z(
            true_z=true_z,
            true_z_views=true_z_views,
            timer=timer,
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            direct_prompt_ids=direct_prompt_ids,
            direct_prompt_mask=direct_prompt_mask,
            course_cot_ids=course_cot_ids,
            course_cot_mask=course_cot_mask,
            course_answer_ids=course_answer_ids,
            course_answer_mask=course_answer_mask,
            course_view_to_sample=course_view_to_sample,
            course_view_to_course_sample=course_view_to_course_sample,
            course_sample_side=course_sample_side,
            c_native_cot_ids=c_native_cot_ids,
            c_native_cot_mask=c_native_cot_mask,
            c_view_to_sample=c_view_to_sample,
            wrong_candidate_mask=wrong_candidate_mask,
            specificity_positive_mask=specificity_positive_mask,
            specificity_donor_prompt_ids=specificity_donor_prompt_ids,
            specificity_donor_prompt_mask=specificity_donor_prompt_mask,
            answer_course_geometry=answer_course_geometry,
            course_reduction=course_reduction,
            global_course_c_sample_count=int(global_course_c_sample_count),
            global_course_direct_side_sample_count=int(
                global_course_direct_side_sample_count
            ),
            global_match_sample_count=int(global_match_sample_count),
            global_specific_sample_count=int(global_specific_sample_count),
            distill_row_weights=distill_row_weights,
            distill_cot_ids=distill_cot_ids,
            distill_cot_mask=distill_cot_mask,
            course_weight=float(course_weight),
            match_weight=float(match_weight),
            specific_weight=float(specific_weight),
            specificity_tau=float(specificity_tau),
            physical_chunk_size=int(physical_chunk_size),
            wrong_control_chunk_size=int(wrong_control_chunk_size),
            component_gradient_audit=bool(component_gradient_audit),
            specificity_step_audit=bool(specificity_step_audit),
            timing_sink=timing_sink,
            free_generation_provider=free_generation_provider,
            rollout_update=int(rollout_update),
            rollout_micro_step=int(rollout_micro_step),
            rollout_sample_keys=tuple(rollout_sample_keys),
            prepared_state=prepared_state,
        )

    def forward(self, *, route: str, **kwargs):
        if route != "route1_bridge":
            raise ValueError("route must be route1")
        return self.forward_route1_bridge(**kwargs)
