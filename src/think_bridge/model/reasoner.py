"""A single prompt reader followed by two shared latent self-attention loops."""

from __future__ import annotations
import math
from collections.abc import Sequence
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F_func
from think_bridge.model.feedback_precision import (
    QUESTION_READER_INPUT_MODES,
    REASONER_INPUT_MODES,
)
from think_bridge.model.layers import (
    RMSNorm,
    _sinusoidal_embed,
)


@dataclass(frozen=True)
class LatentView:
    """One dropout-regularized form of a deterministic latent hidden state."""

    base_z: torch.Tensor
    z: torch.Tensor
    seeds: tuple[int, ...] | None
    keep_fraction: torch.Tensor


class CrossBlock(nn.Module):
    """Noncausal latent self-attention followed by a SwiGLU feedforward layer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        *,
        cross_attention: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        if num_heads <= 0 or d_model % num_heads:
            raise ValueError("R hidden size must be divisible by its attention heads")
        self.head_dim = d_model // num_heads
        from transformers import Qwen3Config
        from think_bridge.model.looped_transformer import LoopedTransformerLayer

        config = Qwen3Config(
            hidden_size=d_model,
            intermediate_size=int(dim_feedforward * 2 / 3),
            num_hidden_layers=1,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            head_dim=self.head_dim,
            attention_bias=True,
            attention_dropout=0.0,
            rms_norm_eps=1e-6,
        )
        config._attn_implementation = "sdpa"
        self.core = LoopedTransformerLayer(config, layer_idx=0)
        if cross_attention:
            raise ValueError("latent layers use self-attention only")
        self.use_cross_attention = False

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        return self.core.feedforward_step(self.core.attention_step(slots, causal=False))


class QuestionReader(nn.Module):
    """Queries choose input content; there is no query residual or value bias."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if num_heads <= 0 or d_model % num_heads:
            raise ValueError(
                "reader hidden size must be divisible by its attention heads"
            )
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        for projection in (self.q, self.k, self.v, self.out):
            nn.init.xavier_uniform_(projection.weight)

    def forward(self, queries, context, mask):
        B, N, D = queries.shape

        def heads(tensor):
            return tensor.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(heads(self.q(queries)))
        k = self.k_norm(heads(self.k(context)))
        v = heads(self.v(context))
        value = F_func.scaled_dot_product_attention(
            q, k, v, attn_mask=mask[:, None, None, :].bool()
        )
        return self.out(value.transpose(1, 2).reshape(B, N, D))


class RecurrentReasoner(nn.Module):
    """Read final-layer prompt states once, then apply shared latent layers."""

    def __init__(
        self,
        d_model: int,
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
        torch_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.latent_steps = int(latent_steps)
        self.latents_per_step = int(latents_per_step)
        self.num_slots = self.latent_steps * self.latents_per_step
        if (
            isinstance(loop_steps, bool)
            or not isinstance(loop_steps, int)
            or loop_steps < 1
        ):
            raise ValueError("reasoner loop_steps must be a positive integer")
        if (
            isinstance(num_layers, bool)
            or not isinstance(num_layers, int)
            or num_layers < 1
        ):
            raise ValueError("reasoner num_layers must be a positive integer")
        from think_bridge.model.looped_transformer import looped_layer_schedule

        self.layer_schedule = looped_layer_schedule(num_layers, loop_steps)
        self.loop_steps = loop_steps
        self.depth = num_layers
        self.recurrent_steps = self.latent_steps
        if not math.isfinite(float(dropout_p)) or not 0.0 <= float(dropout_p) < 1.0:
            raise ValueError("dropout_p must be finite in [0, 1)")
        if isinstance(dropout_views, bool) or not 1 <= int(dropout_views) <= 8:
            raise ValueError("dropout_views must be an integer in [1, 8]")
        self.latent_view_dropout = float(dropout_p)
        self.dropout_views = int(dropout_views)
        if input_mode not in REASONER_INPUT_MODES:
            raise ValueError("unknown reasoner input_mode")
        if input_mode in QUESTION_READER_INPUT_MODES and (
            latent_steps != 1 or zero_residual_init
        ):
            raise ValueError(
                "question reader requires one external step and nonzero initialization"
            )
        self.input_mode = input_mode
        self.tap_count = 1 if input_mode in QUESTION_READER_INPUT_MODES else 6
        self.zero_residual_init = bool(zero_residual_init)
        if output_normalization != "residual":
            raise ValueError("R output_normalization must be residual")
        self.output_normalization = output_normalization
        if (self.latent_steps, self.latents_per_step, num_layers, loop_steps) != (
            1,
            64,
            2,
            2,
        ):
            raise ValueError(
                "R requires 64 slots, one input read, and two layers repeated twice"
            )
        if not math.isfinite(float(bound_scale_init)) or bound_scale_init <= 0:
            raise ValueError("bound_scale_init must be finite and positive")
        self.query_base = nn.Parameter(
            torch.randn(1, self.latents_per_step, self.d_model) * 0.02
        )
        self.context_norms = nn.ModuleList(
            [RMSNorm(self.d_model) for _ in range(self.tap_count)]
        )
        self.input_reader = QuestionReader(self.d_model, int(num_heads))
        self.blocks = nn.ModuleList(
            CrossBlock(self.d_model, int(num_heads), int(dim_feedforward))
            for _ in range(num_layers)
        )
        physical_blocks = tuple(self.blocks)
        self.state_norm = RMSNorm(self.d_model)
        # Each physical layer is independent; repeated executions share it.
        for block in physical_blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            projections = (block.core.self_attn.o_proj, block.core.mlp.down_proj)
            # Retain the existing initialization gain; only remove the CA path.
            for projection in projections:
                nn.init.xavier_uniform_(projection.weight, gain=1.0 / math.sqrt(3.0))
        with torch.no_grad():
            steps = torch.arange(1, self.latent_steps + 1, dtype=torch.float32)
            sinusoid = _sinusoidal_embed(steps, self.d_model)
            sinusoid_rms = sinusoid.square().mean(dim=-1, keepdim=True).sqrt()
            query_rms = self.query_base.float().square().mean().sqrt()
            step_embed = sinusoid / sinusoid_rms * query_rms
            step_embed = (
                step_embed[:, None, :]
                .expand(-1, self.latents_per_step, -1)
                .contiguous()
            )
        self.register_buffer("step_embed", step_embed, persistent=True)
        self.register_buffer(
            "tau_bound", torch.tensor(1.0, dtype=torch.float32), persistent=True
        )
        self.log_s_bound = nn.Parameter(
            torch.tensor(math.log(float(bound_scale_init)), dtype=torch.float32)
        )
        if torch_dtype is not None:
            self.to(torch_dtype)
        self.step_embed.data = self.step_embed.float()
        self.tau_bound.data = self.tau_bound.float()
        self.log_s_bound.data = self.log_s_bound.float()
        self._gradient_checkpointing = False

    def step_queries(self, step_index: int, batch_size: int) -> torch.Tensor:
        if not 0 <= int(step_index) < self.latent_steps:
            raise IndexError("latent step is out of range")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        return (
            self.query_base
            + self.step_embed[int(step_index)]
            .unsqueeze(0)
            .to(device=self.query_base.device, dtype=self.query_base.dtype)
        ).expand(int(batch_size), -1, -1)

    def emit_step(
        self,
        tap_histories: Sequence[torch.Tensor],
        sequence_mask: torch.Tensor,
        *,
        step_index: int,
        previous_state: torch.Tensor | None = None,
        reader_context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Update and return the live state, also used as this round's raw z.

        The caller owns state; never store it on the module or detach it.
        Internal slots are parallel latent states, not autoregressive tokens.
        Context includes only the prompt and already emitted blocks.
        """
        taps = tuple(tap_histories)
        if len(taps) != self.tap_count:
            raise ValueError(
                f"emitter requires {self.tap_count} frozen-F taps, got {len(taps)}"
            )
        first = taps[0]
        if first.ndim != 3 or any((tap.shape != first.shape for tap in taps)):
            raise ValueError("all taps must have shape [batch, sequence, hidden]")
        if tuple(sequence_mask.shape) != tuple(first.shape[:2]):
            raise ValueError("sequence_mask must cover the full tap span")
        if not 0 <= int(step_index) < self.latent_steps:
            raise IndexError("latent step is out of range")
        if (previous_state is None) != (int(step_index) == 0):
            raise ValueError(
                "first loop requires no state; later loops require previous_state"
            )
        expected_shape = (int(first.size(0)), self.latents_per_step, self.d_model)
        if previous_state is not None and tuple(previous_state.shape) != expected_shape:
            raise ValueError("previous_state must match [batch, block slots, hidden]")
        if not bool(sequence_mask.bool().any(dim=-1).all()):
            raise ValueError("each R context requires at least one valid position")
        if reader_context_mask is not None:
            if (
                self.training
                or self.input_mode != "last-query-reader-self-loop"
                or self.latent_steps != 1
            ):
                raise ValueError(
                    "Reader context mask is only supported for evaluation of one-step self-loop R"
                )
            if (
                reader_context_mask.shape != sequence_mask.shape
                or reader_context_mask.dtype != torch.bool
                or reader_context_mask.device != sequence_mask.device
                or bool((reader_context_mask & ~sequence_mask.bool()).any())
                or not bool(reader_context_mask.any(-1).all())
            ):
                raise ValueError(
                    "Reader context mask must select nonempty valid context positions"
                )
        dtype = self.query_base.dtype
        if previous_state is None:
            state = self.step_queries(0, int(first.size(0)))
        else:
            state = previous_state + self.step_embed[int(step_index)].unsqueeze(0)

        def _emit_from_histories(
            live_state: torch.Tensor, *history_and_mask: torch.Tensor
        ) -> torch.Tensor:
            history = history_and_mask[:-1]
            live_mask = history_and_mask[-1]
            normalized = [
                norm(tap.to(dtype=dtype))
                for (norm, tap) in zip(self.context_norms, history, strict=True)
            ]
            context = torch.cat(normalized, dim=1)
            context_mask = live_mask.to(torch.bool).repeat(1, len(history))
            # Use unit-RMS initial queries, not tiny vectors followed by an
            # inverse-RMS amplification at every first sublayer. No trainable
            # gain here; state_norm below owns the loop-boundary gain.
            initial_slots = live_state.float()
            initial_slots = initial_slots * torch.rsqrt(
                initial_slots.square().mean(dim=-1, keepdim=True) + 1e-6
            )
            reader_mask = (
                context_mask if reader_context_mask is None else reader_context_mask
            )
            question_update = self.input_reader(
                initial_slots, context, reader_mask
            ).float()
            updated = initial_slots + question_update
            for index in self.layer_schedule:
                updated = self.blocks[index](updated).float()
            emitted = self.normalize_output(updated)
            # Dropout is a training-only stochastic view of the latent state.
            # It is deliberately after the residual/state normalization and
            # before F consumes the block, so evaluation and deployment remain

            # zero.  The caller is responsible for requesting multiple views.
            if self.training and self.latent_view_dropout > 0.0:
                emitted = torch.nn.functional.dropout(
                    emitted,
                    p=self.latent_view_dropout,
                    training=True,
                )
            return emitted

        if self._gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                _emit_from_histories, state, *taps, sequence_mask, use_reentrant=False
            )
        return _emit_from_histories(state, *taps, sequence_mask)

    def normalize_output(self, residual: torch.Tensor) -> torch.Tensor:
        """Apply an FP32 channel gain without dividing by residual RMS."""
        return residual.float() * self.state_norm.weight.float()

    def bound(self, raw: torch.Tensor, *, embedding_rms: torch.Tensor) -> torch.Tensor:
        """Bound one emitted residual block in embedding RMS units."""
        if raw.ndim != 3 or raw.size(-1) != self.d_model:
            raise ValueError("bound input must have shape [batch, slots, hidden]")
        tau = self.tau_bound.detach().float().clone()
        if not bool(torch.isfinite(tau).all()) or not bool(tau.gt(0).all()):
            raise FloatingPointError("tau_bound must be finite and positive")
        source = raw.float()
        mean_square = source.square().mean(dim=-1, keepdim=True)
        scale = torch.exp(self.log_s_bound.float())
        cap = (
            embedding_rms.detach().to(device=raw.device, dtype=torch.float32).clone()
            * scale
        )
        denominator = tau * torch.sqrt(1.0 + mean_square / tau.square())
        return (cap * source / denominator).to(dtype=raw.dtype)

    def bound_diagnostics(self, raw: torch.Tensor) -> dict[str, torch.Tensor]:
        if raw.ndim != 3:
            raise ValueError("raw latent tensor must have shape [B,K,D]")
        rho = (
            raw.float().square().mean(dim=-1).sqrt() / self.tau_bound.float()
        ).reshape(-1)
        gain = 1.0 / (1.0 + rho.square())
        return {
            "rho_min": rho.min(),
            "rho_median": rho.median(),
            "rho_max": rho.max(),
            "radial_gain_min": gain.min(),
            "radial_gain_median": gain.median(),
            "radial_gain_max": gain.max(),
            "s_bound": torch.exp(self.log_s_bound.float()),
            "tau_bound": self.tau_bound.float(),
        }

    @staticmethod
    def deterministic_view(base_z: torch.Tensor) -> LatentView:
        if base_z.ndim != 3:
            raise ValueError("base_z must have shape [batch, slots, hidden]")
        keep_fraction = torch.ones(
            int(base_z.size(0)), dtype=torch.float32, device=base_z.device
        )
        return LatentView(base_z, base_z, None, keep_fraction)

    @staticmethod
    def sample_view_seeds(batch_size: int) -> list[int]:
        if int(batch_size) < 0:
            raise ValueError("batch_size must be nonnegative")
        # Kept as a small compatibility helper for diagnostics.  Training
        # views use PyTorch's device-local dropout RNG instead of replay seeds.
        return [
            int(value)
            for value in torch.randint(0, 2**31 - 1, (int(batch_size),)).tolist()
        ]

    def materialize_view(
        self, base_z: torch.Tensor, *, seeds: Sequence[int] | None = None
    ) -> LatentView:
        if seeds is not None:
            raise RuntimeError(
                "latent replay seeds are unsupported; use BridgeParallelModel.reason_views"
            )
        return self.deterministic_view(base_z)
