"""Shared Qwen3-compatible layer for latent recurrence and causal CoT decoding.

R and D own separate instances. Reusing weights across depth never aliases KV
states across depth; the caller gives every executed D layer a cache index.
"""

from __future__ import annotations

import copy
import inspect
from functools import lru_cache
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3Model


@lru_cache(maxsize=2)
def _mask_parameters(function):
    # Transformers 4.x uses input_embeds/cache_position; newer 5.x renamed
    # the former to inputs_embeds. Match the installed API without catching
    # TypeError from the actual mask computation.
    return frozenset(inspect.signature(function).parameters)


class _DepthCache:
    """Redirect this invocation's cache update without mutating a shared layer."""

    def __init__(self, cache, index: int):
        self.cache = cache
        self.index = int(index)

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        return self.cache.update(key_states, value_states, self.index, *args, **kwargs)


class LoopedTransformerLayer(Qwen3DecoderLayer):
    """Pre-norm SA + optional CA callback + pre-norm SwiGLU FFN.

    With no callback this is an ordinary Qwen3 layer, including native GQA and
    Q/K normalization. The caller owns positions, masks and recurrence count.
    """

    @classmethod
    def from_qwen3(cls, source_layer, *, layer_idx: int = 0):
        config = copy.deepcopy(source_layer.self_attn.config)
        # No temporary source-sized allocation and no change to caller RNG.
        with torch.random.fork_rng(devices=[]), torch.device("meta"):
            result = cls(config, layer_idx=layer_idx)
        state = {
            key: value.detach().to(dtype=torch.float32).clone()
            if value.is_floating_point()
            else value.detach().clone()
            for key, value in source_layer.state_dict().items()
        }
        result.load_state_dict(state, assign=True)
        result.requires_grad_(True)
        return result

    def attention_step(
        self,
        hidden_states,
        *,
        attention_mask=None,
        position_embeddings=None,
        position_ids=None,
        cache=None,
        cache_index=None,
        causal=True,
        **kwargs,
    ):
        if position_embeddings is None:
            shape = (*hidden_states.shape[:2], self.self_attn.head_dim)
            position_embeddings = (
                hidden_states.new_ones(shape),
                hidden_states.new_zeros(shape),
            )
        if not causal and attention_mask is None:
            if cache is not None:
                raise ValueError("noncausal cached attention requires an explicit mask")
            # An explicit mask prevents SDPA from inferring causal attention.
            attention_mask = torch.zeros(
                (1, 1, hidden_states.size(1), hidden_states.size(1)),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        if cache is not None and cache_index is not None:
            cache = _DepthCache(cache, cache_index)
        attention, _ = self.self_attn(
            hidden_states=self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=cache,
            **kwargs,
        )
        return hidden_states + attention

    def feedforward_step(self, hidden_states):
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        position_embeddings=None,
        *,
        cache_index=None,
        causal=True,
        after_attention=None,
        **kwargs,
    ):
        hidden_states = self.attention_step(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            cache=past_key_values,
            cache_index=cache_index,
            causal=causal,
            use_cache=use_cache,
            **kwargs,
        )
        if after_attention is not None:
            hidden_states = after_attention(hidden_states)
        return self.feedforward_step(hidden_states)


def looped_layer_schedule(layer_count: int, loop_steps: int) -> tuple[int, ...]:
    """Execute two independently initialized layers in the order 0, 1, 0, 1."""
    if (
        type(layer_count) is not int
        or type(loop_steps) is not int
        or (layer_count, loop_steps) != (2, 2)
    ):
        raise ValueError("ThinkBridge uses two physical layers repeated twice")
    return (0, 1, 0, 1)


class LoopedQwen3Model(Qwen3Model):
    """Shared physical layers execute at independently indexed cache depths.

    Final normalization runs once, after the final layer. RoPE token positions do
    not change between internal loops. No state or cache is stored on the model.
    """

    def __init__(self, config, *, loop_steps=2):
        super().__init__(config)
        self.layer_schedule = looped_layer_schedule(
            config.num_hidden_layers, loop_steps
        )
        self.loop_steps = loop_steps
        self.cache_config = copy.deepcopy(config)
        self.cache_config.num_hidden_layers = len(self.layer_schedule)
        self.cache_config.layer_types = [
            config.layer_types[i] for i in self.layer_schedule
        ]

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=False,
        layer_after_attention=None,
        first_layer_after_attention=None,
        **kwargs,
    ):
        if (
            layer_after_attention is not None
            and first_layer_after_attention is not None
        ):
            raise ValueError(
                "choose per-layer or first-execution attention conditioning"
            )
        if (
            self.loop_steps == 1
            and layer_after_attention is None
            and first_layer_after_attention is None
        ):
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                **kwargs,
            )
        from transformers.cache_utils import DynamicCache
        from transformers.masking_utils import (
            create_causal_mask,
            create_sliding_window_causal_mask,
        )
        from transformers.modeling_outputs import BaseModelOutputWithPast

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if self.is_gradient_checkpointing and self.training and torch.is_grad_enabled():
            if use_cache or past_key_values is not None:
                raise ValueError(
                    "looped D training checkpointing cannot mutate a KV cache"
                )
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.cache_config)
        cache_position = kwargs.get("cache_position")
        if cache_position is None:
            offset = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = (
                torch.arange(inputs_embeds.size(1), device=inputs_embeds.device)
                + offset
            )
        if position_ids is None:
            position_ids = cache_position[None]
        if isinstance(attention_mask, dict):
            masks = attention_mask
        else:
            args = dict(
                config=self.config,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
            )

            def build_mask(function):
                parameters = _mask_parameters(function)
                arguments = dict(args)
                arguments[
                    "inputs_embeds" if "inputs_embeds" in parameters else "input_embeds"
                ] = inputs_embeds
                if "cache_position" in parameters:
                    arguments["cache_position"] = cache_position
                return function(**arguments)

            masks = {"full_attention": build_mask(create_causal_mask)}
            if self.has_sliding_layers:
                masks["sliding_attention"] = build_mask(
                    create_sliding_window_causal_mask
                )
        hidden = inputs_embeds
        positions = self.rotary_emb(hidden, position_ids)
        # cache_index is bound in each call/checkpoint closure, never a mutable
        # self_attn.layer_idx shared between two virtual depths.
        for virtual_index, physical_index in enumerate(self.layer_schedule):
            hidden = self.layers[physical_index](
                hidden,
                attention_mask=masks[self.config.layer_types[physical_index]],
                position_ids=position_ids,
                position_embeddings=positions,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_index=virtual_index,
                cache_position=cache_position,
                after_attention=(
                    first_layer_after_attention
                    if virtual_index == 0 and first_layer_after_attention is not None
                    else None
                    if layer_after_attention is None
                    else layer_after_attention[physical_index]
                ),
            )
        return BaseModelOutputWithPast(
            last_hidden_state=self.norm(hidden),
            past_key_values=past_key_values if use_cache else None,
        )
