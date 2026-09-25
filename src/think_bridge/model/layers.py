"""Reusable normalization, conditioning, feed-forward, and rotary layers."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square normalization with a learned scale."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * (x / rms)).to(dtype)


def _sinusoidal_embed(
    t: torch.Tensor, dim: int, max_period: float = 10000.0
) -> torch.Tensor:
    """Sinusoidal timestep embedding: [B] scalar -> [B, dim]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / half
    )
    args = t[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepEmbedder(nn.Module):
    """Project sinusoidal scalar timesteps into the model width."""

    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] scalar timestep -> [B, hidden_size]."""
        emb = _sinusoidal_embed(t, self.freq_dim)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward projection with a two-thirds hidden width."""

    def __init__(self, dim: int, dim_feedforward: int):
        super().__init__()
        hidden = int(dim_feedforward * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden)
        self.w3 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class RotaryEmbedding(nn.Module):
    """One-dimensional rotary position embedding cache."""

    def __init__(self, dim: int, max_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_len)

    def _build_cache(self, max_len: int) -> None:
        dev = self.inv_freq.device
        # Length varies by rank once H_x is part of D's condition. These are
        # disposable local caches, not buffers for DDP to broadcast. Always
        # construct in FP32, including growth inside a BF16 forward.
        with torch.autocast(device_type=dev.type, enabled=False):
            pos = torch.arange(max_len, dtype=torch.float32, device=dev)
            freqs = torch.outer(pos, self.inv_freq.float()).repeat_interleave(2, dim=-1)
            self.cos_cached = torch.cos(freqs)
            self.sin_cached = torch.sin(freqs)

    def forward(
        self, x: torch.Tensor, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cosine and sine tables covering ``seq_len`` positions."""
        if seq_len > self.cos_cached.size(0) or self.cos_cached.device != x.device:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]
