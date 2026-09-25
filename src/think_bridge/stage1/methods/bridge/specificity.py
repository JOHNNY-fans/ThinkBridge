"""Same-prompt capped soft InfoNCE and detached, mean-one similarity weights."""

from __future__ import annotations

import math
import torch
import torch.nn.functional as torch_functional


def similarity_negative_weights(
    owner_repr: torch.Tensor, donor_repr: torch.Tensor
) -> torch.Tensor:
    """Detached cosine-similarity prior before legal-donor normalization."""
    with (
        torch.no_grad(),
        torch.autocast(device_type=owner_repr.device.type, enabled=False),
    ):
        owner = torch_functional.normalize(owner_repr.detach().float(), dim=-1)
        donor = torch_functional.normalize(donor_repr.detach().float(), dim=-1)
        cosine = (owner @ donor.transpose(0, 1)).clamp(-1.0, 1.0)
        return ((1.0 + cosine) * 0.5).clamp_min(1e-3)


def mean_one_donor_weights(weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Normalize stopped priors over selected legal donors only (not the bank)."""
    legal = weights.detach().masked_fill(~mask, 0.0)
    mean = legal.sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp_min(1)
    return legal / mean.clamp_min(torch.finfo(legal.dtype).tiny)


def masked_capped_soft_infonce(
    d_true: torch.Tensor,
    d_donors: torch.Tensor,
    donor_mask: torch.Tensor,
    donor_weights: torch.Tensor | None = None,
    *,
    negative_kl_cap: float,
    temperature: float = 0.1,
    normalize_weights: bool = False,
    positive_mask: torch.Tensor | None = None,
    normalize_candidate_counts: bool = False,
) -> torch.Tensor:
    """Multi-positive probability mass against soft-weighted, saturated negatives.

    -log(sum_p exp(-d_p/tau) / (sum_p exp(-d_p/tau)
                              + sum_j w_j exp(-min(d_j, cap)/tau))).
    With normalize_candidate_counts=True, each sum is divided by its valid
    candidate count; positive_mask permits ragged same-prompt positive sets.
    The cap is in raw token-mean FKL units, with zero negative derivative at
    and above the boundary. Positives remain live. Weights are detached;
    normalize_weights=True uses a mean-one prior over legal donors.
    The additive log(P) is omitted. Empty or
    zero-weight negative sets yield connected zeros. Reduction is per owner;
    the caller applies the global eligible-owner denominator.
    """
    if (
        d_true.ndim != 2
        or d_true.size(1) < 1
        or d_donors.ndim != 2
        or d_donors.size(0) != d_true.size(0)
        or donor_mask.shape != d_donors.shape
        or donor_mask.dtype != torch.bool
        or d_donors.device != d_true.device
        or donor_mask.device != d_true.device
    ):
        raise ValueError(
            "capped soft InfoNCE requires [B,P] true and aligned [B,K] donors/mask"
        )
    cap, tau = float(negative_kl_cap), float(temperature)
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError("specificity_negative_kl_cap must be finite and positive")
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("specificity_temperature must be finite and positive")
    dtype = (
        torch.float64
        if d_true.dtype == torch.float64 or d_donors.dtype == torch.float64
        else torch.float32
    )
    negative = d_donors.to(dtype).masked_fill(~donor_mask, 0.0)
    weights = torch.ones_like(negative)
    if donor_weights is not None:
        if (
            donor_weights.shape != d_donors.shape
            or donor_weights.device != d_true.device
        ):
            raise ValueError("soft weights must align with donors")
        weights = donor_weights.detach().to(dtype).masked_fill(~donor_mask, 0.0)
        if not bool((torch.isfinite(weights) & (weights >= 0) & (weights <= 1)).all()):
            raise ValueError("soft weights must be finite in [0, 1]")
    if normalize_weights:
        weights = mean_one_donor_weights(weights, donor_mask)
    effective = donor_mask & (weights > 0)
    eligible = effective.any(dim=1)
    positive = d_true.to(dtype).masked_fill(~eligible[:, None], 0.0)
    if negative.size(1) == 0:
        return positive.sum(dim=1).mul(0.0) + negative.sum(dim=1).mul(0.0)
    # torch.clamp(max=cap) uses a live derivative at equality; explicitly stop it.
    bounded = torch.where(negative < cap, negative, negative.new_full((), cap))
    negative_scores = -bounded / tau + weights.clamp_min(torch.finfo(dtype).tiny).log()
    negative_scores = negative_scores.masked_fill(~effective, float("-inf"))
    # Avoid undefined logsumexp derivatives for a row with no effective negatives.
    negative_scores = negative_scores.masked_fill(~eligible[:, None], 0.0)
    if positive_mask is None:
        positive_mask = torch.ones_like(positive, dtype=torch.bool)
    if (
        positive_mask.shape != positive.shape
        or positive_mask.dtype != torch.bool
        or positive_mask.device != positive.device
        or not bool(positive_mask.any(1).all())
    ):
        raise ValueError("each owner requires an aligned nonempty positive mask")
    positive_scores = (-positive / tau).masked_fill(~positive_mask, float("-inf"))
    log_positive = torch.logsumexp(positive_scores, dim=1)
    log_negative = torch.logsumexp(negative_scores, dim=1)
    if normalize_candidate_counts:
        log_positive = log_positive - positive_mask.sum(1).to(dtype).log()
        log_negative = log_negative - effective.sum(1).clamp_min(1).to(dtype).log()
    result = torch_functional.softplus(log_negative - log_positive)
    return result.masked_fill(~eligible, 0.0)
