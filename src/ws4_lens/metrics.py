"""Lens evaluation metrics: entropy, KL divergence, and top-1 accuracy (Issue L2).

This module implements evaluation metrics for the mechanistic lens spectrum,
computing vocabulary prediction uncertainty, divergence to the model's final
layer, and discrete top-1 accuracy with numerical stability.
"""

from __future__ import annotations

import torch


def entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute Shannon entropy in nats from unnormalized logits.

    Logits are cast to float32 for numerical stability.

    Formula:
        logp = torch.log_softmax(logits.float(), dim=-1)
        entropy = -(logp.exp() * logp).sum(dim=-1)

    Parameters
    ----------
    logits : torch.Tensor
        Logits tensor of shape [..., V].

    Returns
    -------
    torch.Tensor
        Entropy tensor of shape [...]. Values are clamped at 0 from below.
    """
    logits_f32 = logits.float()
    logp = torch.log_softmax(logits_f32, dim=-1)
    ent = -(logp.exp() * logp).sum(dim=-1)
    return torch.clamp(ent, min=0.0)


def kl_to_final(logits_l: torch.Tensor, logits_L: torch.Tensor) -> torch.Tensor:
    """
    Compute Kullback-Leibler divergence KL(P_L || P_l) in nats.

    Measures the distributional distance from intermediate layer l to the
    model's final layer L. Logits are cast to float32 for numerical stability.

    Formula:
        logp_l = torch.log_softmax(logits_l.float(), dim=-1)
        logp_L = torch.log_softmax(logits_L.float(), dim=-1)
        kl = (logp_L.exp() * (logp_L - logp_l)).sum(dim=-1)

    Parameters
    ----------
    logits_l : torch.Tensor
        Logits of layer l, shape [..., V].
    logits_L : torch.Tensor
        Logits of final layer L, shape [..., V].

    Returns
    -------
    torch.Tensor
        KL divergence tensor of shape [...]. Clamped at 0 from below to prevent
        tiny negative values due to floating-point rounding.
    """
    logits_l_f32 = logits_l.float()
    logits_L_f32 = logits_L.float()
    logp_l = torch.log_softmax(logits_l_f32, dim=-1)
    logp_L = torch.log_softmax(logits_L_f32, dim=-1)
    kl = (logp_L.exp() * (logp_L - logp_l)).sum(dim=-1)
    return torch.clamp(kl, min=0.0)


def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute Top-1 prediction accuracy over valid target positions (targets >= 0).

    Positions where targets < 0 (such as -100 for ignored or padding tokens)
    are excluded from the computation, following Hugging Face convention.
    Targets are automatically moved to the device of logits if needed.

    Parameters
    ----------
    logits : torch.Tensor
        Logits tensor of shape [..., V].
    targets : torch.Tensor
        Target token indices matching batch/sequence dimensions [...].

    Returns
    -------
    float
        Fraction of correctly predicted tokens in [0.0, 1.0].
        Returns 0.0 if no valid target positions exist.
    """
    targets = targets.to(logits.device)
    preds = logits.argmax(dim=-1)
    mask = targets >= 0
    if not mask.any():
        return 0.0
    return float((preds[mask] == targets[mask]).float().mean().item())
