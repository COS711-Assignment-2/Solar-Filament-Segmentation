"""Configurable binary segmentation and auxiliary boundary losses."""

from __future__ import annotations

import torch
from torch import nn


def segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float,
    dice_weight: float,
) -> torch.Tensor:
    """Return weighted BCE plus soft Dice loss."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    probabilities = torch.sigmoid(logits)
    dimensions = tuple(range(1, logits.ndim))
    intersection = (probabilities * target).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + target.sum(dim=dimensions)
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0))
    return bce_weight * bce + dice_weight * dice.mean()


def boundary_target(instance_ids: torch.Tensor) -> torch.Tensor:
    """Mark both sides of horizontal/vertical borders between instances."""
    boundary = torch.zeros_like(instance_ids, dtype=torch.bool)
    horizontal = (instance_ids[..., 1:] != instance_ids[..., :-1]) & (
        (instance_ids[..., 1:] > 0) | (instance_ids[..., :-1] > 0)
    )
    vertical = (instance_ids[..., 1:, :] != instance_ids[..., :-1, :]) & (
        (instance_ids[..., 1:, :] > 0) | (instance_ids[..., :-1, :] > 0)
    )
    boundary[..., 1:] |= horizontal
    boundary[..., :-1] |= horizontal
    boundary[..., 1:, :] |= vertical
    boundary[..., :-1, :] |= vertical
    return boundary.float().unsqueeze(1)
