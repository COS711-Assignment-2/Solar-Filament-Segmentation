"""Configurable binary segmentation, focal, soft clDice, and auxiliary boundary losses."""

from __future__ import annotations

import torch
from torch import nn


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    """Morphological erosion via min-pooling for 2D probability maps."""
    p1 = -nn.functional.max_pool2d(-img, (3, 1), stride=1, padding=(1, 0))
    p2 = -nn.functional.max_pool2d(-img, (1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def soft_skeletonize(img: torch.Tensor, iterations: int = 3) -> torch.Tensor:
    """Differentiable soft morphological skeletonization (Shit et al., CVPR 2021)."""
    skel = torch.zeros_like(img)
    current = img
    for _ in range(iterations):
        eroded = soft_erode(current)
        opened = nn.functional.max_pool2d(eroded, kernel_size=3, stride=1, padding=1)
        skel = skel + torch.relu(current - opened)
        current = eroded
    return torch.clamp(skel, 0.0, 1.0)


def soft_cldice_loss(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    iterations: int = 3,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Centerline Dice loss for topological continuity on thin curvilinear structures."""
    skel_pred = soft_skeletonize(probabilities, iterations=iterations)
    skel_true = soft_skeletonize(target, iterations=iterations)

    dims = tuple(range(1, probabilities.ndim))
    tprec = (skel_pred * target).sum(dim=dims) / (skel_pred.sum(dim=dims) + eps)
    tsens = (skel_true * probabilities).sum(dim=dims) / (skel_true.sum(dim=dims) + eps)

    cldice = (2.0 * tprec * tsens + eps) / (tprec + tsens + eps)
    return (1.0 - cldice).mean()


def segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float,
    dice_weight: float,
    cldice_weight: float = 0.0,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    """Return weighted BCE/Focal, soft Dice, and optional clDice loss."""
    probabilities = torch.sigmoid(logits)

    if focal_gamma > 0.0:
        bce_raw = nn.functional.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
        p_t = probabilities * target + (1.0 - probabilities) * (1.0 - target)
        focal_factor = (1.0 - p_t) ** focal_gamma
        bce = (focal_factor * bce_raw).mean()
    else:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target)

    dimensions = tuple(range(1, logits.ndim))
    intersection = (probabilities * target).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + target.sum(dim=dimensions)
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0))

    total = bce_weight * bce + dice_weight * dice.mean()
    if cldice_weight > 0.0:
        total = total + cldice_weight * soft_cldice_loss(probabilities, target)
    return total


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
