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


def instance_affinity_targets(
    instance_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return right/down same-instance labels and valid foreground edge masks."""
    horizontal_left = instance_ids[..., :, :-1]
    horizontal_right = instance_ids[..., :, 1:]
    vertical_top = instance_ids[..., :-1, :]
    vertical_bottom = instance_ids[..., 1:, :]

    targets = torch.zeros(
        (instance_ids.shape[0], 2, *instance_ids.shape[-2:]),
        dtype=torch.float32,
        device=instance_ids.device,
    )
    valid = torch.zeros_like(targets, dtype=torch.bool)
    horizontal_valid = (horizontal_left > 0) & (horizontal_right > 0)
    vertical_valid = (vertical_top > 0) & (vertical_bottom > 0)
    targets[:, 0, :, :-1] = (
        (horizontal_left == horizontal_right) & horizontal_valid
    ).float()
    targets[:, 1, :-1, :] = (
        (vertical_top == vertical_bottom) & vertical_valid
    ).float()
    valid[:, 0, :, :-1] = horizontal_valid
    valid[:, 1, :-1, :] = vertical_valid
    return targets, valid


def instance_affinity_loss(
    logits: torch.Tensor,
    instance_ids: torch.Tensor,
) -> torch.Tensor:
    """Balance same-instance and cross-instance neighboring foreground edges."""
    targets, valid = instance_affinity_targets(instance_ids)
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    valid_float = valid.float()
    positive = (targets * valid_float).sum(dim=(0, 2, 3))
    negative = ((1.0 - targets) * valid_float).sum(dim=(0, 2, 3))
    positive_weight = (negative / positive.clamp_min(1.0)).clamp_min(1.0)
    weighted = losses * (1.0 + targets * (positive_weight.view(1, 2, 1, 1) - 1.0))
    return (weighted * valid_float).sum() / valid_float.sum().clamp_min(1.0)


def instance_embedding_loss(
    embeddings: torch.Tensor,
    instance_ids: torch.Tensor,
    pull_margin: float = 0.5,
    push_margin: float = 1.5,
    sample_count: int = 256,
) -> torch.Tensor:
    """Pull sampled same-instance embeddings together and repel different instances."""
    if embeddings.ndim != 4 or instance_ids.shape != (
        embeddings.shape[0],
        embeddings.shape[2],
        embeddings.shape[3],
    ):
        raise ValueError("Embedding and instance-ID tensor shapes are incompatible")
    if pull_margin <= 0 or push_margin <= pull_margin or sample_count < 2:
        raise ValueError("Invalid instance-embedding loss margins or sample count")

    batch_size, embedding_dim, height, width = embeddings.shape
    flattened_embeddings = embeddings.permute(0, 2, 3, 1).reshape(
        batch_size, height * width, embedding_dim
    )
    flattened_ids = instance_ids.reshape(batch_size, height * width)
    sample_count = min(sample_count, height * width)
    scores = torch.rand(
        flattened_ids.shape,
        device=embeddings.device,
        dtype=embeddings.dtype,
    )
    scores = scores.masked_fill(flattened_ids <= 0, -1.0)
    sampled_indices = scores.topk(sample_count, dim=1).indices
    sampled_ids = flattened_ids.gather(1, sampled_indices)
    sampled_embeddings = flattened_embeddings.gather(
        1, sampled_indices.unsqueeze(-1).expand(-1, -1, embedding_dim)
    )

    differences = sampled_embeddings.unsqueeze(2) - sampled_embeddings.unsqueeze(1)
    distances = torch.sqrt((differences.square().sum(dim=-1)).clamp_min(1e-8))
    valid = sampled_ids > 0
    same_instance = (
        (sampled_ids.unsqueeze(2) == sampled_ids.unsqueeze(1))
        & valid.unsqueeze(2)
        & valid.unsqueeze(1)
    )
    different_instance = (
        (sampled_ids.unsqueeze(2) != sampled_ids.unsqueeze(1))
        & valid.unsqueeze(2)
        & valid.unsqueeze(1)
    )
    diagonal = torch.eye(
        sample_count, dtype=torch.bool, device=embeddings.device
    ).unsqueeze(0)
    same_instance &= ~diagonal
    pull = torch.relu(distances - pull_margin).square()
    push = torch.relu(push_margin - distances).square()
    pull_loss = (pull * same_instance).sum() / same_instance.sum().clamp_min(1)
    push_loss = (
        (push * different_instance).sum()
        / different_instance.sum().clamp_min(1)
    )
    return pull_loss + push_loss
