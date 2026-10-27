"""Compact CondInst-style instance segmentation model.

This implementation uses a shared multi-scale feature map, an anchor-free
center controller, and per-instance dynamic convolution mask heads. It is
purpose-built for the project's grayscale images and raster instance targets;
it does not depend on Detectron2.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Sequential):
    """Two convolutional layers used by the compact feature pyramid."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class CondInst(nn.Module):
    """Anchor-free conditional instance segmentation for grayscale imagery."""

    is_instance_model = True
    output_stride = 4
    dynamic_hidden_channels = 8

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 24,
        mask_channels: int = 32,
        max_instances: int = 100,
        mask_threshold: float = 0.5,
        nms_distance: int = 2,
    ) -> None:
        super().__init__()
        if base_channels < 4 or mask_channels < 4:
            raise ValueError("base_channels and mask_channels must be at least 4")
        if max_instances < 1 or nms_distance < 1:
            raise ValueError("max_instances and nms_distance must be positive")
        if not 0.0 < mask_threshold < 1.0:
            raise ValueError("mask_threshold must be strictly between 0 and 1")
        self.max_instances = max_instances
        self.mask_threshold = mask_threshold
        self.nms_distance = nms_distance

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.stage2 = ConvBlock(base_channels, base_channels * 2, stride=2)
        self.stage3 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)
        self.stage4 = ConvBlock(base_channels * 4, base_channels * 8, stride=2)
        self.lateral2 = nn.Conv2d(base_channels * 2, mask_channels, 1)
        self.lateral3 = nn.Conv2d(base_channels * 4, mask_channels, 1)
        self.lateral4 = nn.Conv2d(base_channels * 8, mask_channels, 1)
        self.mask_tower = ConvBlock(mask_channels, mask_channels)
        self.controller_tower = ConvBlock(mask_channels, mask_channels)
        self.center_head = nn.Conv2d(mask_channels, 1, 3, padding=1)
        self.dynamic_parameter_count = (
            mask_channels * self.dynamic_hidden_channels
            + self.dynamic_hidden_channels
            + self.dynamic_hidden_channels * self.dynamic_hidden_channels
            + self.dynamic_hidden_channels
            + self.dynamic_hidden_channels
            + 1
        )
        self.controller = nn.Conv2d(
            mask_channels, self.dynamic_parameter_count, kernel_size=1
        )
        nn.init.constant_(self.center_head.bias, -2.19)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return shared mask features, center logits, and dynamic parameters."""
        features2 = self.stage2(self.stem(images))
        features3 = self.stage3(features2)
        features4 = self.stage4(features3)
        target_size = features2.shape[-2:]
        pyramid = (
            self.lateral2(features2)
            + F.interpolate(self.lateral3(features3), size=target_size, mode="bilinear", align_corners=False)
            + F.interpolate(self.lateral4(features4), size=target_size, mode="bilinear", align_corners=False)
        )
        mask_features = self.mask_tower(pyramid)
        controller_features = self.controller_tower(mask_features)
        return {
            "mask_features": mask_features,
            "center_logits": self.center_head(controller_features),
            "dynamic_parameters": self.controller(controller_features),
        }

    @staticmethod
    def _center_focal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
        positive = target.eq(1.0).float()
        negative = target.lt(1.0).float()
        negative_weights = (1.0 - target).pow(4)
        positive_loss = torch.log(probabilities) * (1.0 - probabilities).pow(2) * positive
        negative_loss = (
            torch.log(1.0 - probabilities)
            * probabilities.pow(2)
            * negative_weights
            * negative
        )
        return -(positive_loss.sum() + negative_loss.sum()) / positive.sum().clamp_min(1.0)

    def _dynamic_mask(self, features: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        """Apply one controller-generated three-layer 1x1 mask network."""
        hidden = self.dynamic_hidden_channels
        channels = features.shape[0]
        cursor = 0

        def take(size: int) -> torch.Tensor:
            nonlocal cursor
            values = parameters[cursor:cursor + size]
            cursor += size
            return values

        weights1 = take(hidden * channels).view(hidden, channels, 1, 1)
        bias1 = take(hidden)
        weights2 = take(hidden * hidden).view(hidden, hidden, 1, 1)
        bias2 = take(hidden)
        weights3 = take(hidden).view(1, hidden, 1, 1)
        bias3 = take(1)
        result = F.relu(F.conv2d(features.unsqueeze(0), weights1, bias1))
        result = F.relu(F.conv2d(result, weights2, bias2))
        return F.conv2d(result, weights3, bias3)[0, 0]

    def compute_loss(
        self,
        images: torch.Tensor,
        instance_targets: torch.Tensor,
        loss_config: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Compute center focal loss and per-object dynamic mask losses."""
        outputs = self(images)
        center_logits = outputs["center_logits"]
        mask_features = outputs["mask_features"]
        dynamic_parameters = outputs["dynamic_parameters"]
        batch_size, _, height, width = center_logits.shape
        center_targets = torch.zeros_like(center_logits)
        mask_losses: list[torch.Tensor] = []

        for batch_index in range(batch_size):
            instance_ids = torch.unique(instance_targets[batch_index])
            instance_ids = instance_ids[instance_ids > 0]
            occupied_centers = torch.zeros(
                (height, width), dtype=torch.bool, device=center_logits.device
            )
            for instance_id in instance_ids:
                rows, columns = torch.where(instance_targets[batch_index] == instance_id)
                if rows.numel() == 0:
                    continue
                center_y = int(torch.round(rows.float().mean()).item())
                center_x = int(torch.round(columns.float().mean()).item())
                grid_y = min(max(center_y // self.output_stride, 0), height - 1)
                grid_x = min(max(center_x // self.output_stride, 0), width - 1)
                if occupied_centers[grid_y, grid_x]:
                    available = torch.nonzero(~occupied_centers, as_tuple=False)
                    distances = (
                        (available[:, 0] - grid_y).float().square()
                        + (available[:, 1] - grid_x).float().square()
                    )
                    nearest = available[torch.argmin(distances)]
                    grid_y, grid_x = int(nearest[0]), int(nearest[1])
                occupied_centers[grid_y, grid_x] = True
                center_targets[batch_index, 0, grid_y, grid_x] = 1.0

                target_mask = F.interpolate(
                    (instance_targets[batch_index] == instance_id)
                    .float()[None, None],
                    size=(height, width),
                    mode="nearest",
                )[0, 0]
                if not torch.any(target_mask):
                    continue
                params = dynamic_parameters[batch_index, :, grid_y, grid_x]
                mask_logits = self._dynamic_mask(mask_features[batch_index], params)
                bce = F.binary_cross_entropy_with_logits(mask_logits, target_mask)
                probabilities = torch.sigmoid(mask_logits)
                intersection = (probabilities * target_mask).sum()
                dice = 1.0 - (2.0 * intersection + 1.0) / (
                    probabilities.sum() + target_mask.sum() + 1.0
                )
                mask_losses.append(
                    float(loss_config.get("mask_bce_weight", 1.0)) * bce
                    + float(loss_config.get("mask_dice_weight", 1.0)) * dice
                )

        center_loss = self._center_focal_loss(center_logits, center_targets)
        if mask_losses:
            mask_loss = torch.stack(mask_losses).mean()
        else:
            mask_loss = mask_features.sum() * 0.0 + dynamic_parameters.sum() * 0.0
        total = (
            float(loss_config.get("center_weight", 1.0)) * center_loss
            + mask_loss
        )
        return {"total": total, "center": center_loss, "mask": mask_loss}

    @torch.no_grad()
    def predict_instances(
        self,
        images: torch.Tensor,
        score_threshold: float = 0.05,
        max_instances: int | None = None,
        min_instance_size: int = 1,
    ) -> list[list[tuple[float, np.ndarray]]]:
        """Predict scored instance masks for each image in a batch."""
        outputs = self(images)
        center_scores = torch.sigmoid(outputs["center_logits"])
        pooled = F.max_pool2d(center_scores, kernel_size=3, stride=1, padding=1)
        is_peak = (center_scores >= pooled) & (center_scores >= score_threshold)
        limit = max_instances or self.max_instances
        results: list[list[tuple[float, np.ndarray]]] = []

        for batch_index in range(images.shape[0]):
            peak_indices = torch.nonzero(is_peak[batch_index, 0], as_tuple=False)
            if peak_indices.numel() == 0:
                results.append([])
                continue
            peak_scores = center_scores[batch_index, 0, peak_indices[:, 0], peak_indices[:, 1]]
            order = torch.argsort(peak_scores, descending=True)
            selected: list[tuple[int, int, float]] = []
            for position in order.tolist():
                y, x = peak_indices[position].tolist()
                if any(
                    (y - previous_y) ** 2 + (x - previous_x) ** 2
                    < self.nms_distance ** 2
                    for previous_y, previous_x, _ in selected
                ):
                    continue
                selected.append((y, x, float(peak_scores[position].item())))
                if len(selected) >= limit:
                    break

            image_results: list[tuple[float, np.ndarray]] = []
            for y, x, score in selected:
                parameters = outputs["dynamic_parameters"][batch_index, :, y, x]
                mask_logits = self._dynamic_mask(outputs["mask_features"][batch_index], parameters)
                mask_probability = torch.sigmoid(
                    F.interpolate(
                        mask_logits[None, None],
                        size=images.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]
                )
                binary_mask = (mask_probability >= self.mask_threshold).cpu().numpy()
                if int(binary_mask.sum()) >= min_instance_size:
                    image_results.append((score, binary_mask))
            results.append(image_results)
        return results
