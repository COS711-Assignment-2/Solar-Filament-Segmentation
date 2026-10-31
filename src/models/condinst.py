"""Compact CondInst-style instance segmentation model.

This implementation uses a shared multi-scale feature map, an anchor-free
center controller, and per-instance dynamic convolution mask heads. It is
purpose-built for the project's grayscale images and raster instance targets;
it does not depend on Detectron2.
"""

from __future__ import annotations

from typing import Any, Callable

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
    mask_stride = 2
    dynamic_hidden_channels = 8
    dynamic_mask_chunk_size = 16

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
        self.mask_lateral1 = nn.Conv2d(base_channels, mask_channels, 1)
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
        features1 = self.stem(images)
        features2 = self.stage2(features1)
        features3 = self.stage3(features2)
        features4 = self.stage4(features3)
        center_size = features2.shape[-2:]
        pyramid = self.mask_lateral1(features1)
        pyramid = pyramid + F.interpolate(
            self.lateral2(features2), size=features1.shape[-2:], mode="bilinear", align_corners=False
        )
        pyramid = pyramid + F.interpolate(
            self.lateral3(features3), size=features1.shape[-2:], mode="bilinear", align_corners=False
        )
        pyramid = pyramid + F.interpolate(
            self.lateral4(features4), size=features1.shape[-2:], mode="bilinear", align_corners=False
        )
        mask_features = self.mask_tower(pyramid)
        controller_features = self.controller_tower(
            F.interpolate(mask_features, size=center_size, mode="bilinear", align_corners=False)
        )
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

    @staticmethod
    def _draw_center_gaussian(
        target: torch.Tensor, center_y: int, center_x: int, radius: int
    ) -> None:
        """Draw a CenterNet-style soft target around an object center."""
        diameter = 2 * radius + 1
        coordinates = torch.arange(
            -radius, radius + 1, device=target.device, dtype=target.dtype
        )
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        sigma = diameter / 6.0
        gaussian = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma * sigma))
        height, width = target.shape
        y0, y1 = max(0, center_y - radius), min(height, center_y + radius + 1)
        x0, x1 = max(0, center_x - radius), min(width, center_x + radius + 1)
        gy0, gx0 = y0 - (center_y - radius), x0 - (center_x - radius)
        gy1, gx1 = gy0 + (y1 - y0), gx0 + (x1 - x0)
        target[y0:y1, x0:x1] = torch.maximum(
            target[y0:y1, x0:x1], gaussian[gy0:gy1, gx0:gx1]
        )

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

    def _dynamic_masks(
        self, features: torch.Tensor, parameters: torch.Tensor
    ) -> torch.Tensor:
        """Apply each instance's controller-generated mask network as grouped convolutions."""
        instance_count, channels, height, width = features.shape
        hidden = self.dynamic_hidden_channels
        parameter_groups = parameters.split(
            (
                hidden * channels,
                hidden,
                hidden * hidden,
                hidden,
                hidden,
                1,
            ),
            dim=1,
        )
        weights1, bias1, weights2, bias2, weights3, bias3 = parameter_groups

        result = F.conv2d(
            features.reshape(1, instance_count * channels, height, width),
            weights1.reshape(instance_count * hidden, channels, 1, 1),
            bias1.reshape(instance_count * hidden),
            groups=instance_count,
        )
        result = F.relu(result.reshape(instance_count, hidden, height, width))
        result = F.conv2d(
            result.reshape(1, instance_count * hidden, height, width),
            weights2.reshape(instance_count * hidden, hidden, 1, 1),
            bias2.reshape(instance_count * hidden),
            groups=instance_count,
        )
        result = F.relu(result.reshape(instance_count, hidden, height, width))
        result = F.conv2d(
            result.reshape(1, instance_count * hidden, height, width),
            weights3.reshape(instance_count, hidden, 1, 1),
            bias3.reshape(instance_count),
            groups=instance_count,
        )
        return result.reshape(instance_count, height, width)

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
        target_device = center_logits.device
        cpu_targets = instance_targets.detach().to(device="cpu")
        center_targets = torch.zeros(
            (batch_size, 1, height, width), dtype=center_logits.dtype
        )
        object_locations: list[tuple[int, int, int]] = []
        target_masks: list[torch.Tensor] = []

        for batch_index in range(batch_size):
            instance_ids = torch.unique(cpu_targets[batch_index])
            instance_ids = instance_ids[instance_ids > 0]
            occupied_centers = torch.zeros((height, width), dtype=torch.bool)
            for instance_id in instance_ids:
                rows, columns = torch.where(cpu_targets[batch_index] == instance_id)
                if rows.numel() == 0:
                    continue
                mean_y = rows.float().mean()
                mean_x = columns.float().mean()
                center_index = torch.argmin(
                    (rows.float() - mean_y).square()
                    + (columns.float() - mean_x).square()
                )
                center_y = int(rows[center_index].item())
                center_x = int(columns[center_index].item())
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
                object_height = int(rows.max() - rows.min() + 1)
                object_width = int(columns.max() - columns.min() + 1)
                radius = min(4, max(1, min(object_height, object_width) // 24))
                self._draw_center_gaussian(
                    center_targets[batch_index, 0], grid_y, grid_x, radius
                )

                target_mask = F.interpolate(
                    (cpu_targets[batch_index] == instance_id)
                    .float()[None, None],
                    size=mask_features.shape[-2:],
                    mode="nearest",
                )[0, 0]
                if not torch.any(target_mask):
                    continue
                object_locations.append((batch_index, grid_y, grid_x))
                target_masks.append(target_mask)

        center_loss = self._center_focal_loss(
            center_logits, center_targets.to(device=target_device)
        )
        if target_masks:
            locations = torch.tensor(object_locations, dtype=torch.long, device=target_device)
            batch_indices, center_ys, center_xs = locations.unbind(dim=1)
            parameter_map = dynamic_parameters.permute(0, 2, 3, 1)
            mask_parameters = parameter_map[batch_indices, center_ys, center_xs]
            mask_loss_sum = center_loss.new_zeros(())
            positive_weight = float(loss_config.get("mask_positive_weight", 4.0))
            bce_weight = float(loss_config.get("mask_bce_weight", 1.0))
            dice_weight = float(loss_config.get("mask_dice_weight", 1.0))
            for start in range(0, len(object_locations), self.dynamic_mask_chunk_size):
                end = min(start + self.dynamic_mask_chunk_size, len(object_locations))
                chunk_batch_indices = batch_indices[start:end]
                chunk_features = mask_features[chunk_batch_indices]
                chunk_parameters = mask_parameters[start:end]
                chunk_targets = torch.stack(target_masks[start:end]).to(
                    device=target_device
                )
                mask_logits = self._dynamic_masks(chunk_features, chunk_parameters)
                pixel_weights = 1.0 + (positive_weight - 1.0) * chunk_targets
                bce_map = F.binary_cross_entropy_with_logits(
                    mask_logits, chunk_targets, reduction="none"
                )
                bce = (bce_map * pixel_weights).sum(dim=(1, 2)) / (
                    pixel_weights.sum(dim=(1, 2)).clamp_min(1.0)
                )
                probabilities = torch.sigmoid(mask_logits)
                intersection = (probabilities * chunk_targets).sum(dim=(1, 2))
                dice = 1.0 - (2.0 * intersection + 1.0) / (
                    probabilities.sum(dim=(1, 2))
                    + chunk_targets.sum(dim=(1, 2))
                    + 1.0
                )
                mask_loss_sum = mask_loss_sum + (
                    bce_weight * bce + dice_weight * dice
                ).sum()
            mask_loss = mask_loss_sum / len(object_locations)
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
        mark_step: Callable[[], None] | None = None,
    ) -> list[list[tuple[float, np.ndarray]]]:
        """Predict scored instance masks for each image in a batch."""
        outputs = self(images)
        center_scores = torch.sigmoid(outputs["center_logits"])
        pooled = F.max_pool2d(center_scores, kernel_size=3, stride=1, padding=1)
        is_peak = (center_scores >= pooled) & (center_scores >= score_threshold)
        limit = max_instances or self.max_instances
        results: list[list[tuple[float, np.ndarray]]] = []

        if mark_step is not None:
            mark_step()
        peak_indices = torch.nonzero(is_peak[:, 0], as_tuple=False)
        peak_values = center_scores[:, 0][is_peak[:, 0]]
        peak_data = torch.cat(
            (peak_indices.to(dtype=peak_values.dtype), peak_values[:, None]), dim=1
        ).cpu().tolist()
        selected: list[tuple[int, int, int, float]] = []
        peaks_by_image: list[list[tuple[int, int, float]]] = [
            [] for _ in range(images.shape[0])
        ]
        for batch_index, y, x, score in peak_data:
            peaks_by_image[int(batch_index)].append((int(y), int(x), score))

        for batch_index, image_peaks in enumerate(peaks_by_image):
            image_peaks.sort(key=lambda peak: peak[2], reverse=True)
            selected_for_image: list[tuple[int, int, float]] = []
            for y, x, score in image_peaks:
                if any(
                    (y - previous_y) ** 2 + (x - previous_x) ** 2
                    < self.nms_distance**2
                    for previous_y, previous_x, _ in selected_for_image
                ):
                    continue
                selected_for_image.append((y, x, score))
                if len(selected_for_image) >= limit:
                    break
            selected.extend(
                (batch_index, y, x, score)
                for y, x, score in selected_for_image
            )
            results.append([])

        if not selected:
            return results

        locations = torch.tensor(
            [item[:3] for item in selected],
            dtype=torch.long,
            device=center_scores.device,
        )
        batch_indices, center_ys, center_xs = locations.unbind(dim=1)
        parameter_map = outputs["dynamic_parameters"].permute(0, 2, 3, 1)
        selected_parameters = parameter_map[batch_indices, center_ys, center_xs]
        selected_features = outputs["mask_features"][batch_indices]
        for start in range(0, len(selected), self.dynamic_mask_chunk_size):
            end = min(start + self.dynamic_mask_chunk_size, len(selected))
            chunk_features = selected_features[start:end]
            chunk_parameters = selected_parameters[start:end]
            valid_count = end - start
            padding = self.dynamic_mask_chunk_size - valid_count
            if padding:
                chunk_features = torch.cat(
                    (
                        chunk_features,
                        chunk_features.new_zeros(
                            (padding, *chunk_features.shape[1:])
                        ),
                    )
                )
                chunk_parameters = torch.cat(
                    (
                        chunk_parameters,
                        chunk_parameters.new_zeros(
                            (padding, chunk_parameters.shape[1])
                        ),
                    )
                )
            mask_logits = self._dynamic_masks(chunk_features, chunk_parameters)
            mask_probabilities = torch.sigmoid(
                F.interpolate(
                    mask_logits[:, None],
                    size=images.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )[:, 0]
            )
            if mark_step is not None:
                mark_step()
            binary_masks = (
                mask_probabilities[:valid_count] >= self.mask_threshold
            ).cpu().numpy()
            for selected_index in range(start, end):
                batch_index = selected[selected_index][0]
                binary_mask = binary_masks[selected_index - start]
                if int(binary_mask.sum()) >= min_instance_size:
                    results[batch_index].append(
                        (selected[selected_index][3], binary_mask)
                    )
        return results
