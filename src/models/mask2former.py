"""Hugging Face Mask2Former wrapper for grayscale filament instance masks."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Mask2Former(nn.Module):
    """Mask2Former with a Swin-T backbone and COCO-pretrained initialization."""

    is_instance_model = True
    default_model_name = "facebook/mask2former-swin-tiny-coco-instance"

    def __init__(
        self,
        pretrained_model_name: str = default_model_name,
        pretrained: bool = True,
        mask_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 < mask_threshold < 1.0:
            raise ValueError("mask_threshold must be strictly between 0 and 1")
        try:
            from transformers import Mask2FormerConfig
            from transformers import Mask2FormerForUniversalSegmentation
        except ImportError as error:
            raise ImportError(
                "Mask2Former requires transformers>=4.40,<5. Install it with "
                "`python3 -m pip install 'transformers>=4.40,<5'`."
            ) from error

        model_config = {
            "num_labels": 1,
            "id2label": {0: "filament"},
            "label2id": {"filament": 0},
        }
        if pretrained:
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                pretrained_model_name,
                **model_config,
                ignore_mismatched_sizes=True,
            )
        else:
            config = Mask2FormerConfig.from_pretrained(pretrained_model_name)
            for name, value in model_config.items():
                setattr(config, name, value)
            self.model = Mask2FormerForUniversalSegmentation(config)

        self.mask_threshold = mask_threshold
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def _prepare_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError(
                "Mask2Former expects grayscale images with shape (N, 1, H, W)"
            )
        rgb = images.expand(-1, 3, -1, -1)
        return (rgb - self.image_mean) / self.image_std

    def compute_loss(
        self,
        images: torch.Tensor,
        instance_targets: torch.Tensor,
        loss_config: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Convert raster instance IDs into the per-image labels Mask2Former expects."""
        del loss_config
        nonempty_indices = []
        empty_indices = []
        mask_labels = []
        class_labels = []
        for batch_index, target in enumerate(instance_targets.detach().cpu()):
            instance_ids = torch.unique(target)
            instance_ids = instance_ids[instance_ids > 0]
            if not len(instance_ids):
                empty_indices.append(batch_index)
                continue
            nonempty_indices.append(batch_index)
            masks = torch.stack(
                [target == instance_id for instance_id in instance_ids],
            ).to(device=images.device, dtype=torch.float32)
            mask_labels.append(masks)
            class_labels.append(
                torch.zeros(
                    len(instance_ids), dtype=torch.long, device=images.device
                )
            )

        total_loss = None
        if nonempty_indices:
            nonempty_images = (
                images
                if len(nonempty_indices) == len(images)
                else images[nonempty_indices]
            )
            outputs = self.model(
                pixel_values=self._prepare_images(nonempty_images),
                mask_labels=mask_labels,
                class_labels=class_labels,
            )
            total_loss = outputs.loss

        if empty_indices:
            empty_images = (
                images
                if len(empty_indices) == len(images)
                else images[empty_indices]
            )
            outputs = self.model(
                pixel_values=self._prepare_images(empty_images)
            )
            class_logits = outputs.class_queries_logits
            background_targets = torch.full(
                class_logits.shape[:2],
                fill_value=self.model.config.num_labels,
                dtype=torch.long,
                device=class_logits.device,
            )
            class_weights = torch.ones(
                self.model.config.num_labels + 1,
                dtype=class_logits.dtype,
                device=class_logits.device,
            )
            class_weights[-1] = self.model.config.no_object_weight
            background_loss = F.cross_entropy(
                class_logits.transpose(1, 2),
                background_targets,
                weight=class_weights,
            )
            total_loss = (
                background_loss if total_loss is None else total_loss + background_loss
            )

        if total_loss is None:
            raise ValueError("Mask2Former loss requires at least one training image")
        return {"total": total_loss}

    @torch.no_grad()
    def predict_instances(
        self,
        images: torch.Tensor,
        score_threshold: float = 0.05,
        max_instances: int | None = None,
        min_instance_size: int = 1,
        mark_step: Callable[[], None] | None = None,
    ) -> list[list[tuple[float, np.ndarray]]]:
        """Return scored per-instance masks using Mask2Former's query predictions."""
        outputs = self.model(pixel_values=self._prepare_images(images))
        if mark_step is not None:
            mark_step()
        height, width = images.shape[-2:]
        class_probabilities = outputs.class_queries_logits.softmax(dim=-1)[..., :-1]
        class_scores, class_ids = class_probabilities.max(dim=-1)
        mask_probabilities = F.interpolate(
            outputs.masks_queries_logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()
        limit = max_instances or int(self.model.config.num_queries)
        predictions: list[list[tuple[float, np.ndarray]]] = []
        for batch_index in range(len(images)):
            eligible = torch.nonzero(
                (class_scores[batch_index] >= score_threshold)
                & (class_ids[batch_index] == 0),
                as_tuple=False,
            ).flatten()
            eligible = eligible[
                torch.argsort(class_scores[batch_index, eligible], descending=True)
            ][:limit]
            image_predictions = []
            for query_index in eligible:
                mask = (
                    mask_probabilities[batch_index, query_index]
                    >= self.mask_threshold
                )
                if int(mask.sum().item()) < min_instance_size:
                    continue
                image_predictions.append(
                    (
                        float(class_scores[batch_index, query_index].item()),
                        mask.cpu().numpy(),
                    )
                )
            predictions.append(image_predictions)
        return predictions
