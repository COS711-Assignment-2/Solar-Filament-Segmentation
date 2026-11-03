"""Residual U-Net that predicts foreground and local instance affinities."""

from __future__ import annotations

import torch
from torch import nn

from src.models.res_unet import ResUNet


class AffinityResUNet(ResUNet):
    """Predict foreground plus right/down same-instance edge probabilities."""

    is_affinity_model = True

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 48,
    ) -> None:
        super().__init__(in_channels=in_channels, base_channels=base_channels)
        self.affinity_head = nn.Conv2d(base_channels, 2, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.forward_features(inputs)
        return {
            "segmentation": self.segmentation_head(features),
            "affinity": self.affinity_head(features),
        }
