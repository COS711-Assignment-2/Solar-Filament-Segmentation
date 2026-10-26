"""U-Net variant with a separate auxiliary boundary prediction head."""

from __future__ import annotations

import torch
from torch import nn

from src.models.unet import UNet


class BoundaryAwareUNet(UNet):
    """U-Net that predicts foreground and instance-boundary logits."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__(in_channels=in_channels, base_channels=base_channels)
        self.boundary_head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.forward_features(inputs)
        return {
            "segmentation": self.segmentation_head(features),
            "boundary": self.boundary_head(features),
        }
