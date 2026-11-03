"""Residual U-Net that predicts foreground and per-pixel instance embeddings."""

from __future__ import annotations

import torch
from torch import nn

from src.models.res_unet import ResUNet


class EmbeddingResUNet(ResUNet):
    """Predict foreground logits and bounded features for instance clustering."""

    is_embedding_model = True

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 48,
        embedding_dim: int = 8,
    ) -> None:
        if embedding_dim < 2:
            raise ValueError("embedding_dim must be at least 2")
        super().__init__(in_channels=in_channels, base_channels=base_channels)
        self.embedding_dim = embedding_dim
        self.embedding_head = nn.Conv2d(base_channels, embedding_dim, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.forward_features(inputs)
        return {
            "segmentation": self.segmentation_head(features),
            "embedding": torch.tanh(self.embedding_head(features)),
        }
