"""Shared U-Net encoder/decoder and the standard U-Net model."""

from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Module):
    """Two convolutional layers with normalization and ReLU activations."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class UNet(nn.Module):
    """Standard binary U-Net."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        channels = [base_channels * (2 ** level) for level in range(4)]
        self.encoder1 = DoubleConv(in_channels, channels[0])
        self.encoder2 = DoubleConv(channels[0], channels[1])
        self.encoder3 = DoubleConv(channels[1], channels[2])
        self.encoder4 = DoubleConv(channels[2], channels[3])
        self.bottleneck = DoubleConv(channels[3], channels[3] * 2)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.decoder4 = self._decoder(channels[3] * 2, channels[3], channels[3])
        self.decoder3 = self._decoder(channels[3], channels[2], channels[2])
        self.decoder2 = self._decoder(channels[2], channels[1], channels[1])
        self.decoder1 = self._decoder(channels[1], channels[0], channels[0])
        self.segmentation_head = nn.Conv2d(channels[0], 1, kernel_size=1)

    @staticmethod
    def _decoder(in_channels: int, skip_channels: int, out_channels: int) -> nn.Module:
        return DoubleConv(in_channels + skip_channels, out_channels)

    @staticmethod
    def _merge(
        inputs: torch.Tensor, skip: torch.Tensor, decoder: nn.Module
    ) -> torch.Tensor:
        inputs = nn.functional.interpolate(
            inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        return decoder(torch.cat((skip, inputs), dim=1))

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        skip1 = self.encoder1(inputs)
        skip2 = self.encoder2(self.pool(skip1))
        skip3 = self.encoder3(self.pool(skip2))
        skip4 = self.encoder4(self.pool(skip3))
        features = self.bottleneck(self.pool(skip4))
        features = self._merge(features, skip4, self.decoder4)
        features = self._merge(features, skip3, self.decoder3)
        features = self._merge(features, skip2, self.decoder2)
        return self._merge(features, skip1, self.decoder1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.segmentation_head(self.forward_features(inputs))
