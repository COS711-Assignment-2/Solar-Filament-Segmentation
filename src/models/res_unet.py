"""Residual U-Net with Squeeze-and-Excitation and Attention Gates.

Key Architectural Improvements for Solar Filaments:
1. Residual Blocks: Shortcut additions allow gradient flow through deep curvilinear features.
2. Squeeze-and-Excitation (SE): Adaptive channel weighting highlights filament contrast.
3. Attention Gates: Skip connections are gated by deeper decoder features to suppress
   bright background solar granulation and flare noise.
"""

from __future__ import annotations

import torch
from torch import nn


class SqueezeExcitation(nn.Module):
    """Channel-wise attention module (Hu et al.)."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        scale = self.fc(x).view(b, c, 1, 1)
        return x * scale


class ResBlock(nn.Module):
    """Two 3x3 convs with batchnorm, residual connection, and SE channel attention."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcitation(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.relu(out + residual)


class AttentionGate(nn.Module):
    """Gating module for skip connections (Oktay et al., Attention U-Net)."""

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = nn.functional.interpolate(
            self.W_g(gate), size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        x = self.W_x(skip)
        coeff = self.psi(self.relu(g + x))
        return skip * coeff


class ResUNet(nn.Module):
    """Deep Residual U-Net with Attention Gates for curvilinear filament segmentation."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 48,
    ) -> None:
        super().__init__()
        c = [base_channels * (2 ** i) for i in range(4)]  # e.g. [48, 96, 192, 384]
        bottleneck_c = c[3] * 2                           # e.g. 768

        # Encoder
        self.encoder1 = ResBlock(in_channels, c[0])
        self.encoder2 = ResBlock(c[0], c[1])
        self.encoder3 = ResBlock(c[1], c[2])
        self.encoder4 = ResBlock(c[2], c[3])
        self.pool = nn.MaxPool2d(kernel_size=2)

        # Bottleneck
        self.bottleneck = ResBlock(c[3], bottleneck_c)

        # Attention Gates
        self.att4 = AttentionGate(bottleneck_c, c[3], c[3] // 2)
        self.att3 = AttentionGate(c[3], c[2], c[2] // 2)
        self.att2 = AttentionGate(c[2], c[1], c[1] // 2)
        self.att1 = AttentionGate(c[1], c[0], c[0] // 2)

        # Decoders
        self.decoder4 = ResBlock(bottleneck_c + c[3], c[3])
        self.decoder3 = ResBlock(c[3] + c[2], c[2])
        self.decoder2 = ResBlock(c[2] + c[1], c[1])
        self.decoder1 = ResBlock(c[1] + c[0], c[0])

        # Segmentation Head
        self.segmentation_head = nn.Conv2d(c[0], 1, kernel_size=1)

    def _up_cat(
        self,
        inputs: torch.Tensor,
        skip: torch.Tensor,
        att_gate: AttentionGate,
        decoder: ResBlock,
    ) -> torch.Tensor:
        inputs = nn.functional.interpolate(
            inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        gated_skip = att_gate(inputs, skip)
        merged = torch.cat((gated_skip, inputs), dim=1)
        return decoder(merged)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Encoder
        s1 = self.encoder1(inputs)
        s2 = self.encoder2(self.pool(s1))
        s3 = self.encoder3(self.pool(s2))
        s4 = self.encoder4(self.pool(s3))

        # Bottleneck
        b = self.bottleneck(self.pool(s4))

        # Decoder with attention-gated skip connections
        d4 = self._up_cat(b, s4, self.att4, self.decoder4)
        d3 = self._up_cat(d4, s3, self.att3, self.decoder3)
        d2 = self._up_cat(d3, s2, self.att2, self.decoder2)
        d1 = self._up_cat(d2, s1, self.att1, self.decoder1)

        return self.segmentation_head(d1)
