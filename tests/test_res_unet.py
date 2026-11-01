"""Unit tests for ResUNet and advanced topological losses."""

import torch
import numpy as np

from src.models import create_model, ResUNet
from src.training.losses import soft_skeletonize, soft_cldice_loss, segmentation_loss


def test_res_unet_forward_backward():
    model = create_model("res_unet", in_channels=1, base_channels=16)
    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    assert out.shape == (2, 1, 64, 64)

    target = torch.ones_like(out)
    loss = segmentation_loss(
        out,
        target,
        bce_weight=1.0,
        dice_weight=1.0,
        cldice_weight=0.5,
        focal_gamma=2.0,
    )
    assert loss.item() > 0
    loss.backward()

    # Check gradients exist
    assert model.encoder1.conv1.weight.grad is not None


def test_soft_cldice_loss():
    probs = torch.sigmoid(torch.randn(2, 1, 32, 32))
    target = torch.randint(0, 2, (2, 1, 32, 32)).float()

    loss = soft_cldice_loss(probs, target)
    assert loss.ndim == 0
    assert 0.0 <= loss.item() <= 2.0
