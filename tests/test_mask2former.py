"""Tests for the Mask2Former instance-model adapter."""

import sys
import types
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from src.models import Mask2Former, MODEL_REGISTRY, create_model


class FakeMask2Former(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            num_queries=10, num_labels=1, no_object_weight=0.1
        )
        self.loss_inputs = None
        self.unlabeled_batch_sizes = []
        self.class_logits_offset = nn.Parameter(torch.zeros(1, 2, 2))

    def forward(self, pixel_values, mask_labels=None, class_labels=None):
        if mask_labels is not None:
            self.loss_inputs = (pixel_values, mask_labels, class_labels)
            return SimpleNamespace(loss=pixel_values.sum() * 0 + 1)
        self.unlabeled_batch_sizes.append(len(pixel_values))
        class_logits = torch.tensor(
            [[[1.386, -0.693], [-2.303, 2.079]]],
            dtype=pixel_values.dtype,
            device=pixel_values.device,
        ).expand(len(pixel_values), -1, -1) + self.class_logits_offset
        mask_logits = torch.full(
            (len(pixel_values), 2, 8, 8),
            -2.0,
            dtype=pixel_values.dtype,
            device=pixel_values.device,
        )
        mask_logits[:, 0, 1:4, 2:5] = 2.0
        return SimpleNamespace(
            class_queries_logits=class_logits,
            masks_queries_logits=mask_logits,
        )


def install_fake_transformers(monkeypatch):
    module = types.ModuleType("transformers")
    module.Mask2FormerConfig = SimpleNamespace()
    module.Mask2FormerForUniversalSegmentation = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: FakeMask2Former()
    )
    monkeypatch.setitem(sys.modules, "transformers", module)


def test_mask2former_is_registered_and_adapts_instance_targets(monkeypatch):
    install_fake_transformers(monkeypatch)
    model = create_model("mask2former")
    assert isinstance(model, Mask2Former)
    assert "mask2former" in MODEL_REGISTRY

    images = torch.zeros((2, 1, 8, 8))
    targets = torch.zeros((2, 8, 8), dtype=torch.long)
    targets[0, 2:6, 3:7] = 4
    targets[0, 5:7, 6:8] = 9

    losses = model.compute_loss(images, targets, {})

    assert losses["total"].item() > 1
    pixel_values, mask_labels, class_labels = model.model.loss_inputs
    assert pixel_values.shape == (1, 3, 8, 8)
    assert len(mask_labels) == len(class_labels) == 1
    assert mask_labels[0].shape == (2, 8, 8)
    assert class_labels[0].tolist() == [0, 0]
    assert model.model.unlabeled_batch_sizes == [1]


def test_mask2former_empty_tiles_train_as_background_without_matching(monkeypatch):
    install_fake_transformers(monkeypatch)
    model = Mask2Former()
    images = torch.zeros((2, 1, 8, 8))
    targets = torch.zeros((2, 8, 8), dtype=torch.long)

    loss = model.compute_loss(images, targets, {})["total"]
    loss.backward()

    assert torch.isfinite(loss)
    assert model.model.loss_inputs is None
    assert model.model.unlabeled_batch_sizes == [2]
    assert model.model.class_logits_offset.grad is not None


def test_mask2former_predicts_scored_instance_masks(monkeypatch):
    install_fake_transformers(monkeypatch)
    model = Mask2Former()
    mark_steps = []

    predictions = model.predict_instances(
        torch.zeros((2, 1, 8, 8)),
        score_threshold=0.3,
        max_instances=2,
        min_instance_size=4,
        mark_step=lambda: mark_steps.append(True),
    )

    assert mark_steps == [True]
    assert len(predictions) == 2
    assert 0.7 < predictions[0][0][0] < 0.9
    assert predictions[0][0][1].dtype == np.bool_
    assert int(predictions[0][0][1].sum()) == 9
