"""Tests for instance-aware ResUNet prediction and affinity post-processing."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models import AffinityResUNet, ResUNet, create_model
from src.training.affinity_postprocessing import postprocess_affinity
from src.training.losses import instance_affinity_loss, instance_affinity_targets
from src.training import trainer
from src.training.trainer import _initialize_model_weights, _run_epoch, load_config


def test_affinity_res_unet_predicts_foreground_and_two_local_affinities():
    model = create_model("affinity_res_unet", base_channels=2)
    output = model(torch.rand(1, 1, 32, 32))

    assert set(output) == {"segmentation", "affinity"}
    assert output["segmentation"].shape == (1, 1, 32, 32)
    assert output["affinity"].shape == (1, 2, 32, 32)
    assert AffinityResUNet.is_affinity_model


def test_instance_affinity_targets_and_loss_distinguish_neighbor_instances():
    instances = torch.tensor([[[1, 1, 2], [1, 1, 2]]])
    targets, valid = instance_affinity_targets(instances)
    assert targets.shape == (1, 2, 2, 3)
    assert valid[:, 0, :, 0].all()
    assert torch.equal(targets[:, 0, :, 0], torch.ones((1, 2)))
    assert torch.equal(targets[:, 0, :, 1], torch.zeros((1, 2)))
    assert torch.isfinite(instance_affinity_loss(torch.zeros_like(targets), instances))


def test_affinity_res_unet_training_step_uses_instance_loss():
    model = AffinityResUNet(base_channels=2)
    batch = {
        "image": torch.rand(1, 32, 32),
        "mask": torch.zeros(1, 32, 32),
        "instances": torch.zeros(32, 32, dtype=torch.long),
    }
    batch["instances"][8:24, 5:14] = 1
    batch["instances"][8:24, 14:23] = 2
    batch["mask"] = (batch["instances"] > 0).float().unsqueeze(0)
    config = {
        "loss": {
            "bce_weight": 1.0,
            "dice_weight": 1.0,
            "cldice_weight": 0.0,
            "focal_gamma": 0.0,
            "affinity_weight": 0.5,
        }
    }

    loss = _run_epoch(
        model,
        DataLoader([batch], batch_size=1),
        torch.optim.Adam(model.parameters(), lr=1e-4),
        torch.device("cpu"),
        config,
    )

    assert np.isfinite(loss)


def test_affinity_postprocessing_splits_touching_instances():
    foreground = np.ones((3, 4), dtype=np.float32)
    affinities = np.ones((2, 3, 4), dtype=np.float32)
    affinities[0, :, 1] = 0.0

    labels = postprocess_affinity(foreground, affinities, 0.5, 0.5)

    assert len(np.unique(labels)) == 2
    assert np.all(labels[:, :2] == labels[0, 0])
    assert np.all(labels[:, 2:] == labels[0, 2])
    assert labels[0, 0] != labels[0, 2]


def test_affinity_validation_scores_local_instance_links(monkeypatch):
    class DummyAffinityModel:
        is_affinity_model = True

        def eval(self):
            return self

    target = np.zeros((3, 4), dtype=np.int32)
    target[:, :2] = 1
    target[:, 2:] = 2
    foreground = (target > 0).astype(np.float32)
    affinities = np.ones((2, 3, 4), dtype=np.float32)
    affinities[0, :, 1] = 0.0
    monkeypatch.setattr(
        trainer,
        "_predict_image",
        lambda *args: (foreground, target, affinities, None),
    )
    config = {
        "evaluation": {
            "thresholds": [0.5],
            "affinity_threshold": 0.5,
            "min_instance_size": 1,
        }
    }

    _, pq, _ = trainer.evaluate_thresholds(
        DummyAffinityModel(), ".", [{}], config, torch.device("cpu")
    )

    assert pq == 1.0


def test_affinity_res_unet_initializes_from_res_unet_checkpoint(tmp_path):
    source = ResUNet(base_channels=2)
    checkpoint_path = tmp_path / "best_model.pt"
    torch.save(
        {
            "model_state_dict": source.state_dict(),
            "config": {"model": {"name": "res_unet"}},
        },
        checkpoint_path,
    )
    target = AffinityResUNet(base_channels=2)

    _initialize_model_weights(target, checkpoint_path, "affinity_res_unet")

    assert torch.equal(
        target.segmentation_head.weight,
        source.segmentation_head.weight,
    )


def test_affinity_res_unet_experiment_config_is_valid():
    config = load_config("experiments/affinity_res_unet/config.yaml")

    assert config["model"]["name"] == "affinity_res_unet"
    assert config["data"]["resolution"] == 512
    assert config["evaluation"]["affinity_threshold"] == 0.5
