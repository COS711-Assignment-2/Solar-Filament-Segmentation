"""Tests for instance-embedding ResUNet and clustering."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models import EmbeddingResUNet, ResUNet, create_model
from src.training import trainer
from src.training.embedding_postprocessing import postprocess_embedding
from src.training.losses import instance_embedding_loss
from src.training.trainer import _initialize_model_weights, _run_epoch, load_config


def test_embedding_res_unet_predicts_foreground_and_embeddings():
    model = create_model("embedding_res_unet", base_channels=2, embedding_dim=4)
    output = model(torch.rand(1, 1, 32, 32))

    assert set(output) == {"segmentation", "embedding"}
    assert output["segmentation"].shape == (1, 1, 32, 32)
    assert output["embedding"].shape == (1, 4, 32, 32)


def test_embedding_loss_separates_instances_and_backpropagates():
    embeddings = torch.zeros(1, 3, 8, 8, requires_grad=True)
    instances = torch.zeros(1, 8, 8, dtype=torch.long)
    instances[:, 1:7, 1:4] = 1
    instances[:, 1:7, 4:7] = 2
    with torch.no_grad():
        embeddings[:, 0, :, 1:4] = -0.8
        embeddings[:, 0, :, 4:7] = 0.8

    loss = instance_embedding_loss(embeddings, instances, sample_count=64)

    assert loss.item() < 0.01
    loss.backward()
    assert embeddings.grad is not None


def test_embedding_res_unet_training_step_uses_instance_loss():
    model = EmbeddingResUNet(base_channels=2, embedding_dim=3)
    instances = torch.zeros(32, 32, dtype=torch.long)
    instances[8:24, 5:14] = 1
    instances[8:24, 14:23] = 2
    batch = {
        "image": torch.rand(1, 32, 32),
        "mask": (instances > 0).float().unsqueeze(0),
        "instances": instances,
    }
    config = {
        "loss": {
            "bce_weight": 1.0,
            "dice_weight": 1.0,
            "cldice_weight": 0.0,
            "focal_gamma": 0.0,
            "embedding_weight": 0.2,
            "embedding_sample_count": 32,
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


def test_embedding_clustering_splits_touching_instances():
    foreground = np.ones((8, 12), dtype=np.float32)
    embeddings = np.zeros((2, 8, 12), dtype=np.float32)
    embeddings[0, :, :6] = -0.8
    embeddings[0, :, 6:] = 0.8

    labels = postprocess_embedding(foreground, embeddings, 0.5, 0.4)

    assert len(np.unique(labels)) == 2
    assert np.all(labels[:, :6] == labels[0, 0])
    assert np.all(labels[:, 6:] == labels[0, 6])
    assert labels[0, 0] != labels[0, 6]


def test_embedding_validation_uses_embedding_clusterer(monkeypatch):
    class DummyEmbeddingModel:
        is_embedding_model = True
        embedding_dim = 2

        def eval(self):
            return self

    target = np.zeros((8, 12), dtype=np.int32)
    target[:, :6] = 1
    target[:, 6:] = 2
    foreground = (target > 0).astype(np.float32)
    embeddings = np.zeros((2, 8, 12), dtype=np.float32)
    embeddings[0, :, :6] = -0.8
    embeddings[0, :, 6:] = 0.8
    monkeypatch.setattr(
        trainer,
        "_predict_image",
        lambda *args: (foreground, target, None, embeddings),
    )
    config = {
        "evaluation": {
            "thresholds": [0.5],
            "embedding_cluster_radius": 0.4,
            "min_instance_size": 1,
        }
    }

    _, pq, _ = trainer.evaluate_thresholds(
        DummyEmbeddingModel(), ".", [{}], config, torch.device("cpu")
    )

    assert pq == 1.0


def test_embedding_res_unet_initializes_from_res_unet_checkpoint(tmp_path):
    source = ResUNet(base_channels=2)
    checkpoint_path = tmp_path / "best_model.pt"
    torch.save(
        {
            "model_state_dict": source.state_dict(),
            "config": {"model": {"name": "res_unet"}},
        },
        checkpoint_path,
    )
    target = EmbeddingResUNet(base_channels=2, embedding_dim=3)

    _initialize_model_weights(target, checkpoint_path, "embedding_res_unet")

    assert torch.equal(target.segmentation_head.weight, source.segmentation_head.weight)


def test_embedding_res_unet_config_is_valid():
    config = load_config("experiments/embedding_res_unet/config.yaml")

    assert config["model"]["name"] == "embedding_res_unet"
    assert config["data"]["resolution"] == 512
    assert config["evaluation"]["embedding_cluster_radius"] == 0.75
