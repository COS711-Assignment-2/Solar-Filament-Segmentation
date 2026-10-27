"""Tests for image preprocessing, segmentation losses, and instance PQ."""

import os

import numpy as np
import torch
from PIL import Image

from src.data.preprocess_images import preprocess_pair
from src.models import (
    BoundaryAwareUNet,
    CondInst,
    MODEL_REGISTRY,
    UNet,
    create_model,
    register_model,
)
from src.training import trainer as train
from src.training.losses import boundary_target, segmentation_loss
from src.training.metrics import panoptic_quality
from src.training.postprocessing import postprocess_probability


def test_preprocessing_keeps_image_separate_from_target_masks(tmp_path):
    image_path = tmp_path / "source.png"
    instance_path = tmp_path / "instances.png"
    category_path = tmp_path / "categories.png"
    output_dir = tmp_path / "images"
    Image.fromarray(np.full((8, 8), 127, dtype=np.uint8)).save(image_path)
    instances = np.zeros((8, 8), dtype=np.uint8)
    instances[2:6, 2:6] = 3
    Image.fromarray(instances).save(instance_path)
    Image.fromarray((instances > 0).astype(np.uint8)).save(category_path)

    result = preprocess_pair(
        image_path, instance_path, category_path, output_dir, size=(4, 4)
    )

    assert result["image"] == str(output_dir / "source.png")
    assert result["instances"] == str(instance_path)
    assert np.asarray(Image.open(result["image"])).shape == (4, 4)
    assert np.all(np.asarray(Image.open(result["image"])) == 127)
    assert set(np.asarray(Image.open(instance_path)).ravel()) <= {0, 3}
    assert Image.open(instance_path).size == (4, 4)
    assert Image.open(category_path).size == (4, 4)


def test_panoptic_quality_matches_instances_and_counts_missing_objects():
    target = np.array([[1, 1, 0], [0, 2, 2]], dtype=np.int32)
    exact = panoptic_quality(target, target)
    assert exact.pq == 1.0
    assert (exact.true_positives, exact.false_positives, exact.false_negatives) == (2, 0, 0)

    incomplete = panoptic_quality(target, np.array([[1, 1, 0], [0, 0, 0]]))
    assert incomplete.true_positives == 1
    assert incomplete.false_negatives == 1


def test_postprocessing_supports_both_configured_methods():
    probability = np.zeros((16, 16), dtype=np.float32)
    probability[3:13, 3:13] = 0.9
    for method in ("connected_components", "watershed"):
        instances = postprocess_probability(probability, 0.5, method)
        assert instances.shape == probability.shape
        assert set(np.unique(instances)) == {0, 1}


def test_boundary_head_target_and_weighted_loss():
    instances = torch.tensor([[[0, 1, 1], [0, 2, 2]]])
    boundary = boundary_target(instances)
    assert boundary.shape == (1, 1, 2, 3)
    assert boundary.sum() > 0

    logits = torch.zeros((1, 1, 2, 3))
    target = torch.zeros_like(logits)
    loss = segmentation_loss(logits, target, bce_weight=2.0, dice_weight=0.5)
    assert torch.isfinite(loss)


def test_boundary_aware_unet_returns_both_prediction_heads():
    model = BoundaryAwareUNet(base_channels=2)
    output = model(torch.zeros((1, 1, 32, 32)))
    assert set(output) == {"segmentation", "boundary"}
    assert output["segmentation"].shape == (1, 1, 32, 32)
    assert output["boundary"].shape == (1, 1, 32, 32)


def test_condinst_dynamic_masks_train_and_predict_instances():
    model = CondInst(base_channels=4, mask_channels=8, max_instances=3)
    images = torch.rand((1, 1, 32, 32))
    instances = torch.zeros((1, 32, 32), dtype=torch.long)
    instances[0, 5:18, 6:20] = 1
    instances[0, 20:29, 21:30] = 2

    losses = model.compute_loss(
        images,
        instances,
        {"center_weight": 1.0, "mask_bce_weight": 1.0, "mask_dice_weight": 1.0},
    )
    assert set(losses) == {"total", "center", "mask"}
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert model.controller.weight.grad is not None

    model.eval()
    predictions = model.predict_instances(
        images, score_threshold=0.0, max_instances=3, min_instance_size=1
    )
    assert len(predictions) == 1
    assert all(mask.shape == (32, 32) for _, mask in predictions[0])


def test_condinst_uses_higher_resolution_masks_and_soft_center_targets():
    model = CondInst(base_channels=4, mask_channels=8)
    output = model(torch.rand((1, 1, 32, 32)))
    assert output["mask_features"].shape[-2:] == (16, 16)
    assert output["center_logits"].shape[-2:] == (8, 8)

    center_target = torch.zeros((16, 16))
    model._draw_center_gaussian(center_target, 8, 8, radius=2)
    assert center_target[8, 8] == 1.0
    assert 0.0 < center_target[8, 9] < 1.0
    assert center_target[0, 0] == 0.0


def test_condinst_labeling_handles_empty_and_overlapping_predictions():
    empty = train._label_instance_predictions([], 0.2, 1, (4, 4))
    assert empty.shape == (4, 4)
    assert not empty.any()

    first = np.zeros((4, 4), dtype=bool)
    first[:3, :3] = True
    second = np.zeros((4, 4), dtype=bool)
    second[1:, 1:] = True
    labels = train._label_instance_predictions(
        [(0.8, first), (0.6, second)], 0.2, 1, (4, 4)
    )
    assert set(np.unique(labels)) == {0, 1, 2}
    assert labels[1, 1] == 1


def test_validation_threshold_is_selected_by_pq(monkeypatch):
    class DummyModel:
        def eval(self):
            return self

    target = np.zeros((8, 8), dtype=np.int32)
    target[2:4, 2:4] = 1
    probability = np.zeros((8, 8), dtype=np.float32)
    probability[2:4, 2:4] = 0.8
    probability[2, 4] = 0.5
    monkeypatch.setattr(
        train,
        "_predict_image",
        lambda *args: (probability, target),
    )
    config = {
        "evaluation": {
            "thresholds": [0.3, 0.7],
            "postprocessing": "connected_components",
            "min_instance_size": 1,
            "watershed_min_distance": 5,
        }
    }

    threshold, pq, _ = train.evaluate_thresholds(
        DummyModel(), ".", [{}], config, torch.device("cpu")
    )

    assert threshold == 0.7
    assert pq == 1.0


def test_condinst_validation_selects_center_score_threshold(monkeypatch):
    class DummyInstanceModel:
        is_instance_model = True

        def eval(self):
            return self

    target = np.zeros((8, 8), dtype=np.int32)
    target[2:5, 2:5] = 1
    predicted_mask = target > 0
    monkeypatch.setattr(
        train,
        "_predict_instance_image",
        lambda *args: ([(0.8, predicted_mask)], target),
    )
    config = {
        "evaluation": {
            "thresholds": [0.5, 0.9],
            "min_instance_size": 1,
        }
    }

    threshold, pq, _ = train.evaluate_thresholds(
        DummyInstanceModel(), ".", [{}], config, torch.device("cpu")
    )

    assert threshold == 0.5
    assert pq == 1.0


def test_device_selection_supports_cpu_cuda_and_tpu(monkeypatch):
    cpu_device, cpu_xla = train._select_device("cpu")
    assert cpu_device == torch.device("cpu")
    assert cpu_xla is None

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    cuda_device, cuda_xla = train._select_device("cuda")
    assert cuda_device == torch.device("cuda")
    assert cuda_xla is None

    class FakeXla:
        @staticmethod
        def xla_device():
            return torch.device("xla:0")

    monkeypatch.delenv("PJRT_DEVICE", raising=False)

    def fake_xla_loader():
        assert os.environ["PJRT_DEVICE"] == "TPU"
        return FakeXla

    tpu_device, tpu_xla = train._select_device(
        "tpu",
        xla_loader=fake_xla_loader,
        tpu_access_check=lambda: None,
    )
    assert tpu_device == torch.device("xla:0")
    assert tpu_xla is FakeXla


def test_auto_device_falls_back_when_xla_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device, xla_model = train._select_device(
        "auto",
        xla_loader=lambda: None,
        tpu_access_check=lambda: None,
    )
    assert device == torch.device("cpu")
    assert xla_model is None


def test_inaccessible_tpu_fails_without_initializing_xla(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("PJRT_DEVICE", raising=False)
    xla_loader_called = []

    try:
        train._select_device(
            "tpu",
            xla_loader=lambda: xla_loader_called.append(True),
            tpu_access_check=lambda: "TPU device access denied for /dev/accel1",
        )
    except RuntimeError as error:
        assert "/dev/accel1" in str(error)
        assert "accelerator device permissions" in str(error)
    else:
        raise AssertionError("Expected inaccessible TPU to fail before XLA init")

    assert xla_loader_called == []
    assert os.environ["PJRT_DEVICE"] == "TPU"


def test_auto_mode_falls_back_before_xla_initialization_when_tpu_inaccessible(
    monkeypatch,
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("PJRT_DEVICE", raising=False)
    xla_loader_called = []
    device, xla_model = train._select_device(
        "auto",
        xla_loader=lambda: xla_loader_called.append(True),
        tpu_access_check=lambda: "TPU device access denied for /dev/accel1",
    )
    assert device == torch.device("cpu")
    assert xla_model is None
    assert xla_loader_called == []
    assert os.environ["PJRT_DEVICE"] == "TPU"


def test_explicit_pjrt_backend_is_not_overridden(monkeypatch):
    monkeypatch.setenv("PJRT_DEVICE", "CPU")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    xla_loader_called = []
    device, xla_model = train._select_device(
        "auto",
        xla_loader=lambda: xla_loader_called.append(True),
        tpu_access_check=lambda: None,
    )
    assert device == torch.device("cpu")
    assert xla_model is None
    assert xla_loader_called == []
    assert os.environ["PJRT_DEVICE"] == "CPU"


def test_xla_training_step_flushes_each_optimizer_step():
    model = torch.nn.Conv2d(1, 1, kernel_size=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    barriers = []

    class FakeXla:
        @staticmethod
        def optimizer_step(optimizer, barrier=False):
            barriers.append(barrier)
            optimizer.step()

    batch = {
        "image": torch.ones((1, 1, 2, 2)),
        "mask": torch.ones((1, 1, 2, 2)),
        "instances": torch.ones((1, 2, 2), dtype=torch.long),
    }

    class FakeLoader:
        dataset = [batch]

        def __iter__(self):
            return iter([batch])

        def __len__(self):
            return 1

    loss = train._run_epoch(
        model,
        FakeLoader(),
        optimizer,
        torch.device("cpu"),
        {"loss": {"bce_weight": 1.0, "dice_weight": 1.0, "boundary_weight": 0.0}},
        FakeXla,
    )

    assert barriers == [True]
    assert np.isfinite(loss)


def test_models_are_separate_registered_architectures():
    assert set(MODEL_REGISTRY) >= {"condinst", "unet", "boundary_aware_unet"}
    assert isinstance(create_model("unet", base_channels=2), UNet)
    assert isinstance(create_model("boundary_aware_unet", base_channels=2), BoundaryAwareUNet)
    assert isinstance(
        create_model("condinst", base_channels=4, mask_channels=8), CondInst
    )

    class TestModel(torch.nn.Module):
        def __init__(self, channels=1):
            super().__init__()
            self.channels = channels

    register_model("test_model", TestModel)
    assert isinstance(create_model("test_model", channels=3), TestModel)
    assert create_model("test_model", channels=3).channels == 3


def test_model_and_experiment_names_validate_from_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
data:
  resolution: 512
model:
  name: unet
loss: {}
training:
  device: auto
  max_epochs: 10
  patience: 3
evaluation:
  postprocessing: connected_components
  thresholds: [0.5]
experiment:
  name: unet-baseline
""",
        encoding="utf-8",
    )
    config = train.load_config(config_path)
    assert config["model"]["name"] == "unet"

    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "name: unet-baseline", "name: ../shared"
        ),
        encoding="utf-8",
    )
    try:
        train.load_config(config_path)
    except ValueError as error:
        assert "path-safe" in str(error)
    else:
        raise AssertionError("Expected unsafe experiment names to be rejected")


def test_training_preflight_reports_missing_processed_data(tmp_path):
    try:
        train._validate_processed_records(
            tmp_path,
            "train",
            [{"image_id": "sample-1"}],
        )
    except FileNotFoundError as error:
        message = str(error)
        assert "data is incomplete" in message
        assert "prepare_data.py --stage all" in message
        assert "sample-1.png" in message
    else:
        raise AssertionError("Expected missing processed data to fail preflight")


def test_early_stopping_uses_validation_patience():
    assert train.should_stop_early(stale_epochs=4, patience=4)
    assert not train.should_stop_early(stale_epochs=3, patience=4)
    assert not train.should_stop_early(stale_epochs=100, patience=None)


def test_config_requires_max_epochs_and_valid_patience(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
data:
  resolution: 512
model:
  name: unet
loss: {}
training:
  device: auto
  max_epochs: 10
  patience: 3
evaluation:
  postprocessing: connected_components
  thresholds: [0.5]
experiment:
  name: unet-baseline
""",
        encoding="utf-8",
    )
    config = train.load_config(config_path)
    assert config["training"]["max_epochs"] == 10
    assert config["training"]["patience"] == 3

    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("patience: 3", "patience: 0"),
        encoding="utf-8",
    )
    try:
        train.load_config(config_path)
    except ValueError as error:
        assert "patience" in str(error)
    else:
        raise AssertionError("Expected invalid patience to be rejected")


def test_training_stops_after_validation_pq_patience(tmp_path, monkeypatch, capsys):
    processed_dir = tmp_path / "processed"
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    for split in ("train", "val"):
        (processed_dir / split / "images").mkdir(parents=True)
        (processed_dir / split / "masks" / "instances").mkdir(parents=True)
        image_id = f"{split}-sample"
        image = np.full((32, 32), 100, dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 8:24] = 1
        Image.fromarray(image).save(
            processed_dir / split / "images" / f"{image_id}.png"
        )
        Image.fromarray(mask).save(
            processed_dir / split / "masks" / "instances" / f"{image_id}.png"
        )
        (splits_dir / f"{split}.csv").write_text(
            f"image_id\n{image_id}\n", encoding="utf-8"
        )
    monkeypatch.setattr(
        train,
        "evaluate_thresholds",
        lambda *args: (0.5, 0.4, {"0.5": {"pq": 0.4}}),
    )
    config = {
        "data": {
            "processed_dir": str(processed_dir),
            "splits_dir": str(splits_dir),
            "resolution": "tiled",
            "tile_size": 32,
        },
        "model": {"name": "unet", "base_channels": 2},
        "experiment": {"name": "early-stop-check"},
        "loss": {"bce_weight": 1.0, "dice_weight": 1.0, "boundary_weight": 0.0},
        "training": {
            "max_epochs": 5,
            "patience": 2,
            "batch_size": 1,
            "workers": 0,
            "learning_rate": 0.001,
            "seed": 42,
            "device": "cpu",
            "output_dir": str(tmp_path / "runs"),
        },
        "evaluation": {
            "postprocessing": "connected_components",
            "thresholds": [0.5],
            "min_instance_size": 1,
            "watershed_min_distance": 5,
        },
    }

    result = train.train(config)

    assert result["best_epoch"] == 1
    assert result["epochs_completed"] == 3
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "=== early-stop-check, seed42 ===" in output
    assert "Starting epoch 1/5" in output
    assert "Epoch 001/005 | train_loss=" in output
    assert "val_pq=0.4000 | threshold=0.500 (best)" in output
    assert "Epoch 002/005 | train_loss=" in output
    assert "val_pq=0.4000 | threshold=0.500\n" in output
    run_dir = tmp_path / "runs" / "unet" / "early-stop-check" / "seed42"
    assert (run_dir / "best_model.pt").is_file()
    assert (run_dir / "metrics.json").is_file()


def test_cli_epoch_and_patience_overrides(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        train,
        "load_config",
        lambda path: {
            "training": {"max_epochs": 50, "patience": 10},
            "model": {"name": "unet"},
            "experiment": {"name": "test"},
        },
    )
    monkeypatch.setattr(
        train,
        "train",
        lambda config, *args: captured.update(
            config=config, args=args
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--experiment",
            "unet",
            "--max-epochs",
            "100",
            "--patience",
            "20",
            "--device",
            "xla",
        ],
    )

    train.main()

    assert captured["args"] == (None, None, "tpu", 100, 20)
