import copy
import json
from argparse import Namespace

import pyhopper
import pytest
import torch
import yaml

from scripts import tune_boundary_aware_unet as tuner
from src.training.trainer import _initialize_model_weights


def _base_config():
    return {
        "data": {"splits_dir": "data/splits"},
        "model": {"name": "boundary_aware_unet", "base_channels": 32},
        "experiment": {"name": "boundary-aware-unet"},
        "loss": {
            "bce_weight": 1.0,
            "dice_weight": 1.0,
            "boundary_weight": 0.25,
        },
        "training": {
            "max_epochs": 100,
            "patience": None,
            "batch_size": 4,
            "learning_rate": 0.0003,
            "output_dir": "runs",
        },
        "evaluation": {"thresholds": [0.2, 0.5, 0.7]},
    }


def test_apply_parameters_keeps_architecture_and_balances_segmentation_loss():
    base = _base_config()
    tuned = tuner.apply_parameters(
        base,
        {
            "learning_rate": 0.0001,
            "boundary_weight": 0.2,
            "batch_size": 8,
            "bce_ratio": 0.65,
            "threshold": 0.45,
        },
        seed=43,
        max_epochs=10,
    )

    assert tuned["model"] == base["model"]
    assert tuned["data"] == base["data"]
    assert tuned["training"]["batch_size"] == 8
    assert tuned["training"]["learning_rate"] == 0.0001
    assert tuned["loss"]["bce_weight"] == 0.65
    assert tuned["loss"]["dice_weight"] == 0.35
    assert tuned["loss"]["boundary_weight"] == 0.2
    assert tuned["evaluation"]["thresholds"] == [0.45]
    assert base["evaluation"]["thresholds"] == [0.2, 0.5, 0.7]


def test_pyhopper_search_writes_best_config_without_running_training(
    tmp_path, monkeypatch
):
    base = _base_config()
    monkeypatch.setattr(tuner, "ROOT", tmp_path)
    monkeypatch.setattr(tuner, "load_config", lambda _path: copy.deepcopy(base))

    class FakeSearch:
        def __init__(self, space):
            assert set(space) == {
                "learning_rate",
                "boundary_weight",
                "batch_size",
                "bce_ratio",
                "threshold",
            }
            self.best_f = None

        def run(self, objective, **kwargs):
            assert kwargs["direction"] == "maximize"
            assert kwargs["runtime"] == "4h"
            assert kwargs["n_jobs"] == 1
            self.best_f = objective(
                {
                    "learning_rate": 0.0001,
                    "boundary_weight": 0.2,
                    "batch_size": 8,
                    "bce_ratio": 0.65,
                    "threshold": 0.45,
                }
            )
            checkpoint_path = kwargs["checkpoint_path"]
            open(checkpoint_path, "w", encoding="utf-8").close()
            return {
                "learning_rate": 0.0001,
                "boundary_weight": 0.2,
                "batch_size": 8,
                "bce_ratio": 0.65,
                "threshold": 0.45,
            }

    monkeypatch.setattr(pyhopper, "Search", FakeSearch)

    def fake_train(config, **kwargs):
        assert kwargs["save_checkpoints"] is False
        assert kwargs["max_epochs_override"] == 6
        assert kwargs["initialize_from"] == tmp_path / "warm-start.pt"
        assert config["training"]["batch_size"] in (4, 8)
        assert len(config["evaluation"]["thresholds"]) == 1
        return {"best_val_pq": config["evaluation"]["thresholds"][0]}

    monkeypatch.setattr(tuner, "train", fake_train)
    result = tuner.run_search(
        Namespace(
            config=tmp_path / "config.yaml",
            output_dir=tmp_path / "search",
            batch_sizes=[4, 8],
            seed=43,
            runtime="4h",
            trial_epochs=6,
            final_epochs=100,
            device="cpu",
            init_checkpoint=tmp_path / "warm-start.pt",
        )
    )

    search_dir = tmp_path / "search"
    saved_results = json.loads((search_dir / "best_params.json").read_text())
    saved_config = yaml.safe_load(
        (search_dir / "best_config.yaml").read_text()
    )
    assert result["best_validation_pq"] == saved_results["best_validation_pq"]
    assert saved_config["training"]["max_epochs"] == 100
    assert saved_config["training"]["patience"] is None
    assert saved_config["evaluation"]["thresholds"] == [
        result["best_params"]["threshold"]
    ]
    assert (search_dir / "pyhopper_search.ckpt").is_file()


def test_initialize_model_weights_loads_weights_without_optimizer_state(tmp_path):
    source = torch.nn.Linear(2, 1)
    checkpoint_path = tmp_path / "warm-start.pt"
    torch.save(
        {
            "model_state_dict": source.state_dict(),
            "config": {"model": {"name": "boundary_aware_unet"}},
            "optimizer_state_dict": {"stale": True},
            "epoch": 84,
        },
        checkpoint_path,
    )
    target = torch.nn.Linear(2, 1)

    _initialize_model_weights(target, checkpoint_path, "boundary_aware_unet")

    for source_parameter, target_parameter in zip(
        source.parameters(), target.parameters()
    ):
        assert torch.equal(source_parameter, target_parameter)


def test_initialize_model_weights_rejects_another_architecture(tmp_path):
    checkpoint_path = tmp_path / "other-model.pt"
    torch.save(
        {
            "model_state_dict": torch.nn.Linear(2, 1).state_dict(),
            "config": {"model": {"name": "unet"}},
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="does not match"):
        _initialize_model_weights(
            torch.nn.Linear(2, 1), checkpoint_path, "boundary_aware_unet"
        )
