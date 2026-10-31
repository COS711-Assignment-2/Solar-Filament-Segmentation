"""Shared trainer for registered filament models."""

from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.models import MODEL_REGISTRY, create_model
from src.training.dataset import (
    FilamentDataset,
    RecordBatchSampler,
    crop_tile,
    load_validation_image,
    read_manifest,
    tile_origins,
)
from src.training.losses import boundary_target, segmentation_loss
from src.training.metrics import PQCounts, panoptic_quality
from src.training.postprocessing import postprocess_probability


def _validate_processed_records(
    processed_dir: str | Path,
    split: str,
    records: list[dict[str, str]],
) -> None:
    """Fail before training if the prepared images or instance masks are missing."""
    root = Path(processed_dir) / split
    missing: list[Path] = []
    for record in records:
        stem = "".join(
            char if char.isalnum() or char in "-_." else "_"
            for char in record["image_id"]
        )
        for path in (
            root / "images" / f"{stem}.png",
            root / "masks" / "instances" / f"{stem}.png",
        ):
            if not path.is_file():
                missing.append(path)
    if missing:
        examples = "\n".join(f"  - {path}" for path in missing[:5])
        more = f"\n  ... and {len(missing) - 5} more" if len(missing) > 5 else ""
        raise FileNotFoundError(
            f"Prepared {split} data is incomplete under '{root}'. Missing "
            f"{len(missing)} required file(s):\n{examples}{more}\n"
            "On a fresh Colab clone, raw/processed data is not included in Git. "
            "Run scripts/prepare_data.py --stage all with the downloaded "
            "--train-images and --annotations paths, then retry training."
        )


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config and validate experiment-critical options."""
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Training config must be a YAML mapping")
    for section in ("data", "model", "loss", "training", "evaluation", "experiment"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Training config must include a '{section}' mapping")
    model_name = config["model"].get("name")
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            "model.name must be one of: " + ", ".join(sorted(MODEL_REGISTRY))
        )
    experiment_name = config["experiment"].get("name")
    if (
        not isinstance(experiment_name, str)
        or not experiment_name.strip()
        or experiment_name in (".", "..")
        or "/" in experiment_name
        or "\\" in experiment_name
    ):
        raise ValueError("experiment.name must be a non-empty path-safe name")
    resolution = config["data"].get("resolution")
    if resolution not in (512, 1024, "tiled"):
        raise ValueError("data.resolution must be 512, 1024, or tiled")
    if config["evaluation"].get("postprocessing") not in (
        "connected_components",
        "watershed",
    ):
        raise ValueError(
            "evaluation.postprocessing must be connected_components or watershed"
        )
    max_epochs = config["training"].get("max_epochs")
    if not isinstance(max_epochs, int) or max_epochs < 1:
        raise ValueError("training.max_epochs must be a positive integer")
    patience = config["training"].get("patience")
    if patience is not None and (not isinstance(patience, int) or patience < 1):
        raise ValueError("training.patience must be null or a positive integer")
    device = config["training"].get("device", "auto")
    if device is not None and device not in ("auto", "cpu", "cuda", "tpu"):
        raise ValueError("training.device must be auto, cpu, cuda, or tpu")
    thresholds = config["evaluation"].get("thresholds", [])
    if not thresholds or any(not 0 < float(value) < 1 for value in thresholds):
        raise ValueError("evaluation.thresholds must contain values strictly between 0 and 1")
    return config


def _load_xla_model() -> Any | None:
    """Load the optional torch_xla model API when installed."""
    try:
        return importlib.import_module("torch_xla.core.xla_model")
    except ImportError:
        return None


def _tpu_device_access_error() -> str | None:
    """Check TPU device-node access before XLA initialization can abort."""
    device_nodes = sorted(glob.glob("/dev/accel*"))
    if not device_nodes:
        return "no TPU device nodes were found under /dev/accel*"
    inaccessible = []
    for path in device_nodes:
        if not os.access(path, os.R_OK | os.W_OK):
            inaccessible.append(path)
            continue
        try:
            descriptor = os.open(
                path, os.O_RDWR | getattr(os, "O_NONBLOCK", 0)
            )
        except OSError:
            inaccessible.append(path)
        else:
            os.close(descriptor)
    if inaccessible:
        return (
            "TPU device access denied for " + ", ".join(inaccessible)
            + "; check TPU VM/device permissions"
        )
    return None


def _configure_pjrt_device(requested: str) -> bool:
    """Select the PJRT TPU backend before torch_xla is imported."""
    configured = os.environ.get("PJRT_DEVICE", "").strip().upper()
    if not configured:
        os.environ["PJRT_DEVICE"] = "TPU"
        return True
    if configured == "TPU":
        return True
    if requested == "tpu":
        raise RuntimeError(
            f"TPU requested but PJRT_DEVICE={configured} is already configured. "
            "Unset PJRT_DEVICE or set it to TPU before launching training."
        )
    return False


def _select_device(
    requested: str | None,
    xla_loader: Any = _load_xla_model,
    tpu_access_check: Any = _tpu_device_access_error,
) -> tuple[torch.device, Any | None]:
    """Select CPU, CUDA, or TPU and return XLA helpers for TPU training."""
    requested = requested or "auto"
    if requested not in ("auto", "cpu", "cuda", "tpu"):
        raise ValueError("training.device must be auto, cpu, cuda, or tpu")
    if requested in ("auto", "tpu"):
        if not _configure_pjrt_device(requested):
            print(
                "TPU selection skipped because PJRT_DEVICE is explicitly "
                "configured for another backend."
            )
            xla_model = None
        else:
            access_error = tpu_access_check()
            if access_error:
                if requested == "tpu":
                    raise RuntimeError(
                        f"Cannot initialize TPU: {access_error}. Run on a TPU VM/runtime "
                        "with accelerator device permissions, or select --device cpu/cuda."
                    )
                print(f"TPU unavailable ({access_error}); selecting another device.")
                xla_model = None
            else:
                xla_model = xla_loader()
        if xla_model is None:
            if requested == "tpu":
                raise RuntimeError(
                    "TPU requested but torch_xla is not installed in this environment"
                )
        else:
            try:
                return xla_model.xla_device(), xla_model
            except RuntimeError as error:
                if requested == "tpu":
                    raise RuntimeError(
                        "TPU requested but torch_xla could not initialize an XLA device"
                    ) from error
                print(f"TPU unavailable ({error}); selecting another device.")
    if requested in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda"), None
    if requested == "cuda":
        raise RuntimeError("CUDA requested but CUDA is not available")
    return torch.device("cpu"), None


def _model_output(
    output: torch.Tensor | dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(output, dict):
        return output["segmentation"], output["boundary"]
    return output, None


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict[str, Any],
    xla_model: Any | None = None,
) -> float:
    model.train()
    losses = []
    loss_config = config["loss"]
    for batch in tqdm(loader, desc="Training batches", unit="batch", leave=False):
        images = batch["image"].to(device)
        instances = batch["instances"]
        optimizer.zero_grad(set_to_none=True)
        if hasattr(model, "compute_loss"):
            if not getattr(model, "is_instance_model", False):
                masks = batch["mask"].to(device)
                instances = instances.to(device)
            loss = model.compute_loss(images, instances, loss_config)["total"]
        else:
            masks = batch["mask"].to(device)
            instances = instances.to(device)
            segmentation, boundary = _model_output(model(images))
            loss = segmentation_loss(
                segmentation,
                masks,
                float(loss_config["bce_weight"]),
                float(loss_config["dice_weight"]),
            )
            if boundary is not None:
                boundary_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    boundary, boundary_target(instances)
                )
                loss = loss + float(loss_config["boundary_weight"]) * boundary_loss
        loss.backward()
        if xla_model is None:
            optimizer.step()
        else:
            xla_model.optimizer_step(optimizer, barrier=True)
        count = len(images)
        losses.append(loss.detach() * count)
    if not losses:
        raise ValueError("Training split produced no batches")
    return float(torch.stack(losses).sum().cpu().item() / len(loader.dataset))


def _predict_image(
    model: torch.nn.Module,
    processed_dir: str | Path,
    record: dict[str, str],
    config: dict[str, Any],
    device: torch.device,
    xla_model: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    resolution = config["data"]["resolution"]
    image, target = load_validation_image(processed_dir, record, resolution)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    if resolution == "tiled":
        tile_size = int(config["data"]["tile_size"])
        origins = tile_origins(image.width, image.height, tile_size)
    else:
        tile_size = int(resolution)
        origins = [(0, 0)]
    batch_size = int(config["training"]["batch_size"])
    probability = np.zeros(target.shape, dtype=np.float32)
    for start in range(0, len(origins), batch_size):
        batch_origins = origins[start:start + batch_size]
        tiles = np.stack([
            crop_tile(image_array, x, y, tile_size) for x, y in batch_origins
        ])
        inputs = torch.from_numpy(tiles).unsqueeze(1).to(device)
        with torch.no_grad():
            segmentation, _ = _model_output(model(inputs))
            if xla_model is not None:
                xla_model.mark_step()
            batch_probability = torch.sigmoid(segmentation).squeeze(1).cpu().numpy()
        for (x, y), tile_probability in zip(batch_origins, batch_probability):
            height = min(tile_size, target.shape[0] - y)
            width = min(tile_size, target.shape[1] - x)
            probability[y:y + height, x:x + width] = tile_probability[:height, :width]
    return probability, target


def _predict_instance_image(
    model: torch.nn.Module,
    processed_dir: str | Path,
    record: dict[str, str],
    config: dict[str, Any],
    device: torch.device,
    xla_model: Any | None = None,
) -> tuple[list[tuple[float, np.ndarray]], np.ndarray]:
    """Predict scored instance masks and place tiled results in image space."""
    resolution = config["data"]["resolution"]
    image, target = load_validation_image(processed_dir, record, resolution)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    if resolution == "tiled":
        tile_size = int(config["data"]["tile_size"])
        origins = tile_origins(image.width, image.height, tile_size)
    else:
        tile_size = int(resolution)
        origins = [(0, 0)]
    batch_size = int(config["training"]["batch_size"])
    minimum_score = min(float(value) for value in config["evaluation"]["thresholds"])
    min_instance_size = int(config["evaluation"]["min_instance_size"])
    max_instances = int(config["model"].get("max_instances", 100))
    candidates: list[tuple[float, np.ndarray]] = []

    for start in range(0, len(origins), batch_size):
        batch_origins = origins[start:start + batch_size]
        tiles = np.stack([
            crop_tile(image_array, x, y, tile_size) for x, y in batch_origins
        ])
        inputs = torch.from_numpy(tiles).unsqueeze(1).to(device)
        predictions = model.predict_instances(
            inputs,
            score_threshold=minimum_score,
            max_instances=max_instances,
            min_instance_size=min_instance_size,
            mark_step=xla_model.mark_step if xla_model is not None else None,
        )
        for (x, y), tile_predictions in zip(batch_origins, predictions):
            valid_height = min(tile_size, target.shape[0] - y)
            valid_width = min(tile_size, target.shape[1] - x)
            for score, tile_mask in tile_predictions:
                full_mask = np.zeros(target.shape, dtype=bool)
                full_mask[y:y + valid_height, x:x + valid_width] = tile_mask[
                    :valid_height, :valid_width
                ]
                candidates.append((float(score), full_mask))
    return candidates, target


def _label_instance_predictions(
    candidates: list[tuple[float, np.ndarray]],
    score_threshold: float,
    min_instance_size: int,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Resolve mask overlaps by confidence and produce a labeled instance map."""
    if not candidates:
        return np.zeros(output_shape, dtype=np.int32)
    labels = np.zeros(output_shape, dtype=np.int32)
    for score, mask in sorted(candidates, key=lambda candidate: candidate[0], reverse=True):
        if score < score_threshold:
            continue
        available = np.asarray(mask, dtype=bool) & (labels == 0)
        if int(available.sum()) < min_instance_size:
            continue
        labels[available] = int(labels.max()) + 1
    return labels


def evaluate_thresholds(
    model: torch.nn.Module,
    processed_dir: str | Path,
    records: list[dict[str, str]],
    config: dict[str, Any],
    device: torch.device,
    xla_model: Any | None = None,
) -> tuple[float, float, dict[str, dict[str, float | int]]]:
    """Choose the threshold with the highest dataset-level validation PQ."""
    counts = {
        float(threshold): PQCounts()
        for threshold in config["evaluation"]["thresholds"]
    }
    model.eval()
    for record in tqdm(records, desc="Validation images", unit="image", leave=False):
        if getattr(model, "is_instance_model", False):
            candidates, target = _predict_instance_image(
                model, processed_dir, record, config, device, xla_model
            )
            for threshold, aggregate in counts.items():
                prediction = _label_instance_predictions(
                    candidates,
                    threshold,
                    int(config["evaluation"]["min_instance_size"]),
                    target.shape,
                )
                aggregate.add(panoptic_quality(target, prediction))
        else:
            probability, target = _predict_image(
                model, processed_dir, record, config, device, xla_model
            )
            for threshold, aggregate in counts.items():
                prediction = postprocess_probability(
                    probability,
                    threshold,
                    config["evaluation"]["postprocessing"],
                    int(config["evaluation"]["min_instance_size"]),
                    int(config["evaluation"]["watershed_min_distance"]),
                )
                aggregate.add(panoptic_quality(target, prediction))
    details = {
        str(threshold): {
            "pq": aggregate.pq,
            "tp": aggregate.true_positives,
            "fp": aggregate.false_positives,
            "fn": aggregate.false_negatives,
            "matched_iou": aggregate.matched_iou,
        }
        for threshold, aggregate in counts.items()
    }
    best_threshold = max(
        counts,
        key=lambda value: (counts[value].pq, -abs(value - 0.5)),
    )
    return best_threshold, counts[best_threshold].pq, details


def should_stop_early(stale_epochs: int, patience: int | None) -> bool:
    """Return whether validation has failed to improve for the patience window."""
    return patience is not None and stale_epochs >= patience


def _capture_rng_state() -> dict[str, Any]:
    """Capture host RNG state so a saved run can continue reproducibly."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG state stored in a resumable checkpoint."""
    def as_tuple_tree(value: Any) -> Any:
        if isinstance(value, (tuple, list)):
            return tuple(as_tuple_tree(item) for item in value)
        return value

    random.setstate(as_tuple_tree(state["python"]))
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _save_checkpoint(
    checkpoint: dict[str, Any],
    path: Path,
    xla_model: Any | None,
) -> None:
    """Write checkpoints atomically so an interrupted save leaves the prior file intact."""
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        if xla_model is None:
            torch.save(checkpoint, temporary_path)
        else:
            xla_model.save(checkpoint, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def train(
    config: dict[str, Any],
    seed_override: int | None = None,
    output_tag: str | None = None,
    device_override: str | None = None,
    max_epochs_override: int | None = None,
    patience_override: int | None = None,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    """Train the selected architecture, optionally resuming a saved checkpoint."""
    if output_tag and (
        Path(output_tag).name != output_tag or output_tag in (".", "..")
    ):
        raise ValueError("output_tag must be a single path-safe folder name")
    if seed_override is not None:
        config["training"]["seed"] = seed_override
    if max_epochs_override is not None:
        if max_epochs_override < 1:
            raise ValueError("max_epochs_override must be a positive integer")
        config["training"]["max_epochs"] = max_epochs_override
    if patience_override is not None:
        if patience_override < 0:
            raise ValueError("patience_override must be zero or a positive integer")
        config["training"]["patience"] = patience_override or None
    if device_override is not None:
        config["training"]["device"] = device_override
    elif os.environ.get("GLS_DEVICE") and config["training"].get("device") == "auto":
        environment_device = os.environ["GLS_DEVICE"]
        config["training"]["device"] = (
            "tpu" if environment_device == "xla" else environment_device
        )
    seed = int(config["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device, xla_model = _select_device(config["training"].get("device"))
    print(f"Training device: {device}")
    data_config = config["data"]
    processed_dir = Path(data_config["processed_dir"])
    train_records = read_manifest(Path(data_config["splits_dir"]) / "train.csv")
    val_records = read_manifest(Path(data_config["splits_dir"]) / "val.csv")
    _validate_processed_records(processed_dir, "train", train_records)
    _validate_processed_records(processed_dir, "val", val_records)
    train_data = FilamentDataset(
        processed_dir,
        "train",
        train_records,
        data_config["resolution"],
        int(data_config["tile_size"]),
    )
    train_loader = DataLoader(
        train_data,
        batch_sampler=RecordBatchSampler(
            train_data, int(config["training"]["batch_size"])
        ),
        num_workers=int(config["training"]["workers"]),
        pin_memory=device.type == "cuda",
    )
    model_options = {
        key: value for key, value in config["model"].items() if key != "name"
    }
    model = create_model(config["model"]["name"], **model_options).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["training"]["learning_rate"])
    )
    output_dir = (
        Path(config["training"]["output_dir"])
        / config["model"]["name"]
        / config["experiment"]["name"]
        / (output_tag or f"seed{seed}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    best_pq = -1.0
    best_epoch = 0
    best_threshold = 0.5
    stale_epochs = 0
    history = []
    start_epoch = 1
    last_completed_epoch = 0
    if resume_from is not None:
        checkpoint_path = Path(resume_from)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict) or not isinstance(
            checkpoint.get("model_state_dict"), dict
        ):
            raise ValueError(f"Invalid training checkpoint: {checkpoint_path}")
        checkpoint_config = checkpoint.get("config", {})
        checkpoint_model = checkpoint_config.get("model", {}).get("name")
        if checkpoint_model != config["model"]["name"]:
            raise ValueError(
                f"Checkpoint model '{checkpoint_model}' does not match configured "
                f"model '{config['model']['name']}'"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        checkpoint_epoch = checkpoint.get("epoch")
        if not isinstance(checkpoint_epoch, int) or checkpoint_epoch < 1:
            raise ValueError(f"Checkpoint has an invalid epoch: {checkpoint_path}")
        start_epoch = checkpoint_epoch + 1
        last_completed_epoch = checkpoint_epoch
        best_epoch = int(checkpoint.get("best_epoch", checkpoint_epoch))
        best_pq = float(checkpoint.get("best_val_pq", checkpoint.get("val_pq", -1.0)))
        best_threshold = float(checkpoint.get("best_threshold", checkpoint.get("threshold", 0.5)))
        stale_epochs = int(checkpoint.get("stale_epochs", 0))
        history = checkpoint.get("history", [])
        if not isinstance(history, list):
            raise ValueError(f"Checkpoint history is invalid: {checkpoint_path}")
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            print(
                "Checkpoint contains model weights only; optimizer state will be "
                "reinitialized and this is a warm restart, not an exact resume."
            )
        if "rng_state" in checkpoint:
            _restore_rng_state(checkpoint["rng_state"])
        else:
            print("Checkpoint has no saved RNG state; using the configured seed.")
        if start_epoch > int(config["training"]["max_epochs"]):
            raise ValueError(
                f"Checkpoint is already at epoch {checkpoint_epoch}; increase "
                "training.max_epochs to continue."
            )
        print(
            f"Resuming from epoch {checkpoint_epoch}; continuing at epoch "
            f"{start_epoch}.",
            flush=True,
        )
    max_epochs = int(config["training"]["max_epochs"])
    patience = config["training"]["patience"]
    print(f"=== {config['experiment']['name']}, seed{seed} ===", flush=True)
    for epoch in range(start_epoch, max_epochs + 1):
        tqdm.write(f"Starting epoch {epoch}/{max_epochs}")
        train_loss = _run_epoch(
            model, train_loader, optimizer, device, config, xla_model
        )
        threshold, val_pq, threshold_metrics = evaluate_thresholds(
            model, processed_dir, val_records, config, device, xla_model
        )
        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_pq": val_pq,
            "best_threshold": threshold,
            "threshold_metrics": threshold_metrics,
        }
        history.append(epoch_result)
        improved = val_pq > best_pq
        if improved:
            best_pq = val_pq
            best_epoch = epoch
            best_threshold = threshold
            stale_epochs = 0
            best_checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_pq": val_pq,
                "threshold": threshold,
            }
            _save_checkpoint(
                best_checkpoint, output_dir / "best_model.pt", xla_model
            )
        else:
            stale_epochs += 1
        last_completed_epoch = epoch
        resumable_checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_pq": best_pq,
            "best_threshold": best_threshold,
            "stale_epochs": stale_epochs,
            "history": history,
            "rng_state": _capture_rng_state(),
        }
        _save_checkpoint(
            resumable_checkpoint, output_dir / "last_checkpoint.pt", xla_model
        )
        best_marker = " (best)" if improved else ""
        instance_counts = ""
        if getattr(model, "is_instance_model", False):
            counts = threshold_metrics[str(threshold)]
            instance_counts = (
                f" | TP={counts['tp']} FP={counts['fp']} FN={counts['fn']}"
            )
        tqdm.write(
            f"Epoch {epoch:03d}/{max_epochs:03d} | "
            f"train_loss={train_loss:.4f} | val_pq={val_pq:.4f} | "
            f"threshold={threshold:.3f}{best_marker}{instance_counts}"
        )
        if should_stop_early(stale_epochs, patience):
            print(
                f"Early stopping at epoch {epoch}: validation PQ did not improve "
                f"for {patience} consecutive epochs (best epoch {best_epoch}, "
                f"PQ={best_pq:.5f})."
            )
            break
    result = {
        "best_val_pq": best_pq,
        "best_epoch": best_epoch,
        "epochs_completed": last_completed_epoch,
        "best_threshold": best_threshold,
        "history": history,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--experiment",
        help="Experiment folder under experiments/ (contains config.yaml).",
    )
    selection.add_argument("--config", help="Explicit path to an experiment YAML config.")
    parser.add_argument("--seed", type=int, help="Override the configured random seed.")
    parser.add_argument(
        "--max-epochs",
        type=int,
        help="Override the experiment config's maximum epoch count.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        help="Override early-stopping patience; use 0 to disable early stopping.",
    )
    parser.add_argument(
        "--output-tag",
        help="Optional run label; otherwise the output folder is seed<seed>.",
    )
    parser.add_argument(
        "--resume-from",
        help="Resume from a last_checkpoint.pt or warm-start from best_model.pt.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "xla", "tpu"),
        help="Override device selection. GLS_DEVICE is also supported.",
    )
    args = parser.parse_args()
    if args.experiment and (
        Path(args.experiment).name != args.experiment
        or args.experiment in (".", "..")
    ):
        raise ValueError("--experiment must be a single folder name")
    if args.max_epochs is not None and args.max_epochs < 1:
        parser.error("--max-epochs must be at least 1")
    if args.patience is not None and args.patience < 0:
        parser.error("--patience must be zero or greater")
    config_path = (
        Path(args.config)
        if args.config
        else Path("experiments") / args.experiment / "config.yaml"
    )
    config = load_config(config_path)
    device_override = "tpu" if args.device == "xla" else args.device
    train(
        config,
        args.seed,
        args.output_tag,
        device_override,
        args.max_epochs,
        args.patience,
        args.resume_from,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
