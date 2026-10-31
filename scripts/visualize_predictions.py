"""Create a qualitative validation comparison for the saved seed-43 models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import create_model
from src.training.dataset import load_validation_image, read_manifest, record_stem
from src.training.metrics import panoptic_quality
from src.training.postprocessing import postprocess_probability
from src.training.trainer import _label_instance_predictions


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = {
    "U-Net (seed 43)": Path("runs/unet/unet-baseline/seed43/best_model.pt"),
    "Boundary-aware U-Net (seed 43)": Path(
        "runs/boundary_aware_unet/boundary-aware-unet/seed43/best_model.pt"
    ),
    "CondInst (seed 43, epoch 100 run)": Path(
        "runs/condinst/condinst-comparison/seed43/best_model.pt"
    ),
}
METRICS = {
    "U-Net": Path("runs/unet/unet-baseline/seed43/metrics.json"),
    "Boundary-aware U-Net": Path(
        "runs/boundary_aware_unet/boundary-aware-unet/seed43/metrics.json"
    ),
    "CondInst": Path("runs/condinst/condinst-comparison/seed43/metrics.json"),
}
MODEL_COLORS = {
    "U-Net": "#2563eb",
    "Boundary-aware U-Net": "#059669",
    "CondInst": "#d97706",
}

PANEL_SIZE = 300
CELL_WIDTH = 330
CELL_HEIGHT = 355
MARGIN = 24
GAP = 12
COLORS = {
    "tp": np.array([40, 220, 100], dtype=np.uint8),
    "fp": np.array([255, 65, 65], dtype=np.uint8),
    "fn": np.array([40, 150, 255], dtype=np.uint8),
    "gt": np.array([40, 220, 100], dtype=np.uint8),
}


def _load_saved_model(path: Path) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint["config"]
    model_options = {
        key: value for key, value in config["model"].items() if key != "name"
    }
    model = create_model(config["model"]["name"], **model_options)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, checkpoint


def _choose_records(
    records: list[dict[str, str]], processed_dir: Path, count: int
) -> list[dict[str, str]]:
    stats = []
    for record in records:
        mask_path = (
            processed_dir
            / "val"
            / "masks"
            / "instances"
            / f"{record_stem(record['image_id'])}.png"
        )
        with Image.open(mask_path) as image:
            mask = np.asarray(image)
        instance_count = int(np.count_nonzero(np.unique(mask) > 0))
        stats.append((instance_count, record))
    stats.sort(key=lambda item: item[0])
    indices = np.linspace(0, len(stats) - 1, num=min(count, len(stats)), dtype=int)
    selected = []
    seen = set()
    for index in indices:
        record = stats[int(index)][1]
        if record["image_id"] not in seen:
            selected.append(record)
            seen.add(record["image_id"])
    return selected


def _predict(
    model: torch.nn.Module,
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    image: Image.Image,
    target: np.ndarray,
) -> tuple[np.ndarray, float]:
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    inputs = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)
    threshold = float(checkpoint["threshold"])
    evaluation = config["evaluation"]
    with torch.inference_mode():
        if getattr(model, "is_instance_model", False):
            candidates = model.predict_instances(
                inputs,
                score_threshold=min(float(value) for value in evaluation["thresholds"]),
                max_instances=int(config["model"].get("max_instances", 100)),
                min_instance_size=int(evaluation["min_instance_size"]),
            )[0]
            prediction = _label_instance_predictions(
                candidates,
                threshold,
                int(evaluation["min_instance_size"]),
                target.shape,
            )
        else:
            output = model(inputs)
            segmentation = (
                output["segmentation"] if isinstance(output, dict) else output
            )
            probability = torch.sigmoid(segmentation)[0, 0].cpu().numpy()
            prediction = postprocess_probability(
                probability,
                threshold,
                evaluation["postprocessing"],
                int(evaluation["min_instance_size"]),
                int(evaluation["watershed_min_distance"]),
            )
    return prediction, panoptic_quality(target, prediction).pq


def _overlay(
    image: Image.Image,
    target: np.ndarray,
    prediction: np.ndarray | None,
) -> Image.Image:
    base = ImageOps.autocontrast(image.convert("L"), cutoff=1).convert("RGB")
    pixels = np.asarray(base).copy()
    if prediction is None:
        pixels[target > 0] = (
            0.48 * pixels[target > 0] + 0.52 * COLORS["gt"]
        ).astype(np.uint8)
        return Image.fromarray(pixels)

    gt = target > 0
    pred = prediction > 0
    tp, fp, fn = gt & pred, ~gt & pred, gt & ~pred
    for mask, color in (
        (tp, COLORS["tp"]),
        (fp, COLORS["fp"]),
        (fn, COLORS["fn"]),
    ):
        pixels[mask] = (0.48 * pixels[mask] + 0.52 * color).astype(np.uint8)
    return Image.fromarray(pixels)


def _font(size: int) -> ImageFont.ImageFont:
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if font_path.is_file():
        return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def create_comparison(output_path: Path, sample_count: int = 3) -> None:
    loaded_models = {}
    for name, relative_path in CHECKPOINTS.items():
        path = ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Required model checkpoint is missing: {path}")
        loaded_models[name] = _load_saved_model(path)

    first_config = next(iter(loaded_models.values()))[1]
    processed_dir = ROOT / first_config["data"]["processed_dir"]
    records = read_manifest(ROOT / first_config["data"]["splits_dir"] / "val.csv")
    selected_records = _choose_records(records, processed_dir, sample_count)
    headers = ["Input", "Ground truth"] + list(loaded_models)
    width = MARGIN * 2 + len(headers) * CELL_WIDTH + (len(headers) - 1) * GAP
    height = 125 + len(selected_records) * (CELL_HEIGHT + GAP) + 115
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (MARGIN, 22),
        "Validation qualitative comparison — seed 43",
        fill="black",
        font=_font(24),
    )
    draw.text(
        (MARGIN, 62),
        "Green = correct foreground overlap   Red = false positive   Blue = missed ground truth",
        fill=(35, 35, 35),
        font=_font(15),
    )
    for column, header in enumerate(headers):
        x = MARGIN + column * (CELL_WIDTH + GAP)
        draw.text((x, 98), header, fill=(20, 20, 20), font=_font(15))

    for row, record in enumerate(selected_records):
        image, target = load_validation_image(
            processed_dir, record, first_config["data"]["resolution"]
        )
        y = 125 + row * (CELL_HEIGHT + GAP)
        panels = [
            ImageOps.autocontrast(image.convert("L"), cutoff=1).convert("RGB"),
            _overlay(image, target, None),
        ]
        scores: list[float | None] = [None, None]
        for model, config, checkpoint in loaded_models.values():
            prediction, pq = _predict(model, config, checkpoint, image, target)
            panels.append(_overlay(image, target, prediction))
            scores.append(pq)

        draw.text(
            (MARGIN, y),
            f"{record['image_id']}  |  {len(np.unique(target[target > 0]))} target instances",
            fill=(35, 35, 35),
            font=_font(13),
        )
        panel_y = y + 23
        for column, panel in enumerate(panels):
            x = MARGIN + column * (CELL_WIDTH + GAP)
            panel = panel.resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.BILINEAR)
            canvas.paste(panel, (x, panel_y))
            if column >= 2:
                score = scores[column]
                draw.text(
                    (x, panel_y + PANEL_SIZE + 5),
                    f"Image PQ: {score:.3f}" if score is not None else "",
                    fill=(20, 20, 20),
                    font=_font(13),
                )
        print(f"{record['image_id']}: {len(np.unique(target[target > 0]))} target instances")
        for name, score in zip(loaded_models, scores[2:]):
            print(f"  {name}: image PQ={score:.4f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"Saved qualitative comparison to {output_path}")


def create_analysis(output_dir: Path) -> None:
    """Plot training trends and best-checkpoint validation metrics for seed 43."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_by_model = {}
    for model_name, relative_path in METRICS.items():
        path = ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Training metrics are missing: {path}")
        metrics = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(metrics, dict) or not metrics.get("history"):
            raise ValueError(f"Training metrics have no history: {path}")
        metrics_by_model[model_name] = metrics

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for name, metrics in metrics_by_model.items():
        history = metrics["history"]
        epochs = [row["epoch"] for row in history]
        losses = np.asarray([row["train_loss"] for row in history], dtype=float)
        if losses[0] == 0:
            raise ValueError(f"Initial training loss is zero for {name}")
        axis.plot(
            epochs,
            losses / losses[0],
            color=MODEL_COLORS[name],
            linewidth=1.8,
            label=f"{name} (start = 1)",
        )
    axis.set(
        title="Seed 43 training-loss trend",
        xlabel="Epoch",
        ylabel="Training loss / first recorded loss",
    )
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.savefig(output_dir / "seed43_loss_trends.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for name, metrics in metrics_by_model.items():
        history = metrics["history"]
        epochs = [row["epoch"] for row in history]
        scores = [row["val_pq"] for row in history]
        best_epoch = metrics["best_epoch"]
        best_pq = metrics["best_val_pq"]
        axis.plot(
            epochs,
            scores,
            color=MODEL_COLORS[name],
            linewidth=1.8,
            label=name,
        )
        axis.scatter(
            [best_epoch],
            [best_pq],
            color=MODEL_COLORS[name],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
    axis.set(
        title="Seed 43 validation Panoptic Quality by epoch",
        xlabel="Epoch",
        ylabel="Validation PQ",
        ylim=(0, 1),
    )
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.savefig(output_dir / "seed43_validation_pq.png", dpi=160)
    plt.close(fig)

    names = list(metrics_by_model)
    best_pqs = [metrics_by_model[name]["best_val_pq"] for name in names]
    best_epochs = [metrics_by_model[name]["best_epoch"] for name in names]
    counts = {"TP": [], "FP": [], "FN": []}
    for name in names:
        metrics = metrics_by_model[name]
        best_epoch_row = next(
            (
                row for row in metrics["history"]
                if row["epoch"] == metrics["best_epoch"]
            ),
            None,
        )
        if best_epoch_row is None:
            raise ValueError(f"Best epoch is absent from {name} metric history")
        threshold_metrics = best_epoch_row["threshold_metrics"].get(
            str(metrics["best_threshold"])
        )
        if threshold_metrics is None:
            raise ValueError(f"Best-threshold metrics are absent for {name}")
        for key, label in (("tp", "TP"), ("fp", "FP"), ("fn", "FN")):
            counts[label].append(threshold_metrics[key])

    fig, (pq_axis, count_axis) = plt.subplots(
        2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [1, 1.4]},
        constrained_layout=True,
    )
    bars = pq_axis.bar(
        names, best_pqs, color=[MODEL_COLORS[name] for name in names]
    )
    pq_axis.set(
        title="Best validation metrics (seed 43)",
        ylabel="Best validation PQ",
        ylim=(0, 1),
    )
    pq_axis.grid(axis="y", alpha=0.25)
    for bar, score, epoch in zip(bars, best_pqs, best_epochs):
        pq_axis.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.02,
            f"{score:.3f} (epoch {epoch})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    x = np.arange(len(names))
    bar_width = 0.24
    for offset, label, color in (
        (-bar_width, "TP", "#16a34a"),
        (0, "FP", "#dc2626"),
        (bar_width, "FN", "#2563eb"),
    ):
        count_axis.bar(
            x + offset, counts[label], bar_width, label=label, color=color
        )
    count_axis.set_xticks(x, names)
    count_axis.set_ylabel("Instance count")
    count_axis.grid(axis="y", alpha=0.25)
    count_axis.legend()
    fig.savefig(output_dir / "seed43_best_instance_counts.png", dpi=160)
    plt.close(fig)

    print(f"Saved analysis graphs under {output_dir}")
    for name, metrics in metrics_by_model.items():
        print(
            f"{name}: best PQ={metrics['best_val_pq']:.4f} "
            f"at epoch {metrics['best_epoch']}, "
            f"threshold={metrics['best_threshold']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis",
        action="store_true",
        help="Plot seed-43 training trends and best-checkpoint comparison graphs.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=ROOT / "outputs/analysis",
        help="Directory for analysis graphs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/qualitative/validation_seed43.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of validation images to sample across instance-count density.",
    )
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1")
    if args.analysis:
        create_analysis(args.analysis_dir)
        return 0
    create_comparison(args.output, args.samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
