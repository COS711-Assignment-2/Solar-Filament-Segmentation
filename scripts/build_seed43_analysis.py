"""Build a shareable seed-43 model comparison and analysis bundle."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import visualize_predictions as visualizer


ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    {
        "key": "unet",
        "name": "U-Net",
        "method": "Semantic foreground prediction with connected-component postprocessing",
        "metrics": Path("runs/unet/unet-baseline/seed43/metrics.json"),
        "checkpoint": Path("runs/unet/unet-baseline/seed43/best_model.pt"),
    },
    {
        "key": "condinst",
        "name": "CondInst",
        "method": "Instance prediction with center and dynamic-mask heads",
        "metrics": Path("runs/condinst/condinst-comparison/seed43/metrics.json"),
        "checkpoint": Path("runs/condinst/condinst-comparison/seed43/best_model.pt"),
    },
    {
        "key": "boundary_aware_unet",
        "name": "Boundary-aware U-Net",
        "method": "Semantic foreground and auxiliary boundary prediction",
        "metrics": Path(
            "runs/boundary_aware_unet/boundary-aware-unet/seed43/metrics.json"
        ),
        "checkpoint": Path(
            "runs/boundary_aware_unet/boundary-aware-unet/seed43/best_model.pt"
        ),
    },
    {
        "key": "tuned_boundary_aware_unet",
        "name": "Tuned boundary-aware U-Net",
        "method": "Boundary-aware U-Net with PyHopper-tuned hyperparameters",
        "metrics": Path(
            "runs/boundary_aware_unet/boundary-aware-unet/"
            "pyhopper-warmstart-seed43/metrics.json"
        ),
        "checkpoint": Path(
            "runs/boundary_aware_unet/boundary-aware-unet/"
            "pyhopper-warmstart-seed43/best_model.pt"
        ),
    },
]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _best_counts(metrics: dict[str, Any]) -> dict[str, int | float]:
    best_row = next(
        (
            row for row in metrics["history"]
            if row["epoch"] == metrics["best_epoch"]
        ),
        None,
    )
    if best_row is None:
        raise ValueError("Best epoch is missing from training history")
    threshold = metrics["best_threshold"]
    threshold_key = str(threshold)
    threshold_metrics = best_row["threshold_metrics"].get(threshold_key)
    if threshold_metrics is None:
        threshold_key = f"{float(threshold):g}"
        threshold_metrics = best_row["threshold_metrics"].get(threshold_key)
    if threshold_metrics is None:
        raise ValueError(f"Metrics for selected threshold {threshold} are missing")
    return threshold_metrics


def _copy_run_data(output_dir: Path) -> list[dict[str, Any]]:
    source_dir = output_dir / "source_data"
    models_dir = source_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for run in RUNS:
        metrics_path = ROOT / run["metrics"]
        checkpoint_path = ROOT / run["checkpoint"]
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing run metrics: {metrics_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing best checkpoint: {checkpoint_path}")

        model_dir = models_dir / run["key"]
        model_dir.mkdir(parents=True, exist_ok=True)
        metrics = _read_json(metrics_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint.get("config")
        if not isinstance(config, dict):
            raise ValueError(f"Checkpoint has no saved config: {checkpoint_path}")
        if int(config["training"]["seed"]) != 43:
            raise ValueError(f"Expected seed 43 checkpoint: {checkpoint_path}")

        shutil.copy2(metrics_path, model_dir / "metrics.json")
        shutil.copy2(checkpoint_path, model_dir / "best_model.pt")
        (model_dir / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        counts = _best_counts(metrics)
        true_positives = int(counts["tp"])
        false_positives = int(counts["fp"])
        false_negatives = int(counts["fn"])
        matched_iou = float(counts["matched_iou"])
        recognition_quality = true_positives / (
            true_positives + 0.5 * false_positives + 0.5 * false_negatives
        )
        segmentation_quality = (
            matched_iou / true_positives if true_positives else 0.0
        )
        history = metrics["history"]
        summaries.append(
            {
                **run,
                "best_validation_pq": float(metrics["best_val_pq"]),
                "best_epoch": int(metrics["best_epoch"]),
                "best_threshold": float(metrics["best_threshold"]),
                "epochs_completed": int(metrics["epochs_completed"]),
                "train_loss_at_best": float(
                    next(row["train_loss"] for row in history
                         if row["epoch"] == metrics["best_epoch"])
                ),
                "final_validation_pq": float(history[-1]["val_pq"]),
                "tp": true_positives,
                "fp": false_positives,
                "fn": false_negatives,
                "recognition_quality": recognition_quality,
                "segmentation_quality": segmentation_quality,
                "metrics_data": metrics,
            }
        )

    tuning_dir = source_dir / "pyhopper_search"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    search_root = ROOT / "runs/boundary_aware_unet/boundary-aware-unet/pyhopper-seed43"
    for filename in ("best_params.json", "pyhopper_search.ckpt"):
        source = search_root / filename
        if not source.is_file():
            raise FileNotFoundError(f"Missing PyHopper artifact: {source}")
        shutil.copy2(source, tuning_dir / filename)
    trial_source = search_root / "trials"
    if trial_source.is_dir():
        shutil.copytree(
            trial_source,
            tuning_dir / "trials",
            dirs_exist_ok=True,
        )

    split_dir = source_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("train.csv", "val.csv"):
        source = ROOT / "data/splits" / filename
        if not source.is_file():
            raise FileNotFoundError(f"Missing data split manifest: {source}")
        shutil.copy2(source, split_dir / filename)

    (source_dir / "run_manifest.json").write_text(
        json.dumps(
            [
                {
                    "name": item["name"],
                    "key": item["key"],
                    "metrics": item["metrics"].as_posix(),
                    "checkpoint": item["checkpoint"].as_posix(),
                    "method": item["method"],
                    "seed": 43,
                }
                for item in summaries
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    return summaries


def _audit_prepared_inputs(output_dir: Path, sample_count: int) -> dict[str, Any]:
    processed_root = ROOT / "data/processed"
    raw_images_root = (
        ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026/train/train_images"
    )
    split_results: dict[str, Any] = {}
    representative_records: list[dict[str, str]] = []
    latest_processed_image_mtime = 0.0
    for split in ("train", "val"):
        manifest_path = ROOT / "data/splits" / f"{split}.csv"
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        low_range = 0
        identical_foreground_support = 0
        raw_dynamic_range = 0
        processed_max_values = []
        raw_max_values = []
        for record in records:
            stem = record["image_id"]
            processed_path = processed_root / split / "images" / f"{stem}.png"
            mask_path = (
                processed_root / split / "masks" / "instances" / f"{stem}.png"
            )
            source_path = raw_images_root / Path(record["file_name"]).name
            if not processed_path.is_file():
                raise FileNotFoundError(f"Processed input is missing: {processed_path}")
            if not mask_path.is_file():
                raise FileNotFoundError(f"Instance mask is missing: {mask_path}")
            if not source_path.is_file():
                raise FileNotFoundError(f"Raw source image is missing: {source_path}")

            latest_processed_image_mtime = max(
                latest_processed_image_mtime, processed_path.stat().st_mtime
            )
            processed = np.asarray(Image.open(processed_path).convert("L"))
            target = np.asarray(Image.open(mask_path))
            raw = np.asarray(Image.open(source_path).convert("L"))
            processed_max = int(processed.max())
            raw_max = int(raw.max())
            processed_max_values.append(processed_max)
            raw_max_values.append(raw_max)
            low_range += processed_max <= 5
            identical_foreground_support += np.array_equal(
                processed > 0, target > 0
            )
            raw_dynamic_range += raw_max > 5
        split_results[split] = {
            "image_count": len(records),
            "processed_max_at_most_5": low_range,
            "processed_foreground_support_identical_to_target": (
                identical_foreground_support
            ),
            "raw_source_max_above_5": raw_dynamic_range,
            "processed_max_value_range": [
                min(processed_max_values),
                max(processed_max_values),
            ],
            "raw_max_value_range": [min(raw_max_values), max(raw_max_values)],
        }
        if split == "val":
            representative_records = visualizer._choose_records(
                records, processed_root, sample_count
            )

    all_images = sum(item["image_count"] for item in split_results.values())
    low_range_count = sum(
        item["processed_max_at_most_5"] for item in split_results.values()
    )
    support_match_count = sum(
        item["processed_foreground_support_identical_to_target"]
        for item in split_results.values()
    )
    raw_signal_count = sum(
        item["raw_source_max_above_5"] for item in split_results.values()
    )
    checkpoint_mtimes = [
        (ROOT / run["checkpoint"]).stat().st_mtime for run in RUNS
    ]
    suspected_leakage = (
        all_images > 0
        and low_range_count == all_images
        and support_match_count == all_images
        and raw_signal_count == all_images
    )
    audit = {
        "status": (
            "suspected_target_leakage" if suspected_leakage else "passed"
        ),
        "total_images": all_images,
        "processed_inputs_max_at_most_5": low_range_count,
        "processed_foreground_support_identical_to_target": support_match_count,
        "raw_images_max_above_5": raw_signal_count,
        "latest_processed_image_mtime_utc": datetime.fromtimestamp(
            latest_processed_image_mtime, timezone.utc
        ).isoformat(),
        "earliest_selected_checkpoint_mtime_utc": datetime.fromtimestamp(
            min(checkpoint_mtimes), timezone.utc
        ).isoformat(),
        "splits": split_results,
        "interpretation": (
            "All processed model inputs have only a 0-3 intensity range and "
            "their foreground support exactly matches the instance target, while "
            "the raw source images retain normal grayscale range. Treat "
            "historical model scores as invalid until preprocessing is "
            "regenerated and all models are retrained."
            if suspected_leakage
            else "No dataset-wide mask-like input issue was detected by these checks."
        ),
    }
    (output_dir / "data_quality_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    _create_input_audit_figure(
        output_dir / "data_quality/processed_inputs_vs_targets.png",
        representative_records,
        processed_root,
        raw_images_root,
    )
    return audit


def _create_input_audit_figure(
    output_path: Path,
    records: list[dict[str, str]],
    processed_root: Path,
    raw_images_root: Path,
) -> None:
    panel_size = 300
    cell_width = 330
    cell_height = 355
    margin = 24
    gap = 12
    headers = ("Raw source image", "Processed model input", "Instance target")
    width = margin * 2 + len(headers) * cell_width + (len(headers) - 1) * gap
    height = 130 + len(records) * (cell_height + gap) + 32
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = (
        ImageFont.truetype(str(font_path), size=15)
        if font_path.is_file()
        else ImageFont.load_default()
    )
    draw.text(
        (margin, 20),
        "DATA QUALITY WARNING — processed images appear target-derived",
        fill=(170, 20, 20),
        font=font,
    )
    draw.text(
        (margin, 52),
        "Raw source vs. model input (contrast-stretched) vs. ground-truth instances",
        fill=(35, 35, 35),
        font=font,
    )
    for column, header in enumerate(headers):
        x = margin + column * (cell_width + gap)
        draw.text((x, 95), header, fill=(20, 20, 20), font=font)

    for row_index, record in enumerate(records):
        stem = record["image_id"]
        raw_path = raw_images_root / Path(record["file_name"]).name
        processed_path = processed_root / "val" / "images" / f"{stem}.png"
        target_path = (
            processed_root / "val" / "masks" / "instances" / f"{stem}.png"
        )
        raw = Image.open(raw_path).convert("L")
        processed = ImageOps.autocontrast(
            Image.open(processed_path).convert("L"), cutoff=0
        )
        target = np.asarray(Image.open(target_path))
        instance_view = visualizer._overlay_instances(raw, target)
        panels = (
            ImageOps.autocontrast(raw, cutoff=1).convert("RGB"),
            processed.convert("RGB"),
            instance_view,
        )
        y = 130 + row_index * (cell_height + gap)
        count = int(np.count_nonzero(np.unique(target) > 0))
        draw.text(
            (margin, y),
            f"{stem} | {count} target instances",
            fill=(35, 35, 35),
            font=font,
        )
        for column, panel in enumerate(panels):
            x = margin + column * (cell_width + gap)
            panel = panel.resize(
                (panel_size, panel_size), Image.Resampling.BILINEAR
            )
            canvas.paste(panel, (x, y + 22))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _write_summary_csv(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    columns = [
        "model",
        "method",
        "best_validation_pq",
        "best_epoch",
        "best_threshold",
        "epochs_completed",
        "final_validation_pq",
        "tp",
        "fp",
        "fn",
        "recognition_quality",
        "segmentation_quality",
        "train_loss_at_best",
    ]
    path = output_dir / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in summaries:
            writer.writerow(
                {"model": item["name"], **{key: item[key] for key in columns[1:]}}
            )


def _write_reports(
    output_dir: Path,
    summaries: list[dict[str, Any]],
    split_counts: dict[str, int],
    input_audit: dict[str, Any],
) -> None:
    baseline = summaries[0]
    boundary = summaries[2]
    tuned = summaries[3]
    condinst = summaries[1]
    search_params = _read_json(
        output_dir / "source_data/pyhopper_search/best_params.json"
    )
    params = search_params["best_params"]
    rows = "\n".join(
        "| {name} | {best_validation_pq:.4f} | {best_epoch} | "
        "{best_threshold:.2f} | {tp} | {fp} | {fn} |".format(**item)
        for item in summaries
    )
    boundary_delta = boundary["best_validation_pq"] - baseline["best_validation_pq"]
    tuned_delta = tuned["best_validation_pq"] - boundary["best_validation_pq"]
    data_invalid = input_audit["status"] == "suspected_target_leakage"
    warning = (
        "> **Critical data-quality warning — do not cite these scores as valid "
        "model performance.** The processed inputs for all "
        f"{input_audit['total_images']} train/validation images have maximum "
        "intensity at most 3, and their foreground support exactly matches their "
        "instance masks. The corresponding raw images have normal grayscale "
        "range. This is consistent with the earlier preprocessing overwrite "
        "issue documented in the project README. The run configs point to "
        "`data/processed`; the latest processed-image timestamp is "
        f"{input_audit['latest_processed_image_mtime_utc']} and the earliest "
        "selected checkpoint timestamp is "
        f"{input_audit['earliest_selected_checkpoint_mtime_utc']}, so the "
        "runs likely consumed these inputs; exact historical data "
        "snapshots were not saved. Regenerate processed data from raw files "
        "and retrain before drawing scientific conclusions.\n\n"
        if data_invalid
        else ""
    )
    report = f"""{warning}# Seed-43 Solar Filament Segmentation Analysis

## Scope and evaluation

This draft records exactly four seed-43 checkpoints: standard U-Net,
CondInst, boundary-aware U-Net, and the PyHopper-tuned boundary-aware U-Net
that was warm-started from the standard boundary-aware run. Seed-42 results
and the tuned configuration trained from scratch are intentionally excluded.

The four runs use the same validation split ({split_counts['val']} images;
training split: {split_counts['train']} images). Each model is scored at its
saved best validation epoch and selected threshold using panoptic quality
(PQ). The standard and boundary-aware U-Nets predict foreground and rely on
postprocessing to form instances; CondInst predicts instances directly.
Thresholds are selected per run and therefore differ.

## Quantitative results

The values below are **historical run outputs whose input provenance is
unverified and likely affected by mask-like processed data**. They are
retained for traceability only; do not cite them as valid performance.

| Model | Best validation PQ | Best epoch | Threshold | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
{rows}

Best PQ is the main comparison metric. TP/FP/FN count matched and unmatched
instances at each run's selected threshold. Recognition quality (RQ) and
segmentation quality (SQ), plus the final-epoch score, are included in
[`summary.csv`](summary.csv).

### U-Net vs. CondInst (historical, invalid inputs)

Historically, U-Net recorded PQ {baseline['best_validation_pq']:.4f}
versus CondInst at {condinst['best_validation_pq']:.4f}. CondInst has
{condinst['fp']} false positives and {condinst['fn']} missed instances at its
selected checkpoint/threshold, versus U-Net's {baseline['fp']} and
{baseline['fn']}. This is evidence that the semantic-foreground-plus-
postprocessing pipeline performed better in these runs; it does not establish
that semantic segmentation is generally superior to instance segmentation.
Architecture, optimization, thresholding, and run-history differences remain
possible contributors. The input corruption makes this comparison invalid
as evidence about segmentation paradigms.

### U-Net vs. boundary-aware U-Net (historical, invalid inputs)

Boundary-aware U-Net's best PQ is {boundary['best_validation_pq']:.4f}, an
absolute change of {boundary_delta:+.4f} ({boundary_delta * 100:+.2f}
percentage points) over standard U-Net. At its selected threshold it also
records {boundary['tp']} TP, {boundary['fp']} FP, and {boundary['fn']} FN,
compared with {baseline['tp']}, {baseline['fp']}, and {baseline['fn']} for
standard U-Net. The historical difference is modest, but the contaminated
inputs invalidate this comparison until the models are retrained on corrected
images. Once retrained, compare the instance-ID qualitative panels on
examples with touching structures to assess whether the boundary head
improves separation.

### Tuned boundary-aware U-Net (historical, invalid inputs)

PyHopper selected learning rate `{params['learning_rate']}`, boundary weight
`{params['boundary_weight']}`, batch size `{params['batch_size']}`, BCE ratio
`{params['bce_ratio']}` (Dice ratio `{1 - params['bce_ratio']:.2f}`), and
foreground threshold `{params['threshold']}`. The warm-started 100-epoch run
reaches PQ {tuned['best_validation_pq']:.4f} at epoch {tuned['best_epoch']},
an absolute change of {tuned_delta:+.4f} ({tuned_delta * 100:+.2f} percentage
points) over the untuned boundary-aware run. It ends at
{tuned['final_validation_pq']:.4f}; training loss decreases while validation
PQ falls after its early peak. This pattern is consistent with overfitting to
the validation objective, so the best checkpoint—not the final epoch—is the
appropriate checkpoint for this historical run only. The tuning search also
used the corrupted processed inputs, so its selected parameters and scores
must be re-evaluated after data repair.

## Training-loss curves

![Seed-43 normalized training loss curves; invalid mask-like inputs](plots/seed43_loss_trends.png)

Each curve is normalized to its own first logged loss, because loss functions
and scales differ between architectures. Historically, the U-Net and
boundary-aware U-Net losses fall sharply in the first few epochs and then
flatten; CondInst declines more gradually, with a pronounced one-epoch spike
around epoch 57; the tuned run declines steadily over the full 100 epochs.
This only describes optimization on the recorded inputs. It does not show
which model learned real solar morphology, and the losses must not be compared
as if they shared a scale.

## Validation PQ by epoch

![Seed-43 validation PQ by epoch; invalid mask-like inputs](plots/seed43_validation_pq.png)

Historically, the U-Net and boundary-aware U-Net curves start near 0.81,
quickly peak around 0.86, and drift toward 0.80. The tuned curve begins near
0.86, peaks at epoch 3, then gradually trends downward while remaining above
the other semantic-model curves. CondInst's recorded curve starts much lower
and rises toward 0.43 by epoch 97. These are patterns in the saved logs, not
reliable evidence of architecture performance: the audit below indicates the
models may have received target-derived inputs. The apparent early peaks and
late decline therefore cannot support an overfitting or convergence claim
until corrected data is used.

## Best-epoch PQ and instance counts

![Historical best validation PQ and TP, FP, FN counts](plots/seed43_best_instance_counts.png)

The top panel makes the historical PQ gap visible: the three U-Net variants
cluster around 0.86, while CondInst is around 0.43. In the lower panel,
CondInst also has far fewer matched instances and substantially more false
positives and false negatives. The two U-Net variants have very similar
counts. The tuned U-Net appears to reduce false positives slightly, but this
is not a valid model ranking because of the processed-data integrity failure.

## Input-data and qualitative audit

![Raw image, processed model input, and ground-truth instance mask](data_quality/processed_inputs_vs_targets.png)

The left column shows recognizable raw solar-disk imagery. The middle column,
which is what the training pipeline reads as the model input, contains only
the sparse filament-shaped foreground at intensity values 1-3. Its foreground
support exactly matches the target mask in every audited train and validation
image. This is consistent with the previously documented image-overwrite
issue, and makes target leakage likely.

The historical prediction visualization is included below **only to
illustrate why it is quarantined**. Its "input" panel is not a real solar
image; model predictions that resemble the ground truth under these
conditions are not valid qualitative evidence and must not be interpreted as
successful segmentation.

![Quarantined historical predictions; not valid qualitative evidence](data_quality/quarantined_invalid_predictions/seed43_instance_ids.png)

Because of this audit, these prediction panels are not offered as evidence
about the semantic-versus-instance question or boundary-based separation.
After data regeneration and retraining, replace them with qualitative
comparisons from the corrected images, especially examples with touching
filaments.

## Interpretation and limitations

1. The integrity audit found that all {input_audit['total_images']} images in
   the train and validation splits have processed intensity at most 3 and
   foreground support identical to the target mask, while raw
   images have grayscale range above 5. Treat all four historical scores and
   model comparisons as invalid until data is repaired and models are retrained.
2. Before the audit, standard U-Net and boundary-aware U-Net appeared to have
   close validation PQ; this cannot establish whether the boundary head helps
   separate connected filaments.
3. Before the audit, CondInst appeared to underperform the semantic models;
   this cannot establish an advantage for semantic segmentation.
4. Hyperparameter tuning and checkpoint selection used the validation set.
   The tuned score is therefore not an unbiased generalization estimate.
   Evaluate the selected checkpoint once on an untouched test set, and use
   multiple seeds for stronger comparative claims.
5. Training losses are not directly comparable between architectures.
6. CondInst's saved history is from its resumed training run; interpret its
   epoch curve in that context.

## Required next steps

Regenerate processed images/masks from the raw source while keeping the
existing split manifests fixed:

```bash
python3 scripts/prepare_data.py --stage process \\
  --train-images data/raw/MAGFiLO_1.0_Kaggle_2026/train/train_images \\
  --annotations data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \\
  --split-dir data/splits --processed-dir data/processed
python3 scripts/verify_prepared_data.py \\
  --split-dir data/splits --processed-dir data/processed
```

Then retrain the four comparison models (and rerun tuning against corrected
data) before replacing this draft's historical metrics. Keep architecture,
split, seed, and training protocol aligned; do not warm-start the corrected
tuned run from a checkpoint trained on the corrupted inputs.

## Packaged evidence

`source_data/models/` contains each selected run's metrics history, saved
configuration, and best checkpoint. `source_data/splits/` contains the train
and validation split manifests. `source_data/pyhopper_search/` contains the
search summary, checkpoint, and all recorded trial metrics. The tuned-from-
scratch run and seed-42 results are not included.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    short_report = f"""# Short report: seed-43 model comparison

> **Critical data-integrity warning:** all {input_audit['total_images']} processed
> train/validation inputs have maximum intensity at most 3 and foreground
> support identical to their target masks, while raw images retain normal
> grayscale range. The run configs use `data/processed`; the latest processed
> image predates the earliest selected checkpoint, so target leakage is likely,
> though exact training inputs
> were not snapshotted. Do not treat the historical metrics as valid.

The bundle contains historical seed-43 metrics for U-Net, CondInst,
boundary-aware U-Net, and tuned warm-started boundary-aware U-Net on the same
{split_counts['val']}-image validation split. It also contains metric plots,
source/config/checkpoint files, PyHopper trials, and a raw-versus-processed
data diagnostic. Model-prediction qualitative comparisons are withheld
because they would be misleading with mask-like inputs.

### Historical PQ curve (invalid inputs)

![Historical validation PQ curves](plots/seed43_validation_pq.png)

The U-Net variants appeared to peak early; CondInst's recorded PQ increased
through its run. Since input integrity failed, these curves are descriptive
of the saved logs only, not a valid model comparison.

### Input audit (reason for withholding model predictions)

![Raw versus processed input and target](data_quality/processed_inputs_vs_targets.png)

The processed input resembles the sparse target, rather than the raw solar
image. Therefore the historical prediction visuals are quarantined and are
not evidence of successful segmentation.

Regenerate processed data from raw files with the existing split manifests,
retrain all four models, rerun tuning, and evaluate on an untouched test set
before drawing conclusions about semantic versus instance segmentation,
boundary-based separation, or the tuned model.

See [`report.md`](report.md) for methods, figures, exact metrics, and
limitations. This bundle excludes seed 42 and the tuned-from-scratch run.
"""
    (output_dir / "short_report.md").write_text(short_report, encoding="utf-8")


def build(output_dir: Path, samples: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = _copy_run_data(output_dir)
    _write_summary_csv(output_dir, summaries)

    with (output_dir / "source_data/splits/train.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        train_records = list(csv.DictReader(handle))
    with (output_dir / "source_data/splits/val.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        val_records = list(csv.DictReader(handle))
    input_audit = _audit_prepared_inputs(output_dir, samples)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    (output_dir / "qualitative").mkdir(parents=True, exist_ok=True)
    warning_note = (
        "INVALID INPUTS — mask-like processed images"
        if input_audit["status"] == "suspected_target_leakage"
        else None
    )
    visualizer.create_analysis(output_dir / "plots", warning_note)
    if input_audit["status"] == "suspected_target_leakage":
        quarantine_dir = (
            output_dir / "data_quality/quarantined_invalid_predictions"
        )
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "seed43_foreground_overlap.png",
            "seed43_instance_ids.png",
            "sample_scores.txt",
        ):
            previous = output_dir / "qualitative" / name
            if previous.is_file():
                previous.replace(quarantine_dir / name)
        (output_dir / "qualitative/README.md").write_text(
            "Model-prediction qualitative panels are withheld because the "
            "processed model inputs appear target-derived. See "
            "`../data_quality_audit.json` and "
            "`../data_quality/processed_inputs_vs_targets.png`. Regenerate "
            "data and retrain before producing qualitative comparisons.\n",
            encoding="utf-8",
        )
    else:
        visualizer.create_comparison(
            output_dir / "qualitative/seed43_foreground_overlap.png",
            output_dir / "qualitative/seed43_instance_ids.png",
            samples,
        )

    split_counts = {"train": len(train_records), "val": len(val_records)}
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(
            {
                "seed": 43,
                "train_image_count": split_counts["train"],
                "validation_image_count": split_counts["val"],
                "training_manifest": "source_data/splits/train.csv",
                "validation_manifest": "source_data/splits/val.csv",
                "qualitative_samples": samples,
                "qualitative_selection": (
                    "Selected across sorted ground-truth instance-count density."
                ),
                "processed_input_integrity": input_audit["status"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_reports(output_dir, summaries, split_counts, input_audit)

    print(f"Built seed-43 analysis bundle at {output_dir}")
    print(
        f"Packaged exactly {len(summaries)} models; seed-42 and scratch run excluded."
    )
    print(f"Processed data audit: {input_audit['status']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/seed43_analysis"),
        help="Directory for metrics, plots, qualitative results, and reports.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=4,
        help="Validation examples to compare across instance-count density.",
    )
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least one")
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    build(output_dir.resolve(), args.samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
