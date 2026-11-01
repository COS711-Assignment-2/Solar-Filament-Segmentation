"""Build a four-seed analysis bundle: U-Net, CondInst, Boundary-aware U-Net, seeds 42-45.

All metrics come from checkpoints and the runner log. The pyhopper-tuned variant is
excluded. CondInst checkpoints are unavailable for seeds 42, 44, 45; those seeds use
completion lines from the runner log plus matching best-epoch log rows.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEEDS = (42, 43, 44, 45)
CONDINST_SEEDS = SEEDS
RUN_TAG = "corrected-data-live-seed{seed}"
OUTPUT_DIR = ROOT / "outputs/five_seed_analysis"
LOG_PATH = ROOT / "runs/parallel.log"
MODELS = (
    {
        "key": "unet",
        "name": "U-Net",
        "model_name": "unet",
        "experiment_name": "unet-baseline",
        "run_root": Path("runs/unet/unet-baseline"),
        "method": "Semantic foreground prediction with connected-component postprocessing.",
    },
    {
        "key": "condinst",
        "name": "CondInst",
        "model_name": "condinst",
        "experiment_name": "condinst-comparison",
        "run_root": Path("runs/condinst/condinst-comparison"),
        "method": "Instance prediction with center and dynamic-mask heads.",
    },
    {
        "key": "boundary_aware_unet",
        "name": "Boundary-aware U-Net",
        "model_name": "boundary_aware_unet",
        "experiment_name": "boundary-aware-unet",
        "run_root": Path("runs/boundary_aware_unet/boundary-aware-unet"),
        "method": "Semantic foreground prediction with an auxiliary boundary head.",
    },
)
COLORS = {
    "U-Net": "#2563eb",
    "CondInst": "#d97706",
    "Boundary-aware U-Net": "#059669",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CondInst log parsing
# ---------------------------------------------------------------------------


def _parse_condinst_log(path: Path) -> dict[int, dict[str, Any]]:
    """Extract best_val_pq and best_epoch for each seed from runner completion lines."""
    pattern = re.compile(
        r"finished condinst-comparison seed=(\d+), "
        r"best_val_pq=([0-9.]+), best_epoch=(\d+)"
    )
    results: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            seed = int(match.group(1))
            if seed not in CONDINST_SEEDS:
                continue
            if seed in results:
                raise ValueError(f"Duplicate CondInst completion for seed {seed}")
            results[seed] = {
                "seed": seed,
                "best_validation_pq": float(match.group(2)),
                "best_epoch": int(match.group(3)),
                "epochs_completed": 100,
                "best_threshold": None,
                "final_validation_pq": None,
                "source": "runner completion line in parallel.log",
            }
    missing = set(CONDINST_SEEDS) - set(results)
    if missing:
        raise ValueError(
            f"CondInst completion scores missing for seeds {sorted(missing)}"
        )
    return {seed: results[seed] for seed in CONDINST_SEEDS}


def _parse_condinst_epoch_log(path: Path) -> list[dict[str, Any]]:
    """Extract all CondInst per-epoch log rows."""
    pattern = re.compile(
        r"Epoch (\d{3})/100 \| train_loss=([0-9.]+) "
        r"\| val_pq=([0-9.]+) \| threshold=([0-9.]+)"
        r"( \(best\))? \| TP=(\d+) FP=(\d+) FN=(\d+)"
    )
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        match = pattern.search(line)
        if match:
            rows.append(
                {
                    "log_line": line_number,
                    "epoch": int(match.group(1)),
                    "train_loss": float(match.group(2)),
                    "val_pq": float(match.group(3)),
                    "threshold": float(match.group(4)),
                    "is_best": bool(match.group(5)),
                    "tp": int(match.group(6)),
                    "fp": int(match.group(7)),
                    "fn": int(match.group(8)),
                }
            )
    if not rows:
        raise ValueError(f"No CondInst epoch records found in {path}")
    return rows


# ---------------------------------------------------------------------------
# Load all seed runs
# ---------------------------------------------------------------------------


def _load_seed_runs() -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    runs_by_model: dict[str, list[dict[str, Any]]] = {}
    manifest: list[dict[str, Any]] = []
    model_data_dir = OUTPUT_DIR / "source_data/model_runs"
    condinst_completions = _parse_condinst_log(LOG_PATH)
    condinst_epoch_rows = _parse_condinst_epoch_log(LOG_PATH)

    for model in MODELS:
        model_runs = []

        if model["key"] == "condinst":
            for seed, record in condinst_completions.items():
                run = {**model, **record}
                # Find the unique log row matching this completion score at the best epoch
                best_rows = [
                    row
                    for row in condinst_epoch_rows
                    if row["epoch"] == record["best_epoch"]
                    and row["is_best"]
                    and round(row["val_pq"], 4) == round(record["best_validation_pq"], 4)
                ]
                if len(best_rows) != 1:
                    raise ValueError(
                        "Expected one CondInst log row matching the "
                        f"completion score for seed {seed}; found {len(best_rows)}"
                    )
                best_observation = best_rows[0]
                run.update(
                    best_epoch_observation=best_observation,
                    best_threshold=best_observation["threshold"],
                    train_loss_at_best=best_observation["train_loss"],
                    tp=best_observation["tp"],
                    fp=best_observation["fp"],
                    fn=best_observation["fn"],
                    source="runner completion line and matching best-epoch log row",
                )

                # If a metrics.json exists for this seed, load it and cross-check
                metrics_path = (
                    ROOT
                    / model["run_root"]
                    / RUN_TAG.format(seed=seed)
                    / "metrics.json"
                )
                if metrics_path.is_file():
                    metrics = _read_json(metrics_path)
                    if int(metrics.get("best_epoch", -1)) != record["best_epoch"] or abs(
                        float(metrics.get("best_val_pq", -1)) - record["best_validation_pq"]
                    ) > 1e-6:
                        raise ValueError(
                            "CondInst recovered metrics disagree with the "
                            f"completion log for seed {seed}"
                        )
                    history = metrics["history"]
                    best_row = next(
                        (r for r in history if r["epoch"] == record["best_epoch"]), None
                    )
                    if best_row is None:
                        raise ValueError(
                            f"CondInst recovered history missing best epoch for seed {seed}"
                        )
                    threshold_key = f"{float(metrics['best_threshold']):g}"
                    counts = best_row["threshold_metrics"].get(threshold_key)
                    if counts is None:
                        raise ValueError(
                            f"CondInst history missing best-threshold counts for seed {seed}"
                        )
                    run.update(
                        metrics=metrics,
                        epochs_completed=int(metrics["epochs_completed"]),
                        final_validation_pq=float(history[-1]["val_pq"]),
                        best_threshold=float(metrics["best_threshold"]),
                        train_loss_at_best=float(best_row["train_loss"]),
                        tp=int(counts["tp"]),
                        fp=int(counts["fp"]),
                        fn=int(counts["fn"]),
                        source="recovered metrics.json, matched to runner log",
                    )
                    recovered_dir = model_data_dir / "condinst" / f"seed{seed}"
                    recovered_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(metrics_path, recovered_dir / "metrics.json")

                # Copy checkpoint if it exists
                checkpoint_path = (
                    ROOT / model["run_root"] / RUN_TAG.format(seed=seed) / "best_model.pt"
                )
                if checkpoint_path.is_file():
                    recovered_dir = model_data_dir / "condinst" / f"seed{seed}"
                    recovered_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(checkpoint_path, recovered_dir / "best_model.pt")
                    manifest.append(
                        {
                            "model": model["name"],
                            "seed": seed,
                            "checkpoint_source": str(
                                checkpoint_path.relative_to(ROOT)
                            ),
                            "checkpoint_sha256": hashlib.sha256(
                                checkpoint_path.read_bytes()
                            ).hexdigest(),
                        }
                    )

                model_runs.append(run)

            config_source = ROOT / "experiments/condinst/config.yaml"
            condinst_dir = model_data_dir / "condinst"
            condinst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(config_source, condinst_dir / "experiment_config.yaml")
            runs_by_model[model["key"]] = model_runs
            continue

        # U-Net and Boundary-aware U-Net: require metrics.json + best_model.pt
        for seed in SEEDS:
            run_dir = ROOT / model["run_root"] / RUN_TAG.format(seed=seed)
            metrics_path = run_dir / "metrics.json"
            checkpoint_path = run_dir / "best_model.pt"
            if not metrics_path.is_file() or not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Missing completed {model['name']} seed {seed} artifacts "
                    f"under {run_dir}"
                )
            metrics = _read_json(metrics_path)
            if metrics.get("epochs_completed") != 100:
                raise ValueError(
                    f"{model['name']} seed {seed} completed "
                    f"{metrics.get('epochs_completed')} epochs, expected 100"
                )
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            config = checkpoint.get("config")
            if not isinstance(config, dict):
                raise ValueError(f"Checkpoint has no config: {checkpoint_path}")
            actual_model = config.get("model", {}).get("name")
            actual_experiment = config.get("experiment", {}).get("name")
            actual_seed = config.get("training", {}).get("seed")
            if (
                actual_model != model["model_name"]
                or actual_experiment != model["experiment_name"]
                or actual_seed != seed
            ):
                raise ValueError(
                    f"Checkpoint identity mismatch for {model['name']} seed {seed}: "
                    f"model={actual_model}, experiment={actual_experiment}, seed={actual_seed}"
                )
            copied_run = model_data_dir / model["key"] / f"seed{seed}"
            copied_run.mkdir(parents=True, exist_ok=True)
            shutil.copy2(metrics_path, copied_run / "metrics.json")
            shutil.copy2(checkpoint_path, copied_run / "best_model.pt")
            (copied_run / "config.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            manifest.append(
                {
                    "model": model["name"],
                    "seed": seed,
                    "checkpoint_source": str(checkpoint_path.relative_to(ROOT)),
                    "checkpoint_sha256": checkpoint_hash,
                }
            )
            history = metrics["history"]
            best_row = next(
                (row for row in history if row["epoch"] == metrics["best_epoch"]), None
            )
            if best_row is None:
                raise ValueError(
                    f"Best epoch missing in {model['name']} seed {seed} history"
                )
            threshold_key = f"{float(metrics['best_threshold']):g}"
            counts = best_row["threshold_metrics"].get(threshold_key)
            if counts is None:
                raise ValueError(
                    f"Best-threshold counts missing for {model['name']} seed {seed}"
                )
            model_runs.append(
                {
                    **model,
                    "seed": seed,
                    "best_validation_pq": float(metrics["best_val_pq"]),
                    "best_epoch": int(metrics["best_epoch"]),
                    "epochs_completed": int(metrics["epochs_completed"]),
                    "best_threshold": float(metrics["best_threshold"]),
                    "final_validation_pq": float(history[-1]["val_pq"]),
                    "train_loss_at_best": float(best_row["train_loss"]),
                    "tp": int(counts["tp"]),
                    "fp": int(counts["fp"]),
                    "fn": int(counts["fn"]),
                    "source": "metrics.json and identity-checked best checkpoint",
                    "metrics": metrics,
                    "config": config,
                    "checkpoint_path": checkpoint_path,
                }
            )
        runs_by_model[model["key"]] = model_runs

    return runs_by_model, manifest


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------


def _aggregate(
    runs_by_model: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    baseline = {run["seed"]: run["best_validation_pq"] for run in runs_by_model["unet"]}
    rows = []
    for model in MODELS:
        runs = runs_by_model[model["key"]]
        scores = np.asarray([run["best_validation_pq"] for run in runs])
        paired_runs = [run for run in runs if run["seed"] in baseline]
        paired = np.asarray(
            [run["best_validation_pq"] - baseline[run["seed"]] for run in paired_runs]
        )
        rows.append(
            {
                "model": model["name"],
                "n_seeds": len(runs),
                "n_paired_seeds_vs_unet": len(paired_runs),
                "mean_best_validation_pq": float(scores.mean()),
                "sample_sd_best_validation_pq": float(scores.std(ddof=1)),
                "min_best_validation_pq": float(scores.min()),
                "max_best_validation_pq": float(scores.max()),
                "mean_paired_delta_vs_unet": (
                    0.0 if model["key"] == "unet" else float(paired.mean())
                ),
                "sample_sd_paired_delta_vs_unet": (
                    0.0 if model["key"] == "unet" else float(paired.std(ddof=1))
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Source data
# ---------------------------------------------------------------------------


def _write_source_data(
    runs_by_model: dict[str, list[dict[str, Any]]],
    manifest: list[dict[str, Any]],
) -> None:
    source_dir = OUTPUT_DIR / "source_data"
    split_dir = source_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train.csv", "val.csv"):
        shutil.copy2(ROOT / "data/splits" / split_name, split_dir / split_name)
    shutil.copy2(LOG_PATH, source_dir / "parallel.log")
    (source_dir / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    condinst_epoch_rows = _parse_condinst_epoch_log(LOG_PATH)
    log_attribution: dict[int, tuple[int, str]] = {}

    # Build log attribution from any CondInst run that has a full metrics history
    for run in runs_by_model["condinst"]:
        if "metrics" not in run:
            obs = run["best_epoch_observation"]
            log_attribution[obs["log_line"]] = (
                run["seed"],
                "completion score and best epoch",
            )
            continue
        for row in run["metrics"]["history"]:
            threshold_key = f"{float(row['best_threshold']):g}"
            counts = row["threshold_metrics"].get(threshold_key)
            if counts is None:
                raise ValueError(
                    f"CondInst history missing selected-threshold counts at epoch {row['epoch']}"
                )
            matches = [
                log_row
                for log_row in condinst_epoch_rows
                if log_row["epoch"] == row["epoch"]
                and round(log_row["train_loss"], 4) == round(float(row["train_loss"]), 4)
                and round(log_row["val_pq"], 4) == round(float(row["val_pq"]), 4)
                and round(log_row["threshold"], 4) == round(float(row["best_threshold"]), 4)
                and log_row["tp"] == int(counts["tp"])
                and log_row["fp"] == int(counts["fp"])
                and log_row["fn"] == int(counts["fn"])
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one CondInst log row for seed {run['seed']} epoch {row['epoch']}; "
                    f"found {len(matches)}"
                )
            log_attribution[matches[0]["log_line"]] = (
                run["seed"],
                "recovered metrics.json match",
            )

    seed_rows = []
    history_rows = []
    for model in MODELS:
        for run in runs_by_model[model["key"]]:
            seed_rows.append(
                {
                    "model": model["name"],
                    "seed": run["seed"],
                    "best_validation_pq": run["best_validation_pq"],
                    "best_epoch": run["best_epoch"],
                    "epochs_completed": run["epochs_completed"],
                    "best_threshold": run["best_threshold"],
                    "final_validation_pq": run["final_validation_pq"],
                    "source": run["source"],
                }
            )
            if "metrics" in run:
                for row in run["metrics"]["history"]:
                    history_rows.append(
                        {
                            "model": model["name"],
                            "seed": run["seed"],
                            "epoch": row["epoch"],
                            "train_loss": row["train_loss"],
                            "validation_pq": row["val_pq"],
                            "selected_threshold": row["best_threshold"],
                            "source": run["source"],
                            "log_line": "",
                        }
                    )
            elif "best_epoch_observation" in run:
                obs = run["best_epoch_observation"]
                history_rows.append(
                    {
                        "model": model["name"],
                        "seed": run["seed"],
                        "epoch": obs["epoch"],
                        "train_loss": obs["train_loss"],
                        "validation_pq": obs["val_pq"],
                        "selected_threshold": obs["threshold"],
                        "source": "logged best-epoch observation",
                        "log_line": obs["log_line"],
                    }
                )

    _write_csv(source_dir / "results_by_seed.csv", seed_rows)
    _write_csv(source_dir / "training_history.csv", history_rows)
    _write_csv(
        source_dir / "condinst_epoch_log.csv",
        [
            {
                **row,
                "attributed_seed": log_attribution.get(row["log_line"], ("", ""))[0],
                "attribution_basis": log_attribution.get(row["log_line"], ("", ""))[1],
            }
            for row in condinst_epoch_rows
        ],
    )


# ---------------------------------------------------------------------------
# Dataset summary
# ---------------------------------------------------------------------------


def _dataset_summary() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from src.training.dataset import read_manifest, record_stem

    processed_dir = ROOT / "data/processed"
    per_image = []
    summary: dict[str, Any] = {}
    for split in ("train", "val"):
        records = read_manifest(ROOT / "data/splits" / f"{split}.csv")
        object_counts = []
        foreground_fractions = []
        image_means = []
        for record in records:
            stem = record_stem(record["image_id"])
            image = np.asarray(
                Image.open(
                    processed_dir / split / "images" / f"{stem}.png"
                ).convert("L")
            )
            mask = np.asarray(
                Image.open(
                    processed_dir / split / "masks" / "instances" / f"{stem}.png"
                )
            )
            if image.shape != mask.shape:
                raise ValueError(f"Image/mask dimensions differ for {stem}")
            object_count = int(np.count_nonzero(np.unique(mask) > 0))
            foreground_fraction = float(np.count_nonzero(mask) / mask.size)
            object_counts.append(object_count)
            foreground_fractions.append(foreground_fraction)
            image_means.append(float(image.mean()))
            per_image.append(
                {
                    "split": split,
                    "image_id": record["image_id"],
                    "instance_count": object_count,
                    "foreground_pixel_fraction": foreground_fraction,
                    "input_mean_intensity_0_255": float(image.mean()),
                    "input_std_intensity_0_255": float(image.std()),
                    "height": int(image.shape[0]),
                    "width": int(image.shape[1]),
                }
            )
        summary[split] = {
            "image_count": len(records),
            "instance_count_mean": float(np.mean(object_counts)),
            "instance_count_sd": float(np.std(object_counts, ddof=1)),
            "foreground_fraction_mean": float(np.mean(foreground_fractions)),
            "foreground_fraction_median": float(np.median(foreground_fractions)),
            "input_mean_intensity_mean": float(np.mean(image_means)),
        }
    return summary, per_image


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_results(
    runs_by_model: dict[str, list[dict[str, Any]]],
    per_image: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = OUTPUT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # --- Bar chart: best validation PQ mean ± SD ---
    names = [model["name"] for model in MODELS]
    means = [
        np.mean([run["best_validation_pq"] for run in runs_by_model[model["key"]]])
        for model in MODELS
    ]
    deviations = [
        np.std(
            [run["best_validation_pq"] for run in runs_by_model[model["key"]]],
            ddof=1,
        )
        for model in MODELS
    ]
    x = np.arange(len(MODELS))
    fig, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    axis.bar(
        x,
        means,
        yerr=deviations,
        color=[COLORS[name] for name in names],
        capsize=5,
    )
    for index, model in enumerate(MODELS):
        scores = [run["best_validation_pq"] for run in runs_by_model[model["key"]]]
        axis.scatter(
            np.full(len(scores), index) + np.linspace(-0.09, 0.09, len(scores)),
            scores,
            color="#1f2937",
            s=28,
            zorder=3,
        )
    axis.set_xticks(x, names, rotation=12, ha="right")
    axis.set(
        title=f"Best validation PQ ({len(SEEDS)} seeds)",
        ylabel="Best validation PQ",
    )
    axis.set_ylim(0, 0.4)
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(plot_dir / "best_validation_pq_mean_sd.png", dpi=180)
    plt.close(fig)

    # --- Validation PQ by epoch (mean ± SD for U-Net & BAU-Net; CondInst dots/lines) ---
    available_models = [model for model in MODELS if model["key"] != "condinst"]
    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for model in available_models:
        histories = [
            np.asarray([row["val_pq"] for row in run["metrics"]["history"]])
            for run in runs_by_model[model["key"]]
        ]
        values = np.stack(histories)
        epochs = np.arange(1, values.shape[1] + 1)
        mean = values.mean(axis=0)
        sd = values.std(axis=0, ddof=1)
        axis.plot(epochs, mean, color=COLORS[model["name"]], label=model["name"])
        axis.fill_between(
            epochs,
            np.maximum(0, mean - sd),
            mean + sd,
            color=COLORS[model["name"]],
            alpha=0.16,
        )
    for run in runs_by_model["condinst"]:
        if "metrics" in run:
            values = np.asarray([row["val_pq"] for row in run["metrics"]["history"]])
            axis.plot(
                np.arange(1, len(values) + 1),
                values,
                color=COLORS["CondInst"],
                linestyle="--",
                alpha=0.6,
                label=f"CondInst history (seed {run['seed']})",
            )
        else:
            obs = run["best_epoch_observation"]
            axis.scatter(
                [obs["epoch"]],
                [obs["val_pq"]],
                color=COLORS["CondInst"],
                marker="D",
                edgecolors="black",
                linewidths=0.5,
                zorder=4,
                label=f"CondInst best epoch (seed {run['seed']})",
            )
    axis.set(
        title="Validation PQ by epoch",
        xlabel="Epoch",
        ylabel="Validation PQ",
        ylim=(0, 0.4),
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.savefig(plot_dir / "validation_pq_mean_sd_by_epoch.png", dpi=180)
    plt.close(fig)

    # --- Normalized training loss by epoch ---
    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for model in available_models:
        losses = []
        for run in runs_by_model[model["key"]]:
            values = np.asarray(
                [row["train_loss"] for row in run["metrics"]["history"]], dtype=float
            )
            losses.append(values / values[0])
        stacked = np.stack(losses)
        mean = stacked.mean(axis=0)
        sd = stacked.std(axis=0, ddof=1)
        epochs = np.arange(1, len(mean) + 1)
        axis.plot(epochs, mean, color=COLORS[model["name"]], label=model["name"])
        axis.fill_between(
            epochs,
            np.maximum(0, mean - sd),
            mean + sd,
            color=COLORS[model["name"]],
            alpha=0.16,
        )
    for run in runs_by_model["condinst"]:
        if "metrics" not in run:
            continue
        values = np.asarray(
            [row["train_loss"] for row in run["metrics"]["history"]], dtype=float
        )
        axis.plot(
            np.arange(1, len(values) + 1),
            values / values[0],
            color=COLORS["CondInst"],
            linestyle="--",
            alpha=0.6,
            label=f"CondInst history (seed {run['seed']})",
        )
    axis.set(
        title="Normalized training loss by epoch",
        xlabel="Epoch",
        ylabel="Training loss / epoch 1 loss",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.savefig(plot_dir / "normalized_training_loss_mean_sd.png", dpi=180)
    plt.close(fig)

    # --- CondInst-only plots ---
    condinst_runs = runs_by_model["condinst"]
    for metric, axis_label, filename, title in (
        ("val_pq", "Validation PQ", "condinst_validation_pq_by_epoch.png", "CondInst validation PQ"),
        ("train_loss", "Training loss", "condinst_training_loss_by_epoch.png", "CondInst training loss"),
    ):
        fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for run in condinst_runs:
            if "metrics" in run:
                values = np.asarray(
                    [row[metric] for row in run["metrics"]["history"]], dtype=float
                )
                axis.plot(
                    np.arange(1, len(values) + 1),
                    values,
                    color=COLORS["CondInst"],
                    linestyle="--",
                    alpha=0.7,
                    label=f"History (seed {run['seed']})",
                )
            else:
                obs = run["best_epoch_observation"]
                axis.scatter(
                    [obs["epoch"]],
                    [obs[metric]],
                    color=COLORS["CondInst"],
                    marker="D",
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=4,
                    label=f"Best epoch (seed {run['seed']})",
                )
        axis.set(title=title, xlabel="Epoch", ylabel=axis_label)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        fig.savefig(plot_dir / filename, dpi=180)
        plt.close(fig)

    # --- CondInst training loss at best epoch ---
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.scatter(
        [run["seed"] for run in condinst_runs],
        [run["train_loss_at_best"] for run in condinst_runs],
        color=COLORS["CondInst"],
        edgecolors="black",
        linewidths=0.5,
        s=60,
    )
    axis.set(
        title="CondInst training loss at best epoch",
        xlabel="Seed",
        ylabel="Training loss",
        xticks=list(SEEDS),
    )
    axis.grid(alpha=0.25)
    fig.savefig(plot_dir / "condinst_best_epoch_training_loss.png", dpi=180)
    plt.close(fig)

    # --- Best epoch by seed (boxplot) ---
    fig, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    best_epochs = [
        [run["best_epoch"] for run in runs_by_model[model["key"]]] for model in MODELS
    ]
    axis.boxplot(best_epochs, labels=names, showmeans=True)
    for index, values in enumerate(best_epochs, start=1):
        axis.scatter(
            np.full(len(values), index) + np.linspace(-0.07, 0.07, len(values)),
            values,
            color=[COLORS[name] for name in names][index - 1],
            zorder=3,
        )
    axis.set(
        title="Best validation epoch across seeds",
        ylabel="Best epoch",
        ylim=(0, 105),
    )
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(plot_dir / "best_epoch_by_seed.png", dpi=180)
    plt.close(fig)

    # --- Instance counts at best checkpoint ---
    count_keys = ("tp", "fp", "fn")
    means_by_count = []
    sds_by_count = []
    for key in count_keys:
        values = [
            [run[key] for run in runs_by_model[model["key"]]] for model in MODELS
        ]
        means_by_count.append([np.mean(row) for row in values])
        sds_by_count.append([np.std(row, ddof=1) for row in values])
    fig, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    bar_width = 0.24
    x = np.arange(len(MODELS))
    count_colors = {"tp": "#16a34a", "fp": "#dc2626", "fn": "#2563eb"}
    for offset, key in zip((-bar_width, 0, bar_width), count_keys):
        count_index = count_keys.index(key)
        axis.bar(
            x + offset,
            means_by_count[count_index],
            bar_width,
            yerr=sds_by_count[count_index],
            capsize=3,
            label=key.upper(),
            color=count_colors[key],
        )
    axis.set_xticks(x, names, rotation=10, ha="right")
    axis.set(title="Instance counts at best checkpoint", ylabel="Instances")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.savefig(plot_dir / "best_checkpoint_instance_counts.png", dpi=180)
    plt.close(fig)

    # --- Dataset distribution plots ---
    for split in ("train", "val"):
        split_records = [row for row in per_image if row["split"] == split]
        counts = [row["instance_count"] for row in split_records]
        bins = np.arange(0, max(counts) + 2, max(1, int(max(counts) / 30)))
        fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        axis.hist(counts, bins=bins, color="#3b82f6", edgecolor="white")
        axis.set(
            title=f"{split.title()} instance-count distribution",
            xlabel="Instances per image",
            ylabel="Images",
        )
        axis.grid(axis="y", alpha=0.25)
        fig.savefig(plot_dir / f"{split}_instance_count_distribution.png", dpi=180)
        plt.close(fig)

        fractions = [row["foreground_pixel_fraction"] for row in split_records]
        fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        axis.hist(fractions, bins=30, color="#8b5cf6", edgecolor="white")
        axis.set(
            title=f"{split.title()} foreground-area distribution",
            xlabel="Foreground pixel fraction",
            ylabel="Images",
        )
        axis.grid(axis="y", alpha=0.25)
        fig.savefig(plot_dir / f"{split}_foreground_fraction_distribution.png", dpi=180)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Qualitative figures
# ---------------------------------------------------------------------------


def _make_qualitative_figure(
    runs_by_model: dict[str, list[dict[str, Any]]],
) -> list[str]:
    from scripts import visualize_predictions as visualizer
    from src.training.dataset import load_validation_image, read_manifest

    specs = {model["key"]: model for model in MODELS}
    # Find the first seed that has a CondInst checkpoint (prefer 43)
    condinst_checkpoint_seed: int | None = None
    for preferred in (43, 42, 44, 45):
        cp = (
            ROOT
            / specs["condinst"]["run_root"]
            / RUN_TAG.format(seed=preferred)
            / "best_model.pt"
        )
        if cp.is_file():
            condinst_checkpoint_seed = preferred
            break

    data_config = None
    models_by_seed: dict[int, list[tuple[str, Any, dict, dict]]] = {}
    provenance_rows = []

    for seed in SEEDS:
        seed_models = []
        for key in ("unet", "boundary_aware_unet"):
            spec = specs[key]
            checkpoint_path = (
                ROOT / spec["run_root"] / RUN_TAG.format(seed=seed) / "best_model.pt"
            )
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Missing qualitative checkpoint for {spec['name']} seed {seed}"
                )
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            config = checkpoint.get("config")
            if not isinstance(config, dict):
                raise ValueError(f"Checkpoint has no config: {checkpoint_path}")
            actual_identity = (
                config.get("model", {}).get("name"),
                config.get("experiment", {}).get("name"),
                config.get("training", {}).get("seed"),
            )
            expected_identity = (spec["model_name"], spec["experiment_name"], seed)
            if actual_identity != expected_identity:
                raise ValueError(
                    f"Qualitative checkpoint identity mismatch at {checkpoint_path}: "
                    f"actual={actual_identity}, expected={expected_identity}"
                )
            if data_config is None:
                data_config = config["data"]
            elif config["data"] != data_config:
                raise ValueError(
                    f"Data config differs at {checkpoint_path}"
                )
            model_options = {k: v for k, v in config["model"].items() if k != "name"}
            network = visualizer.create_model(config["model"]["name"], **model_options)
            network.load_state_dict(checkpoint["model_state_dict"])
            network.eval()
            seed_models.append((spec["name"], network, config, checkpoint))
            provenance_rows.append(
                {
                    "model": spec["name"],
                    "figure_seed": seed,
                    "checkpoint_seed": seed,
                    "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
                    "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                }
            )

        # Add CondInst if checkpoint available for this seed (or shared checkpoint)
        if condinst_checkpoint_seed is not None:
            condinst_cp_path = (
                ROOT
                / specs["condinst"]["run_root"]
                / RUN_TAG.format(seed=condinst_checkpoint_seed)
                / "best_model.pt"
            )
            condinst_checkpoint = torch.load(condinst_cp_path, map_location="cpu")
            condinst_config = condinst_checkpoint.get("config")
            condinst_name = f"CondInst (seed {condinst_checkpoint_seed} checkpoint)"
            model_options = {
                k: v for k, v in condinst_config["model"].items() if k != "name"
            }
            condinst_network = visualizer.create_model(
                condinst_config["model"]["name"], **model_options
            )
            condinst_network.load_state_dict(condinst_checkpoint["model_state_dict"])
            condinst_network.eval()
            seed_models.append(
                (condinst_name, condinst_network, condinst_config, condinst_checkpoint)
            )
            provenance_rows.append(
                {
                    "model": condinst_name,
                    "figure_seed": seed,
                    "checkpoint_seed": condinst_checkpoint_seed,
                    "checkpoint_path": str(condinst_cp_path.relative_to(ROOT)),
                    "sha256": hashlib.sha256(
                        condinst_cp_path.read_bytes()
                    ).hexdigest(),
                }
            )

        models_by_seed[seed] = seed_models

    if data_config is None:
        raise ValueError("No qualitative checkpoints were loaded")

    _write_csv(
        OUTPUT_DIR / "source_data/qualitative_checkpoint_provenance.csv",
        provenance_rows,
    )

    processed = ROOT / data_config["processed_dir"]
    records = read_manifest(ROOT / data_config["splits_dir"] / "val.csv")
    selected = visualizer._choose_records(records, processed, 3)
    visualizer._reject_mask_derived_inputs(processed, selected)

    import matplotlib.pyplot as plt

    sample_rows = []
    plot_dir = OUTPUT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for seed, models in models_by_seed.items():
        columns = ["Input", "Ground truth"] + [name for name, *_ in models]
        fig, axes = plt.subplots(
            len(selected),
            len(columns),
            figsize=(3.15 * len(columns), 3.35 * len(selected)),
            squeeze=False,
            constrained_layout=True,
        )
        for column, title in enumerate(columns):
            axes[0, column].set_title(title, fontsize=9)
        for row_index, record in enumerate(selected):
            image, target = load_validation_image(
                processed, record, data_config["resolution"]
            )
            panels = [
                ImageOps.autocontrast(image.convert("L"), cutoff=1).convert("RGB"),
                visualizer._overlay(image, target, None),
            ]
            score_row = {
                "image_id": record["image_id"],
                "figure_seed": seed,
                "target_instances": int(len(np.unique(target[target > 0]))),
            }
            for name, network, config, checkpoint in models:
                prediction, score = visualizer._predict(
                    network, config, checkpoint, image, target
                )
                panels.append(visualizer._overlay(image, target, prediction))
                sample_rows.append(
                    {
                        **score_row,
                        "model": name,
                        "checkpoint_seed": int(config["training"]["seed"]),
                        "image_pq": float(score),
                    }
                )
            for column, panel in enumerate(panels):
                axes[row_index, column].imshow(panel)
                axes[row_index, column].axis("off")
            axes[row_index, 0].set_ylabel(
                f"{record['image_id']}\n{len(np.unique(target[target > 0]))} instances",
                fontsize=7,
            )
        fig.suptitle(f"Validation predictions (seed {seed})", fontsize=14)
        fig.savefig(plot_dir / f"qualitative_seed{seed}_examples.png", dpi=180)
        plt.close(fig)

    _write_csv(OUTPUT_DIR / "source_data/qualitative_image_scores.csv", sample_rows)
    return [record["image_id"] for record in selected]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _write_report(
    runs_by_model: dict[str, list[dict[str, Any]]],
    aggregate: list[dict[str, Any]],
    dataset: dict[str, Any],
    qualitative_ids: list[str],
) -> None:
    by_key = {row["model"]: row for row in aggregate}
    n = len(SEEDS)
    seed_range = f"{min(SEEDS)}–{max(SEEDS)}"

    # Determine CondInst data availability summary
    condinst_with_history = [
        run for run in runs_by_model["condinst"] if "metrics" in run
    ]
    condinst_log_only = [
        run for run in runs_by_model["condinst"] if "metrics" not in run
    ]

    if condinst_with_history:
        history_seeds = sorted(r["seed"] for r in condinst_with_history)
        log_seeds = sorted(r["seed"] for r in condinst_log_only)
        condinst_data_note = (
            f"Full per-epoch histories are available for CondInst seeds "
            f"{', '.join(map(str, history_seeds))} from recovered metrics.json files. "
            f"Seeds {', '.join(map(str, log_seeds))} use runner completion lines and "
            f"matching best-epoch log rows only."
            if log_seeds
            else f"Full per-epoch histories are available for all CondInst seeds "
            f"from recovered metrics.json files."
        )
    else:
        condinst_data_note = (
            "No CondInst metrics.json files are available. "
            "Best scores and epochs come from runner completion lines; "
            "training-curve points show only the verified best-epoch log row per seed."
        )

    seed_cols = " | ".join(f"Seed {s}" for s in SEEDS)
    header = f"| Model | n | {seed_cols} | Mean ± sample SD | Paired mean delta vs U-Net |"
    sep = "|---|---:|" + "---:|" * len(SEEDS) + "---:|---:|"
    table_rows = [header, sep]
    for model in MODELS:
        runs = runs_by_model[model["key"]]
        values_by_seed = {run["seed"]: run["best_validation_pq"] for run in runs}
        row = by_key[model["name"]]
        table_rows.append(
            f"| {model['name']} | {row['n_seeds']} | "
            + " | ".join(
                f"{values_by_seed[s]:.4f}" if s in values_by_seed else "—"
                for s in SEEDS
            )
            + f" | {row['mean_best_validation_pq']:.4f} ± "
            f"{row['sample_sd_best_validation_pq']:.4f} | "
            f"{row['mean_paired_delta_vs_unet']:+.4f} |"
        )

    baseline_mean = by_key["U-Net"]["mean_best_validation_pq"]
    bau_mean = by_key["Boundary-aware U-Net"]["mean_best_validation_pq"]
    bau_delta = by_key["Boundary-aware U-Net"]["mean_paired_delta_vs_unet"]
    condinst_mean = by_key["CondInst"]["mean_best_validation_pq"]
    condinst_sd = by_key["CondInst"]["sample_sd_best_validation_pq"]

    lines = [
        f"# Corrected-data model comparison ({n} seeds)",
        "",
        "## Scope and methods",
        "",
        f"This analysis compares U-Net, CondInst, and boundary-aware U-Net across seeds {seed_range}. "
        "The pyhopper-tuned boundary-aware variant is excluded. "
        f"{condinst_data_note}",
        "",
        "U-Net and boundary-aware U-Net checkpoints are identity-checked: "
        "`config.model.name` and `config.experiment.name` are verified for each seed. "
        "Checkpoint identities and SHA-256 hashes are in `source_data/checkpoint_manifest.json`.",
        "",
        "Validation PQ is the project implementation's instance-level panoptic quality "
        "using greedy instance matching at IoU > 0.5. The reported score is the best "
        "validation PQ selected over the configured thresholds during training. "
        f"Means and sample standard deviations are descriptive summaries over {n} seeds; "
        "paired deltas use the same seeds.",
        "",
        "## Dataset",
        "",
        f"- Train images: {dataset['train']['image_count']}",
        f"- Validation images: {dataset['val']['image_count']}",
        f"- Train instances/image: {dataset['train']['instance_count_mean']:.2f} ± "
        f"{dataset['train']['instance_count_sd']:.2f} (sample SD)",
        f"- Validation instances/image: {dataset['val']['instance_count_mean']:.2f} ± "
        f"{dataset['val']['instance_count_sd']:.2f} (sample SD)",
        f"- Mean validation foreground pixel fraction: {dataset['val']['foreground_fraction_mean']:.4f}",
        "- `scripts/verify_prepared_data.py` passed before the bundle was built.",
        "",
        "![Training instance-count distribution](plots/train_instance_count_distribution.png)",
        "",
        "![Validation instance-count distribution](plots/val_instance_count_distribution.png)",
        "",
        "![Training foreground-area distribution](plots/train_foreground_fraction_distribution.png)",
        "",
        "![Validation foreground-area distribution](plots/val_foreground_fraction_distribution.png)",
        "",
        "## Best validation PQ",
        "",
        *table_rows,
        "",
        "![Best validation PQ](plots/best_validation_pq_mean_sd.png)",
        "",
        f"U-Net's {n}-seed mean best validation PQ is {baseline_mean:.4f}. "
        f"Boundary-aware U-Net mean is {bau_mean:.4f} (paired mean delta {bau_delta:+.4f}). "
        "This does not show a consistent improvement from the auxiliary boundary head; "
        "inspect per-seed points and qualitative examples before drawing conclusions about "
        "touching-filament separation.",
        "",
        f"CondInst's {n}-seed mean best validation PQ is {condinst_mean:.4f} "
        f"(sample SD {condinst_sd:.4f}), substantially below both U-Net and boundary-aware U-Net.",
        "",
        "## Training dynamics",
        "",
        f"U-Net and boundary-aware U-Net curves show mean and sample SD over seeds {seed_range}. "
        "CondInst histories or best-epoch markers are shown per seed; see `source_data/training_history.csv` "
        "for provenance of each row.",
        "",
        "![Validation PQ by epoch](plots/validation_pq_mean_sd_by_epoch.png)",
        "",
        "![CondInst validation PQ by epoch](plots/condinst_validation_pq_by_epoch.png)",
        "",
        "![Normalized training loss by epoch](plots/normalized_training_loss_mean_sd.png)",
        "",
        "![CondInst training loss by epoch](plots/condinst_training_loss_by_epoch.png)",
        "",
        "![CondInst training loss at best epoch](plots/condinst_best_epoch_training_loss.png)",
        "",
        "![Best epoch across seeds](plots/best_epoch_by_seed.png)",
        "",
        "## Instance-level errors",
        "",
        "TP, FP, and FN are taken from each run's best-PQ epoch and selected threshold.",
        "",
        "![Instance counts at best checkpoint](plots/best_checkpoint_instance_counts.png)",
        "",
        "## Qualitative examples",
        "",
        "Three validation images are shown per seed for U-Net and boundary-aware U-Net "
        "using identity-checked per-seed checkpoints. CondInst panels use the best available "
        "checkpoint (see `source_data/qualitative_checkpoint_provenance.csv`). "
        "Green = correct foreground overlap, red = false-positive, blue = missed foreground.",
        "",
    ]
    for seed in SEEDS:
        lines += [
            f"![Seed {seed} validation predictions](plots/qualitative_seed{seed}_examples.png)",
            "",
        ]
    lines += [
        f"Selected validation image IDs: {', '.join(qualitative_ids)}. "
        "Per-image PQ scores are in `source_data/qualitative_image_scores.csv`.",
        "",
        "## Limitations and interpretation",
        "",
        "- All PQ values are validation-selected; they are not independent test-set estimates.",
        f"- {n} seeds give a small sample; mean and SD are descriptive, not significance tests.",
        "- CondInst training curves are partial: only seeds with available metrics.json have "
        "full histories; other seeds show only the verified best-epoch point.",
        "- Qualitative comparisons use selected validation images and do not replace aggregate results.",
        "- The three model families use different prediction formulations; thresholds are not "
        "calibrated identically across models.",
        "",
        "## Reproducibility contents",
        "",
        "- `source_data/results_by_seed.csv`: per-model, per-seed best scores, best epoch, and provenance.",
        "- `source_data/summary_statistics.csv`: mean, sample SD, range, and paired deltas.",
        "- `source_data/training_history.csv`: per-epoch histories (where available) and "
        "verified best-epoch observations.",
        "- `source_data/condinst_epoch_log.csv`: all CondInst epoch records from the log "
        "with seed attribution where determinable.",
        "- `source_data/model_runs/`: per-seed metrics, config, and checkpoints.",
        "- `source_data/qualitative_checkpoint_provenance.csv`: model, figure seed, checkpoint "
        "seed, path, and SHA-256.",
        "- `source_data/parallel.log`: raw runner log.",
        "- `source_data/splits/`: exact train/validation manifests.",
        "- `source_data/checkpoint_manifest.json`: model identity and SHA-256 per checkpoint.",
        "",
    ]
    (OUTPUT_DIR / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {OUTPUT_DIR / 'analysis.md'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in (LOG_PATH, ROOT / "data/splits/train.csv", ROOT / "data/splits/val.csv"):
        if not path.is_file():
            raise FileNotFoundError(f"Required analysis source missing: {path}")

    print(f"Seeds: {SEEDS}")
    print(f"Output: {OUTPUT_DIR}")

    runs_by_model, manifest = _load_seed_runs()
    dataset_summary, per_image = _dataset_summary()
    _write_source_data(runs_by_model, manifest)
    _write_csv(OUTPUT_DIR / "source_data/dataset_per_image.csv", per_image)
    (OUTPUT_DIR / "source_data/dataset_summary.json").write_text(
        json.dumps(dataset_summary, indent=2), encoding="utf-8"
    )

    aggregate = _aggregate(runs_by_model)
    _write_csv(OUTPUT_DIR / "source_data/summary_statistics.csv", aggregate)

    _plot_results(runs_by_model, per_image)
    qualitative_ids = _make_qualitative_figure(runs_by_model)
    _write_report(runs_by_model, aggregate, dataset_summary, qualitative_ids)

    print(f"\nAnalysis bundle created at {OUTPUT_DIR}")
    for row in aggregate:
        print(
            f"  {row['model']}: mean PQ={row['mean_best_validation_pq']:.4f} "
            f"+/- {row['sample_sd_best_validation_pq']:.4f} (n={row['n_seeds']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
