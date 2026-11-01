"""Tune boundary-aware U-Net hyperparameters with PyHopper and validation PQ."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.trainer import load_config, train


ROOT = Path(__file__).resolve().parents[1]


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def build_search_space(pyhopper: Any, batch_sizes: list[int]) -> Any:
    """Build the requested search space, keeping the architecture fixed."""
    return pyhopper.Search(
        {
            "learning_rate": pyhopper.float(
                1e-5, 1e-3, log=True, init=3e-4
            ),
            "boundary_weight": pyhopper.float(0.05, 0.5, init=0.25),
            "batch_size": pyhopper.choice(batch_sizes, init_index=0),
            "bce_ratio": pyhopper.float(0.25, 0.75, init=0.5),
            "threshold": pyhopper.float(
                0.3, 0.7, precision=2, init=0.5
            ),
        }
    )


def apply_parameters(
    config: dict[str, Any],
    parameters: dict[str, Any],
    seed: int,
    max_epochs: int,
) -> dict[str, Any]:
    """Apply searched hyperparameters to a copied fixed-model config."""
    tuned = copy.deepcopy(config)
    if tuned["model"]["name"] != "boundary_aware_unet":
        raise ValueError(
            "This tuner only supports the boundary_aware_unet architecture"
        )
    tuned["training"]["learning_rate"] = float(parameters["learning_rate"])
    tuned["training"]["batch_size"] = int(parameters["batch_size"])
    tuned["training"]["seed"] = seed
    tuned["training"]["max_epochs"] = max_epochs
    tuned["training"]["patience"] = None
    tuned["loss"]["boundary_weight"] = float(parameters["boundary_weight"])
    tuned["loss"]["bce_weight"] = float(parameters["bce_ratio"])
    tuned["loss"]["dice_weight"] = 1.0 - float(parameters["bce_ratio"])
    tuned["evaluation"]["thresholds"] = [float(parameters["threshold"])]
    return tuned


def run_search(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import pyhopper
    except ImportError as error:
        raise RuntimeError(
            "PyHopper is required for tuning. Install dependencies with "
            "`python3 -m pip install pyhopper`."
        ) from error

    config_path = (ROOT / args.config).resolve()
    base_config = load_config(config_path)
    if base_config["model"]["name"] != "boundary_aware_unet":
        raise ValueError(
            f"Expected a boundary_aware_unet config, got "
            f"{base_config['model']['name']!r}"
        )
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_dir = output_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    search_checkpoint = output_dir / "pyhopper_search.ckpt"
    search = build_search_space(pyhopper, args.batch_sizes)
    trial_number = 0

    def objective(parameters: dict[str, Any]) -> float:
        nonlocal trial_number
        trial_number += 1
        trial_config = apply_parameters(
            base_config, parameters, args.seed, args.trial_epochs
        )
        trial_config["training"]["output_dir"] = str(trials_dir)
        parameter_key = hashlib.sha256(
            json.dumps(parameters, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:12]
        tag = f"trial-{parameter_key}-seed{args.seed}"
        print(
            f"PyHopper trial {trial_number}: "
            f"lr={parameters['learning_rate']:.3g}, "
            f"boundary={parameters['boundary_weight']:.3f}, "
            f"batch={parameters['batch_size']}, "
            f"bce_ratio={parameters['bce_ratio']:.3f}, "
            f"threshold={parameters['threshold']:.2f}",
            flush=True,
        )
        result = train(
            trial_config,
            seed_override=args.seed,
            output_tag=tag,
            device_override=args.device,
            max_epochs_override=args.trial_epochs,
            patience_override=0,
            save_checkpoints=False,
            initialize_from=args.init_checkpoint,
        )
        score = float(result["best_val_pq"])
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Trial returned invalid validation PQ: {score}")
        print(
            f"PyHopper trial {trial_number}: best validation PQ={score:.6f}",
            flush=True,
        )
        return score

    best_parameters = search.run(
        objective,
        direction="maximize",
        runtime=args.runtime,
        n_jobs=1,
        checkpoint_path=str(search_checkpoint),
    )
    if best_parameters is None:
        raise RuntimeError("PyHopper completed without returning best parameters")

    best_config = apply_parameters(
        base_config, best_parameters, args.seed, args.final_epochs
    )
    best_config["training"]["output_dir"] = "runs"
    best_config_path = output_dir / "best_config.yaml"
    with best_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(best_config, handle, sort_keys=False)

    result = {
        "model": base_config["model"]["name"],
        "seed": args.seed,
        "validation_split": base_config["data"]["splits_dir"],
        "objective": "maximum validation panoptic quality",
        "search_runtime": args.runtime,
        "trial_epochs": args.trial_epochs,
        "warm_start_checkpoint": display_path(args.init_checkpoint),
        "final_epochs": args.final_epochs,
        "best_params": best_parameters,
        "best_validation_pq": float(search.best_f),
        "best_config": display_path(best_config_path),
        "search_checkpoint": display_path(search_checkpoint),
    }
    results_path = output_dir / "best_params.json"
    results_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    final_command = (
        f"python3 scripts/train.py --config {shlex.quote(display_path(best_config_path))} "
        f"--seed {args.seed} --output-tag pyhopper-best-seed{args.seed} "
        f"--max-epochs {args.final_epochs} --patience 0 --device {args.device}"
    )
    print(f"Best validation PQ: {search.best_f:.6f}")
    print(f"Best parameters: {json.dumps(best_parameters, sort_keys=True)}")
    print(f"Saved results: {display_path(results_path)}")
    print(f"Saved final-training config: {best_config_path}")
    print(f"Final training command:\n{final_command}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/boundary_aware_unet/config.yaml"),
        help="Fixed boundary-aware U-Net experiment config.",
    )
    parser.add_argument(
        "--runtime",
        default="4h",
        help="PyHopper search budget, e.g. 4h or 30min.",
    )
    parser.add_argument(
        "--trial-epochs",
        type=int,
        default=6,
        help="Fixed training epochs per candidate; use the same value for every trial.",
    )
    parser.add_argument(
        "--final-epochs",
        type=int,
        default=100,
        help="Epoch limit written into the winning full-training config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=43,
        help="Fixed random seed used for every candidate.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "xla", "tpu"),
        default="xla",
        help="Training device. Trials run sequentially on one device.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        choices=(4, 8, 16),
        default=[4, 8, 16],
        help="Batch sizes to search; reduce this list if TPU memory is limited.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "runs/boundary_aware_unet/boundary-aware-unet/pyhopper-seed43"
        ),
        help="Directory for search state, trial metrics, and winning config.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=Path(
            "runs/boundary_aware_unet/boundary-aware-unet/seed43/best_model.pt"
        ),
        help="Model weights to warm-start every candidate from.",
    )
    args = parser.parse_args()
    if not args.runtime.strip():
        parser.error("--runtime must be a non-empty PyHopper duration")
    if args.trial_epochs < 1:
        parser.error("--trial-epochs must be at least 1")
    if args.final_epochs < args.trial_epochs:
        parser.error("--final-epochs must be >= --trial-epochs")
    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        parser.error("--batch-sizes must not contain duplicates")
    args.init_checkpoint = (ROOT / args.init_checkpoint).resolve()
    run_search(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
