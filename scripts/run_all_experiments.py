"""Run every experiment config for one or more random seeds."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument(
        "--max-epochs",
        type=int,
        help="Optional maximum-epoch override for every experiment run.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        help="Optional early-stopping patience override; 0 disables early stopping.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "xla", "tpu"),
        help="Optional device override passed to each training run.",
    )
    parser.add_argument(
        "--experiments-dir",
        default="experiments",
        help="Directory containing one experiment folder and config.yaml per run.",
    )
    args = parser.parse_args()
    if args.n_runs < 1:
        parser.error("--n-runs must be at least 1")
    if args.max_epochs is not None and args.max_epochs < 1:
        parser.error("--max-epochs must be at least 1")
    if args.patience is not None and args.patience < 0:
        parser.error("--patience must be zero or greater")
    experiments_dir = Path(args.experiments_dir)
    experiment_configs = sorted(experiments_dir.glob("*/config.yaml"))
    if not experiment_configs:
        raise FileNotFoundError(
            f"No experiment configs found in {experiments_dir}/<name>/config.yaml"
        )
    trainer = Path(__file__).with_name("train.py")
    for config_path in experiment_configs:
        for run_index in range(args.n_runs):
            seed = args.seed_start + run_index
            command = [
                sys.executable,
                str(trainer),
                "--config",
                str(config_path),
                "--seed",
                str(seed),
                "--output-tag",
                f"seed{seed}",
            ]
            if args.device:
                command.extend(("--device", args.device))
            if args.max_epochs is not None:
                command.extend(("--max-epochs", str(args.max_epochs)))
            if args.patience is not None:
                command.extend(("--patience", str(args.patience)))
            print(
                f"Running experiment={config_path.parent.name} seed={seed}",
                flush=True,
            )
            subprocess.run(
                command,
                check=True,
                cwd=Path(__file__).resolve().parents[1],
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
