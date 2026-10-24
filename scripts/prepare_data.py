"""Orchestrate raw validation and reproducible split generation.

This conservative entry point currently performs discovery-independent
validation and splitting.  Mask conversion and preprocessing are exposed as
library functions so callers can choose an explicit annotation policy before
materialising targets.  Use ``--stage validate`` or ``--stage split``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.inspect_annotations import inspect_annotations
from src.data.split_data import create_split
from src.data.validate_data import validate_dataset


def main() -> int:
    """Run selected preparation stages."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("validate", "split", "all"), default="all")
    parser.add_argument("--train-images", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory for generated masks and preprocessed images.",
    )
    parser.add_argument("--audit-dir", default="outputs/data_audit")
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    audit_dir = Path(args.audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    validation = validate_dataset(args.train_images, args.annotations)
    with (audit_dir / "validation.json").open("w", encoding="utf-8") as handle:
        json.dump(validation, handle, indent=2)
    if not validation["valid"]:
        raise SystemExit("Raw data validation failed; inspect validation.json")
    if args.stage in ("split", "all"):
        create_split(args.annotations, args.split_dir, args.val_ratio, args.seed)
        with (audit_dir / "annotations.json").open("w", encoding="utf-8") as handle:
            json.dump(inspect_annotations(args.annotations), handle, indent=2)
        Path(args.processed_dir).mkdir(parents=True, exist_ok=True)
    print("Preparation stages completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
