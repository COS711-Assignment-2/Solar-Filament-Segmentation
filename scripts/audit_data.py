"""Run inventory, validation, annotation, and duplicate audits.

Usage: ``python scripts/audit_data.py --root data/raw/... --annotations ...``.
Reports are JSON files under ``outputs/data_audit`` by default.  A critical
validation failure returns a non-zero exit status.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.deduplicate import duplicate_report
from src.data.inspect_annotations import inspect_annotations
from src.data.inventory import build_inventory
from src.data.validate_data import validate_dataset


def main() -> int:
    """Execute all audit stages and return a shell-friendly status."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--train-images", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output-dir", default="outputs/data_audit")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reports = {
        "inventory": build_inventory(args.root, args.annotations),
        "validation": validate_dataset(args.train_images, args.annotations),
        "annotations": inspect_annotations(args.annotations),
        "duplicates": duplicate_report(args.root, args.annotations),
    }
    for name, report in reports.items():
        with (output / f"{name}.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    print(f"Audit reports written to {output.resolve()}")
    return 0 if reports["validation"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
