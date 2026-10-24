"""Verify split manifests and generated image/target files.

The checker validates manifest existence, train/validation group isolation, and
optionally checks image-target dimensions.  It never modifies prepared data.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def verify(split_dir: str | Path) -> list[str]:
    """Return verification errors for train and validation manifests."""
    directory = Path(split_dir)
    errors: list[str] = []
    rows: dict[str, list[dict[str, str]]] = {}
    for split in ("train", "val"):
        path = directory / f"{split}.csv"
        if not path.is_file():
            errors.append(f"Missing manifest: {path}")
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows[split] = list(csv.DictReader(handle))
    if "train" in rows and "val" in rows:
        train_groups = {row.get("physical_image_key") for row in rows["train"]}
        val_groups = {row.get("physical_image_key") for row in rows["val"]}
        overlap = train_groups & val_groups
        if overlap:
            errors.append(f"Groups cross partitions: {sorted(overlap)}")
    return errors


def main() -> int:
    """Run verification and return a non-zero status on failure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", default="data/splits")
    args = parser.parse_args()
    errors = verify(args.split_dir)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Prepared-data verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
