"""Verify split manifests and generated image/target files.

The checker validates manifest existence, train/validation group isolation,
image/instance/category target presence, target dimensions, and permitted
non-negative integer mask values.  It never modifies prepared data.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _safe_stem(value: str) -> str:
    """Match the preparation script's filesystem-safe identifier."""
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in value)


def verify(
    split_dir: str | Path,
    processed_dir: str | Path | None = "data/processed",
) -> list[str]:
    """Return verification errors for manifests and processed pairs."""
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
    if processed_dir is not None:
        for split, split_rows in rows.items():
            for row in split_rows:
                stem = _safe_stem(row["image_id"])
                directory = Path(processed_dir) / split
                image_path = directory / "images" / f"{stem}.png"
                instance_path = directory / "masks" / "instances" / f"{stem}.png"
                category_path = directory / "masks" / "categories" / f"{stem}.png"
                binary_path = directory / "masks" / "binary" / f"{stem}.png"
                metadata_path = directory / "metadata" / f"{stem}.json"
                for path in (
                    image_path, instance_path, category_path, binary_path,
                    metadata_path,
                ):
                    if not path.is_file():
                        errors.append(f"Missing processed output: {path}")
                if not all(path.is_file() for path in (
                    image_path, instance_path, category_path, binary_path,
                    metadata_path,
                )):
                    continue
                with Image.open(image_path) as image, Image.open(instance_path) as instance, \
                        Image.open(category_path) as category, Image.open(binary_path) as binary:
                    if not (image.size == instance.size == category.size == binary.size):
                        errors.append(f"Dimension mismatch for {stem}")
                    if np.any(np.asarray(instance) < 0) or np.any(np.asarray(category) < 0):
                        errors.append(f"Negative mask value for {stem}")
                    if not set(np.unique(np.asarray(binary))).issubset({0, 255}):
                        errors.append(f"Binary mask has non-binary values for {stem}")
                with metadata_path.open(encoding="utf-8") as handle:
                    metadata = json.load(handle)
                if str(metadata.get("image_id")) != row["image_id"]:
                    errors.append(f"Metadata ID mismatch for {stem}")
    return errors


def main() -> int:
    """Run verification and return a non-zero status on failure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()
    errors = verify(args.split_dir, args.processed_dir)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Prepared-data verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
