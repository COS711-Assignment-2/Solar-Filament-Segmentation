"""Orchestrate validation, splitting, mask generation, and preprocessing.

The script validates the raw COCO dataset, creates grouped train/validation
manifests, rasterises each annotation record into instance/category masks, and
writes deterministic grayscale images and aligned targets under
``data/processed/{train,val}``.  Raw files are never modified.  Use
``--stage validate``, ``split``, ``process``, or ``all``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.data.convert_annotations import rasterize_annotations, save_targets
from src.data.inspect_annotations import inspect_annotations
from src.data.preprocess_images import preprocess_pair
from src.data.split_data import create_split
from src.data.validate_data import validate_dataset


def main() -> int:
    """Run selected preparation stages."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("validate", "split", "process", "all"), default="all"
    )
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
    parser.add_argument(
        "--target-width", type=int, default=None,
        help="Optional processed width; defaults to each source width.",
    )
    parser.add_argument(
        "--target-height", type=int, default=None,
        help="Optional processed height; defaults to each source height.",
    )
    args = parser.parse_args()
    audit_dir = Path(args.audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    print("Stage 1/3: validating raw dataset...")
    validation = validate_dataset(args.train_images, args.annotations)
    with (audit_dir / "validation.json").open("w", encoding="utf-8") as handle:
        json.dump(validation, handle, indent=2)
    if not validation["valid"]:
        raise SystemExit("Raw data validation failed; inspect validation.json")
    if args.stage in ("split", "all"):
        print("Stage 2/3: creating grouped train/validation split...")
        create_split(args.annotations, args.split_dir, args.val_ratio, args.seed)
        with (audit_dir / "annotations.json").open("w", encoding="utf-8") as handle:
            json.dump(inspect_annotations(args.annotations), handle, indent=2)
        Path(args.processed_dir).mkdir(parents=True, exist_ok=True)
    if args.stage in ("process", "all"):
        print("Stage 3/3: generating processed images and masks...")
        process_prepared_data(
            args.train_images,
            args.annotations,
            args.split_dir,
            args.processed_dir,
            (args.target_width, args.target_height)
            if args.target_width and args.target_height else None,
        )
    print("Preparation stages completed.")
    return 0


def process_prepared_data(
    train_images: str | Path,
    annotations_path: str | Path,
    split_dir: str | Path,
    processed_dir: str | Path,
    target_size: tuple[int, int] | None = None,
) -> dict[str, int]:
    """Generate aligned images, instance masks, category masks, and metadata."""
    with Path(annotations_path).open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    annotations_by_image: dict[object, list[dict]] = {}
    for annotation in document.get("annotations", []):
        annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)
    image_records = {record["id"]: record for record in document.get("images", [])}
    source_by_name = {
        path.name: path for path in Path(train_images).iterdir() if path.is_file()
    }
    processed_path = Path(processed_dir)
    counts = {"train": 0, "val": 0, "skipped": 0}
    for split in ("train", "val"):
        manifest = Path(split_dir) / f"{split}.csv"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing split manifest: {manifest}")
        destination = processed_path / split
        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
            for row in tqdm(
                rows,
                desc=f"Processing {split}",
                unit="record",
                dynamic_ncols=True,
            ):
                image_id = row["image_id"]
                record = image_records.get(image_id, image_records.get(
                    _coerce_id(image_id)
                ))
                if record is None:
                    raise ValueError(f"Manifest image ID is absent from JSON: {image_id}")
                filename = Path(row["file_name"].replace("\\", "/")).name
                image_path = source_by_name.get(filename)
                if image_path is None:
                    raise FileNotFoundError(f"Manifest image is missing: {filename}")
                width, height = int(record["width"]), int(record["height"])
                instance, category, metadata = rasterize_annotations(
                    width, height, annotations_by_image.get(record["id"], [])
                )
                stem = _safe_stem(image_id)
                metadata.update({"image_id": record["id"], "file_name": filename,
                                 "width": width, "height": height})
                split_root = processed_path / split
                instance_dir = split_root / "masks" / "instances"
                category_dir = split_root / "masks" / "categories"
                binary_dir = split_root / "masks" / "binary"
                image_dir = split_root / "images"
                metadata_dir = split_root / "metadata"
                category_dir.mkdir(parents=True, exist_ok=True)
                binary_dir.mkdir(parents=True, exist_ok=True)
                image_dir.mkdir(parents=True, exist_ok=True)
                metadata_dir.mkdir(parents=True, exist_ok=True)
                save_targets(instance_dir, stem, instance, category, metadata)
                # Move the category mask into its dedicated directory.
                (instance_dir / f"{stem}_categories.png").replace(
                    category_dir / f"{stem}.png"
                )
                (instance_dir / f"{stem}_instances.png").replace(
                    instance_dir / f"{stem}.png"
                )
                (instance_dir / f"{stem}_metadata.json").replace(
                    metadata_dir / f"{stem}.json"
                )
                binary_mask = np.where(instance > 0, 255, 0).astype(np.uint8)
                Image.fromarray(binary_mask, mode="L").save(
                    binary_dir / f"{stem}.png"
                )
                preprocess_pair(
                    image_path, instance_dir / f"{stem}.png",
                    category_dir / f"{stem}.png", image_dir, target_size,
                    output_stem=stem,
                )
                counts[split] += 1
    with (processed_path / "preparation_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(counts, handle, indent=2)
    return counts


def _coerce_id(value: str) -> object:
    """Match CSV string IDs to numeric JSON IDs without changing provenance."""
    try:
        return int(value)
    except ValueError:
        return value


def _safe_stem(value: str) -> str:
    """Return a filesystem-safe manifest identifier."""
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in value)


if __name__ == "__main__":
    raise SystemExit(main())
