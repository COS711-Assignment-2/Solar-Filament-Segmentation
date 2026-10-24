"""Convert the validated COCO records into an Ultralytics segmentation set.

The converter writes one image and one YOLO polygon-label file per JSON image
record.  Output stems use the JSON ``image_id`` so multiple annotation records
for one physical filename remain separate and are never overwritten.  COCO
category IDs 1-4 are mapped to YOLO class IDs 0-3 in the supplied dataset
configuration.  Polygon coordinates are clipped to image bounds and
normalised to [0, 1].  The official treatment of category semantics and
overlapping polygons remains subject to the competition rules.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from tqdm import tqdm


def _safe_stem(value: object) -> str:
    """Return a filesystem-safe image-record identifier."""
    return "".join(char if char.isalnum() or char in "-_." else "_"
                   for char in str(value))


def _record_lookup(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index JSON image records by both native and string IDs."""
    result: dict[str, dict[str, Any]] = {}
    for record in document.get("images", []):
        result[str(record["id"])] = record
    return result


def _polygon_line(
    category_id: int,
    polygon: list[float],
    width: int,
    height: int,
) -> str | None:
    """Convert one COCO polygon into one normalised YOLO label line."""
    if len(polygon) < 6 or len(polygon) % 2:
        return None
    points: list[str] = []
    for index in range(0, len(polygon), 2):
        x = min(max(float(polygon[index]), 0.0), float(width))
        y = min(max(float(polygon[index + 1]), 0.0), float(height))
        points.extend((f"{x / width:.6f}", f"{y / height:.6f}"))
    if len(points) < 6:
        return None
    return f"{category_id - 1} " + " ".join(points)


def create_yolo_dataset(
    annotations_path: str | Path,
    source_images: str | Path,
    split_dir: str | Path,
    output_dir: str | Path = "data/yolo26",
) -> dict[str, int]:
    """Write YOLO images and polygon labels for train and validation splits."""
    with Path(annotations_path).open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    records = _record_lookup(document)
    annotations_by_image: dict[str, list[dict[str, Any]]] = {}
    for annotation in document.get("annotations", []):
        annotations_by_image.setdefault(str(annotation["image_id"]), []).append(annotation)
    source_by_name = {
        path.name: path for path in Path(source_images).iterdir() if path.is_file()
    }
    output = Path(output_dir)
    counts = {"train": 0, "val": 0, "labels": 0, "skipped_polygons": 0}
    for split in ("train", "val"):
        image_output = output / "images" / split
        label_output = output / "labels" / split
        image_output.mkdir(parents=True, exist_ok=True)
        label_output.mkdir(parents=True, exist_ok=True)
        manifest = Path(split_dir) / f"{split}.csv"
        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in tqdm(rows, desc=f"Preparing YOLO {split}", unit="record",
                        dynamic_ncols=True):
            image_id = str(row["image_id"])
            record = records.get(image_id)
            if record is None:
                raise ValueError(f"Image ID {image_id} is absent from annotations")
            filename = Path(row["file_name"].replace("\\", "/")).name
            source = source_by_name.get(filename)
            if source is None:
                raise FileNotFoundError(f"Source image is missing: {filename}")
            stem = _safe_stem(image_id)
            shutil.copy2(source, image_output / f"{stem}{source.suffix.lower()}")
            lines: list[str] = []
            width, height = int(record["width"]), int(record["height"])
            for annotation in annotations_by_image.get(image_id, []):
                category_id = int(annotation["category_id"])
                if category_id not in {1, 2, 3, 4}:
                    raise ValueError(f"Unsupported category ID: {category_id}")
                for polygon in annotation.get("segmentation", []):
                    line = _polygon_line(category_id, polygon, width, height)
                    if line is None:
                        counts["skipped_polygons"] += 1
                    else:
                        lines.append(line)
            (label_output / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            counts[split] += 1
            counts["labels"] += len(lines)
    return counts
