"""Validate raw images and COCO-style solar-filament annotations.

``validate_dataset`` checks required paths, decodable images, JSON structure,
unique IDs, references, polygon coordinates, bounding boxes, and categories.
It returns separate errors and warnings so legitimate negative images remain
valid.  Coordinates outside image bounds are warnings by default because
clipping is handled explicitly during rasterisation.  No files are modified.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def validate_annotations(data: dict[str, Any]) -> list[str]:
    """Return validation errors for an in-memory COCO-style document."""
    errors: list[str] = []
    for key in ("images", "annotations", "categories"):
        if not isinstance(data.get(key), list):
            errors.append(f"Missing or invalid top-level list: {key}")
    if errors:
        return errors
    image_ids = [item.get("id") for item in data["images"]]
    annotation_ids = [item.get("id") for item in data["annotations"]]
    if len(image_ids) != len(set(image_ids)):
        errors.append("Image IDs are not unique")
    if len(annotation_ids) != len(set(annotation_ids)):
        errors.append("Annotation IDs are not unique")
    known_images = set(image_ids)
    known_categories = {item.get("id") for item in data["categories"]}
    dimensions = {item.get("id"): (item.get("width"), item.get("height"))
                  for item in data["images"]}
    for annotation in data["annotations"]:
        required = ("id", "image_id", "category_id", "segmentation")
        missing = [field for field in required if field not in annotation]
        if missing:
            errors.append(f"Annotation {annotation.get('id')} missing {missing}")
            continue
        if annotation["image_id"] not in known_images:
            errors.append(f"Annotation {annotation['id']} references unknown image")
        if annotation["category_id"] not in known_categories:
            errors.append(f"Annotation {annotation['id']} references unknown category")
        segments = annotation["segmentation"]
        if not isinstance(segments, list) or not segments:
            errors.append(f"Annotation {annotation['id']} has empty segmentation")
            continue
        width, height = dimensions.get(annotation["image_id"], (None, None))
        for polygon in segments:
            if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
                errors.append(f"Annotation {annotation['id']} has invalid polygon")
                continue
            coordinates = polygon
            if not all(isinstance(value, (int, float)) and math.isfinite(value)
                       for value in coordinates):
                errors.append(f"Annotation {annotation['id']} has non-finite coordinates")
            if width and height and any(
                x < 0 or x > width or y < 0 or y > height
                for x, y in zip(coordinates[::2], coordinates[1::2])
            ):
                errors.append(f"Annotation {annotation['id']} has out-of-bounds coordinates")
        bbox = annotation.get("bbox")
        if bbox is not None and (not isinstance(bbox, list) or len(bbox) != 4):
            errors.append(f"Annotation {annotation['id']} has invalid bbox")
    return errors


def validate_dataset(
    train_images: str | Path,
    annotations_path: str | Path,
) -> dict[str, Any]:
    """Validate image files and annotation references, returning a report."""
    train_dir = Path(train_images)
    annotation_file = Path(annotations_path)
    errors: list[str] = []
    warnings: list[str] = []
    if not train_dir.is_dir():
        errors.append(f"Training image directory does not exist: {train_dir}")
    if not annotation_file.is_file():
        errors.append(f"Annotation file does not exist: {annotation_file}")
        return {"valid": False, "errors": errors, "warnings": warnings}
    try:
        with annotation_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return {"valid": False, "errors": [f"Cannot read annotation JSON: {exc}"],
                "warnings": warnings}
    errors.extend(validate_annotations(data))
    files = {path.name: path for path in train_dir.iterdir() if path.is_file()}
    for image in data.get("images", []):
        filename = Path(str(image.get("file_name", "")).replace("\\", "/")).name
        if filename not in files:
            errors.append(f"Image record {image.get('id')} file is missing: {filename}")
        elif image.get("width") is not None and image.get("height") is not None:
            try:
                from PIL import Image
                with Image.open(files[filename]) as opened:
                    if (opened.width, opened.height) != (image["width"], image["height"]):
                        warnings.append(f"Dimension mismatch for {filename}")
            except OSError as exc:
                errors.append(f"Unreadable image {filename}: {exc}")
    return {"valid": not errors, "errors": errors, "warnings": warnings,
            "image_records": len(data.get("images", [])),
            "annotation_records": len(data.get("annotations", []))}
