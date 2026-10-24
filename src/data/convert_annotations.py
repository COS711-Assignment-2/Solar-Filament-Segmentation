"""Rasterise COCO polygons into aligned instance and category masks.

``rasterize_annotations`` uses Pillow's polygon renderer and preserves one
instance ID per annotation, a semantic category map, and provenance metadata.
Coordinates are clipped by the raster canvas.  Overlaps are reported and
resolved deterministically by annotation order; this is an implementation
choice that must be compared with the official competition rules before final
evaluation.  Empty annotations produce all-background masks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def rasterize_annotations(
    image_width: int,
    image_height: int,
    annotations: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return instance IDs, category IDs, and provenance for annotations."""
    instance = np.zeros((image_height, image_width), dtype=np.int32)
    category = np.zeros((image_height, image_width), dtype=np.int16)
    provenance: list[dict[str, Any]] = []
    overlaps = 0
    for next_id, annotation in enumerate(annotations, start=1):
        before = instance.copy()
        layer = Image.new("1", (image_width, image_height), 0)
        draw = ImageDraw.Draw(layer)
        for polygon in annotation.get("segmentation", []):
            if isinstance(polygon, list) and len(polygon) >= 6:
                points = [(float(polygon[index]), float(polygon[index + 1]))
                          for index in range(0, len(polygon) - 1, 2)]
                draw.polygon(points, fill=1)
        pixels = np.asarray(layer, dtype=bool)
        overlaps += int(np.count_nonzero(pixels & (before > 0)))
        instance[pixels] = next_id
        category[pixels] = int(annotation.get("category_id", 0))
        provenance.append({"instance_id": next_id,
                           "annotation_id": annotation.get("id"),
                           "category_id": annotation.get("category_id")})
    return instance, category, {"instances": provenance,
                                "overlap_pixels": overlaps}


def save_targets(
    output_dir: str | Path,
    stem: str,
    instance: np.ndarray,
    category: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Save lossless PNG masks and JSON provenance metadata."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    Image.fromarray(instance.astype(np.int32), mode="I").save(
        destination / f"{stem}_instances.png"
    )
    Image.fromarray(category.astype(np.int32), mode="I").save(
        destination / f"{stem}_categories.png"
    )
    with (destination / f"{stem}_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
