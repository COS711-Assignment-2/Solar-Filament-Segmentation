"""Produce detailed, non-destructive audits of COCO annotations.

The ``inspect_annotations`` function reports category counts, empty images,
polygon and duplicate-ID issues, repeated physical filenames, and basic
instance distributions.  It does not merge annotation records or infer
annotator identity when the source JSON does not provide it.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def inspect_annotations(path: str | Path) -> dict[str, Any]:
    """Return annotation statistics and provenance-preserving relationships."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    by_image: defaultdict[Any, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        by_image[annotation.get("image_id")].append(annotation)
    filenames = [Path(str(item.get("file_name", "")).replace("\\", "/")).name
                 for item in images]
    category_names = {item.get("id"): item.get("name", str(item.get("id")))
                      for item in data.get("categories", [])}
    category_counts = Counter(category_names.get(item.get("category_id"),
                                                 str(item.get("category_id")))
                              for item in annotations)
    areas = [item.get("area") for item in annotations
             if isinstance(item.get("area"), (int, float))]
    return {
        "image_records": len(images),
        "unique_physical_filenames": len(set(filenames)),
        "annotation_records": len(annotations),
        "category_counts": dict(category_counts),
        "images_without_annotations": sum(
            not by_image.get(item.get("id")) for item in images
        ),
        "instances_per_image": {
            str(image_id): len(items) for image_id, items in by_image.items()
        },
        "polygon_area": {
            "count": len(areas),
            "minimum": min(areas) if areas else None,
            "maximum": max(areas) if areas else None,
            "mean": sum(areas) / len(areas) if areas else None,
        },
        "repeated_annotation_ids": [
            key for key, count in Counter(item.get("id") for item in annotations).items()
            if count > 1
        ],
        "repeated_physical_filenames": [
            key for key, count in Counter(filenames).items() if count > 1
        ],
        "malformed_segmentations": [
            item.get("id") for item in annotations
            if not isinstance(item.get("segmentation"), list)
        ],
    }
