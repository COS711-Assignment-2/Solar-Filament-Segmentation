"""Create deterministic, leakage-safe train and validation manifests.

The splitter groups COCO image records by physical filename, retains each
JSON image ID and annotation count, and writes CSV manifests plus JSON audit
metadata.  This is a conservative default supported by the project
documentation; stronger temporal grouping requires reliable metadata and is
not inferred.  No test images are read or included.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def physical_image_key(file_name: str) -> str:
    """Return a normalised basename used as the physical-observation key."""
    return Path(file_name.replace("\\", "/")).name


def create_split(
    annotations_path: str | Path,
    output_dir: str | Path,
    val_ratio: float = 0.20,
    seed: int = 42,
) -> dict[str, Any]:
    """Create train.csv and val.csv and return the split audit dictionary."""
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    with Path(annotations_path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    images = data.get("images", [])
    if len(images) < 2:
        raise ValueError("At least two image records are required")
    categories = {str(item["id"]): item.get("name", str(item["id"]))
                  for item in data.get("categories", [])}
    annotations_by_image: defaultdict[Any, list[dict[str, Any]]] = defaultdict(list)
    for annotation in data.get("annotations", []):
        annotations_by_image[annotation.get("image_id")].append(annotation)
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for image in images:
        groups[physical_image_key(str(image["file_name"]))].append(image)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_val = max(1, min(round(len(keys) * val_ratio), len(keys) - 1))
    val_keys, train_keys = set(keys[:n_val]), set(keys[n_val:])

    fields = ["split", "image_id", "file_name", "physical_image_key",
              "width", "height", "annotation_count"]
    fields.extend(f"category_{key}_{name}" for key, name in categories.items())
    def rows(selected: set[str], split: str) -> list[dict[str, Any]]:
        output = []
        for key in sorted(selected):
            for image in groups[key]:
                anns = annotations_by_image.get(image["id"], [])
                row = {"split": split, "image_id": image["id"],
                       "file_name": image["file_name"], "physical_image_key": key,
                       "width": image.get("width", ""), "height": image.get("height", ""),
                       "annotation_count": len(anns)}
                counts = Counter(str(item.get("category_id")) for item in anns)
                for category_id, name in categories.items():
                    row[f"category_{category_id}_{name}"] = counts[category_id]
                output.append(row)
        return output
    train_rows, val_rows = rows(train_keys, "train"), rows(val_keys, "val")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for name, values in (("train.csv", train_rows), ("val.csv", val_rows)):
        with (output_path / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)
    audit = {
        "seed": seed, "validation_ratio": val_ratio,
        "grouping_rule": "normalised physical filename basename",
        "total_image_records": len(images), "total_groups": len(groups),
        "train_groups": len(train_keys), "validation_groups": len(val_keys),
        "shared_groups": sorted(train_keys & val_keys),
        "train_records": len(train_rows), "validation_records": len(val_rows),
    }
    with (output_path / "split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    return audit
