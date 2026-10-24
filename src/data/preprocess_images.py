"""Apply deterministic image preprocessing while preserving target alignment.

The module converts source images to grayscale, optionally resizes them, and
uses nearest-neighbour interpolation for categorical or instance masks.
Random augmentation is intentionally excluded.  ``preprocess_pair`` writes
lossless PNG targets and a configurable image output, preserving the source
files and making validation/test preprocessing deterministic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def preprocess_pair(
    image_path: str | Path,
    instance_mask_path: str | Path,
    category_mask_path: str | Path,
    output_dir: str | Path,
    size: tuple[int, int] | None = None,
    normalize: bool = False,
) -> dict[str, Any]:
    """Preprocess and save one image and its aligned target masks."""
    image = Image.open(image_path).convert("L")
    instances = Image.open(instance_mask_path)
    categories = Image.open(category_mask_path)
    if size:
        image = image.resize(size, Image.Resampling.BILINEAR)
        instances = instances.resize(size, Image.Resampling.NEAREST)
        categories = categories.resize(size, Image.Resampling.NEAREST)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    image.save(destination / f"{stem}.png")
    instances.save(destination / f"{stem}_instances.png")
    categories.save(destination / f"{stem}_categories.png")
    result = {"image": str(destination / f"{stem}.png"),
            "instances": str(destination / f"{stem}_instances.png"),
            "categories": str(destination / f"{stem}_categories.png"),
            "size": image.size}
    if normalize:
        normalized_path = destination / f"{stem}.npy"
        np.save(normalized_path, np.asarray(image, dtype=np.float32) / 255.0)
        result["normalized_array"] = str(normalized_path)
    return result
