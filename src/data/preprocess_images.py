"""Apply deterministic image preprocessing while preserving target alignment.

The module converts source images to grayscale, optionally resizes them, and
uses nearest-neighbour interpolation for categorical or instance masks.
Random augmentation is intentionally excluded.  ``preprocess_pair`` writes
the image output and updates the generated target masks in place when resizing,
preserving the source files and making validation/test preprocessing
deterministic.
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
    output_stem: str | None = None,
) -> dict[str, Any]:
    """Preprocess one image and keep its generated target masks aligned."""
    image = Image.open(image_path).convert("L")
    instances = Image.open(instance_mask_path)
    categories = Image.open(category_mask_path)
    if size:
        image = image.resize(size, Image.Resampling.BILINEAR)
        instances = instances.resize(size, Image.Resampling.NEAREST)
        categories = categories.resize(size, Image.Resampling.NEAREST)
        instances.save(instance_mask_path)
        categories.save(category_mask_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stem = output_stem or Path(image_path).stem
    image_path_out = destination / f"{stem}.png"
    image.save(image_path_out)
    result = {"image": str(destination / f"{stem}.png"),
            "instances": str(instance_mask_path),
            "categories": str(category_mask_path),
            "size": image.size}
    if normalize:
        normalized_path = image_path_out.with_suffix(".npy")
        np.save(normalized_path, np.asarray(image, dtype=np.float32) / 255.0)
        result["normalized_array"] = str(normalized_path)
    return result
