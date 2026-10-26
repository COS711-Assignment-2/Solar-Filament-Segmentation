"""Convert foreground probabilities into instance masks."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def postprocess_probability(
    probability: np.ndarray,
    threshold: float,
    method: str,
    min_instance_size: int = 1,
    watershed_min_distance: int = 5,
) -> np.ndarray:
    """Extract instances using connected components or watershed."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1")
    if method not in ("connected_components", "watershed"):
        raise ValueError(f"Unsupported post-processing method: {method}")
    foreground = np.asarray(probability) >= threshold
    if method == "connected_components":
        labels, _ = ndimage.label(foreground)
    else:
        distance = ndimage.distance_transform_edt(foreground)
        coordinates = peak_local_max(
            distance, min_distance=watershed_min_distance, labels=foreground
        )
        markers = np.zeros(foreground.shape, dtype=np.int32)
        for marker_id, (row, column) in enumerate(coordinates, start=1):
            markers[row, column] = marker_id
        if len(coordinates) == 0:
            labels, _ = ndimage.label(foreground)
        else:
            labels = watershed(-distance, markers, mask=foreground)
    labels = labels.astype(np.int32, copy=False)
    if min_instance_size > 1:
        sizes = np.bincount(labels.ravel())
        remove = sizes < min_instance_size
        remove[0] = False
        labels[remove[labels]] = 0
        labels, _ = ndimage.label(labels > 0)
    return labels
