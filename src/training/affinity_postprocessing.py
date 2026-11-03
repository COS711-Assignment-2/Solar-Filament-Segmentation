"""Convert foreground and local affinities into instance labels."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def postprocess_affinity(
    foreground_probability: np.ndarray,
    affinity_probability: np.ndarray,
    foreground_threshold: float,
    affinity_threshold: float,
    min_instance_size: int = 1,
) -> np.ndarray:
    """Label foreground pixels connected by predicted same-instance edges."""
    if not 0.0 < foreground_threshold < 1.0:
        raise ValueError("foreground_threshold must be strictly between 0 and 1")
    if not 0.0 < affinity_threshold < 1.0:
        raise ValueError("affinity_threshold must be strictly between 0 and 1")
    if min_instance_size < 1:
        raise ValueError("min_instance_size must be positive")
    if affinity_probability.shape != (2, *foreground_probability.shape):
        raise ValueError("Affinity map must have shape (2, height, width)")

    foreground = np.asarray(foreground_probability) >= foreground_threshold
    height, width = foreground.shape
    expanded = np.zeros((2 * height - 1, 2 * width - 1), dtype=bool)
    expanded[::2, ::2] = foreground
    horizontal = (
        foreground[:, :-1]
        & foreground[:, 1:]
        & (affinity_probability[0, :, :-1] >= affinity_threshold)
    )
    vertical = (
        foreground[:-1, :]
        & foreground[1:, :]
        & (affinity_probability[1, :-1, :] >= affinity_threshold)
    )
    expanded[::2, 1::2] = horizontal
    expanded[1::2, ::2] = vertical
    structure = ndimage.generate_binary_structure(2, 1)
    labels, count = ndimage.label(expanded, structure=structure)
    pixel_labels = labels[::2, ::2].astype(np.int32, copy=False)

    if min_instance_size > 1 and count:
        sizes = np.bincount(pixel_labels.ravel())
        remove = sizes < min_instance_size
        remove[0] = False
        pixel_labels[remove[pixel_labels]] = 0
    return pixel_labels
