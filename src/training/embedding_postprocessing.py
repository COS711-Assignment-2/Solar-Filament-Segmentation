"""Cluster foreground pixels by their learned instance embeddings."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def postprocess_embedding(
    foreground_probability: np.ndarray,
    embeddings: np.ndarray,
    foreground_threshold: float,
    cluster_radius: float,
    min_instance_size: int = 1,
) -> np.ndarray:
    """Greedily cluster foreground pixels in embedding space."""
    if not 0.0 < foreground_threshold < 1.0:
        raise ValueError("foreground_threshold must be strictly between 0 and 1")
    if cluster_radius <= 0:
        raise ValueError("cluster_radius must be positive")
    if min_instance_size < 1:
        raise ValueError("min_instance_size must be positive")
    if embeddings.ndim != 3 or embeddings.shape[1:] != foreground_probability.shape:
        raise ValueError("Embedding map must have shape (channels, height, width)")

    foreground = np.asarray(foreground_probability) >= foreground_threshold
    coordinates = np.argwhere(foreground)
    labels = np.zeros(foreground.shape, dtype=np.int32)
    if not len(coordinates):
        return labels
    features = np.asarray(embeddings[:, foreground].T, dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Embedding map contains non-finite values")

    remaining = np.ones(len(coordinates), dtype=bool)
    next_label = 1
    while remaining.any():
        remaining_indices = np.flatnonzero(remaining)
        remaining_features = features[remaining_indices]
        tree = cKDTree(remaining_features)
        counts = tree.query_ball_point(
            remaining_features, r=cluster_radius, return_length=True
        )
        seed_index = int(np.argmax(counts))
        seed_feature = remaining_features[seed_index]
        neighbors = tree.query_ball_point(seed_feature, r=cluster_radius)
        cluster_indices = remaining_indices[np.asarray(neighbors, dtype=np.int64)]
        center = features[cluster_indices].mean(axis=0)
        cluster_indices = remaining_indices[
            tree.query_ball_point(center, r=cluster_radius)
        ]
        remaining[cluster_indices] = False
        if len(cluster_indices) >= min_instance_size:
            cluster_coordinates = coordinates[cluster_indices]
            labels[cluster_coordinates[:, 0], cluster_coordinates[:, 1]] = next_label
            next_label += 1
    return labels
