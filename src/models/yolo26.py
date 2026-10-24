"""Build and train the official YOLO26 nano segmentation architecture.

``build_yolo26n`` constructs a model from the checked-in public YAML and does
not load weights.  ``train_yolo26n`` delegates optimisation to Ultralytics
with ``pretrained=False``.  The Ultralytics package is an external dependency
under its AGPL-3.0 licence; review that licence before distribution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_yolo26n(model_yaml: str | Path) -> Any:
    """Construct YOLO26n-seg from YAML with random initialisation."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required; install requirements.txt first"
        ) from exc
    return YOLO(str(model_yaml))


def train_yolo26n(
    model_yaml: str | Path,
    data_yaml: str | Path,
    **train_options: Any,
) -> Any:
    """Train YOLO26n-seg from random initialisation using Ultralytics."""
    model = build_yolo26n(model_yaml)
    options = {"data": str(data_yaml), "pretrained": False}
    options.update(train_options)
    return model.train(**options)
