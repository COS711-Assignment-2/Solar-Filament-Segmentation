"""Build and train the official YOLO26 nano segmentation architecture.

``build_yolo26n`` constructs a model from the checked-in public YAML and does
not load weights.  ``train_yolo26n`` delegates optimisation to Ultralytics
with ``pretrained=False``.  The Ultralytics package is an external dependency
under its AGPL-3.0 licence; review that licence before distribution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_device(requested: str = "auto") -> str:
    """Resolve a portable device setting to an Ultralytics device value.

    Supported values are ``auto``, ``cpu``, ``gpu``/``cuda``, ``mps``, a CUDA
    device expression such as ``0`` or ``0,1``, and ``tpu``.  Ultralytics'
    standard ``YOLO.train`` interface does not support TPU execution through
    its ``device`` argument, so TPU requests fail explicitly rather than
    silently falling back to CPU.
    """
    normalized = requested.strip().lower()
    if normalized == "tpu":
        raise RuntimeError(
            "TPU training is not supported by the standard Ultralytics "
            "YOLO.train interface. Use a TPU-specific PyTorch/XLA training "
            "implementation rather than passing tpu as an Ultralytics device."
        )
    if normalized in {"cpu", "mps"}:
        return normalized
    if normalized in {"gpu", "cuda"}:
        return "0"
    if normalized not in {"auto", ""}:
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
    options["device"] = resolve_device(str(options.get("device", "auto")))
    return model.train(**options)
