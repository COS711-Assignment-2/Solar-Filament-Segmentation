"""Model construction and training helpers for the COS711-A2 project."""

from __future__ import annotations

from typing import Callable

from torch import nn

from src.models.boundary_aware_unet import BoundaryAwareUNet
from src.models.unet import UNet

ModelFactory = Callable[..., nn.Module]
MODEL_REGISTRY: dict[str, ModelFactory] = {
    "unet": UNet,
    "boundary_aware_unet": BoundaryAwareUNet,
}


def register_model(name: str, factory: ModelFactory) -> None:
    """Register a model constructor under a unique experiment-facing name."""
    if not name or name.strip() != name or "/" in name or "\\" in name:
        raise ValueError("Model names must be non-empty path-safe identifiers")
    if name in MODEL_REGISTRY:
        raise ValueError(f"Model name is already registered: {name}")
    MODEL_REGISTRY[name] = factory


def create_model(name: str, **kwargs: object) -> nn.Module:
    """Create a model by its stable registry/configuration name."""
    try:
        factory = MODEL_REGISTRY[name]
    except KeyError as error:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model '{name}'. Registered models: {available}"
        ) from error
    return factory(**kwargs)


__all__ = [
    "BoundaryAwareUNet",
    "MODEL_REGISTRY",
    "UNet",
    "create_model",
    "register_model",
]
