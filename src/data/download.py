"""Download and discover the MAGFiLO Kaggle competition dataset.

This module wraps ``kagglehub.competition_download`` and never modifies an
existing download.  It discovers training images, test images, and the COCO
annotation JSON below the returned path instead of assuming a fixed layout.
The public ``download_dataset`` and ``discover_dataset`` functions return
structured paths and raise explicit errors for unavailable or incomplete data.
Kaggle credentials are handled by KaggleHub and must not be stored here.

Use ``python scripts/download_data.py`` for the command-line interface.
Competition-specific submission and PQ rules are intentionally outside this
module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class DatasetDiscoveryError(RuntimeError):
    """Raised when a downloaded directory does not contain expected files."""


@dataclass(frozen=True)
class DatasetPaths:
    """Discovered paths for the training and test portions of the dataset."""

    root: Path
    train_images: Path
    test_images: Path
    annotations: Path


def discover_dataset(root: str | Path) -> DatasetPaths:
    """Discover image directories and a COCO-like annotation file below root."""
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists() or not root_path.is_dir():
        raise DatasetDiscoveryError(f"Dataset directory does not exist: {root_path}")

    image_dirs = [
        path for path in root_path.rglob("*")
        if path.is_dir() and path.name.lower() in {"train_images", "test_images"}
    ]
    train_dirs = [path for path in image_dirs if path.name.lower() == "train_images"]
    test_dirs = [path for path in image_dirs if path.name.lower() == "test_images"]
    annotations = [
        path for path in root_path.rglob("*.json")
        if "annotation" in path.name.lower() and "train" in path.name.lower()
    ]
    if not annotations:
        annotations = [
            path for path in root_path.rglob("*.json")
            if _looks_like_coco(path)
        ]
    if not train_dirs or not test_dirs or not annotations:
        raise DatasetDiscoveryError(
            "Could not discover train_images, test_images, and a training "
            f"annotation JSON below {root_path}"
        )
    return DatasetPaths(root_path, train_dirs[0], test_dirs[0], annotations[0])


def _looks_like_coco(path: Path) -> bool:
    """Return whether a small JSON probe has COCO top-level keys."""
    try:
        import json
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return all(key in value for key in ("images", "annotations", "categories"))
    except (OSError, ValueError, TypeError):
        return False


def download_dataset(
    staging_dir: str | Path = "data/raw",
    competition: str = "filament-segmentation-2026",
) -> DatasetPaths:
    """Download the competition through KaggleHub and discover its contents.

    ``staging_dir`` is passed to KaggleHub's supported ``output_dir`` option.
    The directory is created if necessary, and raw files remain separate from
    generated files under ``data/processed``.
    """
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is required for downloads; install requirements.txt"
        ) from exc
    try:
        from kagglehub.exceptions import UnauthenticatedError
    except ImportError:
        UnauthenticatedError = ()  # type: ignore[assignment]
    raw_dir = Path(staging_dir).expanduser().resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        returned_path = kagglehub.competition_download(
            competition, output_dir=str(raw_dir)
        )
    except UnauthenticatedError as exc:
        raise RuntimeError(
            "KaggleHub is not authenticated. Authenticate with one of the "
            "supported methods: set KAGGLE_API_TOKEN, place your Kaggle API "
            "credentials at %USERPROFILE%\\.kaggle\\kaggle.json, or run "
            "kagglehub.login() from Python. Then rerun this command."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "KaggleHub could not download the competition. Check credentials, "
            "network access, and competition permissions."
        ) from exc
    dataset_path = Path(returned_path)
    if staging_dir:
        LOGGER.info("Requested staging directory is %s; KaggleHub returned %s",
                    Path(staging_dir).resolve(), dataset_path.resolve())
    return discover_dataset(dataset_path)
