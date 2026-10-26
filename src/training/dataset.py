"""Load processed images and instance masks for training and validation."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    """Read a split CSV and reject empty manifests."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    if not records:
        raise ValueError(f"Split manifest contains no records: {path}")
    return records


def record_stem(image_id: str) -> str:
    """Match the safe file stem used by the preparation pipeline."""
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in image_id)


def load_record(
    processed_dir: str | Path,
    split: str,
    record: dict[str, str],
    resolution: int | str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one image and its instance mask at the configured resolution."""
    root = Path(processed_dir) / split
    stem = record_stem(record["image_id"])
    image_path = root / "images" / f"{stem}.png"
    mask_path = root / "masks" / "instances" / f"{stem}.png"
    if not image_path.is_file():
        raise FileNotFoundError(f"Processed image is missing: {image_path}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"Processed instance mask is missing: {mask_path}")
    image = Image.open(image_path).convert("L")
    mask = Image.open(mask_path)
    if image.size != mask.size:
        raise ValueError(f"Image and instance mask dimensions differ for {stem}")
    if resolution != "tiled":
        size = (int(resolution), int(resolution))
        image = image.resize(size, Image.Resampling.BILINEAR)
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return (
        np.asarray(image, dtype=np.float32) / 255.0,
        np.asarray(mask, dtype=np.int32),
    )


def tile_origins(width: int, height: int, tile_size: int) -> list[tuple[int, int]]:
    """Return a non-overlapping tile grid covering the whole image."""
    if tile_size < 16:
        raise ValueError("tile_size must be at least 16 pixels")
    return [
        (x, y)
        for y in range(0, height, tile_size)
        for x in range(0, width, tile_size)
    ]


def crop_tile(array: np.ndarray, x: int, y: int, tile_size: int) -> np.ndarray:
    """Crop a tile, padding right and bottom with zeros when needed."""
    height, width = array.shape
    tile = array[y:min(y + tile_size, height), x:min(x + tile_size, width)]
    padded = np.zeros((tile_size, tile_size), dtype=array.dtype)
    padded[:tile.shape[0], :tile.shape[1]] = tile
    return padded


class FilamentDataset(Dataset):
    """Dataset of fixed-size images or non-overlapping image tiles."""

    def __init__(
        self,
        processed_dir: str | Path,
        split: str,
        records: list[dict[str, str]],
        resolution: int | str,
        tile_size: int,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.split = split
        self.records = records
        self.resolution = resolution
        self.tile_size = tile_size
        self.samples: list[tuple[int, int, int]] = []
        self.samples_by_record: list[list[int]] = []
        self._cached_record_index = -1
        self._cached_arrays: tuple[np.ndarray, np.ndarray] | None = None
        for record_index, record in enumerate(records):
            if resolution == "tiled":
                image_path = (
                    self.processed_dir / split / "images"
                    / f"{record_stem(record['image_id'])}.png"
                )
                if not image_path.is_file():
                    raise FileNotFoundError(f"Processed image is missing: {image_path}")
                with Image.open(image_path) as image:
                    origins = tile_origins(image.width, image.height, tile_size)
            else:
                origins = [(0, 0)]
            indices = []
            for x, y in origins:
                indices.append(len(self.samples))
                self.samples.append((record_index, x, y))
            self.samples_by_record.append(indices)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record_index, x, y = self.samples[index]
        if record_index != self._cached_record_index:
            self._cached_arrays = load_record(
                self.processed_dir,
                self.split,
                self.records[record_index],
                self.resolution,
            )
            self._cached_record_index = record_index
        if self._cached_arrays is None:
            raise RuntimeError("Dataset record cache was not populated")
        image, instances = self._cached_arrays
        if self.resolution == "tiled":
            image = crop_tile(image, x, y, self.tile_size)
            instances = crop_tile(instances, x, y, self.tile_size)
        binary = instances > 0
        return {
            "image": torch.from_numpy(image.copy()).float().unsqueeze(0),
            "mask": torch.from_numpy(binary.copy()).float().unsqueeze(0),
            "instances": torch.from_numpy(instances.copy()).long(),
        }


class RecordBatchSampler(Sampler):
    """Shuffle records while keeping each record's tiles adjacent for I/O."""

    def __init__(
        self, dataset: FilamentDataset, batch_size: int, shuffle: bool = True
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        record_order = list(range(len(self.dataset.samples_by_record)))
        if self.shuffle:
            random.shuffle(record_order)
        indices = []
        for record_index in record_order:
            record_samples = list(self.dataset.samples_by_record[record_index])
            if self.shuffle:
                random.shuffle(record_samples)
            indices.extend(record_samples)
        for start in range(0, len(indices), self.batch_size):
            yield indices[start:start + self.batch_size]

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def load_validation_image(
    processed_dir: str | Path,
    record: dict[str, str],
    resolution: int | str,
) -> tuple[Image.Image, np.ndarray]:
    """Load original-resolution image and scaled target mask for validation."""
    root = Path(processed_dir) / "val"
    stem = record_stem(record["image_id"])
    image_path = root / "images" / f"{stem}.png"
    mask_path = root / "masks" / "instances" / f"{stem}.png"
    if not image_path.is_file():
        raise FileNotFoundError(f"Processed image is missing: {image_path}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"Processed instance mask is missing: {mask_path}")
    image = Image.open(image_path).convert("L")
    mask = Image.open(mask_path)
    if image.size != mask.size:
        raise ValueError(f"Image and instance mask dimensions differ for {stem}")
    if resolution != "tiled":
        size = (int(resolution), int(resolution))
        image = image.resize(size, Image.Resampling.BILINEAR)
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return image, np.asarray(mask, dtype=np.int32)
