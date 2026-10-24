"""Build a streaming file and image inventory for the raw dataset.

The module records file paths, sizes, extensions, SHA-256 hashes, image
dimensions, and modes without retaining all images in memory.  It can also
count COCO records and writes JSON/CSV-friendly dictionaries for audit reports.
Pillow is required for image inspection; malformed files are recorded rather
than hidden.  Use ``build_inventory`` from an audit script.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_inventory(
    root: str | Path,
    annotations_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return an inventory dictionary for every file below ``root``."""
    root_path = Path(root)
    records: list[dict[str, Any]] = []
    for path in sorted(item for item in root_path.rglob("*") if item.is_file()):
        record: dict[str, Any] = {
            "path": str(path.relative_to(root_path)),
            "name": path.name,
            "extension": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            try:
                from PIL import Image
                with Image.open(path) as image:
                    record.update({"width": image.width, "height": image.height,
                                   "mode": image.mode, "readable": True})
            except (OSError, ValueError) as exc:
                record.update({"readable": False, "error": str(exc)})
        records.append(record)
    result: dict[str, Any] = {"root": str(root_path.resolve()), "files": records}
    if annotations_path:
        with Path(annotations_path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        result["annotation_counts"] = {
            "images": len(data.get("images", [])),
            "annotations": len(data.get("annotations", [])),
            "categories": len(data.get("categories", [])),
        }
    return result
