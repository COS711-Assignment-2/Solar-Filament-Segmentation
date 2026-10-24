"""Detect duplicate files and repeated annotation records without deleting data.

``duplicate_report`` distinguishes repeated names, exact binary duplicates,
and repeated physical filenames in annotation records.  Hashes are SHA-256
and are computed incrementally.  Suspected duplicates are only reported; the
module deliberately does not canonicalise or merge annotations automatically.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duplicate_report(
    root: str | Path,
    annotations_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return duplicate paths, names, content hashes, and annotation groups."""
    root_path = Path(root)
    by_name: defaultdict[str, list[str]] = defaultdict(list)
    by_hash: defaultdict[str, list[str]] = defaultdict(list)
    paths = [path for path in root_path.rglob("*") if path.is_file()]
    for path in tqdm(paths, desc="Hashing files", unit="file",
                     dynamic_ncols=True):
        if path.is_file():
            by_name[path.name].append(str(path.relative_to(root_path)))
            by_hash[_hash(path)].append(str(path.relative_to(root_path)))
    result: dict[str, Any] = {
        "repeated_filenames": {key: value for key, value in by_name.items()
                               if len(value) > 1},
        "exact_binary_duplicates": {key: value for key, value in by_hash.items()
                                    if len(value) > 1},
    }
    if annotations_path:
        with Path(annotations_path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        groups: defaultdict[str, list[Any]] = defaultdict(list)
        for image in data.get("images", []):
            name = Path(str(image.get("file_name", "")).replace("\\", "/")).name
            groups[name].append(image.get("id"))
        result["annotation_records_by_physical_filename"] = {
            key: value for key, value in groups.items() if len(value) > 1
        }
    return result
