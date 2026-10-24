"""Prepare the processed COCO records for Ultralytics YOLO segmentation.

This script copies original training images into ``data/yolo26/images`` and
writes one polygon-label file per annotation record under
``data/yolo26/labels``.  It preserves duplicate physical image records by
using JSON image IDs as output names and displays a progress bar.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.yolo_dataset import create_yolo_dataset


def main() -> int:
    """Prepare the YOLO dataset and write a summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--source-images", required=True)
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument("--output-dir", default="data/yolo26")
    args = parser.parse_args()
    summary = create_yolo_dataset(
        args.annotations, args.source_images, args.split_dir, args.output_dir
    )
    with (Path(args.output_dir) / "preparation_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
