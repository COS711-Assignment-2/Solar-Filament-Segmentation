"""Train YOLO26n-seg from random initialisation.

The script builds the model from ``configs/yolo26n-seg.yaml`` and explicitly
passes ``pretrained=False``.  It expects an Ultralytics dataset YAML and keeps
training arguments visible on the command line.  No training starts during
import; execute this script only when a run is intended.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.yolo26 import train_yolo26n


def main() -> int:
    """Parse training options and launch a from-scratch run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="configs/yolo26n-seg.yaml")
    parser.add_argument("--data", default="configs/yolo26_dataset.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="outputs/yolo26")
    parser.add_argument("--name", default="yolo26n-seg-from-scratch")
    parser.add_argument("--cache", action="store_true")
    args = parser.parse_args()
    options = {
        "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
        "workers": args.workers, "project": args.project, "name": args.name,
        "cache": args.cache,
    }
    if args.device is not None:
        options["device"] = args.device
    train_yolo26n(args.model, args.data, **options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
