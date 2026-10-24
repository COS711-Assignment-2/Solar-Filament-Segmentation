"""Command-line entry point for downloading and discovering the Kaggle data.

The script delegates to ``src.data.download`` and reports the discovered
training image, test image, and annotation paths.  It performs no destructive
filesystem operations and requires Kaggle credentials only when downloading.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Running a file below scripts/ does not automatically add the project root to
# sys.path, so make the local src package available for direct CLI execution.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.download import download_dataset


def main() -> int:
    """Parse arguments, download the dataset, and print discovered paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--staging-dir",
        default="data/raw",
        help="Directory for the original downloaded competition files.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    paths = download_dataset(args.staging_dir)
    for label, path in (("root", paths.root), ("train_images", paths.train_images),
                        ("test_images", paths.test_images),
                        ("annotations", paths.annotations)):
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
