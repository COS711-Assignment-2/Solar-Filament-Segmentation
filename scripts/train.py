"""CLI entry point for training one named experiment."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.trainer import main


if __name__ == "__main__":
    raise SystemExit(main())
