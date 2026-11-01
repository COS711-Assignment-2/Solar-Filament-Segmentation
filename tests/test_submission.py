"""Unit tests for submission generation and COCO RLE encoding."""

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from scripts.generate_submission import (
    decode_rle_to_mask,
    encode_mask_to_rle,
    generate_submission,
)
from src.models import UNet


def test_rle_encode_decode_roundtrip():
    """Verify pycocotools RLE encoding preserves the exact binary mask."""
    mask = np.zeros((2048, 2048), dtype=np.uint8)
    mask[100:150, 200:250] = 1
    mask[500:520, 1000:1050] = 1

    rle_string = encode_mask_to_rle(mask)
    assert isinstance(rle_string, str)
    assert len(rle_string) > 0

    decoded = decode_rle_to_mask(rle_string, height=2048, width=2048)
    assert decoded.shape == (2048, 2048)
    assert np.array_equal(mask, decoded)


def test_generate_submission_pipeline(tmp_path):
    """Test full end-to-end submission generation on synthetic test images."""
    test_dir = tmp_path / "test_images"
    test_dir.mkdir()

    # Create 2 synthetic test images
    for stem in ["test_img_A", "test_img_B"]:
        img = Image.fromarray(np.random.randint(50, 200, (2048, 2048), dtype=np.uint8))
        img.save(test_dir / f"{stem}.jpeg")

    # Create dummy checkpoint
    model = UNet(in_channels=1, base_channels=8)
    checkpoint_path = tmp_path / "dummy_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "threshold": 0.5,
            "config": {
                "model": {"name": "unet", "base_channels": 8},
                "evaluation": {"postprocessing": "connected_components", "min_instance_size": 1},
            },
        },
        checkpoint_path,
    )

    out_csv = tmp_path / "submission.csv"
    generate_submission(
        checkpoints=[checkpoint_path],
        test_dir=test_dir,
        output_csv=out_csv,
        input_resolution=64,
        target_size=(2048, 2048),
        device="cpu",
    )

    assert out_csv.is_file()
    with out_csv.open("r", encoding="utf-8") as f:
        reader = list(csv.reader(f))

    assert len(reader) >= 1
    assert reader[0] == ["filament_id", "segmentation_rle"]

    for row in reader[1:]:
        assert len(row) == 2
        filament_id, rle = row
        assert "_" in filament_id
        stem, idx = filament_id.rsplit("_", 1)
        assert stem in ["test_img_A", "test_img_B"]
        assert idx.isdigit() and int(idx) >= 1
        assert len(rle) > 0
        decoded = decode_rle_to_mask(rle, 2048, 2048)
        assert decoded.shape == (2048, 2048)
        assert np.isin(decoded, [0, 1]).all()
