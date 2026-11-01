import numpy as np
import pytest
from PIL import Image

from scripts.visualize_predictions import _reject_mask_derived_inputs


def test_qualitative_comparison_refuses_mask_derived_processed_inputs(tmp_path):
    processed_dir = tmp_path / "processed"
    image_dir = processed_dir / "val" / "images"
    mask_dir = processed_dir / "val" / "masks" / "instances"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    records = []
    for image_id in ("sample-a", "sample-b"):
        target = np.zeros((8, 8), dtype=np.uint8)
        target[2:5, 3:6] = 1
        Image.fromarray(target).save(image_dir / f"{image_id}.png")
        Image.fromarray(target).save(mask_dir / f"{image_id}.png")
        records.append({"image_id": image_id})

    with pytest.raises(RuntimeError, match="appear mask-derived"):
        _reject_mask_derived_inputs(processed_dir, records)
