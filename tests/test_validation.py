"""Test raw-file and COCO validation with small temporary fixtures."""

import json

from PIL import Image

from src.data.validate_data import validate_annotations, validate_dataset


def test_valid_dataset(tmp_path):
    """A readable image and matching annotation document validate."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("L", (10, 10)).save(image_dir / "a.jpg")
    document = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                         "segmentation": [[1, 1, 8, 1, 8, 8]]}],
        "categories": [{"id": 1, "name": "Left"}],
    }
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(json.dumps(document), encoding="utf-8")
    assert validate_dataset(image_dir, annotation_path)["valid"]


def test_unknown_reference_is_reported():
    """Annotations must reference declared images and categories."""
    errors = validate_annotations({
        "images": [{"id": 1, "width": 5, "height": 5}],
        "annotations": [{"id": 2, "image_id": 9, "category_id": 4,
                         "segmentation": [[0, 0, 1, 0, 1, 1]]}],
        "categories": [{"id": 1}],
    })
    assert any("unknown image" in error for error in errors)
    assert any("unknown category" in error for error in errors)
