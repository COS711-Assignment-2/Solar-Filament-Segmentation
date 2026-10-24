"""Test annotation inspection and validation edge cases."""

from src.data.validate_data import validate_annotations


def test_invalid_polygon_is_reported():
    """Polygons need at least three coordinate pairs."""
    errors = validate_annotations({
        "images": [{"id": 1, "width": 4, "height": 4}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                         "segmentation": [[0, 0, 1, 1]]}],
        "categories": [{"id": 1}],
    })
    assert any("invalid polygon" in error for error in errors)


def test_out_of_bounds_polygon_is_reported():
    """Out-of-bounds coordinates are visible to the audit."""
    errors = validate_annotations({
        "images": [{"id": 1, "width": 4, "height": 4}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                         "segmentation": [[0, 0, 5, 0, 5, 5]]}],
        "categories": [{"id": 1}],
    })
    assert any("out-of-bounds" in error for error in errors)
