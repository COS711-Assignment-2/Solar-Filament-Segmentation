"""Test polygon rasterisation, identity preservation, and overlap reporting."""

from src.data.convert_annotations import rasterize_annotations


def test_masks_preserve_instances_and_categories():
    """Separate polygons receive separate IDs and their category values."""
    instances, categories, metadata = rasterize_annotations(10, 10, [
        {"id": 11, "category_id": 1, "segmentation": [[1, 1, 3, 1, 3, 3]]},
        {"id": 12, "category_id": 2, "segmentation": [[6, 6, 8, 6, 8, 8]]},
    ])
    assert instances.shape == (10, 10)
    assert set(instances.ravel()) == {0, 1, 2}
    assert categories[2, 2] == 1
    assert categories[7, 7] == 2
    assert [item["annotation_id"] for item in metadata["instances"]] == [11, 12]


def test_overlaps_are_reported():
    """Overlapping polygons are counted rather than silently ignored."""
    _, _, metadata = rasterize_annotations(5, 5, [
        {"id": 1, "category_id": 1, "segmentation": [[0, 0, 3, 0, 3, 3]]},
        {"id": 2, "category_id": 1, "segmentation": [[1, 1, 4, 1, 4, 4]]},
    ])
    assert metadata["overlap_pixels"] > 0
