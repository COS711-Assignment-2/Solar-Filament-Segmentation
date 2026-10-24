"""Test deterministic grouped splitting and partition isolation."""

import json

from src.data.split_data import create_split


def _annotations():
    return {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 4, "height": 4},
            {"id": 2, "file_name": "a.jpg", "width": 4, "height": 4},
            {"id": 3, "file_name": "b.jpg", "width": 4, "height": 4},
            {"id": 4, "file_name": "c.jpg", "width": 4, "height": 4},
        ],
        "annotations": [],
        "categories": [{"id": 1, "name": "Left"}],
    }


def test_split_is_reproducible_and_grouped(tmp_path):
    """Same seed produces the same manifests and repeated names do not cross."""
    source = tmp_path / "annotations.json"
    source.write_text(json.dumps(_annotations()), encoding="utf-8")
    first, second = tmp_path / "one", tmp_path / "two"
    create_split(source, first, seed=42)
    create_split(source, second, seed=42)
    assert (first / "train.csv").read_text() == (second / "train.csv").read_text()
    assert (first / "val.csv").read_text() == (second / "val.csv").read_text()
    summary = json.loads((first / "split_summary.json").read_text())
    assert summary["shared_groups"] == []
