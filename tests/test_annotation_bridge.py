"""YOLO-to-native annotation conversion and synchronization tests."""

import json

import numpy as np
import pytest

from autolabel.annotation_bridge import prepare_review_dataset, sync_review_to_yolo


def dataset(tmp_path):
    import cv2

    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    (tmp_path / "classes.txt").write_text("person\nphone\n", encoding="utf-8")
    cv2.imwrite(str(images / "frame.jpg"), np.zeros((100, 200, 3), dtype=np.uint8))
    (labels / "frame.txt").write_text("0 0.500000 0.500000 0.400000 0.600000\n",
                                      encoding="utf-8")
    return tmp_path


def test_yolo_json_review_roundtrip(tmp_path):
    from autolabel.workbench.views.labeling.label_file import LabelFile

    root = dataset(tmp_path)
    result = prepare_review_dataset(root)
    assert result["json_created"] == 1
    annotation = root / "images" / "frame.json"
    parsed = LabelFile(str(annotation))
    assert parsed.shapes[0].label == "person"
    data = json.loads(annotation.read_text(encoding="utf-8"))
    data["shapes"][0]["label"] = "phone"
    annotation.write_text(json.dumps(data), encoding="utf-8")
    assert prepare_review_dataset(root)["json_created"] == 0
    assert sync_review_to_yolo(root)["labels"] == 1
    assert (root / "labels" / "frame.txt").read_text(encoding="utf-8").startswith("1 ")


def test_sync_rejects_invalid_category_without_overwrite(tmp_path):
    root = dataset(tmp_path)
    prepare_review_dataset(root)
    annotation = root / "images" / "frame.json"
    data = json.loads(annotation.read_text(encoding="utf-8"))
    data["shapes"][0]["label"] = "unknown"
    annotation.write_text(json.dumps(data), encoding="utf-8")
    previous = (root / "labels" / "frame.txt").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="类别"):
        sync_review_to_yolo(root)
    assert (root / "labels" / "frame.txt").read_text(encoding="utf-8") == previous


def test_numbered_bom_classes_use_same_mapping_for_validation_and_sync(tmp_path):
    from autolabel.validate import validate_dataset

    root = dataset(tmp_path)
    (root / "classes.txt").write_text("1 phone\n0 person\n", encoding="utf-8-sig")
    assert validate_dataset(root)["classes"] == ["person", "phone"]
    prepare_review_dataset(root)
    assert sync_review_to_yolo(root)["classes"] == ["person", "phone"]


def test_sync_commit_failure_preserves_all_existing_labels(tmp_path, monkeypatch):
    import cv2
    from autolabel.workbench.views.labeling import annotation_io

    root = dataset(tmp_path)
    cv2.imwrite(str(root / "images" / "second.jpg"), np.zeros((100, 200, 3), dtype=np.uint8))
    (root / "labels" / "second.txt").write_text("0 0.5 0.5 0.4 0.6\n")
    prepare_review_dataset(root)
    previous = {path: path.read_bytes() for path in (root / "labels").glob("*.txt")}
    original_replace = annotation_io.os.replace

    def fail_second(source, destination):
        if str(source).endswith(".tmp") and str(destination).endswith("second.txt"):
            raise PermissionError("blocked second label")
        return original_replace(source, destination)

    monkeypatch.setattr(annotation_io.os, "replace", fail_second)
    with pytest.raises(PermissionError):
        sync_review_to_yolo(root)
    assert {path: path.read_bytes() for path in (root / "labels").glob("*.txt")} == previous


def test_cancel_sync_before_commit_preserves_original_labels(tmp_path):
    root = dataset(tmp_path)
    prepare_review_dataset(root)
    previous = (root / "labels" / "frame.txt").read_bytes()
    checks = []

    def cancelled():
        checks.append(True)
        return len(checks) == 3  # After conversion, immediately before commit.

    with pytest.raises(InterruptedError):
        sync_review_to_yolo(root, cancelled)
    assert (root / "labels" / "frame.txt").read_bytes() == previous
    assert not list((root / "labels").glob(".autolabel_*"))
