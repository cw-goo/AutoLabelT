"""YOLO-to-native annotation conversion and synchronization tests."""

import json

import numpy as np
import pytest

from autolabelt.annotation_bridge import prepare_review_dataset, sync_review_to_yolo


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
    from autolabelt.workbench.views.labeling.label_file import LabelFile

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
    from autolabelt.validate import validate_dataset

    root = dataset(tmp_path)
    (root / "classes.txt").write_text("1 phone\n0 person\n", encoding="utf-8-sig")
    assert validate_dataset(root)["classes"] == ["person", "phone"]
    prepare_review_dataset(root)
    assert sync_review_to_yolo(root)["classes"] == ["person", "phone"]


def test_sync_uses_external_class_mapping_when_dataset_classes_are_missing(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    dataset(root)
    prepare_review_dataset(root)
    external = tmp_path / "categories.txt"
    (root / "classes.txt").replace(external)
    annotation = root / "images" / "frame.json"
    data = json.loads(annotation.read_text(encoding="utf-8"))
    data["shapes"][0]["label"] = "phone"
    annotation.write_text(json.dumps(data), encoding="utf-8")
    assert sync_review_to_yolo(root, classes_file=external)["classes"] == ["person", "phone"]
    assert (root / "labels" / "frame.txt").read_text(encoding="utf-8").startswith("1 ")
    assert (root / "classes.txt").read_text(encoding="utf-8") == "person\nphone\n"


def test_sync_rejects_external_mapping_with_different_class_order(tmp_path):
    root = dataset(tmp_path)
    prepare_review_dataset(root)
    alternate = tmp_path / "alternate.txt"
    alternate.write_text("phone\nperson\n", encoding="utf-8")
    label = root / "labels" / "frame.txt"
    original = label.read_bytes()
    with pytest.raises(ValueError, match="顺序"):
        sync_review_to_yolo(root, classes_file=alternate)
    assert label.read_bytes() == original


def test_sync_button_opens_class_picker_and_applies_selected_file(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from PyQt6 import QtWidgets
    from autolabelt.video_dock import VideoDatasetDock

    root = dataset(tmp_path)
    prepare_review_dataset(root)
    external = root / "selected.txt"
    (root / "classes.txt").replace(external)
    window = QtWidgets.QMainWindow()
    window.labeling_widget = SimpleNamespace(view=SimpleNamespace(
        image_path=None, save_annotations_for_sync=lambda: True))
    dock = VideoDatasetDock(window)
    dock.config.output_dir = str(root)
    selection = [""]
    starts = []

    def pick(_parent, _title, start, _filter):
        starts.append(start)
        return selection[0], ""

    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", pick)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        lambda _parent, _title, error: pytest.fail(error))
    dock.sync_button.click()
    assert dock.sync_worker is None
    selection[0] = str(external)
    dock.sync_button.click()
    assert dock.sync_worker.wait(5000)
    QtWidgets.QApplication.processEvents()
    assert starts == [str(root), str(root)]
    assert (root / "classes.txt").read_text(encoding="utf-8") == "person\nphone\n"
    assert "YOLO 标签已更新" in dock.log.toPlainText()
    window.close()


@pytest.mark.parametrize("saved", [True, False], ids=["saved-json", "pending-canvas-edit"])
def test_sync_button_writes_canvas_changes_to_original_split_label(tmp_path, monkeypatch, saved):
    from PIL import Image
    from PyQt6 import QtCore, QtWidgets

    from autolabelt.desktop import AutoLabelMainWindow
    from autolabelt.workbench import config
    from autolabelt.workbench.resources import resources

    root = dataset(tmp_path)
    prepare_review_dataset(root)
    training_images = root / "images" / "train"
    training_labels = root / "labels" / "train"
    training_images.mkdir()
    training_labels.mkdir()
    image = training_images / "frame.png"
    Image.new("RGB", (200, 100)).save(image)
    (root / "images" / "frame.jpg").unlink()
    annotation = image.with_suffix(".json")
    data = json.loads((root / "images" / "frame.json").read_text(encoding="utf-8"))
    data["imagePath"] = image.name
    annotation.write_text(json.dumps(data), encoding="utf-8")
    (root / "images" / "frame.json").unlink()
    target = training_labels / "frame.txt"
    (root / "labels" / "frame.txt").replace(target)
    original = target.read_bytes()

    validation_image = root / "images" / "val" / image.name
    validation_image.parent.mkdir()
    Image.new("RGB", (200, 100)).save(validation_image)
    other_target = root / "labels" / "val" / "frame.txt"
    other_target.parent.mkdir()
    other_target.write_bytes(original)
    stale_root = tmp_path / "earlier_project"
    stale_root.mkdir()
    dataset(stale_root)
    stale_target = stale_root / "labels" / "frame.txt"
    selected = []

    def pick(_parent, _title, start, _filter):
        selected.append(start)
        return str(root / "classes.txt"), ""

    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", pick)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        lambda _parent, _title, error: pytest.fail(error))
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        lambda *_: pytest.fail("sync should save the review automatically"))
    cfg = config.get_default_config()
    cfg.update(auto_save=False, exif_scan_enabled=False)
    app = QtWidgets.QApplication.instance()
    window = AutoLabelMainWindow(app, config=cfg)
    try:
        dock = window.video_dock
        view = window.labeling_widget.view
        dock.config.output_dir = str(stale_root)
        view.import_image_folder(str(training_images))
        assert view.image_path == str(image)
        shape = view.canvas.shapes[0]
        shape.label = "phone"
        for point in shape.points:
            point.setX(point.x() + 20)
        view.set_dirty()
        if saved:
            view.save_file()
            assert json.loads(annotation.read_text(encoding="utf-8"))["shapes"][0]["label"] == "phone"
        assert target.read_bytes() == original

        dock.sync_button.click()
        assert dock.sync_worker.wait(5000)
        app.processEvents()
        assert selected == [str(root / "classes.txt")]
        assert target.read_text(encoding="utf-8").strip() == "1 0.600000 0.500000 0.400000 0.600000"
        assert other_target.read_bytes() == original
        assert stale_target.read_bytes() == original
        assert not view.dirty
        log = dock.log.toPlainText()
        assert f"YOLO 标签已更新：1 个文件 · {root / 'labels'}" in log
        assert str(target) not in log
        assert "1 张图片未保存复核 JSON" in log

        view.canvas.select_shapes([shape])
        view.delete_selected_shape()
        dock.sync_button.click()
        assert dock.sync_worker.wait(5000)
        app.processEvents()
        assert target.read_bytes() == b""
    finally:
        view.set_clean()
        window.close()
        window.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


def test_sync_commit_failure_preserves_all_existing_labels(tmp_path, monkeypatch):
    import cv2
    from autolabelt.workbench.views.labeling import annotation_io

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
