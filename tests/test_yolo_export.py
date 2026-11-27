"""Regression tests for empty YOLO exports and class mapping errors."""

import json
from types import SimpleNamespace

import pytest

from autolabelt.workbench.views.labeling.label_converter import LabelConverter


def annotation(tmp_path, shape_type="rectangle", label="person"):
    path = tmp_path / "frame.json"
    points = [[60, 20], [140, 20], [140, 80], [60, 80]]
    path.write_text(json.dumps({
        "imageWidth": 200,
        "imageHeight": 100,
        "shapes": [{"label": label, "shape_type": shape_type, "points": points}],
    }), encoding="utf-8")
    return path


@pytest.mark.parametrize("mode, shape_type, columns", [
    ("hbb", "rectangle", 5),
    ("seg", "polygon", 9),
    ("obb", "rotation", 9),
])
def test_export_accepts_bom_and_class_whitespace(tmp_path, mode, shape_type, columns):
    classes = tmp_path / "classes.txt"
    classes.write_text(" person \nphone\n", encoding="utf-8-sig")
    source = annotation(tmp_path, shape_type)
    destination = tmp_path / "frame.txt"
    converter = LabelConverter(classes_file=str(classes))

    assert converter.custom_to_yolo(str(source), str(destination), mode) is False
    values = destination.read_text(encoding="utf-8").split()
    assert len(values) == columns
    assert values[0] == "0"
    if mode == "hbb":
        assert list(map(float, values[1:])) == pytest.approx([0.5, 0.5, 0.4, 0.6])


@pytest.mark.parametrize("mode, shape_type, columns", [
    ("hbb", "rectangle", 5), ("seg", "polygon", 9), ("obb", "rotation", 9),
])
@pytest.mark.parametrize("content", [
    "0 hat\n1 hat2\n2 hat3\n",
    "2 hat3\n0 hat\n1 hat2\n",
    " 0\that \n 1\that2 \n 2\that3 \n",
    "hat\nhat2\nhat3\n",
])
def test_export_maps_hat2_to_class_one(tmp_path, mode, shape_type, columns, content):
    classes = tmp_path / "classes.txt"
    classes.write_text(content, encoding="utf-8-sig")
    source = annotation(tmp_path, shape_type, label="hat2")
    destination = tmp_path / "frame.txt"
    converter = LabelConverter(classes_file=str(classes))

    assert converter.classes == ["hat", "hat2", "hat3"]
    assert converter.custom_to_yolo(str(source), str(destination), mode) is False
    values = destination.read_text(encoding="utf-8").split()
    assert len(values) == columns
    assert values[0] == "1"


@pytest.mark.parametrize("content", ["0 hat\n0 hat2\n", "0 hat\n2 hat2\n"])
def test_numbered_classes_reject_duplicate_or_missing_ids(tmp_path, content):
    classes = tmp_path / "classes.txt"
    classes.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="class IDs"):
        LabelConverter(classes_file=str(classes))


def test_plain_classes_preserve_multiword_and_numeric_names(tmp_path):
    classes = tmp_path / "classes.txt"
    classes.write_text("hard hat\n7 segment display\n42\n", encoding="utf-8")

    assert LabelConverter(classes_file=str(classes)).classes == [
        "hard hat", "7 segment display", "42"]


def test_numbered_classes_import_export_roundtrip(tmp_path):
    from PIL import Image

    classes = tmp_path / "classes.txt"
    classes.write_text("0 hat\n1 hat2\n2 hat3\n", encoding="utf-8")
    image = tmp_path / "frame.jpg"
    Image.new("RGB", (200, 100)).save(image)
    source = tmp_path / "source.txt"
    source.write_text("1 0.5 0.5 0.4 0.6\n", encoding="utf-8")
    native = tmp_path / "frame.json"
    exported = tmp_path / "exported.txt"
    converter = LabelConverter(classes_file=str(classes))

    converter.yolo_to_custom(str(source), str(native), str(image), "hbb")
    data = json.loads(native.read_text(encoding="utf-8"))
    assert data["shapes"][0]["label"] == "hat2"
    assert converter.custom_to_yolo(str(native), str(exported), "hbb") is False
    assert exported.read_text(encoding="utf-8") == (
        "1 0.500000 0.500000 0.400000 0.600000\n")


def test_yolo_export_dialog_with_numbered_classes(tmp_path, monkeypatch):
    from PyQt6 import QtWidgets
    from autolabelt.workbench.views.labeling.utils import export

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    images = tmp_path / "images"
    images.mkdir()
    image = images / "frame.jpg"
    image.touch()
    source = annotation(images, label="hat2")
    classes = tmp_path / "classes.txt"
    original = "0 hat\n1 hat2\n2 hat3\n"
    classes.write_text(original, encoding="utf-8")

    parent = QtWidgets.QWidget()
    parent.filename = str(image)
    parent.last_open_dir = str(images)
    parent.image_list = [str(image)]
    parent.output_dir = None
    parent.label_file = SimpleNamespace(filename=str(source))
    parent.may_continue = lambda: True
    errors, messages = [], []

    class Popup:
        def __init__(self, message, *args, **kwargs):
            messages.append(message)

        def show_popup(self, *args, **kwargs):
            pass

    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                        lambda *args: (str(classes), ""))
    monkeypatch.setattr(QtWidgets.QDialog, "exec",
                        lambda self: QtWidgets.QDialog.DialogCode.Accepted)
    monkeypatch.setattr(export, "_show_yolo_export_error",
                        lambda *args: errors.append(str(args[-1])))
    monkeypatch.setattr(export, "Popup", Popup)

    export.export_yolo_annotation(parent, "hbb")

    assert errors == []
    assert messages and "successfully" in messages[-1]
    assert (tmp_path / "labels" / "frame.txt").read_text(
        encoding="utf-8") == "1 0.500000 0.500000 0.400000 0.600000\n"
    assert classes.read_text(encoding="utf-8") == original
    parent.deleteLater()
    assert application is not None


@pytest.mark.parametrize("mode, shape_type", [
    ("hbb", "rectangle"), ("seg", "polygon"), ("obb", "rotation"),
])
def test_unknown_class_is_reported_without_overwriting(tmp_path, mode, shape_type):
    classes = tmp_path / "classes.txt"
    classes.write_text("phone\n", encoding="utf-8")
    source = annotation(tmp_path, shape_type)
    destination = tmp_path / "frame.txt"
    previous = "1 0.5 0.5 0.4 0.6\n"
    destination.write_text(previous, encoding="utf-8")

    with pytest.raises(ValueError, match="person"):
        LabelConverter(classes_file=str(classes)).custom_to_yolo(
            str(source), str(destination), mode)
    assert destination.read_text(encoding="utf-8") == previous


def test_incompatible_export_mode_is_reported(tmp_path):
    classes = tmp_path / "classes.txt"
    classes.write_text("person\n", encoding="utf-8")
    source = annotation(tmp_path, "polygon")
    destination = tmp_path / "frame.txt"

    with pytest.raises(ValueError, match="hbb"):
        LabelConverter(classes_file=str(classes)).custom_to_yolo(
            str(source), str(destination), "hbb")
    assert not destination.exists()


def test_empty_annotation_still_exports_negative_sample(tmp_path):
    source = tmp_path / "frame.json"
    source.write_text(json.dumps({
        "imageWidth": 200, "imageHeight": 100, "shapes": [],
    }), encoding="utf-8")
    destination = tmp_path / "frame.txt"

    assert LabelConverter().custom_to_yolo(str(source), str(destination), "hbb") is True
    assert destination.read_text(encoding="utf-8") == ""


@pytest.mark.parametrize("content", ["", "person\n\nphone\n"])
def test_invalid_classes_file_does_not_shift_class_ids(tmp_path, content):
    classes = tmp_path / "classes.txt"
    classes.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="class names"):
        LabelConverter(classes_file=str(classes))


def test_export_reads_annotation_saved_to_custom_path(tmp_path):
    from autolabelt.workbench.views.labeling.utils.export import _get_yolo_label_file

    classes = tmp_path / "classes.txt"
    classes.write_text("person\n", encoding="utf-8")
    source = annotation(tmp_path)
    saved = tmp_path / "custom_name.json"
    source.rename(saved)
    image = tmp_path / "frame.jpg"
    parent = SimpleNamespace(
        filename=str(image), output_dir=None,
        label_file=SimpleNamespace(filename=str(saved)),
    )
    destination = tmp_path / "frame.txt"

    source_file = _get_yolo_label_file(parent, str(image))
    assert LabelConverter(classes_file=str(classes)).custom_to_yolo(
        source_file, str(destination), "hbb") is False
    assert destination.read_text(encoding="utf-8").startswith("0 ")
    # Other images continue to use the configured annotation directory.
    parent.output_dir = str(tmp_path / "annotations")
    other = tmp_path / "other.jpg"
    assert _get_yolo_label_file(parent, str(other)) == str(
        tmp_path / "annotations" / "other.json")
