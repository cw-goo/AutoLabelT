"""Regression coverage for fresh inference and annotation file access."""

from types import SimpleNamespace

import pytest


class Palette(list):
    def count(self):
        return len(self)

    def item(self, index):
        return SimpleNamespace(data=lambda _: self[index])


def test_io_open_releases_handles_on_success_and_failure(tmp_path):
    from autolabelt.workbench.views.labeling.utils._io import io_open

    path = tmp_path / "annotation.json"
    with io_open(path, "w") as stream:
        stream.write("{}")
    assert stream.closed
    with pytest.raises(RuntimeError):
        with io_open(path, "r") as stream:
            raise RuntimeError("conversion failed")
    assert stream.closed
    path.unlink()


def test_fresh_batch_clears_selected_sidecars_and_palette_before_inference(tmp_path, monkeypatch):
    from autolabelt.workbench.views.labeling.utils import batch

    images, output = tmp_path / "images", tmp_path / "output"
    images.mkdir()
    output.mkdir()
    first, second = images / "one.jpg", images / "two.jpg"
    for image in (first, second):
        image.touch()
        image.with_suffix(".json").write_text('{"shapes": [{"label": "old"}]}')
        (output / image.with_suffix(".json").name).write_text("{}")
    unrelated = images / "metadata.json"
    unrelated.write_text("{}")
    yolo = tmp_path / "one.txt"
    yolo.write_text("1 0.5 0.5 0.5 0.5\n")

    palette, history, events = Palette(["old"]), ["old"], []
    state = SimpleNamespace(
        image_list=[str(first), str(second)], filename=str(first),
        output_dir=str(output), _batch_start_index=0, _batch_end_index=1,
        _batch_clear_annotations=True,
        unique_label_list=palette,
        label_dialog=SimpleNamespace(label_list=SimpleNamespace(
            clear=history.clear, addItems=history.extend)),
        auto_labeling_widget=SimpleNamespace(model_manager=SimpleNamespace(
            loaded_model_config={"classes": ["hat", "hat2"]})),
        load_labels=palette.extend,
        set_clean=lambda: events.append("clean"),
        load_file=lambda path: events.append(("reload", path)),
    )

    def start(_state):
        assert not first.with_suffix(".json").exists()
        assert not (output / "one.json").exists()
        assert palette == history == ["hat", "hat2"]
        events.append("inference")

    monkeypatch.setattr(batch, "show_progress_dialog_and_process", start)
    batch._start_batch_processing(state)
    assert events == ["clean", ("reload", str(first)), "inference"]
    assert second.with_suffix(".json").exists()
    assert (output / "two.json").exists()
    assert unrelated.exists() and yolo.read_text().startswith("1 ")
    state._batch_transaction.rollback()
    assert first.with_suffix(".json").exists()
    assert (output / "one.json").exists()


def test_batch_does_not_start_when_old_annotations_cannot_be_deleted(monkeypatch, tmp_path):
    from autolabelt.workbench.views.labeling.utils import batch

    messages = []
    state = SimpleNamespace(_batch_clear_annotations=True, _batch_processing_active=False,
                            image_list=[str(tmp_path / "image.jpg")], output_dir=None,
                            _batch_start_index=0, _batch_end_index=1)

    def denied(_state):
        raise PermissionError("denied: old.json")

    monkeypatch.setattr(batch.AnnotationTransaction, "preflight", denied)
    monkeypatch.setattr(batch, "show_progress_dialog_and_process",
                        lambda _: pytest.fail("inference started after cleanup failed"))
    monkeypatch.setattr(batch, "Popup", lambda message, *a, **kw: SimpleNamespace(
        show_popup=lambda *a, **kw: messages.append(message)))
    batch._start_batch_processing(state)
    assert not state._batch_processing_active
    assert len(messages) == 1 and "denied: old.json" in messages[0]


def test_batch_save_permission_error_is_propagated(tmp_path, monkeypatch):
    from autolabelt.workbench.views.labeling.utils import batch

    path = tmp_path / "one.jpg"
    path.with_suffix(".json").write_text('{"shapes": []}')

    def denied(*args):
        raise PermissionError("denied annotation")

    monkeypatch.setattr(batch, "io_open", denied)
    state = SimpleNamespace(output_dir=None, auto_labeling_widget=SimpleNamespace(
        model_manager=SimpleNamespace(loaded_model_config={})))
    result = SimpleNamespace(shapes=[], description="", replace=True)
    with pytest.raises(PermissionError, match="denied annotation"):
        batch.save_auto_labeling_result(state, str(path), result)


@pytest.mark.parametrize("cancelled", [False, True])
def test_batch_finish_commits_only_on_success_and_restores_on_cancel(tmp_path, monkeypatch, cancelled):
    from autolabelt.workbench.views.labeling.utils import batch
    from autolabelt.workbench.views.labeling.annotation_io import AnnotationTransaction, atomic_write_json

    image = tmp_path / "one.jpg"
    image.touch()
    annotation = image.with_suffix(".json")
    annotation.write_text("old")
    transaction = AnnotationTransaction([annotation])
    transaction.preflight()
    transaction.clear()
    atomic_write_json(transaction.stage_path(annotation), {"new": True})
    palette, history, messages = Palette(["hat"]), ["hat"], []
    state = SimpleNamespace(
        _batch_processing_active=True, _batch_transaction=transaction,
        _batch_palette=["old"], cancel_processing=cancelled,
        image_list=[str(image)], current_index=0, fn_to_index={str(image): 0},
        unique_label_list=palette, load_labels=palette.extend,
        auto_labeling_widget=SimpleNamespace(model_manager=SimpleNamespace(
            loaded_model_config={"classes": ["hat"]})),
        label_dialog=SimpleNamespace(label_list=SimpleNamespace(clear=history.clear, addItems=history.extend)),
        import_image_folder=lambda *a, **kw: None, load_file=lambda _: None,
        file_list_widget=SimpleNamespace(blockSignals=lambda _: False, setCurrentRow=lambda _: None),
        tr=lambda text: text,
    )
    monkeypatch.setattr(batch.QApplication, "processEvents", lambda: None)
    monkeypatch.setattr(batch, "Popup", lambda message, *a, **kw: SimpleNamespace(
        show_popup=lambda *a, **kw: messages.append(message)))
    batch.finish_processing(state, SimpleNamespace(close=lambda: None))
    assert not state._batch_processing_active
    if cancelled:
        assert annotation.read_text() == "old" and palette == ["old"]
        assert messages == ["已取消，旧标注已恢复。"]
    else:
        assert annotation.read_text().find('"new": true') >= 0
        assert palette == ["hat"] and "successfully" in messages[0]
    assert not list(tmp_path.glob(".autolabel_*"))


@pytest.mark.parametrize("preserve", [False, True])
def test_yolo_import_replaces_old_class_palette_unless_preserving(tmp_path, monkeypatch, preserve):
    import cv2
    import numpy as np
    from PyQt6 import QtCore, QtWidgets
    from autolabelt.workbench.views.labeling.utils import upload
    from autolabelt.workbench.views.labeling.widgets.unique_label_qlist_widget import UniqueLabelQListWidget

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    image = images / "one.jpg"
    cv2.imwrite(str(image), np.zeros((40, 80, 3), dtype=np.uint8))
    (labels / "one.txt").write_text("1 0.5 0.5 0.5 0.5\n")
    classes = tmp_path / "classes.txt"
    classes.write_text("0 hat\n1 hat2\n", encoding="utf-8-sig")
    parent = QtWidgets.QWidget()
    parent.filename, parent.output_dir = str(image), None
    parent.may_continue = lambda: True
    parent.unique_label_list = UniqueLabelQListWidget()
    parent.unique_label_list.addItem(parent.unique_label_list.create_item_from_label("old"))
    parent.label_dialog = SimpleNamespace(label_list=QtWidgets.QListWidget())
    parent.label_dialog.label_list.addItem("old")
    parent._get_rgb_by_label = lambda _: (255, 0, 0)
    from autolabelt.workbench.views.labeling.label_widget import LabelingWidget
    parent.load_labels = lambda labels: LabelingWidget.load_labels(parent, labels)
    messages = []

    def accept_dialog(dialog):
        dialog.findChild(QtWidgets.QLineEdit).setText(str(labels))
        dialog.findChild(QtWidgets.QCheckBox).setChecked(preserve)
        return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(QtWidgets.QDialog, "exec", accept_dialog)
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", lambda _: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", lambda *a: (str(classes), ""))
    monkeypatch.setattr(upload, "_refresh_after_annotation_upload", lambda _: None)
    monkeypatch.setattr(upload, "Popup", lambda message, *a, **kw: SimpleNamespace(
        show_popup=lambda *a, **kw: messages.append(message)))
    upload.upload_yolo_annotation(parent, "hbb", 128)
    palette = [parent.unique_label_list.item(i).data(QtCore.Qt.ItemDataRole.UserRole)
               for i in range(parent.unique_label_list.count())]
    assert palette == (["old", "hat", "hat2"] if preserve else ["hat", "hat2"])
    assert "successfully" in messages[-1]
    assert image.with_suffix(".json").is_file()
    parent.deleteLater()
    assert app is not None
