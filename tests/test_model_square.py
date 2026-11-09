"""Shared catalogue selection and local-weight actions remain independent."""

from types import SimpleNamespace

import pytest


@pytest.fixture
def application(monkeypatch):
    from PyQt6 import QtWidgets
    from autolabel.workbench.resources import resources

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app
    app.processEvents()


def test_square_uses_native_providers_and_models_and_marks_video_compatibility(application):
    from PyQt6 import QtCore
    from autolabel.model_square import ModelSquareDialog

    data = {"Vendor One": {
        "yolo11s-r20240930": {"display_name": "YOLO11s"},
        "segmentation-model": {"display_name": "Segment model"},
    }, "Custom": {
        "load_custom_model": {"display_name": "Load YAML"},
        "import_local_weight": {"display_name": "Import weight"},
        "local-model": {"display_name": "My model"},
    }}
    native = SimpleNamespace(init_model_data=lambda: data)
    dialog = ModelSquareDialog(native)
    dialog.only_video.setChecked(False)
    groups = {dialog.tree.topLevelItem(i).text(0): dialog.tree.topLevelItem(i)
              for i in range(dialog.tree.topLevelItemCount())}
    assert [groups["Vendor One"].child(i).text(0) for i in range(2)] == ["YOLO11s", "Segment model"]
    assert groups["Custom"].childCount() == 1
    supported, unsupported = [groups["Vendor One"].child(i) for i in range(2)]
    dialog.tree.setCurrentItem(unsupported)
    assert not dialog.choose_button.isEnabled()
    assert unsupported.text(1) == "仅图片标注"
    dialog.tree.setCurrentItem(supported)
    assert dialog.choose_button.isEnabled()
    assert dialog.selected_model_id is None
    assert supported.data(0, QtCore.Qt.ItemDataRole.UserRole) == "yolo11_s"
    dialog.search.setText("Vendor One")
    assert not groups["Vendor One"].isHidden() and groups["Custom"].isHidden()
    dialog.accept_selection()
    assert dialog.selected_model_id == "yolo11_s"
    dialog.deleteLater()


def test_square_cancel_preserves_video_model(application, monkeypatch):
    from PyQt6 import QtWidgets
    from autolabel.video_dock import VideoDatasetDock

    dock = VideoDatasetDock(QtWidgets.QMainWindow())
    dock.model_combo.setCurrentIndex(dock.model_combo.findData("yolo11_s"))
    before = dock.config.model_id, dock.config.model_path
    monkeypatch.setattr(dock.model_dialog.model_square, "exec",
                        lambda: QtWidgets.QDialog.DialogCode.Rejected)
    dock.model_dialog.square_button.click()
    assert (dock.config.model_id, dock.config.model_path) == before
    dock.close()


def test_model_task_locks_both_selection_entries(application, monkeypatch):
    from PyQt6 import QtWidgets
    from autolabel.video_dock import VideoDatasetDock

    dock = VideoDatasetDock(QtWidgets.QMainWindow())
    dock._set_model_selection_enabled(False)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                        lambda *a: pytest.fail("weight dialog opened during task"))
    monkeypatch.setattr(dock.model_dialog.model_square, "exec",
                        lambda: pytest.fail("model square opened during task"))
    dock.choose_weight()
    dock.model_dialog.open_model_square()
    assert not dock.weight_button.isEnabled()
    assert not dock.model_dialog.square_button.isEnabled()
    dock._set_model_selection_enabled(True)
    assert dock.weight_button.isEnabled() and dock.model_dialog.square_button.isEnabled()
    dock.close()


def test_local_weight_cancel_invalid_and_success(application, tmp_path, monkeypatch):
    from PyQt6 import QtWidgets
    from autolabel.video_dock import VideoDatasetDock

    dock = VideoDatasetDock(QtWidgets.QMainWindow())
    dock.model_combo.setCurrentIndex(dock.model_combo.findData("yolo11_s"))
    before = dock.config.model_id, dock.config.model_path
    selection = [""]
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", lambda *a: (selection[0], ""))
    errors = []
    monkeypatch.setattr(dock, "_error", errors.append)
    dock.weight_button.click()
    assert (dock.config.model_id, dock.config.model_path) == before
    selection[0] = str(tmp_path / "missing.pt")
    dock.weight_button.click()
    assert errors and (dock.config.model_id, dock.config.model_path) == before
    weight = tmp_path / "trained.pt"
    weight.write_bytes(b"local weight")
    selection[0] = str(weight)
    dock.weight_button.click()
    assert dock.config.model_id == "local_yolo" and dock.config.model_path == str(weight)
    assert weight.read_bytes() == b"local weight"
    assert dock.weight_label.text() == weight.name
    dock.close()
