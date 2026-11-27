"""Shared catalogue selection and local-weight actions remain independent."""

from types import SimpleNamespace

import pytest


@pytest.fixture
def application(monkeypatch):
    from PyQt6 import QtWidgets
    from autolabelt.workbench.resources import resources

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app
    app.processEvents()


def test_square_only_offers_grounding_dino_and_sam(application):
    from PyQt6 import QtCore
    from autolabelt.model_square import ModelSquareDialog, SQUARE_MODEL_IDS

    native = SimpleNamespace(init_model_data=lambda: pytest.fail("unrelated native models were queried"))
    dialog = ModelSquareDialog(native)
    groups = {dialog.tree.topLevelItem(i).text(0): dialog.tree.topLevelItem(i)
              for i in range(dialog.tree.topLevelItemCount())}
    assert set(groups) == {"IDEA-Research", "Meta"}
    offered = {group.child(i).data(0, QtCore.Qt.ItemDataRole.UserRole)
               for group in groups.values() for i in range(group.childCount())}
    assert offered == set(SQUARE_MODEL_IDS)
    assert len(offered) == 4
    supported = groups["IDEA-Research"].child(0)
    dialog.tree.setCurrentItem(supported)
    assert dialog.choose_button.isEnabled()
    assert dialog.selected_model_id is None
    assert supported.data(0, QtCore.Qt.ItemDataRole.UserRole) == "grounding_dino"
    dialog.search.setText("IDEA-Research")
    assert not groups["IDEA-Research"].isHidden() and groups["Meta"].isHidden()
    dialog.accept_selection()
    assert dialog.selected_model_id == "grounding_dino"
    dialog.deleteLater()


def test_square_cancel_preserves_video_model(application, monkeypatch):
    from PyQt6 import QtWidgets
    from autolabelt.video_dock import VideoDatasetDock

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
    from autolabelt.video_dock import VideoDatasetDock

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
    from autolabelt.video_dock import VideoDatasetDock

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


def test_model_details_follow_selection_and_filter_clears_hidden_model(application):
    from PyQt6 import QtCore
    from autolabelt.model_square import ModelSquareDialog

    dialog = ModelSquareDialog()
    items = {}
    for index in range(dialog.tree.topLevelItemCount()):
        provider = dialog.tree.topLevelItem(index)
        for row in range(provider.childCount()):
            item = provider.child(row)
            items[item.data(0, QtCore.Qt.ItemDataRole.UserRole)] = item
    dialog.tree.setCurrentItem(items["grounding_dino"])
    assert dialog.info.facts["许可证"].text() == "Apache-2.0"
    dialog.tree.setCurrentItem(items["sam3_1"])
    assert dialog.info.facts["许可证"].text() == "Meta SAM 3 License"
    assert "权重授权" in dialog.info.guidance.text()
    dialog.search.setText("no such model")
    assert not dialog.choose_button.isEnabled()
    assert dialog.info.facts["许可证"].text() == "—"
    assert dialog.selected_model_id is None
    dialog.close()
