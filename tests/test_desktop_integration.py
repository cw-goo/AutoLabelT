"""Integrated desktop workbench and live review tests."""

from pathlib import Path


def test_restored_workbench_has_video_dock(monkeypatch, tmp_path):
    import cv2
    import numpy as np
    from PyQt6 import QtCore, QtGui, QtWidgets
    from autolabelt.workbench import config as settings
    from autolabelt.workbench.resources import resources
    from autolabelt.desktop import AutoLabelMainWindow

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    settings.current_config_file = str(tmp_path / ".autolabeltrc")
    window = AutoLabelMainWindow(application, config=settings.get_config())
    temporary_settings = QtCore.QSettings(
        str(tmp_path / "window.ini"), QtCore.QSettings.Format.IniFormat)
    window.settings = temporary_settings
    window.labeling_widget.view.settings = temporary_settings
    assert window.labeling_widget.view.__class__.__name__ == "LabelingWidget"
    assert window.windowTitle() == "AutoLabelT"
    assert not window.windowIcon().isNull()
    assert not window.windowIcon().pixmap(64, 64).isNull()
    assert application.applicationName() == "AutoLabel"
    assert application.applicationDisplayName() == "AutoLabelT"
    assert application.applicationVersion() == "0.1.0"
    assert window.video_dock.windowTitle() == "视频数据集"
    assert window.video_dock.sync_button.text() == "同步更新标签"
    view = window.labeling_widget.view
    visible = set(view.tools.actions())
    assert view.actions.create_rectangle_mode in visible
    assert view.actions.toggle_auto_labeling_widget in visible
    assert view.actions.open_next_image in visible
    assert view.actions.open_chatbot in visible
    assert view.actions.open_vqa in visible
    assert view.actions.open_pointcloud in visible
    assert view.auto_labeling_widget is not None
    assert view.actions.open_next_image in view.menus.file.actions()
    sidebar_action = window.findChild(QtGui.QAction, "rightSidebarAction")
    assert sidebar_action is not None
    assert window.findChild(QtGui.QAction, "nativeImageModelsAction") is None
    assert window.video_dock.titleBarWidget().height() == 0
    window.show()
    application.processEvents()
    assert window.video_dock.isVisible()
    toolbar = window.findChild(QtWidgets.QToolBar, "autolabelToolbar")
    assert toolbar.isVisible()
    assert toolbar.findChild(QtWidgets.QLabel, "brandName").text() == "AutoLabelT"
    assert toolbar.widgetForAction(toolbar.actions()[-1]) is window.sidebar_toggle
    expanded_width = window.video_dock.width()
    toggle_position = window.sidebar_toggle.pos()
    window.sidebar_toggle.click()
    application.processEvents()
    assert window.video_dock.collapsed
    assert not window.video_dock.isVisible()
    assert not window.video_dock.sidebar_scroll.isVisible()
    assert window.sidebar_toggle.isVisible()
    assert window.sidebar_toggle.pos() == toggle_position
    assert window.video_dock.training_button.text() == "开始训练"
    assert sidebar_action.text() == ""
    assert not sidebar_action.icon().isNull()
    assert sidebar_action.toolTip() == "展开右侧边栏"
    window.sidebar_toggle.click()
    application.processEvents()
    assert window.video_dock.isVisible()
    assert window.video_dock.width() == expanded_width
    assert not window.video_dock.collapsed
    assert window.video_dock.training_footer.isVisible()
    dialog = window.video_dock.model_dialog
    assert not dialog.findChildren(QtWidgets.QTabWidget)
    assert not hasattr(dialog, "image_page")
    assert dialog.native_models is view.auto_labeling_widget
    native = view.auto_labeling_widget
    assert native.model_manager.loaded_model_config is None
    assert native.model_status_label.text() == native.model_manager.tr("No model selected.")
    view.auto_labeling_widget.model_selection_button.click()
    assert view.auto_labeling_widget.model_dropdown.isVisible()
    assert "import_local_weight" in view.auto_labeling_widget.model_dropdown.models_data["Custom"]
    view.auto_labeling_widget.model_dropdown.hide()
    window.video_dock.open_model_dialog()
    assert dialog.isVisible()
    dialog.hide()
    dock = window.video_dock
    dock.local_model_path = str(tmp_path / "staffv2.pt")
    dock.model_combo.setCurrentIndex(dock.model_combo.findData("grounding_dino"))
    dock.model_combo.setCurrentIndex(dock.model_combo.findData("local_yolo"))
    assert dock.config.model_path == dock.local_model_path

    images = tmp_path / "dataset" / "images"
    labels = tmp_path / "dataset" / "labels"
    images.mkdir(parents=True)
    labels.mkdir()
    (tmp_path / "dataset" / "classes.txt").write_text("person\n", encoding="utf-8")
    cv2.imwrite(str(images / "frame.jpg"), np.zeros((60, 80, 3), dtype=np.uint8))
    (labels / "frame.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    from autolabelt.annotation_bridge import prepare_review_dataset
    prepare_review_dataset(tmp_path / "dataset")
    dock.config.output_dir = str(tmp_path / "dataset")
    dock.open_review()
    assert len(view.image_list) == 1
    dock.live_review_dir = str(images)
    second = images / "frame_2.jpg"
    cv2.imwrite(str(second), np.zeros((60, 80, 3), dtype=np.uint8))
    (labels / "frame_2.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    from autolabelt.annotation_bridge import prepare_review_image
    prepare_review_image(tmp_path / "dataset", second)
    dock._show_live_image(str(second))
    application.processEvents()
    assert str(second) in view.image_list
    assert view.filename == str(second)
    assert len(view.canvas.shapes) == 1
    third = images / "frame_3.jpg"
    cv2.imwrite(str(third), np.zeros((60, 80, 3), dtype=np.uint8))
    view.dirty = True
    dock._show_live_image(str(third))
    assert str(third) in view.image_list
    assert view.filename == str(second)
    view.dirty = False
    other_images = tmp_path / "other_dataset" / "images"
    other_images.mkdir(parents=True)
    (other_images.parent / "classes.txt").write_text("hat\nhat2\n", encoding="utf-8")
    cv2.imwrite(str(other_images / "other.jpg"), np.zeros((60, 80, 3), dtype=np.uint8))
    view.import_image_folder(str(other_images))
    from autolabelt.workbench.views.labeling.label_schema import palette_names
    assert palette_names(view) == ["hat", "hat2"]
    assert [view.label_dialog.label_list.item(i).text()
            for i in range(view.label_dialog.label_list.count())] == ["hat", "hat2"]
    from autolabelt.workbench.views.labeling.annotation_io import AnnotationTransaction
    target = other_images / "other.json"
    with AnnotationTransaction([target]) as transaction:
        view._batch_transaction = transaction
        assert view.save_labels(str(target))
        assert not target.exists()
        assert transaction.staged[target].is_file()
        assert view.label_file.filename == str(target)
    del view._batch_transaction
    view.load_labels(["third"], clear_existing=False)
    from autolabelt.workbench.services.auto_labeling.types import AutoLabelingResult
    with monkeypatch.context() as scope:
        scope.setattr(view.auto_labeling_widget.model_manager, "loaded_model_config",
                      {"classes": ["hat", "hat2"]})
        view.new_shapes_from_auto_labeling(AutoLabelingResult([], replace=True))
        assert palette_names(view) == ["hat", "hat2"]
    view.set_clean()
    window.close()
    application.processEvents()


def test_video_sidebar_spacing_and_scroll_keep_controls_reachable(monkeypatch):
    from PyQt6 import QtCore, QtWidgets
    from autolabelt.appearance import stylesheet
    from autolabelt.video_dock import VideoDatasetDock

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = QtWidgets.QMainWindow()
    window.setStyleSheet(stylesheet("light"))
    dock = VideoDatasetDock(window)
    window.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
    dock.stats.setText("构建中\n整批 1,560 / 2,000 张")
    window.resize(1000, 1100)
    window.show()
    application.processEvents()

    scroll = dock.sidebar_scroll
    assert scroll.verticalScrollBar().maximum() == 0
    for upper, lower in zip(dock.workflow_cards, dock.workflow_cards[1:]):
        assert lower.y() - upper.geometry().bottom() - 1 == dock.workflow_layout.spacing() == 12
    assert dock.batch_progress.y() - dock.start_button.geometry().bottom() - 1 >= 10
    assert dock.stats.y() - dock.batch_progress.geometry().bottom() - 1 >= 8
    assert dock.stats.height() >= 2 * dock.stats.fontMetrics().lineSpacing()

    window.resize(1000, 500)
    application.processEvents()
    viewport = scroll.viewport()
    assert scroll.verticalScrollBar().maximum() > 0
    assert dock.workflow_body.width() == viewport.width()
    footer_y = dock.training_button.mapTo(dock, QtCore.QPoint()).y()
    scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
    application.processEvents()
    review_corner = dock.sync_button.mapTo(viewport, dock.sync_button.rect().bottomRight())
    assert viewport.rect().contains(review_corner)
    assert dock.training_button.mapTo(dock, QtCore.QPoint()).y() == footer_y
    window.close()
    application.processEvents()


def test_image_model_accepts_local_yolo_weight(monkeypatch, tmp_path):
    import pytest
    import yaml
    from PyQt6 import QtWidgets
    from autolabelt.workbench import config as settings
    from autolabelt.workbench.resources import resources
    from autolabelt.desktop import AutoLabelMainWindow

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    settings.current_config_file = str(tmp_path / ".autolabeltrc")
    window = AutoLabelMainWindow(application, config=settings.get_config())
    widget = window.labeling_widget.view.auto_labeling_widget
    monkeypatch.setattr(
        "autolabelt.workbench.views.labeling.widgets.auto_labeling.auto_labeling.get_work_directory",
        lambda: str(tmp_path))
    loaded = []
    monkeypatch.setattr(widget, "load_custom_model_config",
                        lambda path: loaded.append(path) or True)
    weight = tmp_path / "trained.pt"
    weight.write_bytes(b"weight")
    monkeypatch.setattr(
        "autolabelt.workbench.views.labeling.widgets.auto_labeling.auto_labeling.QInputDialog.getText",
        lambda *_: ("person, car", True))
    assert widget.prompt_and_import_local_weight(weight)
    config = yaml.safe_load(Path(loaded[0]).read_text(encoding="utf-8"))
    assert config["type"] == "autolabel_local_yolo"
    assert config["model_path"] == str(weight)
    assert config["display_name"] == "本地 YOLO · trained"
    assert config["classes"] == ["person", "car"]
    from autolabelt.models.image_yolo import parse_class_names
    assert parse_class_names("人，汽车") == ["人", "汽车"]
    with pytest.raises(ValueError):
        parse_class_names("person, ")
    with pytest.raises(ValueError):
        parse_class_names("person, Person")
    window.close()


def test_local_yolo_image_model_converts_detection_to_native_shape(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import numpy as np
    from PyQt6 import QtGui
    import ultralytics
    from autolabelt.models.image_yolo import LocalYOLOImageModel

    class Tensor:
        def __init__(self, values):
            self.values = np.asarray(values)

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class FakeYOLO:
        task = "detect"
        names = {0: "old name"}

        def __init__(self, path):
            self.path = path

        def predict(self, **kwargs):
            assert kwargs["source"].shape == (40, 80, 3)
            assert kwargs["classes"] == [0]
            return [SimpleNamespace(
                names={0: "person"},
                boxes=SimpleNamespace(xyxy=Tensor([[10, 5, 30, 25]]),
                                      cls=Tensor([0]), conf=Tensor([0.9])))]

    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    weight = tmp_path / "trained.pt"
    weight.write_bytes(b"weight")
    model = LocalYOLOImageModel(
        {"model_path": str(weight), "classes": ["worker"]}, lambda _: None)
    image = QtGui.QImage(80, 40, QtGui.QImage.Format.Format_RGB32)
    image.fill(QtGui.QColor("white"))
    result = model.predict_shapes(image)
    assert len(result.shapes) == 1
    assert result.shapes[0].label == "worker"
    assert result.shapes[0].shape_type == "rectangle"
    assert result.shapes[0].score == 0.9


def test_model_download_status_and_progress(monkeypatch, tmp_path):
    import json
    from PyQt6 import QtWidgets
    from autolabelt.models.downloader import model_download_state, model_path
    from autolabelt.video_dock import VideoDatasetDock

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("AUTOLABEL_MODEL_DIR", str(tmp_path))
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dock = VideoDatasetDock(QtWidgets.QMainWindow())
    index = dock.model_combo.findData("grounding_dino")
    dock.model_combo.setCurrentIndex(index)
    assert model_download_state("grounding_dino") == "missing"
    assert "未下载" in dock.model_status.text()
    location = model_path("grounding_dino")
    location.mkdir(parents=True)
    (location / "manifest.json").write_text(json.dumps({
        "model_id": "grounding_dino", "revision": "a2bb814dd30d776dcf7e30523b00659f4f141c71",
        "status": "downloaded"}), encoding="utf-8")
    (location / "model.safetensors").write_bytes(b"test")
    dock.model_changed()
    assert "本机已有" in dock.model_status.text()
    dock._model_progress({"downloaded": 5, "total": 10, "file": "weights", "unit": "files"})
    assert dock.model_progress.value() == 50
    assert "5/10" in dock.model_status.text()
    dock._model_progress({"downloaded": 1048576, "total": 2097152, "file": "model.pt"})
    assert "1.0 MB / 2.0 MB" in dock.model_status.text()
    dock.close()
    application.processEvents()


def test_model_dialog_sources_and_required_labels(monkeypatch):
    from PyQt6 import QtWidgets
    from autolabelt.video_dock import VideoDatasetDock

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dock = VideoDatasetDock(QtWidgets.QMainWindow())
    assert dock.model_button.text() == "模型加载"
    assert dock.model_combo.parent() is dock
    dock.model_button.click()
    dialog = dock.model_dialog
    assert dialog.isVisible()
    assert dialog.square_button.text() == "打开模型广场"
    assert dock.weight_button.text() == "选择本地权重"
    selected = ["grounding_dino"]

    def choose():
        dialog.model_square.selected_model_id = selected[0]
        return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(dialog.model_square, "exec", choose)
    dialog.square_button.click()
    assert dock.config.model_id == "grounding_dino"
    assert "huggingface.co" in dialog.source_link.text()
    selected[0] = "sam3_1"
    dialog.square_button.click()
    assert dock.config.model_id == "sam3_1"
    assert "huggingface.co/facebook/sam3.1" in dialog.source_link.text()
    selected[0] = "yolo26_x"
    dialog.square_button.click()
    assert dock.config.model_id == "sam3_1"
    selected[0] = "grounding_dino"
    dialog.square_button.click()
    dock.prompts.setText(" ")
    try:
        dock._current_config()
    except ValueError as error:
        assert "至少输入一个检测类别" in str(error)
    else:
        assert False, "空检测类别必须被拒绝"
    dock.prompts.setText("person, phone")
    dialog.confirm_selection()
    assert not dialog.isVisible()
    assert "person, phone" in dock.model_summary.text()
    dock.model_combo.setCurrentIndex(dock.model_combo.findData("local_yolo"))
    try:
        dock._current_config()
    except ValueError as error:
        assert "只检测 person" in str(error)
    else:
        assert False, "本地基线不应声称支持任意类别"
    dock.close()
    application.processEvents()


def test_grouped_parameter_dialog_keeps_baseline(monkeypatch):
    from PyQt6 import QtWidgets
    from autolabelt.core import ProjectConfig
    from autolabelt.video_dock import PARAMETER_GROUPS, ParameterDialog

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dialog = ParameterDialog(ProjectConfig(step_single=17))
    assert len(PARAMETER_GROUPS) == 3
    assert dialog.controls["step_single"].value() == 17
    spin = dialog.controls["step_single"]
    assert spin.buttonSymbols() == QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons
    buttons = {button.accessibleName(): button for button in
               dialog.findChildren(QtWidgets.QPushButton)}
    buttons["调大单目标跳帧"].click()
    assert spin.value() == 18
    buttons["调小单目标跳帧"].click()
    assert spin.value() == 17
    ratio = dialog.controls["small_target_ratio"]
    buttons["调大小目标占比"].click()
    assert ratio.value() == 0.65
    buttons["调小小目标占比"].click()
    assert ratio.value() == 0.60
    spin.setValue(1)
    buttons["调小单目标跳帧"].click()
    assert spin.value() == 1
    dialog.restore_defaults()
    assert dialog.controls["step_single"].value() == ProjectConfig().step_single
    assert dialog.controls["device"].currentData() == "auto"
    dialog.close()
    application.processEvents()


def test_video_task_completion_retains_session_model(monkeypatch, tmp_path):
    from PyQt6 import QtCore, QtWidgets
    from autolabelt.models.downloader import model_path
    from autolabelt.video_dock import VideoDatasetDock

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("AUTOLABEL_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("AUTOLABEL_SESSION_MODEL_DIR", str(tmp_path))
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = QtWidgets.QMainWindow()
    dock = VideoDatasetDock(window)
    dock.model_combo.setCurrentIndex(dock.model_combo.findData("grounding_dino"))
    location = model_path("grounding_dino")
    location.mkdir(parents=True)
    (location / "model.safetensors").write_bytes(b"test")
    dock.pipeline_worker = QtCore.QThread(dock)
    dock._pipeline_finished()
    assert (location / "model.safetensors").is_file()
    assert dock.pipeline_worker is None
    assert not dock.start_button.isEnabled() and dock.analyze_button.isEnabled()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    application.processEvents()
