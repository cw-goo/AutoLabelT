"""Launch the integrated AutoLabelT desktop workbench."""

import os
import sys
import tempfile
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets

from autolabel.workbench.views.mainwindow import MainWindow

from .video_dock import DEFAULT_VIDEO_DOCK_WIDTH, VideoDatasetDock
from .branding import APP_NAME, APP_SUMMARY, QT_APPLICATION_ID, app_version
from .appearance import SoftButtonMotion


def brand_icon():
    return QtGui.QIcon(str(Path(__file__).parent / "assets" / "logo.png"))


class AutoLabelMainWindow(MainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setObjectName("autolabelWindow")
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(brand_icon())
        self.app.setApplicationName(QT_APPLICATION_ID)
        self.app.setApplicationDisplayName(APP_NAME)
        self.app.setApplicationVersion(app_version())
        self.app.setWindowIcon(brand_icon())
        self.statusBar().showMessage(APP_SUMMARY)
        self._brand_about_dialog()
        self.video_dock = VideoDatasetDock(self)
        self.training_window = None
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea,
                           self.video_dock)
        self.resizeDocks([self.video_dock], [DEFAULT_VIDEO_DOCK_WIDTH],
                         QtCore.Qt.Orientation.Horizontal)
        toolbar = QtWidgets.QToolBar(APP_NAME, self)
        toolbar.setObjectName("autolabelToolbar")
        toolbar.setMovable(False)
        logo = QtWidgets.QLabel(toolbar)
        logo.setPixmap(brand_icon().pixmap(28, 28))
        logo.setContentsMargins(6, 0, 4, 0)
        toolbar.addWidget(logo)
        name = QtWidgets.QLabel(APP_NAME, toolbar)
        name.setObjectName("brandName")
        toolbar.addWidget(name)
        tagline = QtWidgets.QLabel("标注 / 训练", toolbar)
        tagline.setObjectName("brandTagline")
        toolbar.addWidget(tagline)
        self.training_action = QtGui.QAction("开始训练", self)
        self.training_action.setObjectName("trainingCenterAction")
        self.training_action.triggered.connect(self.open_training_center)
        spacer = QtWidgets.QWidget(toolbar)
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        self.sidebar_action = QtGui.QAction(self)
        self.sidebar_action.setObjectName("rightSidebarAction")
        self.sidebar_action.triggered.connect(self._toggle_video_sidebar)
        self.sidebar_toggle = QtWidgets.QToolButton(toolbar)
        self.sidebar_toggle.setObjectName("sidebarToggleButton")
        self.sidebar_toggle.setFixedSize(32, 30)
        self.sidebar_toggle.setIconSize(QtCore.QSize(22, 22))
        self.sidebar_toggle.setDefaultAction(self.sidebar_action)
        self._update_sidebar_action(True)
        toolbar.addWidget(self.sidebar_toggle)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, toolbar)
        self.button_motion = SoftButtonMotion(self)
        for surface in (self.video_dock, toolbar, self.labeling_widget.view.tools):
            self.button_motion.attach(surface)

    def _brand_about_dialog(self):
        view = self.labeling_widget.view
        about_action = view.menus.help.actions()[-1]
        about_action.triggered.disconnect()
        about_action.triggered.connect(self._show_about)

    def _show_about(self):
        QtWidgets.QMessageBox.about(
            self, f"关于 {APP_NAME}",
            f"{APP_NAME}\n{APP_SUMMARY}\n\n"
            "标注画布及原生图片自动标注基于 X-AnyLabeling 4.2.0，"
            "遵循 GPL-3.0；许可文本见 LICENSE。",
        )

    def _toggle_video_sidebar(self):
        self.video_dock.set_collapsed(not self.video_dock.collapsed)
        self._update_sidebar_action(not self.video_dock.collapsed)

    def _update_sidebar_action(self, visible):
        asset = "sidebar-collapse.svg" if visible else "sidebar-expand.svg"
        self.sidebar_action.setIcon(QtGui.QIcon(str(Path(__file__).parent / "assets" / asset)))
        description = "收起右侧边栏" if visible else "展开右侧边栏"
        self.sidebar_action.setToolTip(description)
        self.sidebar_toggle.setAccessibleName(description)

    def closeEvent(self, event):
        if self.training_window and self.training_window.is_busy():
            self.training_window.stop_training()
        if self.video_dock.has_running_workers():
            event.ignore()
            self.video_dock.stop_workers()
            self.statusBar().showMessage("正在结束后台任务，完成后自动关闭…")
            if not hasattr(self, "_close_timer"):
                self._close_timer = QtCore.QTimer(self)
                self._close_timer.setInterval(100)
                self._close_timer.timeout.connect(self._close_when_idle)
            self._close_timer.start()
            return
        super().closeEvent(event)
        if event.isAccepted():
            self.video_dock.stop_workers()
            if self.training_window:
                self.training_window.hide()

    def open_training_center(self):
        if self.training_window is None:
            from .training_window import TrainingWindow
            self.training_window = TrainingWindow(self, can_start=self._can_start_training)
            self.training_window.busy_changed.connect(self._training_busy_changed)
        self.training_window.set_source(self.video_dock.config.output_dir)
        self.training_window.showNormal()
        self.training_window.raise_()
        self.training_window.activateWindow()

    def _can_start_training(self):
        manager = self.labeling_widget.view.auto_labeling_widget.model_manager
        threads = [getattr(manager, name, None) for name in (
            "model_execution_thread", "model_download_thread")]
        return not self.video_dock.has_running_workers() and not any(
            thread and thread.isRunning() for thread in threads)

    def _training_busy_changed(self, busy):
        self.video_dock._refresh_task_controls()
        self.labeling_widget.view.auto_labeling_widget.setEnabled(not busy)

    def _close_when_idle(self):
        if not self.video_dock.has_running_workers():
            self._close_timer.stop()
            self.close()


def main():
    os.environ["AUTOLABEL_INTEGRATED_WORKBENCH"] = "1"
    if "--no-auto-update-check" not in sys.argv:
        sys.argv.insert(1, "--no-auto-update-check")
    from autolabel.workbench.app import main as start_workbench

    previous_store = os.environ.get("AUTOLABEL_MODEL_DIR")
    previous_session = os.environ.get("AUTOLABEL_SESSION_MODEL_DIR")
    with tempfile.TemporaryDirectory(prefix="autolabel-models-") as temporary:
        os.environ["AUTOLABEL_MODEL_DIR"] = temporary
        os.environ["AUTOLABEL_SESSION_MODEL_DIR"] = temporary
        try:
            return start_workbench()
        finally:
            if previous_store is None:
                os.environ.pop("AUTOLABEL_MODEL_DIR", None)
            else:
                os.environ["AUTOLABEL_MODEL_DIR"] = previous_store
            if previous_session is None:
                os.environ.pop("AUTOLABEL_SESSION_MODEL_DIR", None)
            else:
                os.environ["AUTOLABEL_SESSION_MODEL_DIR"] = previous_session
