"""Launch AutoLabel inside the restored AutoLabel workbench."""

import os
import sys
import tempfile
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets

from autolabel.workbench.views.mainwindow import MainWindow

from .video_dock import VideoDatasetDock
from .branding import app_version
from .appearance import SoftButtonMotion


def brand_icon():
    return QtGui.QIcon(str(Path(__file__).parent / "assets" / "logo.png"))


class AutoLabelMainWindow(MainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setObjectName("autolabelWindow")
        self.setWindowTitle("AutoLabel")
        self.setWindowIcon(brand_icon())
        self.app.setApplicationName("AutoLabel")
        self.app.setApplicationVersion(app_version())
        self.app.setWindowIcon(brand_icon())
        self.statusBar().showMessage("AutoLabel · 智能视频与图像标注")
        self._brand_about_dialog()
        self.video_dock = VideoDatasetDock(self)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea,
                           self.video_dock)
        toolbar = QtWidgets.QToolBar("AutoLabel", self)
        toolbar.setObjectName("autolabelToolbar")
        logo = QtWidgets.QLabel(toolbar)
        logo.setPixmap(brand_icon().pixmap(28, 28))
        logo.setContentsMargins(6, 0, 4, 0)
        toolbar.addWidget(logo)
        name = QtWidgets.QLabel("AutoLabel", toolbar)
        name.setObjectName("brandName")
        toolbar.addWidget(name)
        tagline = QtWidgets.QLabel("VIDEO  /  DATASET", toolbar)
        tagline.setObjectName("brandTagline")
        toolbar.addWidget(tagline)
        spacer = QtWidgets.QWidget(toolbar)
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                             QtWidgets.QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        self.sidebar_action = toolbar.addAction("")
        self.sidebar_action.setObjectName("rightSidebarAction")
        self.sidebar_action.setToolTip("收起右侧边栏")
        sidebar_button = toolbar.widgetForAction(self.sidebar_action)
        sidebar_button.setObjectName("sidebarToggleButton")
        sidebar_button.setFixedSize(36, 34)
        sidebar_button.setIconSize(QtCore.QSize(24, 24))
        self.sidebar_action.triggered.connect(self._toggle_video_sidebar)
        self.video_dock.visibilityChanged.connect(self._update_sidebar_action)
        self._update_sidebar_action(self.video_dock.isVisible())
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
            self, "关于 AutoLabel",
            "AutoLabel · 智能视频与图像标注\n"
            "视频工作流与品牌标识由 AutoLabel 提供。\n"
            "标注画布及原生图片自动标注基于 X-AnyLabeling 4.2.0，"
            "遵循 GPL-3.0；许可文本见 LICENSE。",
        )

    def _toggle_video_sidebar(self):
        self.video_dock.setVisible(not self.video_dock.isVisible())

    def _update_sidebar_action(self, visible):
        asset = "sidebar-collapse.svg" if visible else "sidebar-expand.svg"
        self.sidebar_action.setIcon(QtGui.QIcon(str(Path(__file__).parent / "assets" / asset)))
        self.sidebar_action.setToolTip("收起右侧边栏" if visible else "展开右侧边栏")

    def closeEvent(self, event):
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
