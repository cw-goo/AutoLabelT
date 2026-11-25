"""Presentation and small motion effects for the training center."""

from pathlib import Path

from PyQt6 import QtCore, QtWidgets


class AnimatedComboBox(QtWidgets.QComboBox):
    """Keep Qt's keyboard/selection behavior and animate the native popup."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaxVisibleItems(7)
        self.setItemDelegate(QtWidgets.QStyledItemDelegate(self))
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._motion = None
        self._popup_end = None
        self._opacity = QtWidgets.QGraphicsOpacityEffect(self.view())
        self.view().setGraphicsEffect(self._opacity)
        self._opacity.setOpacity(1)

    def showPopup(self):
        if self._motion:
            self._motion.stop()
            self._motion.deleteLater()
        self._opacity.setOpacity(1)
        super().showPopup()
        popup = self.view().window()
        if not popup.isVisible():
            return
        self._popup_end = popup.pos()
        self._motion = QtCore.QParallelAnimationGroup(self)
        slide = QtCore.QPropertyAnimation(popup, b"pos", self._motion)
        slide.setDuration(130)
        slide.setStartValue(self._popup_end + QtCore.QPoint(0, -2))
        slide.setEndValue(self._popup_end)
        slide.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        fade = QtCore.QPropertyAnimation(self._opacity, b"opacity", self._motion)
        fade.setDuration(110)
        fade.setStartValue(0.65)
        fade.setEndValue(1)
        fade.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self._motion.addAnimation(slide)
        self._motion.addAnimation(fade)
        self._motion.start()

    def hidePopup(self):
        if self._motion:
            self._motion.stop()
            self._motion.deleteLater()
            self._motion = None
        self._opacity.setOpacity(1)
        if self._popup_end is not None:
            self.view().window().move(self._popup_end)
        super().hidePopup()


class OnnxConversionDialog(QtWidgets.QDialog):
    stop_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("onnxConversion")
        self.setWindowTitle("格式转换")
        self.setMinimumWidth(390)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 18)
        layout.setSpacing(16)
        sources = QtWidgets.QHBoxLayout()
        self.local = QtWidgets.QPushButton("导入本地 PT")
        self.local.setObjectName("chooseLocalCheckpoint")
        self.current = QtWidgets.QPushButton("转换当前 best.pt")
        sources.addWidget(self.local, 1)
        sources.addWidget(self.current, 1)
        layout.addLayout(sources)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setObjectName("onnxExportProgress")
        self.progress.setAccessibleName("ONNX 转换进度")
        self.progress.setTextVisible(False)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.close_button = QtWidgets.QPushButton("关闭")
        self.close_button.setMinimumWidth(106)
        self.close_button.clicked.connect(self._close_or_stop)
        layout.addWidget(self.close_button, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

    def set_controls(self, *, busy, has_checkpoint):
        self.local.setEnabled(not busy)
        self.current.setEnabled(not busy and has_checkpoint)

    def _close_or_stop(self):
        if self.progress.maximum() == 0:
            self.stop_requested.emit()
        else:
            self.reject()

    def start_conversion(self):
        self.setWindowTitle("正在转换为 ONNX")
        self.progress.setRange(0, 0)
        self.close_button.setText("停止转换")

    def finish_conversion(self, title, *, completed=False):
        self.setWindowTitle(title)
        self.progress.setRange(0, 100)
        self.progress.setValue(100 if completed else 0)
        self.close_button.setText("关闭")


def training_stylesheet():
    arrow = QtCore.QDir.fromNativeSeparators(str(
        Path(__file__).parent / "assets" / "training-chevron.svg"))
    return """
        QMainWindow#trainingWindow, QWidget#trainingBody, QDialog#advancedTrainingSettings,
        QDialog#onnxConversion {
            background: #f2f6fa; }
        QWidget { color: #213b50; font-size: 12px; }
        QLabel#trainingTitle { font-size: 23px; font-weight: 700; color: #193248; }
        QLabel#trainingBrand { color: #087f88; font-size: 11px; font-weight: 700; }
        QLabel#sectionTitle { font-size: 14px; font-weight: 600; }
        QLabel#fieldLabel { color: #6a8091; font-size: 11px; }
        QLabel#trainingState { color: #087f88; font-weight: 600; }
        QLabel#trainingElapsed { color: #718695; font-size: 11px; }
        QFrame[card="true"] { background: #ffffff; border: 1px solid #e0e9ef;
            border-radius: 13px; }
        QGroupBox#trainingParameterGroup { background: #ffffff; border: 1px solid #e0e9ef;
            border-radius: 9px; margin-top: 11px; padding-top: 5px; font-weight: 600; }
        QGroupBox#trainingParameterGroup::title { subcontrol-origin: margin; left: 12px;
            padding: 1px 5px; color: #087f88; }
        QFrame#trainingSidebar { background: #142c3f; border: 0; border-radius: 14px; }
        QFrame#trainingSidebar QLabel { color: #d4e5ec; }
        QListWidget { background: transparent; border: 0; outline: none; padding: 0; }
        QListWidget::item { color: #b9ced9; padding: 14px 10px; margin-bottom: 6px;
            border-radius: 8px; }
        QListWidget::item:hover { background: #203f52; }
        QListWidget::item:selected { background: #28566a; color: #ffffff; }
        QLineEdit, QComboBox, QAbstractSpinBox { background: #f8fafc; color: #213b50;
            border: 1px solid #dce6ed; border-radius: 8px; min-height: 22px;
            padding: 6px 10px; selection-background-color: #087f88; }
        QLineEdit:hover, QComboBox:hover, QAbstractSpinBox:hover { border-color: #9ebfc9; }
        QLineEdit:focus, QComboBox:focus, QAbstractSpinBox:focus { border-color: #15989e; background: #ffffff; }
        QLineEdit:disabled, QComboBox:disabled, QAbstractSpinBox:disabled { color: #899ba8; background: #f1f5f8; }
        QComboBox { padding-right: 26px; }
        QComboBox QLineEdit { border: 0; padding: 0; min-height: 20px; background: transparent; }
        QComboBox::drop-down { width: 30px; border: 0; }
        QComboBox QAbstractItemView { background: #f8fafc; color: #213b50;
            border: 1px solid #dce6ed; selection-background-color: #087f88;
            selection-color: #ffffff; outline: none; }
        QComboBox QAbstractItemView::item { min-height: 28px; padding: 3px 10px; }
        QComboBox::down-arrow { image: url("%s"); width: 12px; height: 12px; }
        QSpinBox::up-button, QSpinBox::down-button { width: 18px; border: 0; background: transparent; }
        QPushButton, QToolButton { background: #ffffff; color: #345267;
            border: 1px solid #dce6ed; border-radius: 8px; padding: 7px 13px; min-height: 20px; }
        QPushButton:hover, QToolButton:hover { background: #eef8f7; border-color: #9cc8c9; }
        QPushButton:pressed, QToolButton:pressed { background: #dcefed; }
        QPushButton:disabled, QToolButton:disabled { color: #9baab4; background: #f3f6f8; border-color: #e6edf1; }
        QPushButton#startTraining, QPushButton#chooseLocalCheckpoint {
            color: #ffffff; background: #087f88; border-color: #087f88;
            font-weight: 600; padding: 9px 24px; }
        QPushButton#startTraining:hover, QPushButton#chooseLocalCheckpoint:hover { background: #0b98a0; }
        QPushButton#startTraining:disabled, QPushButton#chooseLocalCheckpoint:disabled {
            color: #a2b9bc; background: #e0eeee; border-color: #e0eeee; }
        QPushButton#stopTraining { color: #b05a56; }
        QProgressBar { background: #eaf0f5; border: 0; border-radius: 5px;
            color: #456174; min-height: 19px; text-align: center; font-size: 11px; }
        QProgressBar::chunk { background: #a9dbd5; border-radius: 5px; }
        QProgressBar#epochProgress::chunk { background: #77c5c0; }
        QProgressBar#onnxExportProgress { min-height: 10px; max-height: 10px; }
        QProgressBar#onnxExportProgress::chunk { background: #087f88; }
        QTabWidget::pane { border: 1px solid #e0e9ef; background: #ffffff; border-radius: 12px; }
        QWidget#trainingCharts { background: #f3f7fa; }
        QFrame#trainingChart { background: #ffffff; border: 1px solid #e0e9ef; border-radius: 11px; }
        QLabel#chartHeading { color: #20394e; font-size: 15px; font-weight: 700; }
        QLabel#chartSummary { color: #235e74; background: #eaf5f8; border-radius: 7px;
            padding: 5px 8px; font-size: 11px; font-weight: 600; }
        QToolButton#chartSeries { border: 1px solid #e1eaf1; border-radius: 6px; padding: 3px 7px;
            min-height: 15px; color: #8192a3; background: #fafbfd; font-size: 11px; }
        QToolButton#chartSeries:checked { color: #2e485f; background: #f1f7fb; }
        QToolButton#chartSeries:hover { border-color: #9cb8cd; }
        QTabBar::tab { color: #758b9a; background: transparent; padding: 10px 16px;
            border: 0; border-bottom: 2px solid transparent; margin-right: 8px; }
        QTabBar::tab:selected { color: #087f88; border-bottom-color: #087f88; }
        QTabBar::tab:hover { color: #087f88; }
        QPlainTextEdit { color: #355367; background: #ffffff; border: 0;
            border-radius: 10px; padding: 12px; selection-background-color: #b6dedb; }
        QScrollBar:vertical { background: transparent; width: 6px; margin: 2px; }
        QScrollBar::handle:vertical { background: #c4d3dd; border-radius: 3px; min-height: 25px; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
    """ % arrow
