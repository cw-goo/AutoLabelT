"""Colors and subtle button motion for the integrated workbench."""

from PyQt6 import QtCore, QtWidgets


PALETTES = {
    "light": {
        "background": "#f2f6fa", "surface": "#ffffff", "surface_hover": "#eaf6f7",
        "border": "#dbe5ed", "text": "#193248", "muted": "#647a8d",
        "field": "#fbfdff", "accent": "#087f88", "accent_hover": "#0b98a0",
        "accent_pressed": "#076973", "accent_soft": "#def4f2", "orange": "#eea75c",
        "danger": "#a85451", "danger_soft": "#fff1ef", "track": "#e3edf2",
    },
    "dark": {
        "background": "#121f2c", "surface": "#1a2a39", "surface_hover": "#24404b",
        "border": "#334b5a", "text": "#e8f4f6", "muted": "#a5bbc6",
        "field": "#203342", "accent": "#30beb6", "accent_hover": "#51d5ca",
        "accent_pressed": "#20988f", "accent_soft": "#194641", "orange": "#ffb86b",
        "danger": "#f2998c", "danger_soft": "#493436", "track": "#2a4351",
    },
}


def stylesheet(mode):
    palette = PALETTES["dark" if mode == "dark" else "light"]
    return """
        QMainWindow#autolabelWindow, QDialog { background: %(background)s; color: %(text)s; }
        QMenuBar { background: %(surface)s; color: %(text)s; border-bottom: 1px solid %(border)s; padding: 3px; }
        QMenuBar::item { padding: 6px 10px; border-radius: 7px; }
        QMenuBar::item:selected { background: %(accent_soft)s; color: %(accent)s; }
        QMenu { background: %(surface)s; color: %(text)s; border: 1px solid %(border)s; border-radius: 9px; padding: 5px; }
        QMenu::item { padding: 7px 22px; border-radius: 5px; }
        QMenu::item:selected { background: %(accent_soft)s; color: %(accent)s; }
        QToolBar#autolabelToolbar { background: #14283b; border: 0; border-bottom: 2px solid %(accent)s; spacing: 7px; padding: 7px 12px; }
        QToolBar#autolabelToolbar QLabel#brandName { color: #f6fcfd; font-size: 17px; font-weight: 700; }
        QToolBar#autolabelToolbar QLabel#brandTagline { color: %(orange)s; font-size: 12px;
            font-weight: 600; border-left: 1px solid #456477; padding-left: 9px; }
        QToolBar#autolabelToolbar QToolButton { background: #24435a; color: #f5fcfd; border: 1px solid #355970; border-radius: 8px; padding: 6px 12px; }
        QToolBar#autolabelToolbar QToolButton:hover { background: #2d596d; border-color: #58c5c1; }
        QToolBar#autolabelToolbar QToolButton#sidebarToggleButton { background: transparent; border: 1px solid #456477; border-radius: 8px; padding: 4px; }
        QToolBar#autolabelToolbar QToolButton#sidebarToggleButton:hover { background: #244b59; border-color: #65cfc8; }
        QToolBar#autolabelToolbar QToolButton#sidebarToggleButton:pressed { background: #1b3d4a; border-color: #30beb6; }
        QToolBar#ToolsToolBar { background: %(surface)s; border-right: 1px solid %(border)s; }
        QToolBar#ToolsToolBar QToolButton { background: transparent; border: 0; border-radius: 8px; margin: 2px; padding: 3px; }
        QToolBar#ToolsToolBar QToolButton:hover { background: %(accent_soft)s; }
        QToolBar#ToolsToolBar QToolButton:checked { background: %(accent_soft)s; border-left: 2px solid %(accent)s; }
        QDockWidget#videoDatasetDock { color: %(text)s; }
        QWidget#videoSidebarShell { background: %(background)s; }
        QLabel#sidebarHeading { color: %(text)s; font-size: 15px; font-weight: 700; }
        QDockWidget#videoDatasetDock::title { background: %(surface)s; color: %(text)s; padding: 8px; border-bottom: 1px solid %(border)s; }
        QScrollArea#videoDockScroll, QWidget#videoDockContent { background: %(background)s; border: 0; }
        QWidget#videoDockContent { font-size: 14px; }
        QFrame[workflowCard="true"] { background: %(surface)s; border: 1px solid %(border)s;
            border-radius: 12px; }
        QFrame[workflowCard="true"] QLabel { color: %(text)s; }
        QLabel#workflowBadge { background: %(accent_soft)s; color: %(accent)s;
            border-radius: 5px; font-size: 11px; font-weight: 700; }
        QLabel#workflowCardTitle { color: %(text)s; font-size: 16px; font-weight: 700; }
        QLabel#workflowCardCaption { color: %(muted)s; font-size: 12px; }
        QLabel#workflowSummary { color: %(muted)s; }
        QWidget#videoDockContent QPushButton { min-height: 18px; padding: 3px 7px;
            border-radius: 7px; }
        QWidget#videoDockContent QSpinBox, QWidget#videoDockContent QDoubleSpinBox {
            min-height: 18px; padding: 3px 6px; border-radius: 7px; }
        QWidget#videoDockContent QPlainTextEdit { padding: 3px 6px; border-radius: 7px; }
        QProgressBar#videoBatchProgress { min-height: 10px; max-height: 10px; border-radius: 5px; }
        QProgressBar#videoBatchProgress::chunk { border-radius: 5px; }
        QFrame#samplingAnalysisResults { background: %(field)s; border: 1px solid %(border)s;
            border-radius: 7px; }
        QTreeWidget#videoQueue, QTreeWidget#samplingPlanTable {
            background: %(field)s; alternate-background-color: %(background)s;
            color: %(text)s; border: 1px solid %(border)s; border-radius: 7px;
            outline: 0; }
        QTreeWidget#videoQueue::item, QTreeWidget#samplingPlanTable::item {
            min-height: 21px; padding: 0 3px; border: 0; }
        QTreeWidget#videoQueue::item:selected { background: %(accent_soft)s; color: %(accent)s; }
        QTreeWidget#videoQueue QHeaderView::section, QTreeWidget#samplingPlanTable QHeaderView::section {
            background: %(background)s;
            color: %(muted)s; border: 0; border-bottom: 1px solid %(border)s;
            padding: 2px 4px; font-size: 12px; }
        QPushButton#startTrainingButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
            stop:0 #102c44,stop:1 #164b5c); color: #ffffff; border: 1px solid #23596d;
            border-radius: 10px; font-size: 13px; font-weight: 700; padding: 4px 14px; }
        QPushButton#startTrainingButton:hover { background: #1b5268; border-color: #348299; }
        QPushButton#startTrainingButton:pressed { background: #0b2236; }
        QPushButton#startTrainingButton:disabled { background: %(track)s;
            color: %(muted)s; border-color: %(border)s; }
        QFrame#workflowHeader { background: #14283b; border: 1px solid #2b5266; border-radius: 13px; }
        QLabel#workflowTitle { color: #f4fcfd; font-size: 17px; font-weight: 700; }
        QLabel#workflowSubtitle { color: #afd6dc; font-size: 11px; }
        QLabel#workflowSteps { color: #ffcc8e; font-size: 10px; }
        QLabel[tone="muted"] { color: %(muted)s; }
        QDockWidget#videoDatasetDock QGroupBox { background: %(surface)s; color: %(text)s; border: 1px solid %(border)s; border-radius: 12px; margin-top: 13px; padding: 13px 10px 10px; font-weight: 600; }
        QDockWidget#videoDatasetDock QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 12px; padding: 0 7px; color: %(accent)s; }
        QDialog#modelLoadDialog, QDialog#modelSquareDialog, QDialog#parameterDialog { background: %(background)s; }
        QDialog#modelLoadDialog QLabel, QDialog#modelLoadDialog QCheckBox,
        QDialog#modelSquareDialog QLabel, QDialog#modelSquareDialog QCheckBox { color: %(text)s; }
        QDialog#modelLoadDialog QLabel[tone="muted"],
        QDialog#modelSquareDialog QLabel[tone="muted"] { color: %(muted)s; }
        QLabel#dialogHeading { color: %(text)s; font-size: 21px; font-weight: 700; }
        QLabel#dialogHint { color: %(muted)s; font-size: 11px; }
        QFrame[modelCard="true"] { background: %(surface)s; border: 1px solid %(border)s; border-radius: 12px; }
        QLabel#modelColumnTitle { color: %(accent)s; font-size: 14px; font-weight: 700; }
        QDialog#modelSquareDialog QTreeWidget#modelSquareTree {
            background: %(surface)s; alternate-background-color: %(background)s;
            color: %(text)s; border: 1px solid %(border)s; border-radius: 12px;
            outline: 0; padding: 6px; }
        QTreeWidget#modelSquareTree::item { min-height: 30px; padding: 3px 5px;
            border: 0; border-radius: 0; }
        QTreeWidget#modelSquareTree::item:hover { background: transparent; }
        QTreeWidget#modelSquareTree::item:selected { background: %(background)s; color: %(text)s; }
        QTreeWidget#modelSquareTree::branch { background: transparent; border: 0; }
        QTreeWidget#modelSquareTree QHeaderView::section { background: %(background)s;
            color: %(muted)s; border: 0; padding: 8px 5px; }
        QFrame#modelInfoPanel { background: %(surface)s; border: 1px solid %(border)s; border-radius: 12px; }
        QFrame#modelInfoPanel QLabel { color: %(text)s; background: transparent; border: 0; }
        QFrame#modelInfoPanel QLabel#modelInfoTitle { font-size: 21px; font-weight: 700; }
        QFrame#modelInfoPanel QLabel#modelTaskBadge { background: %(accent_soft)s;
            color: %(accent)s; border-radius: 6px; padding: 5px 8px; font-size: 12px; }
        QFrame#modelInfoPanel QLabel#modelStateBadge { background: %(background)s;
            color: %(muted)s; border-radius: 6px; padding: 5px 8px; font-size: 12px; }
        QFrame#modelInfoDivider { background: %(border)s; border: 0; }
        QFrame#modelInfoPanel QLabel#modelUsageHint { background: %(accent_soft)s;
            color: %(accent)s; border-radius: 8px; padding: 11px; }
        QPushButton[selected="true"] { background: %(accent_soft)s; color: %(accent)s; border-color: %(accent)s; font-weight: 700; }
        QListWidget#modelSourceList { background: %(field)s; color: %(text)s; border: 1px solid %(border)s; border-radius: 8px; outline: 0; padding: 4px; }
        QListWidget#modelSourceList::item { padding: 11px 8px; border-radius: 6px; }
        QListWidget#modelSourceList::item:selected { background: %(accent_soft)s; color: %(accent)s; }
        QLabel#modelDetails { background: %(surface)s; color: %(text)s; border: 1px solid %(border)s; border-radius: 8px; padding: 10px; }
        QScrollArea#parameterScroll, QWidget#parameterBody { background: %(background)s; border: 0; }
        QLabel#parameterSectionTitle { color: %(accent)s; font-size: 12px; font-weight: 700; padding: 9px 0 2px; }
        QLabel#parameterFieldLabel { color: %(muted)s; font-size: 11px; }
        QFrame#parameterStepper { background: %(surface)s; border: 1px solid %(border)s; border-radius: 9px; }
        QDialog#parameterDialog QSpinBox, QDialog#parameterDialog QDoubleSpinBox { background: transparent; border: 0; min-height: 30px; padding: 0 8px; }
        QDialog#parameterDialog QSpinBox:focus, QDialog#parameterDialog QDoubleSpinBox:focus { border: 0; }
        QDialog#parameterDialog QComboBox { background: %(surface)s; min-height: 33px; }
        QPushButton#parameterStepButton { background: %(accent_soft)s; color: %(accent)s; border: 0; border-radius: 7px; font-size: 16px; font-weight: 700; padding: 0; }
        QPushButton#parameterStepButton:hover { background: %(accent)s; color: #ffffff; }
        QPushButton#parameterStepButton:pressed { background: %(accent_pressed)s; color: #ffffff; }
        QPushButton { background: %(surface)s; color: %(text)s; border: 1px solid %(border)s; border-radius: 8px; padding: 6px 11px; min-height: 25px; }
        QPushButton:hover { background: %(surface_hover)s; border-color: %(accent)s; }
        QPushButton:pressed { background: %(accent_soft)s; }
        QPushButton:disabled { background: %(background)s; color: %(muted)s; border-color: %(border)s; }
        QPushButton[role="primary"] { background: %(accent)s; color: #ffffff; border: 1px solid %(accent)s; font-weight: 700; }
        QPushButton[role="primary"]:hover { background: %(accent_hover)s; border-color: %(accent_hover)s; }
        QPushButton[role="primary"]:pressed { background: %(accent_pressed)s; }
        QPushButton[role="primary"]:disabled { background: %(track)s; color: %(muted)s; border-color: %(track)s; }
        QPushButton[role="danger"] { color: %(danger)s; background: %(danger_soft)s; border-color: %(danger_soft)s; }
        QPushButton[role="danger"]:hover { border-color: %(danger)s; }
        QPushButton[role="danger"]:disabled { color: %(muted)s; background: %(background)s; border-color: %(border)s; }
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit { background: %(field)s; color: %(text)s; border: 1px solid %(border)s; border-radius: 8px; padding: 6px 8px; selection-background-color: %(accent)s; }
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus { border: 2px solid %(accent)s; }
        QComboBox::drop-down { border: 0; width: 22px; }
        QProgressBar { background: %(track)s; color: %(text)s; border: 0; border-radius: 6px; min-height: 12px; text-align: center; }
        QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 %(accent)s,stop:1 #59c9c0); border-radius: 6px; }
        QListWidget#FileList { background: %(surface)s; color: %(text)s; border: 1px solid %(border)s; border-radius: 9px; outline: 0; }
        QListWidget#FileList::item { padding: 4px 6px; border-radius: 5px; }
        QListWidget#FileList::item:selected { background: %(accent_soft)s; color: %(accent)s; }
        QStatusBar { background: %(surface)s; color: %(muted)s; border-top: 1px solid %(border)s; }
    """ % palette


class SoftButtonMotion(QtCore.QObject):
    def __init__(self, parent):
        super().__init__(parent)
        self.animations = {}

    def attach(self, root):
        if isinstance(root, QtWidgets.QToolBar):
            buttons = [root.widgetForAction(action) for action in root.actions()]
        else:
            buttons = root.findChildren(QtWidgets.QPushButton) + root.findChildren(QtWidgets.QToolButton)
        for button in buttons:
            if not isinstance(button, QtWidgets.QAbstractButton):
                continue
            if button in self.animations or button.graphicsEffect() is not None:
                continue
            button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            effect = QtWidgets.QGraphicsOpacityEffect(button)
            effect.setOpacity(1.0)
            button.setGraphicsEffect(effect)
            animation = QtCore.QPropertyAnimation(effect, b"opacity", button)
            animation.setDuration(170)
            animation.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
            button.installEventFilter(self)
            self.animations[button] = (effect, animation)
            button.destroyed.connect(lambda _=None, item=button: self.animations.pop(item, None))

    def eventFilter(self, watched, event):
        if watched in self.animations and watched.isEnabled():
            targets = {
                QtCore.QEvent.Type.Enter: 0.93,
                QtCore.QEvent.Type.Leave: 1.0,
                QtCore.QEvent.Type.MouseButtonPress: 0.80,
                QtCore.QEvent.Type.MouseButtonRelease: 0.93 if watched.underMouse() else 1.0,
            }
            target = targets.get(event.type())
            if target is not None:
                effect, animation = self.animations[watched]
                animation.stop()
                animation.setStartValue(effect.opacity())
                animation.setEndValue(target)
                animation.start()
        return super().eventFilter(watched, event)
