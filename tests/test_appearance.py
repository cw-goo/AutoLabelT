"""The branded theme stays scoped and button motion remains lightweight."""

from PyQt6 import QtCore, QtWidgets

from autolabelt.appearance import SoftButtonMotion, stylesheet


def test_light_and_dark_palettes_cover_controls():
    light = stylesheet("light")
    dark = stylesheet("dark")
    assert "#f2f6fa" in light
    assert "#121f2c" in dark
    assert 'QPushButton[role="primary"]' in light
    assert "QToolBar#autolabelToolbar" in light
    assert "QDockWidget#videoDatasetDock" in dark
    assert "QProgressBar::chunk" in light


def test_button_hover_uses_soft_animation(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    container = QtWidgets.QWidget()
    button = QtWidgets.QPushButton("开始构建", container)
    motion = SoftButtonMotion(container)
    motion.attach(container)
    effect, animation = motion.animations[button]
    application.sendEvent(button, QtCore.QEvent(QtCore.QEvent.Type.Enter))
    animation.setCurrentTime(170)
    assert effect.opacity() == 0.93
    application.sendEvent(button, QtCore.QEvent(QtCore.QEvent.Type.Leave))
    animation.setCurrentTime(170)
    assert effect.opacity() == 1.0
    container.close()
