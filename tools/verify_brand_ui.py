"""Render small icons and light/dark workbenches using isolated preview settings."""
from pathlib import Path
import argparse
import os
import tempfile

from PyQt6 import QtCore, QtGui, QtWidgets

from autolabelt.workbench import config
from autolabelt.workbench.resources import resources
from autolabelt.workbench.views.labeling.utils.theme import init_theme, get_app_stylesheet
from autolabelt.appearance import stylesheet
from autolabelt.desktop import AutoLabelMainWindow, brand_icon


ROOT = Path(__file__).resolve().parents[1]


def render_previews(output):
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    # Offscreen Qt on Windows needs explicit font registration for Chinese text.
    for filename in ("msyh.ttc", "msyhbd.ttc"):
        font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / filename
        if os.name == "nt" and font.is_file():
            QtGui.QFontDatabase.addApplicationFont(str(font))
    if "Microsoft YaHei" in QtGui.QFontDatabase.families():
        app.setFont(QtGui.QFont("Microsoft YaHei", 9))
    translator = QtCore.QTranslator()
    if not translator.load(":/languages/translations/zh_CN.qm"):
        raise RuntimeError("Chinese translation is missing from the Qt resource bundle")
    app.installTranslator(translator)
    canvas = QtGui.QImage(850, 340, QtGui.QImage.Format.Format_ARGB32)
    canvas.fill(QtGui.QColor("#f2f6fa"))
    painter = QtGui.QPainter(canvas)
    painter.fillRect(0, 170, 850, 170, QtGui.QColor("#14283b"))
    painter.setFont(app.font())
    for row, color in ((0, "#16263c"), (170, "#f4fafb")):
        for x, size in ((40, 16), (170, 28), (320, 56), (520, 128)):
            pixmap = brand_icon().pixmap(size, size)
            if pixmap.isNull():
                raise RuntimeError(f"Cannot load the brand icon at {size}px")
            painter.drawPixmap(x, row + 18, pixmap)
            painter.setPen(QtGui.QColor(color))
            painter.drawText(x, row + 162, f"{size} px")
    painter.end()
    if not canvas.save(str(output / "logo-sizes.png")):
        raise OSError("Cannot save icon previews")
    for mode in ("light", "dark"):
        init_theme(mode)
        app.setStyleSheet(get_app_stylesheet() + "\n" + stylesheet(mode))
        window = AutoLabelMainWindow(app, config=config.get_config())
        try:
            window.resize(1500, 1000)
            window.show()
            app.processEvents()
            if not window.grab().save(str(output / f"workbench-{mode}.png")):
                raise OSError(f"Cannot save {mode} workbench preview")
        finally:
            window.close()
            app.processEvents()
    print("Saved light/dark workbench and 16/28/56/128px icon previews to", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/branding")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="preview-", dir=output) as temporary:
        config._work_directory = temporary
        config.current_config_file = str(Path(temporary) / ".autolabeltrc")
        settings_type = QtCore.QSettings

        class PreviewSettings(settings_type):
            def __init__(self, *args, **kwargs):
                super().__init__(str(Path(temporary) / "preview.ini"), settings_type.Format.IniFormat)

        QtCore.QSettings = PreviewSettings
        try:
            render_previews(output)
        finally:
            QtCore.QSettings = settings_type


if __name__ == "__main__":
    main()
