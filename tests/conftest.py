"""Keep every test's application configuration inside its temporary directory."""
import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def shared_qt_application():
    # Keep Qt-owned renderers alive across tests that create and close windows.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6 import QtWidgets

    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield application
    application.processEvents()


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch, tmp_path):
    from autolabel.workbench import config
    from PyQt6 import QtCore

    monkeypatch.setattr(config, "_work_directory", str(tmp_path))
    monkeypatch.setattr(config, "current_config_file", str(tmp_path / ".autolabelrc"))
    settings_type = QtCore.QSettings

    class IsolatedSettings(settings_type):
        def __init__(self, *args, **kwargs):
            if len(args) >= 2 and args[1] == settings_type.Format.IniFormat:
                super().__init__(*args, **kwargs)
            else:
                super().__init__(str(tmp_path / "qt.ini"), settings_type.Format.IniFormat)

    monkeypatch.setattr(QtCore, "QSettings", IsolatedSettings)
