"""Regression checks for the renamed package and distributable resources."""
from importlib.resources import files
from pathlib import Path
import ast
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_all_application_imports_resolve_in_autolabel_namespace():
    for path in (ROOT / "autolabel").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            imports = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                       else [node.module] if isinstance(node, ast.ImportFrom) and not node.level
                       and node.module else [])
            for name in imports:
                assert name != "anylabeling" and not name.startswith("anylabeling."), path
                if name.startswith("autolabel."):
                    module = ROOT.joinpath(*name.split("."))
                    assert module.is_dir() or module.with_suffix(".py").is_file(), (path, name)


def test_packaged_qt_resources_and_translations_load():
    from PyQt6 import QtCore
    from autolabel.workbench.resources import resources

    assert files("autolabel.workbench").joinpath("configs/autolabel_config.yaml").is_file()
    assert files("autolabel.workbench").joinpath("resources/resources.rcc").is_file()
    assert QtCore.QFile.exists(":/images/images/icon.png")
    assert files("autolabel.workbench").joinpath("configs/models.yaml").is_file()
    translator = QtCore.QTranslator()
    assert translator.load(":/languages/translations/zh_CN.qm")
    assert translator.translate("ExportFormatDialog", "Generate the model configuration and load it in AutoLabel") == "生成模型配置并在 AutoLabel 中加载"


def test_all_launch_modes_use_autolabel_identity(monkeypatch):
    from autolabel.workbench import app_info, config
    from autolabel.workbench.app import build_startup_banner

    assert app_info.__appname__ == "AutoLabel"
    assert app_info.__version__ == "0.1.0"
    assert "AutoLabel" in build_startup_banner(lambda text, color: text)
    monkeypatch.delenv("AUTOLABEL_INTEGRATED_WORKBENCH", raising=False)
    assert config.get_user_config_name() == ".autolabelrc"
    monkeypatch.setenv("AUTOLABEL_INTEGRATED_WORKBENCH", "1")
    assert config.get_user_config_name() == ".autolabelrc"


def test_default_application_update_checks_do_not_contact_upstream(monkeypatch):
    import requests
    from autolabel.workbench.views.labeling.utils.update_checker import (
        check_for_updates_async, check_for_updates_sync,
    )
    from autolabel.workbench.services.updates.protocol import DEFAULT_FEED

    def unexpected_request(*args, **kwargs):
        raise AssertionError("No application feed has been configured")

    monkeypatch.setattr(requests, "get", unexpected_request)
    assert DEFAULT_FEED == ""
    assert check_for_updates_sync() is None
    assert check_for_updates_async() is None


def test_distribution_uses_only_current_package_and_license_paths():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["tool"]["setuptools"]["packages"]["find"]["include"] == ["autolabel*"]
    assert metadata["project"]["license-files"] == ["LICENSE", "THIRD_PARTY_NOTICES.md"]
    assert (ROOT / "LICENSE").is_file()
    assert (ROOT / "THIRD_PARTY_NOTICES.md").is_file()
    assert not (ROOT / "anylabeling").exists()
    assert not (ROOT / "autolabel/ui.py").exists()
    assert not (ROOT / "autolabel/review.py").exists()
