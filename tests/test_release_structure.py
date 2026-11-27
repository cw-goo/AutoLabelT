"""Regression checks for the renamed package and distributable resources."""
from importlib.resources import files
from pathlib import Path
import ast
import tomllib
import xml.etree.ElementTree as ET

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_all_application_imports_resolve_in_autolabelt_namespace():
    for path in (ROOT / "autolabelt").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            imports = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                       else [node.module] if isinstance(node, ast.ImportFrom) and not node.level
                       and node.module else [])
            for name in imports:
                assert name.partition(".")[0] not in {"anylabeling", "autolabel"}, path
                if name.startswith("autolabelt."):
                    module = ROOT.joinpath(*name.split("."))
                    assert module.is_dir() or module.with_suffix(".py").is_file(), (path, name)


def test_packaged_qt_resources_and_translations_load():
    from PyQt6 import QtCore
    from autolabelt.workbench.resources import resources

    assert files("autolabelt.workbench").joinpath("configs/autolabelt_config.yaml").is_file()
    assert files("autolabelt.workbench").joinpath("resources/resources.rcc").is_file()
    assert QtCore.QFile.exists(":/images/images/icon.png")
    # Both workbench icon aliases must ship the current shared brand asset.
    for extension in ("png", "svg"):
        logo = files("autolabelt").joinpath(f"assets/logo.{extension}").read_bytes()
        for name in ("icon", "logo"):
            resource = QtCore.QFile(f":/images/images/{name}.{extension}")
            assert resource.open(QtCore.QIODevice.OpenModeFlag.ReadOnly)
            assert bytes(resource.readAll()) == logo
            resource.close()
    assert files("autolabelt.workbench").joinpath("configs/models.yaml").is_file()
    translator = QtCore.QTranslator()
    assert translator.load(":/languages/translations/zh_CN.qm")
    assert translator.translate(
        "ExportFormatDialog", "Generate the model configuration and load it in the workbench",
    ) == "生成模型配置并在工作台中加载"


def test_brand_logo_is_editable_vector_and_renders_at_icon_sizes():
    from PyQt6 import QtCore, QtGui, QtSvg

    source = files("autolabelt").joinpath("assets/logo.svg").read_text(encoding="utf-8")
    tree = ET.fromstring(source)
    assert tree.findall(".//{http://www.w3.org/2000/svg}path")
    assert not tree.findall(".//{http://www.w3.org/2000/svg}image")
    assert "data:image" not in source
    renderer = QtSvg.QSvgRenderer(source.encode("utf-8"))
    assert renderer.isValid()
    for size in (28, 256):
        image = QtGui.QImage(size, size, QtGui.QImage.Format.Format_ARGB32)
        image.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(image)
        renderer.render(painter)
        painter.end()
        assert image.pixelColor(0, 0).alpha() == 0
        assert any(image.pixelColor(x, y).alpha() for x in range(size) for y in range(size))


def test_all_launch_modes_display_autolabelt_and_keep_configuration(monkeypatch):
    from autolabelt.workbench import app_info, config
    from autolabelt.workbench.app import build_startup_banner

    assert app_info.__appname__ == "AutoLabelT"
    assert app_info.__version__ == "0.1.0"
    assert "AutoLabelT" in build_startup_banner(lambda text, color: text)
    monkeypatch.delenv("AUTOLABEL_INTEGRATED_WORKBENCH", raising=False)
    assert config.get_user_config_name() == ".autolabeltrc"
    monkeypatch.setenv("AUTOLABEL_INTEGRATED_WORKBENCH", "1")
    assert config.get_user_config_name() == ".autolabeltrc"


def test_default_application_update_checks_do_not_contact_upstream(monkeypatch):
    import requests
    from autolabelt.workbench.views.labeling.utils.update_checker import (
        check_for_updates_async, check_for_updates_sync,
    )
    from autolabelt.workbench.services.updates.protocol import DEFAULT_FEED

    def unexpected_request(*args, **kwargs):
        raise AssertionError("No application feed has been configured")

    monkeypatch.setattr(requests, "get", unexpected_request)
    assert DEFAULT_FEED == ""
    assert check_for_updates_sync() is None
    assert check_for_updates_async() is None


def test_distribution_uses_only_current_package_and_license_paths():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["name"] == "autolabelt"
    assert metadata["project"]["scripts"] == {"autolabelt": "autolabelt.desktop:main"}
    assert metadata["tool"]["setuptools"]["packages"]["find"]["include"] == ["autolabelt*"]
    assert metadata["project"]["license-files"] == ["LICENSE", "THIRD_PARTY_NOTICES.md"]
    assert (ROOT / "LICENSE").is_file()
    assert (ROOT / "THIRD_PARTY_NOTICES.md").is_file()
    assert not (ROOT / "anylabeling").exists()
    assert not (ROOT / "autolabel").exists()
    assert not (ROOT / "autolabelt/ui.py").exists()
    assert not (ROOT / "autolabelt/review.py").exists()


def test_packaged_onnx_runtime_loads_runs_and_reads_model_metadata(tmp_path):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper
    from autolabelt.workbench.services.auto_labeling.engines.build_onnx_engine import OnnxBaseModel

    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "inference_check",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])],
    )
    model = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("", 13)])
    helper.set_model_props(model, {"purpose": "image_model_runtime_check"})
    weights = tmp_path / "inference_check.onnx"
    onnx.save(model, weights)

    engine = OnnxBaseModel(str(weights), device_type="cpu")
    assert engine.providers == ["CPUExecutionProvider"]
    image = np.array([[2, 3]], dtype=np.float32)
    np.testing.assert_array_equal(engine.get_ort_inference(image), image)
    assert engine.get_metadata_info("purpose") == "image_model_runtime_check"


@pytest.mark.parametrize(
    "device, cuda_available, use_cuda",
    [("auto", True, True), ("auto", False, False), ("cpu", True, False)],
)
def test_onnx_session_prefers_cuda_and_respects_cpu(monkeypatch, device, cuda_available, use_cuda):
    from autolabelt.workbench.services.auto_labeling.engines import build_onnx_engine as engine

    cuda = engine.CUDA_PROVIDER
    cpu = engine.CPU_PROVIDER
    available = ["TensorrtExecutionProvider", cuda, cpu] if cuda_available else [cpu]
    calls = []
    monkeypatch.setattr(engine, "get_preferred_device", lambda: "GPU")
    monkeypatch.setattr(engine.ort, "get_available_providers", lambda: available)
    monkeypatch.setattr(engine, "_preload_cuda_libraries", lambda: calls.append("preload"))

    class Session:
        def __init__(self, path, *, providers, sess_options):
            calls.append(providers)
            self.providers = providers

        def get_providers(self):
            return self.providers

    monkeypatch.setattr(engine.ort, "InferenceSession", Session)
    engine.create_onnx_session("model.onnx", device)
    assert calls == (["preload", [cuda, cpu]] if use_cuda else [[cpu]])
