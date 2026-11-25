"""Model acquisition, lifecycle, and parameter wiring tests."""

import numpy as np
import pytest

from autolabel.core import ProjectConfig, classify_boxes, format_yolo
from autolabel.models.adapters import _to_yolo, parse_prompts
from autolabel.models.downloader import download_model


def test_temporary_model_cleanup_only_removes_current_session(tmp_path, monkeypatch):
    from autolabel.models.downloader import model_path, release_session_model

    temporary = tmp_path / "session"
    monkeypatch.setenv("AUTOLABEL_MODEL_DIR", str(temporary))
    monkeypatch.setenv("AUTOLABEL_SESSION_MODEL_DIR", str(temporary))
    location = model_path("grounding_dino")
    location.mkdir(parents=True)
    (location / "model.safetensors").write_bytes(b"test")
    personal_weight = tmp_path / "staffv2.pt"
    personal_weight.write_bytes(b"user")
    assert release_session_model("local_yolo") is False
    assert personal_weight.is_file()
    assert release_session_model("grounding_dino") is True
    assert not location.exists()
    monkeypatch.delenv("AUTOLABEL_SESSION_MODEL_DIR")
    location.mkdir(parents=True)
    assert release_session_model("grounding_dino") is False
    assert location.is_dir()


def test_desktop_entry_cleans_temporary_models_on_exit(monkeypatch):
    import os
    from pathlib import Path
    import sys
    import autolabel.workbench.app
    from autolabel.desktop import main

    monkeypatch.setattr(sys, "argv", ["autolabel"])
    created = []

    def stop_application():
        location = Path(os.environ["AUTOLABEL_SESSION_MODEL_DIR"])
        assert location == Path(os.environ["AUTOLABEL_MODEL_DIR"])
        (location / "weight.pt").write_bytes(b"test")
        created.append(location)
        raise SystemExit(0)

    monkeypatch.setattr(autolabel.workbench.app, "main", stop_application)
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 0
    assert not created[0].exists()


def test_prompt_class_mapping_and_coordinates():
    assert parse_prompts(" person, phone\nperson ") == ["person", "phone"]
    boxes = {"xyxy": np.asarray([[10, 20, 30, 60], [50, 10, 70, 30]]),
             "scores": np.asarray([0.9, 0.8])}
    predicted = _to_yolo(boxes, ["phone", "person"], ["person", "phone"], 100, 100)
    valid, small = classify_boxes(predicted, 100, 30)
    assert format_yolo(valid, small, companion_min_height=20) == (
        "1 0.200000 0.400000 0.200000 0.400000\n"
        "0 0.600000 0.200000 0.200000 0.200000\n")


def test_prompt_mapping_rejects_unknown_and_invalid_boxes():
    boxes = {"xyxy": np.asarray([[0, 0, 10, 10], [5, 5, 5, 9]]),
             "scores": np.asarray([0.8, 0.9])}
    assert _to_yolo(boxes, ["dog", "person"], ["person"], 100, 100) == []
    with pytest.raises(ValueError):
        parse_prompts(" , ")


def test_prompt_mapping_does_not_guess_ambiguous_or_empty_names():
    boxes = {"xyxy": np.asarray([[0, 0, 10, 10]]), "scores": np.asarray([0.8])}
    assert _to_yolo(boxes, ["shopping cart"], ["car", "cart"], 100, 100) == []
    assert _to_yolo(boxes, [""], ["car", "cart"], 100, 100) == []
    assert _to_yolo(boxes, ["cart"], ["car", "cart"], 100, 100)[0][0] == 1


def test_florence_requires_explicit_code_trust(tmp_path):
    from autolabel.models.adapters import Florence2Detector

    with pytest.raises(ValueError, match="信任模型代码"):
        Florence2Detector(tmp_path, ["person"], trust_remote_code=False)


def test_model_download_failure_keeps_no_ready_manifest(tmp_path, monkeypatch):
    import autolabel.models.downloader as downloader

    monkeypatch.setenv("AUTOLABEL_MODEL_DIR", str(tmp_path))
    def fail(*_args, **_kwargs):
        raise OSError("network failed")
    monkeypatch.setattr(downloader.urllib.request, "urlopen", fail)
    with pytest.raises(OSError, match="network failed"):
        download_model("yolo_world")
    assert not list(tmp_path.rglob("manifest.json"))


def test_model_manifest_belongs_to_the_tested_model(tmp_path, monkeypatch):
    import json
    from autolabel.models.downloader import mark_model_ready, model_download_state, model_path
    from autolabel.models.registry import get_model

    monkeypatch.setenv("AUTOLABEL_MODEL_DIR", str(tmp_path))
    spec = get_model("grounding_dino")
    location = model_path(spec.model_id)
    location.mkdir(parents=True)
    weight = location / "model.safetensors"
    weight.write_bytes(b"weights")
    manifest = location / "manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    assert model_download_state(spec.model_id) == "missing"
    manifest.write_text(json.dumps({"model_id": "another_model", "revision": spec.revision,
                                    "status": "downloaded"}), encoding="utf-8")
    mark_model_ready(spec.model_id, weight)
    assert json.loads(manifest.read_text())["status"] == "downloaded"
    manifest.write_text(json.dumps({"model_id": spec.model_id, "revision": spec.revision,
                                    "status": "downloaded"}), encoding="utf-8")
    mark_model_ready(spec.model_id, weight)
    assert model_download_state(spec.model_id) == "ready"


def test_pretrained_yolo_catalog_and_managed_download(tmp_path, monkeypatch):
    import io
    import json
    import autolabel.models.downloader as downloader
    from autolabel.models.registry import get_model

    for family, release in (("yolov8", "v8.2.0"), ("yolo11", "v8.3.0"),
                            ("yolo26", "v8.4.0")):
        for size in "nsmlx":
            spec = get_model(f"{family}_{size}")
            assert spec.weight_name == f"{family}{size}.pt"
            assert f"/{release}/{spec.weight_name}" in spec.weight_url
            assert spec.backend == "ultralytics_coco"
    monkeypatch.setenv("AUTOLABEL_MODEL_DIR", str(tmp_path))
    data = b"model" * 300000

    class Response(io.BytesIO):
        headers = {"Content-Length": str(len(data))}

    monkeypatch.setattr(downloader.urllib.request, "urlopen",
                        lambda *_args, **_kwargs: Response(data))
    progress = []
    location = downloader.download_model("yolov8_n", progress.append)
    assert (location / "yolov8n.pt").read_bytes() == data
    assert downloader.model_weight_path("yolov8_n") == location / "yolov8n.pt"
    assert downloader.model_download_state("yolov8_n") == "downloaded"
    manifest = json.loads((location / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["weights_sha256"]["yolov8n.pt"]
    assert progress[-1]["downloaded"] == len(data)


def test_pretrained_yolo_maps_existing_classes_and_rejects_unknown(monkeypatch):
    import sys
    from types import SimpleNamespace
    from autolabel.models.adapters import CocoYOLODetector

    class FakeYOLO:
        names = {0: "person", 1: "bicycle", 67: "cell phone"}

        def __init__(self, _path):
            self.requested = None

        def predict(self, **kwargs):
            self.requested = kwargs["classes"]
            boxes = SimpleNamespace(
                xywhn=SimpleNamespace(cpu=lambda: SimpleNamespace(
                    numpy=lambda: np.asarray([[0.5, 0.5, 0.2, 0.2]]))),
                cls=SimpleNamespace(cpu=lambda: SimpleNamespace(
                    numpy=lambda: np.asarray([67]))),
                conf=SimpleNamespace(cpu=lambda: SimpleNamespace(
                    numpy=lambda: np.asarray([0.9]))))
            return [SimpleNamespace(boxes=boxes)]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    detector = CocoYOLODetector("model.pt", ["cell phone", "person"], device="cpu")
    assert detector.predict(np.zeros((20, 20, 3), dtype=np.uint8))[0][0] == 0
    assert set(detector.model.requested) == {0, 67}
    with pytest.raises(ValueError, match="不包含类别：phone"):
        CocoYOLODetector("model.pt", ["phone"], device="cpu")


def test_sam3_uses_text_prompts_and_reports_gated_download(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    import autolabel.models.downloader as downloader
    from autolabel.models.adapters import SAM3Detector

    class FakeSAM3:
        def __init__(self, overrides):
            self.overrides = overrides
            self.queries = []

        def set_image(self, frame):
            self.frame = frame

        def __call__(self, text):
            self.queries.append(text)
            boxes = SimpleNamespace(
                xywhn=SimpleNamespace(cpu=lambda: SimpleNamespace(
                    numpy=lambda: np.asarray([[0.5, 0.5, 0.2, 0.2]]))),
                conf=SimpleNamespace(cpu=lambda: SimpleNamespace(
                    numpy=lambda: np.asarray([0.8]))))
            return [SimpleNamespace(boxes=boxes)]

    monkeypatch.setitem(sys.modules, "ultralytics.models.sam",
                        SimpleNamespace(SAM3SemanticPredictor=FakeSAM3))
    detector = SAM3Detector("sam3.pt", ["person", "phone"], device="cpu")
    boxes = detector.predict(np.zeros((20, 20, 3), dtype=np.uint8))
    assert [box[0] for box in boxes] == [0, 1]
    assert detector.predictor.queries == [["person"], ["phone"]]

    monkeypatch.setenv("AUTOLABEL_MODEL_DIR", str(tmp_path))
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda **_kwargs: (_ for _ in ()).throw(PermissionError("gated")))
    with pytest.raises(PermissionError, match="申请访问并登录"):
        downloader.download_model("sam3")
    assert not list(tmp_path.rglob("manifest.json"))
