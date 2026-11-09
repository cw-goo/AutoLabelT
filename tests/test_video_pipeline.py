"""Regression coverage for the verified video sampling and filtering baseline."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from autolabel.core import (ProjectConfig, classify_boxes, format_yolo,
                            is_frame_qualified, next_jump, run_pipeline)
from autolabel.validate import validate_dataset


def baseline_filter():
    source = ast.parse(Path("databuild.py").read_text(encoding="utf-8"))
    function = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                    and node.name == "is_frame_qualified")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "databuild.py", "exec"), namespace)
    return namespace["is_frame_qualified"]


@pytest.mark.parametrize("heights,expected", [
    ([], False), ([109], False), ([110], True),
    ([100, 20, 20], False), ([100, 100, 20, 20], True),
    ([100, 20, 20, 20], False), ([100, 100, 20, 20, 20], True),
])
def test_baseline_quality(heights, expected):
    valid = [(0, 0, 0, 0, 0, height) for height in heights if height >= 100]
    small = [(0, 0, 0, 0, 0, height) for height in heights if height < 100]
    assert is_frame_qualified(valid, small) == baseline_filter()(valid, small)
    assert is_frame_qualified(valid, small)[0] is expected


def test_companion_and_yolo_format():
    valid, small = classify_boxes([(0.5, 0.5, 0.2, 0.5),
                                   (0.2, 0.2, 0.1, 0.2),
                                   (0.8, 0.8, 0.1, 0.195)], 200, 100)
    assert format_yolo(valid, small) == ("0 0.500000 0.500000 0.200000 0.500000\n"
                                                "0 0.200000 0.200000 0.100000 0.200000\n")


def test_jumps_and_actual_defaults():
    config = ProjectConfig()
    assert [next_jump(keep, count, config) for keep, count in
            [(False, 0), (True, 1), (True, 2), (True, 4)]] == [500, 300, 160, 100]


def test_config_roundtrip_and_validation(tmp_path):
    config = ProjectConfig(video_path="source.mp4", output_dir="dataset",
                           confidence=0.42, step_single=7)
    path = tmp_path / "config.json"
    config.save(path)
    assert ProjectConfig.load(path) == config
    config.step_single = 0
    with pytest.raises(ValueError, match="step_single"):
        config.save(path)


class FakeCapture:
    def __init__(self):
        self.position = 0
        self.released = False
        self.visited = []

    def isOpened(self):
        return True

    def get(self, property_id):
        import cv2
        return {cv2.CAP_PROP_FRAME_COUNT: 6, cv2.CAP_PROP_FPS: 2,
                cv2.CAP_PROP_FRAME_HEIGHT: 200}.get(property_id, self.position)

    def set(self, property_id, value):
        self.position = int(value)
        return True

    def read(self):
        self.visited.append(self.position)
        return True, np.zeros((200, 100, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class FakeDetector:
    def predict(self, frame):
        return [(0.5, 0.5, 0.2, 0.6)]


def test_pipeline_source_indices_and_cancel(tmp_path):
    capture = FakeCapture()
    config = ProjectConfig(video_path="sample_ch2.mp4", output_dir=str(tmp_path),
                           step_single=2)
    events = []
    result = run_pipeline(config, events.append, lambda: len(events) == 2,
                          FakeDetector(), lambda _: capture)
    assert capture.visited == [0, 2]
    assert capture.released
    assert result == {"scanned": 2, "saved": 2, "discarded": 0, "cancelled": True}
    assert (tmp_path / "labels" / "ch2_f000002.txt").is_file()
    assert '"frame_index": 2' in (tmp_path / "metadata" / "samples.jsonl").read_text(encoding="utf-8")
    assert validate_dataset(tmp_path)["images"] == 2


def test_invalid_video_and_write_failure(tmp_path, monkeypatch):
    class Invalid(FakeCapture):
        def isOpened(self):
            return False

    config = ProjectConfig(video_path="bad.mp4", output_dir=str(tmp_path))
    capture = Invalid()
    with pytest.raises(ValueError):
        run_pipeline(config, detector=FakeDetector(), capture_factory=lambda _: capture)
    assert capture.released
    monkeypatch.setattr("cv2.imwrite", lambda *_: False)
    with pytest.raises(OSError, match="图片写入失败"):
        run_pipeline(config, detector=FakeDetector(), capture_factory=lambda _: FakeCapture())
    assert not list((tmp_path / "images").glob("*.jpg"))


def test_pipeline_rejects_model_class_ids_outside_dataset_before_writing(tmp_path):
    class InvalidClassDetector:
        def predict(self, _frame):
            return [(2, 0.5, 0.5, 0.5, 0.5, 0.9)]

    capture = FakeCapture()
    config = ProjectConfig(video_path="sample_ch2.mp4", output_dir=str(tmp_path))
    with pytest.raises(ValueError, match="类别 ID"):
        run_pipeline(config, detector=InvalidClassDetector(), capture_factory=lambda _: capture)
    assert capture.released
    assert not list((tmp_path / "images").glob("*.jpg"))
    assert not list((tmp_path / "labels").glob("*.txt"))


def test_dataset_validation_rejects_invalid_class_and_coordinates(tmp_path):
    import cv2
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    (tmp_path / "classes.txt").write_text("person\n", encoding="utf-8")
    cv2.imwrite(str(tmp_path / "images" / "one.jpg"), np.zeros((10, 10, 3), dtype=np.uint8))
    label = tmp_path / "labels" / "one.txt"
    label.write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="类别 ID"):
        validate_dataset(tmp_path)
    label.write_text("0 0.9 0.5 0.4 0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="坐标"):
        validate_dataset(tmp_path)
    label.write_text("0 0.508903 0.491324 0.982195 0.963007\n", encoding="utf-8")
    assert validate_dataset(tmp_path)["images"] == 1


@pytest.mark.parametrize("conflict", ["classes", "sample"])
def test_rejected_run_preserves_previous_config(tmp_path, conflict):
    configs = tmp_path / "configs"
    configs.mkdir()
    previous = configs / "run_config.json"
    previous.write_text("previous config", encoding="utf-8")
    (tmp_path / "classes.txt").write_text(
        "dog\n" if conflict == "classes" else "person\n", encoding="utf-8")
    if conflict == "sample":
        (tmp_path / "images").mkdir()
        (tmp_path / "images" / "ch2_f000300.jpg").write_bytes(b"existing")
    config = ProjectConfig(video_path="sample_ch2.mp4", output_dir=str(tmp_path))
    with pytest.raises((ValueError, FileExistsError)):
        run_pipeline(config, detector=FakeDetector(), capture_factory=lambda _: FakeCapture())
    assert previous.read_text(encoding="utf-8") == "previous config"
    assert list(configs.iterdir()) == [previous]


@pytest.mark.parametrize("model_id,expected", [("local_yolo", [0.9, 0.3]),
                                               ("florence2", [None, None])])
def test_video_scores_survive_review_and_recovery(tmp_path, model_id, expected):
    from autolabel.annotation_bridge import prepare_review_dataset

    class ScoredDetector:
        def predict(self, frame):
            return [(0, 0.5, 0.5, 0.2, 0.6, 0.9),
                    (0, 0.2, 0.2, 0.1, 0.2, 0.3),
                    (0, 0.8, 0.8, 0.1, 0.1, 0.1)]

    config = ProjectConfig(video_path="sample_ch2.mp4", output_dir=str(tmp_path),
                           model_id=model_id, small_target_ratio=1.0)
    run_pipeline(config, detector=ScoredDetector(), capture_factory=lambda _: FakeCapture())
    prepare_review_dataset(tmp_path)
    annotation = tmp_path / "images" / "ch2_f000000.json"
    data = json.loads(annotation.read_text(encoding="utf-8"))
    assert [shape.get("score") for shape in data["shapes"]] == expected
    data["shapes"][0]["score"] = 0.12
    annotation.write_text(json.dumps(data), encoding="utf-8")
    prepare_review_dataset(tmp_path)
    assert json.loads(annotation.read_text(encoding="utf-8"))["shapes"][0]["score"] == 0.12
    record = json.loads((tmp_path / "metadata" / "samples.jsonl").read_text(encoding="utf-8"))
    assert (tmp_path / "configs" / f"run_{record['run_id']}.json").is_file()


def test_cancel_during_prediction_writes_no_sample(tmp_path):
    cancelled = []

    class CancellingDetector:
        def predict(self, frame):
            cancelled.append(True)
            return [(0.5, 0.5, 0.2, 0.6)]

    result = run_pipeline(ProjectConfig(video_path="test.mp4", output_dir=str(tmp_path)),
                          cancelled=lambda: bool(cancelled), detector=CancellingDetector(),
                          capture_factory=lambda _: FakeCapture())
    assert result["cancelled"] and result["saved"] == 0
    assert not list((tmp_path / "images").iterdir())
