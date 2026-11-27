"""Batch sampling regressions without loading model weights."""

from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from autolabelt.core import DYNAMIC_JUMP_KEYS, ProjectConfig
from autolabelt.video_batch import (TargetSampler, allocate, analyze_videos, plan_signature,
                                   recommend_sampling, run_batch, source_prefix)


class Capture:
    def __init__(self, path, frames=120):
        self.path, self.frames, self.position = path, frames, 0
        self.visited = []
        self.released = False

    def isOpened(self):
        return True

    def get(self, key):
        return {cv2.CAP_PROP_FRAME_COUNT: self.frames, cv2.CAP_PROP_FPS: 2,
                cv2.CAP_PROP_FRAME_HEIGHT: 200}.get(key, self.position)

    def set(self, key, value):
        self.position = int(value)

    def read(self):
        self.visited.append(self.position)
        image = np.zeros((200, 20, 3), dtype=np.uint8)
        image[0, 0, 0] = self.position
        image[0, 0, 1] = int("empty" in self.path)
        return True, image

    def release(self):
        self.released = True


class Detector:
    def __init__(self, qualified=lambda frame: True):
        self.qualified = qualified

    def predict(self, frame):
        return [(0.5, 0.5, 0.2, 0.6)] if self.qualified(frame) else []


def config_for(tmp_path, paths=None, target=24):
    paths = paths or ["clip_ch1_a.mp4", "clip_ch1_b.mp4"]
    return ProjectConfig(video_path=paths[0], video_paths=paths,
                         output_dir=str(tmp_path), target_count=target, step_single=10)


def test_analysis_covers_timeline_without_writing_and_releases(tmp_path):
    captures, events = [], []
    def factory(path):
        cap = Capture(path)
        captures.append(cap)
        return cap
    plan = analyze_videos(config_for(tmp_path), events.append, detector=Detector(),
                          capture_factory=factory)
    assert sum(report["quota"] for report in plan["videos"]) == 24
    assert all(report["rate"] == 1 for report in plan["videos"])
    assert all(min(cap.visited) < 15 and max(cap.visited) > 105 for cap in captures)
    assert all(cap.released for cap in captures)
    assert not list(tmp_path.iterdir())
    assert events[-1]["video_index"] == 1


def test_analysis_estimates_detection_counts_separately_from_qualification(tmp_path):
    class MixedDetector:
        def predict(self, frame):
            count = {9: 0, 22: 1, 37: 2, 51: 4, 69: 0, 82: 1, 97: 2, 111: 4}[int(frame[0, 0, 0])]
            return [(0.5, 0.5, 0.2, 0.2 if count == 1 else 0.6)] * count

    config = replace(config_for(tmp_path, ["mixed.mp4"], 0), output_dir="")
    plan = analyze_videos(config, detector=MixedDetector(), capture_factory=Capture)
    report = plan["videos"][0]
    assert plan["total_frames"] == 120
    assert report["target_counts"] == dict(empty=2, single=2, small_group=2, large_group=2)
    assert report["estimated_frames"] == dict(empty=30, single=30, small_group=30, large_group=30)
    assert report["rate"] == 0.5 and report["recommended_jump"] is None
    assert report["jump_counts"] == dict(zip(DYNAMIC_JUMP_KEYS, (4, 0, 2, 2)))
    assert plan["dynamic_jumps"] == dict(zip(DYNAMIC_JUMP_KEYS, (20, 12, 6, 4)))
    targeted = recommend_sampling(plan["videos"], replace(config, target_count=8))
    assert targeted["dynamic_jumps"] == dict(zip(DYNAMIC_JUMP_KEYS, (30, 18, 10, 6)))
    dense = recommend_sampling(plan["videos"], replace(config, target_count=200))
    assert dense["dynamic_jumps"] == dict.fromkeys(DYNAMIC_JUMP_KEYS, 2)


def test_batch_shares_detector_and_preserves_same_camera_sources(tmp_path, monkeypatch):
    import autolabelt.video_batch as batch
    loads = []
    monkeypatch.setattr(batch, "UltralyticsDetector", lambda config: loads.append(config) or Detector())
    events = []
    result = run_batch(config_for(tmp_path), events.append, capture_factory=Capture)
    assert len(loads) == 1
    assert result["saved"] == 24 and result["shortfall"] == 0
    assert len(list((tmp_path / "images").glob("*.jpg"))) == 24
    records = [json.loads(line) for line in (tmp_path / "metadata" / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    sources = list(dict.fromkeys(record["source_video"] for record in records))
    assert sources == config_for(tmp_path).video_paths
    for source in sources:
        indices = sorted(record["frame_index"] for record in records if record["source_video"] == source and record["kept"])
        assert min(indices) < 15 and max(indices) > 105
        assert all(b - a >= 2 for a, b in zip(indices, indices[1:]))
    assert len({source_prefix(path) for path in sources}) == 2
    assert events[-1]["phase"] == "video_done"
    assert events[-1]["batch_saved"] == 24


def test_empty_later_video_transfers_deficit_back_to_earlier(tmp_path):
    config = config_for(tmp_path, ["good.mp4", "empty.mp4"], 30)
    result = run_batch(config, detector=Detector(lambda frame: not frame[0, 0, 1]), capture_factory=Capture)
    assert result["saved"] == 30
    assert result["shortfall"] == 0
    assert sum(video["saved"] for video in result["videos"]) == 30
    assert not list((tmp_path / "images").glob(f"{source_prefix('empty.mp4')}_*.jpg"))
    assert len(list((tmp_path / "images").glob("*.jpg"))) == 30


def test_impossible_target_reports_shortfall_without_relaxing_quality(tmp_path):
    result = run_batch(config_for(tmp_path, target=200), detector=Detector(), capture_factory=Capture)
    assert result["saved"] == 120  # 60 minimum-gap positions per video
    assert result["shortfall"] == 80
    other = tmp_path / "empty"
    result = run_batch(config_for(other), detector=Detector(lambda _: False), capture_factory=Capture)
    assert result["saved"] == 0 and result["shortfall"] == 24


def test_sampler_supplements_sparse_content_without_duplicate_predictions():
    sampler = TargetSampler(120, 2, 15, rate=0)
    visited = []
    for frame in sampler:
        visited.append(frame)
        sampler.observe(frame, frame % 4 == 0)
    assert len(sampler.saved_indices) == 15
    assert len(visited) == len(set(visited))
    assert all(frame % 4 == 0 for frame in sampler.saved_indices)


def test_sampler_uses_recommended_initial_density():
    sparse = iter(TargetSampler(100, 1, 2, initial_jump=50))
    dense = iter(TargetSampler(100, 1, 2, initial_jump=10))
    assert next(sparse) == 25
    assert next(dense) == 5


def test_sampler_can_fill_off_grid_between_existing_frames():
    sampler = TargetSampler(12, 4, 1, existing_frames=(1, 9))
    positions = []
    for frame in sampler:
        positions.append(frame)
        sampler.observe(frame, True)
    assert positions == [5]


def test_cancel_during_analysis_writes_no_dataset(tmp_path):
    events = []
    result = run_batch(config_for(tmp_path), events.append,
                       cancelled=lambda: len(events) == 2, detector=Detector(), capture_factory=Capture)
    assert result["cancelled"] and not result["saved"]
    assert not list(tmp_path.iterdir())


def test_cancel_processing_stops_queue_and_preserves_samples(tmp_path):
    events = []
    config = config_for(tmp_path)
    plan = analyze_videos(config, detector=Detector(), capture_factory=Capture)
    result = run_batch(config, events.append,
                       cancelled=lambda: any(event.get("batch_saved", 0) >= 3 for event in events),
                       detector=Detector(), plan=plan, capture_factory=Capture)
    assert result["cancelled"] and result["saved"] == 3
    assert len(list((tmp_path / "images").glob("*.jpg"))) == 3
    assert not any(event.get("video_index") == 1 for event in events)


def test_batch_preflight_rejects_collision_before_first_video(tmp_path):
    config = config_for(tmp_path)
    images = tmp_path / "images"
    images.mkdir()
    existing = images / f"{source_prefix(config.video_paths[1])}_f000000.jpg"
    existing.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        run_batch(config, detector=Detector(), capture_factory=Capture)
    assert list(images.iterdir()) == [existing]
    assert existing.read_bytes() == b"original"


def test_repeat_build_counts_existing_images_and_preserves_old_labels(tmp_path):
    config = replace(config_for(tmp_path, target=12), min_sample_frames=3)
    first = run_batch(config, detector=Detector(), capture_factory=Capture)
    assert first["dataset_total"] == 12
    labels = {path: path.read_bytes() for path in (tmp_path / "labels").glob("*.txt")}
    manifest = tmp_path / "metadata" / "samples.jsonl"
    lines = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    for record in lines:
        record.pop("image_name")
    manifest.write_text("".join(json.dumps(record) + "\n" for record in lines), encoding="utf-8")

    config = replace(config, target_count=24, min_sample_frames=4)
    continued = run_batch(config, detector=Detector(), capture_factory=Capture)
    assert continued["existing"] == 12 and continued["saved"] == 12
    assert continued["dataset_total"] == 24 and continued["shortfall"] == 0
    assert all(path.read_bytes() == content for path, content in labels.items())
    assert len(list((tmp_path / "images").glob("*.jpg"))) == 24
    kept = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    indices = [(record["source_video"], record["frame_index"]) for record in kept if record["kept"]]
    assert len(set(indices)) == len(indices) == 24
    for index, (source, frame) in enumerate(indices[12:], 12):
        assert all(abs(other_frame - frame) >= 4 for other_source, other_frame in indices[:index]
                   if source == other_source)
    already_done = run_batch(config, capture_factory=Capture)
    assert already_done["dataset_total"] == 24 and already_done["saved"] == 0


def test_full_batch_uses_baseline_jumps(tmp_path):
    config = config_for(tmp_path, target=0)
    result = run_batch(config, detector=Detector(), capture_factory=Capture)
    assert result["saved"] == 24
    single = replace(config, video_paths=["legacy_ch2.mp4"], output_dir=str(tmp_path / "single"))
    result = run_batch(single, detector=Detector(), capture_factory=Capture)
    assert result["saved"] == 12
    assert (tmp_path / "single" / "images" / "ch2_f000010.jpg").is_file()
    continued = run_batch(single, detector=Detector(), capture_factory=Capture)
    assert continued["saved"] > 0 and continued["dataset_total"] > 12


def test_config_backward_compatibility_and_queue_roundtrip(tmp_path):
    config = replace(config_for(tmp_path), min_sample_frames=7)
    path = tmp_path / "config.json"
    config.save(path)
    from autolabelt.core import ProjectConfig
    assert ProjectConfig.load(path) == config
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("video_paths", "target_count", "min_sample_seconds", "min_sample_frames"):
        data.pop(key)
    data["schema_version"] = 2
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = ProjectConfig.load(path)
    assert loaded.target_count == 0 and loaded.video_paths == [] and loaded.min_sample_frames is None
    assert sum(allocate(5, [1, 4, 2])) == 5


def test_queue_ui_reorders_removes_and_reads_target(monkeypatch):
    from PyQt6 import QtWidgets
    from autolabelt.video_dock import VideoDatasetDock
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = QtWidgets.QMainWindow()
    dock = VideoDatasetDock(window)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames", lambda *args: (["a.mp4", "b.mp4", "a.mp4"], ""))
    monkeypatch.setattr("autolabelt.video_dock.video_info", lambda path: {
        "width": 100, "height": 200, "fps": 2, "frames": 120, "duration": 60})
    dock.choose_video()
    assert dock.config.video_paths == ["a.mp4", "b.mp4"]
    dock.video_queue.setCurrentItem(dock.video_queue.topLevelItem(1))
    dock.move_video(-1)
    assert dock.config.video_path == "b.mp4"
    dock.target_count.setValue(3000)
    assert dock._current_config().target_count == 3000
    dock.remove_video()
    assert dock.config.video_paths == ["a.mp4"]
    dock.close()
    window.close()
    application.processEvents()


def test_dock_fills_sampling_table_and_replans_without_inference(tmp_path, monkeypatch):
    from PyQt6 import QtCore, QtWidgets
    from autolabelt.video_dock import ParameterDialog, VideoDatasetDock

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = QtWidgets.QMainWindow()
    dock = VideoDatasetDock(window)
    source, weight = tmp_path / "video.mp4", tmp_path / "model.pt"
    source.touch()
    weight.touch()
    config = replace(config_for(tmp_path, [str(source)], 0), model_path=str(weight),
                     min_sample_seconds=2)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames", lambda *args: ([str(source)], ""))
    monkeypatch.setattr("autolabelt.video_dock.video_info", lambda path: {
        "width": 100, "height": 200, "fps": 2, "frames": 120, "duration": 60})
    dock.choose_video()
    dock.config = replace(config, output_dir="")
    plan = analyze_videos(config, detector=Detector(), capture_factory=Capture)
    signature = plan_signature(config)
    dock._completed({"analysis": True, "plan": plan, "signature": signature})
    assert "待确认参数" in dock.stats.text() and not dock.start_button.isEnabled()
    output = tmp_path / "unlimited"
    dock.config.output_dir = str(output)
    applied = dock._current_config()
    assert applied.step_single == 24 and applied.target_count == 0
    result = run_batch(applied, detector=Detector(), capture_factory=Capture)
    assert result["saved"] == 5
    assert (output / "images" / "video_f000024.jpg").is_file()
    assert ProjectConfig.load(output / "configs" / "run_config.json").step_single == 24

    dock.target_count.setValue(12)
    dock.sample_gap.setValue(4)
    assert dock.analysis_signature == plan_signature(dock._current_config()) == signature
    assert "120" in dock.analysis_summary.text()
    assert dock.video_queue.topLevelItem(0).text(1) == "12"
    assert "未检出目标" in dock.video_queue.topLevelItem(0).toolTip(1)
    table = dock.sampling_plan_table
    assert table.parentWidget() is dock.analysis_results
    assert dock.analysis_results.parentWidget() is dock.workflow_cards[2]
    assert table.columnCount() == 3
    assert [table.topLevelItem(0).text(column) for column in (1, 2)] == ["12 张", "10 帧"]
    assert dock._current_config().min_sample_seconds == 2
    assert dock._current_config().min_sample_frames == 4

    def check_dialog_and_edit(dialog):
        assert dialog.findChild(QtWidgets.QTreeWidget, "samplingPlanTable") is None
        headings = [label.text() for label in dialog.findChildren(QtWidgets.QLabel)
                    if label.objectName() == "parameterSectionTitle"]
        assert "动态抽帧" in headings
        assert {key: dialog.controls[key].value() for key in DYNAMIC_JUMP_KEYS} == dock.analysis_plan["dynamic_jumps"]
        dialog.controls["step_single"].setValue(11)
        return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(ParameterDialog, "exec", check_dialog_and_edit)
    dock.edit_parameters()
    assert dock.analysis_signature == signature
    assert dock._current_config().step_single == 11
    assert dock.start_button.isEnabled()
    dock.target_count.setValue(16)
    assert dock._current_config().step_single == 11
    assert not dock.start_button.isEnabled()
    dock.prompts.setText("phone")
    assert dock.analysis_plan is None
    assert table.topLevelItemCount() == 0
    assert dock.video_queue.topLevelItem(0).toolTip(1) == ""
    assert dock.config.step_single == 11
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    application.processEvents()


def test_build_updates_a_single_status_and_logs_only_video_milestones(monkeypatch):
    from PyQt6 import QtCore, QtWidgets
    from autolabelt.video_dock import VideoDatasetDock

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = QtWidgets.QMainWindow()
    dock = VideoDatasetDock(window)
    event = {"video_index": 0, "video_count": 2}
    dock._progress({**event, "phase": "video_start"})
    for scanned in range(1, 11):
        dock._progress({**event, "phase": "processing", "scanned": scanned,
                        "batch_saved": min(scanned, 6), "target": 10, "fraction": scanned / 10,
                        "reason": "合格"})
    dock._progress({**event, "phase": "video_done", "cancelled": False,
                    "saved": 6, "batch_saved": 6, "target": 10})
    assert dock.batch_progress.value() == 600
    assert "60%" in dock.stats.text() and "6/10" in dock.stats.text()
    assert dock.log.toPlainText().splitlines() == [
        "视频 1/2 · 开始构建", "视频 1/2 · 已完成，本轮保存 6 张"]

    dock._progress({**event, "phase": "video_start", "supplemental": True})
    assert dock.batch_progress.value() == 600 and "开始补采" in dock.stats.text()
    dock._progress({**event, "phase": "processing", "scanned": 1, "batch_saved": 7,
                    "target": 10, "fraction": 0.1, "supplemental": True})
    assert dock.batch_progress.value() == 700 and "补采" in dock.stats.text()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    application.processEvents()


def test_worker_analyzes_then_builds_with_cached_plan_and_live_labels(tmp_path, monkeypatch):
    from autolabelt.video_dock import VideoPipelineWorker
    from autolabelt.validate import validate_dataset
    sources = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    model = tmp_path / "model.pt"
    for path in [*sources, model]:
        path.touch()
    config = config_for(tmp_path / "dataset", [str(path) for path in sources], 12)
    config.model_path = str(model)
    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    monkeypatch.setattr("autolabelt.video_batch.UltralyticsDetector", lambda _: Detector())
    failures, analyzed, built = [], [], []
    analysis = VideoPipelineWorker(config, analyze_only=True)
    analysis.failed.connect(failures.append)
    analysis.completed.connect(analyzed.append)
    analysis.run()
    assert not failures and analyzed[0]["analysis"]
    assert not (tmp_path / "dataset").exists()
    builder = VideoPipelineWorker(config, plan=analyzed[0]["plan"])
    builder.failed.connect(failures.append)
    builder.completed.connect(built.append)
    builder.run()
    assert not failures and built[0]["saved"] == 12
    assert validate_dataset(config.output_dir)["images"] == 12
    labels = list((tmp_path / "dataset" / "images").glob("*.json"))
    assert len(labels) == 12
    assert json.loads(labels[0].read_text(encoding="utf-8"))["shapes"][0]["label"] == "person"


def test_desktop_reanalyzes_and_continues_with_one_video_import(tmp_path, monkeypatch):
    from PyQt6 import QtCore, QtTest, QtWidgets
    from autolabelt.models.downloader import model_path
    from autolabelt.video_dock import ParameterDialog, VideoDatasetDock

    app = QtWidgets.QApplication.instance()
    window = QtWidgets.QMainWindow()
    dock = VideoDatasetDock(window)
    window.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
    window.show()
    app.processEvents()
    paths = [str(tmp_path / name) for name in ("first.mp4", "second.mp4")]
    for path in paths:
        Path(path).touch()
    store = tmp_path / "models"
    monkeypatch.setenv("AUTOLABEL_MODEL_DIR", str(store))
    monkeypatch.setenv("AUTOLABEL_SESSION_MODEL_DIR", str(store))
    ids = ("grounding_dino", "grounding_dino_base")
    for model_id in ids:
        location = model_path(model_id)
        location.mkdir(parents=True)
        (location / "model.safetensors").touch()
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames", lambda *args: (paths, ""))
    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    loads = []
    monkeypatch.setattr("autolabelt.video_batch.UltralyticsDetector",
                        lambda config: loads.append(config.model_id) or Detector())
    errors = []
    monkeypatch.setattr(dock, "_error", errors.append)
    output = tmp_path / "dataset"
    dock.config.output_dir = str(output)
    monkeypatch.setattr(dock, "_show_live_image",
                        lambda image: setattr(dock, "live_review_dir", str(output / "images")))
    dock.choose_video()
    assert dock.target_count.text() == "0"
    result_height = dock.analysis_results.height()
    dock.target_count.setValue(6)
    dock.sample_gap.setValue(5)
    assert dock.target_count.text() == "6" and dock.sample_gap.text() == "5 帧"
    assert dock.target_count.y() == dock.sample_gap.y()
    assert not dock.start_button.isEnabled()

    def run(button):
        assert button.isEnabled()
        button.click()
        assert dock.pipeline_worker is not None
        for _ in range(200):
            QtTest.QTest.qWait(20)
            if dock.pipeline_worker is None:
                break
        assert dock.pipeline_worker is None and not errors
        assert dock.analyze_button.isEnabled()
        assert dock.analysis_results.height() == result_height
        assert dock.config.video_paths == paths

    def confirm(confidence=None, cancel=False):
        def select(dialog):
            assert any(button.text() == "使用参数" for button in dialog.findChildren(QtWidgets.QPushButton))
            if confidence is not None:
                dialog.controls["confidence"].setValue(confidence)
            return QtWidgets.QDialog.DialogCode.Rejected if cancel else QtWidgets.QDialog.DialogCode.Accepted

        with monkeypatch.context() as scope:
            scope.setattr(ParameterDialog, "exec", select)
            dock.parameter_button.click()

    dock.model_combo.setCurrentIndex(dock.model_combo.findData(ids[0]))
    run(dock.analyze_button)
    dock.model_combo.setCurrentIndex(dock.model_combo.findData(ids[1]))
    run(dock.analyze_button)
    assert all(report["gap"] == 5 for report in dock.analysis_plan["videos"])
    assert not dock.start_button.isEnabled()
    confirm(cancel=True)
    assert not dock.start_button.isEnabled()
    confirm()
    run(dock.start_button)
    assert len(list((output / "images").glob("*.jpg"))) == 6
    dock.target_count.setValue(12)
    assert not dock.start_button.isEnabled()
    run(dock.analyze_button)
    original_signature = dock.analysis_signature
    confirm(confidence=0.50)
    assert dock.confirmed_signature != original_signature
    run(dock.start_button)
    assert dock.analysis_signature == dock.confirmed_signature
    assert "本次新增 6 张" in dock.stats.text()
    assert len(list((output / "images").glob("*.jpg"))) == 12
    assert len(list((output / "images").glob("*.json"))) == 12
    assert loads == [ids[0], *([ids[1]] * 4)]
    assert all(model_path(model_id).is_dir() for model_id in ids)
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    app.processEvents()


def test_direct_build_publishes_plan_for_parameter_table(tmp_path, monkeypatch):
    from autolabelt.video_dock import VideoPipelineWorker

    source, model = tmp_path / "a.mp4", tmp_path / "model.pt"
    source.touch()
    model.touch()
    config = replace(config_for(tmp_path / "dataset", [str(source)], 4), model_path=str(model))
    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    monkeypatch.setattr("autolabelt.video_batch.UltralyticsDetector", lambda _: Detector())
    worker = VideoPipelineWorker(config)
    events, failures = [], []
    worker.progress.connect(events.append)
    worker.failed.connect(failures.append)
    worker.run()
    assert not failures
    event = next(event for event in events if event["phase"] == "plan")
    assert event["signature"] == plan_signature(config)
    assert event["plan"]["videos"][0]["quota"] == 4
