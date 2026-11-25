"""Training snapshots, worker telemetry and independent window lifecycle."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from autolabel.training_data import prepare_training_dataset


def dataset(root, filenames):
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    (root / "classes.txt").write_text("helmet\n", encoding="utf-8")
    for name in filenames:
        Image.new("RGB", (32, 32), "white").save(root / "images" / f"{name}.jpg")
        (root / "labels" / f"{name}.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    return root


def test_snapshot_groups_video_frames_and_freezes_labels(tmp_path):
    import json
    source = dataset(tmp_path / "source", ["a_f000001", "a_f000002", "b_f000001", "b_f000002"])
    result = prepare_training_dataset(source, tmp_path / "snapshot")
    assert result["train"] == result["val"] == 2
    manifest = json.loads((tmp_path / "snapshot/manifest.json").read_text())
    split_by_video = {}
    for row in manifest["samples"]:
        split_by_video.setdefault(Path(row["source"]).stem.split("_f")[0], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in split_by_video.values())
    original = next((source / "labels").glob("*.txt"))
    original.write_text("")
    assert all(p.read_text().strip() for p in (tmp_path / "snapshot/labels").rglob("*.txt"))


def test_generated_yaml_creates_directory_splits_and_can_prepare_snapshot(tmp_path):
    import yaml
    source = dataset(tmp_path / "source", ["one", "two", "three", "four"])
    destination = tmp_path / "configuration"
    result = prepare_training_dataset(source, destination, yaml_path=tmp_path / "data.yaml")
    config = yaml.safe_load(Path(result["data"]).read_text())
    assert config == {"path": str(destination), "train": "images/train",
                      "val": "images/val", "names": {0: "helmet"}}
    for split in ("train", "val"):
        images = {path.stem for path in (destination / config[split]).glob("*.jpg")}
        labels = {path.stem for path in (destination / "labels" / split).glob("*.txt")}
        assert images == labels and len(images) == result[split]
        assert not (destination / f"{split}.txt").exists()
    assert len(list((source / "images").glob("*.jpg"))) == 4
    snapshot = prepare_training_dataset(result["data"], tmp_path / "snapshot")
    assert snapshot["train"] == result["train"]
    assert snapshot["val"] == result["val"]


@pytest.mark.parametrize("names_in_yaml", [False, True])
def test_generated_yaml_reuses_existing_split_directories(tmp_path, names_in_yaml):
    import yaml

    source = dataset(tmp_path / "source", ["one", "two", "three"])
    for folder, suffix in (("images", "jpg"), ("labels", "txt")):
        for split, filenames in (("train", ["one"]), ("val", ["two", "three"])):
            (source / folder / split).mkdir()
            for name in filenames:
                filename = f"{name}.{suffix}"
                (source / folder / filename).rename(source / folder / split / filename)
    if names_in_yaml:
        (source / "classes.txt").unlink()
        (source / "data.yaml").write_text("train: images/train\nval: images/val\nnames: {0: helmet}\n")
    destination = tmp_path / "unused"
    yaml_path = tmp_path / "data.yaml"
    result = prepare_training_dataset(source, destination, yaml_path=yaml_path)
    assert yaml_path.read_text(encoding="utf-8") == (
        f"path: {source}\ntrain: images/train\nval: images/val\n\nnames:\n  0: helmet\n"
    )
    assert (result["train"], result["val"]) == (1, 2)
    assert not destination.exists()
    snapshot = prepare_training_dataset(source, tmp_path / "snapshot")
    assert (snapshot["train"], snapshot["val"]) == (1, 2)
    saved = tmp_path / "saved"
    copied = prepare_training_dataset(
        source, saved / "data_splits",
        yaml_path=saved / "data.yaml", copy_existing_splits=True,
    )
    split_root = Path(yaml.safe_load(Path(copied["data"]).read_text())["path"])
    assert split_root.parent == saved
    assert len(list((split_root / "images/train").glob("*.jpg"))) == 1
    assert len(list((split_root / "labels/val").glob("*.txt"))) == 2


def test_generate_yaml_and_split_in_selected_output_without_starting_training(tmp_path, monkeypatch):
    import yaml
    from PyQt6 import QtWidgets, QtTest
    from autolabel.training_window import TrainingWindow
    source = dataset(tmp_path / "source", ["one", "two", "three", "four"])
    window = TrainingWindow()
    window.source.setText(str(source))
    output = tmp_path / "runs"
    window.output.setText(str(output))
    selected = output / "data.yaml"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        lambda *args: pytest.fail("generation should use the selected output directory"))
    monkeypatch.setattr(window.manager, "start_training", lambda args: pytest.fail("generation must not train"))
    def wait_for_preparation():
        for _ in range(500):
            QtWidgets.QApplication.processEvents()
            if not window.is_busy():
                break
            QtTest.QTest.qWait(10)
        QtWidgets.QApplication.processEvents()

    def generate(ratio):
        window.ratio.setValue(ratio)
        window.generate_yaml()
        wait_for_preparation()
        return Path(yaml.safe_load(selected.read_text())["path"])

    first = generate(80)
    latest = generate(50)
    assert first != latest
    assert first.parent == latest.parent == output
    assert not (source / "data.yaml").exists()
    assert len(list((latest / "images/train").glob("*.jpg"))) == 2
    assert len(list((latest / "images/val").glob("*.jpg"))) == 2
    assert Path(window.yaml_path.text()) == selected
    assert selected.is_file()
    assert window.record is None
    assert not window.manager.is_training
    monkeypatch.setattr(window, "launch", lambda: None)
    for _ in range(2):
        window.ratio.setValue(80)
        window.start_training()
        wait_for_preparation()
        task = Path(window.record["directory"])
        assert "resume" not in window.record["args"]
        assert not (task / "dataset").exists()
        assert yaml.safe_load((task / "data.yaml").read_text())["path"] == str(latest)
        assert Path(window.record["sample_manifest"]) == task / "manifest.json"
        assert (task / "manifest.json").is_file()
        assert Path(window.yaml_path.text()) == selected
    window.timer.stop()
    window.deleteLater()


def test_training_uses_yaml_field_instead_of_dataset_directory(tmp_path, monkeypatch):
    import yaml
    from PyQt6 import QtWidgets, QtTest
    from autolabel.training_window import TrainingWindow
    original = dataset(tmp_path / "original", ["one", "two"])
    imported = dataset(tmp_path / "imported", ["one", "two"])
    (imported / "classes.txt").write_text("vest\n")
    generated = prepare_training_dataset(imported, tmp_path / "config",
                                         yaml_path=tmp_path / "imported.yaml")
    window = TrainingWindow()
    window.source.setText(str(original))
    window.yaml_path.setText(generated["data"])
    window.output.setText(str(tmp_path / "runs"))
    assert (window.epochs.value(), window.imgsz.value(), window.batch.value()) == (150, 1024, 24)
    assert window.device.currentText() == "1"
    assert window.initial_lr.isEnabled()
    window.optimizer.setText("auto")
    assert not window.initial_lr.isEnabled()
    window.optimizer.setText("AdamW")
    assert window.initial_lr.isEnabled()
    window.initial_lr.setValue(0.0002)
    window.final_lr_ratio.setValue(0.1)
    window.warmup_epochs.setValue(2)
    window.patience.setValue(20)
    window.mosaic.setValue(0.3)
    window.workers.setValue(2)
    window.cache.setText("disk")
    window.seed.setValue(21)
    window.amp.setChecked(False)
    captured = []
    def launch():
        captured.append(dict(window.record["args"]))
    monkeypatch.setattr(window, "launch", launch)
    window.start_training()
    for _ in range(500):
        QtWidgets.QApplication.processEvents()
        if not window.is_busy():
            break
        QtTest.QTest.qWait(10)
    QtWidgets.QApplication.processEvents()
    assert len(captured) == 1
    expected = {
        "optimizer": "AdamW", "lr0": 0.0002, "lrf": 0.1, "warmup_epochs": 2.0,
        "device": 1, "patience": 20, "close_mosaic": 20, "cache": "disk",
        "mosaic": 0.3, "scale": 0.3, "fliplr": 0.5, "hsv_h": 0.01, "hsv_s": 0.3,
        "hsv_v": 0.3, "erasing": 0.1, "workers": 2, "seed": 21, "amp": False,
        "deterministic": False,
    }
    assert {key: captured[0][key] for key in expected} == expected
    data = yaml.safe_load(Path(captured[0]["data"]).read_text())
    assert data["names"] == {0: "vest"}
    assert window.yaml_path.text() == generated["data"]
    window.timer.stop()
    window.deleteLater()


def test_repeat_training_reuses_membership_and_reads_latest_source_labels(tmp_path):
    import json

    source = dataset(tmp_path / "source", ["one", "two", "three", "four"])
    first = prepare_training_dataset(source, tmp_path / "first", ratio=0.75)
    first_manifest = json.loads(first["automatic_split_manifest"].read_text())
    (source / "labels/one.txt").write_text("0 0.4 0.5 0.5 0.5\n")
    second = prepare_training_dataset(
        source, tmp_path / "second", ratio=0.50, split_manifest=first["automatic_split_manifest"]
    )
    second_manifest = json.loads(second["automatic_split_manifest"].read_text())
    assert (second["train"], second["val"]) == (3, 1)
    assert second_manifest["ratio"] == first_manifest["ratio"]
    original, current = (
        next(row for row in samples["samples"] if Path(row["source"]).stem == "one")
        for samples in (first_manifest, second_manifest)
    )
    assert original["split"] == current["split"]
    assert original["label_sha256"] != current["label_sha256"]


def test_repeat_training_reuses_generated_files_and_refreshes_labels(tmp_path):
    import hashlib
    import json
    import yaml

    source = dataset(tmp_path / "source", ["one", "two", "three"])
    first = prepare_training_dataset(source, tmp_path / "first" / "dataset", reference_splits=True)
    original_partition = Path(first["automatic_split_manifest"]).parent
    target = next((original_partition / "labels").rglob("*_one.txt"))
    cache = target.parent.with_suffix(".cache")
    cache.write_text("stale label cache")
    (source / "labels/one.txt").write_text("0 0.4 0.5 0.5 0.5\n")
    second_dir = tmp_path / "second"
    second = prepare_training_dataset(source, second_dir / "dataset",
                                      split_manifest=first["automatic_split_manifest"], reference_splits=True)
    assert yaml.safe_load(Path(second["data"]).read_text())["path"] == str(original_partition)
    assert not (second_dir / "dataset").exists()
    assert not cache.exists()
    assert target.read_text() == "0 0.4 0.5 0.5 0.5\n"
    first_manifest, second_manifest = (
        json.loads(result["manifest"].read_text(encoding="utf-8")) for result in (first, second)
    )
    sample = next(row for row in second_manifest["samples"] if Path(row["label_path"]) == target)
    original = next(row for row in first_manifest["samples"] if row["source"] == sample["source"])
    assert sample["label_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert original["label_sha256"] != sample["label_sha256"]


def test_training_history_labels_records_and_converts_selected_checkpoint(tmp_path, monkeypatch):
    import json
    import sys

    from PyQt6 import QtWidgets
    from autolabel.training_window import TrainingWindow

    task = tmp_path / "20261124_122928_24e6eb"
    checkpoint = task / "train/weights/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    (task / "task.json").write_text(json.dumps({
        "directory": str(task), "source": str(tmp_path / "staffv4/data.yaml"),
        "python": sys.executable, "status": "训练完成",
        "args": {"model": str(tmp_path / "weights/staffv3.pt"), "imgsz": 1280, "model_family": "yolo"},
    }, ensure_ascii=False), encoding="utf-8")
    window = TrainingWindow()
    window.output.setText(str(tmp_path))
    window.yaml_path.setText("next-training.yaml")
    window.load_history()
    item = window.history.item(0)
    assert item.text() == "staffv4 · staffv3\n11-24 12:29 · 训练完成"
    assert str(task) in item.toolTip()
    window.history.setCurrentItem(item)
    assert window.yaml_path.text() == "next-training.yaml"
    assert window.export_button.isEnabled()
    assert window.export_button.text() == "格式转换"
    assert not {"保存权重", "用于标注"}.intersection(
        button.text() for button in window.findChildren(QtWidgets.QPushButton)
    )
    target = tmp_path / "staff.onnx"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *_: (str(target), ""))
    calls = []

    def start_export(*args, **kwargs):
        calls.append((args, kwargs))
        return True, ""

    monkeypatch.setattr(window.export_manager, "start_checkpoint_export", start_export)
    window.export_button.click()
    assert window.export_dialog.current.isEnabled()
    window.export_dialog.current.click()
    assert calls == [((str(checkpoint),),
                      {"destination": str(target), "settings": {"imgsz": 1280, "model_family": "yolo"}})]
    window.on_export_event("export_completed", {"exported_path": str(target)})
    assert str(target) in window.results.toPlainText()
    window.export_dialog.close()
    window.timer.stop()
    window.deleteLater()


def test_format_conversion_imports_local_pt_without_training_history(tmp_path, monkeypatch):
    from PyQt6 import QtWidgets
    from autolabel.training_window import TrainingWindow

    source = tmp_path / "local.pt"
    source.write_bytes(b"test checkpoint")
    destination = tmp_path / "converted.onnx"
    window = TrainingWindow()
    assert window.record is None
    assert window.export_button.isEnabled()
    assert window.export_button.menu() is None
    window.export_button.click()
    dialog = window.export_dialog
    assert dialog.isVisible()
    assert not dialog.current.isEnabled()
    assert not dialog.findChildren(QtWidgets.QLabel)
    assert not dialog.progress.isTextVisible()
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", lambda *_: (str(source), ""))
    calls = []

    def start_export(checkpoint, **settings):
        calls.append((checkpoint, settings))
        window.export_manager.is_exporting = True
        return True, ""

    monkeypatch.setattr(window.export_manager, "start_checkpoint_export", start_export)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *_: ("", ""))
    dialog.local.click()
    assert calls == []
    assert dialog.progress.value() == 0

    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *_: (str(destination), ""))
    dialog.local.click()
    assert calls == [(str(source), {"destination": str(destination),
                                   "settings": {"model_family": None, "imgsz": None}})]
    assert dialog.progress.maximum() == 0
    assert not dialog.local.isEnabled()
    assert dialog.close_button.text() == "停止转换"
    dialog.close()
    window.export_button.click()
    assert dialog.isVisible() and dialog.progress.maximum() == 0
    window.export_manager.is_exporting = False
    window.on_export_event("export_completed", {"exported_path": str(destination)})
    assert dialog.progress.value() == dialog.progress.maximum() == 100
    assert dialog.local.isEnabled()
    assert str(destination) in window.state.text()
    assert window.record is None
    dialog.close_button.click()
    assert not dialog.isVisible()
    window.timer.stop()
    window.deleteLater()


def test_training_worker_passes_grouped_parameters_to_yolo(monkeypatch):
    import ultralytics
    from autolabel.workbench.services.auto_training.ultralytics import worker

    arguments = {}
    model = SimpleNamespace(
        add_callback=lambda *_: None,
        train=lambda **kwargs: arguments.update(kwargs),
    )
    monkeypatch.setattr(ultralytics, "YOLO", lambda _weight: model)
    monkeypatch.setattr(worker, "emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker, "configure_weights_directory", lambda _path: None)
    settings = {"cache": "ram", "mosaic": 1.0, "close_mosaic": 20, "fliplr": 0.5,
                "erasing": 0.1, "deterministic": False}
    worker.run_train({
        "args": {"model": "yolo11n.pt", "device": "cpu", **settings},
        "paths": {"weights_directory": ""},
    })
    assert {key: arguments[key] for key in settings} == settings


def test_advanced_settings_dialog_handles_manual_input_and_cancel():
    from PyQt6 import QtCore, QtWidgets
    from autolabel.training_window import TrainingWindow

    window = TrainingWindow()
    dialog = window.advanced_dialog
    assert not dialog.findChildren(QtWidgets.QComboBox)
    assert {group.title() for group in dialog.findChildren(QtWidgets.QGroupBox)} == {
        "数据加载", "迁移训练", "训练控制", "数据增强", "性能设置"
    }
    assert window.workers.value() == 8
    assert window.cache.text() == "ram"
    assert window.patience.value() == 30
    assert window.initial_lr.value() == 0.001
    assert all(field.buttonSymbols() == QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons
               for field in window.advanced_numbers)
    assert any("0–1" in label.text() for label in dialog.findChildren(QtWidgets.QLabel))
    window.show()
    QtWidgets.QApplication.processEvents()
    button_position = window.advanced_button.mapTo(window, QtCore.QPoint())
    ratio_position = window.ratio.mapTo(window, QtCore.QPoint())
    yaml_position = window.yaml_path.mapTo(window, QtCore.QPoint())
    output_position = window.output.mapTo(window, QtCore.QPoint())
    assert abs(yaml_position.y() - ratio_position.y()) <= 2
    assert output_position.y() < yaml_position.y()
    assert button_position.y() > ratio_position.y()
    assert abs(window.epochs.mapTo(window, QtCore.QPoint()).y() - button_position.y()) <= 2
    assert (button_position.x() + window.advanced_button.width()
            == ratio_position.x() + window.ratio.width())
    assert "继续训练" not in {button.text() for button in window.findChildren(QtWidgets.QPushButton)}
    assert "训练前保存并同步标注" not in {
        label.text() for label in window.findChildren(QtWidgets.QLabel)
    }

    def cancel():
        window.mosaic.setValue(0.2)
        window.cache.setText("disk")
        window.deterministic.setChecked(True)
        dialog.reject()

    QtCore.QTimer.singleShot(0, cancel)
    window.advanced_button.click()
    assert window.mosaic.value() == 1.0
    assert window.cache.text() == "ram"
    assert not window.deterministic.isChecked()

    def accept():
        window.optimizer.setText("adamw")
        window.cache.setText("false")
        window.mosaic.lineEdit().setText("0.25")
        window.mosaic.interpretText()
        window.accept_advanced_settings()

    QtCore.QTimer.singleShot(0, accept)
    window.advanced_button.click()
    assert window.mosaic.value() == 0.25
    assert window.optimizer.text() == "AdamW"
    assert window.cache.text() == "false"
    window.timer.stop()
    window.hide()
    window.deleteLater()


@pytest.mark.parametrize("label", [None, "2 0.5 0.5 0.5 0.5", "0 nan 0.5 0.5 0.5"])
def test_snapshot_rejects_missing_or_invalid_labels(tmp_path, label):
    source = dataset(tmp_path / "source", ["one", "two"])
    path = source / "labels/one.txt"
    if label is None:
        path.unlink()
    else:
        path.write_text(label)
    with pytest.raises(ValueError):
        prepare_training_dataset(source, tmp_path / "snapshot")


def test_prepared_yaml_keeps_split_and_rejects_overlap(tmp_path):
    import yaml
    source = dataset(tmp_path / "source", ["one", "two"])
    config = {"path": str(source), "train": "images", "val": "images", "names": ["helmet"]}
    path = source / "data.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="重复"):
        prepare_training_dataset(path, tmp_path / "snapshot")


def test_training_can_still_read_existing_txt_split_lists(tmp_path):
    source = dataset(tmp_path / "source", ["one", "two"])
    (source / "train.txt").write_text("images/one.jpg\n")
    (source / "val.txt").write_text("images/two.jpg\n")
    yaml_path = source / "legacy.yaml"
    yaml_path.write_text("train: train.txt\nval: val.txt\nnames: [helmet]\n")
    result = prepare_training_dataset(yaml_path, tmp_path / "snapshot")
    assert (result["train"], result["val"]) == (1, 1)


def test_training_callbacks_report_epoch_and_metrics():
    from autolabel.workbench.services.auto_training.ultralytics.worker import attach_training_events
    callbacks, events = {}, []
    model = SimpleNamespace(add_callback=lambda name, callback: callbacks.update({name: callback}))
    attach_training_events(model, lambda event, **data: events.append((event, data)))
    trainer = SimpleNamespace(epoch=2, epochs=10, train_loader=[1, 2], save_dir="runs/train",
                              metrics={"metrics/mAP50(B)": 0.6, "invalid": float("nan")}, tloss=None)
    callbacks["on_train_epoch_start"](trainer)
    callbacks["on_fit_epoch_end"](trainer)
    callbacks["on_fit_epoch_end"](trainer)
    assert events[0][1]["epoch"] == 3
    assert "on_train_batch_end" not in callbacks
    assert events[-1][1]["metrics"] == {"metrics/mAP50(B)": 0.6}
    assert [event for event, _ in events].count("training_metrics") == 1


def test_metric_chart_animates_updates_and_legend_filters_hover_values():
    from PyQt6 import QtCore, QtGui, QtWidgets

    from autolabel.training_chart import MetricChart

    chart = MetricChart("验证指标")
    chart.resize(440, 260)
    chart.show()
    QtWidgets.QApplication.processEvents()
    rows = [(1, {"metrics/mAP50-95(B)": 0.4, "metrics/recall(B)": 0.6, "train/box_loss": 1.2}),
            (5, {"metrics/mAP50-95(B)": 0.65, "metrics/recall(B)": 0.8})]
    chart.set_metrics(rows, animate=True, reset=True)
    assert chart.plot.animation.state() == QtCore.QAbstractAnimation.State.Running
    assert "65.00%" in chart.summary.text()
    assert set(chart.legend_buttons) == {"metrics/mAP50-95(B)", "metrics/recall(B)"}
    chart.plot.animation.setCurrentTime(chart.plot.animation.duration())
    plot = chart.plot
    position = plot.plot_rect().topRight() + QtCore.QPointF(-2, 12)
    move = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseMove, position, position, QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.MouseButton.NoButton, QtCore.Qt.KeyboardModifier.NoModifier,
    )
    QtWidgets.QApplication.sendEvent(plot, move)
    assert plot.hovered_epoch == 5
    chart.legend_buttons["metrics/mAP50-95(B)"].click()
    assert [curve.label for curve in plot.curves] == ["Recall"]
    assert "80.00%" in chart.summary.text()
    assert not chart.grab().isNull()
    chart.set_metrics([], reset=True)
    assert plot.hovered_epoch is None
    assert not chart.legend.isVisible()
    chart.close()
    chart.deleteLater()


def test_window_live_progress_and_close_hides(tmp_path, monkeypatch):
    from autolabel.training_window import TrainingWindow
    from autolabel.workbench.services.auto_training.ultralytics import worker
    window = TrainingWindow()
    window.record = {"args": {"model": "rtdetr-l.pt"}}
    window.on_event("training_progress", {"epoch": 3, "total_epochs": 10})
    assert window.epoch_progress.format() == "已完成 2 / 10 轮"
    assert window.epoch_progress.value() == 2
    assert window.loss_chart.rows == []
    assert window.metric_chart.rows == []
    window.record = None
    window.on_event("training_metrics", {"epoch": 3, "total_epochs": 10,
                                          "metrics": {"metrics/recall(B)": 0.7,
                                                      "train/box_loss": 0.5}})
    assert window.epoch_progress.value() == 3
    assert window.loss_chart.rows == window.metric_chart.rows == [
        (3, {"metrics/recall(B)": 0.7, "train/box_loss": 0.5})
    ]
    assert "Epoch 3/10  验证 all  R 0.7000" in window.log.toPlainText()
    window.record = {"directory": str(tmp_path)}
    monkeypatch.setattr(worker, "emit", lambda event, **data: window.on_event(event, data))
    stream = worker.EventLogStream("training_log")
    for row in ("train 1/99", "train 99/99", "\n", "val 1/17", "val 17/17", "\n"):
        stream.write(row if row == "\n" else "\r\033[K" + row)
        stream.flush()
    assert window.log.toPlainText().splitlines()[-2:] == ["train 99/99", "val 17/17"]
    assert (tmp_path / "training.log").read_text().splitlines() == ["train 99/99", "val 17/17"]
    window.family.setCurrentIndex(1)
    assert window.model.currentText() == "rtdetr-l.pt"
    window.show()
    window.close()
    assert not window.isVisible()
    window.timer.stop()
    window.deleteLater()


def test_integrated_training_entry_reuses_window(tmp_path):
    from PyQt6 import QtCore, QtWidgets
    from autolabel.workbench import config
    from autolabel.workbench.resources import resources
    from autolabel.desktop import AutoLabelMainWindow
    previous_output = tmp_path / "training-runs"
    QtCore.QSettings().setValue("training/output", str(previous_output))
    window = AutoLabelMainWindow(QtWidgets.QApplication.instance(), config=config.get_config())
    window.video_dock.config.output_dir = str(tmp_path / "dataset")
    window.training_action.trigger()
    training = window.training_window
    assert training.source.text() == str(tmp_path / "dataset")
    assert training.windowTitle() == "AutoLabelT · 训练中心"
    assert training.findChild(QtWidgets.QLabel, "trainingBrand").text() == "AutoLabelT"
    assert training.output.text() == str(previous_output)
    assert training.isVisible()
    training.close()
    window.training_action.trigger()
    assert window.training_window is training
    training.timer.stop()
    training.hide()
    window.close()
