"""Failure injection for reusable annotation I/O, without touching user datasets."""

import errno
import json
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import pytest

from autolabel.workbench.views.labeling.annotation_io import (
    AnnotationTransaction, atomic_write_json, describe_file_error,
)


def test_atomic_save_preserves_original_on_serialization_or_replace_failure(tmp_path, monkeypatch):
    from autolabel.workbench.views.labeling import annotation_io

    target = tmp_path / "frame.json"
    target.write_text("original")
    with pytest.raises(TypeError):
        atomic_write_json(target, {"invalid": object()})
    assert target.read_text() == "original"

    def denied(*_):
        raise PermissionError(errno.EACCES, "denied", str(target))

    monkeypatch.setattr(annotation_io.os, "replace", denied)
    with pytest.raises(PermissionError):
        atomic_write_json(target, {"shapes": []})
    assert target.read_text() == "original"
    assert list(tmp_path.iterdir()) == [target]


def test_preflight_checks_every_output_before_moving_any_file(tmp_path, monkeypatch):
    first, second = tmp_path / "one.json", tmp_path / "two.json"
    first.write_text("first")
    second.write_text("second")
    original_open = Path.open

    def denied(path, *args, **kwargs):
        if path == second and args == ("r+b",):
            raise PermissionError(errno.EACCES, "denied", str(second))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied)
    with pytest.raises(PermissionError):
        with AnnotationTransaction([first, second]) as transaction:
            transaction.clear()
    assert first.read_text() == "first" and second.read_text() == "second"
    assert len(list(tmp_path.iterdir())) == 2


def test_cancel_fresh_inference_restores_all_originals_and_removes_staged_outputs(tmp_path):
    first, second = tmp_path / "one.json", tmp_path / "two.json"
    first.write_text("old")
    with AnnotationTransaction([first, second]) as transaction:
        transaction.clear()
        assert not first.exists()
        atomic_write_json(transaction.stage_path(first), {"new": 1})
        atomic_write_json(transaction.stage_path(second), {"new": 2})
        # Leaving without committing represents cancellation or inference failure.
    assert first.read_text() == "old"
    assert not second.exists()
    assert list(tmp_path.iterdir()) == [first]


def test_failed_second_commit_restores_originals_and_removes_new_files(tmp_path, monkeypatch):
    from autolabel.workbench.views.labeling import annotation_io

    first, second, new = [tmp_path / name for name in ("one.json", "two.json", "new.json")]
    first.write_text("first")
    second.write_text("second")
    original_replace = annotation_io.os.replace
    with pytest.raises(PermissionError):
        with AnnotationTransaction([first, second, new]) as transaction:
            for path in (first, new, second):
                atomic_write_json(transaction.stage_path(path), {"new": True})
            failed_stage = transaction.staged[second]

            def fail_second(source, destination):
                if Path(source) == failed_stage:
                    raise PermissionError(errno.EACCES, "locked", str(second))
                return original_replace(source, destination)

            monkeypatch.setattr(annotation_io.os, "replace", fail_second)
            transaction.commit()
    assert first.read_text() == "first" and second.read_text() == "second"
    assert not new.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["one.json", "two.json"]


def test_successful_fresh_inference_commits_and_removes_sidecar_backup(tmp_path):
    sidecar, output = tmp_path / "sidecar.json", tmp_path / "output.json"
    sidecar.write_text("old-sidecar")
    output.write_text("old-output")
    with AnnotationTransaction([sidecar, output]) as transaction:
        transaction.clear()
        atomic_write_json(transaction.stage_path(output), {"label": "hat2"})
        transaction.commit()
    assert not sidecar.exists()
    assert json.loads(output.read_text()) == {"label": "hat2"}
    assert list(tmp_path.iterdir()) == [output]


def test_rollback_failure_preserves_recoverable_backup(tmp_path, monkeypatch):
    from autolabel.workbench.views.labeling import annotation_io

    target = tmp_path / "frame.json"
    target.write_text("original")
    transaction = AnnotationTransaction([target])
    transaction.preflight()
    transaction.clear()
    backup = transaction.backups[target]

    def denied(*_):
        raise PermissionError("still locked")

    monkeypatch.setattr(annotation_io.os, "replace", denied)
    with pytest.raises(OSError, match=str(backup).replace("\\", "\\\\")):
        transaction.rollback()
    assert backup.read_text() == "original"


@pytest.mark.parametrize("failure,expected", [
    (PermissionError(errno.EACCES, "denied", "frame.txt"), "安全权限"),
    (OSError(errno.ENOSPC, "full", "frame.json"), "磁盘空间不足"),
])
def test_file_errors_include_path_and_actionable_reason(failure, expected):
    wrapped = ValueError(f"annotation failed: {failure}")
    wrapped.__cause__ = failure
    message = describe_file_error(wrapped)
    assert expected in message and "frame." in message


@pytest.mark.parametrize("result", ["raises", None])
def test_model_manager_does_not_convert_prediction_failure_to_empty_annotations(result):
    from autolabel.workbench.services.auto_labeling.model_manager import ModelManager

    def predict(*_):
        if result == "raises":
            raise RuntimeError("inference failed")
        return None

    state = SimpleNamespace(
        loaded_model_config_lock=Lock(),
        loaded_model_config={"model": SimpleNamespace(predict_shapes=predict)},
        new_model_status=SimpleNamespace(emit=lambda *_: None),
        prediction_finished=SimpleNamespace(emit=lambda *_: None),
        tr=lambda text: text,
    )
    with pytest.raises(RuntimeError):
        ModelManager.predict_shapes(state, None, "frame.jpg", batch=True)


def import_dataset(tmp_path):
    import cv2
    import numpy as np
    from autolabel.workbench.views.labeling.label_converter import LabelConverter

    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for name in ("one", "two"):
        cv2.imwrite(str(images / f"{name}.jpg"), np.zeros((40, 80, 3), dtype=np.uint8))
        (labels / f"{name}.txt").write_text("1 0.5 0.5 0.5 0.5\n")
        (images / f"{name}.json").write_text('{"shapes": [], "old": true}')
    classes = tmp_path / "classes.txt"
    classes.write_text("hat\nhat2\n")
    return images, labels, LabelConverter(str(classes))


@pytest.mark.parametrize("failure", ["class", "permission", "cancel"])
def test_yolo_import_failure_or_cancel_never_leaves_mixed_results(tmp_path, monkeypatch, failure):
    from autolabel.workbench.views.labeling.utils.upload import (
        _import_yolo_annotations, AnnotationImportCancelled,
    )

    images, labels, converter = import_dataset(tmp_path)
    originals = {path: path.read_bytes() for path in images.glob("*.json")}
    processed = []
    progress = SimpleNamespace(setMaximum=lambda _: None,
                               setValue=processed.append,
                               wasCanceled=lambda: failure == "cancel" and bool(processed))
    expected = AnnotationImportCancelled
    if failure == "class":
        (labels / "two.txt").write_text("2 0.5 0.5 0.5 0.5\n")
        expected = ValueError
    elif failure == "permission":
        original_read = converter.read_lines

        def denied(path, *args):
            if Path(path).stem == "two":
                raise PermissionError(errno.EACCES, "denied", str(path))
            return original_read(path, *args)

        monkeypatch.setattr(converter, "read_lines", denied)
        expected = PermissionError
    with pytest.raises(expected):
        _import_yolo_annotations(converter, "hbb", str(images), str(labels), str(images),
                                 False, progress)
    assert {path: path.read_bytes() for path in images.glob("*.json")} == originals
    assert not list(images.glob(".autolabel_*"))


@pytest.mark.parametrize("mode,columns", [
    ("hbb", "0.5 0.5 0.5 0.5"),
    ("obb", "0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5"),
    ("seg", "0.1 0.1 0.5 0.1 0.5 0.5"),
    ("pose", "0.5 0.5 0.5 0.5 0.5 0.5 2"),
])
@pytest.mark.parametrize("class_id", [-1, 2])
def test_yolo_class_ids_cannot_wrap_or_create_implicit_classes(tmp_path, mode, columns, class_id):
    from autolabel.workbench.views.labeling.label_converter import LabelConverter

    source = tmp_path / "frame.txt"
    source.write_text(f"{class_id} {columns}\n")
    converter = LabelConverter()
    converter.classes = ["hat", "hat2"]
    converter.pose_classes = {"hat": ["point"], "hat2": ["point"]}
    converter.has_visible = True
    with pytest.raises(ValueError, match=r"frame.txt:1:.*class ID"):
        list(converter.iter_yolo_rows(source, mode))


def test_yolo_import_tolerates_bom_whitespace_and_blank_lines(tmp_path):
    from autolabel.workbench.views.labeling.label_converter import LabelConverter

    source = tmp_path / "frame.txt"
    source.write_text("\n1\t 0.5   0.5  0.5  0.5\n\n", encoding="utf-8-sig")
    converter = LabelConverter()
    converter.classes = ["hat", "hat2"]
    assert [label for _, label in converter.iter_yolo_rows(source, "hbb")] == ["hat2"]


@pytest.mark.parametrize("result", [None, "unknown", "negative"])
def test_fresh_batch_distinguishes_failure_unknown_class_and_valid_negative_sample(tmp_path, result):
    from autolabel.workbench.services.auto_labeling.types import AutoLabelingResult
    from autolabel.workbench.views.labeling.utils.batch import save_auto_labeling_result

    images, _, _ = import_dataset(tmp_path)
    image = images / "one.jpg"
    target = image.with_suffix(".json")
    before = target.read_bytes()
    output = None if result is None else AutoLabelingResult([], replace=True)
    if result == "unknown":
        output.shapes = [SimpleNamespace(to_dict=lambda: {"label": "third"})]
    state = SimpleNamespace(output_dir=None, _config={"store_data": False},
        auto_labeling_widget=SimpleNamespace(model_manager=SimpleNamespace(
            loaded_model_config={"type": "autolabel_local_yolo", "classes": ["hat", "hat2"]})))
    if result == "negative":
        with AnnotationTransaction([target]) as transaction:
            transaction.clear()
            state._batch_transaction = transaction
            save_auto_labeling_result(state, str(image), output)
            transaction.commit()
        assert json.loads(target.read_text())["shapes"] == []
    else:
        with pytest.raises((RuntimeError, ValueError)):
            with AnnotationTransaction([target]) as transaction:
                transaction.clear()
                state._batch_transaction = transaction
                save_auto_labeling_result(state, str(image), output)
        assert target.read_bytes() == before
