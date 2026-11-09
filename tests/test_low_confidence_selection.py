"""Opening/navigating images focuses the least confident box without editing."""

from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("scores,expected", [
    ([0.9, 0.2, 0.7], 1),
    ([0.0, 0.2], 0),
    ([0.4, 0.4, 0.9], 0),
    ([None, "0.3", 0.8], 1),
    ([None, float("nan"), float("inf"), -0.1, 1.1, "invalid", True], None),
    ([], None),
])
def test_lowest_confidence_box_selection_ignores_missing_or_invalid_scores(scores, expected):
    from autolabel.workbench.views.labeling.label_widget import LabelingWidget

    shapes = [SimpleNamespace(shape_type="rectangle", score=score) for score in scores]
    selections = []
    state = SimpleNamespace(canvas=SimpleNamespace(
        shapes=shapes, is_visible=lambda _: True, select_shapes=selections.append))
    selected = LabelingWidget.select_lowest_confidence_box(state)
    if expected is None:
        assert selected is None and selections == [[]]
    else:
        assert selected is shapes[expected] and selections == [[shapes[expected]]]


def test_default_box_selection_respects_visibility_and_ignores_non_boxes():
    from autolabel.workbench.views.labeling.label_widget import LabelingWidget

    hidden = SimpleNamespace(shape_type="rectangle", score=0.01)
    point = SimpleNamespace(shape_type="point", score=0.02)
    rotated = SimpleNamespace(shape_type="rotation", score=0.3)
    rectangle = SimpleNamespace(shape_type="rectangle", score=0.9)
    selections = []
    state = SimpleNamespace(canvas=SimpleNamespace(
        shapes=[hidden, point, rotated, rectangle],
        is_visible=lambda shape: shape is not hidden, select_shapes=selections.append))
    assert LabelingWidget.select_lowest_confidence_box(state) is rotated
    assert selections == [[rotated]]


def test_image_navigation_selects_lowest_score_in_canvas_and_object_list(tmp_path, monkeypatch):
    import cv2
    import numpy as np
    from PyQt6 import QtCore, QtWidgets
    from autolabel.workbench import config as settings
    from autolabel.workbench.resources import resources
    from autolabel.workbench.views.labeling.label_file import LabelFile
    from autolabel.workbench.views.labeling.pointcloud.icons import _svg_renderer
    from autolabel.desktop import AutoLabelMainWindow

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    # Earlier Qt tests may have disposed their QApplication and its SVG renderers.
    _svg_renderer.cache_clear()
    settings.current_config_file = str(tmp_path / ".autolabelrc")
    config = settings.get_config()
    config.update(auto_save=False, keep_prev=False)
    window = AutoLabelMainWindow(app, config=config)
    temporary_settings = QtCore.QSettings(str(tmp_path / "window.ini"), QtCore.QSettings.Format.IniFormat)
    window.settings = temporary_settings
    view = window.labeling_widget.view
    view.settings = temporary_settings
    view.output_dir = None
    images = tmp_path / "images"
    images.mkdir()
    scores_by_image = ([0.9, 0.15, 0.6], [0.1, 0.8], [None], [])
    originals = {}
    for index, scores in enumerate(scores_by_image):
        image = images / f"frame_{index}.jpg"
        cv2.imwrite(str(image), np.zeros((100, 200, 3), dtype=np.uint8))
        shapes = [{
            "label": "hat", "score": score, "shape_type": "rectangle",
            "points": [[10 + row * 50, 10], [40 + row * 50, 10],
                       [40 + row * 50, 50], [10 + row * 50, 50]],
            "flags": {}, "group_id": 1,
        } for row, score in enumerate(scores)]
        annotation = image.with_suffix(".json")
        LabelFile().save(filename=str(annotation), shapes=shapes, image_path=image.name,
                         image_height=100, image_width=200)
        originals[annotation] = annotation.read_bytes()
    try:
        view.import_image_folder(str(images))
        assert view.filename == str(images / "frame_0.jpg")
        assert view.canvas.selected_shapes == [view.canvas.shapes[1]]
        selected_item = view.label_list.find_item_by_shape(view.canvas.shapes[1])
        assert view.label_list.selected_items() == [selected_item]
        assert view.actions.delete.isEnabled() and not view.dirty
        # A manual choice on the current image must stay selected until navigation.
        view.canvas.select_shapes([view.canvas.shapes[0]])
        app.processEvents()
        assert view.canvas.selected_shapes == [view.canvas.shapes[0]]
        view.open_next_image()
        assert view.filename == str(images / "frame_1.jpg")
        assert view.canvas.selected_shapes == [view.canvas.shapes[0]]
        assert view.label_list.selected_items()[0].shape().score == 0.1
        view.open_prev_image()
        assert view.canvas.selected_shapes == [view.canvas.shapes[1]]
        # The file-list jump uses the same default selection as next/previous.
        view.file_list_widget.setCurrentRow(1)
        assert view.canvas.selected_shapes[0].score == 0.1
        view.open_next_image()
        assert view.canvas.selected_shapes == []
        view.open_next_image()
        assert view.canvas.shapes == [] and view.canvas.selected_shapes == []
        assert not view.dirty
        assert all(path.read_bytes() == original for path, original in originals.items())
    finally:
        view.set_clean()
        window.close()
        app.processEvents()
