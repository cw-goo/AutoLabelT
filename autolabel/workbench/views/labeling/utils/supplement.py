"""Match supplementary detections against saved image annotations."""

import math
from pathlib import Path

import numpy as np

from autolabel.workbench.services.auto_labeling.utils.box import box_iou


def _bounds(shape):
    if shape.get("shape_type") not in ("rectangle", "rotation", "polygon"):
        return None
    points = shape.get("points")
    if not points or len(points) < 2:
        return None
    xs, ys = zip(*points)
    xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
    if not all(math.isfinite(coord) for coord in (xmin, ymin, xmax, ymax)):
        return None
    if xmin >= xmax or ymin >= ymax:
        return None
    return (xmin, ymin, xmax, ymax)


def _exact_geometry(shape):
    return (shape.get("shape_type"), shape.get("label"),
            tuple(map(tuple, shape.get("points") or ())))


def supplementary_shapes(existing, predictions, threshold):
    """Keep the first annotation when two object bounds overlap sufficiently."""
    boxes, geometry = [], set()
    for shape in existing:
        bounds = _bounds(shape)
        if bounds is None:
            geometry.add(_exact_geometry(shape))
        else:
            boxes.append(bounds)
    added = []
    for shape in predictions:
        bounds = _bounds(shape)
        if bounds is not None:
            if boxes and (box_iou(np.asarray([bounds]), np.asarray(boxes)) >= threshold).any():
                continue
            boxes.append(bounds)
        else:
            key = _exact_geometry(shape)
            if key in geometry:
                continue
            geometry.add(key)
        added.append(shape)
    return added


def yolo_label_path(image_file):
    """Find a conventional YOLO label alongside a mirrored images directory."""
    image = Path(image_file)
    for images in image.parents:
        if images.name.casefold() == "images":
            return (images.parent / "labels" / image.relative_to(images)).with_suffix(".txt"), images.parent
    return None, None


def read_yolo_baseline(image_file, label, root, watch):
    from autolabel.workbench.views.labeling.label_converter import LabelConverter

    classes = root / "classes.txt"
    if not classes.is_file():
        raise ValueError(f"发现 YOLO 标签 {label.name}。请先通过 YOLO 标注导入加载类别表，"
                         "或在数据集目录放入 classes.txt 后再运行漏标补充。")
    watch(classes)
    converter = LabelConverter(classes_file=str(classes))
    return converter.read_yolo_annotation(str(label), str(image_file), "hbb")
