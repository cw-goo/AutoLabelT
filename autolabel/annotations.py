"""Read and write standard YOLO bounding-box annotations."""

from pathlib import Path
import math

from autolabel.workbench.views.labeling.annotation_io import atomic_write_text


def read_labels(path, class_count):
    path = Path(path)
    if not path.exists():
        return []
    boxes = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{number}: expected class and four coordinates")
        try:
            category = int(parts[0])
            center_x, center_y, width, height = map(float, parts[1:])
        except ValueError as error:
            raise ValueError(f"{path}:{number}: invalid value") from error
        if not 0 <= category < class_count:
            raise ValueError(f"{path}:{number}: unknown class ID")
        if not all(map(math.isfinite, (center_x, center_y, width, height))):
            raise ValueError(f"{path}:{number}: non-finite coordinate")
        left, top = center_x - width / 2, center_y - height / 2
        right, bottom = center_x + width / 2, center_y + height / 2
        if width <= 0 or height <= 0 or min(left, top) < -1e-6 or max(right, bottom) > 1 + 1e-6:
            raise ValueError(f"{path}:{number}: invalid box")
        boxes.append((category, max(0.0, left), max(0.0, top),
                      min(1.0, right), min(1.0, bottom)))
    return boxes


def write_labels(path, boxes):
    lines = []
    for category, left, top, right, bottom in boxes:
        if not 0 <= left < right <= 1 or not 0 <= top < bottom <= 1:
            raise ValueError("Invalid bounding box")
        lines.append(f"{category} {(left + right) / 2:.6f} {(top + bottom) / 2:.6f} "
                     f"{right - left:.6f} {bottom - top:.6f}\n")
    path = Path(path)
    atomic_write_text(path, "".join(lines))
