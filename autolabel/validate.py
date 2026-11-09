"""Validate standard YOLO output before handoff or export."""

from pathlib import Path
import math

from autolabel.workbench.views.labeling.label_mapping import read_class_names


def validate_dataset(root):
    from PIL import Image

    coordinate_tolerance = 1e-6
    root = Path(root)
    classes_file = root / "classes.txt"
    if not classes_file.is_file():
        raise ValueError("缺少 classes.txt")
    classes = read_class_names(classes_file)
    images = {file.stem: file for file in (root / "images").glob("*.jpg")}
    labels = {file.stem: file for file in (root / "labels").glob("*.txt")}
    if images.keys() != labels.keys():
        raise ValueError(f"图片与标签不匹配: 图片 {len(images)}，标签 {len(labels)}")
    for stem, image in images.items():
        try:
            with Image.open(image) as opened:
                opened.verify()
        except Exception as error:
            raise ValueError(f"无法读取图片 {image}: {error}") from error
        try:
            lines = labels[stem].read_text(encoding="utf-8-sig").splitlines()
        except Exception as error:
            raise ValueError(f"无法读取标签 {labels[stem]}: {error}") from error
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"{stem}:{number} 应为类别加 4 个坐标")
            try:
                category = int(parts[0])
                center_x, center_y, width, height = map(float, parts[1:])
            except ValueError as error:
                raise ValueError(f"{stem}:{number} 包含非法数值") from error
            if not 0 <= category < len(classes):
                raise ValueError(f"{stem}:{number} 类别 ID 越界")
            if (not all(math.isfinite(value) for value in
                        (center_x, center_y, width, height)) or
                    width <= 0 or height <= 0 or
                    center_x - width / 2 < -coordinate_tolerance or
                    center_x + width / 2 > 1 + coordinate_tolerance or
                    center_y - height / 2 < -coordinate_tolerance or
                    center_y + height / 2 > 1 + coordinate_tolerance):
                raise ValueError(f"{stem}:{number} 坐标越界或宽高非法")
    return {"images": len(images), "labels": len(labels), "classes": classes}
