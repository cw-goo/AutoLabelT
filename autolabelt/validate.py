"""Validate standard YOLO output before handoff or export."""

from pathlib import Path

from .annotations import read_labels
from autolabelt.workbench.views.labeling.label_mapping import read_class_names


def validate_dataset(root):
    from PIL import Image

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
            read_labels(labels[stem], len(classes), allow_missing=False)
        except (OSError, UnicodeError) as error:
            raise ValueError(f"无法读取标签 {labels[stem]}: {error}") from error
    return {"images": len(images), "labels": len(labels), "classes": classes}
