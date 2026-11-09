"""Bridge standard YOLO datasets with AutoLabel's native JSON format."""

import json
from pathlib import Path
import tempfile

from .annotations import read_labels
from .validate import validate_dataset
from autolabel.workbench.views.labeling.annotation_io import AnnotationTransaction
from autolabel.workbench.views.labeling.annotation_io import atomic_write_json


def prepare_review_dataset(root, cancelled=None):
    from autolabel.workbench.views.labeling.label_file import LabelFile

    root = Path(root)
    summary = validate_dataset(root)
    created = 0
    for image in sorted((root / "images").glob("*.jpg")):
        if cancelled and cancelled():
            raise InterruptedError("已取消复核准备")
        annotation = image.with_suffix(".json")
        if annotation.exists():
            LabelFile(str(annotation), image_path=str(image))
            continue
        prepare_review_image(root, image)
        created += 1
    return {**summary, "json_created": created}


def prepare_review_image(root, image):
    from autolabel.workbench.views.labeling.label_converter import LabelConverter
    from autolabel.workbench.views.labeling.label_file import LabelFile

    root = Path(root)
    image = Path(image)
    annotation = image.with_suffix(".json")
    if annotation.exists():
        LabelFile(str(annotation), image_path=str(image))
        return annotation
    label = root / "labels" / f"{image.stem}.txt"
    converter = LabelConverter(classes_file=str(root / "classes.txt"))
    read_labels(label, len(converter.classes))
    with tempfile.NamedTemporaryFile(dir=image.parent, suffix=".json", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        converter.yolo_to_custom(str(label), str(temporary_path), str(image), "hbb")
        scores_path = root / "metadata" / f"{image.stem}.scores.json"
        if scores_path.is_file():
            scores = json.loads(scores_path.read_text(encoding="utf-8"))["scores"]
            data = json.loads(temporary_path.read_text(encoding="utf-8"))
            if len(scores) != len(data["shapes"]):
                raise ValueError(f"检测分数与标注数量不匹配：{image.name}")
            for shape, score in zip(data["shapes"], scores):
                shape["score"] = score
            atomic_write_json(temporary_path, data, ensure_ascii=False)
        LabelFile(str(temporary_path), image_path=str(image))
        temporary_path.replace(annotation)
    finally:
        temporary_path.unlink(missing_ok=True)
    return annotation


def sync_review_to_yolo(root, cancelled=None):
    from autolabel.workbench.views.labeling.label_converter import LabelConverter

    root = Path(root)
    converter = LabelConverter(classes_file=str(root / "classes.txt"))
    classes = converter.classes
    images = sorted((root / "images").glob("*.jpg"))
    if not images or not classes:
        raise ValueError("缺少图片或类别映射")
    for image in images:
        if cancelled and cancelled():
            raise InterruptedError("已取消标签同步")
        annotation = image.with_suffix(".json")
        if not annotation.is_file():
            raise ValueError(f"缺少复核标注：{annotation.name}")
        data = json.loads(annotation.read_text(encoding="utf-8"))
        if data.get("imagePath") != image.name:
            raise ValueError(f"图片路径不匹配：{annotation.name}")
        for shape in data.get("shapes", []):
            if shape.get("shape_type") != "rectangle" or len(shape.get("points", [])) != 4:
                raise ValueError(f"仅支持矩形框：{annotation.name}")
            if shape.get("label") not in classes:
                raise ValueError(f"类别不在 classes.txt 中：{shape.get('label')}")
    labels_dir = root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    with AnnotationTransaction(labels_dir / f"{image.stem}.txt" for image in images) as transaction:
        for image in images:
            if cancelled and cancelled():
                raise InterruptedError("已取消标签同步")
            converted = transaction.stage_path(labels_dir / f"{image.stem}.txt")
            converter.custom_to_yolo(str(image.with_suffix(".json")),
                                     str(converted), "hbb")
            read_labels(converted, len(classes))
        if cancelled and cancelled():
            raise InterruptedError("已取消标签同步")
        transaction.commit()
    return validate_dataset(root)
