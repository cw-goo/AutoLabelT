"""Bridge standard YOLO datasets with AutoLabel's native JSON format."""

import json
from pathlib import Path
import tempfile

from .annotations import IMAGE_SUFFIXES, read_labels
from .validate import validate_dataset
from autolabelt.workbench.views.labeling.annotation_io import AnnotationTransaction
from autolabelt.workbench.views.labeling.annotation_io import atomic_write_json
from autolabelt.workbench.views.labeling.label_mapping import read_class_names


def review_sync_location(image_path, output_dir):
    from autolabelt.workbench.views.labeling.utils.supplement import yolo_label_path

    if image_path:
        image = Path(image_path).absolute()
        _, root = yolo_label_path(image)
        if root is None:
            raise ValueError(f"无法定位当前图片的 YOLO 标签。请从数据集的 images 目录打开图片：{image}")
        return image, root
    if output_dir:
        return None, Path(output_dir).absolute()
    raise ValueError("请先选择要同步标签的数据集输出目录")


def prepare_review_dataset(root, cancelled=None):
    from autolabelt.workbench.views.labeling.label_file import LabelFile

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
    from autolabelt.workbench.views.labeling.label_converter import LabelConverter
    from autolabelt.workbench.views.labeling.label_file import LabelFile

    root = Path(root)
    image = Path(image)
    annotation = image.with_suffix(".json")
    if annotation.exists():
        LabelFile(str(annotation), image_path=str(image))
        return annotation
    label = root / "labels" / f"{image.stem}.txt"
    converter = LabelConverter(classes_file=str(root / "classes.txt"))
    read_labels(label, len(converter.classes), allow_missing=False)
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


def sync_review_to_yolo(root, cancelled=None, *, classes_file=None, active_annotation=None):
    from PIL import Image
    from autolabelt.workbench.views.labeling.label_converter import LabelConverter

    root = Path(root).absolute()
    images_dir = root / "images"
    labels_dir = root / "labels"
    dataset_classes = root / "classes.txt"
    classes_file = Path(classes_file) if classes_file is not None else dataset_classes
    images = sorted(image for image in images_dir.rglob("*")
                    if image.suffix.lower() in IMAGE_SUFFIXES and image.is_file())
    if not images:
        raise ValueError(f"没有可同步的图片：{images_dir}")
    active_image, active_json = (map(Path, active_annotation) if active_annotation
                                 else (None, None))
    if active_image is not None:
        active_image = active_image.absolute()
        if active_image not in images:
            raise ValueError(f"当前图片不在可同步的数据集中：{active_image}")
        if not active_json.is_file():
            raise ValueError(f"当前图片的复核标注无法读取：{active_json}")
    pairs = []
    targets = set()
    for image in images:
        annotation = active_json if image == active_image else image.with_suffix(".json")
        target = (labels_dir / image.relative_to(images_dir)).with_suffix(".txt")
        if target in targets:
            raise ValueError(f"多张同名图片对应同一个 YOLO 标签：{target}")
        targets.add(target)
        if annotation.is_file():
            pairs.append((image, annotation, target))
    if not pairs:
        raise ValueError(f"没有找到已保存的复核 JSON：{images_dir}")
    outputs = [target for _, _, target in pairs]
    for directory in {target.parent for target in outputs}:
        directory.mkdir(parents=True, exist_ok=True)
    install_classes = not dataset_classes.exists()
    if install_classes:
        outputs.append(dataset_classes)
    with AnnotationTransaction(outputs) as transaction:
        transaction.watch(classes_file)
        transaction.watch(dataset_classes)
        converter = LabelConverter(classes_file=str(classes_file))
        classes = converter.classes
        if dataset_classes.exists():
            if read_class_names(dataset_classes) != classes:
                raise ValueError(f"所选类别及其顺序须与数据集类别表一致：{dataset_classes}")
        elif install_classes:
            transaction.stage_path(dataset_classes).write_text(
                "\n".join(classes) + "\n", encoding="utf-8"
            )
        else:
            raise OSError(f"数据集类别表已移除，请重试：{dataset_classes}")
        for image, annotation, target in pairs:
            if cancelled and cancelled():
                raise InterruptedError("已取消标签同步")
            transaction.watch(image)
            transaction.watch(annotation)
            transaction.watch(target)
            with Image.open(image) as opened:
                size = opened.size
                opened.verify()
            data = converter.read_json(str(annotation))
            image_path = data.get("imagePath")
            if (not isinstance(image_path, str)
                    or (annotation.parent / image_path).resolve() != image.resolve()):
                raise ValueError(f"复核 JSON 所指向的图片与当前图片不一致：{annotation} → {image}")
            if (data.get("imageWidth"), data.get("imageHeight")) != size:
                raise ValueError(f"复核 JSON 记录的图片尺寸不匹配：{annotation}")
            for shape in data.get("shapes", []):
                if shape.get("shape_type") != "rectangle" or len(shape.get("points", [])) not in (2, 4):
                    raise ValueError(f"仅支持矩形框：{annotation}")
                if shape.get("label") not in classes:
                    raise ValueError(f"所选类别文件中没有类别：{shape.get('label')}（{annotation}）")
            if cancelled and cancelled():
                raise InterruptedError("已取消标签同步")
            read_labels(target, len(classes))
            converted = transaction.stage_path(target)
            converter.custom_to_yolo(str(annotation), str(converted), "hbb", annotation=data)
            read_labels(converted, len(classes), allow_missing=False)
        if cancelled and cancelled():
            raise InterruptedError("已取消标签同步")
        transaction.commit()
    return {"images": len(images), "labels": len(pairs), "classes": classes,
            "labels_dir": str(labels_dir), "skipped": len(images) - len(pairs)}
