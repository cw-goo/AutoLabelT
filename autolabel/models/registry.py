"""Pinned catalog of detection models supported by AutoLabel."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    title: str
    task: str
    source: str
    license: str
    source_url: str
    revision: str
    backend: str
    description: str
    weight_url: str = ""
    weight_name: str = ""


MODEL_REGISTRY = {
    "local_yolo": ModelSpec(
        "local_yolo", "自定义 YOLO 权重", "封闭词表目标检测", "用户本地文件", "随权重来源",
        "", "local", "ultralytics", "加载本地 .pt 权重。默认只检测 person 类别。"),
    "yolo_world": ModelSpec(
        "yolo_world", "YOLO-World v2 S", "文本提示开放词汇检测", "Ultralytics / YOLO-World", "AGPL-3.0",
        "https://github.com/ultralytics/ultralytics", "v8.4.0", "ultralytics_world",
        "速度快，可输入多个类别词，如 person、phone。", 
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s-worldv2.pt"),
    "grounding_dino": ModelSpec(
        "grounding_dino", "Grounding DINO Tiny", "文本提示开放词汇检测", "IDEA-Research", "Apache-2.0",
        "https://huggingface.co/IDEA-Research/grounding-dino-tiny", "a2bb814dd30d776dcf7e30523b00659f4f141c71",
        "transformers_grounding_dino", "适合按自然语言类别查找目标，模型约 689 MB。"),
    "florence2": ModelSpec(
        "florence2", "Florence-2 Base", "视觉语言 / 开放词汇检测", "Microsoft", "MIT",
        "https://huggingface.co/microsoft/Florence-2-base", "5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac",
        "transformers_florence2", "可执行开放词汇检测等多种视觉任务，首次加载需确认模型代码来源。"),
    "grounding_dino_base": ModelSpec(
        "grounding_dino_base", "Grounding DINO Base", "文本提示开放词汇检测",
        "IDEA-Research", "Apache-2.0",
        "https://huggingface.co/IDEA-Research/grounding-dino-base",
        "12bdfa3120f3e7ec7b434d90674b3396eccf88eb",
        "transformers_grounding_dino", "较大的 DINO 开放词汇模型，权重约 933 MB。"),
    "sam3": ModelSpec(
        "sam3", "SAM 3", "文本概念分割转检测框", "Meta", "Meta SAM 3 License",
        "https://huggingface.co/facebook/sam3",
        "3c879f39826c281e95690f02c7821c4de09afae7",
        "ultralytics_sam3", "需先获 Meta 权重访问许可并登录 Hugging Face；输出分割框供复核。",
        weight_name="sam3.pt"),
    "sam3_1": ModelSpec(
        "sam3_1", "SAM 3.1", "文本概念分割转检测框", "Meta", "Meta SAM 3 License",
        "https://huggingface.co/facebook/sam3.1",
        "daa63191845a41281374e725f4c9e51c7a824460",
        "ultralytics_sam3", "需先获 Meta 权重访问许可并登录 Hugging Face；输出分割框供复核。",
        weight_name="sam3.1_multiplex.pt"),
}

for family, release, label in (("yolov8", "v8.2.0", "YOLOv8"),
                               ("yolo11", "v8.3.0", "YOLO11"),
                               ("yolo26", "v8.4.0", "YOLO26")):
    for size in "nsmlx":
        weight_name = f"{family}{size}.pt"
        MODEL_REGISTRY[f"{family}_{size}"] = ModelSpec(
            f"{family}_{size}", f"{label} {size.upper()}", "COCO 预训练目标检测",
            "Ultralytics", "AGPL-3.0", f"https://docs.ultralytics.com/models/{family}",
            release, "ultralytics_coco",
            "COCO 预训练权重；仅能选择其已有类别，不支持任意文本概念。",
            f"https://github.com/ultralytics/assets/releases/download/{release}/{weight_name}",
            weight_name)


def get_model(model_id):
    try:
        return MODEL_REGISTRY[model_id]
    except KeyError as error:
        raise ValueError(f"不支持的模型：{model_id}") from error
