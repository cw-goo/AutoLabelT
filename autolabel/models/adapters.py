"""Inference adapters return normalized boxes and stable prompt class IDs."""

import numpy as np
import os
import threading

os.environ["YOLO_AUTOINSTALL"] = "false"
_clip_load_lock = threading.Lock()


def parse_prompts(value):
    prompts = []
    for item in value.replace("\n", ",").split(","):
        label = item.strip()
        if label and label.casefold() not in {entry.casefold() for entry in prompts}:
            prompts.append(label)
    if not prompts:
        raise ValueError("至少输入一个检测类别，例如 person, phone")
    return prompts


def _device(device):
    if device != "auto":
        return device
    import torch
    return "0" if torch.cuda.is_available() else "cpu"


def _to_yolo(boxes, names, labels, image_width, image_height):
    label_ids = {name.casefold(): index for index, name in enumerate(labels)}
    result = []
    for box, score, name in zip(boxes["xyxy"], boxes["scores"], names):
        x1, y1, x2, y2 = map(float, box)
        x1, x2 = sorted((min(image_width, max(0.0, x1)),
                         min(image_width, max(0.0, x2))))
        y1, y2 = sorted((min(image_height, max(0.0, y1)),
                         min(image_height, max(0.0, y2))))
        label = str(name).strip().casefold()
        class_id = label_ids.get(label)
        if class_id is None and label:
            matches = [index for index, prompt in enumerate(labels)
                       if prompt.casefold() in label or label in prompt.casefold()]
            class_id = matches[0] if len(matches) == 1 else None
        if class_id is None or x2 <= x1 or y2 <= y1:
            continue
        result.append((class_id, (x1 + x2) / (2 * image_width),
                       (y1 + y2) / (2 * image_height),
                       (x2 - x1) / image_width, (y2 - y1) / image_height,
                       float(score)))
    return result


class YOLODetector:
    def __init__(self, path, prompts, device="auto", image_size=1280, confidence=0.35):
        from ultralytics import YOLO
        self.model = YOLO(str(path))
        self.prompts = prompts
        self.device = _device(device)
        self.image_size = image_size
        self.confidence = confidence

    def predict(self, frame):
        result = self.model.predict(source=frame, imgsz=self.image_size,
                                    conf=self.confidence, classes=[0],
                                    device=self.device, verbose=False)[0]
        if result.boxes is None:
            return []
        boxes = result.boxes.xywhn.cpu().numpy()
        return [(0, *map(float, box), float(score))
                for box, score in zip(boxes, result.boxes.conf.cpu().numpy())]


class CocoYOLODetector:
    def __init__(self, path, prompts, device="auto", image_size=1280, confidence=0.35):
        from ultralytics import YOLO
        self.model = YOLO(str(path))
        names = self.model.names
        self.class_map = {int(class_id): index for index, prompt in enumerate(prompts)
                          for class_id, name in names.items()
                          if str(name).casefold() == prompt.casefold()}
        if len(self.class_map) != len(prompts):
            available = {str(name).casefold() for name in names.values()}
            unknown = [prompt for prompt in prompts if prompt.casefold() not in available]
            raise ValueError(f"预训练模型不包含类别：{', '.join(unknown)}；请换用开放词汇模型")
        self.device = _device(device)
        self.image_size = image_size
        self.confidence = confidence

    def predict(self, frame):
        result = self.model.predict(
            source=frame, imgsz=self.image_size, conf=self.confidence,
            classes=list(self.class_map), device=self.device, verbose=False)[0]
        if result.boxes is None:
            return []
        boxes = result.boxes.xywhn.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        return [(self.class_map[int(category)], *map(float, box), float(score))
                for category, box, score in zip(classes, boxes, scores)]


class YOLOWorldDetector:
    def __init__(self, path, prompts, device="auto", image_size=1280, confidence=0.35):
        from ultralytics import YOLOWorld
        self.model = YOLOWorld(str(path))
        import clip
        original_load = clip.load
        text_encoder = os.fspath(os.path.join(os.path.dirname(path), "text_encoder"))

        def load_with_managed_cache(name, device=None, jit=False, download_root=None):
            return original_load(name, device=device, jit=jit, download_root=text_encoder)

        with _clip_load_lock:
            clip.load = load_with_managed_cache
            try:
                self.model.set_classes(prompts)
            finally:
                clip.load = original_load
        self.device = _device(device)
        self.image_size = image_size
        self.confidence = confidence

    def predict(self, frame):
        result = self.model.predict(source=frame, imgsz=self.image_size,
                                    conf=self.confidence, device=self.device,
                                    verbose=False)[0]
        if result.boxes is None:
            return []
        boxes = result.boxes.xywhn.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        return [(int(category), *map(float, box), float(score))
                for category, box, score in zip(classes, boxes, scores)]


class GroundingDinoDetector:
    def __init__(self, path, prompts, device="auto", image_size=1280, confidence=0.35):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.device = torch.device("cuda:0" if device in ("auto", "0") and torch.cuda.is_available()
                                   else "cpu" if device == "auto" else device)
        self.processor = AutoProcessor.from_pretrained(str(path), local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            str(path), local_files_only=True, use_safetensors=True).to(self.device).eval()
        self.prompts = prompts
        self.confidence = confidence
        self.torch = torch

    def predict(self, frame):
        import cv2
        from PIL import Image
        height, width = frame.shape[:2]
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        caption = ". ".join(self.prompts) + "."
        inputs = self.processor(images=image, text=caption, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        prediction = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=self.confidence,
            text_threshold=max(0.15, self.confidence - 0.1),
            target_sizes=[(height, width)])[0]
        names = list(prediction.get("text_labels", prediction.get("labels", [])))
        if names and isinstance(names[0], int):
            names = [self.prompts[index] if index < len(self.prompts) else ""
                     for index in names]
        return _to_yolo({"xyxy": prediction["boxes"].detach().cpu().numpy(),
                         "scores": prediction["scores"].detach().cpu().numpy()},
                        names, self.prompts, width, height)


class SAM3Detector:
    def __init__(self, path, prompts, device="auto", image_size=1280, confidence=0.35):
        from ultralytics.models.sam import SAM3SemanticPredictor
        self.predictor = SAM3SemanticPredictor(overrides={
            "model": str(path), "conf": confidence, "task": "segment",
            "mode": "predict", "imgsz": image_size, "device": _device(device),
            "save": False, "verbose": False})
        self.prompts = prompts

    def predict(self, frame):
        self.predictor.set_image(frame)
        predictions = []
        for class_id, prompt in enumerate(self.prompts):
            results = self.predictor(text=[prompt])
            for result in results:
                if result.boxes is None:
                    continue
                boxes = result.boxes.xywhn.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                predictions.extend((class_id, *map(float, box), float(score))
                                   for box, score in zip(boxes, scores))
        return predictions


class Florence2Detector:
    def __init__(self, path, prompts, device="auto", image_size=1280, confidence=0.35,
                 trust_remote_code=False):
        import torch
        if not trust_remote_code:
            raise ValueError("Florence-2 官方权重需要执行固定版本的模型代码；请先确认来源并勾选信任模型代码。")
        from transformers import AutoModelForCausalLM, AutoProcessor
        self.device = torch.device("cuda:0" if device in ("auto", "0") and torch.cuda.is_available()
                                   else "cpu" if device == "auto" else device)
        self.processor = AutoProcessor.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(path), local_files_only=True, use_safetensors=True,
            trust_remote_code=True, attn_implementation="eager").to(self.device).eval()
        self.prompts = prompts
        self.confidence = confidence
        self.torch = torch

    def predict(self, frame):
        import cv2
        from PIL import Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        task = "<OPEN_VOCABULARY_DETECTION>"
        prompt = task + ". ".join(self.prompts)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=256, use_cache=False)
        decoded = self.processor.batch_decode(generated, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(decoded, task=task, image_size=image.size)[task]
        boxes = np.asarray(parsed.get("bboxes", []), dtype=float)
        names = parsed.get("bboxes_labels", [])
        scores = np.asarray(parsed.get("scores", np.ones(len(boxes))), dtype=float)
        if len(boxes) == 0:
            return []
        return _to_yolo({"xyxy": boxes, "scores": scores}, names,
                        self.prompts, image.width, image.height)


def create_detector(model_id, path, prompt_text, device="auto", image_size=1280,
                    confidence=0.35, trust_remote_code=False):
    from .registry import get_model
    prompts = parse_prompts(prompt_text)
    spec = get_model(model_id)
    arguments = (path, prompts, device, image_size, confidence)
    if spec.backend == "ultralytics":
        return YOLODetector(*arguments)
    if spec.backend == "ultralytics_coco":
        if model_id.startswith("yolo26_"):
            from importlib.metadata import version
            from packaging.version import Version
            if Version(version("ultralytics")) < Version("8.4.0"):
                raise RuntimeError("YOLO26 需要 Ultralytics 8.4+；请在模型窗口确认安装兼容版本")
        return CocoYOLODetector(*arguments)
    if spec.backend == "ultralytics_sam3":
        from importlib.metadata import version
        from packaging.version import Version
        if Version(version("ultralytics")) < Version("8.4.0"):
            raise RuntimeError("SAM 3 需要 Ultralytics 8.4+；请在模型窗口确认安装兼容版本")
        return SAM3Detector(*arguments)
    if spec.backend == "ultralytics_world":
        return YOLOWorldDetector(*arguments)
    if spec.backend == "transformers_grounding_dino":
        return GroundingDinoDetector(*arguments)
    if spec.backend == "transformers_florence2":
        return Florence2Detector(*arguments, trust_remote_code=trust_remote_code)
    raise ValueError(f"没有可用的模型适配器：{model_id}")
