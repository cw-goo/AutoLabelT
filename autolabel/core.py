"""Qt-independent implementation of the verified databuild.py rules."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import re
import time
import uuid

from autolabel.workbench.views.labeling.annotation_io import atomic_write_text
from autolabel.workbench.views.labeling.label_mapping import read_class_names


DYNAMIC_JUMP_KEYS = ("idle_jump", "step_single", "step_small_group", "step_large_group")


@dataclass
class ProjectConfig:
    video_path: str = ""
    output_dir: str = ""
    model_path: str = "staffv2.pt"
    model_id: str = "local_yolo"
    prompt_labels: str = "person"
    trust_remote_code: bool = False
    confidence: float = 0.35
    image_size: int = 1280
    device: str = "auto"
    min_pixel_height: int = 100
    idle_jump: int = 500
    step_single: int = 300
    step_small_group: int = 160
    step_large_group: int = 100
    companion_min_height: int = 40
    small_target_ratio: float = 0.60
    reject_single_below: int = 110
    video_paths: list[str] = field(default_factory=list)
    target_count: int = 0
    min_sample_seconds: float = 1.0
    min_sample_frames: int | None = None

    def validate(self, *, require_output=True):
        if not self.video_path or not self.model_path or (require_output and not self.output_dir):
            raise ValueError("视频、输出目录和模型路径不能为空")
        if not 0 <= self.confidence <= 1 or not 0 <= self.small_target_ratio <= 1:
            raise ValueError("置信度和小目标占比必须在 0 到 1 之间")
        if not isinstance(self.target_count, int) or self.target_count < 0:
            raise ValueError("目标张数必须是非负整数，0 表示完整扫描")
        if not 0.1 <= self.min_sample_seconds <= 3600:
            raise ValueError("最小采样间隔必须在 0.1 到 3600 秒之间")
        if self.min_sample_frames is not None and (
                type(self.min_sample_frames) is not int or self.min_sample_frames < 1):
            raise ValueError("最小采样间隔必须是大于 0 的帧数")
        if not isinstance(self.video_paths, list) or any(
                not isinstance(path, str) or not path for path in self.video_paths):
            raise ValueError("视频队列必须包含有效路径")
        for key in ("image_size", "min_pixel_height", *DYNAMIC_JUMP_KEYS, "companion_min_height",
                    "reject_single_below"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} 必须大于 0")

    def save(self, path):
        self.validate()
        atomic_write_text(path, json.dumps({"schema_version": 4, **asdict(self)},
                                         ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.pop("schema_version", 1) not in (1, 2, 3, 4):
            raise ValueError("不支持的配置版本")
        config = cls(**data)
        config.validate()
        return config


def classify_boxes(boxes, frame_height, min_pixel_height):
    valid, small = [], []
    for prediction in boxes:
        if len(prediction) == 4:
            class_id, bx, by, bw, bh, score = 0, *prediction, 1.0
        elif len(prediction) == 5:
            class_id, bx, by, bw, bh = prediction
            score = 1.0
        elif len(prediction) == 6:
            class_id, bx, by, bw, bh, score = prediction
        else:
            raise ValueError("检测结果应包含归一化坐标、可选类别 ID 和置信度")
        box = (int(class_id), bx, by, bw, bh, bh * frame_height, float(score))
        (valid if box[5] >= min_pixel_height else small).append(box)
    return valid, small


def is_frame_qualified(valid_boxes, small_boxes, ratio=0.60, single_below=110):
    count = len(valid_boxes) + len(small_boxes)
    if not valid_boxes:
        return False, "无合格目标"
    if count >= 3 and len(small_boxes) / count > ratio and len(valid_boxes) <= 1:
        return False, "小目标占比过高"
    if count == 1 and len(valid_boxes) == 1 and valid_boxes[0][5] < single_below:
        return False, "单人边缘弱特征"
    return True, "合格"


def dynamic_jump_key(keep, valid_count):
    if not keep:
        return "idle_jump"
    if valid_count == 1:
        return "step_single"
    if valid_count in (2, 3):
        return "step_small_group"
    return "step_large_group"


def next_jump(keep, valid_count, config):
    return getattr(config, dynamic_jump_key(keep, valid_count))


def format_yolo(valid, small, companion_min_height=40):
    targets = valid + [box for box in small if box[5] >= companion_min_height]
    return "".join(f"{int(box[0])} {box[1]:.6f} {box[2]:.6f} "
                   f"{box[3]:.6f} {box[4]:.6f}\n" for box in targets)


def video_info(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {path}")
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        return {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "frames": frames, "fps": fps,
                "duration": frames / fps if fps > 0 else None}
    finally:
        cap.release()


class UltralyticsDetector:
    def __init__(self, config):
        from .models.adapters import create_detector
        self.model = create_detector(config.model_id, config.model_path,
                                     config.prompt_labels, config.device,
                                     config.image_size, config.confidence,
                                     config.trust_remote_code)

    def predict(self, frame):
        return self.model.predict(frame)


def legacy_source_prefix(path):
    name = Path(path).stem
    match = re.search(r"ch\d+", name, re.IGNORECASE)
    return match.group(0).lower() if match else name


def run_pipeline(config, progress=None, cancelled=None, detector=None, capture_factory=None,
                 sampler=None, source_prefix=None, existing_frames=None):
    import cv2
    config.validate()
    info = video_info(config.video_path) if capture_factory is None else None
    cap = (capture_factory(config.video_path) if capture_factory else
           cv2.VideoCapture(config.video_path))
    try:
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {config.video_path}")
        total = info["frames"] if info else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = info["fps"] if info else float(cap.get(cv2.CAP_PROP_FPS))
        height = info["height"] if info else int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if total <= 0 or height <= 0:
            raise ValueError("视频缺少有效帧数或分辨率")
        detector = detector or UltralyticsDetector(config)
        root = Path(config.output_dir)
        images, labels, metadata = root / "images", root / "labels", root / "metadata"
        for directory in (images, labels, metadata, root / "configs"):
            directory.mkdir(parents=True, exist_ok=True)
        classes = root / "classes.txt"
        from .models.adapters import parse_prompts
        class_names = (parse_prompts(config.prompt_labels)
                       if config.model_id != "local_yolo" else ["person"])
        if not classes.exists():
            atomic_write_text(classes, "\n".join(class_names) + "\n")
        elif read_class_names(classes) != class_names:
            raise ValueError("输出目录现有类别表与当前提示类别不同，请使用新目录或沿用原类别表")
        prefix = source_prefix or legacy_source_prefix(config.video_path)
        # Reject colliding sources before changing the previous run's provenance.
        for directory, suffix in ((images, "jpg"), (images, "json"), (labels, "txt")):
            if existing_frames is None and (not sampler or not sampler.saved_indices) and any(
                    directory.glob(f"{prefix}_f*.{suffix}")):
                raise FileExistsError(f"输出已存在: {prefix}，请使用新目录")
        run_id = uuid.uuid4().hex
        config.save(root / "configs" / f"run_{run_id}.json")
        config.save(root / "configs" / "run_config.json")
        current, scanned, saved, discarded = 0, 0, 0, 0
        indices = iter(sampler) if sampler is not None else None
        start = time.monotonic()
        manifest = metadata / "samples.jsonl"
        with manifest.open("a", encoding="utf-8") as records:
            while current < total:
                if cancelled and cancelled():
                    break
                if indices is not None:
                    current = next(indices, total)
                    if current >= total:
                        break
                elif existing_frames and current in existing_frames:
                    current += 1
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, current)
                actual = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"视频解码失败，目标帧 {current}")
                if actual != current:
                    raise RuntimeError(f"视频定位失败，目标帧 {current}，实际帧 {actual}")
                scanned += 1
                valid, small = classify_boxes(detector.predict(frame), height,
                                              config.min_pixel_height)
                if cancelled and cancelled():
                    break
                if any(not 0 <= box[0] < len(class_names) for box in valid + small):
                    raise ValueError(f"模型输出了类别表之外的类别 ID，已停止处理：帧 {current}")
                keep, reason = is_frame_qualified(valid, small,
                                                   config.small_target_ratio,
                                                   config.reject_single_below)
                scores = ([box[6] if config.model_id != "florence2" else None
                           for box in valid + [box for box in small
                                               if box[5] >= config.companion_min_height]]
                          if keep else [])
                if keep:
                    base = f"{prefix}_f{current:06d}"
                    image_path, label_path = images / f"{base}.jpg", labels / f"{base}.txt"
                    if image_path.exists() or label_path.exists() or image_path.with_suffix(".json").exists():
                        raise FileExistsError(f"输出已存在: {base}")
                    temporary = images / f"{base}.partial.jpg"
                    scores_path = metadata / f"{base}.scores.json"
                    if scores_path.exists():
                        raise FileExistsError(f"检测分数已存在: {scores_path}")
                    try:
                        if not cv2.imwrite(str(temporary), frame):
                            raise OSError(f"图片写入失败: {temporary}")
                        temporary.replace(image_path)
                        label_path.write_text(format_yolo(valid, small,
                                             config.companion_min_height), encoding="utf-8")
                        atomic_write_text(scores_path, json.dumps({"scores": scores}))
                    except Exception:
                        temporary.unlink(missing_ok=True)
                        image_path.unlink(missing_ok=True)
                        label_path.unlink(missing_ok=True)
                        scores_path.unlink(missing_ok=True)
                        raise
                    saved += 1
                else:
                    discarded += 1
                if sampler is not None:
                    sampler.observe(current, keep, len(valid))
                records.write(json.dumps({"run_id": run_id, "source_video": config.video_path,
                                          "frame_index": current,
                                          "timestamp_seconds": current / fps if fps > 0 else None,
                                          "model": config.model_path, "kept": keep,
                                          "model_id": config.model_id,
                                          "prompt_labels": config.prompt_labels,
                                          "image_name": image_path.name if keep else None,
                                          "reason": reason,
                                          "valid_count": len(valid), "small_count": len(small),
                                          "mean_confidence": (sum(box[6] for box in valid + small) /
                                                              max(1, len(valid) + len(small)))},
                                         ensure_ascii=False) + "\n")
                records.flush()
                jump = next_jump(keep, len(valid), config)
                if progress:
                    progress({"frame": current, "total": total, "scanned": scanned,
                              "saved": saved, "discarded": discarded,
                              "elapsed": time.monotonic() - start, "reason": reason,
                              "image": str(image_path) if keep else None,
                              "scores": scores})
                current = current + jump if indices is None else 0
        return {"scanned": scanned, "saved": saved, "discarded": discarded,
                "cancelled": bool(cancelled and cancelled())}
    finally:
        cap.release()
