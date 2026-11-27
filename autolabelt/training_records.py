"""Paths and display metadata for training runs."""

import json
import uuid
from datetime import datetime
from pathlib import Path


RUN_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


def create_run_directory(output):
    name = f"{datetime.now():{RUN_TIMESTAMP_FORMAT}}_{uuid.uuid4().hex[:6]}"
    directory = output / name
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def read_training_record(path):
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or not isinstance(record.get("args"), dict):
        raise ValueError(f"无法读取训练记录：{path}")
    record["directory"] = str(path.parent)
    record.setdefault("status", "未知状态")
    return record


def training_history(output):
    rows = []
    for path in output.glob("*/task.json"):
        try:
            record = read_training_record(path)
            try:
                created = datetime.strptime(path.parent.name[:15], RUN_TIMESTAMP_FORMAT)
            except ValueError:
                created = datetime.fromtimestamp(path.stat().st_mtime)
            args = record["args"]
            model = Path(record.get("base_model") or args.get("model") or "").stem
            if model.casefold() in ("", "best", "last"):
                model = "RT-DETR" if args.get("model_family") == "rtdetr" else "YOLO"
            source = Path(record.get("source") or "")
            dataset = (source.parent if source.suffix.lower() in (".yaml", ".yml") else source).name
            title = model if dataset.casefold() in ("", "dataset", "train", model.casefold()) else f"{dataset} · {model}"
            rows.append((created, path, title, record["status"], source))
        except (OSError, TypeError, ValueError):
            continue
    return sorted(rows, key=lambda row: row[:2], reverse=True)


def training_weights(directory):
    return Path(directory) / "train" / "weights"
