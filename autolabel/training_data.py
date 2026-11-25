"""Validate and freeze detection datasets for reproducible local training."""

import hashlib
import json
import random
import re
import shutil
from pathlib import Path

import yaml
from PIL import Image

from .annotations import IMAGE_SUFFIXES, parse_labels
from .workbench.views.labeling.annotation_io import AnnotationTransaction, atomic_write_json, atomic_write_text
from .workbench.views.labeling.label_mapping import read_class_names


def _read_dataset_yaml(path):
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError(f"无效的数据集 YAML：{path}")
    return config


def _split_entries(root, splits):
    for split, paths in splits.items():
        paths = paths if isinstance(paths, list) else [paths]
        for value in paths:
            if not isinstance(value, str):
                raise ValueError(f"缺少 {split} 图片路径")
            path = root / value
            if path.is_dir():
                images = sorted(p for p in path.rglob("*")
                                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
            elif path.is_file() and path.suffix.lower() == ".txt":
                images = [
                    path.parent / line.strip()
                    for line in path.read_text(encoding="utf-8-sig").splitlines()
                    if line.strip()
                ]
            else:
                raise ValueError(f"图片路径不存在：{path}")
            for image in images:
                parts = list(image.resolve().parts)
                if "images" not in parts:
                    raise ValueError(f"图片须位于 images 目录中：{image}")
                index = len(parts) - 1 - parts[::-1].index("images")
                parts[index] = "labels"
                yield split, image, Path(*parts).with_suffix(".txt")


def _flat_entries(source, ratio, seed, split_manifest=None):
    images = sorted(p for p in (source / "images").iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if len(images) < 2:
        raise ValueError("至少需要两张图片才能划分训练集和验证集")
    if split_manifest:
        try:
            saved = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
            assignments = {Path(row["source"]): row["split"] for row in saved["samples"]}
            if assignments.keys() == set(images) and set(assignments.values()) == {"train", "val"}:
                entries = [(assignments[image], image, source / "labels" / f"{image.stem}.txt")
                           for image in images]
                return entries, "沿用上一次的训练集和验证集划分。", saved["ratio"], saved["seed"], Path(split_manifest)
        except (OSError, ValueError, TypeError, KeyError):
            pass
    groups = {}
    for image in images:
        match = re.match(r"(.+)_f\d+(?:_.*)?$", image.stem)
        groups.setdefault(match[1] if match else image.stem, []).append(image)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    warning = ""
    if len(keys) > 1:
        cut = max(1, min(len(keys) - 1, round(len(keys) * ratio)))
        training = {p for key in keys[:cut] for p in groups[key]}
    else:
        cut = max(1, min(len(images) - 1, round(len(images) * ratio)))
        training = set(images[:cut])
        warning = "只有一个来源视频：已按时间顺序划分；相邻片段仍可能相似，建议使用独立视频验证。"
    entries = [("train" if p in training else "val", p,
                source / "labels" / f"{p.stem}.txt") for p in images]
    return entries, warning, ratio, seed, None


def _partition_pair(root, index, split, image):
    name = f"{index:07d}_{image.name}"
    return root / "images" / split / name, root / "labels" / split / Path(name).with_suffix(".txt")


def _reusable_partition(manifest, entries):
    root = manifest.parent
    for index, (split, image, _) in enumerate(entries):
        cached, _ = _partition_pair(root, index, split, image)
        try:
            original, saved = image.stat(), cached.stat()
        except OSError:
            return None
        if (original.st_size, original.st_mtime_ns) != (saved.st_size, saved.st_mtime_ns):
            return None
    return root


def prepare_training_dataset(source, destination, ratio=0.8, seed=42, cancelled=lambda: False,
                             yaml_path=None, split_manifest=None, reference_splits=False,
                             copy_existing_splits=False):
    """Validate the samples and create an Ultralytics dataset configuration."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not 0 < ratio < 1:
        raise ValueError("训练集比例必须在 0 和 1 之间")
    warning = ""
    splits = {"train": "images/train", "val": "images/val"}
    dataset_root = destination
    copy_samples = True
    automatic_split = False
    reused_manifest = None
    reused_partition = None
    if source.is_dir():
        names_file = source / "classes.txt"
        if any((source / path).is_dir() for path in splits.values()):
            entries = list(_split_entries(source, splits))
            if reference_splits or (yaml_path is not None and not copy_existing_splits):
                dataset_root, copy_samples = source, False
            if not names_file.is_file():
                names_file = next((path for path in (source / "data.yaml", source / "data.yml")
                                   if path.is_file()), names_file)
        else:
            automatic_split = True
            entries, warning, ratio, seed, reused_manifest = _flat_entries(source, ratio, seed, split_manifest)
            if reference_splits and reused_manifest:
                reused_partition = _reusable_partition(reused_manifest, entries)
                if reused_partition:
                    dataset_root, copy_samples = reused_partition, False
        names = (read_class_names(names_file) if names_file.suffix == ".txt"
                 else _read_dataset_yaml(names_file).get("names"))
    else:
        config = _read_dataset_yaml(source)
        names = config.get("names")
        root = (source.parent / (config.get("path") or "")).resolve()
        input_splits = {split: config.get(split) for split in splits}
        entries = list(_split_entries(root, input_splits))
        if reference_splits and all(
            (root / path).is_dir() for paths in input_splits.values()
            for path in (paths if isinstance(paths, list) else [paths])
        ):
            dataset_root, copy_samples, splits = root, False, input_splits
    if isinstance(names, dict):
        if names.keys() != set(range(len(names))):
            raise ValueError("类别 ID 必须从 0 连续编号")
        names = [names[i] for i in range(len(names))]
    if (not isinstance(names, list) or not names
            or any(not isinstance(n, str) or not n.strip() for n in names)
            or len(set(names)) != len(names)):
        raise ValueError("类别表为空、重复或格式无效")
    if not all(any(s == split for s, _, _ in entries) for split in ("train", "val")):
        raise ValueError("训练集和验证集都必须包含图片")
    data_dir = destination.parent if reference_splits else destination
    data = Path(yaml_path) if yaml_path is not None else data_dir / "data.yaml"
    manifest_path = (destination if copy_samples else data_dir) / "manifest.json" if yaml_path is None else None
    config = {"path": str(dataset_root), **splits}
    yaml_text = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    yaml_text += "\n" + yaml.safe_dump({"names": dict(enumerate(names))}, allow_unicode=True)
    if copy_samples:
        destination.mkdir(parents=True, exist_ok=False)
    try:
        manifest, seen, positive, changed_labels = [], set(), 0, {}
        label_cache_paths = set()
        for index, (split, image, label) in enumerate(entries):
            if cancelled():
                raise InterruptedError("已取消数据准备")
            identity = str(image.resolve()).casefold()
            if identity in seen:
                raise ValueError(f"训练集和验证集有重复图片：{image}")
            seen.add(identity)
            if not label.is_file():
                raise ValueError(f"缺少标签（负样本请提供空 TXT）：{label}")
            target_image, target_label = image, label
            if copy_samples:
                target_image, target_label = _partition_pair(destination, index, split, image)
                target_image.parent.mkdir(parents=True, exist_ok=True)
                target_label.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image, target_image)
                shutil.copy2(label, target_label)
            contents = target_label.read_bytes()
            positive += len(parse_labels(contents.decode("utf-8-sig"), len(names), target_label))
            with Image.open(target_image) as opened:
                opened.verify()
            if reused_partition:
                target_image, cached_label = _partition_pair(reused_partition, index, split, image)
                if not cached_label.is_file() or cached_label.read_bytes() != contents:
                    changed_labels[cached_label] = contents
                target_label = cached_label
            if not copy_samples:
                label_cache_paths.add(target_label.parent.with_suffix(".cache"))
            if manifest_path is not None:
                manifest.append({
                    "source": str(image), "split": split, "image": target_image.name,
                    "image_path": str(target_image), "label_path": str(target_label),
                    "label_sha256": hashlib.sha256(contents).hexdigest(),
                })
        if not positive:
            raise ValueError("数据集没有任何目标框，无法训练目标检测模型")
        if cancelled():
            raise InterruptedError("已取消数据准备")
        if changed_labels:
            with AnnotationTransaction(changed_labels) as transaction:
                for path, contents in changed_labels.items():
                    transaction.stage_path(path).write_bytes(contents)
                if cancelled():
                    raise InterruptedError("已取消数据准备")
                transaction.commit()
        # Ultralytics may reuse cached labels after a same-length edit.
        if reference_splits:
            for cache in label_cache_paths:
                cache.unlink(missing_ok=True)
        data.parent.mkdir(parents=True, exist_ok=True)
        if manifest_path is not None:
            atomic_write_json(manifest_path, {
                "data": str(data), "dataset_root": str(dataset_root), "seed": seed, "ratio": ratio,
                "names": names, "samples": manifest,
            }, ensure_ascii=False, indent=2)
        atomic_write_text(data, yaml_text)
    except Exception:
        if copy_samples:
            shutil.rmtree(destination, ignore_errors=True)
        raise
    return {"data": str(data), "names": names, "warning": warning, "manifest": manifest_path,
            "automatic_split_manifest": (dataset_root / "manifest.json"
                                         if automatic_split and yaml_path is None else None),
            "train": sum(s == "train" for s, _, _ in entries),
            "val": sum(s == "val" for s, _, _ in entries)}
