"""Sequential video jobs, bounded pilot analysis and time-distributed sampling."""

from bisect import bisect_left, insort
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
import json
import math
import re
import time

from .core import (DYNAMIC_JUMP_KEYS, ProjectConfig, UltralyticsDetector, classify_boxes,
                   dynamic_jump_key, is_frame_qualified, legacy_source_prefix, run_pipeline)
from .models.adapters import parse_prompts
from .workbench.views.labeling.annotation_io import atomic_write_text
from .workbench.views.labeling.label_mapping import read_class_names


TARGET_COUNT_BUCKETS = (("empty", "未检出目标"), ("single", "1 个目标"),
                        ("small_group", "2–3 个目标"), ("large_group", "4 个及以上目标"))


def video_paths(config):
    return config.video_paths or [config.video_path]


def plan_signature(config):
    sources = []
    for path in video_paths(config):
        stat = Path(path).stat()
        sources.append((str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns))
    model = Path(config.model_path).stat()
    settings = asdict(config)
    for key in ("output_dir", "target_count", "min_sample_seconds", "min_sample_frames",
                *DYNAMIC_JUMP_KEYS):
        settings.pop(key)
    return json.dumps([settings, sources, model.st_mtime_ns], sort_keys=True)


def source_prefix(path):
    # Full stem plus path digest: ch1 clips and equal basenames remain distinct.
    digest = sha256(str(Path(path).resolve()).casefold().encode()).hexdigest()[:10]
    stem = re.sub(r"[^\w.-]", "_", Path(path).stem)[:80]
    return f"{stem}_{digest}"


def allocate(total, weights, *, reserve_each=True):
    """Largest remainder allocation with an optional minimum of one."""
    if not weights:
        return []
    base = int(reserve_each and total >= len(weights))
    remaining = total - base * len(weights)
    weight_sum = sum(weights)
    raw = ([remaining * weight / weight_sum for weight in weights] if weight_sum else
           [remaining / len(weights)] * len(weights))
    result = [base + int(value) for value in raw]
    order = sorted(range(len(weights)), key=lambda i: raw[i] - int(raw[i]), reverse=True)
    for index in order[:total - sum(result)]:
        result[index] += 1
    return result


def analyze_videos(config, progress=None, cancelled=None, detector=None, capture_factory=None):
    """Sample short windows throughout each source, without writing a dataset."""
    import cv2
    config.validate(require_output=False)
    detector = detector or UltralyticsDetector(config)
    reports = []
    for index, path in enumerate(video_paths(config)):
        if cancelled and cancelled():
            return None
        cap = (capture_factory or cv2.VideoCapture)(path)
        try:
            if not cap.isOpened():
                raise ValueError(f"无法打开视频: {path}")
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if frames <= 0 or height <= 0 or fps <= 0 or not math.isfinite(fps):
                raise ValueError(f"视频缺少有效帧数、FPS 或分辨率: {path}")
            # Up to 80 predictions: 20 temporal windows, four spaced probes each.
            windows = min(20, max(1, math.ceil(frames / fps / 30)))
            positions = sorted({min(frames - 1, int((window + fraction) * frames / windows))
                                for window in range(windows)
                                for fraction in (0.15, 0.38, 0.62, 0.85)})
            counts = dict.fromkeys((key for key, _ in TARGET_COUNT_BUCKETS), 0)
            jump_counts = dict.fromkeys(DYNAMIC_JUMP_KEYS, 0)
            started = time.monotonic()
            for number, position in enumerate(positions):
                if cancelled and cancelled():
                    return None
                cap.set(cv2.CAP_PROP_POS_FRAMES, position)
                actual = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                ok, frame = cap.read()
                if not ok or frame is None or actual != position:
                    raise RuntimeError(f"预分析无法解码或定位: {path}，帧 {position}")
                valid, small = classify_boxes(detector.predict(frame), height, config.min_pixel_height)
                count = len(valid) + len(small)
                bucket = "empty" if count == 0 else "single" if count == 1 else (
                    "small_group" if count < 4 else "large_group")
                counts[bucket] += 1
                keep, _ = is_frame_qualified(valid, small, config.small_target_ratio,
                                             config.reject_single_below)
                jump_counts[dynamic_jump_key(keep, len(valid))] += 1
                if progress:
                    progress({"phase": "analysis", "video_index": index,
                              "video_count": len(video_paths(config)),
                              "fraction": (number + 1) / len(positions),
                              "reason": f"预分析 {Path(path).name} · {number + 1}/{len(positions)}"})
            probes = len(positions)
            rate = (probes - jump_counts["idle_jump"]) / probes
            estimated = allocate(frames, counts.values(), reserve_each=False)
            reports.append({"path": path, "frames": frames, "fps": fps,
                            "duration": frames / fps, "target_counts": counts,
                            "estimated_frames": dict(zip(counts, estimated)),
                            "rate": rate, "probes": probes, "jump_counts": jump_counts,
                            "baseline_jump": sum(count * getattr(config, key)
                                                 for key, count in jump_counts.items()) / probes,
                            "seconds_per_probe": (time.monotonic() - started) / probes})
        finally:
            cap.release()
    return recommend_sampling(reports, config)


def recommended_jump(report, quota):
    return max(report["gap"], round(report["frames"] * max(report["rate"], 0.05) / quota))


def recommend_dynamic_jumps(reports, target):
    """Scale the validated step ratios using the pilot's eligible-frame mix."""
    baseline = ProjectConfig()
    total_frames = sum(report["frames"] for report in reports)
    representative_fps = total_frames / sum(report["duration"] for report in reports)
    eligible = {
        key: sum(report["frames"] * report["jump_counts"][key] / report["probes"]
                 for report in reports)
        for key in DYNAMIC_JUMP_KEYS[1:]
    }
    eligible_frames = sum(eligible.values())
    scale = representative_fps / 25  # The databuild baseline steps are expressed at 25 FPS.
    minimum = 1
    if eligible_frames and target:
        # With the baseline ratios, expected saves are approximately sum(frames / step).
        scale = sum(frames / getattr(baseline, key) for key, frames in eligible.items()) / target
        minimum = max(report["gap"] for report in reports)
    elif eligible_frames:
        # A sparse batch benefits from denser probes, capped at twice baseline density.
        scale *= max(0.5, eligible_frames / total_frames)
    return {key: min(100_000, max(minimum, round(getattr(baseline, key) * scale)))
            for key in DYNAMIC_JUMP_KEYS}


def recommend_sampling(reports, config):
    videos = []
    for report in reports:
        gap = config.min_sample_frames or max(1, math.ceil(report["fps"] * config.min_sample_seconds))
        capacity = math.ceil(report["frames"] / gap)
        videos.append({**report, "gap": gap, "capacity": capacity,
                       "weight": capacity * max(report["rate"], 0.05)})
    quotas = allocate(config.target_count, [report["weight"] for report in videos])
    for report, quota in zip(videos, quotas):
        report["quota"] = quota
        report["recommended_jump"] = recommended_jump(report, quota) if quota else None
        report["estimated_saved"] = min(quota, round(report["capacity"] * report["rate"]))
    return {"videos": videos, "target": config.target_count,
            "dynamic_jumps": recommend_dynamic_jumps(videos, config.target_count),
            "total_frames": sum(report["frames"] for report in videos),
            "estimated_saved": sum(report["estimated_saved"] for report in videos),
            "estimated_seconds": sum(
                min(report["capacity"], report["quota"] / max(report["rate"], 0.05)) *
                report["seconds_per_probe"] for report in videos)}


class TargetSampler:
    """Fill temporal quotas, then supplement empty regions on a bounded grid.

    Feedback updates the next round's density. A fixed minimum gap bounds work
    and separates accepted images; no confidence or quality threshold is changed.
    """

    def __init__(self, frames, gap, target, rate=0.5, cancelled=None, initial_jump=None,
                 existing_frames=()):
        self.frames, self.gap, self.target = frames, gap, target
        self.initial_jump = initial_jump
        self.rate = max(0.05, rate)
        self.prior_rate = self.rate
        self.cancelled = cancelled or (lambda: False)
        self.visited = set()
        self.saved_indices = sorted(existing_frames)
        self.has_off_grid_samples = any(frame % gap for frame in self.saved_indices)
        self.target += len(self.saved_indices)
        self.attempts = self.accepted = 0

    def observe(self, frame, keep, valid_count=0):
        self.visited.add(frame)
        self.attempts += 1
        if keep:
            insort(self.saved_indices, frame)
            self.accepted += 1
        self.rate = (self.accepted + 4 * self.prior_rate) / (self.attempts + 4)

    def available(self, position):
        index = bisect_left(self.saved_indices, position)
        return (position not in self.visited and
                (index == 0 or position - self.saved_indices[index - 1] >= self.gap) and
                (index == len(self.saved_indices) or self.saved_indices[index] - position >= self.gap))

    def sample_region(self, start, end, goal, initial_jump=None):
        slots = max(0, math.ceil((end - start) / self.gap))
        if not slots:
            return
        first_round = math.ceil((end - start) / initial_jump) if initial_jump else goal
        density = min(slots, max(1, goal, first_round))
        while len(self.saved_indices) < self.target:
            if self.cancelled():
                return
            saved_here = bisect_left(self.saved_indices, end) - bisect_left(self.saved_indices, start)
            if saved_here >= goal:
                return
            for slot in range(density):
                if self.cancelled() or len(self.saved_indices) >= self.target:
                    return
                grid_index = min(slots - 1, int((slot + 0.5) * slots / density))
                position = start + grid_index * self.gap
                if self.available(position):
                    yield position
                    saved_here = bisect_left(self.saved_indices, end) - bisect_left(self.saved_indices, start)
                    if saved_here >= goal:
                        return
            if initial_jump or density == slots:
                return
            remaining = max(1, goal - saved_here)
            density = min(slots, max(density * 2, math.ceil(remaining / max(0.05, self.rate))))

    def __iter__(self):
        windows = min(20, self.target, self.frames)
        if not windows:
            return
        quotas = allocate(self.target, [1] * windows)
        # Align every region to the same minimum-gap grid.
        slots = math.ceil(self.frames / self.gap)
        for index, quota in enumerate(quotas):
            start = (index * slots // windows) * self.gap
            end = min(self.frames, ((index + 1) * slots // windows) * self.gap)
            yield from self.sample_region(start, end, quota, self.initial_jump)
        # Empty windows can transfer their deficit to any remaining useful slot.
        yield from self.sample_region(0, self.frames, self.target)
        if self.has_off_grid_samples:
            # Earlier runs may have used a different gap. Fill valid slots
            # between those existing frames after exhausting the new grid.
            position = 0
            while position < self.frames and len(self.saved_indices) < self.target and not self.cancelled():
                index = bisect_left(self.saved_indices, position)
                if index and position - self.saved_indices[index - 1] < self.gap:
                    position = self.saved_indices[index - 1] + self.gap
                elif index < len(self.saved_indices) and self.saved_indices[index] - position < self.gap:
                    position = self.saved_indices[index] + self.gap
                elif position in self.visited:
                    position += 1
                else:
                    yield position
                    position += self.gap


def existing_samples(root, paths):
    """Match existing generated files to their recorded source and frame."""
    images = {file.stem for file in (root / "images").glob("*.jpg")
              if not file.name.endswith(".partial.jpg")}
    artifacts = images | {file.stem for file in (root / "images").glob("*.json")}
    artifacts.update(file.stem for file in (root / "labels").glob("*.txt"))
    artifacts.update(file.name.removesuffix(".scores.json")
                     for file in (root / "metadata").glob("*.scores.json"))
    source_keys = {str(Path(path).resolve()).casefold(): index for index, path in enumerate(paths)}
    prefixes = [(source_prefix(path), legacy_source_prefix(path)) for path in paths]
    all_prefixes = {prefix for pair in prefixes for prefix in pair}
    frames = [set() for _ in paths]
    matching = {stem for stem in artifacts
                if stem.rpartition("_f")[0] in all_prefixes and stem.rpartition("_f")[2].isdecimal()}
    if not matching:
        return len(images), frames

    matched = {}
    manifest = root / "metadata" / "samples.jsonl"
    if manifest.is_file():
        with manifest.open(encoding="utf-8") as records:
            for line in records:
                record = json.loads(line)
                if not record.get("kept"):
                    continue
                index = source_keys.get(str(Path(record["source_video"]).resolve()).casefold())
                frame = record["frame_index"]
                if index is None:
                    continue
                names = [f"{prefix}_f{frame:06d}" for prefix in prefixes[index]]
                if record.get("image_name"):
                    names = [Path(record["image_name"]).stem]
                for stem in matching.intersection(names):
                    if stem in matched and matched[stem] != index:
                        raise FileExistsError(f"输出的源视频记录有冲突：{stem}，请选择新的输出目录")
                    matched[stem] = index
                    frames[index].add(frame)
    if unmatched := matching - matched.keys():
        raise FileExistsError(f"输出已存在但缺少来源记录：{min(unmatched)}，请选择新的输出目录")
    return len(images), frames


def run_batch(config, progress=None, cancelled=None, detector=None, plan=None,
              capture_factory=None, refresh_analysis=False):
    config.validate()
    paths = video_paths(config)
    if len({str(Path(path).resolve()).casefold() for path in paths}) != len(paths):
        raise ValueError("视频队列包含重复文件")
    cancelled = cancelled or (lambda: False)
    if cancelled():
        return {"saved": 0, "scanned": 0, "discarded": 0, "cancelled": True}
    root = Path(config.output_dir)
    classes = root / "classes.txt"
    class_names = ["person"] if config.model_id == "local_yolo" else parse_prompts(config.prompt_labels)
    if classes.is_file() and read_class_names(classes) != class_names:
        raise ValueError("输出目录现有类别表与当前提示类别不同，请使用新目录或沿用原类别表")
    existing, saved_frames = existing_samples(root, paths)
    prefixes = [source_prefix(path) for path in paths]
    if len(paths) == 1 and not config.target_count:
        prefixes = [legacy_source_prefix(paths[0])]
    target = config.target_count
    totals = {"saved": 0, "scanned": 0, "discarded": 0}
    if target and existing >= target:
        return {**totals, "existing": existing, "dataset_total": existing, "cancelled": False,
                "target": target, "shortfall": 0, "videos": [], "plan": plan}
    if progress and existing:
        progress({"phase": "existing", "batch_saved": existing, "target": target})
    detector = detector or UltralyticsDetector(config)
    if plan is None and (target or refresh_analysis):
        plan = analyze_videos(config, progress, cancelled, detector, capture_factory)
        if plan is None:
            return {**totals, "existing": existing, "dataset_total": existing, "cancelled": True}
        if progress:
            progress({"phase": "plan", "plan": plan, "batch_saved": existing})
    allocation = plan
    if target and existing:
        allocation = recommend_sampling(plan["videos"], replace(config, target_count=target - existing))
    results, samplers = [], []

    def process(index, sampler, supplemental=False):
        high_water = 0
        def report(event):
            nonlocal high_water
            high_water = max(high_water, event["frame"])
            if progress:
                progress({**event, "phase": "processing", "video_index": index,
                          "video_count": len(paths), "video_path": paths[index],
                          "batch_saved": existing + totals["saved"] + event["saved"],
                          "target": target, "supplemental": supplemental,
                          "fraction": high_water / max(1, event["total"])})
        if progress:
            progress({"phase": "video_start", "video_index": index,
                      "video_count": len(paths), "supplemental": supplemental})
        result = run_pipeline(replace(config, video_path=paths[index]), report, cancelled,
                              detector, capture_factory, sampler, prefixes[index], saved_frames[index])
        for key in totals:
            totals[key] += result[key]
        if progress:
            progress({"phase": "video_done", "video_index": index,
                      "video_count": len(paths), "saved": result["saved"],
                      "batch_saved": existing + totals["saved"], "target": target,
                      "cancelled": result["cancelled"]})
        return result

    for index, path in enumerate(paths):
        if cancelled():
            break
        sampler = None
        if target:
            report = allocation["videos"][index]
            remaining = max(0, target - existing - totals["saved"])
            upcoming = allocation["videos"][index:]
            deficit = max(0, remaining - sum(entry["quota"] for entry in upcoming))
            extra = allocate(deficit, [entry["weight"] for entry in upcoming], reserve_each=False)
            quota = min(remaining, report["quota"] + extra[0])
            jump = report["recommended_jump"]
            if quota != report["quota"]:
                jump = recommended_jump(report, quota) if quota else None
            sampler = TargetSampler(report["frames"], report["gap"], quota, report["rate"],
                                    cancelled, initial_jump=jump, existing_frames=saved_frames[index])
        samplers.append(sampler)
        results.append(process(index, sampler))
    # Revisit earlier sources when later videos cannot supply their share.
    if target and not cancelled():
        for index, sampler in enumerate(samplers):
            deficit = target - existing - totals["saved"]
            if deficit <= 0 or cancelled():
                break
            sampler.target = len(sampler.saved_indices) + deficit
            extra = process(index, sampler, supplemental=True)
            for key in ("saved", "scanned", "discarded"):
                results[index][key] += extra[key]
    dataset_total = existing + totals["saved"]
    result = {**totals, "existing": existing, "dataset_total": dataset_total,
              "cancelled": cancelled(), "target": target, "shortfall": max(0, target - dataset_total),
              "videos": results, "plan": plan}
    if results:
        atomic_write_text(root / "metadata" / f"batch_{time.time_ns()}.json",
                          json.dumps({"config": asdict(config), **result}, ensure_ascii=False, indent=2))
    return result
