from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback

EVENT_PREFIX = "__AUTOLABEL_WORKER_EVENT__="
PROTOCOL_VERSION = 1
EXPORT_DEPENDENCIES = {
    "onnx": (
        ("onnx", "onnx>=1.15.0"),
        ("onnxslim", "onnxslim>=0.1.59"),
        ("onnxruntime", "onnxruntime"),
    ),
    "openvino": (("openvino", "openvino>=2024.0.0"),),
    "engine": (("tensorrt", "tensorrt>7.0.0,!=10.1.0"),),
    "coreml": (("coremltools", "coremltools>=8.0"),),
    "saved_model": (("tensorflow", "tensorflow>=2.0.0"),),
    "pb": (("tensorflow", "tensorflow>=2.0.0"),),
    "tflite": (("tensorflow", "tensorflow>=2.0.0"),),
    "edgetpu": (("tensorflow", "tensorflow>=2.0.0"),),
    "tfjs": (("tensorflow", "tensorflow>=2.0.0"),),
    "paddle": (("paddle", "paddlepaddle"), ("x2paddle", "x2paddle")),
    "mnn": (("MNN", "MNN>=2.9.6"),),
    "ncnn": (("ncnn", "ncnn"),),
    "imx": (
        ("imx500_converter", "imx500-converter[pt]>=3.16.1"),
        ("mct_quantizers", "mct-quantizers>=1.6.0"),
    ),
    "rknn": (("rknn", "rknn-toolkit2"),),
    "torchscript": (),
}


def emit(event: str, **data) -> None:
    payload = {"version": PROTOCOL_VERSION, "event": event, **data}
    stream = sys.__stdout__ or sys.stdout
    stream.write(f"{EVENT_PREFIX}{json.dumps(payload, ensure_ascii=False)}\n")
    stream.flush()


class EventLogStream:
    def __init__(self, event: str):
        self._event = event
        self._buffer = ""
        self._progress = False

    def write(self, text: str) -> None:
        parts = re.split(r"(\r\n|\r|\n)", text or "")
        for segment, ending in zip(parts[::2], parts[1::2]):
            self._buffer += segment
            self.flush()
            if ending == "\r":
                self._progress = True
            elif self._progress:
                emit(self._event, message="", progress="finish")
                self._progress = False
        self._buffer += parts[-1]

    def flush(self) -> None:
        if self._progress and self._buffer == "\033[K":
            emit(self._event, message="", progress="finish")
            self._progress = False
        elif self._buffer.strip():
            if self._progress:
                emit(self._event, message=self._buffer.rstrip(), progress="update")
            else:
                emit(self._event, message=self._buffer.rstrip())
        self._buffer = ""


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def validate_cpu(torch) -> None:
    tensor = torch.tensor([1.0], device="cpu") + 1
    if tensor.item() != 2.0:
        raise RuntimeError("CPU tensor returned an unexpected result")


def probe_cuda(torch, cuda_version) -> tuple[bool, list[dict], str | None]:
    gpus = []
    error = None
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
            if cuda_version:
                error = (
                    "CUDA runtime is installed but no usable driver or device "
                    "was found"
                )
            return False, gpus, error
        for index in range(torch.cuda.device_count()):
            gpu = {
                "index": index,
                "name": f"GPU {index}",
                "available": False,
                "error": None,
            }
            try:
                gpu["name"] = torch.cuda.get_device_name(index)
                tensor = torch.tensor([1.0], device=f"cuda:{index}") + 1
                torch.cuda.synchronize(index)
                if tensor.item() != 2.0:
                    raise RuntimeError(
                        "CUDA tensor returned an unexpected result"
                    )
                gpu["available"] = True
            except Exception as exc:
                gpu["error"] = str(exc)
            gpus.append(gpu)
    except Exception as exc:
        error = str(exc)
    return any(gpu["available"] for gpu in gpus), gpus, error


def probe_mps(torch) -> tuple[bool, str | None]:
    try:
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            return False, None
        tensor = torch.tensor([1.0], device="mps") + 1
        if tensor.item() != 2.0:
            raise RuntimeError("MPS tensor returned an unexpected result")
        return True, None
    except Exception as exc:
        return False, str(exc)


def run_probe() -> None:
    result = {
        "python_executable": sys.executable,
        "python_version": ".".join(
            str(value) for value in sys.version_info[:3]
        ),
        "torch_version": None,
        "ultralytics_version": package_version("ultralytics"),
        "cuda_version": None,
        "cpu_available": False,
        "cuda_available": False,
        "mps_available": False,
        "gpus": [],
        "cuda_error": None,
        "mps_error": None,
    }
    try:
        import torch
    except Exception as exc:
        emit(
            "environment_error",
            error=f"PyTorch is not available: {exc}",
            error_type="torch_missing",
            environment=result,
        )
        return
    result["torch_version"] = str(torch.__version__)
    result["cuda_version"] = getattr(torch.version, "cuda", None)
    try:
        validate_cpu(torch)
        result["cpu_available"] = True
    except Exception as exc:
        emit(
            "environment_error",
            error=f"CPU tensor validation failed: {exc}",
            error_type="cpu_unavailable",
            environment=result,
        )
        return
    try:
        import ultralytics

        result["ultralytics_version"] = str(ultralytics.__version__)
    except Exception as exc:
        emit(
            "environment_error",
            error=f"Ultralytics is not available: {exc}",
            error_type="ultralytics_missing",
            environment=result,
        )
        return
    (
        result["cuda_available"],
        result["gpus"],
        result["cuda_error"],
    ) = probe_cuda(torch, result["cuda_version"])
    result["mps_available"], result["mps_error"] = probe_mps(torch)
    emit("environment_detected", **result)


def validate_selected_device(torch, device) -> None:
    if device == "cpu":
        tensor = torch.tensor([1.0], device="cpu") + 1
        if tensor.item() != 2.0:
            raise RuntimeError("CPU device failed tensor validation")
        return
    if device == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS device is no longer available")
        tensor = torch.tensor([1.0], device="mps") + 1
        if tensor.item() != 2.0:
            raise RuntimeError("MPS device failed tensor validation")
        return
    if isinstance(device, int) and device >= 0:
        device = [device]
    if not isinstance(device, list):
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        raise RuntimeError("CUDA is no longer available")
    for raw_index in device:
        index = int(raw_index)
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {index} is no longer available")
        tensor = torch.tensor([1.0], device=f"cuda:{index}") + 1
        torch.cuda.synchronize(index)
        if tensor.item() != 2.0:
            raise RuntimeError(f"CUDA device {index} failed tensor validation")


def configure_weights_directory(weights_directory: str) -> None:
    os.makedirs(weights_directory, exist_ok=True)
    os.chdir(weights_directory)
    from ultralytics import settings

    settings.update({"weights_dir": weights_directory})


def attach_training_events(model, send):
    """Publish structured progress without scraping terminal progress bars."""
    reported_epoch = None

    def epoch_start(trainer):
        device = str(getattr(trainer, "device", ""))
        memory = None
        if device.startswith("cuda"):
            import torch
            memory = torch.cuda.memory_reserved(trainer.device) / 1024 ** 3
        send("training_progress", epoch=int(trainer.epoch) + 1,
             total_epochs=int(trainer.epochs), save_dir=str(trainer.save_dir),
             device=device, gpu_reserved_gb=memory)

    def epoch_end(trainer):
        nonlocal reported_epoch
        epoch = int(trainer.epoch) + 1
        if reported_epoch == epoch:
            return
        reported_epoch = epoch
        metrics = dict(getattr(trainer, "metrics", {}) or {})
        loss = getattr(trainer, "tloss", None)
        if loss is not None:
            metrics.update(trainer.label_loss_items(loss, prefix="train"))
        values = {}
        for key, value in metrics.items():
            try:
                value = float(value)
                if math.isfinite(value):
                    values[str(key)] = value
            except (ValueError, TypeError):
                continue
        send("training_metrics", epoch=epoch, total_epochs=int(trainer.epochs), metrics=values,
             save_dir=str(trainer.save_dir))

    model.add_callback("on_train_epoch_start", epoch_start)
    model.add_callback("on_fit_epoch_end", epoch_end)
    model.add_callback("on_val_start", lambda validator: send("training_phase", phase="validation"))


def run_train(payload: dict) -> None:
    train_args = dict(payload["args"])
    family = train_args.pop("model_family", "yolo")
    if family not in ("yolo", "rtdetr"):
        raise ValueError("Unsupported model family")
    allowed = {
        "data",
        "model",
        "project",
        "name",
        "device",
        "epochs",
        "batch",
        "imgsz",
        "workers",
        "single_cls",
        "classes",
        "time",
        "patience",
        "mosaic",
        "close_mosaic",
        "optimizer",
        "cos_lr",
        "amp",
        "multi_scale",
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_epochs",
        "warmup_momentum",
        "warmup_bias_lr",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "degrees",
        "translate",
        "scale",
        "fliplr",
        "erasing",
        "shear",
        "perspective",
        "dropout",
        "fraction",
        "rect",
        "box",
        "cls",
        "dfl",
        "pose",
        "kobj",
        "save_period",
        "val",
        "plots",
        "save",
        "resume",
        "pretrained",
        "deterministic",
        "seed",
        "cache",
    }
    unsupported = set(train_args) - allowed
    if unsupported:
        raise ValueError(
            f"Unsupported training arguments: {', '.join(sorted(unsupported))}"
        )
    import torch

    if train_args.get("device") == "auto":
        if torch.cuda.is_available():
            train_args["device"] = [0]
        elif torch.backends.mps.is_available():
            train_args["device"] = "mps"
        else:
            train_args["device"] = "cpu"
    validate_selected_device(torch, train_args.get("device"))
    weights_directory = payload["paths"]["weights_directory"]
    configure_weights_directory(weights_directory)
    import matplotlib

    matplotlib.use("Agg")
    from ultralytics import YOLO, RTDETR

    emit("training_started", total_epochs=train_args.get("epochs", 100))
    log_stream = EventLogStream("training_log")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout = log_stream
        sys.stderr = log_stream
        model = (RTDETR if family == "rtdetr" else YOLO)(train_args.pop("model"))
        attach_training_events(model, emit)
        train_args["verbose"] = False
        train_args["show"] = False
        model.train(**train_args)
    finally:
        log_stream.flush()
        sys.stdout, sys.stderr = original_stdout, original_stderr
    emit("training_completed", results="Training completed successfully")


def missing_export_packages(export_format: str, *, simplify: bool = True) -> list[str]:
    return [
        requirement
        for module, requirement in EXPORT_DEPENDENCIES[export_format]
        if simplify or export_format != "onnx" or module == "onnx"
        if importlib.util.find_spec(module) is None
    ]


def install_export_packages(packages: list[str]) -> None:
    emit(
        "export_log",
        message=f"Installing required packages: {', '.join(packages)}",
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "pip", "install", *packages],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if process.stdout is None:
        raise RuntimeError("Package installer output is unavailable")
    output_queue = queue.Queue()
    reader_done = threading.Event()
    recent_output = []

    def read_output() -> None:
        try:
            for line in process.stdout:
                output_queue.put(line)
        finally:
            reader_done.set()

    threading.Thread(target=read_output, daemon=True).start()
    deadline = time.monotonic() + 1800
    while (
        process.poll() is None
        or not reader_done.is_set()
        or not output_queue.empty()
    ):
        if time.monotonic() >= deadline:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            raise TimeoutError(
                "Package installation timed out after 30 minutes"
            )
        try:
            line = output_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if line.strip():
            cleaned_line = line.rstrip()
            recent_output.append(cleaned_line)
            recent_output = recent_output[-20:]
            emit("export_log", message=cleaned_line)
    if process.returncode != 0:
        if any("No module named pip" in line for line in recent_output):
            raise RuntimeError(
                "pip is not available in Training Python. Install pip "
                "manually before exporting."
            )
        raise RuntimeError(
            "Failed to install required export packages "
            f"(exit code {process.returncode})"
        )


def run_export(payload: dict) -> None:
    export_format = payload["args"].get("format", "onnx")
    if export_format not in EXPORT_DEPENDENCIES:
        raise ValueError(f"Unsupported export format: {export_format}")
    weights_path = payload["paths"]["weights_path"]
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")
    emit("export_started", weights_path=weights_path, format=export_format)
    destination = payload["paths"].get("destination")
    if destination and export_format != "onnx":
        raise ValueError("A destination is only supported for ONNX conversion")
    missing = missing_export_packages(export_format, simplify=not destination)
    if missing and not payload["options"].get("auto_install_packages", True):
        command = f'"{sys.executable}" -m pip install ' + " ".join(missing)
        raise RuntimeError(
            f"Missing required packages: {', '.join(missing)}. Install them with: {command}"
        )
    if missing:
        install_export_packages(missing)
        remaining = missing_export_packages(export_format, simplify=not destination)
        if remaining:
            raise RuntimeError(
                f"Packages remain unavailable after installation: {', '.join(remaining)}"
            )
    from ultralytics import RTDETR, YOLO

    emit("export_log", message=f"Loading model from {weights_path}")
    log_stream = EventLogStream("export_log")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout = log_stream
        sys.stderr = log_stream
        if destination:
            import onnx

            target = Path(destination)
            if target.suffix.lower() != ".onnx":
                raise ValueError(f"Expected an .onnx output path: {target}")
            model_type = RTDETR if payload["args"].get("model_family") == "rtdetr" else YOLO
            settings = {"format": "onnx", "device": "cpu", "simplify": False}
            if payload["args"].get("imgsz"):
                settings["imgsz"] = payload["args"]["imgsz"]
            with tempfile.TemporaryDirectory(prefix="autolabel-onnx-") as staging:
                copied_weights = Path(staging) / Path(weights_path).name
                shutil.copyfile(weights_path, copied_weights)
                result = Path(model_type(str(copied_weights)).export(**settings))
                onnx.checker.check_model(str(result))
                descriptor, filename = tempfile.mkstemp(prefix=".autolabel-onnx-", dir=target.parent)
                pending = Path(filename)
                try:
                    with os.fdopen(descriptor, "wb") as output, result.open("rb") as source:
                        shutil.copyfileobj(source, output)
                    os.replace(pending, target)
                finally:
                    pending.unlink(missing_ok=True)
            result = target
        else:
            result = YOLO(weights_path).export(format=export_format)
    finally:
        log_stream.flush()
        sys.stdout, sys.stderr = original_stdout, original_stderr
    exported_path = (
        os.fspath(result)
        if isinstance(result, (str, os.PathLike))
        else str(result)
    )
    if not exported_path or not os.path.exists(exported_path):
        fallback = str(Path(weights_path).with_suffix(f".{export_format}"))
        exported_path = fallback if os.path.exists(fallback) else exported_path
    if not exported_path or not os.path.exists(exported_path):
        raise FileNotFoundError(
            "Export completed but its output was not found"
        )
    emit("export_completed", exported_path=exported_path, format=export_format)


def load_payload(payload_path: str) -> dict:
    with open(payload_path, encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("version") != PROTOCOL_VERSION:
        raise ValueError("Unsupported worker payload version")
    if payload.get("action") not in {"probe", "train", "export"}:
        raise ValueError("Unsupported worker action")
    unsupported_top_level = set(payload) - {
        "version",
        "action",
        "args",
        "paths",
        "options",
    }
    if unsupported_top_level:
        raise ValueError("Unsupported worker payload fields")
    for key in ("args", "paths", "options"):
        if not isinstance(payload.get(key), dict):
            raise ValueError(f"Worker payload field '{key}' must be an object")
    action = payload["action"]
    allowed_fields = {
        "probe": {"args": set(), "paths": set(), "options": set()},
        "train": {
            "args": set(payload["args"]),
            "paths": {"data_directory", "weights_directory"},
            "options": {"auto_install_packages"},
        },
        "export": {
            "args": {"format", "model_family", "imgsz"},
            "paths": {"data_directory", "weights_path", "destination"},
            "options": {"auto_install_packages"},
        },
    }[action]
    for key, allowed in allowed_fields.items():
        if set(payload[key]) - allowed:
            raise ValueError(f"Unsupported {action} payload field in '{key}'")
    required_paths = {
        "probe": set(),
        "train": {"weights_directory"},
        "export": {"weights_path"},
    }[action]
    if not required_paths.issubset(payload["paths"]):
        raise ValueError(f"Missing required paths for {action}")
    if any(not isinstance(value, str) for value in payload["paths"].values()):
        raise ValueError("Worker paths must be strings")
    auto_install = payload["options"].get("auto_install_packages")
    if auto_install is not None and not isinstance(auto_install, bool):
        raise ValueError("auto_install_packages must be a boolean")
    return payload


def main(payload_path: str) -> None:
    action = "environment"
    try:
        payload = load_payload(payload_path)
        action = payload["action"]
        if action == "probe":
            run_probe()
        elif action == "train":
            run_train(payload)
        else:
            run_export(payload)
    except Exception as exc:
        event = {
            "probe": "environment_error",
            "train": "training_error",
            "export": "export_error",
        }.get(action, "environment_error")
        emit(
            event,
            error=str(exc),
            exception_type=type(exc).__name__,
            traceback=traceback.format_exc(),
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: worker.py <payload.json>")
    main(sys.argv[1])
