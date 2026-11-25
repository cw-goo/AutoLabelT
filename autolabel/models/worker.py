"""Qt workers for model downloads, dependency setup, and inference checks."""

import subprocess
import sys
import tempfile
from pathlib import Path

import psutil
from PyQt6 import QtCore

from .adapters import create_detector
from .downloader import download_model, mark_model_ready, model_path


def _stop_install_process(process):
    # Windows venv launchers spawn an interpreter that inherits the log handle.
    # Stop and reap the entire tree before closing/removing the log directory.
    try:
        parent = psutil.Process(process.pid)
        processes = [parent, *parent.children(recursive=True)]
    except psutil.NoSuchProcess:
        process.wait()
        return
    for child in processes:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=3)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=3)
    if alive:
        raise TimeoutError("Model installer processes did not exit")
    process.wait()


class ModelWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(dict)
    message = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)
    completed = QtCore.pyqtSignal(dict)

    def __init__(self, action, model_id, path="", prompts="person", device="auto",
                 image_size=640, confidence=0.25, trust_remote_code=False, parent=None):
        super().__init__(parent)
        self.action = action
        self.model_id = model_id
        self.path = path
        self.prompts = prompts
        self.device = device
        self.image_size = image_size
        self.confidence = confidence
        self.trust_remote_code = trust_remote_code
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _check_cancelled(self):
        if self._cancel:
            raise InterruptedError("已取消模型任务")

    def run(self):
        try:
            self._check_cancelled()
            if self.action == "download":
                path = download_model(self.model_id, self.progress.emit,
                                      lambda: self._cancel)
                self.completed.emit({"path": str(path), "action": self.action})
            elif self.action == "test":
                import numpy as np
                path = self.path or str(model_path(self.model_id))
                detector = create_detector(self.model_id, path, self.prompts,
                                           self.device, self.image_size,
                                           self.confidence, self.trust_remote_code)
                self._check_cancelled()
                detector.predict(np.zeros((384, 384, 3), dtype=np.uint8))
                self._check_cancelled()
                mark_model_ready(self.model_id, path)
                self.completed.emit({"path": path, "action": self.action,
                                     "status": "ready"})
            elif self.action == "install_runtime":
                self._install_runtime()
            else:
                raise ValueError(f"未知模型任务：{self.action}")
        except Exception as error:
            self.failed.emit(str(error))

    def _install_runtime(self):
        if self.model_id.startswith("yolo26_") or self.model_id in ("sam3", "sam3_1"):
            commands = [[sys.executable, "-m", "pip", "install", "--no-deps",
                         "ultralytics>=8.4.0,<9"]]
            if self.model_id in ("sam3", "sam3_1"):
                commands.extend([
                    [sys.executable, "-m", "pip", "install", "ftfy>=6.1", "regex>=2023.0"],
                    [sys.executable, "-m", "pip", "install", "--no-deps",
                     "git+https://github.com/ultralytics/CLIP.git@7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40"],
                ])
            self.message.emit("正在安装所选模型运行组件；不会升级 PyTorch/CUDA。")
        elif self.model_id == "yolo_world":
            commands = [
                [sys.executable, "-m", "pip", "install", "ftfy>=6.1", "regex>=2023.0"],
                [sys.executable, "-m", "pip", "install", "--no-deps",
                 "git+https://github.com/ultralytics/CLIP.git@7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40"],
            ]
            self.message.emit("正在安装固定版本的 YOLO-World 文本编码依赖，不会升级 PyTorch/CUDA。")
        else:
            commands = [[sys.executable, "-m", "pip", "install",
                         "--upgrade-strategy", "only-if-needed", "transformers>=4.56,<5"]]
            if self.model_id == "florence2":
                commands.append([sys.executable, "-m", "pip", "install", "--no-deps", "einops>=0.8", "timm>=1.0"])
            self.message.emit("正在安装 Transformers 模型运行组件；PyTorch/CUDA 不在安装目标内。")
        for command in commands:
            self._check_cancelled()
            # A regular file avoids blocking cancellation on a silent stdout pipe.
            with tempfile.TemporaryDirectory(prefix="autolabel-install-") as temporary:
                output_path = Path(temporary) / "pip.log"
                output = output_path.open("wb")
                try:
                    process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
                except BaseException:
                    output.close()
                    raise
                offset = 0
                try:
                    while True:
                        self._check_cancelled()
                        try:
                            status = process.wait(timeout=0.1)
                        except subprocess.TimeoutExpired:
                            status = None
                        with output_path.open("rb") as reader:
                            reader.seek(offset)
                            chunk = reader.read()
                        offset += len(chunk)
                        if chunk:
                            self.message.emit(chunk.decode("utf-8", errors="replace").rstrip())
                        if status is not None:
                            break
                finally:
                    if process.poll() is None:
                        _stop_install_process(process)
                    output.close()
            self._check_cancelled()
            if status:
                raise RuntimeError(f"模型运行组件安装失败，pip exit code={status}")
        self.completed.emit({"action": self.action, "status": "installed"})
