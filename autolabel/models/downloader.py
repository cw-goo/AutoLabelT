"""Cancelable, version-pinned downloads and local model manifests."""

from pathlib import Path
import hashlib
import json
import os
import shutil
import urllib.request

from .registry import get_model

CLIP_URL = ("https://openaipublic.azureedge.net/clip/models/"
            "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt")
CLIP_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"


def model_store():
    configured = os.environ.get("AUTOLABEL_MODEL_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".local" / "share" / "autolabel" / "models"


def model_path(model_id):
    spec = get_model(model_id)
    return model_store() / model_id / spec.revision


def model_weight_path(model_id):
    spec = get_model(model_id)
    location = model_path(model_id)
    if spec.backend == "ultralytics_world":
        return location / "yolov8s-worldv2.pt"
    if spec.weight_name:
        return location / spec.weight_name
    return location


def release_session_model(model_id):
    """Remove only weights owned by the current AutoLabel temporary session."""
    session_root = os.environ.get("AUTOLABEL_SESSION_MODEL_DIR")
    if not session_root or model_id == "local_yolo":
        return False
    root = Path(session_root).resolve()
    location = model_path(model_id).resolve()
    if root != model_store().resolve() or root not in location.parents:
        raise ValueError("模型路径不属于当前临时会话")
    if location.is_dir():
        shutil.rmtree(location)
        return True
    return False


def model_download_state(model_id):
    spec = get_model(model_id)
    location = model_path(model_id)
    manifest_path = location / "manifest.json"
    if not manifest_path.is_file():
        return "missing"
    if spec.backend == "ultralytics_world" and not all((location / name).is_file()
                                                        for name in ("yolov8s-worldv2.pt",
                                                                     "text_encoder/ViT-B-32.pt")):
        return "missing"
    if spec.weight_name and not (location / spec.weight_name).is_file():
        return "missing"
    if not spec.weight_url and not spec.weight_name and not any(location.glob("*.safetensors")):
        return "missing"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "missing"
    if manifest.get("model_id") != model_id or manifest.get("revision") != spec.revision:
        return "missing"
    return "ready" if manifest.get("status") == "ready" else "downloaded"


def mark_model_ready(model_id, path):
    spec = get_model(model_id)
    candidate = Path(path)
    manifest_path = (candidate.parent if candidate.is_file() else candidate) / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "ready"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                 encoding="utf-8")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as model_file:
        for block in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_model(model_id, progress=None, cancelled=None):
    spec = get_model(model_id)
    root = model_store()
    root.mkdir(parents=True, exist_ok=True)
    final_path = model_path(model_id)
    if model_download_state(model_id) != "missing":
        return final_path
    temporary = root / ".partial" / model_id / spec.revision
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=True)
    if spec.backend == "ultralytics_world" and (final_path / "yolov8s-worldv2.pt").is_file():
        shutil.copy2(final_path / "yolov8s-worldv2.pt", temporary / "yolov8s-worldv2.pt")
    try:
        if spec.weight_url:
            destination = temporary / (spec.weight_name or "yolov8s-worldv2.pt")
            files = [(spec.weight_url, destination, None)]
            if spec.backend == "ultralytics_world":
                encoder = temporary / "text_encoder" / "ViT-B-32.pt"
                files.append((CLIP_URL, encoder, CLIP_SHA256))
            digests = {}
            for url, target, expected_sha in files:
                target.parent.mkdir(parents=True, exist_ok=True)
                cached_encoder = Path.home() / ".cache" / "clip" / "ViT-B-32.pt"
                if target.name == "ViT-B-32.pt" and cached_encoder.is_file():
                    cached_digest = _sha256(cached_encoder)
                    if expected_sha and cached_digest.startswith(expected_sha):
                        shutil.copy2(cached_encoder, target)
                        digests[str(target.relative_to(temporary))] = cached_digest
                        if progress:
                            progress({"downloaded": target.stat().st_size,
                                      "total": target.stat().st_size,
                                      "file": target.name})
                        continue
                request = urllib.request.Request(url, headers={"User-Agent": "AutoLabel/0.1"})
                with urllib.request.urlopen(request, timeout=30) as response, target.open("wb") as output:
                    total = int(response.headers.get("Content-Length", 0))
                    received = 0
                    while True:
                        if cancelled and cancelled():
                            raise InterruptedError("用户取消了模型下载")
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                        received += len(block)
                        if progress:
                            progress({"downloaded": received, "total": total,
                                      "file": target.name})
                if target.stat().st_size < 1024 * 1024:
                    raise ValueError(f"下载文件太小，权重文件可能不完整：{target.name}")
                digest = _sha256(target)
                if expected_sha and not digest.startswith(expected_sha):
                    raise ValueError(f"SHA256 校验失败：{target.name}")
                digests[str(target.relative_to(temporary))] = digest
        else:
            from huggingface_hub import snapshot_download
            from tqdm.auto import tqdm

            class DownloadProgress(tqdm):
                def __init__(self, *args, **kwargs):
                    self.file_description = kwargs.get("desc", "model")
                    super().__init__(*args, **kwargs)

                def update(self, amount=1):
                    if cancelled and cancelled():
                        raise InterruptedError("用户取消了模型下载")
                    result = super().update(amount)
                    if progress:
                        progress({"downloaded": self.n, "total": self.total or 0,
                                  "file": self.file_description, "unit": "files"})
                    return result

            snapshot_download(
                repo_id=spec.source_url.removeprefix("https://huggingface.co/"),
                revision=spec.revision,
                local_dir=temporary,
                cache_dir=temporary / ".hf_cache",
                local_dir_use_symlinks=False,
                allow_patterns=([spec.weight_name] if spec.weight_name else
                                ["*.json", "*.txt", "*.safetensors", "*.py", "LICENSE", "README.md"]),
                max_workers=1,
                tqdm_class=DownloadProgress,
            )
            if cancelled and cancelled():
                raise InterruptedError("用户取消了模型下载")
            weights = ([temporary / spec.weight_name] if spec.weight_name else
                       sorted(temporary.glob("*.safetensors")))
            if not weights:
                raise ValueError("模型仓库没有安全张量权重文件 (.safetensors)")
            if not all(path.is_file() for path in weights):
                raise ValueError("授权仓库没有下载到预期模型权重；请确认访问权限与登录状态")
            digests = {path.name: _sha256(path) for path in weights}
            shutil.rmtree(temporary / ".hf_cache", ignore_errors=True)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if final_path.exists():
            shutil.rmtree(final_path)
        temporary.replace(final_path)
        manifest = {"model_id": spec.model_id, "title": spec.title,
                    "source": spec.source, "source_url": spec.source_url,
                    "license": spec.license, "revision": spec.revision,
                    "weights_sha256": digests, "status": "downloaded"}
        (final_path / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return final_path
    except Exception as error:
        shutil.rmtree(temporary, ignore_errors=True)
        if spec.backend == "ultralytics_sam3" and (
                "gated" in str(error).casefold() or "403" in str(error)):
            raise PermissionError(
                "SAM 权重需要先在 Hugging Face 模型页面申请访问并登录，"
                "获批后重试下载") from error
        raise
