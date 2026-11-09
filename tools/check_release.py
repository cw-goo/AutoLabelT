"""Check source names, required assets and optional Git staging without changing files."""
from pathlib import Path
import argparse
import ast
import json
import re
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DIRECTORIES = {".git", "artifacts", "build", "dist", ".venv", "venv", "__pycache__",
                     ".pytest_cache", "软件著作权文档", "SoftwareCopyright", "datasets", "dataset", "data", "outputs",
                     "runs", "logs", "weights", "checkpoints", "pretrained", "model_cache"}
WEIGHT_SUFFIXES = {".pt", ".pth", ".onnx", ".engine", ".safetensors", ".ckpt", ".tflite", ".gguf"}


def local_path(path):
    if path.parts and path.parts[0] in LOCAL_DIRECTORIES:
        return True
    return any(part in {"__pycache__", "build", "dist", ".git"}
               or part.endswith(".egg-info") for part in path.parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-index", action="store_true", help="Also reject local files in the Git index")
    args = parser.parse_args()
    errors = []
    files = [p for p in ROOT.rglob("*") if p.is_file() and not local_path(p.relative_to(ROOT))]
    obsolete = re.compile(r"anylabel|xany", re.I)
    secret = re.compile(r"sk-[A-Za-z0-9_-]{24,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
    source_count = 0
    for path in files:
        relative = path.relative_to(ROOT)
        if any(obsolete.search(part) for part in relative.parts):
            errors.append(f"Obsolete path: {relative}")
        if path.suffix == ".py":
            source_count += 1
            try:
                tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(relative))
            except SyntaxError as error:
                errors.append(f"Syntax error: {relative}:{error.lineno}")
                continue
            for node in ast.walk(tree):
                imports = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                           else [node.module] if isinstance(node, ast.ImportFrom) and not node.level
                           and node.module else [])
                for name in imports:
                    if name == "anylabeling" or name.startswith("anylabeling."):
                        errors.append(f"Obsolete import: {relative}:{node.lineno}")
                    if name.startswith("autolabel."):
                        module = ROOT.joinpath(*name.split("."))
                        if not module.is_dir() and not module.with_suffix(".py").is_file():
                            errors.append(f"Unresolved local import: {relative}:{node.lineno}: {name}")
        if path.suffix in {".py", ".yaml", ".yml", ".toml", ".md", ".sh", ".ps1"}:
            for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
                if secret.search(line):
                    errors.append(f"Possible credential: {relative}:{number} (value hidden)")
    required = ["LICENSE", "THIRD_PARTY_NOTICES.md", "autolabel/assets/logo.svg", "autolabel/assets/logo.png",
                "autolabel/workbench/configs/autolabel_config.yaml",
                "autolabel/workbench/resources/resources.rcc", "docs/images/workbench.png"]
    errors.extend(f"Missing required file: {name}" for name in required if not (ROOT / name).is_file())
    qrc = ROOT / "autolabel/workbench/resources/resources.qrc"
    for element in ET.parse(qrc).findall(".//file"):
        if not (qrc.parent / element.text).is_file():
            errors.append(f"Missing Qt source asset: {element.text}")
    indexed_count = None
    if args.check_index:
        result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True)
        indexed = result.stdout.decode("utf-8").rstrip("\0").split("\0")
        indexed_count = len(indexed)
        for name in indexed:
            path = Path(name)
            if local_path(path) or path.suffix in WEIGHT_SUFFIXES:
                errors.append(f"Local/generated file staged: {name}")
            if any(obsolete.search(part) for part in path.parts):
                errors.append(f"Obsolete path staged: {name}")
            if not (ROOT / path).is_file():
                errors.append(f"Deleted file still staged: {name}")
        for path in files:
            name = path.relative_to(ROOT).as_posix()
            if path.suffix not in WEIGHT_SUFFIXES and name not in indexed:
                errors.append(f"Source/asset not staged: {name}")
    print(json.dumps({"source_files_checked": len(files), "python_files_parsed": source_count,
                      "indexed_files": indexed_count, "errors": errors}, ensure_ascii=True, indent=2))
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
