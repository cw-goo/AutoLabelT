"""Audit a built wheel and source archive against the current source assets."""
from pathlib import Path
import argparse
import hashlib
import json
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--sdist", type=Path)
    args = parser.parse_args()
    errors = []
    with zipfile.ZipFile(args.wheel) as archive:
        names = set(archive.namelist())
    source = [p.relative_to(ROOT).as_posix() for p in (ROOT / "autolabel").rglob("*")
              if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    errors.extend(f"Missing application file: {name}" for name in source if name not in names)
    with zipfile.ZipFile(args.wheel) as archive:
        errors.extend(f"Stale application file in wheel: {name}" for name in source if name in names
                      and hashlib.sha256(archive.read(name)).digest()
                      != hashlib.sha256((ROOT / name).read_bytes()).digest())
    errors.extend(f"Obsolete namespace in wheel: {name}" for name in names if "anylabel" in name.lower())
    for basename in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        if not any(name.endswith("/licenses/" + basename) for name in names):
            errors.append(f"Missing wheel license/notice: {basename}")
    sdist_count = None
    if args.sdist:
        with tarfile.open(args.sdist) as archive:
            members = [member.name.split("/", 1)[-1] for member in archive.getmembers() if member.isfile()]
        sdist_count = len(members)
        errors.extend(f"Missing source archive file: {name}" for name in source if name not in members)
        for name in members:
            path = Path(name)
            if any(part in {"artifacts", "build", ".git", "__pycache__", "软件著作权文档"} for part in path.parts):
                errors.append(f"Local file leaked into source archive: {name}")
            if path.suffix in {".pt", ".pth", ".onnx", ".safetensors", ".pyc"}:
                errors.append(f"Weight/cache leaked into source archive: {name}")
    print(json.dumps({"application_files": len(source), "wheel_files": len(names),
                      "sdist_files": sdist_count, "errors": errors}, indent=2))
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
