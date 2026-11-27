"""Audit a built wheel and source archive against the current source assets."""
from pathlib import Path
import argparse
import hashlib
import json
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def audit_distribution(wheel, sdist=None, root=ROOT):
    errors = []
    source = {p.relative_to(root).as_posix() for p in (root / "autolabelt").rglob("*")
              if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        for name in sorted(source & names):
            with archive.open(name) as built, (root / name).open("rb") as local:
                archive_hash = hashlib.file_digest(built, "sha256").digest()
                source_hash = hashlib.file_digest(local, "sha256").digest()
            if archive_hash != source_hash:
                errors.append(f"Stale application file in wheel: {name}")
    errors.extend(f"Missing application file: {name}" for name in sorted(source - names))
    errors.extend(f"Unexpected application file in wheel: {name}" for name in sorted(names)
                  if name.startswith("autolabelt/") and not name.endswith("/") and name not in source)
    errors.extend(f"Obsolete namespace in wheel: {name}" for name in names
                  if name.startswith("autolabel/") or "anylabel" in name.lower())
    for basename in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        if not any(name.endswith("/licenses/" + basename) for name in names):
            errors.append(f"Missing wheel license/notice: {basename}")
    sdist_count = None
    if sdist:
        with tarfile.open(sdist) as archive:
            members = {member.name.split("/", 1)[-1] for member in archive if member.isfile()}
        sdist_count = len(members)
        expected = source | ({"environment.yml"} if (root / "environment.yml").is_file() else set())
        errors.extend(f"Missing source archive file: {name}" for name in sorted(expected - members))
        errors.extend(f"Unexpected application file in source archive: {name}" for name in members
                      if name.startswith("autolabelt/") and name not in source)
        errors.extend(f"Obsolete namespace in source archive: {name}" for name in members
                      if name.startswith("autolabel/"))
        for name in members:
            path = Path(name)
            if any(part in {"artifacts", "build", ".git", "__pycache__", "软件著作权文档", "SoftwareCopyright"} for part in path.parts):
                errors.append(f"Local file leaked into source archive: {name}")
            if path.suffix in {".pt", ".pth", ".onnx", ".safetensors", ".pyc"}:
                errors.append(f"Weight/cache leaked into source archive: {name}")
    return {"application_files": len(source), "wheel_files": len(names),
            "sdist_files": sdist_count, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--sdist", type=Path)
    args = parser.parse_args()
    report = audit_distribution(args.wheel, args.sdist)
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(report["errors"]))


if __name__ == "__main__":
    main()
