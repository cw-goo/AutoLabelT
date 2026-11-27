"""Catch deleted source files retained by incremental package builds."""
import io
import tarfile
import zipfile

import pytest

from tools.build_distribution import BUILD_DIRECTORIES, BUILD_FILES, copy_build_inputs
from tools.check_distribution import audit_distribution


def make_wheel(root, extras=None, changed=None):
    wheel = root / "test.whl"
    files = {"autolabelt/__init__.py": b"", "autolabelt/assets/logo.png": b"current logo"}
    for name, content in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, content in {**files, **(extras or {}), **(changed or {})}.items():
            archive.writestr(name, content)
        for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
            archive.writestr(f"test.dist-info/licenses/{name}", b"license")
    return wheel


def test_current_wheel_passes(tmp_path):
    wheel = make_wheel(tmp_path)
    assert audit_distribution(wheel, root=tmp_path)["errors"] == []


def test_wheel_rejects_legacy_package(tmp_path):
    name = "autolabel/__init__.py"
    wheel = make_wheel(tmp_path, extras={name: b""})
    assert f"Obsolete namespace in wheel: {name}" in audit_distribution(wheel, root=tmp_path)["errors"]


@pytest.mark.parametrize("name", ["autolabelt/assets/logo.svg",
                                  "autolabelt/workbench/resources/images/icon.png",
                                  "autolabelt/__pycache__/old.pyc"])
def test_wheel_rejects_deleted_assets_and_cache(tmp_path, name):
    wheel = make_wheel(tmp_path, extras={name: b"leftover"})
    assert f"Unexpected application file in wheel: {name}" in audit_distribution(wheel, root=tmp_path)["errors"]


def test_wheel_rejects_old_contents_at_current_path(tmp_path):
    name = "autolabelt/assets/logo.png"
    wheel = make_wheel(tmp_path, changed={name: b"old logo"})
    assert f"Stale application file in wheel: {name}" in audit_distribution(wheel, root=tmp_path)["errors"]


def test_sdist_rejects_deleted_assets_and_local_material(tmp_path):
    wheel = make_wheel(tmp_path)
    (tmp_path / "environment.yml").write_text("name: autolabelt\n", encoding="utf-8")
    sdist = tmp_path / "test.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        for name in ("autolabelt/__init__.py", "autolabelt/assets/logo.png",
                     "autolabelt/assets/logo.svg", "SoftwareCopyright/private.md"):
            entry = tarfile.TarInfo("test-0.1/" + name)
            entry.size = 4
            archive.addfile(entry, io.BytesIO(b"test"))
    errors = audit_distribution(wheel, sdist, root=tmp_path)["errors"]
    assert "Unexpected application file in source archive: autolabelt/assets/logo.svg" in errors
    assert "Local file leaked into source archive: SoftwareCopyright/private.md" in errors
    assert "Missing source archive file: environment.yml" in errors


def test_clean_source_copy_excludes_old_builds_and_local_data(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    for name in BUILD_FILES:
        (root / name).write_text("source", encoding="utf-8")
    for name in BUILD_DIRECTORIES:
        (root / name).mkdir()
    for name in ("autolabelt/__init__.py", "autolabelt/__pycache__/old.pyc",
                 "build/lib/autolabelt/deleted.py", "artifacts/private.json",
                 "SoftwareCopyright/private.md", "autolabelt.egg-info/SOURCES.txt"):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("test", encoding="utf-8")
    copied = tmp_path / "clean"
    copy_build_inputs(root, copied)
    assert (copied / "autolabelt/__init__.py").is_file()
    assert (copied / "environment.yml").read_bytes() == (root / "environment.yml").read_bytes()
    assert not (copied / "autolabelt/__pycache__").exists()
    assert not (copied / "build").exists()
    assert not (copied / "artifacts").exists()
    assert not (copied / "SoftwareCopyright").exists()
    assert not (copied / "autolabelt.egg-info").exists()
