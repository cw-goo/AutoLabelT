"""Build a portable source wheel bundle for installation on Linux."""

from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tomllib


def main():
    if sys.version_info[:2] != (3, 11):
        raise SystemExit("Build with Python 3.11")

    root = Path(__file__).resolve().parents[2]
    with (root / "pyproject.toml").open("rb") as stream:
        version = tomllib.load(stream)["project"]["version"]
    output = root / "artifacts" / "release"
    stage = output / f"AutoLabel-{version}-Linux"
    stage.mkdir(parents=True, exist_ok=True)
    for old_wheel in stage.glob("autolabel-*.whl"):
        old_wheel.unlink()

    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps",
         "--no-build-isolation", "--wheel-dir", str(stage), str(root)],
        check=True,
        cwd=root,
    )
    wheels = list(stage.glob("autolabel-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Expected exactly one AutoLabel wheel")

    shutil.copy2(Path(__file__).with_name("install.sh"), stage)
    shutil.copy2(Path(__file__).with_name("INSTALL.txt"), stage)
    shutil.copy2(root / "LICENSE", stage)
    shutil.copy2(root / "THIRD_PARTY_NOTICES.md", stage)

    archive = output / f"AutoLabel-{version}-Linux.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path in stage.iterdir():
            info = bundle.gettarinfo(str(path), arcname=f"{stage.name}/{path.name}")
            info.mode = 0o755 if path.name == "install.sh" else 0o644
            with path.open("rb") as stream:
                bundle.addfile(info, stream)
    print(f"Created {archive}")


if __name__ == "__main__":
    main()
