"""Build and audit distributions in a fresh source copy, without stale build files."""
from pathlib import Path
import argparse
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
BUILD_FILES = ("pyproject.toml", "MANIFEST.in", "README.md", "LICENSE",
               "THIRD_PARTY_NOTICES.md", "AGENTS.md", "databuild.py")
BUILD_DIRECTORIES = ("autolabel", "tests", "tools", "docs", "packaging", ".github")


def copy_build_inputs(root, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for name in BUILD_FILES:
        shutil.copy2(root / name, destination / name)
    for name in BUILD_DIRECTORIES:
        shutil.copytree(root / name, destination / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo",
                                                      "*.egg-info", "build", "dist"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--sdist", action="store_true", help="Also build and audit the source archive")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    scratch = ROOT / "artifacts/build"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="distribution-", dir=scratch) as temporary:
        temporary = Path(temporary)
        source = temporary / "source"
        built = temporary / "dist"
        built.mkdir()
        copy_build_inputs(ROOT, source)
        subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps",
                        "--no-build-isolation", "--wheel-dir", str(built), str(source)], check=True)
        wheel, = built.glob("*.whl")
        check = [sys.executable, str(ROOT / "tools/check_distribution.py"), str(wheel)]
        if args.sdist:
            result = subprocess.run([sys.executable, "-c",
                                     "import sys; from setuptools.build_meta import build_sdist; build_sdist(sys.argv[1])",
                                     str(built)], cwd=source, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            if result.returncode:
                print(result.stdout)
            result.check_returncode()
            sdist, = built.glob("*.tar.gz")
            check.extend(["--sdist", str(sdist)])
        subprocess.run(check, check=True)
        output.mkdir(parents=True, exist_ok=True)
        for artifact in sorted(built.iterdir()):
            shutil.copy2(artifact, output / artifact.name)
            print("Created", output / artifact.name)


if __name__ == "__main__":
    main()
