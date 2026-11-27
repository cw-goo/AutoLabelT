"""Rebuild bundled Qt resources after changing icons, presets or translations.

Requires Qt's rcc and lrelease build tools, not a running desktop application.
The generated resources.rcc is shipped so users do not need these build tools.
"""
from pathlib import Path
import argparse
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rcc", default=shutil.which("rcc"))
    parser.add_argument("--lrelease", default=shutil.which("lrelease"))
    args = parser.parse_args()
    if not args.rcc or not args.lrelease:
        parser.error("Qt rcc and lrelease are required; supply their absolute paths")
    root = Path(__file__).resolve().parents[1]
    resources = root / "autolabelt/workbench/resources"
    for source in sorted((resources / "translations").glob("*.ts")):
        subprocess.run([args.lrelease, str(source), "-qm", str(source.with_suffix(".qm"))], check=True)
    subprocess.run([
        args.rcc, "--binary", "--format-version", "2", "--compress-algo", "zlib",
        str(resources / "resources.qrc"), "-o", str(resources / "resources.rcc"),
    ], check=True)
    print("Built", resources / "resources.rcc")


if __name__ == "__main__":
    main()
