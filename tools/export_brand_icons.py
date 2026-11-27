"""Render the vector master to PNG and platform icons; rebuild Qt resources afterwards."""
from pathlib import Path
import argparse
import os
import math

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def render_svg(source, output, size):
    from PyQt6 import QtCore, QtGui, QtSvg

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtGui.QGuiApplication.instance() or QtGui.QGuiApplication([])
    renderer = QtSvg.QSvgRenderer(str(source))
    if not renderer.isValid():
        raise ValueError(f"Invalid SVG logo: {source}")
    # Render at the SVG's native coordinate grid before scaling. This avoids
    # translucent seams between tightly fitted colour-gradient paths in Qt.
    native_size = max(1, math.ceil(max(renderer.viewBoxF().width(), renderer.viewBoxF().height())))
    image = QtGui.QImage(native_size, native_size, QtGui.QImage.Format.Format_ARGB32)
    image.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(image)
    renderer.render(painter)
    painter.end()
    if native_size != size:
        image = image.scaled(size, size, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                             QtCore.Qt.TransformationMode.SmoothTransformation)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not image.save(str(output)):
        raise OSError(f"Cannot save rendered logo: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "autolabelt/assets/logo.svg")
    parser.add_argument("--png-output", type=Path, default=ROOT / "autolabelt/assets/logo.png")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "autolabelt/workbench/resources/images")
    args = parser.parse_args()
    source = args.source
    if source.suffix.lower() == ".svg":
        if args.size < 256:
            parser.error("PNG export size must be at least 256 pixels")
        render_svg(source, args.png_output, args.size)
        source = args.png_output
    with Image.open(source) as image:
        if image.mode != "RGBA" or image.width != image.height:
            parser.error("The logo must be a square RGBA PNG")
        if image.getpixel((0, 0))[3] != 0:
            parser.error("The logo must retain its transparent background")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        image.save(args.output_dir / "icon.ico",
                   sizes=[(size, size) for size in (16, 24, 32, 48, 64, 128, 256)])
        image.save(args.output_dir / "icon.icns")
        print(f"Exported Windows ICO and macOS ICNS from {args.source} ({image.size}, {image.mode})")


if __name__ == "__main__":
    main()
