"""Export platform icons from the shared PNG logo; rebuild Qt resources afterwards."""
from pathlib import Path
import argparse

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "autolabel/assets/logo.png")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "autolabel/workbench/resources/images")
    args = parser.parse_args()
    with Image.open(args.source) as image:
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
