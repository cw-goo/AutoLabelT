"""Application identity shared by the desktop and command-line entry points."""

from importlib.metadata import PackageNotFoundError, version
import os
import sys


APP_NAME = "AutoLabel"


def app_version():
    try:
        return version("autolabel")
    except PackageNotFoundError:
        return "0.1.0"


def startup_banner(use_color=False):
    border = "38;2;65;110;128"
    cyan = "38;2;80;224;211"
    amber = "38;2;255;185;105"
    white = "38;2;230;240;244"
    muted = "38;2;150;171;183"
    width = 54

    def tint(text, color):
        return f"\x1b[{color}m{text}\x1b[0m" if use_color else text

    def row(*parts):
        length = sum(len(text) for text, _ in parts)
        content = "".join(tint(text, color) for text, color in parts)
        return tint("  |  ", border) + content + " " * (width - length) + tint("  |", border)

    edge = tint("  +" + "-" * (width + 4) + "+", border)
    return "\n".join((
        edge,
        row(("[ A > ]  ", amber), (APP_NAME, cyan), (f"  v{app_version()}", white)),
        row(("", muted)),
        row(("VIDEO", cyan), ("  >  FRAMES  >  LABELS  >  DATASET", white)),
        row(("Intelligent video and image annotation", muted)),
        edge,
    ))


def write_startup_banner(stream=None):
    stream = stream or sys.stderr
    interactive = stream.isatty() and "NO_COLOR" not in os.environ
    interactive = interactive and os.environ.get("TERM") != "dumb"
    interactive = interactive and os.environ.get("CLICOLOR") != "0"
    if interactive and os.name == "nt":
        try:
            import ctypes
            import msvcrt

            handle = msvcrt.get_osfhandle(stream.fileno())
            mode = ctypes.c_ulong()
            kernel = ctypes.windll.kernel32
            interactive = bool(kernel.GetConsoleMode(handle, ctypes.byref(mode)))
            if interactive:
                interactive = bool(kernel.SetConsoleMode(handle, mode.value | 0x0004))
        except (OSError, AttributeError, ValueError):
            interactive = False
    stream.write("\n" + startup_banner(use_color=interactive) + "\n\n")
    stream.flush()


HELP_MESSAGE = """AutoLabel 智能视频与图像标注

启动：python -m autolabel
帮助：python -m autolabel --help
版本：python -m autolabel version
配置：python -m autolabel config

标注画布基于 AutoLabel；许可信息见 LICENSES/。
"""
