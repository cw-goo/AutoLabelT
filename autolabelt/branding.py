"""Application identity shared by the desktop and command-line entry points."""

from importlib.metadata import PackageNotFoundError, version
import os
import sys


APP_NAME = "AutoLabelT"
LAUNCH_COMMAND = "autolabelt"
QT_APPLICATION_ID = "AutoLabel"  # Keep existing Qt settings and application data paths.
APP_SUMMARY = "视频与图像标注 · 可视化训练"


def app_version():
    try:
        return version("autolabelt")
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
        row(("VIDEO", cyan), ("  >  FRAMES  >  LABELS  >  DATASET  >  TRAIN", white)),
        row(("From frames and annotations to trained models", muted)),
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


HELP_MESSAGE = f"""{APP_NAME}
{APP_SUMMARY}

启动：{LAUNCH_COMMAND}
帮助：{LAUNCH_COMMAND} --help
版本：{LAUNCH_COMMAND} version
配置：{LAUNCH_COMMAND} config

标注画布基于 X-AnyLabeling。许可信息见 LICENSE。
"""
