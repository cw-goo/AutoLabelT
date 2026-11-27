"""Register the packaged Qt resource bundle.

Rebuild with tools/build_resources.py after changing presets or translations.
"""
from pathlib import Path
from PyQt6 import QtCore

_path = str(Path(__file__).with_suffix(".rcc"))
_registered = False


def qInitResources():
    global _registered
    if not _registered:
        if not QtCore.QResource.registerResource(_path):
            raise RuntimeError("Cannot load AutoLabel Qt resources; rebuild with tools/build_resources.py")
        _registered = True


def qCleanupResources():
    global _registered
    if _registered:
        QtCore.QResource.unregisterResource(_path)
        _registered = False


qInitResources()
