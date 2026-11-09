"""The registered Windows command works outside the repository/environment."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows command registration")
REPO = Path(__file__).resolve().parents[1]


def test_registered_command_runs_from_any_directory_and_registration_is_repeatable(tmp_path):
    commands = tmp_path / "command directory with spaces"
    unrelated = tmp_path / "outside project"
    unrelated.mkdir()
    register = [
        "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
        str(REPO / "packaging" / "windows" / "register-command.ps1"),
        "-Python", sys.executable, "-CommandDir", str(commands), "-NoPathUpdate",
    ]
    for _ in range(2):
        result = subprocess.run(register, cwd=unrelated, capture_output=True, timeout=60)
        assert result.returncode == 0, (result.stdout + result.stderr).decode(errors="replace")
    command = commands / "autolabel.exe"
    assert command.is_file()
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("CONDA_PREFIX", None)
    environment["PATH"] = str(commands) + os.pathsep + environment["PATH"]
    # cmd performs PATH resolution itself, rather than using the Python executable.
    version = subprocess.run(["cmd.exe", "/d", "/c", "autolabel version"],
                             cwd=unrelated, env=environment, capture_output=True, timeout=30)
    assert version.returncode == 0, version.stderr.decode(errors="replace")
    assert version.stdout.decode().strip() == "0.1.0"
    help_result = subprocess.run([str(command), "--help"], cwd=unrelated,
                                 env=environment, capture_output=True, timeout=30)
    assert help_result.returncode == 0
    assert b"usage:" in help_result.stdout.lower()
    assert b"usage: autolabel" in help_result.stdout.lower()
