"""The registered Windows command works outside the repository/environment."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows command registration")
REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("without_file_hash", [False, True])
def test_registered_command_runs_from_any_directory_and_registration_is_repeatable(
        tmp_path, without_file_hash):
    commands = tmp_path / "command directory with spaces"
    unrelated = tmp_path / "outside project"
    unrelated.mkdir()
    script = REPO / "packaging" / "windows" / "register-command.ps1"
    if without_file_hash:
        wrapper = tmp_path / "without-file-hash.ps1"
        escaped_script = str(script).replace("'", "''")
        wrapper.write_text(
            "ConvertFrom-Json '{}' | Out-Null\n"
            "function Get-FileHash { throw 'Get-FileHash is unavailable in this host' }\n"
            f"& '{escaped_script}' @args\n",
            encoding="utf-8",
        )
        script = wrapper
    register = [
        "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
        str(script),
        "-Python", sys.executable, "-CommandDir", str(commands), "-NoPathUpdate",
    ]
    for _ in range(2):
        result = subprocess.run(register, cwd=unrelated, capture_output=True, timeout=60)
        assert result.returncode == 0, (result.stdout + result.stderr).decode(errors="replace")
    command = commands / "autolabelt.exe"
    legacy_command = commands / "autolabel.exe"
    assert command.is_file()
    assert not legacy_command.exists()
    expected_launcher = command.read_bytes()
    legacy_command.write_bytes(expected_launcher)
    command.write_bytes(b"outdated launcher")
    result = subprocess.run(register, cwd=unrelated, capture_output=True, timeout=60)
    assert result.returncode == 0, (result.stdout + result.stderr).decode(errors="replace")
    assert command.read_bytes() == expected_launcher
    assert not legacy_command.exists()
    legacy_command.write_bytes(b"unrelated launcher")
    result = subprocess.run(register, cwd=unrelated, capture_output=True, timeout=60)
    assert result.returncode == 0, (result.stdout + result.stderr).decode(errors="replace")
    assert legacy_command.read_bytes() == b"unrelated launcher"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("CONDA_PREFIX", None)
    environment["PATH"] = str(commands) + os.pathsep + environment["PATH"]
    # cmd performs PATH resolution itself, rather than using the Python executable.
    version = subprocess.run(["cmd.exe", "/d", "/c", "autolabelt version"],
                             cwd=unrelated, env=environment, capture_output=True, timeout=30)
    assert version.returncode == 0, version.stderr.decode(errors="replace")
    assert version.stdout.decode().strip() == "0.1.0"
    help_result = subprocess.run([str(command), "--help"], cwd=unrelated,
                                 env=environment, capture_output=True, timeout=30)
    assert help_result.returncode == 0
    assert b"usage:" in help_result.stdout.lower()
    assert b"usage: autolabelt" in help_result.stdout.lower()
