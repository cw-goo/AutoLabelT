param(
    [string]$Python = "",
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA "Programs/AutoLabel"),
    [switch]$NoShortcut
)

$ErrorActionPreference = "Stop"
$wheel = Get-ChildItem -LiteralPath $PSScriptRoot -Filter "autolabel-*.whl" -File | Select-Object -First 1
if (-not $wheel) { throw "AutoLabel wheel is missing from this folder" }

if (-not $Python) {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        $candidate = & py -3.11 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0) { $Python = $candidate }
    }
    if (-not $Python) { $Python = "python" }
}
& $Python -c "import sys; assert sys.version_info[:2] == (3, 11) and sys.maxsize > 2**32, 'Use 64-bit Python 3.11'"
if ($LASTEXITCODE -ne 0) { throw "Install 64-bit Python 3.11, then rerun with -Python path-to-python.exe" }

$InstallDir = [System.IO.Path]::GetFullPath($InstallDir)
$venvPython = Join-Path $InstallDir "Scripts/python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    & $Python -m venv $InstallDir
    if ($LASTEXITCODE -ne 0) { throw "Could not create virtual environment" }
}
& $venvPython -m pip install --upgrade $wheel.FullName
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed; check network access and disk space" }
& $venvPython -c "import autolabel.workbench, PyQt6, cv2, torch, ultralytics; from importlib.resources import files; assert files('autolabel').joinpath('assets/logo.svg').is_file(); assert files('autolabel.workbench').joinpath('configs/autolabel_config.yaml').is_file(); from autolabel.workbench.resources import resources"
if ($LASTEXITCODE -ne 0) { throw "Installation verification failed" }

& (Join-Path $PSScriptRoot "register-command.ps1") -Python $venvPython

$launcherPath = Join-Path $InstallDir "AutoLabel.cmd"
@"
@echo off
"$venvPython" -m autolabel %*
"@ | Set-Content -LiteralPath $launcherPath -Encoding ASCII

if (-not $NoShortcut) {
    $startMenu = Join-Path $env:APPDATA "Microsoft/Windows/Start Menu/Programs"
    $shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $startMenu "AutoLabel.lnk"))
    $shortcut.TargetPath = (Join-Path $InstallDir "Scripts/pythonw.exe")
    $shortcut.Arguments = "-m autolabel"
    $shortcut.WorkingDirectory = $InstallDir
    $shortcut.Description = "AutoLabel video and image annotation"
    $shortcut.Save()
}
Write-Host "Installed AutoLabel to $InstallDir"
Write-Host "Open a new terminal and run autolabel, or use the Start menu."
