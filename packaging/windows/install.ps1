param(
    [string]$Python = "",
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA "Programs/AutoLabel"),
    [switch]$NoShortcut
)

$ErrorActionPreference = "Stop"
$wheel = Get-ChildItem -LiteralPath $PSScriptRoot -Filter "autolabel-*.whl" -File | Select-Object -First 1
if (-not $wheel) { throw "AutoLabelT wheel is missing from this folder" }

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
$scripts = Join-Path $InstallDir 'Scripts'
if (-not (Test-Path -LiteralPath (Join-Path $scripts 'autolabelt.exe') -PathType Leaf) -or
    (Test-Path -LiteralPath (Join-Path $scripts 'autolabel.exe') -PathType Leaf)) {
    & $venvPython -m pip install --force-reinstall --no-deps $wheel.FullName
    if ($LASTEXITCODE -ne 0) { throw "Could not update the AutoLabelT launcher" }
}
& $venvPython -c "import autolabel.workbench, PyQt6, cv2, torch, ultralytics, onnx, onnxruntime; from importlib.resources import files; assert files('autolabel').joinpath('assets/logo.png').is_file(); assert files('autolabel.workbench').joinpath('configs/autolabel_config.yaml').is_file(); from autolabel.workbench.resources import resources"
if ($LASTEXITCODE -ne 0) { throw "Installation verification failed" }

& (Join-Path $PSScriptRoot "register-command.ps1") -Python $venvPython

$legacyLauncher = Join-Path $InstallDir 'AutoLabel.cmd'
if (Test-Path -LiteralPath $legacyLauncher -PathType Leaf) {
    $legacyContents = ([string](Get-Content -LiteralPath $legacyLauncher -Raw)).Trim() -replace '\r\n', "`n"
    if ($legacyContents -eq "@echo off`n`"$venvPython`" -m autolabel %*") {
        Remove-Item -LiteralPath $legacyLauncher
    }
}

if (-not $NoShortcut) {
    $startMenu = Join-Path $env:APPDATA "Microsoft/Windows/Start Menu/Programs"
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut((Join-Path $startMenu "AutoLabelT.lnk"))
    $shortcut.TargetPath = Join-Path $scripts 'pythonw.exe'
    $shortcut.Arguments = "-m autolabel"
    $shortcut.WorkingDirectory = $InstallDir
    $shortcut.Description = "AutoLabelT desktop workbench for annotation and visual model training"
    $shortcut.Save()
    foreach ($name in ('AutoLabel.lnk', 'LabelTrain.lnk')) {
        $legacyPath = Join-Path $startMenu $name
        if (Test-Path -LiteralPath $legacyPath -PathType Leaf) {
            $legacyShortcut = $shell.CreateShortcut($legacyPath)
            if ($legacyShortcut.TargetPath -eq $shortcut.TargetPath -and $legacyShortcut.Arguments -eq $shortcut.Arguments) {
                Remove-Item -LiteralPath $legacyPath
            }
        }
    }
}
Write-Host "Installed AutoLabelT to $InstallDir"
Write-Host "Open a new terminal and run autolabelt, or use the Start menu."
