param(
    [Parameter(Mandatory = $true)]
    [string]$Python,
    [string]$CommandDir = (Join-Path $env:USERPROFILE ".local/bin"),
    [switch]$NoPathUpdate
)

$ErrorActionPreference = "Stop"

function Get-LauncherHash([string]$Path) {
    # Get-FileHash may be unavailable when Windows PowerShell inherits a pwsh module path.
    $algorithm = [Security.Cryptography.SHA256]::Create()
    $stream = $null
    try {
        $stream = [IO.File]::OpenRead($Path)
        return [Convert]::ToBase64String($algorithm.ComputeHash($stream))
    } finally {
        if ($null -ne $stream) { $stream.Dispose() }
        $algorithm.Dispose()
    }
}

$CommandDir = [IO.Path]::GetFullPath([Environment]::ExpandEnvironmentVariables($CommandDir))
$legacyCommand = Join-Path $CommandDir 'autolabel.exe'
$probe = @'
import json
import sys
import sysconfig
from importlib.metadata import distribution
from pathlib import Path
from zipfile import BadZipFile, ZipFile

if not (3, 11) <= sys.version_info[:2] < (3, 13):
    raise SystemExit('AutoLabelT requires Python 3.11 or 3.12')
entries = distribution('autolabel').entry_points
launcher = Path(sysconfig.get_path('scripts')) / 'autolabelt.exe'
if not launcher.is_file() or not any(
    entry.group == 'console_scripts' and entry.name == 'autolabelt'
    and entry.value == 'autolabel.desktop:main' for entry in entries
):
    raise SystemExit('The autolabelt launcher is missing. Reinstall AutoLabelT first.')

# pip appends a ZIP containing the entry point to its Windows executable.
try:
    with ZipFile(sys.argv[1]) as legacy:
        script = legacy.read('__main__.py')
        remove_legacy = b'from autolabel.desktop import main' in script and b'sys.exit(main())' in script
except (OSError, BadZipFile, KeyError, RuntimeError):
    remove_legacy = False

print(json.dumps({'python': sys.executable, 'launcher': str(launcher), 'remove_legacy': remove_legacy}))
'@
$result = & $Python -c $probe $legacyCommand
if ($LASTEXITCODE -ne 0) { throw "Could not find the installed autolabelt command in $Python" }
$installation = ($result -join "`n") | ConvertFrom-Json

# Use pip's native console launcher. Its embedded interpreter path remains valid
# when copied, including paths containing spaces or non-ASCII characters.
& $installation.launcher version
if ($LASTEXITCODE -ne 0) { throw "The autolabelt command failed its startup check" }
New-Item -ItemType Directory -Force -Path $CommandDir | Out-Null
$commandPath = Join-Path $CommandDir 'autolabelt.exe'
if ($commandPath -ne $installation.launcher) {
    $copyRequired = -not (Test-Path -LiteralPath $commandPath)
    if (-not $copyRequired) {
        $copyRequired = (Get-LauncherHash $commandPath) -ne (Get-LauncherHash $installation.launcher)
    }
    if ($copyRequired) {
        Copy-Item -LiteralPath $installation.launcher -Destination $commandPath -Force
    }
}
& $commandPath version
if ($LASTEXITCODE -ne 0) { throw "The registered command failed its startup check: $commandPath" }
if ($installation.remove_legacy -and (Test-Path -LiteralPath $legacyCommand -PathType Leaf)) {
    Remove-Item -LiteralPath $legacyCommand
}
Write-Host "Registered: $commandPath"

function Test-CommandDirectoryInPath([string]$PathValue) {
    foreach ($entry in ($PathValue -split ';')) {
        if (-not $entry.Trim()) { continue }
        $expanded = [Environment]::ExpandEnvironmentVariables($entry.Trim().Trim('"'))
        try { $expanded = [IO.Path]::GetFullPath($expanded).TrimEnd('\', '/') }
        catch { continue }
        if ([string]::Equals($expanded, $CommandDir.TrimEnd('\', '/'),
                            [StringComparison]::OrdinalIgnoreCase)) { return $true }
    }
    return $false
}

if (-not $NoPathUpdate) {
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    if (-not (Test-CommandDirectoryInPath ($machinePath + ';' + $userPath))) {
        $newUserPath = if ([string]::IsNullOrWhiteSpace($userPath)) {
            $CommandDir
        } else {
            $userPath.TrimEnd(';') + ';' + $CommandDir
        }
        [Environment]::SetEnvironmentVariable("Path", $newUserPath, "User")
    }
    if (-not (Test-CommandDirectoryInPath $env:Path)) {
        $env:Path = $CommandDir + ';' + $env:Path
    }
}
Write-Host "Python environment: $($installation.python)"
Write-Host "Open a new terminal and run: autolabelt"
