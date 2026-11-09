param(
    [Parameter(Mandatory = $true)]
    [string]$Python,
    [string]$CommandDir = (Join-Path $env:USERPROFILE ".local/bin"),
    [switch]$NoPathUpdate
)

$ErrorActionPreference = "Stop"
$probe = @'
import json
import sys
import sysconfig
from importlib.metadata import distribution
from pathlib import Path

if not (3, 11) <= sys.version_info[:2] < (3, 13):
    raise SystemExit('AutoLabel requires Python 3.11 or 3.12')
entries = distribution('autolabel').entry_points
if not any(e.group == 'console_scripts' and e.name == 'autolabel'
           and e.value == 'autolabel.desktop:main' for e in entries):
    raise SystemExit('Install AutoLabel in this Python environment first')
launcher = Path(sysconfig.get_path('scripts')) / 'autolabel.exe'
if not launcher.is_file():
    raise SystemExit('AutoLabel launcher is missing; reinstall AutoLabel first')
print(json.dumps({'python': sys.executable, 'launcher': str(launcher)}))
'@
$result = & $Python -c $probe
if ($LASTEXITCODE -ne 0) { throw "Could not find the installed AutoLabel command in $Python" }
$installation = ($result -join "`n") | ConvertFrom-Json

# Use pip's native console launcher. Its embedded interpreter path remains valid
# when copied, including paths containing spaces or non-ASCII characters.
& $installation.launcher version
if ($LASTEXITCODE -ne 0) { throw "The AutoLabel command failed its startup check" }
$CommandDir = [IO.Path]::GetFullPath([Environment]::ExpandEnvironmentVariables($CommandDir))
New-Item -ItemType Directory -Force -Path $CommandDir | Out-Null
$commandPath = Join-Path $CommandDir "autolabel.exe"
if ($commandPath -ne $installation.launcher) {
    $copyRequired = -not (Test-Path -LiteralPath $commandPath)
    if (-not $copyRequired) {
        $copyRequired = (Get-FileHash -LiteralPath $commandPath).Hash -ne
                        (Get-FileHash -LiteralPath $installation.launcher).Hash
    }
    if ($copyRequired) {
        Copy-Item -LiteralPath $installation.launcher -Destination $commandPath -Force
    }
}
& $commandPath version
if ($LASTEXITCODE -ne 0) { throw "The registered command failed its startup check: $commandPath" }

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
Write-Host "Registered: $commandPath"
Write-Host "Python environment: $($installation.python)"
Write-Host "Open a new terminal and run: autolabel"
