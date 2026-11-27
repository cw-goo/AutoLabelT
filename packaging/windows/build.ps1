param(
    [string]$Python = "python",
    [string]$OutputDir = "artifacts/release"
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$output = [System.IO.Path]::GetFullPath((Join-Path $repo $OutputDir))

& $Python -c "import sys; assert sys.version_info[:2] == (3, 11) and sys.maxsize > 2**32, 'Use 64-bit Python 3.11'"
if ($LASTEXITCODE -ne 0) { throw "Python 3.11 x64 is required" }

$version = (& $Python -c "import sys, tomllib; print(tomllib.load(open(sys.argv[1], 'rb'))['project']['version'])" (Join-Path $repo "pyproject.toml") 2>&1)
if ($LASTEXITCODE -ne 0) { throw "Cannot read package version" }
$stage = Join-Path $output "AutoLabelT-$version-Windows"
New-Item -ItemType Directory -Force -Path $stage | Out-Null

Push-Location $repo
try {
    & $Python tools/build_distribution.py --output-dir $stage
    if ($LASTEXITCODE -ne 0) { throw "Wheel build failed" }
} finally {
    Pop-Location
}

Copy-Item (Join-Path $PSScriptRoot "install.ps1") $stage
Copy-Item (Join-Path $PSScriptRoot "register-command.ps1") $stage
Copy-Item (Join-Path $PSScriptRoot "INSTALL.txt") $stage
Copy-Item (Join-Path $repo "LICENSE") $stage
Copy-Item (Join-Path $repo "THIRD_PARTY_NOTICES.md") $stage
$archive = Join-Path $output "AutoLabelT-$version-Windows.zip"
if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive }
Compress-Archive -Path $stage -DestinationPath $archive
Write-Host "Created $archive"
