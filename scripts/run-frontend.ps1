Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Error ".venv is missing. Run .\scripts\setup.ps1 first."
    exit 1
}

. ".\.venv\Scripts\Activate.ps1"

Write-Host "Starting frontend at http://localhost:5500"
python -m http.server 5500 -d frontend
