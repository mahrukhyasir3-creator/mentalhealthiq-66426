Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

Write-Host "Setting up MentalHealthIQ..."

try {
    $PythonVersion = python --version
    Write-Host "Found $PythonVersion"
} catch {
    Write-Error "Python is not available on PATH. Install Python 3.11+ and try again."
    exit 1
}

if (-not (Test-Path ".venv")) {
    Write-Host "Creating .venv..."
    python -m venv .venv
}

. ".\.venv\Scripts\Activate.ps1"

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Installing requirements.txt..."
pip install -r requirements.txt

if (-not (Test-Path ".env")) {
    Write-Host "Creating .env from .env.example..."
    Copy-Item ".env.example" ".env"
} else {
    Write-Host ".env already exists; leaving it unchanged."
}

$Folders = @(
    "data/raw",
    "data/processed",
    "data/models",
    "data/fairness_reports"
)

foreach ($Folder in $Folders) {
    New-Item -ItemType Directory -Force -Path $Folder | Out-Null
}

Write-Host ""
Write-Host "Setup complete."
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Place demographic.csv and questionnaire.csv in data/raw"
Write-Host "2. Run: .\scripts\bootstrap.ps1"
Write-Host "3. Run: .\scripts\run-all.ps1"
Write-Host ""
Write-Host "Manual API only: .\scripts\run-api.ps1"
Write-Host "Manual frontend only: .\scripts\run-frontend.ps1"
