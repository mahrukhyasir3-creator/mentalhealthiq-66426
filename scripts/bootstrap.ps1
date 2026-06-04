Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

$DemographicPath = "data/raw/demographic.csv"
$QuestionnairePath = "data/raw/questionnaire.csv"

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Error ".venv is missing. Run .\scripts\setup.ps1 first."
    exit 1
}

if (-not (Test-Path $DemographicPath) -or -not (Test-Path $QuestionnairePath)) {
    Write-Error "Raw data files are missing. Place demographic.csv and questionnaire.csv in data/raw, then run this script again."
    exit 1
}

. ".\.venv\Scripts\Activate.ps1"

Write-Host "Bootstrapping preprocessing, model training, and fairness artifacts..."
python scripts/bootstrap_ml.py
