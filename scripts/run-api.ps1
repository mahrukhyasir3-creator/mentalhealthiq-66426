Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

function Get-DotEnvValue {
    param(
        [string]$Name,
        [string]$Default
    )

    if (Test-Path ".env") {
        foreach ($Line in Get-Content ".env") {
            if ($Line -match "^\s*$Name\s*=\s*(.+?)\s*$") {
                return $Matches[1]
            }
        }
    }

    return $Default
}

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Error ".venv is missing. Run .\scripts\setup.ps1 first."
    exit 1
}

. ".\.venv\Scripts\Activate.ps1"

$HostName = Get-DotEnvValue "API_HOST" "127.0.0.1"
$Port = Get-DotEnvValue "API_PORT" "8000"

Write-Host "Starting API at http://localhost:$Port"
python -m uvicorn mentalhealthiq.api:app --reload --host $HostName --port $Port
