Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Error ".venv is missing. Run .\scripts\setup.ps1 first."
    exit 1
}

$Docker = Get-Command docker -ErrorAction SilentlyContinue
if ($Docker) {
    try {
        docker compose version | Out-Null
        Write-Host "Starting MongoDB and Mongo Express with Docker Compose..."
        docker compose up -d mongodb mongo-express
    } catch {
        Write-Host "Docker Compose is not available or not running. Continuing without MongoDB."
    }
} else {
    Write-Host "Docker is not available. Continuing without MongoDB."
}

$ApiScript = Join-Path $PSScriptRoot "run-api.ps1"
$FrontendScript = Join-Path $PSScriptRoot "run-frontend.ps1"

Write-Host "Starting API and frontend in separate PowerShell windows..."
Start-Process powershell.exe -WorkingDirectory $ProjectRoot -WindowStyle Normal -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-File", $ApiScript)
Start-Process powershell.exe -WorkingDirectory $ProjectRoot -WindowStyle Normal -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-File", $FrontendScript)

Write-Host ""
Write-Host "MentalHealthIQ URLs:"
Write-Host "API:           http://localhost:8000"
Write-Host "Health:        http://localhost:8000/health"
Write-Host "Frontend:      http://localhost:5500"
Write-Host "Mongo Express: http://localhost:8081"
Write-Host ""
Write-Host "Close the API/frontend PowerShell windows to stop those services."
