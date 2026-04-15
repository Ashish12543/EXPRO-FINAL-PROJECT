$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$activatePath = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"

if (Test-Path $activatePath) {
    . $activatePath
} else {
    Write-Host "No .venv found. Running with system Python instead." -ForegroundColor Yellow
}

python .\smart_fall_activity_report.py
