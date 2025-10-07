param(
    [string]$Port = "8000"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Push-Location $PSScriptRoot

if (-not (Test-Path .venv)) {
    python -m venv .venv
}

& .\.venv\Scripts\python -m pip install --upgrade pip
& .\.venv\Scripts\pip install -r .\requirements.txt
& .\.venv\Scripts\pip install fastapi uvicorn[standard] python-multipart

if (-not (Test-Path .\uploads)) { New-Item -ItemType Directory -Path .\uploads | Out-Null }

& .\.venv\Scripts\python -m uvicorn ML.server:app --host 0.0.0.0 --port $Port --reload

Pop-Location


