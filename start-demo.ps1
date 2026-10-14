param([int]$Port = 8080, [switch]$NativeHttps)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
$taskPython = '.venv\Scripts\python.exe'
& $taskPython -m pip install -r requirements-demo.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
$taskArgs = @('scripts/run_demo.py', '--port', "$Port")
if ($NativeHttps) { $taskArgs += '--native-https' }
& $taskPython @taskArgs
