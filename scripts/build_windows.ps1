<#
.SYNOPSIS
  Builds the Windows x64 executables locally and zips them.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1

  Result: dist\VideoMotionControl\ (VideoMotionControl.exe + vmc.exe)
          dist\VideoMotionControl-windows-x64.zip
#>
param(
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Exe $Arguments" }
}

# 64-bit Python is required for a 64-bit executable.
$bits = (& python -c "import struct; print(struct.calcsize('P') * 8)").Trim()
if ($bits -ne "64") { throw "A 64-bit Python (x64) is required, found $bits-bit." }

if (-not (Test-Path ".venv")) { Invoke-Checked "python" @("-m", "venv", ".venv") }
$py = Join-Path (Get-Location) ".venv\Scripts\python.exe"

Invoke-Checked $py @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Checked $py @("-m", "pip", "install", "-r", "requirements.txt", "-r", "requirements-dev.txt")

& "$PSScriptRoot\fetch_ffmpeg.ps1"

if (-not $SkipTests) { Invoke-Checked $py @("-m", "pytest", "-q") }

Invoke-Checked $py @("-m", "PyInstaller", "--noconfirm", "--clean", "vmc.spec")

$zip = "dist\VideoMotionControl-windows-x64.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path "dist\VideoMotionControl" -DestinationPath $zip
Write-Host ""
Write-Host "Done:"
Write-Host "  dist\VideoMotionControl\VideoMotionControl.exe   (GUI)"
Write-Host "  dist\VideoMotionControl\vmc.exe                  (CLI)"
Write-Host "  $zip"
