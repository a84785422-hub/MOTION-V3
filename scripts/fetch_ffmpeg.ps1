<#
.SYNOPSIS
  Downloads a static FFmpeg build (ffmpeg.exe + ffprobe.exe) into vendor/ffmpeg.

.DESCRIPTION
  The Windows build bundles these two executables. The default sources are GPL builds that
  include libx264, libx265, ProRes (prores_ks), FFV1 and NVENC support. Pass -Url to use
  your own build (a .zip containing ffmpeg.exe and ffprobe.exe somewhere inside).
#>
param(
    [string[]]$Url = @(
        "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
        "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    ),
    [string]$Destination = "vendor/ffmpeg"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # makes Invoke-WebRequest much faster

$root = Split-Path -Parent $PSScriptRoot
$dest = Join-Path $root $Destination

if ((Test-Path (Join-Path $dest "ffmpeg.exe")) -and (Test-Path (Join-Path $dest "ffprobe.exe"))) {
    Write-Host "FFmpeg already present in $dest"
    exit 0
}

New-Item -ItemType Directory -Force -Path $dest | Out-Null
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("vmc-ffmpeg-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

$zip = Join-Path $tmp "ffmpeg.zip"
$downloaded = $false
foreach ($u in $Url) {
    for ($attempt = 1; $attempt -le 3 -and -not $downloaded; $attempt++) {
        try {
            Write-Host "Downloading $u (attempt $attempt)"
            Invoke-WebRequest -Uri $u -OutFile $zip -UseBasicParsing
            $downloaded = $true
        } catch {
            Write-Warning "Download failed: $($_.Exception.Message)"
            Start-Sleep -Seconds (3 * $attempt)
        }
    }
    if ($downloaded) { break }
}
if (-not $downloaded) { throw "Could not download FFmpeg from any source." }

$extract = Join-Path $tmp "x"
Expand-Archive -Path $zip -DestinationPath $extract -Force
$ffmpeg = Get-ChildItem -Path $extract -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
if (-not $ffmpeg) { throw "ffmpeg.exe not found in the downloaded archive." }
$bin = $ffmpeg.DirectoryName
$ffprobe = Join-Path $bin "ffprobe.exe"
if (-not (Test-Path $ffprobe)) { throw "ffprobe.exe not found next to ffmpeg.exe." }

Copy-Item $ffmpeg.FullName (Join-Path $dest "ffmpeg.exe") -Force
Copy-Item $ffprobe (Join-Path $dest "ffprobe.exe") -Force
$license = Get-ChildItem -Path $extract -Recurse -Filter "LICENSE*" | Select-Object -First 1
if ($license) { Copy-Item $license.FullName (Join-Path $dest "LICENSE.txt") -Force }

Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue

& (Join-Path $dest "ffmpeg.exe") -hide_banner -version | Select-Object -First 1
Write-Host "FFmpeg installed in $dest"
