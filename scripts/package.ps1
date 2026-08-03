<# VRChat Agent Package Script #>
param(
    [string]$OutputDir = ".\deploy\"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$OutputDir = [System.IO.Path]::GetFullPath([System.IO.Path]::Combine($ProjectRoot, $OutputDir))

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  VRChat Agent Package Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Project: $ProjectRoot"
Write-Host "Output:  $OutputDir"
Write-Host ""

Write-Host "[1/4] Prepare output directory..." -ForegroundColor Yellow
if (Test-Path $OutputDir) {
    Write-Host "  Cleaning..."
    Remove-Item -Path $OutputDir -Recurse -Force -ErrorAction SilentlyContinue
}
$null = New-Item -ItemType Directory -Force -Path $OutputDir
Write-Host "  Ready." -ForegroundColor Green

function Copy-Filtered {
    param(
        [string]$Source, [string]$Dest,
        [string[]]$NoDirs, [string[]]$NoFiles
    )
    if (-not (Test-Path $Source)) {
        Write-Host "  SKIP (not found): $Source" -ForegroundColor DarkYellow
        return
    }
    if (-not (Test-Path $Dest)) {
        $null = New-Item -ItemType Directory -Force -Path $Dest
    }
    $items = Get-ChildItem -Path $Source -ErrorAction SilentlyContinue
    foreach ($it in $items) {
        if ($it.PSIsContainer) {
            $skip = $false
            foreach ($pat in $NoDirs) { if ($it.Name -like $pat) { $skip = $true; break } }
            if (-not $skip) {
                Copy-Filtered -Source $it.FullName -Dest (Join-Path $Dest $it.Name) -NoDirs $NoDirs -NoFiles $NoFiles
            }
        } else {
            $skip = $false
            foreach ($pat in $NoFiles) { if ($it.Name -like $pat) { $skip = $true; break } }
            if (-not $skip) {
                Copy-Item -Path $it.FullName -Destination (Join-Path $Dest $it.Name) -Force
            }
        }
    }
}

Write-Host "[2/4] Copy project files..." -ForegroundColor Yellow

Write-Host "  src/ ..."
Copy-Filtered -Source (Join-Path $ProjectRoot "src") -Dest (Join-Path $OutputDir "src") -NoDirs @("__pycache__", "*egg-info*") -NoFiles @("*.pyc", "*.pyo")

Write-Host "  node_bridge/ ..."
Copy-Filtered -Source (Join-Path $ProjectRoot "node_bridge") -Dest (Join-Path $OutputDir "node_bridge") -NoDirs @("node_modules", "logs", "*qa*") -NoFiles @("qa-*", "_qa*", "*.log")

Write-Host "  mappings/ ..."
Copy-Item -Path (Join-Path $ProjectRoot "mappings") -Destination (Join-Path $OutputDir "mappings") -Recurse -Force

Write-Host "  scripts/ ..."
Copy-Filtered -Source (Join-Path $ProjectRoot "scripts") -Dest (Join-Path $OutputDir "scripts") -NoDirs @("__pycache__") -NoFiles @("*.pyc", "*.pyo")

Write-Host "  Root files..."
$rootFiles = @("config.yaml", "pyproject.toml", "requirements.txt", "README.md")
foreach ($f in $rootFiles) {
    $src = Join-Path $ProjectRoot $f
    if (Test-Path $src) {
        Copy-Item -Path $src -Destination (Join-Path $OutputDir $f) -Force
        Write-Host "    + $f"
    } else { Write-Host "    - $f (not found)" -ForegroundColor DarkYellow }
}

Write-Host "[3/4] Generate deploy.bat..." -ForegroundColor Yellow
$bat = @("@echo off", "chcp 65001 >nul", "title VRChat Agent Deploy",
       "echo ========================================",
       "echo   VRChat Agent One-Click Deploy",
       "echo ========================================",
       "echo.", "echo Starting deployment...", "echo.",
       "powershell -ExecutionPolicy Bypass -File ""%~dp0scripts\deploy.ps1""",
       "pause")
Set-Content -Path (Join-Path $OutputDir "deploy.bat") -Value ($bat -join [System.Environment]::NewLine) -Encoding ASCII
Write-Host "  deploy.bat generated." -ForegroundColor Green

Write-Host "[4/4] Statistics..." -ForegroundColor Yellow
$count = 0; $size = 0
Get-ChildItem -Path $OutputDir -Recurse -File | ForEach-Object { $count++; $size += $_.Length }

if ($size -gt 1GB) { $disp = "{0:N2} GB" -f ($size / 1GB) }
elseif ($size -gt 1MB) { $disp = "{0:N2} MB" -f ($size / 1MB) }
else { $disp = "{0:N2} KB" -f ($size / 1KB) }

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Package Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Files:  $count"
Write-Host "  Size:   $disp"
Write-Host "  Output: $OutputDir"
Write-Host "========================================" -ForegroundColor Cyan
