<# VRChat Agent One-Click Deploy Script (Windows) #>

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  VRChat Agent One-Click Deploy" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Project: $ProjectRoot"
Write-Host ""

$step = 0
function Step { param([string]$m) $script:step++; Write-Host ("[{0}/7] {1}" -f $script:step, $m) -ForegroundColor Yellow }
function Die { param([string]$m) Write-Host "ERROR: $m" -ForegroundColor Red; Write-Host "Deploy aborted." -ForegroundColor Red; exit 1 }

# Step 1: Check Python
Step "Check Python..."
$pyCmd = $null; $pyVer = $null
foreach ($c in @("python", "python3", "py")) {
    try { $r = & $c --version 2>&1; if ($LASTEXITCODE -eq 0) { $pyCmd = $c; $pyVer = $r; break } } catch {}
}
if ($null -eq $pyCmd) { Die "Python not found. Install Python 3.11+ from https://www.python.org/downloads/" }
Write-Host "  Found: $pyVer"

$m = [regex]::Match($pyVer, '(\d+)\.(\d+)\.(\d+)')
if ($m.Success) {
    $ma = [int]$m.Groups[1].Value; $mi = [int]$m.Groups[2].Value
    if (($ma -lt 3) -or ($ma -eq 3 -and $mi -lt 11)) { Die "Python too old: $pyVer. Need 3.11+." }
    Write-Host "  Version OK (>=3.11)" -ForegroundColor Green
} else { Write-Host "  Warning: cannot parse version, continuing..." -ForegroundColor DarkYellow }

# Step 2: Create venv
Step "Create venv..."
$venv = Join-Path $ProjectRoot ".venv"
if (Test-Path $venv) {
    Write-Host "  venv already exists, skipping."
} else {
    Write-Host "  Creating .venv ..."
    $r = & $pyCmd -m venv $venv 2>&1
    if ($LASTEXITCODE -ne 0) { Die "Failed to create venv.`n$r" }
    Write-Host "  venv created." -ForegroundColor Green
}

$venvPy = Join-Path $venv "Scripts" "python.exe"
$venvPip = Join-Path $venv "Scripts" "pip.exe"
if (-not (Test-Path $venvPy)) { Die "venv python.exe not found: $venvPy" }

# Step 3: Install Python deps
Step "Install Python dependencies..."
$pyproject = Join-Path $ProjectRoot "pyproject.toml"
$reqs = Join-Path $ProjectRoot "requirements.txt"

if (Test-Path $pyproject) {
    Write-Host "  Using pyproject.toml (pip install -e .)..."
    Write-Host "  Note: some packages (faster-whisper, pyaudiowpatch) are large..."
    $r = & $venvPip install -e $ProjectRoot 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Warning: pip install -e failed, trying requirements.txt fallback..." -ForegroundColor DarkYellow
        if (Test-Path $reqs) {
            $r = & $venvPip install -r $reqs 2>&1
            if ($LASTEXITCODE -ne 0) { Die "Python deps install failed.`n$r" }
        } else { Die "Python deps install failed and no requirements.txt found." }
    }
} else {
    if (Test-Path $reqs) {
        Write-Host "  Using requirements.txt..."
        $r = & $venvPip install -r $reqs 2>&1
        if ($LASTEXITCODE -ne 0) { Die "Python deps install failed.`n$r" }
    } else { Die "No pyproject.toml or requirements.txt found." }
}
Write-Host "  Python deps installed." -ForegroundColor Green

# Step 4: Install Node deps
Step "Install Node.js bridge..."
$nodeOk = $false
try { $null = & node --version 2>&1; if ($LASTEXITCODE -eq 0) { $nodeOk = $true } } catch {}
if (-not $nodeOk) { Die "Node.js not found. Install Node.js 18+ from https://nodejs.org/" }

try { $null = & npm --version 2>&1; if ($LASTEXITCODE -ne 0) { Die "npm not available." } } catch { Die "npm not available." }

$nb = Join-Path $ProjectRoot "node_bridge"
if (-not (Test-Path $nb)) { Die "node_bridge not found: $nb" }
if (-not (Test-Path (Join-Path $nb "package.json"))) { Die "node_bridge/package.json not found." }

Write-Host "  Running npm install (postinstall runs 4 patches, 28 MCP tools)..."
Push-Location $nb
try {
    $r = & npm install 2>&1
    if ($LASTEXITCODE -ne 0) { Pop-Location; Die "npm install failed.`n$r" }
} finally { Pop-Location }
Write-Host "  Node deps installed, postinstall patches applied." -ForegroundColor Green

# Step 5: Check config
Step "Check config..."
$cfg = Join-Path $ProjectRoot "config.yaml"
if (-not (Test-Path $cfg)) { Die "config.yaml not found. It should be included in the package." }
Write-Host "  config.yaml ready." -ForegroundColor Green

# Step 6: Env vars
Step "Environment variables..."
$raw = Get-Content -Path $cfg -Encoding UTF8 -Raw
$vars = @()
foreach ($m in [regex]::Matches($raw, 'api_key_env:\s*(\S+)')) {
    $vn = $m.Groups[1].Value.Trim()
    if ($vn -and $vars -notcontains $vn) { $vars += $vn }
}

if ($vars.Count -eq 0) {
    Write-Host "  No api_key_env references found."
} else {
    Write-Host ""
    Write-Host "  ========================================" -ForegroundColor Magenta
    Write-Host "  The following ENV VARS are needed:" -ForegroundColor Magenta
    Write-Host "  ========================================" -ForegroundColor Magenta
    Write-Host ""

    foreach ($vn in $vars) {
        $set = $false
        foreach ($scope in @("User", "Machine", "Process")) {
            $v = [System.Environment]::GetEnvironmentVariable($vn, $scope)
            if ($null -ne $v -and $v.Trim() -ne "") { $set = $true; break }
        }
        if ($set) { Write-Host "    [SET]   $vn" -ForegroundColor Green }
        else      { Write-Host "    [MISS]  $vn  <-- needs config" -ForegroundColor Red }
    }

    Write-Host ""
    Write-Host "  How to set:" -ForegroundColor White
    Write-Host "    Method 1: Create .env file in project root:"
    foreach ($vn in $vars) { Write-Host "      ${vn}=your_api_key_here" }
    Write-Host "    Method 2: System env (permanent):  setx VAR_NAME your_key"
    Write-Host "    Method 3: Temp (current shell):    `$env:VAR_NAME = 'your_key'"

    Write-Host ""
    Write-Host "  Usage:" -ForegroundColor White
    foreach ($vn in $vars) {
        $ls = $raw -split "`r`n"
        $ctx = ""
        for ($i = 0; $i -lt $ls.Count; $i++) {
            if ($ls[$i] -match "api_key_env:\s*$vn") {
                for ($j = $i - 1; $j -ge 0; $j--) {
                    if ($ls[$j] -match '^\s{0,4}(\w[\w_]*):') { $ctx = $ls[$j].Trim(); break }
                }
                break
            }
        }
        if ($ctx) { Write-Host "      $vn -> $ctx" -ForegroundColor Gray }
        else      { Write-Host "      $vn" -ForegroundColor Gray }
    }
    Write-Host ""
    Write-Host "  ========================================" -ForegroundColor Magenta
}

# Step 7: Done
Step "Deploy complete!"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Deploy Successful!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Project: $ProjectRoot"
Write-Host "  venv:    $venv"
Write-Host ""
Write-Host "  Start commands:" -ForegroundColor White
Write-Host "    1. powershell scripts/run.ps1            (recommended)"
Write-Host "    2. python -m vrchat_agent.main           (activate venv first)"
Write-Host "    3. vrchat-agent                          (after pip install -e .)"
Write-Host ""
Write-Host "  Checklist:" -ForegroundColor White
Write-Host "    [ ] Set required env vars (LLM_API_KEY, etc.)"
Write-Host "    [ ] Download model weights to model/ (faster-whisper)"
Write-Host "    [ ] Enable OSC in VRChat (Menu -> OSC -> Enable)"
Write-Host "    [ ] If using OBS, install and configure OBS WebSocket"
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
exit 0
