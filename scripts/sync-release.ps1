# VRChat Agent — 开发环境 → 发布目录同步脚本
# 用法: .\scripts\sync-release.ps1 [-Source .\] [-Dest "D:\biaozhu\分享\vrchat-agent-release\"] [-WhatIf]
#
# 同步策略:
#   - 复制生产文件（src/、mappings/、scripts/、node_bridge/、config.yaml 等）
#   - 目标目录多了源没有的文件 → 删除（"删减"）
#   - 源有目标没有 → 新增
#   - 两边都有但源更新 → 覆盖（"替换"）
#
# 排除:
#   - tests/、__pycache__/、*.pyc、*.egg-info
#   - node_modules/、logs/、*.log、*.jsonl
#   - _qa*、_yan*、_tmp*、_r2*、qa-*
#   - .env、config.yaml.bak、.git、deploy-test/

[CmdletBinding()]
param(
    [string]$Source = $PSScriptRoot,
    [string]$Dest = "D:\biaozhu\分享\vrchat-agent-release"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Dest)) {
    Write-Error "发布目录不存在: $Dest"
    exit 1
}

Write-Host "同步: $Source -> $Dest" -ForegroundColor Cyan
if ($WhatIfPreference) { Write-Host "[预览模式 — 不会实际修改文件]" -ForegroundColor Yellow }

# robocopy 排除列表
$XDirs  = @("tests","__pycache__","node_modules","logs",".git","deploy-test","deploy-vrfy-*","data",".venv",".venv-qa")
$XFiles = @("*.pyc","*.log","*.jsonl","_qa*","_yan*","_tmp*","_r2*","qa-*",".env","config.yaml.bak","*.egg-info","*.egg-info\*","_qa_probe*","_qa_probe*\*")

# robocopy 公共参数
function Sync-Dir($SubDir, $ExtraXD = @(), $ExtraXF = @()) {
    $from = Join-Path $Source $SubDir
    $to   = Join-Path $Dest   $SubDir
    if (-not (Test-Path $from)) { Write-Host "  跳过(源不存在): $SubDir" -ForegroundColor DarkGray; return }

    $xd = ($XDirs + $ExtraXD | ForEach-Object { "/XD" ; $_ })
    $xf = ($XFiles + $ExtraXF | ForEach-Object { "/XF" ; $_ })
    
    $args = @($from, $to, "/MIR", "/NJH", "/NJS", "/NP", "/R:2", "/W:2") + $xd + $xf
    $result = & robocopy @args 2>&1
    $code = $LASTEXITCODE
    if ($code -ge 8) {
        Write-Warning "robocopy $SubDir 出错 (exit=$code): $result"
    } elseif ($code -gt 0) {
        $summary = ($result | Select-String "Files|Dirs|Bytes" | ForEach-Object { $_.Line.Trim() }) -join " | "
        if ($summary) { Write-Host "  $SubDir : $summary" -ForegroundColor Green }
        else           { Write-Host "  $SubDir : 已同步" -ForegroundColor Green }
    } else {
        Write-Host "  $SubDir : 无变化" -ForegroundColor DarkGray
    }
}

# 逐目录同步
Sync-Dir "src"
Sync-Dir "mappings"
Sync-Dir "scripts"
Sync-Dir "web"

# node_bridge: 排除 node_modules 但保留 scripts/
Sync-Dir "node_bridge" -ExtraXD @("node_modules")

# 根级文件: 只复制不删除（避免丢了 deploy.bat、README 等手工文件）
function Sync-RootFile($Name) {
    $from = Join-Path $Source $Name
    $to   = Join-Path $Dest   $Name
    if (Test-Path $from) {
        Copy-Item $from $to -Force
        Write-Host "  $Name : 已更新" -ForegroundColor Green
    }
}
Sync-RootFile "config.yaml"
Sync-RootFile "pyproject.toml"
Sync-RootFile "requirements.txt"
Sync-RootFile "README.md"

Write-Host "同步完成" -ForegroundColor Cyan
