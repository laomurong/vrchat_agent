# ============================================================
# 独立急停脚本 [Win]
# 功能：向正在运行的主进程发送急停信号（写入 flag 文件）
#       主进程 EmergencyStop 轮询到该文件后立即静音输出 + 复位动作。
# 用法：powershell -ExecutionPolicy Bypass -File scripts/emergency_stop.ps1
# ============================================================
$Root = Split-Path -Parent $PSScriptRoot
$Flag = Join-Path $Root "logs\emergency_stop.flag"

New-Item -ItemType Directory -Force -Path (Split-Path $Flag) | Out-Null
New-Item -ItemType File -Force -Path $Flag | Out-Null

Write-Host "[急停] 已发送急停信号: $Flag" -ForegroundColor Yellow
Write-Host "[急停] 主进程将在 <1s 内停止所有输出（TTS 静音 + 动作复位）。"
