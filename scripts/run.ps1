# ============================================================
# VRChat Agent 一键启动脚本 [Win]
# 功能：检查 Python → 检查/安装依赖 → 检查 OSC 桥接 → 启动主进程
# ============================================================
# 修复中文乱码：强制 UTF-8 输出（Windows PowerShell 5.1 默认 GBK）
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
$OutputEncoding = [System.Text.Encoding]::UTF8
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "[1/4] 检查 Python 3.11+ ..."
# 优先使用 conda 环境（若已激活）；否则回退 PATH 中的 python
$py = $null
if ($env:CONDA_PREFIX -and (Test-Path "$env:CONDA_PREFIX\python.exe")) {
    $py = "$env:CONDA_PREFIX\python.exe"
    Write-Host "       使用 conda 环境: $($env:CONDA_PREFIX)"
} else {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { $py = $cmd.Source }
}
if (-not $py) {
    Write-Error "未找到 Python，请先激活 conda 环境（conda activate vrchat-agent）或安装 Python 3.11+ 并加入 PATH。"
    exit 1
}
$ver = & $py -c "import sys; print(sys.version_info[:2])" 2>$null
Write-Host "       Python 版本: $ver"

# ------------------------------------------------------------
# [2/4] 依赖检查：关键包可导入则跳过安装（避免每次启动都 pip install）
# 说明：探测 requirements.txt 全部运行时依赖对应的 import 名；只要任一
#       缺失就执行 pip install（缺失时全量安装天然兜底，requirements.txt
#       新增依赖也能覆盖）。依赖已满足时显示"跳过安装"，启动更快。
# ------------------------------------------------------------
Write-Host "[2/4] 检查 Python 依赖 ..."
$ProbeModules = @(
    "numpy", "yaml", "dotenv", "openai", "edge_tts",
    "faster_whisper", "webrtcvad", "sounddevice", "obswebsocket", "cv2",
    "mss", "mcp", "keyboard", "miniaudio",
    "fastapi", "uvicorn", "httpx", "rich"
)
$modList = ($ProbeModules | ForEach-Object { "'$_'" }) -join ','
$probeCode = "import importlib.util; mods=[$modList]; " +
    "missing=[m for m in mods if importlib.util.find_spec(m) is None]; " +
    "print('MISSING:'+','.join(missing) if missing else 'ALL_OK')"
$probeOut = & $py -c $probeCode 2>$null
if ($LASTEXITCODE -ne 0) { $probeOut = "MISSING:probe_error" }

if ($probeOut -match "ALL_OK") {
    Write-Host "       依赖已满足，跳过安装（$($ProbeModules.Count) 个核心包可导入）"
} else {
    $missingNames = ($probeOut -replace '^MISSING:', '')
    Write-Host "       检测到缺失依赖: $missingNames"
    Write-Host "       正在安装依赖（仅缺失时执行） ..."
    & $py -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Error "依赖安装失败，请检查网络或 pip 源。"
        exit 1
    }
}

# ------------------------------------------------------------
# [3/4] OSC 桥接检查：依赖目录 + config.yaml 的 node.exe（不再以 PATH 为准）
# 说明：node 可能不在 PATH（如 .workbuddy 自带 node），config.yaml
#       action.bridge.command 已配置绝对路径；只要桥接依赖目录存在或
#       该 node.exe 存在，即认为桥接可用，避免误报"降级模式"。
# ------------------------------------------------------------
Write-Host "[3/4] 检查 OSC 桥接（Node.js） ..."
$bridgePkg = "$Root\node_bridge\node_modules\@iflow-mcp\krekun-vrchat-mcp-osc"
$bridgeReady = $false
if (Test-Path $bridgePkg) {
    Write-Host "       OSC 桥接依赖已就绪: node_modules/@iflow-mcp/krekun-vrchat-mcp-osc"
    $bridgeReady = $true
} else {
    Write-Host "       桥接依赖未安装（缺 node_modules/@iflow-mcp/krekun-vrchat-mcp-osc），尝试 npm install ..."
    $npm = Get-Command npm -ErrorAction SilentlyContinue
    if ($npm) {
        Push-Location "$Root\node_bridge"
        npm install 2>$null
        Pop-Location
    } else {
        Write-Host "       未找到 npm 命令，跳过自动安装（请手动 cd node_bridge && npm install）"
    }
    if (Test-Path $bridgePkg) {
        Write-Host "       OSC 桥接依赖已就绪（npm install 完成）"
        $bridgeReady = $true
    }
}

$bridgeCmd = $null
if (Test-Path "$Root\config.yaml") {
    $cmdLine = Select-String -Path "$Root\config.yaml" -Pattern '^\s*command:\s*(.+)$' |
        Select-Object -Last 1
    if ($cmdLine) {
        $bridgeCmd = ($cmdLine.Line -replace '^\s*command:\s*', '').Trim().Trim('"').Trim("'")
    }
}
$nodeOk = $false
if ($bridgeCmd -and (Test-Path $bridgeCmd)) {
    Write-Host "       Node.js 可执行文件就绪: $bridgeCmd"
    $nodeOk = $true
}

if (-not $bridgeReady) {
    if ($nodeOk) {
        Write-Host "       提示：OSC 桥接依赖未安装，但 node.exe 可用；请 cd node_bridge && npm install 后重启。"
    } else {
        Write-Warning "OSC 桥接未就绪（桥接依赖缺失且未找到 node.exe）。将进入降级模式：仅语音无动作。请 cd node_bridge && npm install，并确认 config.yaml action.bridge.command 指向有效的 node.exe。"
    }
} elseif (-not $nodeOk -and $bridgeCmd) {
    Write-Warning "提示：OSC 桥接依赖已就绪，但 config.yaml 中 node.exe 不存在（$bridgeCmd），动作桥接可能无法启动。"
}

Write-Host "[4/4] 启动 VRChat Agent 主进程 ..."
Write-Host "       急停快捷键: Ctrl+Alt+X ；独立急停: scripts/emergency_stop.ps1"
Write-Host "       Web 控制台: 启动后自动打开浏览器（URL 见日志/控制台）"
$env:PYTHONPATH = "$Root\src"
& $py -m vrchat_agent.main
