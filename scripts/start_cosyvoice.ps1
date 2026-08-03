# ============================================================================
# 一键启动 CosyVoice 2.0-0.5B 克隆语音推理服务（任务 1）
#
# 用法：
#   双击 scripts/start_cosyvoice.bat（推荐，或本脚本）
#   或 powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_cosyvoice.ps1
#
# 前置条件：
#   1. 模型文件位于 model/（llm.pt / hift.pt / campplus.onnx /
#      speech_tokenizer_v2.onnx / cosyvoice2.yaml / flow.pt 或 flow.cache.pt）；
#      缺 flow.cache.pt（.crdownload 0 字节 = 之前下载中断）时本脚本会明确提示；
#   2. 推理依赖（torch / torchaudio / onnxruntime / cosyvoice 包）已安装在
#      当前 Python 环境（推荐 conda 创建 cosyvoice 环境，见 README 6.2 节）。
#
# 参数：
#   -ModelDir  模型目录（默认 <项目根>\model）
#   -Port      服务端口（默认 8000）
#   -Python    python.exe 路径（留空自动探测 PATH）
# ============================================================================
param(
    [string]$ModelDir = "",
    [int]$Port = 8000,
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot          # scripts\.. = 项目根
if (-not $ModelDir) { $ModelDir = Join-Path $ProjectRoot "model" }
$SrcDir = Join-Path $ProjectRoot "src"

function Write-Step($msg)  { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)    { Write-Host "    $msg" -ForegroundColor Green }
function Write-Warn2($msg) { Write-Host "    $msg" -ForegroundColor Yellow }
function Write-Err($msg)   { Write-Host "[错误] $msg" -ForegroundColor Red }

function Exit-WithPause($code) {
    if (-not $env:COSYVOICE_NO_PAUSE) { Read-Host "按回车退出" }
    exit $code
}

# 运行 Python 探测命令并返回退出码。
# 注意：PS 5.1 下 $ErrorActionPreference="Stop" 会把原生命令（python）的 stderr
# 当作终止错误，导致依赖检查误中断；这里局部降级为 Continue 再取 $LASTEXITCODE。
function Invoke-PythonProbe([string]$code) {
    $oldPref = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python -c $code 2>$null
    $rc = $LASTEXITCODE
    $ErrorActionPreference = $oldPref
    return $rc
}

Write-Step "CosyVoice 2.0-0.5B 克隆语音服务启动器"
Write-Step "模型目录: $ModelDir"

# ---------------------------------------------------------------------------
# 1. 模型文件完整性检查（缺关键文件 → 报错退出；重点提示 flow.cache.pt.crdownload）
# ---------------------------------------------------------------------------
$required = @("llm.pt", "hift.pt", "campplus.onnx", "speech_tokenizer_v2.onnx", "cosyvoice2.yaml")
$missing = @()
foreach ($f in $required) {
    if (-not (Test-Path (Join-Path $ModelDir $f))) { $missing += $f }
}
# flow 权重：flow.pt（官方命名）或 flow.cache.pt（镜像命名），至少一个
$hasFlowPt    = Test-Path (Join-Path $ModelDir "flow.pt")
$hasFlowCache = Test-Path (Join-Path $ModelDir "flow.cache.pt")
if (-not ($hasFlowPt -or $hasFlowCache)) {
    $missing += "flow.pt / flow.cache.pt（flow 流模型权重，必需）"
}

if ($missing.Count -gt 0) {
    Write-Err "模型文件不完整，缺少："
    foreach ($m in $missing) { Write-Host "    - $m" -ForegroundColor Red }
    $crd = Join-Path $ModelDir "flow.cache.pt.crdownload"
    if (Test-Path $crd) {
        $size = (Get-Item $crd).Length
        Write-Err "检测到 flow.cache.pt.crdownload（${size} 字节）→ 这是未完成的下载！"
        Write-Err "请重新下载 flow.cache.pt（之前下载中断），完整文件约 1GB+；"
        Write-Err "下载完成后删除 .crdownload 残留文件再启动。"
    }
    $crd2 = Join-Path $ModelDir "flow.pt.crdownload"
    if (Test-Path $crd2) {
        Write-Err "检测到 flow.pt.crdownload → 同样未完成，请重新下载 flow.pt。"
    }
    Write-Host ""
    Write-Err "请先补全模型文件再启动（部署指引见 README「CosyVoice 克隆语音」节）。"
    Exit-WithPause 1
}
Write-Ok "模型文件完整（含 flow 权重）"

# 可选文件缺失提示（不阻断）
if (-not (Test-Path (Join-Path $ModelDir "flow.encoder.fp16.zip"))) {
    Write-Warn2 "未找到 flow.encoder.fp16.zip（可选）：将使用纯 PyTorch flow encoder（不影响推理，GPU 上略慢）"
}
if (-not (Test-Path (Join-Path $ModelDir "CosyVoice-BlankEN"))) {
    Write-Warn2 "未找到 CosyVoice-BlankEN（可选）：中文克隆不需要；若克隆英文等多语种建议补下"
}

# ---------------------------------------------------------------------------
# 2. Python 探测
# ---------------------------------------------------------------------------
if (-not $Python) { $Python = (Get-Command python -ErrorAction SilentlyContinue).Source }
if (-not $Python) {
    Write-Err "未找到 Python。请先安装 Python 3.11+ 或激活 CosyVoice 官方 conda 环境，"
    Write-Err "或通过 -Python 参数指定 python.exe 路径（如 E:\...\cosyvoice\python.exe）。"
    Exit-WithPause 1
}
Write-Ok "Python: $Python"

# ---------------------------------------------------------------------------
# 3. 推理依赖检查（torch / torchaudio / onnxruntime / cosyvoice / fastapi / uvicorn）
# ---------------------------------------------------------------------------
if ((Invoke-PythonProbe "import torch, torchaudio, onnxruntime") -ne 0) {
    Write-Err "当前 Python 缺少推理依赖 torch / torchaudio / onnxruntime。"
    Write-Err "请按 README「CosyVoice 克隆语音」节安装推理依赖（推荐 conda 创建 cosyvoice 环境），再重试。"
    Exit-WithPause 1
}
Write-Ok "推理依赖 torch/torchaudio/onnxruntime 已就绪"

if ((Invoke-PythonProbe "import cosyvoice") -ne 0) {
    Write-Err "缺少 cosyvoice 包：请 git clone https://github.com/FunAudioLLM/CosyVoice "
    Write-Err "并 pip install -e CosyVoice 后重试（部署指引见 README 6.2 节）。"
    Exit-WithPause 1
}
Write-Ok "cosyvoice 包已就绪"

if ((Invoke-PythonProbe "import fastapi, uvicorn, python_multipart") -ne 0) {
    Write-Warn2 "缺少 fastapi/uvicorn/python-multipart，自动安装…"
    & $Python -m pip install -q fastapi uvicorn python-multipart
    if ($LASTEXITCODE -ne 0) {
        Write-Err "安装失败，请手动执行: pip install fastapi uvicorn python-multipart"
        Exit-WithPause 1
    }
    Write-Ok "fastapi/uvicorn/python-multipart 已安装"
}

# ---------------------------------------------------------------------------
# 4. GPU 信息提示（可选）
# ---------------------------------------------------------------------------
$nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidia) {
    Write-Step "GPU 信息："
    & nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader |
        ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }
}
else {
    Write-Warn2 "未检测到 nvidia-smi（可能无 NVIDIA GPU 或不在 PATH）。CosyVoice2 在 CPU 上可跑但很慢，建议使用 GPU。"
}

# ---------------------------------------------------------------------------
# 5. 启动服务
# ---------------------------------------------------------------------------
$env:PYTHONPATH = "$SrcDir;$env:PYTHONPATH"
$env:COSYVOICE_MODEL_DIR = $ModelDir

Write-Step "启动 CosyVoice 服务（http://127.0.0.1:$Port）…"
Write-Ok "首次 /tts 会加载模型（llm.pt 2GB）约 1~3 分钟，请耐心等待；"
Write-Ok "加载完成后在 Agent Web 控制台设置: provider=clone, base_url=http://127.0.0.1:$Port, method=post"
Write-Ok "健康检查: http://127.0.0.1:$Port/health   合成接口: POST http://127.0.0.1:$Port/tts"
Write-Host ""

& $Python -m vrchat_agent.cosyvoice.server --model-dir $ModelDir --host 127.0.0.1 --port $Port
$exitCode = $LASTEXITCODE
Write-Err "CosyVoice 服务已退出（code=$exitCode）。"
Exit-WithPause $exitCode
