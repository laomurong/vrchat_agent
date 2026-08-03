# VRChat 全自主具身 AI Agent

让 AI 像真人玩家一样在 VRChat 中游玩：**听**（WASAPI 回环 + VAD + 本地 Whisper）→ **决策**（OpenAI 兼容 LLM，多模态视觉理解）→ **说 + 动作**（云端/本地 TTS → VB-Audio 虚拟声卡 + OSC 表情/移动/聊天框）。

> 技术栈：**Python 3.11+ 主编排（asyncio 事件驱动）** + **Node 仅作 OSC 桥接子进程**（复用 `vrchat-mcp-osc`）。
> 合规红线：仅使用 **VRChat 官方 OSC 协议** + **系统级音频/视频回环**；**严禁**任何内存读取 Mod（MelonLoader 等）。

---

## 1. 架构总览（7 大模块）

| 模块 | 能力 | 关键实现 | 平台依赖 |
| --- | --- | --- | --- |
| A 语音输入 | WASAPI 回环捕获 → VAD → faster-whisper ASR → 自抑制 | `audio/` | [Win] WASAPI 回环 |
| B 大脑 | LLM 决策 JSON + pydantic 校验 + 上下文 + 人设 | `brain/` | — |
| C 语音输出 | TTS 抽象（Edge 本地 / 云端 / 克隆语音·本地模型，默认云端）→ 播放到虚拟声卡 | `speech/` | [Win] VB-Audio Virtual Cable |
| D 动作控制 | Actuator 抽象 + MCP OSC 实现 + 语义动作映射 YAML | `action/` `bridge/` | [Win] Node.js + vrchat-mcp-osc |
| E 视觉感知 | OBS WebSocket 取帧（首选）/ OpenCV 回退 + 0.5~1fps 采样 | `vision/` | [Win] OBS Studio |
| F 编排主循环 | 状态机 IDLE→LISTENING→THINKING→SPEAKING + 打断 + 急停 | `core/` | — |
| G 配置可观测 | config.yaml 热重载 + JSON Lines 日志 + 仪表盘 + 急停 | `config.py` `logging_setup.py` `core/dashboard.py` | [Win] 快捷键 |

模块间**只通过 `EventBus` 通信**（装配层除外）；音频全链路 **16kHz mono float32**；API keys 走 `.env` 不进 yaml。

## 2. 目录结构

```
vrchat_agent/
├── pyproject.toml / requirements.txt / requirements-dev.txt
├── config.yaml                 # 主配置（模型/阈值/映射/keys 引用 .env）
├── .env.example                # API keys 模板（复制为 .env）
├── scripts/
│   ├── run.ps1                 # 一键启动 [Win]
│   └── emergency_stop.ps1      # 独立急停脚本 [Win]
├── node_bridge/                # Node OSC 桥接（复用 vrchat-mcp-osc）
├── src/vrchat_agent/
│   ├── main.py                 # 入口装配
│   ├── config.py / events.py / logging_setup.py
│   ├── audio/  brain/  speech/  action/  vision/  core/  bridge/
├── mappings/avatar_default.yaml  # 语义动作 → OSC 映射表（占位模板）
└── tests/                      # 离线可跑 pytest（外部依赖全 mock）
```

## 3. 安装

```powershell
# 1) Python 3.11+
python --version

# 2) 安装 Python 依赖
pip install -r requirements.txt
# 开发/测试：pip install -r requirements-dev.txt

# 3) Node 桥接（动作控制需要）
cd node_bridge
npm install
cd ..

# 4) 配置 .env
copy .env.example .env   # 填写 LLM_API_KEY / TTS_API_KEY
```

**系统前置（[Win]）**：
- 安装 **VB-Audio Virtual Cable**，把 VRChat 麦克风设备设为 `CABLE Output`
- 安装 **OBS Studio**（WebSocket 端口 4455；或启动虚拟摄像头作回退）
- 安装 **NVIDIA 驱动**（faster-whisper GPU 推理；无 GPU 改 config `asr.device=cpu, compute_type=int8`）

## 4. 运行

```powershell
# 一键启动（检查依赖 + 装依赖 + 主进程）
powershell -ExecutionPolicy Bypass -File scripts/run.ps1

# 或手动：
$env:PYTHONPATH = "src"
python -m vrchat_agent.main
```

启动前建议：
1. 打开 VRChat（Desktop 模式）并进入房间，开启 **OSC**（菜单 → OSC → Enable）
2. 选择音频输入来源（Web 控制台「参数」页下拉框，或 config.yaml `audio.input.device`）：
   - **麦克风**：听周围玩家说话（经 VRChat 语音传输）；
   - **电脑内录（回环）**：`<输出设备名> [内录]`（WASAPI 回环，听 VRChat 内所有声音，
     需 VRChat 正在运行；基于 pyaudiowpatch，`python -c "from vrchat_agent.audio.audio_input
     import AudioInput; print(AudioInput.list_loopback_devices())"` 可枚举）
3. 确认 TTS 输出设备为 VB-Audio 虚拟声卡

## 5. 急停

| 方式 | 操作 | 生效 |
| --- | --- | --- |
| 全局快捷键 | `Ctrl+Alt+X` | < 1s 静音 + 复位动作 |
| 独立脚本 | `powershell -File scripts/emergency_stop.ps1` | < 1s |

急停后状态复位 IDLE，等待下次输入。

## 5.1 Web 控制台操控（任务 2：B1/B2/B3）

Web 控制台「① 状态」页顶部新增**控制区**，可在浏览器里实际操作 Agent：

| 操作 | 接口 | 行为 |
| --- | --- | --- |
| 紧急停止 | `POST /api/control/emergency_stop`（`/api/emergency` 同义） | 发布 `EMERGENCY_STOP`：静音输出 + 复位动作（旁路） |
| 暂停 | `POST /api/control/pause` | 主循环进入暂停态，忽略语音触发（不退出进程） |
| 恢复 | `POST /api/control/resume` | 恢复监听 |
| 重启 | `POST /api/control/restart` | 进程级重启：优雅关闭（走 shutdown 流程）→ 资源释放完成后 `subprocess.Popen` 拉起新进程（重新加载配置；新进程在旧端口释放后才启动，无端口冲突） |
| 关闭 Agent | `POST /api/control/shutdown` | 触发主循环 `stop()` → 现有优雅关闭流程（web/急停/动作/MCP/视觉依次收尾） |
| 状态 | `GET /api/control/status` | `{running, paused, emergency_stopped, state, uptime_s, processed_count, restartable, ...}` |

- 控制端点仅本机可访问（`web.host` 默认 `127.0.0.1`）；
- 紧急停止/重启/关闭需二次确认；暂停 ↔ 恢复为同一按钮切换；`restartable=false`
  时前端隐藏「重启」按钮；
- 页面每 2s 轮询控制状态，外部触发（快捷键急停等）也会反映到按钮与状态徽标。

## 6. 配置说明（config.yaml 要点）

- `asr.device`：`cuda`（GPU）或 `cpu`；`model_size` 低配可下调 `small/base`
- `brain.llm.base_url`：云端 OpenAI / 本地 Ollama（`http://localhost:11434/v1`）/ vLLM（`/v1`）
- `speech.tts.provider`：`cloud`（默认，需 TTS_API_KEY）、`edge`（本地免费）、
  `clone`（克隆语音·本地模型，见 6.2）或 `mimo`（小米 MiMo 克隆语音·云，见 6.3）。
  无 TTS_API_KEY 或 cloud 合成失败（认证/端点不支持/解码失败）会自动回退 Edge 本地合成，
  不会阻断说话；`clone` 服务不可达/失败、`mimo` 克隆服务失败（无 key/参考音频
  缺失/认证/超时/网络）同样自动回退 Edge。
- `audio.input.device` / `speech.output.device`：设备用**数字 index**（Web 下拉框值）。
  Web 下拉框已**实测 16k 兼容并屏蔽不支持 16k 的设备**（WASAPI/WDM-KS 只认 48k 或
  打开失败 → 不显示；MME/DirectSound 支持 16k → 正常显示）。设备插拔导致 index
  变化时，用 Web 控制台「参数 → 音频输入/输出 → 设备」重新选择即可。
- `audio.input.filter_*`：**ASR 误识别过滤**——本地 Whisper 有时把环境音/视频字幕/
  BGM 误识别为"用户说过的话"，命中规则的文本**不会进入 LLM 流程**（热重载，
  Web「参数 → 音频输入」保存即生效）：
  - `filter_enabled`：总开关（默认 `true`）；
  - `filter_blocklist`：**子串匹配**屏蔽词（长词/多字词用包含匹配，覆盖
    `"字幕志愿者 李宗盛"` 这类带前后缀的变体；默认含 `字幕志愿者/谢谢观看/字幕/BGM/背景音乐`）；
  - `filter_exact_words`：**精确匹配**屏蔽词（仅单字/短回应词，`text.strip() == 词`
    才命中，避免 `"你好"` 被 `"好"` 误滤；默认含 `好/嗯/啊/哦`）；
  - `filter_min_length`：去空白后**最短保留长度**（默认 `1` = 仅滤空/纯空白；
    要滤单字请加进 `filter_exact_words` 而非调大此值，避免误伤正常短句）。
- `agent.startup_check`：**启动自检**（默认 `true`，可关）——启动后异步后台执行
  功能自检（LLM/TTS/视觉/OSC 配置可达性，不消耗 token）+ OSC 自动校准（枚举当前
  avatar 参数与映射表比对，缺失记 WARNING 提示运行 `scripts/calibrate_actions.py`，
  **只读不写映射表**）；结果记 INFO 日志并展示在 Dashboard 自检区，失败不阻塞启动。
- `agent.utterance_queue`：**对话队列 + 批量合并**（默认开）——模型忙时新对话入队
  不再在锁上隐式等待；队列待处理 ≥ `max_batch`（默认 3）条时合并为一条
  `（连续对话）1. A 2. B 3. C` 喂 LLM 综合回应，否则逐条处理；急停清空队列；
  `enabled` / `max_batch` 均热重载。
- `agent.chatbox_sync`：`true`（默认）时每次 AI 回复自动同步发送 VRChat 聊天框
  （chatbox:reply → send_message，消息自动截断 140 字符）；`false` 关闭。
- `action.mapping.path`：不同 avatar 换映射表

### 6.0 热重载 vs 需重启（任务 1：A1/A2/A3）

Web 控制台「参数」页保存配置后，**♻ 可热重载**字段即时生效（无需重启），
**⚠ 需重启**字段会在保存结果里明确提示。

| 类别 | 字段（dotted path） | 生效方式 |
| --- | --- | --- |
| ♻ 热重载 | `agent.loop.*`、`agent.interrupt.enabled`、`agent.degradation.*`、`agent.chatbox_sync` | 保存即生效 |
| ♻ 热重载 | `brain.llm.temperature`、`brain.llm.max_tokens`、`brain.llm.timeout`、`brain.llm.vision_enabled` | 下次决策生效 |
| ♻ 热重载 | `brain.persona.name`、`brain.persona.system_prompt` | 下次决策生效 |
| ♻ 热重载 | `brain.memory.enabled`、`inject_limit`、`auto_extract`、`extract_interval`、`extract_max_items` | 下次决策/提取生效 |
| ♻ 热重载 | `asr.vad_mode`、`asr.min_speech_ms`、`asr.silence_ms`、`asr.max_segment_s`、`audio.input.silence_threshold`、`audio.input.filter_*` | 下个音频块生效 |
| ♻ 热重载 | `vision.sampling.fps`、`max_size`、`jpeg_quality`、`event_trigger`、`change_threshold` | 下个采样周期生效 |
| ♻ 热重载 | `speech.tts.voice`、`rate`、`volume`、`emotion_voice`、`timeout` | 下次合成生效 |
| ♻ 热重载 | `dashboard.refresh_ms`、`logging.level` | 下次循环/保存即生效 |
| ⚠ 需重启 | `audio.input.device`、`audio.input.sample_rate`、`audio.input.chunk_ms`、`audio.output.*`、`speech.output.*` | 设备/采样率重建 |
| ⚠ 需重启 | `asr.model_size`、`asr.model_path`、`asr.device`、`asr.compute_type`、`speech.tts.provider`、`vision.sensor.backend` | 模型/引擎重建 |
| ⚠ 需重启 | `brain.llm.base_url`、`brain.llm.model`、`action.*`、`web.*`、`emergency.hotkey` 等其余字段 | 装配期读取 |

> 原则：**阈值/采样/人设/音色类参数热加载；设备/模型/provider 等资源型参数需重启**。
> 音频输出设备：7.1ch MME/DirectSound 支持 16k；WASAPI/WDM-KS 只认 48k 或打开失败，
> Web 下拉框已实测屏蔽（supports_16k=false 不进可选分组），若已保存值恰为不支持
> 16k 的设备，下拉框会以"⚠ 不支持 16k，已屏蔽"禁用项展示，播放报
> `Invalid sample rate` 时请换支持 16k 的设备。

### 6.2 克隆语音 TTS（CosyVoice 2.0-0.5B，任务 1 增量）

`speech.tts.provider: clone` 时使用**克隆语音模型**（已接入 CosyVoice 2.0-0.5B，
多语种 + 声音克隆）替代/补充云端 TTS，让 AI 用你指定的音色说话。

**方案选型（已选定 CosyVoice2）**：

| 模型 | 声音克隆 | 中文质量 | HTTP 接入成本 | 部署成本 |
| --- | --- | --- | --- | --- |
| **CosyVoice2**（已接入） | zero-shot（3~10s 参考音频） | 优秀 | 低：项目内一键脚本 + `/tts` JSON | 中：torch/onnxruntime（需补 flow.cache.pt） |
| GPT-SoVITS | few-shot（参考音频+文本） | 优秀 | 低：GET `/tts` 参数直白 | 中：需下载模型（~2GB） |
| fish-speech | zero-shot | 好 | 中：OpenAI 兼容 `/v1/tts` | 高：模型大、需 GPU |

**配置**（Web 控制台「参数 → 语音合成」或 config.yaml）：

```yaml
speech:
  tts:
    provider: clone          # cloud / edge / clone
    tts_clone:
      base_url: http://127.0.0.1:8000   # CosyVoice 服务地址（一键启动脚本默认端口）
      method: post            # post=CosyVoice 等 JSON 服务；get=GPT-SoVITS 风格(query)
      ref_audio_path: ""      # 克隆参考音频（服务端可访问路径；3~10s 干净人声 wav/mp3）
      ref_text: ""            # 参考音频对应文本（zero-shot 克隆建议填写）
      lang: zh                # 合成语言：zh/en/ja/ko 等（CosyVoice 自动检测，提示用）
      timeout: 60.0           # 首句合成较慢，建议 ≥60s
```

**CosyVoice 部署指引**：

1. **补下模型文件**：模型目录 `model/` 必须包含 `llm.pt` / `hift.pt` /
   `campplus.onnx` / `speech_tokenizer_v2.onnx` / `cosyvoice2.yaml` / `flow.pt`（或
   镜像命名的 `flow.cache.pt`）。若 `model/flow.cache.pt.crdownload` 是 **0 字节**，
   说明之前下载中断，请**重新下载 flow.cache.pt（约 1GB+）**，下载完成后删除
   `.crdownload` 残留。启动脚本会自动检查并逐项提示。
2. **安装推理依赖**（CosyVoice 官方环境，推荐 conda）：
   ```powershell
   conda create -n cosyvoice python=3.11 -y
   conda activate cosyvoice
   pip install torch torchaudio onnxruntime
   git clone https://github.com/FunAudioLLM/CosyVoice
   cd CosyVoice && pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
   pip install -e .
   pip install fastapi uvicorn python-multipart
   ```
3. **启动服务**：双击 `scripts/start_cosyvoice.bat`（或运行
   `scripts/start_cosyvoice.ps1`），首次 `/tts` 会加载 llm.pt（2GB，约 1~3 分钟）。
   健康检查：`http://127.0.0.1:8000/health`；合成接口：`POST http://127.0.0.1:8000/tts`。
4. **准备参考音频**：3~10s 干净人声（无 BGM / 混响 / 多人声），wav 或 mp3 均可；
   把路径填到 `ref_audio_path`（服务端可访问），`ref_text` 填这段音频说的内容。
5. **接入 Agent**：Web 控制台把 `speech.tts.provider` 改为 `clone` 并保存（provider
   属需重启字段，重启进程生效）；克隆服务未启动/失败时自动回退 Edge 本地合成
   （不阻断说话），日志可见 `克隆语音 TTS 服务不可达` 提示。

**fp16 兼容性结论**（核对官方 `cosyvoice/cli/cosyvoice.py`）：`flow.encoder.fp16.zip`
是 **TorchScript（JIT）** 打包的 flow encoder，仅当 GPU 上用 `load_jit=True` +
`fp16=True` 时才被加载；官方默认推理走**纯 PyTorch flow encoder**，**完全不使用**
该文件，因此「只有 fp16.zip、没有 fp32.zip」不影响基本推理，**无需换 tp16**。
`CosyVoice-BlankEN` 仅英文等多语种克隆需要，中文克隆不需要。

> CosyVoice 真实模型推理需用户自行安装 torch/onnxruntime 等推理依赖并补下
> flow.cache.pt；本机 Agent 代码链路已用假服务 + 全量测试验证（离线可跑）。

**GPT-SoVITS 等其他模型**：把 `base_url` 改为对应服务地址、`method` 按服务风格
设置即可（引擎会同时携带 GPT-SoVITS 与 CosyVoice 两套字段名，CosyVoice 服务端
自动归一化）。

### 6.3 小米 MiMo 克隆语音 TTS（云服务 · 零样本声音克隆，MiMo 接入增量）

`speech.tts.provider: mimo` 时使用**小米 MiMo 克隆 TTS**（`MiMo-V2.5-TTS-VoiceClone`，
云服务，无需本地部署模型）。它**不是** OpenAI 兼容的 `/audio/speech` 端点，而是走
`chat.completions + audio 参数`：参考音频经 `voice` 字段以 base64 data URI 传入
（零样本克隆），文本放 `assistant` 消息，响应 `message.audio.data` 返回 base64 wav。

**配置**（Web 控制台「参数 → 语音合成」或 config.yaml）：

```yaml
speech:
  tts:
    provider: mimo          # cloud / edge / clone / mimo
    tts_mimo:
      base_url: https://api.xiaomimimo.com/v1
      api_key_env: MIMO_TTS_API_KEY
      model: mimo-v2.5-tts-voiceclone
      ref_audio_path: C:/Users/14718/Documents/Voicemeeter/Record 2026-08-02 at 20h19m47s.wav
      timeout: 60.0          # 首句克隆较慢，建议 ≥60s
```

**使用步骤**：

1. **配 key**：`.env` 写 `MIMO_TTS_API_KEY=sk-...`（绝不写入 config.yaml）；
   LLM 同理 `LLM_API_KEY=tp-...`（`brain.llm` 指向 `https://token-plan-cn.xiaomimimo.com/v1`，
   model=`mimo-v2.5`）。
2. **准备参考音频**：3~10s 干净人声 wav（无 BGM/混响/多人声），路径填到
   `ref_audio_path`（支持热重载，改后无需重启）。
3. **接入 Agent**：Web 控制台把 `speech.tts.provider` 改为 `mimo` 并保存（provider
   属需重启字段，重启进程生效）；克隆服务失败（无 key/参考音频缺失/认证/超时/
   网络/返回非音频）自动回退 Edge 中文女声（不阻断说话），日志可见
   `主 TTS 引擎失败，自动回退备用引擎` 提示。

**热重载**：`tts_mimo.base_url` / `ref_audio_path` / `timeout` 保存即生效
（引擎每次合成从 config 读取，base_url 变化自动重建 client）；`model` /
`api_key_env` 属需重启字段，`api_key` 走 `.env`（改后需重启进程加载）。

### 6.4 长期记忆系统（标准版）

AI 记住玩家的长期事实/喜好，**重启不丢、跨会话生效**；Web 控制台「④ 记忆」页
可查看/新增/编辑/删除。

**实现**：`brain/memory.py` 的 `MemoryStore`（SQLite 内置 sqlite3，零新依赖，
文件默认 `data/memory.db`），表 `memories(id, content, kind, created_at,
updated_at, last_used_at, use_count)`，`kind` ∈ `fact`（事实）/`preference`
（喜好）/`note`（手动备注）。

**工作流**：

1. **注入**：每次 `Brain.decide()` 组装 messages 时，把 `MemoryStore.recall()`
   的常用记忆（`inject_limit` 条）以 `[长期记忆]` 节注入 system 消息，让 LLM
   跨会话记住事实（注入后标记 use_count+1，越常用越靠前）。
2. **自动提取**：每轮决策完成后，按 `extract_interval`（默认每 3 轮）**异步、
   低优先级**调同一 LLM 从"最近对话 + 回复"提取值得长期记住的事实（输出
   `{"memories": [...]}`，宽松解析），写入 MemoryStore；失败静默降级不阻塞
   主循环；提取为空不写。
3. **裁剪**：超过 `max_items`（默认 200）自动删除最久未用的记忆。

**配置**（config.yaml `brain.memory`，Web「参数 → 大脑 → 长期记忆」）：

```yaml
brain:
  memory:
    enabled: true              # 总开关（false 时不注入不提取，旧行为不回归）
    db_path: data/memory.db    # SQLite 路径（相对项目根；改后需重启）
    max_items: 200             # 记忆上限，超出删最久未用（改后需重启）
    inject_limit: 5            # 每次注入 LLM 的记忆条数（热重载）
    auto_extract: true         # 对话后自动提取（消耗少量 LLM token，可关；热重载）
    extract_interval: 3        # 每 3 轮决策提取一次，省 token（热重载）
    extract_max_items: 5       # 每次提取最多写入条数（热重载）
```

**Web API**（本机 127.0.0.1 无鉴权，与现有接口一致）：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/memory?kind=fact` | 记忆列表（含 enabled/auto_extract/max_items） |
| POST | `/api/memory` | 新增 `{content, kind}` |
| PUT | `/api/memory/{id}` | 编辑 `{content, kind?}` |
| DELETE | `/api/memory/{id}` | 删除 |

**注意**：自动提取每次会额外消耗 LLM token（默认每 3 轮一次，每次最多 5 条）；
不需要可把 `auto_extract` 关掉，或调大 `extract_interval` 降低频率。SQLite 单机
写入，多进程并发访问同一库文件时建议只保留主 Agent 进程写（Web 与主进程同进程）。

### 6.1 动作校准（换 avatar 后必做，任务 3）

`mappings/avatar_default.yaml` 中的 emote 索引（VRCEmote 1~8）与 gesture 参数名
（Point/ThumbsUp/Wave）是**占位**，需按目标 avatar 的真实 Expressions 菜单与参数校准：

```powershell
conda activate vrchat-agent
python scripts/calibrate_actions.py            # 列出当前 avatar 真实参数 + 生成建议映射
python scripts/calibrate_actions.py --probe    # 额外对常见参数名执行 set_avatar_parameter 探测
```

脚本会拉起 OSC 桥接并调用 `get_avatar_parameters`，打印当前 avatar 真实参数名，
按关键词（Wave/Bow/Clap/Dance/Laugh/Point/ThumbsUp/Jump）自动关联语义生成建议映射。
把建议写入 `mappings/avatar_default.yaml`（或复制为 `mappings/avatar_<名字>.yaml`
后在 `action.mapping.path` 切换），重启 Agent 生效。

## 7. 测试（离线可跑）

```powershell
pip install -r requirements-dev.txt
pytest
```

所有外部依赖（网络/GPU/声卡/OSC/OBS）在测试中全部 mock，CI 或离线环境可跑。

## 8. 已知限制

- **说话人区分**（A-05）：WASAPI 捕获的是整体混合音频，无法精确区分说话人（P1 best-effort）
- **OSC 输入级盲驱动**（D）：无法读取游戏内真实状态，靠视觉通道补偿
- **云端依赖**：LLM/TTS 故障时降级为"提示音 + 不回应"，进程不崩溃
- **自抑制**：默认静音窗口方案（TTS 发声期间丢弃 ASR 输入），回声/啸叫需本机监听静音配合
- **动作映射表为占位**（Q9）：emote/gesture 参数名需按目标 avatar 校准——
  换 avatar 后运行 `python scripts/calibrate_actions.py` 自动获取真实参数并生成建议映射
- **目标机规格**（Q8）：默认 Windows 10/11 + NVIDIA GPU（显存 ≥ 4GB）；CPU 回退可用但延迟不保证

## 9. 合规声明

本项目仅使用：
- VRChat 官方 **OSC 协议**（输入级指令：表情/参数/移动/聊天框）
- 系统级音频回环（WASAPI）+ 虚拟声卡（VB-Audio）
- 系统级视频回环（OBS Studio / 虚拟摄像头）

**不包含**任何游戏内存读取、注入、修改类 Mod（MelonLoader 等）。新增依赖入库前需通过合规审查（架构评审关卡）。
