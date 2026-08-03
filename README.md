# VRChat 全自主具身 AI Agent

让 AI 像真人玩家一样在 VRChat 中游玩：**听**（WASAPI 回环 + VAD + 本地 ASR）→ **看**（窗口截图 / OBS 取帧 + 视觉闭环校准）→ **想**（OpenAI 兼容 LLM，多模态）→ **说 + 动**（云端/本地 TTS → 虚拟声卡 + OSC 表情/移动/聊天框）。

> 技术栈：**Python 3.11+ 主编排（asyncio 事件驱动）** + **Node 仅作 OSC 桥接子进程**（复用 `@iflow-mcp/krekun-vrchat-mcp-osc`，postinstall 补丁链扩展）。
> 合规红线：仅使用 **VRChat 官方 OSC 协议** + **系统级音频/视频回环**；**严禁**任何内存读取 Mod（MelonLoader 等）。

---

## 1. 核心能力一览

| 能力 | 实现 | 引擎/切换 |
| --- | --- | --- |
| **听（ASR）** | WASAPI 回环 → VAD → 本地转写 → 自抑制/过滤 | `faster-whisper`（默认）\| `Dolphin small CTC`（sherpa-onnx，中文更准），`config.asr.engine` 切换 |
| **看（视觉）** | OBS WebSocket / OpenCV / 窗口截图 + 多模态 LLM 理解 | `config.vision.sensor.backend` |
| **视觉校准** | 多步静默闭环：截图→VLM 判偏移→pan/tilt 微调→迭代，最多 3 轮，中间步不发 TTS | `core/visual_calibration.py` |
| **想（LLM）** | OpenAI 兼容（默认小米 MiMo v2.5），JSON 决策 + 宽松解析 + 记忆 + 人设 | `config.brain.llm` |
| **说（TTS）** | 多引擎：`mimo`（默认）/ `cosyvoice`（DashScope 音色克隆）/ `edge` / `cloud` / `local_clone` | `config.speech.tts.provider` 切换 |
| **音色克隆** | CosyVoice `VoiceEnrollmentService`：Web 上传/URL 克隆 → 保存 voice_id → 后续直接复用 | Web「参数 → 音色克隆」 |
| **动（动作）** | MCP → OSC：28 个工具 + YAML 语义映射（69 条），蹲/躺走键盘注入（OSC 无 Crouch/Prone 地址） | `mappings/avatar_default.yaml` |
| **管道并发** | 三段流水线（ASR→LLM→TTS 独立，Semaphore 限流）+ 互斥输出槽，chatbox/动作/语音同刻起跑 | `core/output_slot.py`、`core/utterance.py` |
| **Web 控制台** | FastAPI + SSE：状态面板、流水线面板、参数热改、急停、跳过播放、音色克隆 | `web/` |

---

## 2. 快速开始（Windows）

### 环境要求
- **Python 3.11+**（推荐 conda：`conda create -n vrchat-agent python=3.12`）
- **Node.js 16+**（OSC 桥接需要）
- **VB-Audio Virtual Cable**（虚拟声卡，Agent 语音输出到 VRChat）
- **OBS Studio**（可选，视觉取帧首选；无 OBS 时自动回退窗口截图）
- **VRChat**（桌面版，需先启动并保持窗口）

### 一键启动

```powershell
cd vrchat_agent
.\scripts\run.ps1          # 检查环境 → 装依赖 → 起桥接 → 启动 Agent
```

或手动：

```powershell
# 1. 安装依赖（注意：Agent 实际跑在 conda vrchat-agent 环境，务必装到同一个环境！）
conda activate vrchat-agent
pip install -r requirements.txt

# 2. 准备 OSC 桥接（postinstall 会自动打 4 个补丁，注册 28 个 MCP 工具）
cd node_bridge
npm install

# 3. 配置密钥：复制 .env.example → .env，填入 API keys
#    LLM_API_KEY / MIMO_TTS_API_KEY / DASHSCOPE_API_KEY（cosyvoice 用）/ TTS_API_KEY

# 4. 启动
cd ..
python -m vrchat_agent.main
```

> ⚠️ **环境一致性**：Agent 进程用的 Python 必须是你装依赖的那个（`run.ps1` 优先用 conda 激活的环境）。装错环境会报 `ModuleNotFoundError`。

---

## 3. 配置（config.yaml）

所有配置均支持**热重载**（改文件或 Web 面板保存即生效，白名单字段）。常用项：

| 配置节 | 关键项 | 说明 |
| --- | --- | --- |
| `asr` | `engine: whisper \| dolphin` | 转写引擎；`dolphin` 用 `model_path`（如 `model/sherpa-small-dolphin`，含 model.onnx + tokens.txt）；`decoding_method: greedy_search \| beam_search` |
| `speech.tts` | `provider: mimo \| cosyvoice \| edge \| cloud \| clone` | TTS 引擎切换 |
| `speech.tts.tts_cosyvoice` | `api_key_env: DASHSCOPE_API_KEY`、`voice_id` | CosyVoice 模型 + 克隆音色 ID（Web 上克隆后自动回填） |
| `agent.pipeline` | `llm_semaphore / tts_semaphore / output_queue_max` | 并发限流与输出队列积压上限 |
| `agent.loop` | `think_timeout / tts_timeout / action_timeout` | 各阶段超时 |
| `action.mapping` | `path: mappings/avatar_default.yaml` | 动作语义映射表 |

### API Keys（`.env`，**绝不写入 config.yaml**）

```
LLM_API_KEY=...            # 云端 LLM（MiMo）
MIMO_TTS_API_KEY=...       # MiMo 克隆 TTS
DASHSCOPE_API_KEY=...      # 阿里云 DashScope（CosyVoice 克隆 TTS）
TTS_API_KEY=...            # 其他云端 TTS（provider=cloud 时）
```

---

## 4. 目录结构

```
vrchat_agent/
├── src/vrchat_agent/
│   ├── audio/        # 语音输入：asr(whisper/dolphin) / vad / filter / pipeline
│   ├── brain/        # LLM 决策：brain / prompts / schema / memory
│   ├── speech/       # 语音输出：tts_mimo / tts_cosyvoice / tts_edge / voice_clone / audio_output
│   ├── action/       # 动作：mcp_actuator / action_mapper / key_sender(键盘注入) / utterance
│   ├── core/         # 编排：agent_loop / output_slot / utterance / visual_calibration / state_machine
│   ├── vision/       # 视觉感知（OBS / OpenCV / 窗口截图）
│   ├── web/          # FastAPI 控制台 + SSE + 静态面板
│   ├── bridge/       # MCP 客户端（连 node_bridge）
│   └── main.py       # 入口（python -m vrchat_agent.main）
├── node_bridge/      # OSC 桥接（@iflow-mcp，postinstall 补丁链）
├── mappings/         # 动作语义映射 avatar_default.yaml
├── scripts/          # run / deploy / package / sync-release / emergency_stop / start_cosyvoice
├── config.yaml       # 主配置
├── requirements.txt / pyproject.toml
└── .env.example      # API keys 模板
```

---

## 5. 部署与同步

### 打包 + 一键部署

```powershell
# 打包（排除测试/开发文件，生成可分发目录）
.\scripts\package.ps1 -OutputDir "D:\vrchat-agent-release"

# 目标机部署（产物目录里）
.\deploy.bat          # 检查 Python → venv → pip install → npm install(补丁) → 环境变量提示
```

### 发布目录同步（开发 → 发布）

```powershell
.\scripts\sync-release.ps1 -Source . -Dest "D:\biaozhu\分享\vrchat-agent-release"
```

### 版本控制（GitHub）

```bash
git remote -v                            # origin → git@github.com:laomurong/vachat_agent.git
git add -A && git commit -m "..." && git push   # 增量同步（SSH key 认证，免密码）
```

> 仓库只跟踪**部署所需文件**（src/mappings/node_bridge/scripts/config.yaml 等）；`tests/`、`docs/`、`model/`（权重）、`.env`、日志、QA 残留均被 `.gitignore` 排除，不入库。

---

## 6. 常见问题

| 问题 | 处理 |
| --- | --- |
| `ModuleNotFoundError: No module named 'dashscope'/'sherpa_onnx'` | 依赖装错环境了——装进 **Agent 实际使用的那个 Python**（conda vrchat-agent 或 .venv），二选一别混 |
| 蹲下/躺下没反应 | 走的是键盘注入（VRChat 官方 OSC **没有** Crouch/Prone 地址）；确认 VRChat 窗口在运行（`action/key_sender.py` 找不到窗口只 WARNING） |
| `look:up/down` 不动 | 桌面端不绑 LookUp/Down 按键，已改为 `look_axis`（`/input/LookVertical`）驱动，需重新同步映射表 |
| 视觉校准不准 | 检查启动日志有无「截图尺寸与游戏窗口不一致」WARNING；保持窗口与截图区域一致，或重启 Agent 让校准 prompt 带上实际尺寸/FOV |
| 音色克隆报「dashscope 未安装」 | 同上，依赖装进实际环境 |
| 桥接工具数变 28 / wire guard 失败 | `node_bridge` 的 dist 被打回原版（如 `npm install --no-save` 后），重跑 `npm run postinstall` 恢复补丁链 |

---

## 7. 动作系统速查

- **MCP 工具 28 个** = 上游 12 + expand 补丁 16（`run/use_*/grab_*/drop_*/toggle_sit_stand/afk_toggle/panic_button/quick_menu_right/move_axis/look_axis/set_gesture/set_typing_indicator/send_raw_osc`）
- **语义映射 69 条**，`ActionKind` 9 个：avatar / chatbox / emote / gesture / input / jump / look / move / pose
- **蹲/躺（pose:crouch/prone）**：键盘注入 C/Z 键（OSC 不支持，走 `send_key` 伪工具）
- **上下视角（look:up/down/up_big/down_big）**：`look_axis`（`/input/LookVertical` 轴驱动）
- **自动复位**：动作结束语义完整（shutdown 真复位 / emergency_reset 16 项 / 按通道分槽），急停整体 5s 预算

---

## 8. 许可与声明

- 仅供个人学习与研究使用
- 使用本工具需遵守 VRChat 服务条款与各模型供应商（OpenAI/MiMo/DashScope）使用条款
- 不在任何环境中用于干扰其他玩家体验或破坏游戏公平性
