# VRChat Agent 项目交接文档（2026-08-04）

> 用途：本文件是给**新会话的 AI（主理人）**的完整交接说明。新会话激活 SoftwareCompany 专家后，先读本文件 + `E:\vrc具身agent\.workbuddy\memory\MEMORY.md` + `E:\vrc具身agent\.workbuddy\memory\2026-08-04.md`，即可无缝接手本项目。用户已明确要求：**所有任务必须走团队子 agent 流程**（TeamCreate → 工程师 → QA → 同步 → 交付）。

---

## 一、当前状态（2026-08-04 21:23 定格）

- **Git**：`E:\vrc具身agent\vrchat_agent\`，HEAD=`0b05ee5`，工作区干净，已推送 GitHub（`git@github.com:laomurong/vrchat_agent.git`，SSH key `~/.ssh/id_ed25519_github`）
- **测试基线**：896 passed / 1 failed（live2 预存机器差异）/ 1 skipped
  - 运行：`.venv\Scripts\python.exe -m pytest` 或 `C:\ProgramData\miniconda3\envs\vrchat-agent\python.exe`
  - ⚠️ 实际运行环境是 conda `vrchat-agent`（`C:\ProgramData\miniconda3\envs\vrchat-agent\`），装依赖要装进 conda 不是 .venv！
- **发布目录**：`D:\biaozhu\分享\vrchat-agent-release\`（每次交付后必须同步生产文件，用 robocopy 镜像 src/ + node_bridge/scripts + mappings + 根配置文件，排除 tests/node_modules/__pycache__ 等）

## 二、待办任务（用户已拍板，按序执行）

### ① vrchatapi 集成（换头像/查好友/找世界/更新状态）
- 依赖 `vrchatapi`；新增 `src/vrchat_agent/vrcapi/`（client.py 懒加载登录+2FA+限速器 ≤1次/60s；actions.py 封装四能力）
- 动作映射：ActionKind 9→10（扩 `vrc`），avatar_default.yaml 增量加 `vrc:*` 映射，走 action_mapper 派发
- prompts.py 加 vrc 动作枚举说明；config.yaml 加 `vrcapi` 节（env 引用）
- 测试 `tests/test_qa_vrcapi.py`（mock SDK，不真登录）

### ② 记忆语义检索（embedding 替代关键词）
- 现有记忆：SQLite（data/memory.db）+ Web CRUD + 类型标签，关键词匹配
- 新增语义检索层（轻量本地 embedding，如 sentence-transformers paraphrase-multilingual-MiniLM；模型权重放 model/ 不入 git），写入存向量、查询余弦相似度 top-k
- 与现有 SQLite 兼容（BLOB 列或独立向量表），Web 增删改查不破坏；LLM 记忆注入切到语义检索（关键词兜底）
- 参考：Project Gabriel Remastered 的分层记忆+语义搜索设计

### ④ 空间导航（拓扑图+指纹+路径规划+自主探索）
- **核心约束**：VRChat OSC 无绝对坐标 → 只能**拓扑导航**（节点=视觉地标指纹，边=相对动作序列）
- 新增 `src/vrchat_agent/nav/`：landmark.py（截图像素指纹+回环检测）/ topology.py（SQLite 存节点边）/ planner.py（指纹匹配+BFS+每步重定位）/ explorer.py（自主探索，优先于 idle_instinct）
- prompts.py 加"去往某处/探索世界"意图；复用视觉校准闭环（core/visual_calibration.py）、ImageSampler、ScreenshotSensor
- 测试用 mock 截图/动作执行

## 三、Git 操作铁律（用户拍板，违反=事故）

本项目 git 出过 **5 次事故**（.git 消失/objects 被清/refs 丢失），根因 = 子 agent 沙箱执行 `git stash` 时，WorkBuddy safe-delete shim（NODE_OPTIONS 注入 genie-safe-delete.cjs）把 .git 内部文件误移回收站（中文路径 vrc具身agent 下）。

1. **调度子 agent 时任务说明必须写明**："禁止 git stash / git reset --hard / rm -rf .git / git gc 类危险操作"
2. **git 收尾（commit/push）统一由主理人在主会话执行**；子 agent 只写代码，报告改动文件清单
3. 每次交付后主理人必须 push 到 GitHub（远端是最终保险）
4. 子 agent 报 git 异常时，主理人在主会话 `git init -b main + git fetch origin + git reset --mixed origin/main` 恢复，不派子 agent 修

## 四、项目架构速览

- 管道：ASR(Whisper/Dolphin) → LLM(MiMo) → TTS(MiMo克隆/CosyVoice) → 输出槽并发（动作+语音+chatbox 三路 gather）
- 核心文件：`core/agent_loop.py`（流水线+输出槽+idle_instinct）、`core/output_slot.py`、`core/utterance.py`、`core/idle_instinct.py`（空闲本能微动）、`core/visual_calibration.py`（视觉校准闭环）
- 动作系统：MCP bridge（node_bridge/，28 工具=上游12+expand16，crouch/prone 走 key_sender 键盘注入）、`action/action_mapper.py`（ActionKind 9：avatar/chatbox/emote/gesture/input/jump/look/move/pose）、mappings/avatar_default.yaml（69 项）
- OSC 白名单：26 按钮+4 轴（无 Crouch/Prone，蹲躺=键盘注入 C/Z）；轴必须 float32
- Web：Mission Control 深空主题仪表盘（状态/参数Tab分组/日志占位/记忆），schema 驱动表单+SELECT_OPTIONS 下拉+中文 title 兜底
- 已接入：Dolphin ASR（sherpa-onnx，engine 可切）、CosyVoice TTS+音色克隆（voice_clone.py）、Idle Instinct、全项目审查 22 项修复（A-01~A-20，急停可恢复等）

## 五、近期已完成（避免重复）

- 8-04：全项目审查修复 22 项（P0 急停失能三连/P1 skip-直效/OBS-不落盘/P2 ASR-降级/热重载对齐/P3 SSRF-防护 等），896/1/1
- 8-04：TTS 颜文字过滤（tts.py _clean_tts_text）、ASR 噪音过滤（agent_loop.py _is_noise_text）、Web 中文标签（titleFromSchema）、Tab 分组排版、Mission Control 视觉重设计、Idle Instinct Loop
- 7 月底：动作/语音/chatbox 三路并发、蹲躺键盘注入、look up/down 改 look_axis、流水线并发化、P1/P2 增强项
