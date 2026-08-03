"""VRChat 全自主具身 AI Agent。

Python 主编排（asyncio 事件驱动）+ Node 仅作 OSC 桥接子进程（复用 vrchat-mcp-osc）。

模块划分（与 7 大模块一一对应）：
- audio/    A 语音输入：WASAPI 回环 → VAD → faster-whisper ASR → 自抑制
- brain/    B 大脑：LLM 决策 JSON + pydantic 校验 + 上下文 + 人设
- speech/   C 语音输出：TTS 双实现 → VB-Audio 虚拟声卡
- action/   D 动作控制：Actuator 抽象 + MCP OSC 实现 + 语义动作映射
- vision/   E 视觉感知：OBS WebSocket 取帧 + 低频采样
- core/     F 编排主循环：状态机 + 打断 + 急停 + 仪表盘
- config.py + logging_setup.py + dashboard.py  G 配置与可观测
"""

__version__ = "0.1.0"
