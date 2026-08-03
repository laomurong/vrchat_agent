"""CosyVoice 2.0-0.5B 克隆语音推理服务（任务 1 增量）。

- model_loader：模型文件检查 / 字段归一化 / WAV 编码 / 惰性模型加载（离线可测）；
- server：轻量 FastAPI 服务（POST /tts、/tts/upload、GET /health），
  兼容 Agent LocalCloneTTSEngine 发送的 GPT-SoVITS 命名字段。

本包刻意不在 __init__ 导入 torch/cosyvoice：重依赖只在 model_loader 的
CosyVoiceModelContext.load() 内惰性导入，未装时给出清晰中文错误，
不污染 Agent 主进程的导入面（Web/编排可正常 import vrchat_agent）。
"""
