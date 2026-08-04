"""结构化 JSON Lines 日志 + 用量统计（T01 / T05 增强）。

共享知识约定（架构文档第 8 节第 6 条）：
- JSON Lines，每行 {ts, level, module, event, message, ...extra}
- module 用模块短名（audio.input / brain / speech.tts / action.actuator / loop）
- 关键事件必须带 event 字段与统一枚举
- 急停、LLM 失败、ASR 失败为 ERROR 级
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

# 模块短名 → 便于日志检索
MODULE_SHORT = {
    "vrchat_agent": "main",
    "vrchat_agent.config": "config",
    "vrchat_agent.events": "events",
    "vrchat_agent.logging_setup": "logging",
    "vrchat_agent.audio.audio_input": "audio.input",
    "vrchat_agent.audio.vad": "audio.vad",
    "vrchat_agent.audio.asr": "audio.asr",
    "vrchat_agent.audio.self_suppression": "audio.suppress",
    "vrchat_agent.audio.pipeline": "audio.pipeline",
    "vrchat_agent.vision.vision_sensor": "vision",
    "vrchat_agent.vision.obs_sensor": "vision.obs",
    "vrchat_agent.vision.sampler": "vision.sampler",
    "vrchat_agent.vision.event_detector": "vision.detect",
    "vrchat_agent.brain.llm_client": "brain.llm",
    "vrchat_agent.brain.brain": "brain",
    "vrchat_agent.brain.context": "brain.context",
    "vrchat_agent.brain.schema": "brain.schema",
    "vrchat_agent.brain.prompts": "brain.prompts",
    "vrchat_agent.speech.tts": "speech.tts",
    "vrchat_agent.speech.tts_edge": "speech.tts.edge",
    "vrchat_agent.speech.tts_cloud": "speech.tts.cloud",
    "vrchat_agent.speech.audio_output": "speech.output",
    "vrchat_agent.action.actuator": "action.actuator",
    "vrchat_agent.action.mcp_actuator": "action.actuator",
    "vrchat_agent.action.action_mapper": "action.mapper",
    "vrchat_agent.action.action_schema": "action.schema",
    "vrchat_agent.bridge.mcp_client": "action.bridge",
    "vrchat_agent.core.agent_loop": "loop",
    "vrchat_agent.core.interrupt": "loop.interrupt",
    "vrchat_agent.core.emergency": "loop.emergency",
    "vrchat_agent.core.dashboard": "dashboard",
    "vrchat_agent.web.server": "web",
    "vrchat_agent.web.status": "web",
    "vrchat_agent.web.schema": "web",
}


class JsonlHandler(logging.Handler):
    """JSON Lines 文件处理器：每行一条结构化日志。

    A-17：构造期一次性打开文件句柄并复用（此前每条 emit 都 ``with open`` 重开，
    高频日志下开销大）；提供 close() 关闭句柄，setup_logging 清 handlers 时自动
    关闭旧实例，程序退出由 logging 框架统一 shutdown 调用 close()，不泄漏句柄。
    """

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # A-17：句柄在构造期打开并复用（SIM115 忽略——若用 with 会立即关闭，
        # 正是本修复要避免的"每条日志重开文件"）。
        self._fh = open(self._path, "a", encoding="utf-8")  # noqa: SIM115

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if self._fh is None:
                return  # close() 后不再写（解释器退出时 logging.shutdown 可能再触发）
            entry: dict[str, Any] = {
                "ts": round(record.created, 3),
                "level": record.levelname,
                "module": getattr(record, "module_short", record.name),
                "event": getattr(record, "event", None),
                "message": record.getMessage(),
            }
            extra = getattr(record, "extra_fields", None)
            if isinstance(extra, dict):
                entry.update(extra)
            self._fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._fh.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        """关闭文件句柄（幂等）。"""
        try:
            if self._fh is not None:
                self._fh.close()
        finally:
            self._fh = None
            super().close()


def _short_name(name: str) -> str:
    """把 logger 名映射为模块短名。"""
    return MODULE_SHORT.get(name, name.rsplit(".", 1)[-1])


def setup_logging(level: str = "INFO", file: str | Path | None = None, stream: bool = True) -> logging.Logger:
    """初始化根 logger：可选 JSON Lines 文件 + 控制台输出。

    A-17：清 handlers 前先 close 旧实例（JsonlHandler 持有文件句柄，避免泄漏）。
    """
    root = logging.getLogger("vrchat_agent")
    root.setLevel(level.upper())
    for old in list(root.handlers):
        try:
            old.close()
        except Exception:
            pass
    root.handlers.clear()
    root.propagate = False

    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    if stream:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
    if file:
        jh = JsonlHandler(file)
        root.addHandler(jh)
    return root


def apply_logging_level(level: str = "INFO") -> None:
    """运行中调整日志级别（热重载 A1）：仅改根 logger 的 level，不清 handlers。

    供 Web 保存配置后调用（logging.level 变更即时生效，无需重启）。
    """
    try:
        root = logging.getLogger("vrchat_agent")
        root.setLevel(str(level).upper())
    except Exception:
        logging.getLogger("vrchat_agent.logging_setup").debug("apply_logging_level 失败", exc_info=True)


def get_logger(name: str) -> logging.Logger:
    """获取带模块短名的 logger（短名由 MODULE_SHORT 映射）。"""
    logger = logging.getLogger(name)
    logger.module_short = _short_name(name)  # type: ignore[attr-defined]
    return logger


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, message: str = "", **extra: Any) -> None:
    """打印带 event 字段的结构化日志。"""
    logger.log(level, message, extra={"event": event, "extra_fields": extra})


class UsageTracker:
    """云端用量统计（T05：G-05）。记录 LLM/TTS 调用次数、token 与预估成本。"""

    def __init__(self) -> None:
        self._llm_calls: int = 0
        self._llm_prompt_tokens: int = 0
        self._llm_completion_tokens: int = 0
        self._tts_calls: int = 0
        self._tts_chars: int = 0
        self._cost: float = 0.0

    def record_llm(self, prompt_tokens: int = 0, completion_tokens: int = 0, cost: float = 0.0) -> None:
        """记录一次 LLM 调用。"""
        self._llm_calls += 1
        self._llm_prompt_tokens += int(prompt_tokens or 0)
        self._llm_completion_tokens += int(completion_tokens or 0)
        self._cost += float(cost or 0.0)

    def record_tts(self, chars: int = 0, cost: float = 0.0) -> None:
        """记录一次 TTS 调用。"""
        self._tts_calls += 1
        self._tts_chars += int(chars or 0)
        self._cost += float(cost or 0.0)

    def summary(self) -> dict[str, Any]:
        """用量摘要（供仪表盘展示）。"""
        return {
            "llm_calls": self._llm_calls,
            "llm_prompt_tokens": self._llm_prompt_tokens,
            "llm_completion_tokens": self._llm_completion_tokens,
            "tts_calls": self._tts_calls,
            "tts_chars": self._tts_chars,
            "cost": round(self._cost, 4),
        }

    def reset(self) -> None:
        self.__init__()
