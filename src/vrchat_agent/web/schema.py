"""配置表单 schema 生成 + 校验/备份/原子写/reload 保存服务（M4 增量，W-04）。

写盘安全红线（共享知识约定第 2 条）：
`save_config` 是唯一写 config.yaml 的入口。顺序固定：
model_validate 校验（失败 400 不写盘）→ 剔除 api_key → 备份 config.yaml.bak
→ 临时文件 + os.replace 原子写 → Config.reload() → 返回 restart_required。
任何一步失败保证原文件可用。
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from ..config import AppConfig, Config

logger = logging.getLogger("vrchat_agent.web.schema")

# 敏感字段（dotted path）：GET /api/config 掩码；PUT 写盘时剔除/恢复
SENSITIVE_PATHS = [
    "brain.llm.api_key",
    "speech.tts.api_key",
    "speech.tts.tts_mimo.api_key",
    "speech.tts.tts_cosyvoice.api_key",
    "vision.sensor.obs_password",
]

# 枚举字段下拉选项（dotted path → 合法值列表，字符串形式）：
# GET /api/config/schema 以 select_options 透传给前端，index.html 的 renderField
# 命中该表时把字段渲染为 <select><option>（含「+ 自定义…」自由输入兜底），
# 未命中字段保持原行为（text / number / checkbox / textarea 等）。
# vad_mode 为 int 字段：选项用字符串 "0"~"3"，前端提交时转回 int（后端 pydantic
# lax 模式亦接受字符串数字，双保险）。language / lang 属"常见值"枚举：前端支持
# 选择「+ 自定义…」手动输入列表外的语言代码，故这里只列高频项。
SELECT_OPTIONS: dict[str, list[str]] = {
    # ---- ASR（faster-whisper / Dolphin）----
    "asr.engine": ["whisper", "dolphin"],
    "asr.model_size": [
        "tiny", "base", "small", "medium",
        "large-v1", "large-v2", "large-v3", "large-v3-turbo", "turbo",
        "distil-small.en", "distil-medium.en", "distil-large-v2", "distil-large-v3",
    ],
    "asr.device": ["cuda", "cpu"],
    "asr.compute_type": ["float16", "float32", "int8", "int8_float16", "int16"],
    "asr.language": ["auto", "zh", "en", "ja", "ko", "yue", "fr", "de", "es", "ru", "it", "pt"],
    "asr.decoding_method": ["greedy_search", "beam_search", "modified_beam_search"],
    "asr.vad_mode": ["0", "1", "2", "3"],
    # ---- Brain ----
    "brain.llm.provider": ["openai", "ollama", "vllm"],
    "brain.vision_mode": ["reference_only", "keyword_filter"],
    # ---- TTS ----
    "speech.tts.provider": ["cloud", "edge", "clone", "mimo", "cosyvoice"],
    "speech.tts.tts_clone.method": ["get", "post"],
    "speech.tts.tts_clone.lang": ["zh", "en", "ja", "ko"],
    # ---- Vision ----
    "vision.sensor.backend": ["obs_websocket", "opencv", "window_capture"],
    # ---- Logging ----
    "logging.level": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
}

# 热重载节（前缀匹配，历史兼容）：Config.reload() 实际动态生效的节级范围。
# 新实现（任务 1：A1）升级为叶子级白名单 HOT_RELOAD_PATHS（精确到字段），
# 本集合保留用于 schema 元数据展示与旧测试兼容。
HOT_RELOAD_SECTIONS = {"agent.loop", "agent.degradation"}

# 热重载叶子字段白名单（A1 真热加载范围，精确到 dotted path）：
# 仅以下字段由模块在运行时从 self._config 读取，保存后无需重启。
# 其余字段（设备/采样率/chunk/模型/provider/backend 等）属"需重启"（A2）。
HOT_RELOAD_PATHS = {
    # agent 主循环
    "agent.loop.think_timeout",
    "agent.loop.tts_timeout",
    "agent.loop.action_timeout",
    "agent.loop.latency_budget_s",
    "agent.interrupt.enabled",
    "agent.degradation.message",
    "agent.degradation.beep",
    "agent.chatbox_sync",
    # 对话队列（需求 4：AgentLoop 每次事件读取，保存即生效）
    "agent.utterance_queue.enabled",
    "agent.utterance_queue.max_batch",
    # brain：temperature/max_tokens/timeout/vision_enabled/persona
    "brain.llm.temperature",
    "brain.llm.max_tokens",
    "brain.llm.timeout",
    "brain.llm.vision_enabled",
    "brain.persona.name",
    "brain.persona.system_prompt",
    # 长期记忆（标准版）：Brain 每次决策/提取从 config 读取，保存即生效；
    # db_path / max_items 属装配期读取（需重启），不在此列
    "brain.memory.enabled",
    "brain.memory.inject_limit",
    "brain.memory.auto_extract",
    "brain.memory.extract_interval",
    "brain.memory.extract_max_items",
    # vad/audio 阈值（chunk_ms 需重启，不在此列）
    "asr.vad_mode",
    "asr.min_speech_ms",
    "asr.silence_ms",
    "asr.max_segment_s",
    "audio.input.silence_threshold",
    # ASR 误识别过滤（A-05：UtteranceFilter 每次转写后读取，保存即生效）
    "audio.input.filter_enabled",
    "audio.input.filter_blocklist",
    "audio.input.filter_exact_words",
    "audio.input.filter_min_length",
    # vision 采样
    "vision.sampling.fps",
    "vision.sampling.max_size",
    "vision.sampling.jpeg_quality",
    "vision.sampling.event_trigger",
    "vision.sampling.change_threshold",
    # speech/tts 合成参数（provider 需重启，不在此列）
    "speech.tts.voice",
    "speech.tts.edge_voice",
    "speech.tts.rate",
    "speech.tts.volume",
    "speech.tts.emotion_voice",
    "speech.tts.timeout",
    # 克隆语音 TTS 参数（LocalCloneTTSEngine 每次合成从 config 读取）
    "speech.tts.tts_clone.base_url",
    "speech.tts.tts_clone.method",
    "speech.tts.tts_clone.ref_audio_path",
    "speech.tts.tts_clone.ref_text",
    "speech.tts.tts_clone.lang",
    "speech.tts.tts_clone.timeout",
    # MiMo 克隆 TTS 参数（MimoTTSEngine 每次合成从 config 读取；model/api_key_env
    # 属需重启字段：model 需重启更稳妥，api_key 走 .env 需重启进程加载）
    "speech.tts.tts_mimo.base_url",
    "speech.tts.tts_mimo.ref_audio_path",
    "speech.tts.tts_mimo.timeout",
    # CosyVoice（DashScope）参数：voice_id/model/timeout 每次合成从 config 读取
    # （保存即生效，VoiceCloneManager.set_voice_id 也写回此字段）；api_key_env 属
    # 需重启字段（api_key 走 .env 需重启进程加载）
    "speech.tts.tts_cosyvoice.voice_id",
    "speech.tts.tts_cosyvoice.model",
    "speech.tts.tts_cosyvoice.timeout",
    # idle_instinct 空闲本能微动作循环（IdleInstinctLoop 每次 tick 从 config 读取，保存即生效）
    "agent.idle_instinct.enabled",
    "agent.idle_instinct.interval_min_sec",
    "agent.idle_instinct.interval_max_sec",
    "agent.idle_instinct.jitter_min_deg",
    "agent.idle_instinct.jitter_max_deg",
    "agent.idle_instinct.look_overshoot_prob",
    "agent.idle_instinct.hesitate_prob",
    "agent.idle_instinct.small_step_move_prob",
    "agent.idle_instinct.move_duration_min_sec",
    "agent.idle_instinct.move_duration_max_sec",
    # dashboard / logging
    "dashboard.refresh_ms",
    "logging.level",
}

# obs_password 掩码哨兵：等于该值时不写盘，恢复当前真实值
PASSWORD_MASK = "********"

_BACKUP_SUFFIX = ".bak"

# 写盘串行化锁（WBUG-1）：并发 PUT /api/config 时保护「备份→写临时→os.replace→
# reload」全程。Windows 下 os.replace 与另一线程的并发读/写文件句柄重叠会抛
# PermissionError (WinError 5) → HTTP 500；加锁后同进程内保存严格串行，杜绝竞态。
_SAVE_LOCK = threading.Lock()


class ConfigValidationError(Exception):
    """校验失败（HTTP 400，不写盘）。"""

    def __init__(self, detail: list[dict[str, Any]]) -> None:
        self.detail = detail
        super().__init__("validation_error")


class ConfigWriteError(Exception):
    """写盘 / reload 失败（HTTP 500，保证原文件可用）。"""


@dataclass
class SaveResult:
    """保存结果（对应 HTTP 200 响应体）。"""

    ok: bool = True
    reloaded: bool = False
    restart_required: list[str] = field(default_factory=list)
    path: str = ""


# ---------------------------------------------------------------------------
# 热重载 / 需重启 标注
# ---------------------------------------------------------------------------
def _iter_leaf_paths(model_cls: type[BaseModel], prefix: str = "") -> list[str]:
    """递归收集 pydantic 模型的叶子字段路径（dotted，按声明顺序）。"""
    paths: list[str] = []
    for name, field_info in model_cls.model_fields.items():
        cur = f"{prefix}.{name}" if prefix else name
        ann = field_info.annotation
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            paths.extend(_iter_leaf_paths(ann, cur))
        else:
            paths.append(cur)
    return paths


def _is_hot_reload(path: str) -> bool:
    """路径是否属于热重载字段（叶子级白名单；兼容旧节级前缀匹配）。"""
    if path in HOT_RELOAD_PATHS:
        return True
    for section in HOT_RELOAD_SECTIONS:
        if path == section or path.startswith(section + "."):
            return True
    return False


def restart_required_paths() -> list[str]:
    """返回"需重启生效"的字段路径（按 schema 顺序）。"""
    return [p for p in _iter_leaf_paths(AppConfig) if not _is_hot_reload(p)]


def schema_payload() -> dict[str, Any]:
    """GET /api/config/schema 响应：JSON Schema + 热重载/需重启/敏感元数据。"""
    return {
        "schema": AppConfig.model_json_schema(),
        "hot_reload": sorted(HOT_RELOAD_SECTIONS),
        "hot_reload_paths": sorted(HOT_RELOAD_PATHS),
        "restart_required": restart_required_paths(),
        "sensitive": SENSITIVE_PATHS,
        "select_options": SELECT_OPTIONS,
        "title": "VRChat Agent 配置",
    }


# ---------------------------------------------------------------------------
# 掩码 / 写盘载荷
# ---------------------------------------------------------------------------
def _mask_api_key(value: Any) -> str:
    """掩码 api_key：保留前 3 位 + *** + 后 4 位；空值返回空串。"""
    if not value:
        return ""
    s = str(value)
    if len(s) <= 8:
        return "***" + s[-4:]
    return s[:3] + "***" + s[-4:]


def _deep_copy(data: dict[str, Any]) -> dict[str, Any]:
    """深拷贝（仅用于掩码输出，不污染内部状态）。"""
    import copy

    return copy.deepcopy(data)


def _resolve_parent(data: dict[str, Any], path: str) -> tuple[dict[str, Any] | None, str]:
    """按 dotted path 定位父节点与末键。"""
    parts = path.split(".")
    if not parts:
        return None, ""
    node: Any = data
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            return None, parts[-1]
        node = node[part]
    return (node if isinstance(node, dict) else None), parts[-1]


def mask_config(data: dict[str, Any]) -> dict[str, Any]:
    """深拷贝并对敏感路径掩码（GET /api/config 用）。"""
    result = _deep_copy(data)
    for path in SENSITIVE_PATHS:
        node, key = _resolve_parent(result, path)
        if node is not None and key in node:
            if path.endswith("obs_password"):
                node[key] = PASSWORD_MASK if node[key] else ""
            else:
                node[key] = _mask_api_key(node[key])
    return result


def build_write_payload(validated: AppConfig) -> dict[str, Any]:
    """构造写盘字典：剔除三个 api_key 字段（keys 属 .env，W-07）。"""
    data = validated.model_dump()
    llm = data.get("brain", {}).get("llm", {})
    if isinstance(llm, dict):
        llm.pop("api_key", None)
    tts = data.get("speech", {}).get("tts", {})
    if isinstance(tts, dict):
        tts.pop("api_key", None)
        mimo = tts.get("tts_mimo")
        if isinstance(mimo, dict):
            mimo.pop("api_key", None)
        cosy = tts.get("tts_cosyvoice")
        if isinstance(cosy, dict):
            cosy.pop("api_key", None)
    return data


# ---------------------------------------------------------------------------
# 保存服务（唯一写盘入口）
# ---------------------------------------------------------------------------
def _validation_detail(exc: ValidationError) -> list[dict[str, Any]]:
    """把 pydantic ValidationError 转为字段级 detail（HTTP 400 响应体）。"""
    details: list[dict[str, Any]] = []
    for err in exc.errors():
        details.append(
            {
                "loc": [str(p) for p in err.get("loc", [])],
                "msg": err.get("msg", ""),
                "type": err.get("type", ""),
            }
        )
    return details


def save_config(config: Config, data: dict[str, Any]) -> SaveResult:
    """校验 → 剔除 api_key → 备份 .bak → 原子写 → reload → SaveResult。

    校验失败抛 ConfigValidationError（HTTP 400，不写盘）；
    写盘/reload 失败抛 ConfigWriteError（HTTP 500，原文件保持可用）。
    """
    # 1) 先校验：失败抛 400，绝不写盘
    try:
        validated = AppConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigValidationError(_validation_detail(exc)) from exc

    # 2) 构造写盘字典：剔除 api_key；obs_password 掩码哨兵恢复现值
    payload = build_write_payload(validated)
    sensor = payload.get("vision", {}).get("sensor", {})
    if isinstance(sensor, dict) and sensor.get("obs_password") == PASSWORD_MASK:
        sensor["obs_password"] = config.get("vision.sensor.obs_password") or ""

    path = Path(config.path)
    backup = path.with_name(path.name + _BACKUP_SUFFIX)

    # 写盘全程加锁：并发保存串行化（WBUG-1，Windows os.replace 竞态 PermissionError）。
    # 校验（第 1 步）与载荷构造（第 2 步）为纯 CPU 且不碰文件，放锁外保证非法值
    # 快速 400 返回、不排队等待。
    with _SAVE_LOCK:
        # 3) 备份（保留最近一份）
        try:
            if path.exists():
                backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        except OSError as exc:
            logger.exception("配置备份失败", extra={"event": "web_error"})
            raise ConfigWriteError(f"备份失败: {exc}") from exc

        # 4) 临时文件 + os.replace 原子写（避免写一半损坏）
        tmp_path: str | None = None
        try:
            fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix="config.", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                yaml.safe_dump(payload, fh, allow_unicode=True, sort_keys=False)
            os.replace(tmp_path, path)
        except OSError as exc:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            logger.exception("配置写盘失败", extra={"event": "web_error"})
            raise ConfigWriteError(f"写盘失败: {exc}") from exc

        # 5) reload；失败回滚备份
        reloaded = False
        try:
            config.reload()
            reloaded = True
        except Exception as exc:
            logger.exception("配置 reload 失败，回滚备份", extra={"event": "web_error"})
            try:
                if backup.exists():
                    backup.replace(path)
            except OSError:
                pass
            raise ConfigWriteError(f"reload 失败: {exc}") from exc

    return SaveResult(ok=True, reloaded=reloaded, restart_required=restart_required_paths(), path=str(path))
