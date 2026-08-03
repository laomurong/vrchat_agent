"""C 语音输出：VoiceCloneManager（DashScope CosyVoice 音色克隆管理 + 持久化）。

音色克隆（声音复刻）：给一段**公网可访问**的参考音频 URL，DashScope 会训练一个
克隆音色并返回 voice_id（形如 ``cosyvoice-v3.5-plus-<prefix>-<unique>``）。
后续合成直接用 voice_id 即可，**无需每次重新克隆**——这正是本管理器的核心目标：
把 voice_id 存下来复用。

设计要点：
- **create_voice**：`VoiceEnrollmentService.create_voice()` 提交后返回 voice_id，
  轮询 `query_voice()`（间隔可配）直到 status=OK（就绪）或 UNDEPLOYED（失败），
  超时（默认 30 次 × 10s）抛清晰 TTSError；
- **持久化选 data/voices.json 而非 config.yaml**（简单性考量）：
  ① voice_id 是运行时增长的**数据**（可克隆多个音色），不属于静态配置；
  ② config.yaml 由 Web 表单 schema 驱动回写，塞运行时数据会污染表单/写盘逻辑；
  ③ 独立 json 读写简单、无热重载依赖。结构：
     {"current_voice_id": str, "voices": [{"id","model","prefix","created_at"}]}
  config.yaml 的 ``speech.tts.cosyvoice.voice_id`` 仍保留为"静态指定"入口：
  引擎合成时优先读 config 里的 voice_id；Web「设为当前音色」会**同步写回** config
  （经 schema.save_config 原子写 + reload），保证引擎热重载即时复用、重启不丢。
- **api_key 走环境变量**（DASHSCOPE_API_KEY），绝不写明文；
- **惰性 import + to_thread**：SDK 同步调用放线程执行，不阻塞事件循环。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from .tts import TTSError
from .tts_cosyvoice import _DEFAULT_API_KEY_ENV, TARGET_MODEL, _configure_dashscope

logger = logging.getLogger("vrchat_agent.speech.tts.voice_clone")

# 轮询默认参数（30 次 × 10s = 最长等待 300s）
_DEFAULT_MAX_POLLS = 30
_DEFAULT_POLL_INTERVAL = 10.0


class VoiceCloneManager:
    """CosyVoice 音色克隆管理：创建 / 轮询状态 / 列表 / 当前音色 / 持久化。

    config：可选。注入后，set_voice_id() 会把 voice_id 同步写回
    ``config.speech.tts.cosyvoice.voice_id``（原子写盘 + reload），让持有 config
    引用的 CosyVoiceTTSEngine 下次合成直接复用（热重载，无需重启）。
    """

    def __init__(
        self,
        api_key_env: str = _DEFAULT_API_KEY_ENV,
        data_file: str | Path | None = None,
        model: str = TARGET_MODEL,
        config=None,
        max_polls: int = _DEFAULT_MAX_POLLS,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
    ) -> None:
        self._api_key_env = (api_key_env or _DEFAULT_API_KEY_ENV).strip() or _DEFAULT_API_KEY_ENV
        self._model = (model or TARGET_MODEL).strip() or TARGET_MODEL
        self._config = config
        self._max_polls = max(1, int(max_polls))
        self._poll_interval = max(0.1, float(poll_interval))
        # 持久化文件：默认 <项目根>/data/voices.json（data_file 可注入 tmp_path 测试）
        if data_file is None:
            # voice_clone.py 位于 src/vrchat_agent/speech/ → parents[3] 即项目根
            project_root = Path(__file__).resolve().parents[3]
            data_file = project_root / "data" / "voices.json"
        self._data_file = Path(data_file)
        self._data: dict[str, Any] = {"current_voice_id": "", "voices": []}
        self._lock = threading.Lock()  # 并发 clone/select 写盘保护
        self._load()

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def _load(self) -> None:
        try:
            if self._data_file.exists():
                raw = json.loads(self._data_file.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self._data = {
                        "current_voice_id": str(raw.get("current_voice_id", "") or ""),
                        "voices": raw.get("voices", []) if isinstance(raw.get("voices", []), list) else [],
                    }
        except Exception:
            logger.warning("voices.json 解析失败，使用空状态: %s", self._data_file, exc_info=True)

    def _save(self) -> None:
        """原子写 data/voices.json（临时文件 + os.replace）。"""
        self._data_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._data_file.with_name(self._data_file.name + ".tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._data_file)

    # ------------------------------------------------------------------
    # 环境变量 / 服务
    # ------------------------------------------------------------------
    def _api_key(self) -> str:
        key = os.environ.get(self._api_key_env, "").strip()
        if not key:
            raise TTSError(
                f"音色克隆未配置 API key：请在 .env 设置环境变量 {self._api_key_env}"
            )
        return key

    def _service(self):
        """创建 VoiceEnrollmentService 实例（惰性 import；缺包抛 TTSError）。"""
        _configure_dashscope(self._api_key())
        try:
            from dashscope.audio.tts_v2 import VoiceEnrollmentService
        except ImportError as exc:
            raise TTSError("dashscope 未安装（pip install dashscope）") from exc
        return VoiceEnrollmentService()

    # ------------------------------------------------------------------
    # 克隆
    # ------------------------------------------------------------------
    async def create_voice(
        self, target_model: str | None = None, prefix: str = "myvoice", audio_url: str = ""
    ) -> str:
        """用公网可访问的参考音频 URL 创建克隆音色，轮询到就绪后返回 voice_id。

        状态轮询：status=OK 就绪；UNDEPLOYED 失败；其余（部署中）继续等；
        超时（max_polls × poll_interval）抛清晰 TTSError。
        """
        model = (target_model or self._model).strip() or TARGET_MODEL
        prefix = (prefix or "myvoice").strip() or "myvoice"
        audio_url = str(audio_url or "").strip()
        if not audio_url:
            raise TTSError("音色克隆缺少参考音频 URL（body.audio_url）")
        if not audio_url.lower().startswith(("http://", "https://")):
            raise TTSError(
                "参考音频 URL 必须是公网可访问的 http(s) 地址（DashScope 需拉取音频文件，"
                "如 OSS 链接）；本地文件上传暂不支持"
            )

        service = self._service()
        loop = asyncio.get_running_loop()
        try:
            voice_id = await loop.run_in_executor(
                None,
                lambda: service.create_voice(target_model=model, prefix=prefix, url=audio_url),
            )
        except TTSError:
            raise
        except Exception as exc:
            raise TTSError(f"音色克隆创建失败: {exc}") from exc
        if not voice_id or not str(voice_id).strip():
            raise TTSError("音色克隆创建失败：未返回 voice_id")

        # 轮询 query_voice() 直到就绪 / 失败 / 超时
        last_status = ""
        for _ in range(self._max_polls):
            await asyncio.sleep(self._poll_interval)
            try:
                details = await loop.run_in_executor(
                    None, lambda vid=voice_id: service.query_voice(vid)
                )
                status = self._extract_status(details)
            except Exception:
                # 查询异常不中断轮询（部署中接口偶发未就绪）
                status = ""
            if status:
                last_status = status
            if status == "OK":
                self._record_voice(voice_id, model, prefix)
                logger.info("音色克隆就绪: %s (model=%s)", voice_id, model)
                return str(voice_id)
            if status == "UNDEPLOYED":
                raise TTSError(
                    f"音色克隆失败（status=UNDEPLOYED）：参考音频无法处理，voice_id={voice_id}。"
                    "请检查音频格式（≤30s 干净人声 wav/mp3）与 URL 可访问性"
                )
        raise TTSError(
            f"音色克隆超时（>{self._max_polls * self._poll_interval:.0f}s）："
            f"voice_id={voice_id} 状态={last_status or 'unknown'}。可稍后调用 "
            f"GET /api/voice/clone/{voice_id} 查询"
        )

    @staticmethod
    def _extract_status(details: Any) -> str:
        """从 query_voice() 返回对象提取 status（dict 或 list 兼容）。"""
        if isinstance(details, dict):
            return str(details.get("status", "") or "")
        if isinstance(details, list) and details:
            first = details[0]
            if isinstance(first, dict):
                return str(first.get("status", "") or "")
            if hasattr(first, "status"):
                return str(getattr(first, "status"))
        return str(getattr(details, "status", "") or "")

    async def query_voice_status(self, voice_id: str) -> dict[str, Any]:
        """查询单个克隆音色状态（Web 轮询用）：{voice_id, status}。"""
        voice_id = str(voice_id or "").strip()
        if not voice_id:
            raise TTSError("查询音色克隆状态缺少 voice_id")
        service = self._service()
        loop = asyncio.get_running_loop()
        try:
            details = await loop.run_in_executor(None, lambda: service.query_voice(voice_id))
        except TTSError:
            raise
        except Exception as exc:
            raise TTSError(f"查询音色克隆状态失败: {exc}") from exc
        return {"voice_id": voice_id, "status": self._extract_status(details) or "processing"}

    # ------------------------------------------------------------------
    # 列表 / 当前音色
    # ------------------------------------------------------------------
    def list_voices(self) -> list[dict[str, Any]]:
        """返回本地已保存的克隆音色列表（含当前选中标记）。"""
        current = self._data.get("current_voice_id", "")
        result = []
        for v in self._data.get("voices", []):
            item = dict(v)
            item["current"] = bool(v.get("id") == current and current)
            result.append(item)
        return result

    def get_voice_id(self) -> str:
        """当前生效的 voice_id：config 静态指定优先，否则取持久化的当前音色。"""
        cfg_id = ""
        if self._config is not None:
            try:
                cfg_id = str(self._config.speech.tts.tts_cosyvoice.voice_id or "").strip()
            except Exception:
                cfg_id = ""
        if cfg_id:
            return cfg_id
        return str(self._data.get("current_voice_id", "") or "")

    def set_voice_id(self, voice_id: str) -> bool:
        """设为当前音色：更新 voices.json；若已注入 config 则同步写回 config.yaml。

        返回 True 表示已生效（含 config 写盘成功或无需写盘）；config 写盘失败时
        仅记录日志，json 持久化仍生效。
        """
        voice_id = str(voice_id or "").strip()
        if not voice_id:
            raise TTSError("设为当前音色缺少 voice_id")
        with self._lock:
            self._data["current_voice_id"] = voice_id
            self._save()
        self._sync_to_config(voice_id)
        logger.info("已设为当前音色: %s", voice_id)
        return True

    def _record_voice(self, voice_id: str, model: str, prefix: str) -> None:
        """创建成功后把音色记入本地列表（去重），并设为当前音色。"""
        with self._lock:
            voices = self._data.get("voices", [])
            if not any(v.get("id") == voice_id for v in voices):
                voices.append(
                    {
                        "id": voice_id,
                        "model": model,
                        "prefix": prefix,
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
                self._data["voices"] = voices
            self._data["current_voice_id"] = voice_id
            self._save()
        self._sync_to_config(voice_id)

    def _sync_to_config(self, voice_id: str) -> None:
        """把 voice_id 写回 config.speech.tts.cosyvoice.voice_id（原子写盘 + reload）。

        这样 CosyVoiceTTSEngine（持有 config 引用）下次合成直接读到新 voice_id，
        无需重启、无需每次重新克隆。写盘失败仅记日志，不阻断 json 持久化。
        """
        if self._config is None:
            return
        try:
            cfg = self._config.speech.tts.tts_cosyvoice
            if str(cfg.voice_id or "").strip() == voice_id:
                return
            # 先更新内存 config 对象，再整体 dump 写盘（payload 才会含新 voice_id）
            cfg.voice_id = voice_id
            from ..web.schema import save_config

            result = save_config(self._config, self._config.model_dump())
            logger.info(
                "voice_id 已写回 config.yaml（reloaded=%s）: %s", result.reloaded, voice_id
            )
        except Exception:
            logger.warning("写回 config.yaml 失败（json 已保存，重启后需重设）: %s", voice_id, exc_info=True)
