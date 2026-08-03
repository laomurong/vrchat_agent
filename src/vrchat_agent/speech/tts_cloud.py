"""C 语音输出：CloudTTSEngine（云端 TTS，默认实现，C-01）。

走 OpenAI 兼容 audio/speech 端点（与 LLM 同一 SDK），
emotion 通过 emotion_voice 映射切换音色（P1：C-03）。
"""

from __future__ import annotations

import asyncio
import logging

from .tts import TTSEngine, TTSError, AudioClip, decode_audio_bytes

logger = logging.getLogger("vrchat_agent.speech.tts.cloud")

# 未配置 key 时 AsyncOpenAI 使用的占位值（避免 SDK 抛空 key 参数错误）
_EMPTY_KEY = "EMPTY"


def _classify_cloud_error(exc: Exception) -> str:
    """把 cloud TTS 异常翻译成用户可定位的中文原因（无 key / 端点不支持 / 认证 / 解码）。"""
    status = getattr(exc, "status_code", None)
    if status is not None:
        try:
            code = int(status)
        except (TypeError, ValueError):
            code = None
        if code in (401, 403):
            return f"认证失败（HTTP {code}）：TTS_API_KEY 无效或无权限"
        if code == 404:
            return (
                "端点不支持（HTTP 404）：当前 base_url 未提供 audio/speech 端点"
                "（多为本地 Ollama/LM Studio 等仅聊天端点）"
            )
        if code >= 500:
            return f"服务端错误（HTTP {code}）：{exc}"
        return f"HTTP {code}: {exc}"
    raw = str(exc).strip()
    return raw if raw else type(exc).__name__


class CloudTTSEngine(TTSEngine):
    """云端 TTS（OpenAI 兼容 audio 端点）。

    热重载（A1）：传入 config 引用后，synthesize() 每次从 ``config.speech.tts.*``
    读取 voice / emotion_voice / timeout，改音色/情感映射无需重启；未传 config
    时沿用装配期缓存值。
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str = "tts-1",
        voice: str = "alloy",
        timeout: float = 10.0,
        emotion_voice: dict[str, str] | None = None,
        config=None,
    ) -> None:
        self._base_url = base_url or None
        # 归一化：None/空串 → _EMPTY_KEY（synthesize 前会显式检查并给出清晰错误）
        self._api_key = (api_key or "").strip() or _EMPTY_KEY
        self._model = model
        self._voice = voice
        self._timeout = timeout
        self._emotion_voice = emotion_voice or {}
        self._config = config
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(base_url=self._base_url, api_key=self._api_key)
        return self._client

    # ---- 热重载读取（异常回退装配期缓存值） ----
    def _cfg_voice(self) -> str:
        if self._config is not None:
            try:
                return str(self._config.speech.tts.voice or self._voice)
            except Exception:
                pass
        return self._voice

    def _cfg_emotion_voice(self) -> dict[str, str]:
        if self._config is not None:
            try:
                return dict(self._config.speech.tts.emotion_voice or {}) or self._emotion_voice
            except Exception:
                pass
        return self._emotion_voice

    def _cfg_timeout(self) -> float:
        if self._config is not None:
            try:
                return float(self._config.speech.tts.timeout)
            except Exception:
                pass
        return self._timeout

    async def synthesize(self, text: str, emotion: str | None = None) -> AudioClip:
        if not text or not text.strip():
            raise TTSError("TTS 文本为空")
        if self._api_key == _EMPTY_KEY:
            # 装配层已保证"无 key 用 edge"；此处为直接构造 CloudTTSEngine 时的兜底，
            # 给出清晰原因而不是带着 EMPTY key 发请求。
            raise TTSError("云端 TTS 未配置 API key（请设置 TTS_API_KEY，或使用 provider: edge）")
        client = self._ensure_client()
        voice = self._cfg_emotion_voice().get(emotion or "", self._cfg_voice())
        timeout = self._cfg_timeout()
        try:
            resp = await asyncio.wait_for(
                client.audio.speech.create(
                    model=self._model,
                    voice=voice,
                    input=text,
                    response_format="mp3",
                ),
                timeout=timeout,
            )
            data = resp.content  # bytes
            if not data:
                raise TTSError("云端 TTS 返回空音频")
            clip = decode_audio_bytes(data)
            logger.info("云端 TTS 合成完成: %d 字符, %.2fs", len(text), clip.duration)
            return clip
        except asyncio.TimeoutError:
            raise TTSError(f"云端 TTS 超时（>{timeout}s）") from None
        except TTSError:
            # 已带明确原因的 TTS 错误（含解码失败 + 数据前缀）直接上抛，
            # 由 FallbackTTSEngine 捕获并回退 edge。
            raise
        except Exception as exc:
            raise TTSError(f"云端 TTS 失败: {_classify_cloud_error(exc)}") from exc
