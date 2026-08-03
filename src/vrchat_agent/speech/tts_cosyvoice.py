"""C 语音输出：CosyVoiceTTSEngine（阿里云 DashScope CosyVoice v3.5-plus 云端 TTS）。

与 CloudTTSEngine（OpenAI 兼容 audio/speech）/ MimoTTSEngine（小米 MiMo 克隆 TTS）
都不同：本引擎走 **DashScope SDK 的 SpeechSynthesizer**（WebSocket 协议）合成：

    from dashscope.audio.tts_v2 import SpeechSynthesizer
    dashscope.base_websocket_api_url = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
    dashscope.base_http_api_url    = "https://dashscope.aliyuncs.com/api/v1"
    dashscope.api_key              = <从环境变量读取>
    synthesizer = SpeechSynthesizer(model="cosyvoice-v3.5-plus", voice=voice_id)
    audio = synthesizer.call(text)        # 同步阻塞，返回音频字节（默认 mp3）

设计要点：
- **api_key 绝不写明文**：只从环境变量 ``os.environ[api_key_env]`` 读取（默认
  ``DASHSCOPE_API_KEY``）；缺 key 在构造期即抛清晰 TTSError（提示设置环境变量），
  装配层捕获后自动降级 Edge（与 provider=clone 无 base_url 降级同风格）。
- **SDK 是同步阻塞**：`synthesizer.call()` 放 `asyncio.to_thread`（内部 executor）
  执行，不阻塞事件循环；外围用 asyncio.wait_for 限制超时。
- **voice_id 热重载**：合成时每次从 ``config.speech.tts.tts_cosyvoice.*`` 读取
  model / voice_id / timeout / api_key_env；voice_id 经 Web「音色克隆」设置并写回
  config 后，下次合成即时生效（无需重启、无需重新克隆）。
- **惰性 import**：缺 dashscope 包 → TTSError（提示 ``pip install dashscope``）；
  合成失败/空音频 → TTSError 包装，由装配层 FallbackTTSEngine 自动回退 Edge。
"""

from __future__ import annotations

import asyncio
import logging
import os

from .tts import AudioClip, TTSEngine, TTSError, decode_audio_bytes

logger = logging.getLogger("vrchat_agent.speech.tts.cosyvoice")

# 默认模型（DashScope CosyVoice v3.5-plus）
TARGET_MODEL = "cosyvoice-v3.5-plus"
# 默认 API key 环境变量名（key 走 .env，绝不写入 config.yaml / 代码）
_DEFAULT_API_KEY_ENV = "DASHSCOPE_API_KEY"
_DEFAULT_TIMEOUT = 60.0
# DashScope 地域端点（中国大陆通用域名；如迁移业务空间专属域名改配置即可）
_WS_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
_HTTP_URL = "https://dashscope.aliyuncs.com/api/v1"


def _configure_dashscope(api_key: str):
    """配置 dashscope 全局地址与 api_key（惰性 import；缺包抛 TTSError）。"""
    try:
        import dashscope
    except ImportError as exc:
        raise TTSError("dashscope 未安装（pip install dashscope）") from exc
    dashscope.base_websocket_api_url = _WS_URL
    dashscope.base_http_api_url = _HTTP_URL
    dashscope.api_key = api_key
    return dashscope


class CosyVoiceTTSEngine(TTSEngine):
    """阿里云 DashScope CosyVoice v3.5-plus 云端 TTS（音色克隆 voice_id 驱动）。

    热重载（与 Mimo/Cloud 一致）：传入 config 引用后，synthesize() 每次从
    ``config.speech.tts.tts_cosyvoice.*`` 读取 model / voice_id / timeout /
    api_key_env，改 voice_id 无需重启（provider 切换属"需重启"字段）。
    api_key 每次从环境变量实时读取（热重载 key 也即时生效）。
    """

    def __init__(
        self,
        api_key_env: str = _DEFAULT_API_KEY_ENV,
        model: str = TARGET_MODEL,
        voice_id: str = "",
        timeout: float = _DEFAULT_TIMEOUT,
        config=None,
    ) -> None:
        self._api_key_env = (api_key_env or _DEFAULT_API_KEY_ENV).strip() or _DEFAULT_API_KEY_ENV
        self._model = (model or TARGET_MODEL).strip() or TARGET_MODEL
        self._voice_id = voice_id
        self._timeout = float(timeout)
        self._config = config
        # 构造期校验 key：缺则抛清晰 TTSError（提示设置环境变量）。
        # 装配层（build_tts_engine）捕获后降级 Edge；测试可 monkeypatch 环境变量。
        self._api_key = self._read_api_key()

    # ---- 环境变量 / 热重载读取（异常回退装配期缓存值） ----
    def _read_api_key(self) -> str:
        """从环境变量读 API key（绝不写明文）；缺则抛清晰 TTSError。"""
        env = self._cfg_api_key_env()
        key = os.environ.get(env, "").strip()
        if not key:
            raise TTSError(
                f"CosyVoice TTS 未配置 API key：请在 .env 设置环境变量 {env}"
                "（阿里云百炼 DashScope 密钥，形如 sk-xxx）"
            )
        return key

    def _cfg(self, key: str, default):
        """从 config.speech.tts.tts_cosyvoice 读取字段；异常/未配置回退缓存值。"""
        if self._config is not None:
            try:
                value = getattr(self._config.speech.tts.tts_cosyvoice, key)
                if value is not None and str(value).strip() != "":
                    return value
            except Exception:
                pass
        return default

    def _cfg_api_key_env(self) -> str:
        return str(self._cfg("api_key_env", self._api_key_env)).strip() or _DEFAULT_API_KEY_ENV

    def _cfg_model(self) -> str:
        return str(self._cfg("model", self._model)).strip() or TARGET_MODEL

    def _cfg_voice_id(self) -> str:
        return str(self._cfg("voice_id", self._voice_id)).strip()

    def _cfg_timeout(self) -> float:
        try:
            return float(self._cfg("timeout", self._timeout))
        except (TypeError, ValueError):
            return self._timeout

    # ---- 同步合成（供 to_thread 调用，不阻塞事件循环） ----
    def _synthesize_sync(self, text: str, api_key: str, model: str, voice_id: str) -> bytes:
        """同步调用 DashScope SpeechSynthesizer，返回音频字节（mp3）。"""
        _configure_dashscope(api_key)
        try:
            from dashscope.audio.tts_v2 import SpeechSynthesizer
        except ImportError as exc:
            raise TTSError("dashscope 未安装（pip install dashscope）") from exc
        synthesizer = SpeechSynthesizer(model=model, voice=voice_id)
        audio = synthesizer.call(text)
        if audio is None:
            raise TTSError("CosyVoice TTS 返回空音频（call() 返回 None）")
        data = bytes(audio)
        if not data:
            raise TTSError("CosyVoice TTS 返回空音频")
        return data

    async def synthesize(self, text: str, emotion: str | None = None) -> AudioClip:
        """把文本发到 DashScope CosyVoice，解码为 AudioClip（16k mono float32）。

        - api_key / model / voice_id / timeout 每次从 config 或环境变量实时读取；
        - SDK 同步调用放 asyncio.to_thread，外围 asyncio.wait_for 限时；
        - 音频字节（mp3）→ decode_audio_bytes → AudioClip。
        """
        if not text or not text.strip():
            raise TTSError("TTS 文本为空")
        api_key = self._read_api_key()
        model = self._cfg_model()
        voice_id = self._cfg_voice_id()
        timeout = self._cfg_timeout()

        if not voice_id:
            raise TTSError(
                "CosyVoice TTS 未配置 voice_id：请先在 Web 控制台「音色克隆」用参考音频"
                "创建并设为当前音色，或在 config.yaml 配置 speech.tts.tts_cosyvoice.voice_id"
            )
        try:
            data = await asyncio.wait_for(
                asyncio.to_thread(self._synthesize_sync, text, api_key, model, voice_id),
                timeout=timeout,
            )
        except TimeoutError:
            raise TTSError(f"CosyVoice TTS 超时（>{timeout}s）") from None
        except TTSError:
            raise
        except Exception as exc:
            raise TTSError(f"CosyVoice TTS 合成失败: {exc}") from exc

        clip = decode_audio_bytes(data)
        logger.info(
            "CosyVoice TTS 合成完成: %d 字符, %.2fs, voice=%s", len(text), clip.duration, voice_id
        )
        return clip
