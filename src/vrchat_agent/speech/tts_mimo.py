"""C 语音输出：MimoTTSEngine（小米 MiMo 克隆语音 TTS，云服务，MiMo 接入增量）。

与 CloudTTSEngine（OpenAI 兼容 audio/speech 端点）**完全不同**：MiMo 克隆 TTS
走 **chat.completions + audio 参数**（官方调用方式，不是 /audio/speech）：

    messages=[{"role": "user", "content": ""},
              {"role": "assistant", "content": 要合成的文本}]   # 文本放 assistant
    audio={"format": "wav", "voice": "data:audio/wav;base64,{参考音频 base64}"}  # 零样本克隆
    → message.audio.data = base64 编码的 wav

设计要点：
- 用 openai SDK 的 AsyncOpenAI（async client），与 LLMClient/CloudTTSEngine 同栈，
  惰性 import（便于测试替换 sys.modules["openai"]）；
- 参考音频每次合成时读取（支持热重载 ref_audio_path）；缺失/读失败给出含路径的
  清晰 TTSError；
- 错误分级：认证 401/403（提示配 MIMO_TTS_API_KEY）、404（模型/端点不存在）、
  超时、网络连接失败、返回非音频（JSON/HTML 预览）；
- 热重载（A1）：传入 config 引用后，base_url / ref_audio_path / timeout（以及
  api_key / model）每次合成时从 ``config.speech.tts.tts_mimo.*`` 读取；base_url
  或 api_key 变化时 AsyncOpenAI client 自动重建（base_url 热重载真实生效）。
  model / api_key_env 属"需重启"字段（A2），热读不破坏行为，但 Web 保存时仍
  标注需重启。
"""

from __future__ import annotations

import asyncio
import base64
import logging

from .tts import AudioClip, TTSEngine, TTSError, decode_audio_bytes

logger = logging.getLogger("vrchat_agent.speech.tts.mimo")

_DEFAULT_BASE_URL = "https://api.xiaomimimo.com/v1"
_DEFAULT_MODEL = "mimo-v2.5-tts-voiceclone"
_EMPTY_KEY = "EMPTY"

# 合法音频魔数前缀（wav=RIFF；mp3=ID3/FFFB/FFF3），用于快速识别"返回的是音频"
_AUDIO_MAGIC_PREFIXES = (b"RIFF", b"ID3", b"\xff\xfb", b"\xff\xf3")


def _classify_mimo_error(exc: Exception, timeout: float) -> str:
    """把 MiMo 克隆 TTS 异常翻译成用户可定位的中文原因（错误分级）。

    - HTTP 401/403 → 认证失败（提示 MIMO_TTS_API_KEY）；
    - HTTP 404 → 模型/端点不存在（提示检查 model/base_url）；
    - HTTP >= 500 → 服务端错误；
    - 超时（asyncio.TimeoutError / SDK APITimeoutError）→ 超时秒；
    - 网络连接失败（SDK APIConnectionError）→ 连接失败；
    - 其余返回原始异常信息。
    """
    status = getattr(exc, "status_code", None)
    if status is not None:
        try:
            code = int(status)
        except (TypeError, ValueError):
            code = None
        if code in (401, 403):
            return f"认证失败（HTTP {code}）：MIMO_TTS_API_KEY 无效或无权限（请在 .env 配置）"
        if code == 404:
            return (
                f"模型/端点不存在（HTTP 404）：请确认 speech.tts.tts_mimo.model="
                f"{_DEFAULT_MODEL} 且 base_url 为 MiMo TTS 地址"
                "（MiMo 克隆 TTS 走 chat.completions + audio 参数，非 /audio/speech）"
            )
        if code >= 500:
            return f"MiMo 服务端错误（HTTP {code}）：{exc}"
        return f"MiMo TTS HTTP {code}: {exc}"
    raw = str(exc).strip()
    name = type(exc).__name__
    lowered = raw.lower()
    if name == "APITimeoutError" or "timed out" in lowered:
        return f"请求超时（>{timeout}s）"
    if name == "APIConnectionError" or "connection error" in lowered or "connection" in lowered:
        return f"网络连接失败: {raw}"
    return raw if raw else name


class MimoTTSEngine(TTSEngine):
    """小米 MiMo 克隆语音 TTS（chat.completions + audio 参数，零样本克隆）。

    热重载（A1）：传入 config 引用后，synthesize() 每次从
    ``config.speech.tts.tts_mimo.*`` 读取 base_url / ref_audio_path / timeout /
    api_key / model，改配置无需重启；未传 config 时沿用装配期缓存值。
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        ref_audio_path: str = "",
        timeout: float = 60.0,
        config=None,
    ) -> None:
        self._base_url = (base_url or _DEFAULT_BASE_URL).rstrip("/") or _DEFAULT_BASE_URL
        # 归一化：None/空串 → 空串（synthesize 前显式检查并给出清晰错误）
        self._api_key = (api_key or "").strip()
        self._model = model or _DEFAULT_MODEL
        self._ref_audio_path = ref_audio_path
        self._timeout = float(timeout)
        self._config = config
        self._client = None            # openai.AsyncOpenAI（惰性创建）
        self._client_base_url: str | None = None
        self._client_api_key: str | None = None

    # ---- 热重载读取（异常回退装配期缓存值） ----
    def _cfg(self, key: str, default):
        """从 config.speech.tts.tts_mimo 读取字段；异常/未配置回退缓存值。"""
        if self._config is not None:
            try:
                value = getattr(self._config.speech.tts.tts_mimo, key)
                if value is not None and str(value).strip() != "":
                    return value
            except Exception:
                pass
        return default

    def _cfg_base_url(self) -> str:
        return str(self._cfg("base_url", self._base_url)).strip().rstrip("/") or _DEFAULT_BASE_URL

    def _cfg_api_key(self) -> str:
        return str(self._cfg("api_key", self._api_key)).strip()

    def _cfg_model(self) -> str:
        return str(self._cfg("model", self._model)).strip() or _DEFAULT_MODEL

    def _cfg_ref_audio_path(self) -> str:
        return str(self._cfg("ref_audio_path", self._ref_audio_path)).strip()

    def _cfg_timeout(self) -> float:
        try:
            return float(self._cfg("timeout", self._timeout))
        except (TypeError, ValueError):
            return self._timeout

    def _ensure_client(self, base_url: str, api_key: str):
        """惰性创建 AsyncOpenAI 客户端；base_url/api_key 变化时自动重建。

        与 CloudTTSEngine 的区别：Cloud 的 client 首次创建后固化，base_url 热重载
        不生效；MiMo 的 base_url 属热重载字段，因此按 (base_url, api_key) 缓存，
        任一变化即重建（保持热重载真实生效）。
        """
        if (
            self._client is None
            or self._client_base_url != base_url
            or self._client_api_key != api_key
        ):
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(base_url=base_url, api_key=api_key or _EMPTY_KEY)
            self._client_base_url = base_url
            self._client_api_key = api_key
        return self._client

    def _read_ref_sync(self, ref_audio_path: str) -> bytes:
        """同步读取参考音频字节（供 executor 调用，不阻塞事件循环）。"""
        try:
            with open(ref_audio_path, "rb") as fh:
                return fh.read()
        except OSError as exc:
            raise TTSError(
                f"MiMo 克隆 TTS 参考音频读取失败: {ref_audio_path}（{exc}）。"
                "请检查 speech.tts.tts_mimo.ref_audio_path 是否存在"
            ) from exc

    async def synthesize(self, text: str, emotion: str | None = None) -> AudioClip:
        """把文本发到 MiMo 克隆 TTS，解码为 AudioClip（16k mono float32）。

        - 参考音频 base64 → data URI（零样本克隆）；
        - chat.completions：user 消息为空串，assistant 消息放待合成文本（官方约定）；
        - 响应 message.audio.data 为 base64 wav → decode_audio_bytes。
        """
        if not text or not text.strip():
            raise TTSError("TTS 文本为空")
        api_key = self._cfg_api_key()
        if not api_key or api_key == _EMPTY_KEY:
            raise TTSError(
                "MiMo 克隆 TTS 未配置 API key（请在 .env 设置 MIMO_TTS_API_KEY）"
            )
        base_url = self._cfg_base_url()
        model = self._cfg_model()
        ref_audio_path = self._cfg_ref_audio_path()
        timeout = self._cfg_timeout()

        if not ref_audio_path:
            raise TTSError(
                "MiMo 克隆 TTS 未配置参考音频路径（speech.tts.tts_mimo.ref_audio_path）"
            )
        # 读文件放 executor（async 函数内不阻塞事件循环；与 LocalClone 同风格）
        loop = asyncio.get_running_loop()
        try:
            audio_bytes = await loop.run_in_executor(None, self._read_ref_sync, ref_audio_path)
        except TTSError:
            raise
        except Exception as exc:
            raise TTSError(f"MiMo 克隆 TTS 参考音频读取失败: {ref_audio_path}（{exc}）") from exc
        if not audio_bytes:
            raise TTSError(f"MiMo 克隆 TTS 参考音频为空: {ref_audio_path}")

        voice_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        voice_uri = f"data:audio/wav;base64,{voice_b64}"
        client = self._ensure_client(base_url, api_key)

        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": ""},
                        {"role": "assistant", "content": text},
                    ],
                    audio={"format": "wav", "voice": voice_uri},
                ),
                timeout=timeout,
            )
        except TimeoutError:
            raise TTSError(f"MiMo 克隆 TTS 超时（>{timeout}s）") from None
        except Exception as exc:
            raise TTSError(f"MiMo 克隆 TTS 调用失败: {_classify_mimo_error(exc, timeout)}") from exc

        # 响应结构：choices[0].message.audio.data = base64 wav
        message = resp.choices[0].message
        audio_attr = getattr(message, "audio", None)
        data_b64 = getattr(audio_attr, "data", None) if audio_attr is not None else None
        if not data_b64:
            raise TTSError(
                "MiMo 克隆 TTS 响应缺少 audio.data（请确认 speech.tts.tts_mimo.model "
                f"为 {_DEFAULT_MODEL}）"
            )
        try:
            data = base64.b64decode(data_b64)
        except Exception as exc:
            raise TTSError(f"MiMo 克隆 TTS audio.data 不是合法 base64: {exc}") from exc
        if not data:
            raise TTSError("MiMo 克隆 TTS 返回空音频")

        # 非音频检测：返回 JSON/HTML 错误时直接给出预览（而非"解码失败"难定位）
        if not data.startswith(_AUDIO_MAGIC_PREFIXES):
            preview = data[:200].decode("utf-8", errors="replace")
            stripped = preview.lstrip()
            if stripped.startswith(("{", "<")):
                raise TTSError(
                    f"MiMo 克隆 TTS 返回非音频数据（疑似 JSON/HTML 错误）: {preview!r}"
                )

        clip = decode_audio_bytes(data)
        logger.info("MiMo 克隆 TTS 合成完成: %d 字符, %.2fs", len(text), clip.duration)
        return clip
