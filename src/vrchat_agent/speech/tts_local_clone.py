"""C 语音输出：LocalCloneTTSEngine（克隆语音 · 本地模型 TTS，需求 1 方案变更）。

用户改用克隆语音模型（GPT-SoVITS / CosyVoice / fish-speech 等）替代/补充云端
TTS。本引擎通过 HTTP 调用**用户自部署的本地推理服务**（如 GPT-SoVITS 的
api_v2 服务，默认 http://127.0.0.1:9880/tts；或 CosyVoice 的 HTTP 封装服务），
拿到 wav/mp3 字节后复用 `decode_audio_bytes` 解码为 AudioClip（16k mono float32）。

设计（接口先行 + 参考实现）：
- 依赖只用标准库 urllib（不引入 httpx/requests），符合项目轻依赖风格；
- 同步 HTTP 请求放到 executor 线程执行（不阻塞事件循环）；
- method="get"  → GPT-SoVITS 风格：参数走 query（text/text_language/refer_wav_path
  /prompt_text/prompt_language）；
- method="post" → CosyVoice 等 JSON 服务：同名字段走 JSON body；
- 失败统一抛 TTSError（含"服务不可达 / HTTP 状态码+响应预览 / 解码失败"），
  由装配层 FallbackTTSEngine 自动回退 Edge。

注意：本机需自行部署并启动本地推理服务（部署指引见 README「克隆语音 TTS」节）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.parse
import urllib.request

from .tts import AudioClip, TTSEngine, TTSError, decode_audio_bytes

logger = logging.getLogger("vrchat_agent.speech.tts.clone")

# 默认本地服务路径（GPT-SoVITS api_v2 默认端点）
_DEFAULT_BASE_URL = "http://127.0.0.1:9880"
_DEFAULT_ENDPOINT = "/tts"


class LocalCloneTTSEngine(TTSEngine):
    """克隆语音 TTS：HTTP 调本地推理服务（GPT-SoVITS / CosyVoice）→ AudioClip。

    热重载（与 Edge/Cloud 一致）：传入 config 引用后，synthesize() 每次从
    ``config.speech.tts.tts_clone.*`` 读取 base_url/method/ref_audio_path/ref_text/
    timeout，改配置无需重启（provider 切换属需重启字段）。
    """

    def __init__(
        self,
        base_url: str | None = None,
        endpoint: str = _DEFAULT_ENDPOINT,
        method: str = "get",
        ref_audio_path: str = "",
        ref_text: str = "",
        lang: str = "zh",
        timeout: float = 30.0,
        config=None,
    ) -> None:
        self._base_url = (base_url or _DEFAULT_BASE_URL).rstrip("/")
        self._endpoint = endpoint or _DEFAULT_ENDPOINT
        self._method = (method or "get").lower()
        self._ref_audio_path = ref_audio_path
        self._ref_text = ref_text
        self._lang = lang or "zh"
        self._timeout = float(timeout)
        self._config = config

    # ---- 热重载读取（异常回退装配期缓存值） ----
    def _cfg(self, key: str, default):
        """从 config.speech.tts.tts_clone 读取字段；异常/未配置回退缓存值。"""
        if self._config is not None:
            try:
                value = getattr(self._config.speech.tts.tts_clone, key)
                if value is not None and str(value).strip() != "":
                    return value
            except Exception:
                pass
        return default

    def _cfg_base_url(self) -> str:
        return str(self._cfg("base_url", self._base_url)).strip().rstrip("/") or _DEFAULT_BASE_URL

    def _cfg_method(self) -> str:
        return str(self._cfg("method", self._method)).strip().lower() or "get"

    def _cfg_ref_audio_path(self) -> str:
        return str(self._cfg("ref_audio_path", self._ref_audio_path)).strip()

    def _cfg_ref_text(self) -> str:
        return str(self._cfg("ref_text", self._ref_text)).strip()

    def _cfg_lang(self) -> str:
        """合成语言：zh/en/ja/ko 等（CosyVoice 服务端归一化后透传）。"""
        return str(self._cfg("lang", self._lang)).strip() or "zh"

    def _cfg_timeout(self) -> float:
        try:
            return float(self._cfg("timeout", self._timeout))
        except (TypeError, ValueError):
            return self._timeout

    # ---- 请求构造 / 发送（同步，供 executor 调用） ----
    def _build_params(self, text: str) -> dict[str, str]:
        """构造本地服务参数（GPT-SoVITS 命名；CosyVoice 等 JSON 服务兼容同名字段）。

        - text：待合成文本；
        - text_language / prompt_language：合成语言（默认 zh；可配 lang 改 en/ja/ko）；
        - refer_wav_path：克隆参考音频路径（服务端可访问）；
        - prompt_text：参考音频对应文本（few-shot 克隆）。

        同时携带 ref_audio_path / ref_text（CosyVoice 服务端命名），服务端做字段
        归一化（见 vrchat_agent.cosyvoice.model_loader.normalize_tts_params），
        因此本地服务无需区分 GPT-SoVITS 还是 CosyVoice。
        留空字段不携带（部分服务端对空参敏感）。
        """
        lang = self._cfg_lang()
        params: dict[str, str] = {"text": text, "text_language": lang, "prompt_language": lang}
        ref_audio = self._cfg_ref_audio_path()
        ref_text = self._cfg_ref_text()
        if ref_audio:
            params["refer_wav_path"] = ref_audio
            params["ref_audio_path"] = ref_audio
        if ref_text:
            params["prompt_text"] = ref_text
            params["ref_text"] = ref_text
        return params

    def _request_sync(self, text: str) -> bytes:
        """同步发送 HTTP 请求并返回音频字节（失败抛 TTSError）。"""
        base_url = self._cfg_base_url()
        endpoint = self._endpoint
        method = self._cfg_method()
        timeout = self._cfg_timeout()
        url = f"{base_url}{endpoint}"
        params = self._build_params(text)

        try:
            if method == "post":
                req = urllib.request.Request(
                    url,
                    data=json.dumps(params, ensure_ascii=False).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
            else:
                query = urllib.parse.urlencode(params)
                req = urllib.request.Request(f"{url}?{query}", method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            # 服务端返回错误（如模型未加载/参考音频不存在）：带状态码 + 响应预览，
            # 便于用户直接看到服务端报错（比"解码失败 <html>"可读得多）。
            body = b""
            try:
                body = exc.read(400)
            except Exception:
                pass
            preview = body[:200].decode("utf-8", errors="replace")
            raise TTSError(
                f"克隆语音 TTS 服务返回错误 HTTP {exc.code}（{url}）: {preview}"
            ) from exc
        except urllib.error.URLError as exc:
            raise TTSError(
                f"克隆语音 TTS 服务不可达（{url}）: {exc.reason}。"
                "请确认已启动本地推理服务（GPT-SoVITS 默认 http://127.0.0.1:9880），"
                "或用 Web 控制台修改 speech.tts.tts_clone.base_url"
            ) from exc
        except TimeoutError as exc:
            raise TTSError(f"克隆语音 TTS 超时（>{timeout}s，{url}）") from exc
        except Exception as exc:
            raise TTSError(f"克隆语音 TTS 请求失败: {exc}") from exc

    async def synthesize(self, text: str, emotion: str | None = None) -> AudioClip:
        """把文本发到本地克隆语音服务，解码为 AudioClip（16k mono float32）。"""
        if not text or not text.strip():
            raise TTSError("TTS 文本为空")
        loop = asyncio.get_running_loop()
        try:
            data = await loop.run_in_executor(None, self._request_sync, text)
        except TTSError:
            raise
        except Exception as exc:
            raise TTSError(f"克隆语音 TTS 失败: {exc}") from exc
        if not data:
            raise TTSError("克隆语音 TTS 返回空音频")
        # 复用 decode_audio_bytes：解码失败会带数据前 16 字节 hex / 前 200 字符预览，
        # 便于判断"服务端返回了 JSON/HTML 而非音频"。
        clip = decode_audio_bytes(data)
        logger.info("克隆语音 TTS 合成完成: %d 字符, %.2fs", len(text), clip.duration)
        return clip
