"""C 语音输出：EdgeTTSEngine（本地免费 TTS，C-01 双实现之一）。

依赖 edge-tts（微软 Edge 在线语音合成，免费）。
emotion 映射到语速（rate）微调，实现轻量语气控制（P1：C-03）。
"""

from __future__ import annotations

import logging

from .tts import TTSEngine, TTSError, AudioClip, decode_audio_bytes

logger = logging.getLogger("vrchat_agent.speech.tts.edge")

# 情感 → (rate 增量, volume 增量) 的轻量语气控制
_EMOTION_RATE = {
    "happy": "+12%",
    "excited": "+18%",
    "sad": "-12%",
    "angry": "+8%",
    "surprised": "+10%",
    "calm": "+0%",
    "neutral": "+0%",
}


class EdgeTTSEngine(TTSEngine):
    """Edge-TTS 本地实现（免费）。

    热重载（A1）：传入 config 引用后，synthesize() 每次从 ``config.speech.tts.*``
    读取 edge_voice / voice / rate / volume，改音色/语速/音量无需重启；未传 config
    时沿用装配期缓存值。edge_voice 优先于 voice（BugFix P0：voice 属云端音色名）。
    """

    def __init__(
        self,
        voice: str = "zh-CN-XiaoxiaoNeural",
        rate: str = "+0%",
        volume: str = "+0%",
        timeout: float = 10.0,
        config=None,
    ) -> None:
        self._voice = voice
        self._rate = rate
        self._volume = volume
        self._timeout = timeout
        self._config = config

    # ---- 热重载读取（异常回退装配期缓存值） ----
    def _cfg_voice(self) -> str:
        """读取 Edge 音色：优先 config.speech.tts.edge_voice（微软音色名）。

        BugFix P0（TTS 主备音色不兼容）：voice 字段被云端（OpenAI 兼容）音色名
        占用（默认 "alloy"），Edge-TTS 收到会抛 Invalid voice。故 Edge 专属字段
        edge_voice 优先；未配置 edge_voice（旧配置）时回退 voice 兼容旧行为。
        """
        if self._config is not None:
            try:
                edge_voice = str(self._config.speech.tts.edge_voice or "").strip()
                if edge_voice:
                    return edge_voice
            except Exception:
                pass
            try:
                return str(self._config.speech.tts.voice or self._voice)
            except Exception:
                pass
        return self._voice

    def _cfg_rate(self) -> str:
        if self._config is not None:
            try:
                rate = int(self._config.speech.tts.rate)
                return f"{rate:+d}%"
            except Exception:
                pass
        return self._rate

    def _cfg_volume(self) -> str:
        if self._config is not None:
            try:
                volume = int(self._config.speech.tts.volume)
                return f"{volume:+d}%"
            except Exception:
                pass
        return self._volume

    async def synthesize(self, text: str, emotion: str | None = None) -> AudioClip:
        if not text or not text.strip():
            raise TTSError("TTS 文本为空")
        try:
            import edge_tts
        except ImportError as exc:
            raise TTSError("edge-tts 未安装（pip install edge-tts）") from exc

        voice = self._cfg_voice()
        rate = _EMOTION_RATE.get(emotion or "", self._cfg_rate())
        volume = self._cfg_volume()
        try:
            communicate = edge_tts.Communicate(text, voice, rate=rate, volume=volume)
            buf = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.extend(chunk["data"])
            if not buf:
                raise TTSError("Edge-TTS 返回空音频")
            clip = decode_audio_bytes(bytes(buf))
            logger.info("Edge-TTS 合成完成: %d 字符, %.2fs", len(text), clip.duration)
            return clip
        except TTSError:
            raise
        except Exception as exc:
            raise TTSError(f"Edge-TTS 合成失败: {exc}") from exc
