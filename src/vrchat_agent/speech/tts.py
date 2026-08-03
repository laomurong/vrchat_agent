"""C 语音输出：TTSEngine 抽象 + AudioClip 数据模型（C-01）。

AudioClip：16k mono float32 音频片段（共享知识约定第 5 条）。
TTSEngine：双实现 EdgeTTSEngine（本地）/ CloudTTSEngine（云端），config 切换。
decode_audio_bytes：用 miniaudio 把 mp3/wav 字节解码为 AudioClip。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger("vrchat_agent.speech.tts")


class TTSError(RuntimeError):
    """TTS 合成/解码错误。"""


class AudioClip(BaseModel):
    """统一音频片段：16k mono float32（默认）。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    samples: np.ndarray = Field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    sample_rate: int = 16000

    @model_validator(mode="after")
    def _normalize(self) -> "AudioClip":
        """强制 float32 / 单声道。"""
        samples = np.asarray(self.samples, dtype=np.float32)
        if samples.ndim > 1:
            if samples.shape[1] == 1:
                samples = samples[:, 0]
            else:
                samples = samples.mean(axis=1)
        self.samples = np.ascontiguousarray(samples, dtype=np.float32)
        return self

    @property
    def duration(self) -> float:
        """时长（秒）。"""
        if len(self.samples) == 0 or self.sample_rate <= 0:
            return 0.0
        return len(self.samples) / float(self.sample_rate)


class TTSEngine(ABC):
    """TTS 引擎抽象。"""

    @abstractmethod
    async def synthesize(self, text: str, emotion: str | None = None) -> AudioClip:
        """把文本合成为 AudioClip（16k mono float32）。"""
        raise NotImplementedError


def decode_audio_bytes(data: bytes, sample_rate: int = 16000) -> AudioClip:
    """解码音频字节（mp3/wav 等）为 AudioClip。

    使用 miniaudio（纯 C 扩展，无需 ffmpeg）；解码时直接重采样到 16k mono float32。

    解码失败时错误信息会附带数据前 16 字节的 hex 与数据前 200 字符的文本预览，
    便于用户判断"返回的是不是音频"（如看到 `<html>` 或 `{"error"` 即端点问题）。
    """
    if not data:
        raise TTSError("音频数据为空")
    try:
        import miniaudio
    except ImportError as exc:
        raise TTSError("miniaudio 未安装（pip install miniaudio）") from exc
    # BugFix P0：miniaudio 1.71 的 decode() 经 cffi 调用 ma_decode_memory，
    # 入参只接受 bytes（bytes 可直接转 cdata void*）；memoryview/bytearray
    # 会抛 TypeError: initializer for ctype 'void *' ...。此处统一转 bytes。
    raw = bytes(data)
    try:
        decoded = miniaudio.decode(
            raw,
            output_format=miniaudio.SampleFormat.FLOAT32,
            nchannels=1,
            sample_rate=sample_rate,
        )
        samples = np.frombuffer(decoded.samples, dtype=np.float32).copy()
        return AudioClip(samples=samples, sample_rate=sample_rate)
    except Exception as exc:
        # P0 增强：解码失败时给出数据前 N 字节 hex / 前 200 字符，
        # 帮助定位"端点返回了 HTML/JSON 而非音频"的根因。
        prefix_hex = raw[:16].hex(" ")
        preview = raw[:200].decode("utf-8", errors="replace")
        raise TTSError(
            f"音频解码失败: {exc}；数据前 16 字节 hex=[{prefix_hex}]，"
            f"前 200 字符={preview!r}"
        ) from exc


class FallbackTTSEngine(TTSEngine):
    """带兜底的双引擎 TTS：主引擎失败时自动回退备用引擎。

    装配层用它包装 CloudTTSEngine(primary) + EdgeTTSEngine(fallback)，
    保证"cloud 挂了也不阻断说话"；两引擎都失败才抛 TTSError（含各自原因）。
    """

    def __init__(self, primary: TTSEngine, fallback: TTSEngine) -> None:
        self._primary = primary
        self._fallback = fallback

    @property
    def primary(self) -> TTSEngine:
        """主引擎（供测试/仪表盘识别当前优先引擎）。"""
        return self._primary

    @property
    def fallback(self) -> TTSEngine:
        """备用引擎。"""
        return self._fallback

    async def synthesize(self, text: str, emotion: str | None = None) -> AudioClip:
        if not text or not text.strip():
            raise TTSError("TTS 文本为空")
        try:
            return await self._primary.synthesize(text, emotion)
        except TTSError as primary_exc:
            logger.warning(
                "主 TTS 引擎失败，自动回退备用引擎: %s",
                primary_exc,
                extra={"event": "tts_fallback"},
            )
            try:
                return await self._fallback.synthesize(text, emotion)
            except TTSError as fallback_exc:
                raise TTSError(
                    f"TTS 主备引擎均失败（主: {primary_exc}；备: {fallback_exc}）"
                ) from fallback_exc
