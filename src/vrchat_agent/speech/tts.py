"""C 语音输出：TTSEngine 抽象 + AudioClip 数据模型（C-01）。

AudioClip：16k mono float32 音频片段（共享知识约定第 5 条）。
TTSEngine：双实现 EdgeTTSEngine（本地）/ CloudTTSEngine（云端），config 切换。
decode_audio_bytes：用 miniaudio 把 mp3/wav 字节解码为 AudioClip。
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger("vrchat_agent.speech.tts")

# ── TTS 文本清洗 ──────────────────────────────────────────────
# LLM 回复中可能混入颜文字/emoji/特殊 Unicode 符号，直接送入 TTS 合成会产生怪声。
# 本函数在合成前过滤掉非语音字符，只保留 TTS 能正常朗读的内容。
#
# 策略：按 Unicode 码点范围白名单。相比 unicodedata.category（如 Nd 会误放泰文数字
# ๐-๙），精确码点范围能可靠区分"TTS 可朗读的字符"与"颜文字/emoji 的视觉装饰字符"。
#
# 保留范围：
#   - ASCII 可打印 (U+0020–U+007E)：拉丁字母、数字、英文标点、空格
#   - CJK 统一表意文字 (U+3400–U+9FFF, U+20000–U+323AF)
#   - CJK 兼容表意文字 (U+F900–U+FAFF)
#   - CJK 符号与标点 (U+3000–U+303F)：、。〃々〜 etc.
#   - 全角标点 (U+FF01–U+FF5E)：！～ etc.
#   - CJK 兼容形式 (U+FE30–U+FE4F)：纵向标点变体
#   - 常用例外：EM DASH — (U+2014)、EN DASH – (U+2013)、省略号 … (U+2026)

# 注意：此集合用于 O(1) 行尾检查，实际过滤逻辑基于码点范围而非集合。
_TTS_RETAIN_NEWLINE = frozenset({"\n", "\r"})


def _clean_tts_text(text: str) -> str:
    """清洗 TTS 输入文本：移除颜文字/emoji/特殊 Unicode 符号。

    保留：中文、拉丁字母、数字、常见中英文标点、空格/换行。
    移除：emoji（U+1F300+）、颜文字特殊字符（Thai 字母/数字、声调修饰符、
          零宽字符、组合变音符、装饰符号等）、其他非语音 Unicode block。

    若全部被过滤则返回空字符串，由调用方空检查兜底。
    """
    if not text:
        return ""
    chars: list[str] = []
    for ch in text:
        cp = ord(ch)
        # ── 白名单：按码点范围 ──
        # ASCII 可打印 (space U+0020 ~ U+007E)
        if 0x0020 <= cp <= 0x007E:
            chars.append(ch)
            continue
        # CJK 统一表意文字 (U+3400–U+9FFF 覆盖 Extension A + 基本区)
        if 0x3400 <= cp <= 0x9FFF:
            chars.append(ch)
            continue
        # CJK 统一表意文字 Extension B–H (U+20000–U+323AF)
        if 0x20000 <= cp <= 0x323AF:
            chars.append(ch)
            continue
        # CJK 兼容表意文字 (U+F900–U+FAFF)
        if 0xF900 <= cp <= 0xFAFF:
            chars.append(ch)
            continue
        # CJK 符号与标点 (U+3000–U+303F)：、。〃々〜 etc.
        if 0x3000 <= cp <= 0x303F:
            chars.append(ch)
            continue
        # 全角标点 (U+FF01 ！–U+FF5E ～)
        if 0xFF01 <= cp <= 0xFF5E:
            chars.append(ch)
            continue
        # CJK 兼容形式 (U+FE30–U+FE4F)：纵向标点变体
        if 0xFE30 <= cp <= 0xFE4F:
            chars.append(ch)
            continue
        # 常见中英文排版例外：— (EM DASH) / – (EN DASH) / … (省略号)
        if cp in (0x2014, 0x2013, 0x2026):
            chars.append(ch)
            continue
        # 保留换行符（Cc 类别例外；后续 \s+ 合并会归一化）
        if ch in _TTS_RETAIN_NEWLINE:
            chars.append(ch)
        # 其余字符（emoji、颜文字装饰符、Thai、零宽等）→ 丢弃
    cleaned = "".join(chars)
    # 后处理：移除被掏空的括号对（颜文字被过滤后残留的空括号/纯符号括号，
    # 如 "( )"、"(*^^*)"、"（）"），这些送入 TTS 会产生无意义音节。
    cleaned = re.sub(r"\([^\w]*\)", "", cleaned)
    cleaned = re.sub(r"（[^\w]*）", "", cleaned)
    # 去首尾空白并合并多余空格
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


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
        text = _clean_tts_text(text)
        if not text or not text.strip():
            raise TTSError("TTS 文本为空")
        try:
            return await self._primary.synthesize(text, emotion)
        except Exception as primary_exc:  # noqa: BLE001 - P2：非 TTSError（网络/解析/断言等）也触发回退
            logger.warning(
                "主 TTS 引擎失败，自动回退备用引擎: %s",
                primary_exc,
                extra={"event": "tts_fallback"},
            )
            try:
                return await self._fallback.synthesize(text, emotion)
            except Exception as fallback_exc:  # noqa: BLE001 - 两引擎均失败 → 统一包装 TTSError
                raise TTSError(
                    f"TTS 主备引擎均失败（主: {primary_exc}；备: {fallback_exc}）"
                ) from fallback_exc
