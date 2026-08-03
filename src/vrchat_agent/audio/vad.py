"""A 语音输入：VAD 语音活动检测 + 语音段切分（A-03）。

- VAD：基于 webrtcvad（轻量）；P1 可换 silero-vad。
- SpeechSegmenter：把 30ms 音频块切分成语音段（静音间隔触发切分），
  供 ASR 异步转写。
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("vrchat_agent.audio.vad")


class VAD:
    """语音活动检测：is_speech(chunk) -> bool。

    chunk 为 16k mono float32（内部转 int16 PCM 供 webrtcvad）。
    vad_impl 参数用于测试注入替身。

    热重载（A1）：传入 config 引用后，``sync_from_config()`` 会把
    ``config.asr.vad_mode`` 的最新值同步到 webrtcvad（其 set_mode() 支持运行中
    修改灵敏度，无需重建对象）。
    """

    def __init__(self, mode: int = 2, sample_rate: int = 16000, vad_impl=None, config=None) -> None:
        self._sample_rate = sample_rate
        self._mode = int(mode)
        self._config = config
        if vad_impl is not None:
            self._vad = vad_impl
        else:
            try:
                import webrtcvad

                self._vad = webrtcvad.Vad(mode)
            except ImportError as exc:
                raise RuntimeError(
                    "webrtcvad 未安装，无法进行 VAD（pip install webrtcvad-wheels）"
                ) from exc

    def set_mode(self, mode: int) -> None:
        """运行中更新 VAD 灵敏度（webrtcvad.Vad.set_mode；替身无此方法则跳过）。"""
        self._mode = int(mode)
        try:
            setter = getattr(self._vad, "set_mode", None)
            if setter is not None:
                setter(self._mode)
        except Exception:
            logger.debug("VAD set_mode 失败（忽略，保持当前模式）", exc_info=True)

    def sync_from_config(self) -> None:
        """从 config 读取最新 vad_mode 并同步（无 config 或值未变则跳过）。"""
        if self._config is None:
            return
        try:
            mode = int(self._config.asr.vad_mode)
        except Exception:
            return
        if mode != self._mode:
            self.set_mode(mode)

    def is_speech(self, chunk: np.ndarray) -> bool:
        """判定单个音频帧是否为语音（帧长需为 10/20/30ms @16k）。"""
        if chunk is None or len(chunk) == 0:
            return False
        if chunk.dtype != np.int16:
            pcm = (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16)
        else:
            pcm = chunk
        try:
            return bool(self._vad.is_speech(pcm.tobytes(), self._sample_rate))
        except Exception:
            return False

    def reset(self) -> None:
        """重置内部状态（webrtcvad 无状态，保留接口一致性）。"""
        pass


class SpeechSegmenter:
    """语音段切分器：把连续的音频块累积为语音段，静音达阈值后切分。

    热重载（A1）：传入 config 引用后，feed()/flush() 会先从 ``config`` 读取
    min_speech_ms / silence_ms / max_segment_s / audio.input.silence_threshold，
    并同步到内部帧数/阈值——改这些阈值无需重启（chunk_ms 属需重启字段，固定）。
    未传 config 时完全沿用装配期参数（保持旧测试兼容）。
    """

    def __init__(
        self,
        vad: VAD,
        min_speech_ms: int = 300,
        silence_ms: int = 800,
        max_segment_s: int = 15,
        sample_rate: int = 16000,
        chunk_ms: int = 30,
        silence_threshold: float = 0.005,
        config=None,
    ) -> None:
        self._vad = vad
        self._min_speech_ms = int(min_speech_ms)
        self._silence_ms = int(silence_ms)
        # max_segment_s 保留 float（如 0.3s），int() 会截断为 0 导致 max_frames=1
        self._max_segment_s = float(max_segment_s)
        self._chunk_ms = int(chunk_ms)
        self._sample_rate = int(sample_rate)
        # RMS 静音预筛阈值（归一化 0~1，参考项目 vrc_chat_translator/voice.py 0.005）：
        # 块 RMS 低于阈值直接当静音处理（不调 webrtcvad），避免底噪/环境音被误判为
        # 语音（幻听根因）。与"自抑制静音窗口"（set_muted）互不冲突：前者按能量过滤，
        # 后者按 TTS 发声窗口过滤。
        self._silence_threshold = float(silence_threshold)
        self._config = config
        self._recompute_frames()
        self.reset()

    def _recompute_frames(self) -> None:
        """按当前 ms 参数重算帧数（chunk_ms 固定不变）。"""
        chunk_ms = self._chunk_ms
        self._min_speech_frames = max(1, int(self._min_speech_ms / chunk_ms))
        self._silence_frames = max(1, int(self._silence_ms / chunk_ms))
        self._max_frames = max(1, int(self._max_segment_s * 1000 / chunk_ms))

    def _sync_from_config(self) -> None:
        """从 config 读取最新阈值/时长参数（热重载），异常回退缓存值。"""
        if self._config is None:
            return
        changed = False
        try:
            new_threshold = float(self._config.audio.input.silence_threshold)
            if new_threshold != self._silence_threshold:
                self._silence_threshold = new_threshold
                changed = True
        except Exception:
            pass
        try:
            new_min = int(self._config.asr.min_speech_ms)
            new_silence = int(self._config.asr.silence_ms)
            new_max = int(self._config.asr.max_segment_s)
        except Exception:
            new_min, new_silence, new_max = None, None, None
        if new_min is not None and new_min != self._min_speech_ms:
            self._min_speech_ms = new_min
            changed = True
        if new_silence is not None and new_silence != self._silence_ms:
            self._silence_ms = new_silence
            changed = True
        if new_max is not None and new_max != self._max_segment_s:
            self._max_segment_s = new_max
            changed = True
        if changed:
            self._recompute_frames()
        if self._vad is not None:
            sync = getattr(self._vad, "sync_from_config", None)
            if sync is not None:
                sync()

    def reset(self) -> None:
        self._speech_frames: list[np.ndarray] = []
        self._silence_count = 0
        self._speech_started = False

    @staticmethod
    def _rms(chunk: np.ndarray) -> float:
        """计算音频块 RMS（归一化 0~1）；空/异常块返回 0。

        平方和用 float64 计算：float32 大数值平方可能 overflow→inf，inf 会使
        `rms < silence_threshold` 恒 False → RMS 静音预筛被绕过（幻听）。
        """
        try:
            x = np.asarray(chunk, dtype=np.float32)
            if x.size == 0:
                return 0.0
            return float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))
        except Exception:
            return 0.0

    def feed(self, chunk: np.ndarray) -> np.ndarray | None:
        """输入一个音频块，返回切分好的语音段（1D float32）或 None。

        语音段由若干音频块拼接而成；静音达到 silence_ms 或段长达到上限时切分。
        RMS 预筛：低于 silence_threshold 的块视为静音（跳过 webrtcvad），但仍
        参与静音计数（保证"停顿切句"不失效）。
        """
        self._sync_from_config()  # 热重载：阈值/时长参数运行中可改
        rms = self._rms(chunk)
        if rms < self._silence_threshold:
            # 能量预筛命中：底噪/环境音 → 不触发 VAD（幻听修复），按静音帧处理。
            logger.debug(
                "RMS 预筛静音: rms=%.5f < %.5f（丢弃 VAD，块长=%d）",
                rms, self._silence_threshold, len(chunk),
            )
            is_speech = False
        else:
            is_speech = self._vad.is_speech(chunk)
        if is_speech:
            self._speech_started = True
            self._silence_count = 0
            self._speech_frames.append(np.asarray(chunk, dtype=np.float32).copy())
            if len(self._speech_frames) >= self._max_frames:
                return self._emit_segment()
            return None

        if self._speech_started:
            self._silence_count += 1
            if self._silence_count >= self._silence_frames:
                return self._emit_segment()
        return None

    def flush(self) -> np.ndarray | None:
        """强制切分剩余语音段（用于停止时收尾）。"""
        self._sync_from_config()  # 热重载：与 feed() 保持同一参数源
        return self._emit_segment()

    def _emit_segment(self) -> np.ndarray | None:
        if len(self._speech_frames) < self._min_speech_frames:
            self.reset()
            return None
        segment = np.concatenate(self._speech_frames)
        self.reset()
        return segment

    @property
    def has_pending_speech(self) -> bool:
        return self._speech_started
