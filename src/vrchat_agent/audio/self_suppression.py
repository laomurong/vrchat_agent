"""A 语音输入：自抑制静音窗口（A-04）。

问题：AI 自身 TTS 发声会经 WASAPI 回环被再次捕获，形成"自我对话"死循环。
方案（Q5 默认）：静音窗口 —— 监听 tts_started / tts_finished 事件，
在窗口期间置 AudioInput 静音，丢弃音频块；窗口结束后解除。
P1 可选升级：设备分离 / 能量比对。
"""

from __future__ import annotations

import logging
import time

from ..events import Event, EventType

logger = logging.getLogger("vrchat_agent.audio.suppress")


class SelfSuppression:
    """TTS 发声期间静音窗口控制器。

    用法（装配层）：
        suppression = SelfSuppression(bus=bus, audio_input=audio_input)
        suppression.attach()   # 订阅 tts_started / tts_finished
    """

    def __init__(
        self,
        bus=None,
        audio_input=None,
        strategy: str = "mute_window",
        grace_ms: float = 300.0,
    ) -> None:
        self._bus = bus
        self._audio_input = audio_input
        self._strategy = strategy
        self._grace_s = grace_ms / 1000.0
        self._suppressing = False
        self._started_at = 0.0

    # ---- 事件回调（订阅 EventBus）----
    async def on_tts_started(self, event: Event | None = None) -> None:
        """tts_started → 进入静音窗口。"""
        self._suppressing = True
        self._started_at = time.time()
        if self._audio_input is not None:
            self._audio_input.set_muted(True)
        logger.info("自抑制：进入静音窗口")

    async def on_tts_finished(self, event: Event | None = None) -> None:
        """tts_finished → 退出静音窗口。"""
        self._suppressing = False
        if self._audio_input is not None:
            self._audio_input.set_muted(False)
        logger.info("自抑制：解除静音窗口")

    async def on_emergency_stop(self, event: Event | None = None) -> None:
        """急停 → 强制解除静音窗口（A-02 兜底）。

        打断/急停取消播放时若漏发 TTS_FINISHED，静音窗口永不解除导致麦克风
        永久静音（Agent"聋"）；订阅急停事件强制恢复，保证急停后一定能听到。
        """
        self._suppressing = False
        if self._audio_input is not None:
            self._audio_input.set_muted(False)
        logger.info("自抑制：急停强制解除静音窗口")

    # ---- 查询 ----
    def should_suppress(self) -> bool:
        """当前是否处于静音窗口（应丢弃 ASR 输入）。"""
        if not self._suppressing:
            return False
        # 带宽限：窗口最短维持 grace_ms，防止快速抖动
        if time.time() - self._started_at < self._grace_s:
            return True
        return self._suppressing

    # ---- 装配 ----
    def attach(self) -> None:
        """订阅 TTS 事件（装配层调用）。"""
        if self._bus is None:
            return
        self._bus.subscribe(EventType.TTS_STARTED.value, self.on_tts_started)
        self._bus.subscribe(EventType.TTS_FINISHED.value, self.on_tts_finished)
        # A-02：急停事件强制解除静音（兜底，防止取消路径漏发 TTS_FINISHED 致聋）
        self._bus.subscribe(EventType.EMERGENCY_STOP.value, self.on_emergency_stop)
