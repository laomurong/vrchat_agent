"""F 编排主循环：打断控制（F-03，P1）。

玩家在 AI 说话（SPEAKING）期间再次说话 → request_interrupt()，
主循环检测到后停止当前 TTS 播放并重新决策（SPEAKING → THINKING）。
"""

from __future__ import annotations

import threading
import time


class InterruptController:
    """打断请求控制器（线程安全）。"""

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._requested = False
        self._at = 0.0
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def request_interrupt(self) -> None:
        """请求打断当前说话。"""
        with self._lock:
            self._requested = True
            self._at = time.time()

    def should_interrupt(self) -> bool:
        """是否有待处理的打断请求。"""
        with self._lock:
            return self._requested

    def clear(self) -> None:
        """清除打断请求。"""
        with self._lock:
            self._requested = False
