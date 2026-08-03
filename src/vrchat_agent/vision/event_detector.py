"""E 视觉感知：视觉事件检测器（P1：E-03）。

通过前后帧灰度差分均值判断画面是否发生显著变化（有人靠近、举物等），
用于低频采样循环中按需发布 vision_snapshot，节省 API 成本。
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("vrchat_agent.vision.detect")


class VisualEventDetector:
    """画面显著变化检测。

    热重载（A1）：传入 config 引用后，is_significant() 每次从
    ``config.vision.sampling.change_threshold`` 读取阈值（未显式传 threshold
    参数时），改阈值无需重启；未传 config 时沿用装配期缓存值。
    """

    def __init__(self, threshold: float = 0.12, gray_size: tuple[int, int] = (64, 64), config=None) -> None:
        self._threshold = threshold
        self._gray_size = gray_size
        self._config = config

    def _cfg_threshold(self) -> float | None:
        if self._config is not None:
            try:
                return float(self._config.vision.sampling.change_threshold)
            except Exception:
                pass
        return None

    def is_significant(
        self, new_frame: np.ndarray, old_frame: np.ndarray | None, threshold: float | None = None
    ) -> bool:
        """判定新帧相对旧帧是否发生显著变化。

        首帧（old_frame 为 None）视为显著（需要发布）。
        """
        if old_frame is None:
            return True
        if threshold is None:
            threshold = self._cfg_threshold() or self._threshold
        a = self._downscale(new_frame)
        b = self._downscale(old_frame)
        if a is None or b is None or a.shape != b.shape:
            return True
        import cv2

        diff = cv2.absdiff(a, b)
        score = float(diff.mean()) / 255.0
        return score >= threshold

    def _downscale(self, frame: np.ndarray) -> np.ndarray | None:
        """转灰度并缩小到统一尺寸。"""
        if frame is None or frame.size == 0:
            return None
        try:
            import cv2

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.resize(gray, self._gray_size, interpolation=cv2.INTER_AREA)
        except Exception:
            return None
