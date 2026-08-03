"""E 视觉感知：图像采样器（E-02）。

低频采样（0.5~1fps）→ 缩放 → JPEG 压缩 → base64，供多模态 LLM。
支持后台采样循环：按 fps 发布 vision_snapshot 事件（可配显著变化触发）。
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time

import numpy as np

from ..events import Event, EventType

logger = logging.getLogger("vrchat_agent.vision.sampler")


class ImageSampler:
    """低频采样 + JPEG/base64 压缩。

    热重载（A1）：传入 config 引用后，run()/encode() 每次从
    ``config.vision.sampling.*`` 读取 fps / max_size / jpeg_quality /
    event_trigger——改采样参数无需重启；未传 config 时沿用装配期缓存值。
    """

    def __init__(
        self,
        fps: float = 0.5,
        max_size: int = 768,
        jpeg_quality: int = 80,
        event_trigger: bool = False,
        detector=None,
        config=None,
    ) -> None:
        self._interval = (1.0 / fps) if fps and fps > 0 else 2.0
        self._max_size = max_size
        self._quality = jpeg_quality
        self._event_trigger = event_trigger
        self._detector = detector
        self._config = config

    # ---- 热重载读取（异常回退装配期缓存值） ----
    def _cfg_interval(self) -> float:
        if self._config is not None:
            try:
                fps = float(self._config.vision.sampling.fps)
                return (1.0 / fps) if fps and fps > 0 else self._interval
            except Exception:
                pass
        return self._interval

    def _cfg_max_size(self) -> int:
        if self._config is not None:
            try:
                return int(self._config.vision.sampling.max_size)
            except Exception:
                pass
        return self._max_size

    def _cfg_quality(self) -> int:
        if self._config is not None:
            try:
                return int(self._config.vision.sampling.jpeg_quality)
            except Exception:
                pass
        return self._quality

    def _cfg_event_trigger(self) -> bool:
        if self._config is not None:
            try:
                return bool(self._config.vision.sampling.event_trigger)
            except Exception:
                pass
        return self._event_trigger

    async def sample(self, sensor) -> str | None:
        """捕获一帧 → 压缩 → 返回 JPEG base64 字符串（或 None）。"""
        frame = await sensor.capture()
        if frame is None:
            return None
        return self.encode(frame)

    def encode(self, frame: np.ndarray) -> str | None:
        """BGR 图像 → 缩放 → JPEG base64（max_size/jpeg_quality 热重载）。"""
        try:
            import cv2
        except ImportError as exc:
            logger.error("opencv-python 未安装: %s", exc)
            return None
        max_size = self._cfg_max_size()
        quality = self._cfg_quality()
        img = self._resize(frame, max_size)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")

    @staticmethod
    def _resize(frame: np.ndarray, max_size: int) -> np.ndarray:
        """按最长边缩放，控制送入 LLM 的 token/成本。"""
        h, w = frame.shape[:2]
        scale = max_size / max(h, w)
        if scale < 1.0:
            import cv2

            frame = cv2.resize(
                frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )
        return frame

    async def run(self, sensor, bus, detector=None) -> None:
        """后台采样循环：按 fps 发布 vision_snapshot 事件（阻塞运行）。

        event_trigger 热重载：True → 用事件检测器按需发布；False → 跳过捕获
        与发布（等效"后台循环未启动"），改配置即时生效。
        """
        last_frame = None
        while True:
            await asyncio.sleep(self._cfg_interval())
            if not self._cfg_event_trigger():
                continue  # 事件触发关闭：不采样、不发布
            frame = await sensor.capture()
            if frame is None:
                continue
            det = detector or self._detector
            if det is not None and not det.is_significant(frame, last_frame):
                continue  # 画面无显著变化：不发布（P1：E-03）
            last_frame = frame
            image_b64 = self.encode(frame)
            if image_b64:
                bus.publish(
                    Event(EventType.VISION_SNAPSHOT.value, {"image_b64": image_b64, "ts": time.time()})
                )
