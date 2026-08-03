"""E 视觉感知：VisionSensor 抽象接口（E-01）。

所有视觉传感器实现 capture() → BGR 图像（HxWx3 uint8）或 None（失败）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class VisionSensorError(RuntimeError):
    """视觉传感器错误。"""


class VisionSensor(ABC):
    """视觉传感器抽象：一次捕获一帧画面。"""

    @abstractmethod
    async def capture(self) -> np.ndarray | None:
        """返回 BGR 图像（HxWx3 uint8）或 None（捕获失败）。"""
        raise NotImplementedError

    async def start(self) -> None:
        """打开设备（默认无操作）。"""

    async def stop(self) -> None:
        """关闭设备（默认无操作）。"""
