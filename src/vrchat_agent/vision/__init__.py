"""E 视觉感知模块：OBS WebSocket / OpenCV / 指定窗口截图 → 低频采样 + JPEG/base64。"""

from __future__ import annotations

from .obs_sensor import OBSWebsocketVisionSensor, OpenCVVisionSensor
from .vision_sensor import VisionSensor, VisionSensorError
from .window_sensor import WindowCaptureVisionSensor

__all__ = [
    "OBSWebsocketVisionSensor",
    "OpenCVVisionSensor",
    "VisionSensor",
    "VisionSensorError",
    "WindowCaptureVisionSensor",
    "create_vision_sensor",
]


def create_vision_sensor(cfg) -> VisionSensor:
    """按 config.vision.sensor.backend 选择传感器实例（工厂；不启动，由调用方 start()）。

    - opencv          → OpenCVVisionSensor（OBS 虚拟摄像头回退）
    - window_capture  → WindowCaptureVisionSensor（指定窗口截图，替代 OBS）
    - 其他/默认        → OBSWebsocketVisionSensor
    """
    if cfg.backend == "opencv":
        return OpenCVVisionSensor(cfg.camera_device)
    if cfg.backend == "window_capture":
        return WindowCaptureVisionSensor(cfg.window_title_keyword)
    return OBSWebsocketVisionSensor(
        host=cfg.obs_host,
        port=cfg.obs_port,
        password=cfg.obs_password,
        source_name=cfg.source_name,
    )
