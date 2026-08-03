"""E 视觉感知：OBS 视觉传感器实现。

首选：obs-websocket 取帧（GetSourceScreenshot）[Win: OBS Studio]
回退：OpenCV 直接读 OBS 虚拟摄像头设备 [Win]
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

import numpy as np

from .vision_sensor import VisionSensor, VisionSensorError

logger = logging.getLogger("vrchat_agent.vision.obs")


class OBSWebsocketVisionSensor(VisionSensor):
    """通过 OBS WebSocket 截图取帧（首选，E-01）。

    依赖 obs-websocket-py（BugFix P2：1.x 包名为 obswebsocket，`from obswebsocket
    import obsws, requests`；旧版 0.x 才直接 `import obsws`）。obs-websocket-py 1.x
    的 connect/call/disconnect 均为同步阻塞调用，这里统一放进 executor 执行，
    避免阻塞事件循环。OBS 需开启 WebSocket 服务（工具 → WebSocket 服务器设置）。
    source_name 留空时自动探测窗口/游戏/显示器捕获源。
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 4455,
        password: str = "",
        source_name: str = "",
        width: int = 1280,
        height: int = 720,
    ) -> None:
        self._host = host
        self._port = port
        self._password = password
        self._source_name = source_name
        self._width = width
        self._height = height
        self._ws = None

    async def start(self) -> None:
        # BugFix P2：obs-websocket-py 1.x 提供的是 obswebsocket 包（顶层导出 obsws=类），
        # 旧版 0.x 才是顶层 obsws 模块——两种都试，避免"已安装却误报未安装"。
        try:
            from obswebsocket import obsws, requests
        except ImportError:
            try:
                import obsws  # type: ignore[no-redef]

                requests = None  # 旧版 0.x：requests 对象结构不同，截图降级为不可用
            except ImportError as exc:
                raise self._import_error(exc) from exc
        # BugFix P2（QA 回归）：1.x 的 __init__.py 是 `from .core import obsws`（obsws=类），
        # 不存在 `obsws.obsws`；0.x 的 obsws 才是模块（obsws.obsws=类）。统一取到类再构造，
        # 避免 `obsws.obsws(...)` 抛 AttributeError。
        ws_cls = obsws if isinstance(obsws, type) else obsws.obsws
        self._requests = requests
        self._ws = ws_cls(self._host, self._port, self._password)
        loop = asyncio.get_running_loop()
        try:
            # 1.x 的 connect() 为同步阻塞调用：放 executor，避免卡死事件循环
            await loop.run_in_executor(None, self._ws.connect)
            logger.info("OBS WebSocket 已连接: %s:%s", self._host, self._port)
        except Exception as exc:
            self._ws = None
            raise VisionSensorError(
                f"OBS WebSocket 连接失败（请确认 OBS Studio 已启动并开启 WebSocket 服务）: {exc}"
            ) from exc

    @staticmethod
    def _import_error(exc: ImportError) -> VisionSensorError:
        """区分「未安装」与「已安装但导入失败」两类错误（BugFix P2）。"""
        import importlib.util

        if (
            importlib.util.find_spec("obswebsocket") is None
            and importlib.util.find_spec("obsws") is None
        ):
            return VisionSensorError("obs-websocket-py 未安装（pip install obs-websocket-py）")
        return VisionSensorError(
            f"obs-websocket-py 已安装但导入失败（请检查版本/依赖是否完整）: {exc}"
        )

    async def _run_call(self, req) -> Any:
        """在 executor 中同步执行 OBS 请求（1.x 的 call() 为阻塞式）。

        失败返回 None（与旧行为一致：截图失败降级为无画面，不抛异常）。
        """
        if self._ws is None:
            return None
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, self._ws.call, req)
        except Exception as exc:
            logger.debug("OBS 请求失败: %s", exc)
            return None

    async def capture(self) -> np.ndarray | None:
        """从 OBS 截图一帧并解码为 BGR 图像。"""
        if self._ws is None or self._requests is None:
            return None
        name = self._source_name or await self._detect_source()
        if not name:
            return None
        req = self._requests.GetSourceScreenshot(
            sourceName=name,
            imageFormat="png",
            imageWidth=self._width,
            imageHeight=self._height,
        )
        try:
            resp = await self._run_call(req)
            if resp is None:
                return None
            data = (resp.datain or {}).get("imageData", "")
            if data.startswith("data:"):
                data = data.split(",", 1)[1]
            raw = base64.b64decode(data)
            arr = np.frombuffer(raw, dtype=np.uint8)
            import cv2

            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
        except Exception as exc:
            logger.debug("OBS 截图失败: %s", exc)
            return None

    async def _detect_source(self) -> str | None:
        """自动探测截图源：优先 VRChat 相关的窗口/游戏/显示器捕获源。"""
        if self._ws is None or self._requests is None:
            return None
        resp = await self._run_call(self._requests.GetSceneList())
        if resp is None:
            return None
        try:
            for scene in (resp.datain or {}).get("scenes", []):
                for src in scene.get("sources", []):
                    name = src.get("sourceName", "")
                    stype = src.get("type", "")
                    if any(
                        kw in stype
                        for kw in ("window_capture", "monitor_capture", "game_capture", "display_capture")
                    ) or "vrchat" in name.lower():
                        return name
        except Exception:
            pass
        return None

    async def stop(self) -> None:
        if self._ws is not None:
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._ws.disconnect)
            except Exception:
                pass
            self._ws = None


class OpenCVVisionSensor(VisionSensor):
    """OpenCV 直接读 OBS 虚拟摄像头设备（回退方案，[Win]）。

    需要 OBS Studio 已启动虚拟摄像头（工具 → 虚拟摄像头）。
    """

    def __init__(self, camera_device: int = 0) -> None:
        self._camera_device = camera_device
        self._cap = None

    async def start(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise VisionSensorError("opencv-python 未安装（pip install opencv-python）") from exc
        self._cap = cv2.VideoCapture(self._camera_device)
        if not self._cap.isOpened():
            self._cap = None
            raise VisionSensorError(
                f"OpenCV 打开摄像头失败（设备号 {self._camera_device}，OBS 虚拟摄像头未启动？）"
            )
        logger.info("OpenCV 摄像头已打开: device=%s", self._camera_device)

    async def capture(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    async def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
