"""E 视觉感知：指定窗口截图传感器（window_capture 后端，Windows 限定，E-01 增量）。

替代 OBS：直接截取标题含关键词的窗口（如 VRChat）画面作为视觉源。
- 窗口枚举：ctypes 调用 user32（EnumWindows / GetWindowTextW / GetWindowRect /
  IsWindowVisible），零额外依赖（pywin32 非必需）；
- 取帧：mss 按窗口矩形截取（纯 Python，pip install mss），BGRA → BGR；
- 接口：与 OBS/OpenCV 实现一致（VisionSensor：start/capture/stop），
  sampler / event_detector 复用不变。

错误分级（用户可操作）：
- 未安装 mss        → VisionSensorError("mss 未安装（pip install mss）")
- 非 Windows 平台   → VisionSensorError("window_capture 仅支持 Windows 平台")
- 找不到目标窗口     → VisionSensorError("未找到标题包含「xx」的窗口…")
- 单帧截图失败       → capture() 返回 None（视觉降级为无画面，不抛异常）
"""

from __future__ import annotations

import logging
import sys

import numpy as np

from .vision_sensor import VisionSensor, VisionSensorError

logger = logging.getLogger("vrchat_agent.vision.window")

_IS_WINDOWS = sys.platform.startswith("win")


# ---------------------------------------------------------------------------
# 窗口枚举 / 矩形查询（模块级纯函数，便于单测 monkeypatch；非 Windows 返回空）
# ---------------------------------------------------------------------------
def _enumerate_windows() -> list[tuple[int, str]]:
    """枚举所有顶层可见且有标题的窗口：[(hwnd, title)]。

    Windows 下用 ctypes 调 user32（EnumWindows 回调收集），不依赖 pywin32；
    非 Windows 平台返回空列表（window_capture 后端仅 Windows 可用）。
    """
    if not _IS_WINDOWS:
        return []
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    results: list[tuple[int, str]] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _cb(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True  # 继续枚举
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        if buf.value:
            results.append((int(hwnd), buf.value))
        return True

    user32.EnumWindows(_cb, 0)
    return results


def _get_window_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    """返回窗口屏幕矩形 (left, top, right, bottom)；失败返回 None。"""
    if not _IS_WINDOWS:
        return None
    import ctypes
    from ctypes import wintypes

    rect = wintypes.RECT()
    if not ctypes.windll.user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(rect)):
        return None
    return (rect.left, rect.top, rect.right, rect.bottom)


def _find_window(keyword: str) -> tuple[int, str] | None:
    """按标题关键词查找窗口，返回 (hwnd, title)；找不到返回 None。

    匹配策略（避免误匹配 "vrchat_agent" 等子串窗口，如 VSCode 项目标题）：
    1) 标题**精确相等**优先（VRChat 游戏窗口标题通常就是 "VRChat"）；
    2) 无精确匹配时退回大小写不敏感的子串匹配（如 "VRChat - Home"）。
    """
    kw = (keyword or "").strip().lower()
    if not kw:
        return None
    substring: tuple[int, str] | None = None
    for hwnd, title in _enumerate_windows():
        t = title.strip().lower()
        if t == kw:
            return hwnd, title  # 精确匹配优先
        if substring is None and kw in t:
            substring = (hwnd, title)
    return substring


class WindowCaptureVisionSensor(VisionSensor):
    """截取指定标题关键词窗口画面（首选替代 OBS 的轻量方案）。"""

    def __init__(self, window_title_keyword: str = "VRChat") -> None:
        self._keyword = window_title_keyword
        self._hwnd: int | None = None
        self._title = ""
        self._sct = None

    async def start(self) -> None:
        try:
            import mss
        except ImportError as exc:
            raise VisionSensorError("mss 未安装（pip install mss）") from exc
        if not _IS_WINDOWS:
            raise VisionSensorError("window_capture 仅支持 Windows 平台（当前无法调用 user32）")
        found = _find_window(self._keyword)
        if found is None:
            raise VisionSensorError(
                f"未找到标题包含「{self._keyword}」的窗口"
                "（请确认 VRChat 已启动且窗口可见，或修改 config.yaml 的 "
                "vision.sensor.window_title_keyword）"
            )
        self._hwnd, self._title = found
        # mss 9.x 用 mss.mss()；10.x 弃用该工厂改 mss.MSS 类——两者兼容取其一。
        mss_factory = getattr(mss, "MSS", None) or getattr(mss, "mss", None)
        self._sct = mss_factory()
        logger.info(
            "窗口截图已就绪: hwnd=%s title=%s keyword=%s", self._hwnd, self._title, self._keyword
        )

    async def capture(self) -> np.ndarray | None:
        """截取窗口当前画面 → BGR 图像；失败返回 None（视觉降级为无画面，不抛异常）。"""
        if self._hwnd is None or self._sct is None:
            return None
        try:
            rect = _get_window_rect(self._hwnd)
            if rect is None:
                return None
            region = self._clamp_region(rect)
            if region is None:
                return None
            shot = self._sct.grab(region)
            img = np.asarray(shot, dtype=np.uint8)  # BGRA (HxWx4)
            import cv2

            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except Exception as exc:
            logger.debug("窗口截图失败: %s", exc)
            return None

    def _clamp_region(self, rect: tuple[int, int, int, int]) -> dict | None:
        """把窗口矩形裁剪到 mss 虚拟屏幕范围（多显示器负坐标安全），返回 mss region。"""
        if self._sct is None:
            return None
        virt = self._sct.monitors[0]  # 所有显示器并集（虚拟屏幕）
        v_left, v_top = int(virt["left"]), int(virt["top"])
        v_right = v_left + int(virt["width"])
        v_bottom = v_top + int(virt["height"])
        left = max(rect[0], v_left)
        top = max(rect[1], v_top)
        right = min(rect[2], v_right)
        bottom = min(rect[3], v_bottom)
        if right <= left or bottom <= top:
            return None
        return {"left": left, "top": top, "width": right - left, "height": bottom - top}

    async def stop(self) -> None:
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None
