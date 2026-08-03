"""实机验证 v2：精确匹配优先 + 实机截 VRChat 窗口。"""

from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, r"E:/vrc具身agent/vrchat_agent/src")

from vrchat_agent.vision.window_sensor import _find_window, WindowCaptureVisionSensor


async def main() -> None:
    found = _find_window("VRChat")
    print("_find_window('VRChat') =", found)
    if found is None:
        print("未找到 VRChat 窗口")
        return
    hwnd, title = found

    sensor = WindowCaptureVisionSensor(window_title_keyword="VRChat")
    await sensor.start()  # 走完整 start() 路径（枚举 + mss 实例化）
    frame = await sensor.capture()
    if frame is None:
        print("FAIL: capture 返回 None")
        await sensor.stop()
        return
    print(f"PASS: 截图成功 shape={frame.shape} dtype={frame.dtype}")
    import cv2

    cv2.imwrite(r"E:/vrc具身agent/vrchat_agent/debug_window_capture.png", frame)
    print("已保存: debug_window_capture.png")
    await sensor.stop()


if __name__ == "__main__":
    asyncio.run(main())
