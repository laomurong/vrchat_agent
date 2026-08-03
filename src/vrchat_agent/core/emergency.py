"""F 编排主循环：EmergencyStop（全局急停，G-03）。

共享知识约定（架构文档第 8 节第 9 条）：
- EmergencyStop 是唯一可旁路状态机直接调 AudioOutput.stop() + Actuator.reset() 的入口；
- 急停后状态复位 IDLE，等待下次输入；<1s 生效。

双通道触发：
1. 全局快捷键（Ctrl+Alt+X，[Win]，keyboard 库）
2. 信号文件（scripts/emergency_stop.ps1 写入 logs/emergency_stop.flag）

事件：触发后向 EventBus 发布 emergency_stop 事件（线程安全）。
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from ..events import Event, EventType

logger = logging.getLogger("vrchat_agent.loop.emergency")


class EmergencyStop:
    """全局急停控制器。"""

    def __init__(
        self,
        hotkey: str = "ctrl+alt+x",
        bus=None,
        flag_file: str | Path | None = None,
        poll_interval: float = 0.2,
    ) -> None:
        self._hotkey = hotkey
        self._bus = bus
        self._flag_file = Path(flag_file) if flag_file else None
        self._poll_interval = poll_interval
        self._hotkey_registered = False
        self._hotkey_handle = None
        self._watch_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    async def arm(self) -> None:
        """注册快捷键 + 启动信号文件轮询（阻塞直到 disarm）。

        每次 arm 重置 _stop_event，支持二次 arm（OBS-5）。
        """
        self._stop_event = asyncio.Event()
        self._register_hotkey()
        self._watch_task = asyncio.create_task(self._watch_flag())
        logger.info("急停已布防: 快捷键=%s flag=%s", self._hotkey, self._flag_file)
        try:
            await self._stop_event.wait()
        finally:
            await self.disarm()

    def _register_hotkey(self) -> None:
        """注册全局快捷键（[Win]，keyboard 库；非 Win/未安装则跳过）。"""
        try:
            import keyboard
        except ImportError:
            logger.warning("keyboard 未安装，仅支持信号文件急停")
            return
        try:
            self._hotkey_handle = keyboard.add_hotkey(self._hotkey, self._trigger)
            self._hotkey_registered = True
            logger.info("急停快捷键已注册: %s", self._hotkey)
        except Exception as exc:
            logger.warning("注册急停快捷键失败: %s", exc)

    def _trigger(self) -> None:
        """线程安全触发（keyboard 回调线程调用）。"""
        logger.warning("收到急停信号（快捷键）", extra={"event": "emergency_stop"})
        if self._bus is not None:
            self._bus.publish_threadsafe(Event(EventType.EMERGENCY_STOP.value, {"source": "hotkey"}))

    async def _watch_flag(self) -> None:
        """轮询急停信号文件（独立脚本通道）。"""
        while True:
            await asyncio.sleep(self._poll_interval)
            if self._flag_file is not None and self._flag_file.exists():
                try:
                    self._flag_file.unlink()
                except OSError:
                    pass
                logger.warning("收到急停信号（信号文件）", extra={"event": "emergency_stop"})
                if self._bus is not None:
                    self._bus.publish(Event(EventType.EMERGENCY_STOP.value, {"source": "flag_file"}))

    async def disarm(self) -> None:
        """解除布防：注销快捷键、停止轮询。

        优先只注销本实例注册的热键（OBS-4），避免 remove_all_hotkeys
        影响系统中其它库注册的全局热键。
        """
        if self._hotkey_registered:
            try:
                import keyboard

                if self._hotkey_handle is not None:
                    try:
                        keyboard.remove_hotkey(self._hotkey_handle)
                    except Exception:
                        keyboard.remove_all_hotkeys()
                else:
                    keyboard.remove_all_hotkeys()
            except Exception:
                pass
            self._hotkey_registered = False
            self._hotkey_handle = None
        if self._watch_task is not None:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
            self._watch_task = None
        self._stop_event.set()
