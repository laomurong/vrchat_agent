"""事件总线：模块间唯一通信通道（共享知识约定第 8 节第 1 条）。

事件类型（统一字符串常量）：
- utterance_ready  语音转写完成（A → F）
- vision_snapshot  视觉采样快照（E → F / 仪表盘）
- decision_ready   决策完成（F → 日志/仪表盘）
- tts_started      TTS 开始播放（F → 自抑制）
- tts_finished     TTS 播放结束（F → 自抑制解除）
- action_executed  动作已执行（F → 日志/仪表盘）
- state_changed    状态流转（F → 日志/仪表盘）
- emergency_stop   急停信号（EmergencyStop → F）
- pipeline_state   流水线状态（F → 仪表盘，P2-1）

事件 ts 一律 time.time()（本地墙钟，仅用于排序与延迟统计）。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Awaitable, Callable

logger = logging.getLogger("vrchat_agent.events")


class EventType(str, Enum):
    """统一事件类型枚举（值即事件名字符串）。"""

    UTTERANCE_READY = "utterance_ready"
    VISION_SNAPSHOT = "vision_snapshot"
    DECISION_READY = "decision_ready"
    TTS_STARTED = "tts_started"
    TTS_FINISHED = "tts_finished"
    ACTION_EXECUTED = "action_executed"
    STATE_CHANGED = "state_changed"
    EMERGENCY_STOP = "emergency_stop"
    PIPELINE_STATE = "pipeline_state"


@dataclass
class Event:
    """事件数据：type + payload + ts。"""

    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


Handler = Callable[[Event], Awaitable[None] | None]


class EventBus:
    """异步事件总线。

    - subscribe：注册回调处理器（async 或 sync），publish 时调度执行
    - stream：异步迭代器订阅（每个订阅者独立队列）
    - publish：同步投递（推入流队列 + 调度处理器任务），不阻塞调用方
    - publish_threadsafe：供非事件循环线程（如急停快捷键回调）安全投递
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._streams: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pending: list[Event] = []

    # ---- 生命周期 ----
    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        """绑定事件循环（main 装配时调用）；绑定后补投 pending 事件。"""
        self._loop = loop
        if self._pending:
            pending, self._pending = self._pending, []
            for event in pending:
                self.publish(event)

    # ---- 订阅 ----
    def subscribe(self, event_type: str, handler: Handler) -> None:
        """注册事件处理器（按事件类型分发）。"""
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Handler) -> None:
        """注销事件处理器。"""
        if handler in self._handlers.get(event_type, []):
            self._handlers[event_type].remove(handler)

    async def stream(self, event_type: str) -> AsyncIterator[Event]:
        """异步迭代订阅：消费指定类型事件的无限流。"""
        queue: asyncio.Queue = asyncio.Queue()
        self._streams[event_type].append(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            try:
                self._streams[event_type].remove(queue)
            except ValueError:
                pass

    # ---- 发布 ----
    def publish(self, event: Event) -> None:
        """同步发布事件：推入流队列 + 调度处理器任务。

        无事件循环上下文（未 bind）时缓存到 _pending，bind 后补投一次；
        此时不推流队列，避免"入流队列 + 缓存补投"双路径导致重复投递（OBS-2）。
        """
        if self._get_running_loop() is None and self._loop is None:
            self._pending.append(event)
            return
        # 1) 流式订阅者
        for queue in self._streams.get(event.type, []):
            queue.put_nowait(event)
        # 2) 回调订阅者
        for handler in self._handlers.get(event.type, []):
            self._schedule(handler, event)

    def publish_threadsafe(self, event: Event) -> None:
        """线程安全发布（供非事件循环线程调用，如 keyboard 快捷键回调）。"""
        loop = self._loop or self._get_running_loop()
        if loop is None:
            self._pending.append(event)
            return
        loop.call_soon_threadsafe(self.publish, event)

    # ---- 内部 ----
    def _schedule(self, handler: Handler, event: Event) -> None:
        loop = self._get_running_loop() or self._loop
        if loop is None:
            return  # publish 已在无 loop 时缓存
        try:
            result = handler(event)
        except Exception as exc:
            # 同步 handler 异常隔离：不冒泡到 publish、不阻断其他 handler（BUG-3）
            logger.error("事件处理器异常: %s", exc)
            return
        if asyncio.iscoroutine(result):
            task = loop.create_task(result)
            task.add_done_callback(self._on_task_done)

    @staticmethod
    def _on_task_done(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("事件处理器异常: %s", exc)

    @staticmethod
    def _get_running_loop() -> asyncio.AbstractEventLoop | None:
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return None
