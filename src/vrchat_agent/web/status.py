"""Web 状态存储：订阅 EventBus 维护快照 + 时间线（M4 增量，W-03）。

与 Dashboard 是独立订阅者，互不干扰：Dashboard 继续刷终端，Web 只喂 SSE。
所有事件均为既有类型（state_changed/utterance_ready/decision_ready/
action_executed/tts_started/tts_finished/emergency_stop），不新增类型。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import AsyncIterator
from typing import Any

from ..events import Event, EventBus, EventType

logger = logging.getLogger("vrchat_agent.web.status")

# SSE 转发的事件类型（均为既有事件 + P2-1 vrc_2fa_status，不新增类型以外的通道）
FORWARDED_TYPES = [
    EventType.STATE_CHANGED.value,
    EventType.UTTERANCE_READY.value,
    EventType.DECISION_READY.value,
    EventType.ACTION_EXECUTED.value,
    EventType.TTS_STARTED.value,
    EventType.TTS_FINISHED.value,
    EventType.EMERGENCY_STOP.value,
    EventType.PIPELINE_STATE.value,
    "vrc_2fa_status",
]

_DEFAULT_TIMELINE_MAX = 200
_ACTION_KEEP = 3

# A-12：每个 SSE 连接事件队列上限。慢客户端消费不及时导致积压时，
# 丢弃最旧事件（get_nowait 后 put），保证内存有界、事件循环不被拖垮。
_SSE_QUEUE_MAXSIZE = 500


class WebStatusStore:
    """Web 状态订阅者：维护快照 + 最近时间线 + 每连接事件队列。

    生命周期：create_app 时创建并 subscribe；每个 SSE 连接 register_queue()
    建立独立队列；snapshot() 返回当前快照（含 UsageTracker.summary() 与
    uptime）；连接断开 remove_queue() 清理。
    """

    def __init__(
        self,
        bus: EventBus,
        usage=None,
        max_timeline: int = _DEFAULT_TIMELINE_MAX,
    ) -> None:
        self._bus = bus
        self._usage = usage
        self._max_timeline = max_timeline
        self._started = time.time()
        self._snapshot: dict[str, Any] = {
            "state": "IDLE",
            "utterance": "",
            "reply": "",
            "actions": [],
            "latency_s": 0.0,
            "last_update": time.time(),
            # P2-1：VRChat API 2FA 等待状态（vrc_2fa_status 事件驱动；未发生时为默认空态）
            "vrc_2fa": {
                "waiting": False,
                "methods": [],
                "timeout_s": 120.0,
                "wait_elapsed_s": 0.0,
                "wait_remaining_s": 0.0,
                "code_env_configured": False,
                "submitted": False,
            },
        }
        self._timeline: deque[dict[str, Any]] = deque(maxlen=max_timeline)
        self._queues: list[asyncio.Queue[Event]] = []
        self._subscribe()

    # ---- 订阅 ----
    def _subscribe(self) -> None:
        """订阅全部转发事件类型（与 Dashboard 同款方式，不侵入其类）。"""
        for event_type in FORWARDED_TYPES:
            self._bus.subscribe(event_type, self._on_event)

    def _unsubscribe(self) -> None:
        for event_type in FORWARDED_TYPES:
            self._bus.unsubscribe(event_type, self._on_event)

    # ---- 事件处理 ----
    def _on_event(self, event: Event) -> None:
        """同步处理事件：更新快照 + 追加时间线 + 推入各连接队列。"""
        try:
            self._update_snapshot(event)
            self._timeline.append(
                {"ts": round(event.ts, 3), "type": event.type, "payload": event.payload}
            )
            for queue in self._queues:
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    # A-12：慢客户端队列满 → 丢弃最旧事件后重试一次，仍满则丢新事件
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    try:
                        queue.put_nowait(event)
                    except asyncio.QueueFull:
                        pass
        except Exception:
            logger.exception("WebStatusStore 事件处理异常", extra={"event": "web_error"})

    def _update_snapshot(self, event: Event) -> None:
        """按 Dashboard 同款规则更新快照字段。"""
        payload = event.payload
        if event.type == EventType.STATE_CHANGED.value:
            self._snapshot["state"] = payload.get("to", self._snapshot["state"])
            if "latency_s" in payload:
                self._snapshot["latency_s"] = payload["latency_s"]
            self._snapshot["last_update"] = time.time()
        elif event.type == EventType.UTTERANCE_READY.value:
            self._snapshot["utterance"] = payload.get("text", "")
        elif event.type == EventType.DECISION_READY.value:
            self._snapshot["reply"] = payload.get("reply_text") or ""
        elif event.type == EventType.ACTION_EXECUTED.value:
            params = payload.get("params", {}) or {}
            name = params.get("name", "")
            kind = payload.get("kind", "")
            self._snapshot["actions"].append(f"{kind}:{name}" if kind else str(name))
            self._snapshot["actions"] = self._snapshot["actions"][-_ACTION_KEEP:]
        elif event.type == "vrc_2fa_status":
            # P2-1：2FA 等待状态直接覆盖快照字段（waiting/methods/超时/是否已提交）
            self._snapshot["vrc_2fa"] = dict(payload or {})

    # ---- 快照 ----
    def snapshot(self) -> dict[str, Any]:
        """当前快照（含用量/uptime/时间线），供 GET /api/status 与 SSE 首帧。"""
        snap = dict(self._snapshot)
        if self._usage is not None:
            snap["usage"] = self._usage.summary()
        else:
            snap["usage"] = {}
        snap["uptime_s"] = round(time.time() - self._started, 1)
        snap["timeline"] = list(self._timeline)
        return snap

    # ---- 每连接队列 ----
    def register_queue(self) -> asyncio.Queue[Event]:
        """为单个 SSE 连接注册事件队列（A-12：有界，满时丢弃最旧事件）。"""
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=_SSE_QUEUE_MAXSIZE)
        self._queues.append(queue)
        return queue

    def remove_queue(self, queue: asyncio.Queue[Event]) -> None:
        """连接关闭时清理事件队列。"""
        try:
            self._queues.remove(queue)
        except ValueError:
            pass

    @property
    def active_connections(self) -> int:
        """当前活跃 SSE 连接数（测试/观测用）。"""
        return len(self._queues)

    async def events(self) -> AsyncIterator[Event]:
        """事件流（兼容接口）：为单个连接维护独立队列，断线自动清理。"""
        queue = self.register_queue()
        try:
            while True:
                yield await queue.get()
        finally:
            self.remove_queue(queue)
