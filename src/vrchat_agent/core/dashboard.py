"""G 配置与可观测：Dashboard 状态仪表盘（P1：G-04）。

订阅事件流，维护状态快照；终端面板输出（OBS 叠加 HTML 留作扩展）。
显示：状态 / 最近听到 / 最近回复 / 当前动作 / 延迟 / 用量统计 / 启动自检。

需求 3（事件驱动刷新）：安静时终端不刷屏，有对话/动作时才刷新——
- Dashboard 维护 `_dirty` 标志；监听关键事件（state_changed / utterance_ready /
  decision_ready / action_executed / tts_started / tts_finished / emergency_stop）
  置 dirty；
- run 循环 sleep(refresh_ms) 后仅当 dirty 才重绘并清标志；
- 另加超时兜底（force_refresh_s，默认 30s 无事件也强制刷一次，防卡死无输出）。
"""

from __future__ import annotations

import asyncio
import logging
import time

from ..events import Event, EventBus, EventType

logger = logging.getLogger("vrchat_agent.dashboard")

_STATE_ICON = {
    "IDLE": "🟢",
    "LISTENING": "🟡",
    "THINKING": "🔵",
    "SPEAKING": "🟣",
    "ACTION": "⚪",
    "EMERGENCY_STOP": "🔴",
}

# 需求 3：置 dirty 的关键事件（有消息流才刷新）
_DIRTY_EVENT_TYPES = (
    EventType.STATE_CHANGED.value,
    EventType.UTTERANCE_READY.value,
    EventType.DECISION_READY.value,
    EventType.ACTION_EXECUTED.value,
    EventType.TTS_STARTED.value,
    EventType.TTS_FINISHED.value,
    EventType.EMERGENCY_STOP.value,
)

# 超时兜底：无事件也强制刷新间隔（秒）
_FORCE_REFRESH_S = 30.0


class Dashboard:
    """状态仪表盘：事件订阅 + 终端渲染（事件驱动刷新）。"""

    def __init__(self, bus: EventBus, config=None, usage_tracker=None,
                 force_refresh_s: float = _FORCE_REFRESH_S) -> None:
        self._bus = bus
        self._config = config
        self._usage = usage_tracker
        self._force_refresh_s = max(0.05, float(force_refresh_s))
        self._dirty = False
        # 需求 2：启动自检报告（main 异步自检完成后写入，render 展示自检区）
        self.startup_report: dict | None = None
        self._snapshot: dict = {
            "state": "IDLE",
            "utterance": "",
            "reply": "",
            "actions": [],
            "latency_s": 0.0,
            "last_update": time.time(),
        }
        # 快照更新订阅（与旧实现一致）
        bus.subscribe(EventType.STATE_CHANGED.value, self._on_state)
        bus.subscribe(EventType.UTTERANCE_READY.value, self._on_utterance)
        bus.subscribe(EventType.DECISION_READY.value, self._on_decision)
        bus.subscribe(EventType.ACTION_EXECUTED.value, self._on_action)
        # 需求 3：全部关键事件统一置 dirty（快照更新仍由各自 handler 负责）
        for event_type in _DIRTY_EVENT_TYPES:
            bus.subscribe(event_type, self._mark_dirty)

    # ---- 事件处理器 ----
    def _mark_dirty(self, event: Event | None = None) -> None:
        """置脏（同步 handler；被关键事件订阅调用）。"""
        self._dirty = True

    async def _on_state(self, event: Event) -> None:
        self._snapshot["state"] = event.payload.get("to", self._snapshot["state"])
        if "latency_s" in event.payload:
            self._snapshot["latency_s"] = event.payload["latency_s"]
        self._snapshot["last_update"] = time.time()
        self._mark_dirty()

    async def _on_utterance(self, event: Event) -> None:
        self._snapshot["utterance"] = event.payload.get("text", "")
        self._mark_dirty()

    async def _on_decision(self, event: Event) -> None:
        self._snapshot["reply"] = event.payload.get("reply_text") or ""
        # 本次决策 = 新一轮对话开始：清空上一轮累积的动作快照，只显示本次动作。
        # （修复跨次对话累积：旧实现 actions[-3:] 保留最近 3 条不随状态清空，
        #   面板会混入上一轮的 move:/chatbox:reply 等，如 "chatbox:reply, move:, chatbox:reply"）
        self._snapshot["actions"] = []
        self._mark_dirty()

    async def _on_action(self, event: Event) -> None:
        params = event.payload.get("params", {}) or {}
        name = params.get("name", "")
        self._snapshot["actions"].append(f"{event.payload.get('kind')}:{name}")
        # 单次对话可能执行多条动作（emote + move + chatbox 等），保留最近 3 条展示
        self._snapshot["actions"] = self._snapshot["actions"][-3:]
        self._mark_dirty()

    @property
    def dirty(self) -> bool:
        """是否待重绘（测试/观测用）。"""
        return self._dirty

    # ---- 渲染 ----
    def render(self) -> str:
        """渲染终端面板文本。"""
        state = self._snapshot["state"]
        icon = _STATE_ICON.get(state, "⚪")
        lines = [
            "=" * 58,
            "  VRChat Agent 状态面板",
            "-" * 58,
            f"  状态: {icon} {state}",
            f"  听到: {self._snapshot['utterance'] or '-'}",
            f"  回复: {self._snapshot['reply'] or '-'}",
            f"  动作: {', '.join(self._snapshot['actions']) or '-'}",
            f"  延迟: {self._snapshot['latency_s']:.1f}s",
        ]
        if self._usage is not None:
            s = self._usage.summary()
            lines.append(
                f"  用量: LLM={s['llm_calls']} TTS={s['tts_calls']} 成本≈¥{s['cost']:.4f}"
            )
        # 需求 2：自检区（异步自检完成后展示，未完成/未启用时不显示）
        report = self.startup_report
        if report:
            checks = report.get("checks", {}) or {}

            def _ok(name: str) -> str:
                c = checks.get(name) or {}
                return "✓" if c.get("ok") else "✗"

            lines.append(
                f"  自检: LLM{_ok('llm')} TTS{_ok('tts')} 视觉{_ok('vision')} "
                f"OSC{_ok('osc')} 校准{_ok('calibration')}"
            )
        lines.append("=" * 58)
        return "\n".join(lines)

    def to_overlay_html(self) -> str:
        """生成 OBS 可叠加 HTML（P1 扩展，透明背景）。"""
        state = self._snapshot["state"]
        return f"""<div style="font-family:sans-serif;background:rgba(0,0,0,0.55);color:#fff;
                     padding:12px 16px;border-radius:8px;width:420px;">
  <div style="font-size:18px;font-weight:bold;">{_STATE_ICON.get(state,'⚪')} {state}</div>
  <div>听到: {self._snapshot['utterance']}</div>
  <div>回复: {self._snapshot['reply']}</div>
  <div>动作: {', '.join(self._snapshot['actions']) or '-'}</div>
</div>"""

    async def run(self) -> None:
        """后台刷新循环（阻塞运行）。

        需求 3：事件驱动——仅当 dirty（有关键消息流）或超过 force_refresh_s
        无事件时重绘；热重载（A1）：refresh_ms 每次循环读取，改刷新间隔无需重启。
        """
        last_render = time.time()
        while True:
            refresh = (self._config.dashboard.refresh_ms / 1000.0) if self._config else 0.5
            await asyncio.sleep(refresh)
            now = time.time()
            force = (now - last_render) >= self._force_refresh_s
            if self._dirty or force:
                if self._config is None or self._config.dashboard.terminal:
                    print("\033[2J\033[H" + self.render(), flush=True)
                self._dirty = False
                last_render = now
