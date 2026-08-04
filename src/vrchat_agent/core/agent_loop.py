"""F 编排主循环：状态机 + 事件订阅 + 打断 + 超时降级 + 流水线并发（P0 重构）。

状态机：IDLE → LISTENING → THINKING → SPEAKING → IDLE
        SPEAKING → THINKING（打断，F-03）
        任意状态 → EMERGENCY_STOP → IDLE（急停旁路）

P0 流水线重构：
  旧：_processing_lock 一把大锁覆盖 ASR→LLM→TTS→播放
  新：LLM Stage(Semaphore) + TTS Stage(Semaphore) + OutputSlot(Lock)
      ASR 直接发布 UTTERANCE_READY → Pipeline 立即送 LLM

端到端延迟预算 < 8s：决策返回后，非 chatbox 动作与 TTS 合成并发执行（先发动作再播放语音）；
chatbox 动作从并发任务拆出、延迟到语音播放前同步发出（聊天框与语音同步，"开口即出字幕"）；
云端 LLM/TTS 失败统一超时降级（提示音 + 回到 IDLE），进程不崩溃。
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import unicodedata
from enum import Enum

import numpy as np

from ..action.actuator import Actuator
from ..brain.brain import Brain
from ..brain.schema import Action, Decision
from ..events import Event, EventBus, EventType
from ..speech.audio_output import AudioOutput
from ..speech.tts import TTSEngine, AudioClip
from ..vision.sampler import ImageSampler
from ..vision.vision_sensor import VisionSensor
from .idle_instinct import IdleInstinct
from .interrupt import InterruptController
from .output_slot import OutputSlot
from .utterance import Utterance, UtteranceState
from .utterance_queue import UtteranceQueue

logger = logging.getLogger("vrchat_agent.loop")

# 视觉关键词（仅 vision_mode=keyword_filter 时生效）：玩家输入命中任一关键词
# 才采样画面并传给 LLM；reference_only 模式不使用本列表（始终传画面，靠 prompt 约束）。
_VISION_KEYWORDS = (
    "看到",
    "看见",
    "画面",
    "场景",
    "周围",
    "谁在",
    "这是什么",
    "这是哪",
    "哪里",
    "在哪",
    "看",
    "长什么样",
    "穿",
    "颜色",
    "describe",
    "look",
    "see",
    "what",
    "who",
    "around",
)

# 自动同步 chatbox 的动作名（映射表 chatbox:reply → send_message{message}）
_CHATBOX_ACTION_NAME = "reply"
# VRChat chatbox 长度上限（超出部分由 ActionMapper 截断，此处只负责构造动作）
_CHATBOX_MAX_LEN = 140


def _should_attach_vision(text: str, mode: str) -> bool:
    """是否应采样画面并传给 LLM（reference_only 恒真；keyword_filter 需命中关键词）。"""
    if mode != "keyword_filter":
        return True
    low = (text or "").lower()
    return any(k in low for k in _VISION_KEYWORDS)


def _is_noise_text(text: str) -> bool:
    """判断 ASR 文本是否为纯噪音（标点/符号/空白/无意义短数字）。

    规则（命中任一即判定为噪音）：
    1. 纯标点/空白/符号/控制字符 → True（如 "."、"。"、"！"、"…"、"、"）
    2. strip 标点/符号后真实字符数 < 2 → True（如 "，。" → 0 字；"，好的" → 2 字放行）
    3. 仅含数字（无字母/CJK）且数字位数 < 5 → True（如 "123"、"1.2.3" 无用）
    4. 其余 → False（正常文本放行）
    """
    text = text.strip()
    if not text:
        return True

    # 规则 1：纯标点/空白/符号/控制字符
    if all(unicodedata.category(ch).startswith(('P', 'Z', 'S', 'C')) for ch in text):
        return True

    # 提取真实字符：字母（含 CJK）和数字
    real_chars = [ch for ch in text if unicodedata.category(ch).startswith(('L', 'N'))]

    # 规则 2：真实字符数 < 2
    if len(real_chars) < 2:
        return True

    # 规则 3：仅数字（无字母/CJK）且位数 < 5
    if all(unicodedata.category(ch).startswith('N') for ch in real_chars) and len(real_chars) < 5:
        return True

    return False


def _is_connection_error(exc: BaseException) -> bool:
    """检查异常链中是否包含连接级错误（网络不可达、连接拒绝等）。

    连接错误 → 静默跳过（不是"没听清"而是"连不上"）。
    内容错误（rate limit / 内容过滤等）→ 保留降级。
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None:
        if id(current) in seen:
            break
        seen.add(id(current))
        cls_name = type(current).__name__

        # httpx connection errors
        if cls_name in (
            'ConnectError', 'ConnectTimeout', 'RemoteProtocolError',
            'NetworkError', 'ReadError', 'WriteError', 'ReadTimeout',
            'PoolTimeout', 'WriteTimeout',
        ):
            return True
        # openai connection errors
        if cls_name == 'APIConnectionError':
            return True
        # stdlib connection errors
        if cls_name in (
            'ConnectionError', 'ConnectionRefusedError',
            'ConnectionResetError', 'ConnectionAbortedError',
            'BrokenPipeError',
        ):
            return True

        current = current.__cause__ or current.__context__

    return False


class AgentState(Enum):
    """编排状态机状态。

    P0 重构：全局 AgentState 反映"输出槽是否有活动"（宏观视角）。
    每个 utterance 的 UtteranceState 供 Web 监控（微观视角）。
    """

    IDLE = "IDLE"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    ACTION = "ACTION"
    EMERGENCY_STOP = "EMERGENCY_STOP"


# 合法流转表（急停除外：任意状态 → EMERGENCY_STOP 为旁路）
_ALLOWED_TRANSITIONS: dict[AgentState, set[AgentState]] = {
    AgentState.IDLE: {AgentState.LISTENING, AgentState.THINKING},
    AgentState.LISTENING: {AgentState.THINKING, AgentState.IDLE},
    AgentState.THINKING: {AgentState.SPEAKING, AgentState.ACTION, AgentState.IDLE},
    AgentState.SPEAKING: {AgentState.THINKING, AgentState.IDLE},
    AgentState.ACTION: {AgentState.SPEAKING, AgentState.THINKING, AgentState.IDLE},
    AgentState.EMERGENCY_STOP: {AgentState.IDLE},
}


class AgentLoop:
    """编排主循环：订阅事件，按状态机流转，驱动大脑/TTS/动作/视觉。

    P0 流水线并发：LLM/TTS Semaphore + OutputSlot 互斥播放。
    """

    def __init__(
        self,
        bus: EventBus,
        brain: Brain,
        tts: TTSEngine,
        audio_out: AudioOutput,
        actuator: Actuator,
        vision: VisionSensor | None = None,
        sampler: ImageSampler | None = None,
        config=None,
        interrupt: InterruptController | None = None,
    ) -> None:
        self._bus = bus
        self._brain = brain
        self._tts = tts
        self._audio_out = audio_out
        self._actuator = actuator
        self._vision = vision
        self._sampler = sampler
        self._config = config
        self._interrupt = interrupt or InterruptController(
            enabled=bool(config.agent.interrupt.enabled) if config else True
        )
        self._state = AgentState.IDLE
        self._running = False
        self._paused = False
        self._started_at = 0.0
        self._stop_event = asyncio.Event()
        self._processed_count = 0
        self._last_utterance = ""
        self._last_reply = ""
        self._last_latency_s = 0.0
        self._last_vision_b64: str | None = None
        self._emergency_stopped = False
        # 需求 1：进程级重启标志（Web restart → 置位 + stop）
        self._restart_requested = False
        # 需求 4：对话队列 + 批量合并（旧 UtteranceQueue，兼容保留）
        self._utterance_queue = UtteranceQueue(max_batch=3)
        # A-14：旧 _process_next 路径的互斥锁——在 __init__ 显式创建（此前每次
        # 调用临时新建 asyncio.Lock()，无互斥语义；现恢复"同时只处理一条对话"）。
        self._processing_lock = asyncio.Lock()

        # ---- P0 流水线并发 ----
        # LLM Semaphore：限制并发 LLM 请求数（config agent.pipeline.llm_semaphore）
        self._llm_semaphore = asyncio.Semaphore(self._pipeline_llm_semaphore())
        # TTS Semaphore：限制并发 TTS 请求数（config agent.pipeline.tts_semaphore）
        self._tts_semaphore = asyncio.Semaphore(self._pipeline_tts_semaphore())
        # OutputSlot：输出槽互斥调度
        self._output_slot = OutputSlot(maxsize=self._pipeline_output_queue_max())
        # 飞行中的 pipeline 任务集合（用于打断/急停批量取消）
        self._pipe_tasks: set[asyncio.Task] = set()
        # A-07：事件 handler 任务集合（run()/stop() 时统一取消，防止停后仍在跑）
        self._handler_tasks: set[asyncio.Task] = set()
        # A-07：空闲本能启动任务（run()/stop() 时统一取消）
        self._idle_task: asyncio.Task | None = None
        # 活跃 utterance 集合（key=utt.id，用于打断/急停批量设置 cancel_event）
        self._active_utterances: dict[str, Utterance] = {}
        # P1-1：utt.id → pipeline Task 映射（用于打断时精准取消飞行中 TTS）
        self._pipe_task_map: dict[str, asyncio.Task] = {}
        # 视觉校准后台任务（播放含 look 动作的 utterance 后触发）
        self._calibration_task: asyncio.Task | None = None
        # P1-5：ASR 去重窗口（text → timestamp）
        self._recent_texts: dict[str, float] = {}
        # P2-1：当前播放开始时间 + 音频时长（供仪表盘进度）
        self._play_started_at: float = 0.0
        self._play_clip_duration: float = 0.0
        # P2-3：跳过当前播放标志（供 skip 端点使用）
        self._skip_requested = False
        # 空闲本能微动作循环（IDLE 状态下周期性微小 OSC 动作）
        self._idle_instinct: IdleInstinct | None = None
        # 仅在 actuator 暴露 _mapper 时创建（MCPOSCActuator；NoopActuator 无）
        _mapper = getattr(self._actuator, '_mapper', None)
        if _mapper is not None and self._config is not None:
            try:
                # A-08：传 Config 门面（非构造期 IdleInstinctConfig 快照），
                # 使 idle_instinct.* 保存后真正热重载（每次 tick 读最新值）。
                self._idle_instinct = IdleInstinct(
                    self._actuator, _mapper, self._config,
                )
                logger.info("空闲本能已初始化")
            except Exception:
                logger.warning("空闲本能初始化失败，禁用", exc_info=True)

    # ------------------------------------------------------------------
    # 只读属性（供仪表盘/测试）
    # ------------------------------------------------------------------
    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def processed_count(self) -> int:
        return self._processed_count

    @property
    def last_utterance(self) -> str:
        return self._last_utterance

    @property
    def last_reply(self) -> str:
        return self._last_reply

    @property
    def last_latency_s(self) -> float:
        return self._last_latency_s

    # ------------------------------------------------------------------
    # 控制（Web 操控，任务 2：B3 主循环配合）
    # ------------------------------------------------------------------
    @property
    def paused(self) -> bool:
        """是否处于暂停态（暂停期间忽略 utterance 触发）。"""
        return self._paused

    @property
    def running(self) -> bool:
        """主循环是否存活。"""
        return self._running

    @property
    def emergency_stopped(self) -> bool:
        return self._emergency_stopped

    @property
    def restart_requested(self) -> bool:
        """是否已请求进程级重启（main build_and_run finally 据此拉起新进程）。"""
        return self._restart_requested

    @property
    def utterance_queue(self) -> UtteranceQueue:
        """对话队列（需求 4，测试/观测用）。"""
        return self._utterance_queue

    @property
    def output_slot(self) -> OutputSlot:
        """输出槽（P0 流水线，供测试/仪表盘观测）。"""
        return self._output_slot

    def pause(self) -> None:
        """暂停监听：主循环进入暂停态，忽略后续 utterance 触发（不退出进程）。"""
        if self._paused:
            return
        self._paused = True
        logger.info("主循环已暂停（忽略语音触发）", extra={"event": "control_pause"})

    def resume(self) -> None:
        """恢复监听：重新接受 utterance 触发。"""
        if not self._paused:
            return
        self._paused = False
        logger.info("主循环已恢复监听", extra={"event": "control_resume"})

    def restart(self) -> None:
        """请求进程级重启：置位标志 + 停止主循环（重新加载配置）。"""
        self._restart_requested = True
        self._stop_event.set()
        logger.info("已请求重启（重新加载配置）", extra={"event": "control_restart"})

    def control_status(self) -> dict:
        """Web 控制状态快照（GET /api/control/status 数据源）。"""
        uptime = 0.0
        if self._running and self._started_at > 0:
            uptime = max(0.0, time.time() - self._started_at)
        return {
            "running": self._running,
            "paused": self._paused,
            "emergency_stopped": self._emergency_stopped,
            "state": self._state.value if self._state else "IDLE",
            "uptime_s": round(uptime, 1),
            "processed_count": self._processed_count,
            "last_reply": self._last_reply or "",
            "restartable": True,
        }

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    async def run(self) -> None:
        """启动主循环（阻塞直到 stop()）。

        P0 重构：额外启动 _drain_output_slot 后台任务消费输出槽。
        """
        if self._running:
            return
        self._running = True
        self._started_at = time.time()
        self._stop_event = asyncio.Event()
        tasks = [
            asyncio.create_task(self._consume(EventType.UTTERANCE_READY.value, self._on_utterance)),
            asyncio.create_task(self._consume(EventType.EMERGENCY_STOP.value, self._on_emergency)),
            asyncio.create_task(self._consume(EventType.VISION_SNAPSHOT.value, self._on_vision)),
            asyncio.create_task(self._drain_output_slot()),
        ]
        try:
            await self._stop_event.wait()
        finally:
            self._running = False
            for task in tasks:
                task.cancel()
            # A-07：统一取消飞行中的 pipeline / 事件 handler / 校准 / idle 任务，
            # 防止 stop() 后任务继续向已停止的 OutputSlot enqueue 或发布事件。
            for task in list(self._pipe_tasks):
                task.cancel()
            for task in list(self._handler_tasks):
                task.cancel()
            cal_task = self._calibration_task
            if cal_task is not None and not cal_task.done():
                cal_task.cancel()
            idle_task = self._idle_task
            if idle_task is not None and not idle_task.done():
                idle_task.cancel()
            # 空闲本能内部循环任务一并停止（start() 外层任务已立即返回，需停内层）
            if self._idle_instinct is not None:
                await self._idle_instinct.stop()
            await asyncio.gather(
                *tasks,
                *list(self._pipe_tasks),
                *list(self._handler_tasks),
                *([cal_task] if cal_task is not None else []),
                *([idle_task] if idle_task is not None else []),
                return_exceptions=True,
            )
            self._pipe_tasks.clear()
            self._handler_tasks.clear()
            self._calibration_task = None
            self._idle_task = None

    def stop(self) -> None:
        """请求停止主循环。"""
        self._stop_event.set()

    async def close(self) -> None:
        """关闭主循环：停止空闲本能 + 设置停止事件。"""
        if self._idle_instinct is not None:
            await self._idle_instinct.stop()
        self.stop()

    async def _consume(self, event_type: str, handler) -> None:
        """消费事件流：每个事件调度为独立任务（保证打断/急停可抢占）。"""
        async for event in self._bus.stream(event_type):
            if not self._running:
                break
            try:
                task = asyncio.create_task(self._safe_handle(handler, event))
                # A-07：事件 handler 任务纳入跟踪，stop() 时统一取消
                self._handler_tasks.add(task)
                task.add_done_callback(self._handler_tasks.discard)
            except Exception:
                logger.exception("调度事件处理失败: %s", event_type)

    async def _safe_handle(self, handler, event: Event) -> None:
        """事件处理器包装：异常隔离，不中断消费循环。"""
        try:
            await handler(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("事件处理异常: %s", event.type)

    # ------------------------------------------------------------------
    # P0 流水线：_on_utterance → _run_utterance_pipeline → OutputSlot
    # ------------------------------------------------------------------
    async def _on_utterance(self, event: Event) -> None:
        """收到语音转写 → 创建 Utterance → 创建 pipeline 任务 → 立即返回。

        P0 重构：不再阻塞等待锁。ASR 产出直接进 LLM Stage（通过 Semaphore 限流）。
        打断/急停通过 Utterance.cancel_event 精准取消。
        """
        # 新 utterance 到来 → 打断空闲本能
        if self._idle_instinct is not None:
            await self._idle_instinct.stop()

        if self._paused:
            logger.debug("主循环暂停中，忽略语音触发")
            return
        text = (event.payload.get("text") or "").strip()
        if not text:
            return
        if _is_noise_text(text):
            logger.debug("ASR 噪音过滤: '%s'（纯标点/符号/无意义短文本）", text, extra={"event": "asr_noise_filtered"})
            return
        self._last_utterance = text

        # ---- P1-5：ASR 去重 ----
        now = time.time()
        dedup_window = self._pipeline_asr_dedup_window()
        self._recent_texts = {t: ts for t, ts in self._recent_texts.items() if now - ts < dedup_window}
        for old_text, old_ts in list(self._recent_texts.items()):
            if old_text != text and old_text in text:
                for uid, old_utt in list(self._active_utterances.items()):
                    if old_utt.text == old_text and old_utt.is_active:
                        old_utt.cancel()
                        old_task = self._pipe_task_map.pop(old_utt.id, None)
                        if old_task is not None and not old_task.done():
                            old_task.cancel()
                        logger.info(
                            "ASR 去重: '%s' → '%s'（超集修正，取消旧条）",
                            old_text, text,
                            extra={"event": "asr_dedup"},
                        )
                        break
                break  # 只取消第一个匹配（同一文本不应有多个活跃 utterance）
        self._recent_texts[text] = now

        # 取消正在运行的校准循环（新语音可打断静默校准）
        cal_task = self._calibration_task
        if cal_task is not None and not cal_task.done():
            cal_task.cancel()
            self._calibration_task = None

        # 创建 Utterance（ASR 产出文本，进入流水线）
        utt = Utterance(text=text, state=UtteranceState.ASR_DONE)
        self._active_utterances[utt.id] = utt

        # P2-1：发布流水线状态
        self._publish_pipeline_state()

        # 打断检测：输出槽有活动 + 打断开关开启
        if self._state == AgentState.SPEAKING:
            if self._interrupt_enabled():
                logger.info("收到打断请求（说话期间新语音）", extra={"event": "interrupt_requested"})
                self._interrupt.request_interrupt()
                # 取消所有飞行中的旧 utterance（新 utt 除外）
                for old_utt in list(self._active_utterances.values()):
                    if old_utt is not utt:
                        old_utt.cancel_event.set()
                        # P1-1：取消飞行中 TTS 的 pipeline task
                        if old_utt.state == UtteranceState.TTS_RUNNING:
                            old_task = self._pipe_task_map.pop(old_utt.id, None)
                            if old_task is not None and not old_task.done():
                                old_task.cancel()
                                logger.info(
                                    "打断取消飞行中 TTS: %s (text=%s)",
                                    old_utt.id, old_utt.text,
                                    extra={"event": "interrupt_cancel_tts"},
                                )
                await self._output_slot.interrupt()
                await self._audio_out.stop()
                self._transition(AgentState.THINKING)

        # 非说话态 + 旧队列模式兼容：若 utterance_queue 开关开启且
        # pipeline 已在处理中（有活跃 utterance），仍按旧队列逻辑入队。
        # 但 P0 流水线模式下通常不需要此路径（Semaphore 自动排队）。
        if self._state != AgentState.SPEAKING and self._utterance_queue_enabled():
            # 旧队列兼容：仅当显式配置 semaphore=1（退化串行）且有活跃任务时入队
            if self._pipeline_llm_semaphore() <= 1 and len(self._active_utterances) > 1:
                queued = self._utterance_queue.push(text)
                if queued:
                    logger.info(
                        "对话排队（旧队列兼容，当前 %d 条）", len(self._utterance_queue),
                        extra={"event": "utterance_queued"},
                    )
                    self._active_utterances.pop(utt.id, None)
                    return

        # 创建 pipeline 任务（不 await，让其独立跑）
        task = asyncio.create_task(self._run_utterance_pipeline(utt))
        self._pipe_tasks.add(task)
        self._pipe_task_map[utt.id] = task  # P1-1：utt.id → task 映射
        task.add_done_callback(lambda t: self._pipe_tasks.discard(t))
        task.add_done_callback(lambda t, uid=utt.id: self._pipe_task_map.pop(uid, None))

    async def _run_utterance_pipeline(self, utt: Utterance) -> None:
        """单条 utterance 的完整流水线：LLM → TTS → enqueue OutputSlot。

        各阶段通过 Utterance.cancel_event 支持精准取消（打断/急停）。
        LLM/TTS 通过 Semaphore 限流（可配置并发数）。
        错误隔离：单条失败不影响其他 utterance。
        """
        # 有新的 pipeline 任务 → 停止空闲本能
        if self._idle_instinct is not None:
            await self._idle_instinct.stop()

        try:
            # ---- 阶段 0：检查取消 ----
            if utt.cancel_event.is_set():
                utt.state = UtteranceState.CANCELLED
                self._publish_pipeline_state()
                return

            # ---- 阶段 1：LLM 决策 ----
            utt.state = UtteranceState.LLM_RUNNING
            self._publish_pipeline_state()
            # 首次进入 pipeline → THINKING
            if self._state == AgentState.IDLE:
                self._transition(AgentState.THINKING)

            # 视觉采样（LLM semaphore 外进行，减少持锁时间）
            vision_b64 = None
            if self._vision is not None and self._sampler is not None:
                if _should_attach_vision(utt.text, self._vision_mode()):
                    vision_b64 = await self._sampler.sample(self._vision)
                else:
                    logger.debug("vision_mode=keyword_filter：输入未命中视觉关键词，跳过画面采样")

            async with self._llm_semaphore:
                if utt.cancel_event.is_set():
                    utt.state = UtteranceState.CANCELLED
                    self._publish_pipeline_state()
                    return
                timeout = self._timeout("agent.loop.think_timeout", 20.0)
                try:
                    decision = await asyncio.wait_for(
                        self._brain.decide(utt.text, vision_b64), timeout=timeout
                    )
                except asyncio.TimeoutError:
                    logger.error("LLM 决策超时 (%.1fs)", timeout, extra={"event": "llm_timeout"})
                    decision = Decision(respond=False, reply_text=None, actions=[])
                    await self._degrade()
                except Exception as exc:
                    logger.exception("LLM 决策失败", extra={"event": "llm_error"})
                    decision = Decision(respond=False, reply_text=None, actions=[])
                    if _is_connection_error(exc):
                        logger.warning(
                            "LLM 连接失败，静默跳过（网络不可达，非内容错误）",
                            extra={"event": "llm_connection_error"},
                        )
                    else:
                        await self._degrade()

            utt.decision = decision
            utt.state = UtteranceState.LLM_DONE
            utt.respond = decision.respond

            # P2-2：LLM 决策后立即发送打字指示器（respond=True 时）
            if decision.respond:
                try:
                    await self._actuator.execute(Action(kind="chatbox", params={"name": "typing"}))
                except Exception:
                    logger.debug("打字指示器发送失败（忽略）", extra={"event": "typing_failed"})

            self._publish_pipeline_state()

            self._bus.publish(
                Event(
                    EventType.DECISION_READY.value,
                    {
                        "utterance": utt.text,
                        "respond": decision.respond,
                        "reply_text": decision.reply_text,
                        "emotion": decision.emotion,
                        "actions": [a.model_dump() for a in decision.actions],
                    },
                )
            )

            # ---- 阶段 2：急停/取消检查 ----
            if self._emergency_stopped or utt.cancel_event.is_set():
                utt.state = UtteranceState.CANCELLED
                self._publish_pipeline_state()
                return

            # ---- 阶段 3：respond=False → 直接结束 ----
            if not decision.respond or not decision.reply_text:
                # 仅动作不回应
                await self._execute_actions(decision.actions)
                self._last_reply = decision.reply_text or ""
                self._processed_count += 1
                utt.state = UtteranceState.DONE
                return

            # ---- 阶段 4：TTS 合成 ----
            utt.state = UtteranceState.TTS_RUNNING
            self._publish_pipeline_state()
            async with self._tts_semaphore:
                if utt.cancel_event.is_set():
                    utt.state = UtteranceState.CANCELLED
                    self._publish_pipeline_state()
                    return
                actions = self._attach_chatbox_action(decision.actions, decision.reply_text)
                utt.actions = [a for a in actions if a.kind != "chatbox"]
                utt.chatbox_actions = [a for a in actions if a.kind == "chatbox"]

                tts_timeout = self._timeout("agent.loop.tts_timeout", 10.0)
                clip: AudioClip | None = None
                try:
                    clip = await asyncio.wait_for(
                        self._tts.synthesize(decision.reply_text, decision.emotion),
                        timeout=tts_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.error("TTS 合成超时", extra={"event": "tts_failed"})
                    await self._degrade()
                except Exception:
                    logger.exception("TTS 合成失败", extra={"event": "tts_failed"})
                    await self._degrade()

            # ---- 阶段 5：TTS 完成/失败处理 ----
            if utt.cancel_event.is_set():
                utt.state = UtteranceState.CANCELLED
                self._publish_pipeline_state()
                return

            if clip is not None:
                utt.clip = clip
                utt.state = UtteranceState.TTS_DONE
                self._publish_pipeline_state()
                # 入队输出槽（等待播放）
                try:
                    await self._output_slot.enqueue(utt)
                except asyncio.QueueFull:
                    logger.warning("输出槽队列已满，丢弃 utterance: %s", utt.id)
                    utt.state = UtteranceState.CANCELLED
                    return
                # 延迟统计：从 ASR 产出到 TTS 完成
                self._last_latency_s = time.time() - utt.created_at
            else:
                # TTS 失败：非 chatbox 动作仍照常执行
                await self._safe_exec(utt.actions)
                self._last_reply = decision.reply_text or ""
                self._processed_count += 1
                utt.state = UtteranceState.CANCELLED

        except asyncio.CancelledError:
            utt.state = UtteranceState.CANCELLED
            self._publish_pipeline_state()
            raise
        except Exception:
            logger.exception("pipeline 异常: %s", utt.id)
            utt.state = UtteranceState.CANCELLED
            self._publish_pipeline_state()
        finally:
            self._active_utterances.pop(utt.id, None)
            # 从 pipe_tasks 中移除当前任务（done callback 在 finally 后才触发）
            try:
                self._pipe_tasks.discard(asyncio.current_task())
            except RuntimeError:
                pass
            # 仅在终端状态（DONE/CANCELLED）检查是否回 IDLE
            # QUEUED 的 utterance 由 _drain_output_slot 负责收尾
            if utt.state in (UtteranceState.DONE, UtteranceState.CANCELLED):
                self._maybe_idle()

    # ------------------------------------------------------------------
    # P0 输出槽消费
    # ------------------------------------------------------------------
    async def _drain_output_slot(self) -> None:
        """后台任务：持续消费输出槽队列，逐条播放。

        OutputSlot.wait_and_play() 内部 Lock 互斥 + gather 三路并发。
        播放前通过 on_start 回调做 SPEAKING transition + TTS_STARTED 事件。
        播放完成后写入 _last_reply / _processed_count，并管理全局状态。
        """

        async def _on_play_start(utt: Utterance) -> None:
            """播放开始回调：SPEAKING transition + TTS_STARTED 事件 + 播放计时。"""
            latency = time.time() - utt.created_at
            self._last_latency_s = latency
            self._play_started_at = time.time()
            self._play_clip_duration = utt.clip.duration if utt.clip else 0.0
            # A-02：标记已进入播放（TTS_STARTED 已发布）——取消时据此补发 TTS_FINISHED
            utt.started_playing = True
            self._transition(AgentState.SPEAKING, latency_s=latency)
            self._publish_pipeline_state()
            if utt.decision and utt.decision.reply_text:
                self._bus.publish(
                    Event(EventType.TTS_STARTED.value, {"text": utt.decision.reply_text})
                )

        while self._running:
            # P2-3：跳过当前播放（兜底；A-04 后 skip() 已直接执行跳过）
            if self._skip_requested:
                self._skip_requested = False
                if self._output_slot.is_playing:
                    await self._audio_out.stop()
                    logger.info("已跳过当前播放", extra={"event": "skip_executed"})
                    self._publish_pipeline_state()
                    continue

            utt = await self._output_slot.wait_and_play(
                play_func=self._run_play,
                exec_func=self._safe_exec,
                on_start=_on_play_start,
            )
            if utt is None:
                # 输出槽暂不可用（急停窗口）：事件阻塞等待复位（A-01，替代 50ms 轮询）
                await self._output_slot.wait_until_ready()
                continue

            # 播放完成后的收尾（打断的也计数，匹配旧行为）
            self._processed_count += 1
            if utt.state != UtteranceState.CANCELLED:
                utt.state = UtteranceState.DONE
                if utt.decision and utt.decision.reply_text:
                    self._last_reply = utt.decision.reply_text
                # 发布 TTS_FINISHED（播放完成后）
                self._bus.publish(Event(EventType.TTS_FINISHED.value, {}))
            else:
                # A-02：播放中被取消（打断/急停）也补发 TTS_FINISHED，
                # 解除自抑制静音窗口——否则首次打断后 Agent 永久"聋"。
                # 仅当确实进入过播放（TTS_STARTED 已发布）才补发，避免
                # 未播放即取消的 utterance 误关正在播放的另一条静音窗口。
                if utt.started_playing:
                    self._bus.publish(Event(EventType.TTS_FINISHED.value, {}))

            # 播放完成后检测是否需要触发视觉校准循环
            if utt.state == UtteranceState.DONE and utt.actions:
                if any(a.kind == "look" for a in utt.actions):
                    if self._vision is not None and self._sampler is not None:
                        logger.info(
                            "触发视觉校准循环: %s",
                            utt.text,
                            extra={"event": "calibration_trigger"},
                        )
                        self._calibration_task = asyncio.create_task(
                            self._run_calibration(utt.text)
                        )

            self._maybe_idle()
            # P2-1：播放完成后发布最新流水线状态
            self._publish_pipeline_state()

    def _maybe_idle(self) -> None:
        """检查是否应回 IDLE：输出槽空闲 + 无活跃 utterance + 无飞行 pipeline 任务。"""
        if (
            not self._output_slot.is_playing
            and self._output_slot.queue_size() == 0
            and len(self._active_utterances) == 0
            and len(self._pipe_tasks) == 0
        ):
            if self._state in (AgentState.THINKING, AgentState.SPEAKING, AgentState.ACTION):
                self._transition(AgentState.IDLE)
                # 进入 IDLE 后启动空闲本能微动作循环（A-07：任务纳入跟踪）
                if self._idle_instinct is not None:
                    self._idle_task = asyncio.create_task(self._idle_instinct.start())

    # ------------------------------------------------------------------
    # 视觉校准（播放含 look 动作的 utterance 后触发）
    # ------------------------------------------------------------------
    async def _run_calibration(self, target_desc: str) -> None:
        """触发视觉校准循环（后台任务）。

        截图→LLM判断→look微调，最多3次。期间静默（不发TTS/文字）。
        被新语音打断时通过 task.cancel() 取消。
        """
        from .visual_calibration import _visual_calibration_loop

        try:
            await _visual_calibration_loop(
                target_desc=target_desc,
                vision=self._vision,
                sampler=self._sampler,
                llm_client=self._brain._llm,
                actuator=self._actuator,
                max_iterations=3,
            )
        except asyncio.CancelledError:
            logger.info("校准循环被外部取消")
        except Exception:
            logger.exception("校准循环异常")
        finally:
            self._calibration_task = None

    # ------------------------------------------------------------------
    # 事件处理器（旧语义兼容 + P0 适配）
    # ------------------------------------------------------------------
    async def _on_vision(self, event: Event) -> None:
        """收到视觉快照：保存最新画面（P1：视觉事件触发，默认仅记录）。"""
        self._last_vision_b64 = event.payload.get("image_b64")

    async def _on_emergency(self, event: Event) -> None:
        """急停旁路：任意状态 → EMERGENCY_STOP → IDLE（<1s）。

        P0 重构：取消所有活跃 utterance → 清空输出槽 → 停音频 → 复位动作。
        """
        if self._state == AgentState.EMERGENCY_STOP:
            return
        # 急停 → 停止空闲本能
        if self._idle_instinct is not None:
            await self._idle_instinct.stop()
        logger.warning("执行急停: 取消所有飞行中任务 + 静音输出 + 复位动作", extra={"event": "emergency_stop"})
        self._emergency_stopped = True
        # 需求 4：急停清空对话队列（丢弃未处理对话）
        self._utterance_queue.clear()
        # P0：取消所有活跃 utterance
        for old_utt in list(self._active_utterances.values()):
            logger.info(
                "急停取消 utterance: %s (state=%s, text=%s)",
                old_utt.id, old_utt.state.value, old_utt.text,
                extra={"event": "emergency_cancel"},
            )
            old_utt.cancel()
        self._active_utterances.clear()
        # 取消所有飞行中的 pipeline 任务
        for task in list(self._pipe_tasks):
            if not task.done():
                task.cancel()
        self._pipe_tasks.clear()
        self._pipe_task_map.clear()  # P1-3：清理 task 映射
        # P1-5：急停时清理 ASR 去重缓存
        self._recent_texts.clear()
        # 取消校准循环
        cal_task = self._calibration_task
        if cal_task is not None and not cal_task.done():
            cal_task.cancel()
            self._calibration_task = None
        # 清空输出槽
        await self._output_slot.clear()
        # P2-1：急停后发布最新流水线状态
        self._publish_pipeline_state()

        self._transition(AgentState.EMERGENCY_STOP)
        try:
            await self._audio_out.stop()
        except Exception:
            pass
        try:
            await self._actuator.reset()
        except Exception:
            logger.exception("急停动作复位失败")
        self._state = AgentState.IDLE
        # A-03：急停可恢复 —— 复位紧急停止标志（与 legacy _handle_utterance
        # 开头复位语义一致），否则急停后所有 move/look/emote 动作永久被跳过。
        self._emergency_stopped = False
        self._bus.publish(
            Event(
                EventType.STATE_CHANGED.value,
                {
                    "from": AgentState.EMERGENCY_STOP.value,
                    "to": AgentState.IDLE.value,
                    "ts": time.time(),
                    "emergency": True,
                },
            )
        )

    def emergency_stop(self) -> None:
        """同步急停入口（线程安全，供外部直接调用）。"""
        self._bus.publish_threadsafe(Event(EventType.EMERGENCY_STOP.value, {"source": "external"}))

    # ------------------------------------------------------------------
    # 核心处理：旧 _handle_utterance（保留用于测试/向后兼容）
    # ------------------------------------------------------------------
    async def _handle_utterance(self, text: str) -> None:
        """一次完整交互：决策 → 动作+语音 并发执行（旧实现，保留兼容）。

        P0 重构后，主路径改为 _run_utterance_pipeline + OutputSlot。
        本方法保留用于旧测试兼容和 semaphore=1 退化串行场景。
        """
        self._interrupt.clear()  # 重置打断标志（防止上一轮残留）
        self._emergency_stopped = False
        owner_state = AgentState.THINKING
        t0 = time.time()
        vision_b64 = None
        if self._vision is not None and self._sampler is not None:
            if _should_attach_vision(text, self._vision_mode()):
                vision_b64 = await self._sampler.sample(self._vision)
            else:
                logger.debug("vision_mode=keyword_filter：输入未命中视觉关键词，跳过画面采样")

        timeout = self._timeout("agent.loop.think_timeout", 20.0)
        try:
            decision = await asyncio.wait_for(self._brain.decide(text, vision_b64), timeout=timeout)
        except asyncio.TimeoutError:
            logger.error("LLM 决策超时 (%.1fs)", timeout, extra={"event": "llm_timeout"})
            decision = Decision(respond=False, reply_text=None, actions=[])
            await self._degrade()
        except Exception:
            logger.exception("LLM 决策失败", extra={"event": "llm_error"})
            decision = Decision(respond=False, reply_text=None, actions=[])
            await self._degrade()

        self._bus.publish(
            Event(
                EventType.DECISION_READY.value,
                {
                    "utterance": text,
                    "respond": decision.respond,
                    "reply_text": decision.reply_text,
                    "emotion": decision.emotion,
                    "actions": [a.model_dump() for a in decision.actions],
                },
            )
        )

        if self._emergency_stopped:
            pass
        elif decision.respond and decision.reply_text:
            actions = self._attach_chatbox_action(decision.actions, decision.reply_text)
            chatbox_actions = [a for a in actions if a.kind == "chatbox"]
            other_actions = [a for a in actions if a.kind != "chatbox"]
            clip: AudioClip | None = None
            tts_timeout = self._timeout("agent.loop.tts_timeout", 10.0)
            try:
                clip = await asyncio.wait_for(
                    self._tts.synthesize(decision.reply_text, decision.emotion), timeout=tts_timeout
                )
            except asyncio.TimeoutError:
                logger.error("TTS 合成超时", extra={"event": "tts_failed"})
                await self._degrade()
            except Exception:
                logger.exception("TTS 合成失败", extra={"event": "tts_failed"})
                await self._degrade()

            if clip is not None and not self._interrupt.should_interrupt() and not self._emergency_stopped:
                latency = time.time() - t0
                self._last_latency_s = latency
                self._transition(AgentState.SPEAKING, latency_s=latency)
                owner_state = AgentState.SPEAKING
                self._bus.publish(Event(EventType.TTS_STARTED.value, {"text": decision.reply_text}))
                await asyncio.gather(
                    self._run_play(clip),
                    self._safe_exec(other_actions),
                    self._safe_exec(chatbox_actions),
                )
            else:
                if clip is not None:
                    self._interrupt.clear()
                await self._safe_exec(other_actions)
        else:
            await self._execute_actions(decision.actions)

        self._last_reply = decision.reply_text or ""
        self._processed_count += 1
        if self._state == owner_state:
            self._transition(AgentState.IDLE)

    async def _process_next(self, text: str) -> None:
        """串行处理一条对话（旧 _processing_lock 路径，P0 兼容保留）。

        P0 重构后主路径不再经过此方法。保留用于：
        - 旧 UtteranceQueue 模式兼容
        - 测试引用

        A-14：互斥锁在 __init__ 显式创建（self._processing_lock），并发调用时
        真正串行，恢复互斥语义。
        """
        async with self._processing_lock:
            self._utterance_queue.processing = True
            try:
                if self._state in (
                    AgentState.IDLE,
                    AgentState.LISTENING,
                    AgentState.THINKING,
                    AgentState.ACTION,
                ):
                    if self._state == AgentState.IDLE:
                        self._transition(AgentState.LISTENING)
                    if self._state != AgentState.THINKING:
                        self._transition(AgentState.THINKING)
                    await self._handle_utterance(text)
                await self._drain_utterance_queue()
            finally:
                self._utterance_queue.processing = False

    async def _drain_utterance_queue(self) -> None:
        """消费对话队列：≥max_batch 合并批量处理（旧队列模式，兼容保留）。"""
        if not self._utterance_queue_enabled():
            self._utterance_queue.clear()
            return
        while True:
            if self._state == AgentState.EMERGENCY_STOP:
                self._utterance_queue.clear()
                return
            batch = self._utterance_queue.pop_batch(self._utterance_queue_max_batch())
            if batch is None:
                return
            text, count = batch
            if count > 1:
                logger.info("合并 %d 条对话批量处理", count, extra={"event": "utterance_batched"})
            await self._handle_utterance(text)

    # ------------------------------------------------------------------
    # 播放 / 动作辅助（P0 复用）
    # ------------------------------------------------------------------
    async def _run_play(self, clip: AudioClip) -> None:
        """三路并发中的「语音路」：播放 clip 并保证 TTS_FINISHED 必发。

        单独封装成协程是为了能与动作路 / chatbox 路一起塞进 asyncio.gather；
        播放异常在此就地吞掉（记录日志），避免 gather 提前抛出中断其余两路。
        """
        try:
            await self._audio_out.play(clip)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("语音播放失败", extra={"event": "tts_failed"})

    async def _safe_exec(self, actions: list[Action]) -> None:
        """三路并发中的「动作路 / chatbox 路」：屏蔽异常，单路失败不拖累语音。"""
        if not actions:
            return
        try:
            await self._execute_actions(actions)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("动作分支执行异常", extra={"event": "action_failed"})

    async def _execute_actions(self, actions: list[Action]) -> None:
        """按优先级（大者先）顺序执行动作。

        约定：
        - 回复分支（respond=True）调用方已把 chatbox 动作拆出（chatbox 与语音、
          其余动作在 asyncio.gather 并发起跑），因此本方法单次调用要么只含 chatbox
          动作、要么只含非 chatbox 动作；
        - 仅动作不回应分支（respond=False）无语音可同步，LLM 的 chatbox 动作
          仍通过本方法照常执行；
        - 急停后（_emergency_stopped）跳过全部/剩余动作（G-03，BUG-1）。
        """
        if not actions:
            return
        if self._emergency_stopped:
            return
        timeout = self._timeout("agent.loop.action_timeout", 5.0)
        for action in sorted(actions, key=lambda a: a.priority, reverse=True):
            if self._emergency_stopped:
                logger.warning("急停：跳过剩余动作", extra={"event": "emergency_stop"})
                break
            try:
                await asyncio.wait_for(self._actuator.execute(action), timeout=timeout)
                self._bus.publish(
                    Event(
                        EventType.ACTION_EXECUTED.value,
                        {"kind": action.kind, "params": action.params, "priority": action.priority},
                    )
                )
            except Exception as exc:
                logger.error("动作执行失败 %s: %s", action.kind, exc, extra={"event": "action_failed"})

    # ------------------------------------------------------------------
    # 任务 2：回复时自动同步 VRChat 聊天框（chatbox:reply）
    # ------------------------------------------------------------------
    def _chatbox_sync_enabled(self) -> bool:
        """读取 agent.chatbox_sync 开关（异常回退默认开）。"""
        try:
            return bool(self._config.agent.chatbox_sync) if self._config else True
        except Exception:
            return True

    @staticmethod
    def _truncate_chatbox_text(text: str, max_len: int = _CHATBOX_MAX_LEN) -> str:
        """截断聊天框文本到上限（含省略号）。"""
        text = (text or "").strip()
        if len(text) <= max_len:
            return text
        return text[:max_len].rstrip() + "…"

    def _attach_chatbox_action(self, actions: list[Action], reply_text: str) -> list[Action]:
        """在主循环自动附加 chatbox:reply 动作（与 LLM 决策动作一起执行）。"""
        if not self._chatbox_sync_enabled():
            return list(actions)
        if any(a.kind == "chatbox" for a in actions):
            return list(actions)
        message = self._truncate_chatbox_text(reply_text)
        if not message:
            return list(actions)
        chatbox_action = Action(
            kind="chatbox",
            params={"name": _CHATBOX_ACTION_NAME, "message": message},
            priority=0,
        )
        logger.info("自动附加 chatbox:reply（%d 字符）", len(message))
        return list(actions) + [chatbox_action]

    # ------------------------------------------------------------------
    # 降级
    # ------------------------------------------------------------------
    async def _degrade(self) -> None:
        """云端故障降级：本地提示音（F-04），进程不崩溃。"""
        beep = bool(self._config.agent.degradation.beep) if self._config else True
        if beep:
            try:
                await self._play_beep()
            except Exception:
                pass

    async def _play_beep(self) -> None:
        """生成并播放本地提示音（880Hz 衰减正弦，0.4s）。

        A-06：播放前先 stop()——停掉在播流（若 TTS 正在播）并等待退出，
        再播提示音；避免两条流并发覆盖 AudioOutput._active_stream 导致
        打断/急停失效（原语音流继续播、PortAudio 同设备并发冲突）。
        """
        sample_rate = 16000
        t = np.linspace(0.0, 0.4, int(sample_rate * 0.4), endpoint=False)
        samples = (0.3 * np.sin(2.0 * np.pi * 880.0 * t) * np.exp(-3.0 * t)).astype(np.float32)
        try:
            await self._audio_out.stop()
        except Exception:
            logger.debug("播放提示音前停止在播流失败（忽略）")
        await self._audio_out.play(AudioClip(samples=samples, sample_rate=sample_rate))

    # ------------------------------------------------------------------
    # 状态机
    # ------------------------------------------------------------------
    def _transition(self, next_state: AgentState, **extra) -> None:
        """状态流转（发布 state_changed 事件）。急停为唯一旁路。"""
        if next_state == AgentState.EMERGENCY_STOP:
            pass  # 旁路：任意状态均可进入
        elif next_state not in _ALLOWED_TRANSITIONS.get(self._state, set()):
            logger.warning("非法状态流转: %s -> %s（忽略）", self._state.value, next_state.value)
            return
        old = self._state
        self._state = next_state
        payload = {"from": old.value, "to": next_state.value, "ts": time.time()}
        payload.update(extra)
        self._bus.publish(Event(EventType.STATE_CHANGED.value, payload))
        logger.info("状态流转: %s -> %s", old.value, next_state.value, extra={"event": "state_changed"})

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def _vision_mode(self) -> str:
        """读取视觉接入模式（reference_only / keyword_filter），异常回退默认值。"""
        try:
            return self._config.brain.vision_mode if self._config else "reference_only"
        except Exception:
            return "reference_only"

    def _interrupt_enabled(self) -> bool:
        """读取打断开关（热重载 A1：agent.interrupt.enabled 运行中可改）。"""
        if self._config is not None:
            try:
                return bool(self._config.agent.interrupt.enabled)
            except Exception:
                pass
        return self._interrupt.enabled

    def _utterance_queue_enabled(self) -> bool:
        """读取对话队列开关（热重载：agent.utterance_queue.enabled 运行中可改）。"""
        if self._config is not None:
            try:
                return bool(self._config.agent.utterance_queue.enabled)
            except Exception:
                pass
        return True

    def _utterance_queue_max_batch(self) -> int:
        """读取批量合并阈值（热重载），异常回退默认 3。"""
        if self._config is not None:
            try:
                return max(1, int(self._config.agent.utterance_queue.max_batch))
            except Exception:
                pass
        return 3

    # ---- P0 pipeline 配置读取 ----
    def _pipeline_llm_semaphore(self) -> int:
        """读取 pipeline.llm_semaphore（并发 LLM 请求数），异常回退默认 2。"""
        if self._config is not None:
            try:
                return max(1, int(self._config.agent.pipeline.llm_semaphore))
            except Exception:
                pass
        return 2

    def _pipeline_tts_semaphore(self) -> int:
        """读取 pipeline.tts_semaphore（并发 TTS 请求数），异常回退默认 2。"""
        if self._config is not None:
            try:
                return max(1, int(self._config.agent.pipeline.tts_semaphore))
            except Exception:
                pass
        return 2

    def _pipeline_output_queue_max(self) -> int:
        """读取 pipeline.output_queue_max（输出槽最大容量），异常回退默认 3。"""
        if self._config is not None:
            try:
                return max(1, int(self._config.agent.pipeline.output_queue_max))
            except Exception:
                pass
        return 3

    def _pipeline_asr_dedup_window(self) -> float:
        """读取 pipeline.asr_dedup_window_s（ASR 去重窗口秒数），异常回退默认 2.0s。"""
        if self._config is not None:
            try:
                return max(0.5, float(self._config.agent.pipeline.asr_dedup_window_s))
            except Exception:
                pass
        return 2.0

    # ---- P2-1：流水线状态发布 ----
    def _publish_pipeline_state(self) -> None:
        """发布 PIPELINE_STATE 事件：活跃 utterance 列表 + 输出队列 + 当前播放进度。"""
        utt_list = []
        for utt in list(self._active_utterances.values()):
            reply = utt.decision.reply_text if utt.decision else ""
            utt_list.append({
                "id": utt.id[:8],
                "text": (utt.text or "")[:20],
                "state": utt.state.value,
                "reply_text": (reply or "")[:20],
            })
        # 当前播放进度
        playing = None
        current = self._output_slot.current
        if current is not None and current.state == UtteranceState.SPEAKING:
            elapsed = max(0.0, time.time() - self._play_started_at) if self._play_started_at > 0 else 0.0
            remaining = max(0.0, self._play_clip_duration - elapsed)
            playing = {
                "id": current.id[:8],
                "reply_text": (current.decision.reply_text if current.decision else current.text)[:20],
                "remaining_s": round(remaining, 1),
            }
        payload = {
            "utterances": utt_list,
            "queue_size": self._output_slot.queue_size(),
            "current_playing": playing,
        }
        self._bus.publish(Event(EventType.PIPELINE_STATE.value, payload))

    # ---- P2-3：跳过当前播放 ----
    async def skip(self) -> None:
        """跳过当前播放中的 utterance，_drain_output_slot 自动取队列下一条。

        A-04：直接执行实际跳过（不等下轮检查）——立即停止音频 + 取消当前
        utterance，随后 drain 自然消费队列下一条。安全：仅在有当前播放或队列
        非空时生效；无副作用时调用直接忽略。
        """
        if self._output_slot.current is None and self._output_slot.queue_size() == 0:
            self._skip_requested = False
            return
        self._skip_requested = False
        current = self._output_slot.current
        if current is not None:
            logger.info("跳过当前播放: %s (text=%s)", current.id, current.text,
                        extra={"event": "skip_executed"})
            current.cancel()
        try:
            await self._audio_out.stop()
        except Exception:
            logger.exception("跳过播放时停止音频失败", extra={"event": "audio_output_stop_failed"})
        self._publish_pipeline_state()

    def _timeout(self, dotted_path: str, default: float) -> float:
        """从 config 读取超时值（支持热重载），异常回退默认值。"""
        node = self._config
        if node is None:
            return default
        try:
            for part in dotted_path.split("."):
                node = getattr(node, part)
            return float(node)
        except Exception:
            return default
