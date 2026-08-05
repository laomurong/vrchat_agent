"""OutputSlot：输出槽互斥调度器（P0 流水线重构）。

保证同一时刻只有一条 utterance 在播放（语音 + 动作 + chatbox 三路 gather）。
新 utterance 的 TTS 完成后入队 FIFO，消费者逐条播放。
打断/急停可精准取消当前 + 清空队列。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from .utterance import Utterance, UtteranceState

logger = logging.getLogger("vrchat_agent.loop.output_slot")

# 输出三路并发的回调签名
PlayFunc = Callable[..., Awaitable[None]]
ExecFunc = Callable[..., Awaitable[None]]


class OutputSlot:
    """输出槽：FIFO 队列 + 互斥锁，保证播放原子性。

    - enqueue(utt)：TTS 完成后入队（state → QUEUED）
    - wait_and_play(play_func, exec_func)：消费者阻塞等待，取队首 gather 三路
    - interrupt()：停当前 + 清队列（打断时清理旧 utterance）
    - clear()：停当前 + 清队列（急停用；末尾复位停止标记，急停可恢复）
    """

    def __init__(self, maxsize: int = 3) -> None:
        self._queue: asyncio.Queue[Utterance] = asyncio.Queue(maxsize=maxsize)
        self._lock = asyncio.Lock()
        self._current: Utterance | None = None
        self._stopped = False
        # A-01：输出槽可用性唤醒事件（clear() 复位 _stopped 后 set，
        # 唤醒空转等待的消费者，替代急停后的 50ms 轮询）。
        self._ready_event = asyncio.Event()
        self._ready_event.set()

    # ------------------------------------------------------------------
    # 只读属性
    # ------------------------------------------------------------------
    @property
    def current(self) -> Utterance | None:
        """当前正在播放的 utterance（无播放时返回 None）。"""
        return self._current

    @property
    def is_playing(self) -> bool:
        """输出槽是否正在播放。"""
        return self._current is not None

    def queue_size(self) -> int:
        """等待播放的 utterance 数量。"""
        return self._queue.qsize()

    # ------------------------------------------------------------------
    # 入队
    # ------------------------------------------------------------------
    async def enqueue(self, utt: Utterance) -> None:
        """TTS 完成后入队，等待播放。

        P1-2：队列满时丢弃最旧 utterance（get_nowait + cancel），为新 utterance 腾出空间。
        """
        if self._stopped:
            utt.state = UtteranceState.CANCELLED
            logger.debug("输出槽已停止，丢弃 utterance: %s", utt.id)
            return
        # P1-2：输出队列积压保护 - 满则丢弃最旧
        maxsize = self._queue.maxsize
        if self._queue.qsize() >= maxsize:
            try:
                oldest = self._queue.get_nowait()
                oldest.cancel()
                self._queue.task_done()
                logger.warning(
                    "输出槽队列已满 (%d/%d)，丢弃最旧: %s (text=%s)",
                    self._queue.qsize(), maxsize, oldest.id, oldest.text,
                    extra={"event": "output_queue_full"},
                )
            except asyncio.QueueEmpty:
                pass
        utt.state = UtteranceState.QUEUED
        await self._queue.put(utt)
        logger.debug("utterance 入队: %s (队列 %d)", utt.id, self._queue.qsize())

    # ------------------------------------------------------------------
    # 消费（播放）
    # ------------------------------------------------------------------
    async def wait_and_play(
        self,
        play_func: PlayFunc,
        exec_func: ExecFunc,
        on_start: Callable[[Utterance], Awaitable[None]] | None = None,
    ) -> Utterance | None:
        """阻塞等待队列中的下一条 utterance 并播放。

        内部流程：
        1. 从队列取一条（阻塞直到有数据或 stopped）
        2. 获取 _lock（互斥：同一时刻只有一个 utterance 在播）
        3. 调用 on_start(utt)（AgentLoop 在此做 SPEAKING transition + TTS_STARTED 事件）
        4. asyncio.gather(play(clip), exec(other_actions), exec(chatbox_actions))

        Returns:
            播放完成的 utterance；若输出槽已停止返回 None。
        """
        if self._stopped:
            return None
        try:
            utt = await self._queue.get()
        except asyncio.CancelledError:
            return None

        async with self._lock:
            if self._stopped:
                utt.state = UtteranceState.CANCELLED
                return None
            if utt.is_cancelled:
                logger.debug("utterance 已被取消，跳过播放: %s", utt.id)
                return utt
            self._current = utt
            utt.state = UtteranceState.SPEAKING
            logger.info("开始播放: %s", utt.id)
            # 通知 AgentLoop：SPEAKING transition + TTS_STARTED 事件
            if on_start is not None:
                try:
                    await on_start(utt)
                except Exception:
                    logger.exception("on_start 回调异常: %s", utt.id)
            try:
                # 三路并发起跑：语音播放 / 非 chatbox 动作 / chatbox 文本
                # 任一路失败都不拖累其余两路（play_func / exec_func 各自吞异常）
                await asyncio.gather(
                    play_func(utt.clip),
                    exec_func(utt.actions),
                    exec_func(utt.chatbox_actions),
                )
            except asyncio.CancelledError:
                # P1-2：播放中被取消（打断/急停/任务取消）。
                # 修复前 re-raise 会把 CancelledError 击穿到 _drain_output_slot，
                # 使输出队列消费循环永久停摆（打断后 Agent 失声）。
                # 修复后：标记 utterance 已取消并按"播放完成"返回——由调用方
                # （drain）走收尾逻辑（A-02 补发 TTS_FINISHED 解除静音窗口），
                # 消费循环自然继续；若外层任务确实在停机取消，drain 的
                # ``while self._running`` 会在下一轮退出。
                logger.debug("播放被取消: %s", utt.id)
                utt.state = UtteranceState.CANCELLED
                return utt
            except Exception:
                logger.exception("播放 gather 异常: %s", utt.id)
            finally:
                self._current = None
            logger.info("播放完成: %s", utt.id)
            return utt

    # ------------------------------------------------------------------
    # 打断
    # ------------------------------------------------------------------
    async def interrupt(self) -> None:
        """打断：取消当前播放中的 utterance + 清空等待队列。

        信号式取消（不获取 _lock，避免与播放中的 wait_and_play 死锁）：
        1. 设置 _current.cancel_event（wait_and_play 的 play_func 可感知）
        2. 置 _current = None
        3. 清空队列（逐条 cancel）

        调用方需额外调用 audio_out.stop() 停止硬件播放。
        """
        if self._current is not None:
            logger.info("打断当前播放: %s", self._current.id)
            self._current.cancel()
        self._current = None

        cancelled_count = 0
        while not self._queue.empty():
            try:
                old = self._queue.get_nowait()
                old.cancel()
                cancelled_count += 1
            except asyncio.QueueEmpty:
                break
        if cancelled_count:
            logger.info("打断清空队列: %d 条", cancelled_count)

    # ------------------------------------------------------------------
    # 急停
    # ------------------------------------------------------------------
    async def clear(self) -> None:
        """急停：取消当前播放 + 清空队列；末尾复位停止标记（急停可恢复，A-01）。

        复位语义（A-01）：急停后输出槽立即可复用——后续 enqueue/wait_and_play
        恢复正常，避免"按一次急停 Agent 永久失声"。复位时 set 唤醒事件，
        让空转等待的 _drain_output_slot 立即恢复消费。
        """
        self._stopped = True
        if self._current is not None:
            logger.info("急停取消当前播放: %s", self._current.id)
            self._current.cancel()
        self._current = None

        cancelled_count = 0
        while not self._queue.empty():
            try:
                old = self._queue.get_nowait()
                old.cancel()
                cancelled_count += 1
            except asyncio.QueueEmpty:
                break
        if cancelled_count:
            logger.info("急停清空队列: %d 条", cancelled_count)

        # A-01：急停可恢复 —— 复位停止标记并唤醒等待中的消费者
        self._stopped = False
        self._ready_event.set()

    async def wait_until_ready(self) -> None:
        """阻塞直到输出槽可再次消费（clear() 复位后返回）。

        A-01：替代急停后的 50ms 轮询——clear() 复位 _stopped 时 set 事件唤醒。
        正常未停止时立即返回（事件初始为 set）。
        """
        while self._stopped:
            await self._ready_event.wait()
            self._ready_event.clear()
