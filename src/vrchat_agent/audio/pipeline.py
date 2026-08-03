"""A 语音输入流水线装配（装配层例外，允许直接 import 组件）。

流程：AudioInput 产出 30ms 块 → SpeechSegmenter 切分语音段 →
      ASR worker 异步转写 → 发布 utterance_ready 事件。

自抑制：静音窗口期间丢弃音频块（AudioInput.is_muted() 或 suppression 双保险）。
"""

from __future__ import annotations

import asyncio
import logging

from ..events import Event, EventType

logger = logging.getLogger("vrchat_agent.audio.pipeline")


class AudioPipeline:
    """把捕获 → VAD 切分 → ASR → 事件发布串成一条异步流水线。"""

    def __init__(
        self,
        audio_input,
        segmenter,
        asr,
        bus,
        suppression=None,
        asr_timeout: float = 30.0,
        filter=None,
    ) -> None:
        self._audio_input = audio_input
        self._segmenter = segmenter
        self._asr = asr
        self._bus = bus
        self._suppression = suppression
        self._asr_timeout = asr_timeout
        # ASR 误识别过滤（A-05）：None=不过滤（旧行为）；UtteranceFilter 实例则
        # 在转写成功、publish 前判定，命中（误识别/噪音）直接跳过。
        self._filter = filter
        self._asr_queue: asyncio.Queue = asyncio.Queue(maxsize=8)
        self._running = False

    async def run(self) -> None:
        """启动流水线（阻塞运行，直到 AudioInput.stop()）。"""
        if self._running:
            return
        self._running = True
        worker = asyncio.create_task(self._asr_worker())
        try:
            await self._audio_input.run(self._on_chunk)
        finally:
            self._running = False
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass

    async def _on_chunk(self, chunk) -> None:
        """捕获回调：自抑制 → VAD 切分 → 入 ASR 队列。"""
        if self._suppression is not None and self._suppression.should_suppress():
            return  # 静音窗口：丢弃
        segment = self._segmenter.feed(chunk)
        if segment is not None:
            try:
                self._asr_queue.put_nowait(segment)
            except asyncio.QueueFull:
                logger.warning("ASR 队列积压，丢弃语音段")

    async def _asr_worker(self) -> None:
        """ASR 消费任务：转写并发布 utterance_ready。"""
        while True:
            segment = await self._asr_queue.get()
            try:
                text = await asyncio.wait_for(self._asr.transcribe(segment), timeout=self._asr_timeout)
            except TimeoutError:
                logger.error("ASR 转写超时", extra={"event": "asr_failed"})
                continue
            except Exception as exc:
                logger.error("ASR 转写失败: %s", exc, extra={"event": "asr_failed"})
                continue
            if not text:
                continue
            # ASR 误识别过滤：命中屏蔽规则（环境音/字幕/BGM 误识别）→ 不进入 LLM 流程
            if self._filter is not None and self._filter.should_skip(text):
                logger.info("ASR 文本被过滤（误识别/噪音）: %r", text)
                continue
            logger.info("听到: %s", text, extra={"event": "utterance_ready"})
            self._bus.publish(Event(EventType.UTTERANCE_READY.value, {"text": text}))
