"""A 语音输入流水线装配（装配层例外，允许直接 import 组件）。

流程：AudioInput 产出 30ms 块 → SpeechSegmenter 切分语音段 →
      ASR worker 异步转写 → 发布 utterance_ready 事件。

自抑制：静音窗口期间丢弃音频块（AudioInput.is_muted() 或 suppression 双保险）。
"""

from __future__ import annotations

import asyncio
import logging
import time

from ..events import Event, EventType

logger = logging.getLogger("vrchat_agent.audio.pipeline")

# A-10：ASR 连续失败阈值——达到后进入"ASR 不可用"降级态（停止重试、日志降频），
# 避免无限重试刷屏；_ASR_RETRY_COOLDOWN_S 后自动重置计数重探一次。
_ASR_MAX_CONSECUTIVE_FAILURES = 5
_ASR_RETRY_COOLDOWN_S = 30.0


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
        """启动流水线（阻塞运行，直到 AudioInput.stop()）。

        A-11：AudioInput 打开失败（麦克风被占用/不存在）时异常在 run() 内部消化，
        记 WARNING 后降级为"半聋模式"返回（TTS 仍可用），不冒泡崩溃整个进程——
        与视觉/桥接的 try 降级语义一致。
        """
        if self._running:
            return
        self._running = True
        worker = asyncio.create_task(self._asr_worker())
        try:
            try:
                await self._audio_input.run(self._on_chunk)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "音频输入不可用，进入半聋模式（TTS 仍可用）: %s", exc,
                    extra={"event": "audio_input_failed"},
                )
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
        """ASR 消费任务：转写并发布 utterance_ready。

        A-10：记录连续失败次数，达到阈值（5 次）进入"ASR 不可用"降级态——
        停止调用 ASR、日志降频为一条 WARNING；30s 后自动重置计数重探，
        或收到新语音段时再试一次（重探后失败会重新累计）。
        """
        consecutive_failures = 0
        next_probe_at = 0.0
        degraded_logged = False
        while True:
            segment = await self._asr_queue.get()
            now = time.monotonic()
            # 降级态：连续失败达阈值且未到重探时间 → 丢弃该段，不调 ASR
            if consecutive_failures >= _ASR_MAX_CONSECUTIVE_FAILURES:
                if now < next_probe_at:
                    if not degraded_logged:
                        logger.warning(
                            "ASR 连续失败 %d 次，进入不可用降级态（%.0fs 后自动重探）",
                            consecutive_failures, _ASR_RETRY_COOLDOWN_S,
                            extra={"event": "asr_unavailable"},
                        )
                        degraded_logged = True
                    continue
                # 到重探时间：重置计数再试一次
                consecutive_failures = 0
                degraded_logged = False
            try:
                text = await asyncio.wait_for(self._asr.transcribe(segment), timeout=self._asr_timeout)
            except TimeoutError:
                consecutive_failures += 1
                next_probe_at = time.monotonic() + _ASR_RETRY_COOLDOWN_S
                logger.error(
                    "ASR 转写超时（连续 %d 次）", consecutive_failures,
                    extra={"event": "asr_failed"},
                )
                continue
            except Exception as exc:
                consecutive_failures += 1
                next_probe_at = time.monotonic() + _ASR_RETRY_COOLDOWN_S
                logger.error(
                    "ASR 转写失败: %s（连续 %d 次）", exc, consecutive_failures,
                    extra={"event": "asr_failed"},
                )
                continue
            # 转写成功：重置失败计数与重探时间
            consecutive_failures = 0
            next_probe_at = 0.0
            if not text:
                continue
            # ASR 误识别过滤：命中屏蔽规则（环境音/字幕/BGM 误识别）→ 不进入 LLM 流程
            if self._filter is not None and self._filter.should_skip(text):
                logger.info("ASR 文本被过滤（误识别/噪音）: %r", text)
                continue
            logger.info("听到: %s", text, extra={"event": "utterance_ready"})
            self._bus.publish(Event(EventType.UTTERANCE_READY.value, {"text": text}))
