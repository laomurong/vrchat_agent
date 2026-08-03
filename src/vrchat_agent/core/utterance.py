"""Utterance 数据模型：ASR 产出的单条对话及其全生命周期状态（P0 流水线重构）。

每条 Utterance 从 ASR_DONE 开始，经 LLM→TTS→QUEUED→SPEAKING→DONE，
拥有独立的 cancel_event（精准取消，替代旧的全局 _interrupt 标志）。
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..brain.schema import Action, Decision
    from ..speech.tts import AudioClip


class UtteranceState(Enum):
    """Utterance 全生命周期状态。

    流转路径（正常）：
        ASR_DONE → LLM_RUNNING → LLM_DONE → TTS_RUNNING → TTS_DONE
        → QUEUED → SPEAKING → DONE

    旁路：
        任意状态 → CANCELLED（打断/急停）
        LLM_DONE → DONE（respond=False，跳过 TTS+播放）
        TTS_DONE → CANCELLED（TTS 失败降级）
    """

    ASR_DONE = "asr_done"          # ASR 产出文本，等待 LLM
    LLM_RUNNING = "llm_running"    # LLM 决策中
    LLM_DONE = "llm_done"          # LLM 返回 Decision
    TTS_RUNNING = "tts_running"    # TTS 合成中
    TTS_DONE = "tts_done"          # AudioClip 已就绪
    QUEUED = "queued"              # 在输出槽等待播放
    SPEAKING = "speaking"          # 正在播放
    DONE = "done"                  # 完成
    CANCELLED = "cancelled"        # 被打断/急停取消


@dataclass
class Utterance:
    """一条完整的对话处理单元。

    从 ASR 产出文本到播放完成的全生命周期追踪。
    cancel_event 支持独立取消（精准粒度），替代旧全局 _interrupt 标志。
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    state: UtteranceState = UtteranceState.ASR_DONE
    created_at: float = field(default_factory=time.time)
    decision: Decision | None = None
    clip: AudioClip | None = None
    actions: list[Action] = field(default_factory=list)
    chatbox_actions: list[Action] = field(default_factory=list)
    respond: bool = False
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def is_active(self) -> bool:
        """是否仍在流水线中（未完成且未取消）。"""
        return self.state not in (UtteranceState.DONE, UtteranceState.CANCELLED)

    @property
    def is_cancelled(self) -> bool:
        """是否已被取消。"""
        return self.cancel_event.is_set() or self.state == UtteranceState.CANCELLED

    def cancel(self) -> None:
        """标记取消：置 cancel_event + 状态改为 CANCELLED。"""
        self.cancel_event.set()
        self.state = UtteranceState.CANCELLED
