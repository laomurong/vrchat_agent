"""对话队列 + 批量合并（需求 4），P0 流水线重构后降级为"辅助兼容队列"。

P0 变更：
- 主路径改为 Utterance + OutputSlot + Semaphore 流水线并发（agent_loop.py）。
- 本模块保留用于 semaphore=1 退化串行 + 旧测试兼容。
- push/pop_batch/clear 接口不变，语义为"并行流水线满载时的溢出缓冲队列"。

原始背景：AgentLoop 的 `_on_utterance` 用 `_processing_lock` 串行处理——模型忙时新
utterance 会在锁上隐式排队（无显式队列、无批量合并），连续多人说话会排队很久。

接口：
- `push(text)`：processing 时入队（返回 True）；空闲时不缓存（返回 False）；
- `pop_batch(max_batch)`：队列待处理条数 ≥ max_batch 时合并为一条；否则取单条；
- `clear()`：急停 / 关闭时清空（丢弃未处理对话）。

线程模型：仅事件循环单线程访问（AgentLoop 内部），无需加锁。
"""

from __future__ import annotations

import logging

logger = logging.getLogger("vrchat_agent.loop.queue")


class UtteranceQueue:
    """FIFO 对话队列：模型忙时排队，≥max_batch 条自动合并批量处理。"""

    def __init__(self, max_batch: int = 3) -> None:
        self.pending: list[str] = []
        self.processing: bool = False
        self._max_batch = max(1, int(max_batch))

    # ---- 入队 ----
    def push(self, text: str) -> bool:
        """提交一条对话：处理中 → 入队并返回 True；空闲 → 不缓存返回 False。

        返回 False 时调用方应把 text 直接交给处理流程（不经过队列），
        避免"已入队 + 直接处理"双路径导致同一条被处理两次。
        """
        text = (text or "").strip()
        if not text:
            return False
        if self.processing:
            self.pending.append(text)
            return True
        return False

    # ---- 出队 ----
    def pop_batch(self, max_batch: int | None = None) -> tuple[str, int] | None:
        """取下一批待处理对话。

        返回 (text, count)：
        - 队列待处理 ≥ batch → 取前 batch 条合并为一条（count>1）；
        - 队列 < batch → 取队首单条（count=1）；
        - 队列为空 → None。
        """
        if not self.pending:
            return None
        batch = self._max_batch if max_batch is None else max(1, int(max_batch))
        if len(self.pending) >= batch:
            items = self.pending[:batch]
            del self.pending[:batch]
            return self.merge(items), len(items)
        text = self.pending.pop(0)
        return text, 1

    @staticmethod
    def merge(items: list[str]) -> str:
        """把多条对话拼接为一条（合并格式说明：序号分隔，LLM 更易识别多条并综合回应）。"""
        clean = [str(i).strip() for i in (items or []) if str(i).strip()]
        if not clean:
            return ""
        if len(clean) == 1:
            return clean[0]
        return "（连续对话）" + " ".join(f"{idx}. {t}" for idx, t in enumerate(clean, start=1))

    # ---- 清空 / 状态 ----
    def clear(self) -> None:
        """清空待处理队列（急停 / 关闭时丢弃未处理对话）。"""
        self.pending.clear()

    def __len__(self) -> int:
        return len(self.pending)

    @property
    def pending_count(self) -> int:
        """待处理条数。"""
        return len(self.pending)

    @property
    def max_batch(self) -> int:
        """合并阈值。"""
        return self._max_batch
