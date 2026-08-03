"""B 大脑：对话上下文管理（B-01）。

滚动窗口：保留最近 max_turns 轮（user + assistant 两两一组）。
"""

from __future__ import annotations

from typing import Any


class ConversationContext:
    """多轮对话上下文（滚动窗口）。"""

    def __init__(self, max_turns: int = 10) -> None:
        self._max_turns = max(1, int(max_turns))
        self._messages: list[dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        """追加一条消息（role: system/user/assistant）。"""
        self._messages.append({"role": role, "content": content})
        self._trim()

    def _trim(self) -> None:
        """裁剪到最近 max_turns 轮。"""
        max_msgs = self._max_turns * 2
        if len(self._messages) > max_msgs:
            self._messages = self._messages[-max_msgs:]

    def to_messages(self) -> list[dict[str, str]]:
        """返回上下文消息列表（拷贝，供 LLM 调用）。"""
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    @property
    def size(self) -> int:
        return len(self._messages)

    @property
    def max_turns(self) -> int:
        return self._max_turns
