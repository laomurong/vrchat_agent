"""D 动作控制：Actuator 抽象接口（D-01~D-03）。

共享知识约定（架构文档第 8 节第 7/8 条）：
- 动作映射 YAML：semantic_action → [{tool, args}]；
- 桥接失败降级为"仅语音无动作"（NoopActuator）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..brain.schema import Action


class ActuatorError(RuntimeError):
    """动作执行错误。"""


class Actuator(ABC):
    """动作执行器抽象：所有实现（MCP/自写 OSC）统一接口。"""

    @abstractmethod
    async def execute(self, action: Action) -> None:
        """执行一条语义动作。"""
        raise NotImplementedError

    @abstractmethod
    async def reset(self) -> None:
        """复位所有动作/表情（急停时调用）。"""
        raise NotImplementedError


class NoopActuator(Actuator):
    """降级模式：桥接进程不可用时，仅语音无动作（不崩溃）。"""

    async def execute(self, action: Action) -> None:
        pass

    async def reset(self) -> None:
        pass
