"""D 动作控制：OSCCommand / ResetSpec 数据模型（工具调用级）。

ResetSpec（问题 1：动作自动复位）：
- 持续型动作（emote/gesture 写入 VRChat 参数后不会自动复原）在映射表条目里
  配置 `reset: {delay: ..., tool: ..., args: ...}`（或 tools 列表）；
- 执行器在动作执行成功后延迟 delay 秒自动调用复位工具，实现"动作自动停/复位"。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class OSCCommand(BaseModel):
    """一条 OSC 工具调用：tool 为 vrchat-mcp-osc 暴露的工具名，args 为参数。"""

    tool: str
    args: dict = Field(default_factory=dict)


class ResetSpec(BaseModel):
    """动作自动复位规格（映射条目 reset 字段，问题 1）。

    - delay：执行后延迟多少秒复位；连续触发同一动作时复位计时重置；
    - tools：复位要调用的 OSC 工具列表（如 emote → set_emote_parameter{value:0}，
      gesture → set_avatar_parameter{parameter_name:..., value:false}）。
    """

    delay: float = 3.0
    tools: list[OSCCommand] = Field(default_factory=list)
