"""Skill 式按需加载 Prompt（T01）：技能包 + 注册表 + 常驻核心。

对外导出：
- ``SkillPack`` / ``SkillRegistry``：技能块模型与注册表；
- ``build_system_prompt_resident`` / ``build_decision_instruction_minimal`` /
  ``build_decision_instruction_full``：常驻核心 prompt（精简/全量双模式）。
"""

from __future__ import annotations

from .core import (
    MINIMAL_ACTION_KINDS,
    MINIMAL_RULES,
    build_decision_instruction_full,
    build_decision_instruction_minimal,
    build_system_prompt_resident,
)
from .registry import SkillPack, SkillRegistry

__all__ = [
    "MINIMAL_ACTION_KINDS",
    "MINIMAL_RULES",
    "SkillPack",
    "SkillRegistry",
    "build_decision_instruction_full",
    "build_decision_instruction_minimal",
    "build_system_prompt_resident",
]
