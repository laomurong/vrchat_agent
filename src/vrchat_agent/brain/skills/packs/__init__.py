"""内置技能块注册（T02/T03）：Python 注册入口（YAML 为兜底，二者收敛于 SkillPack）。

- 历史功能拆分（T02）：osc_gesture / osc_input / osc_look / nav / core_extra
  ——把 prompts.py 现有动作枚举/规则逐字搬入技能块，供 on_demand 模式按需注入；
- 新增 vrc 技能块（T03）：vrc.social / vrc.avatar / vrc.world / vrc.account /
  vrc.call_api ——vrchatapi 全量接入的精选动作 + 例句 + 安全说明；
- ``default_registry()``：装配完整内置注册表（每次返回新实例，避免跨 Brain
  共享 sticky 会话状态）。
"""

from __future__ import annotations

from ..registry import SkillRegistry
from . import (
    core_extra,
    nav,
    osc_gesture,
    osc_input,
    osc_look,
    vrc_account,
    vrc_avatar,
    vrc_call_api,
    vrc_social,
    vrc_world,
)

__all__ = [
    "core_extra",
    "default_registry",
    "nav",
    "osc_gesture",
    "osc_input",
    "osc_look",
    "vrc_account",
    "vrc_avatar",
    "vrc_call_api",
    "vrc_social",
    "vrc_world",
]

#: 内置技能块清单（default_registry 注册顺序）
BUILTIN_PACKS = (
    core_extra.PACK,
    osc_gesture.PACK,
    osc_input.PACK,
    osc_look.PACK,
    nav.PACK,
    vrc_social.PACK,
    vrc_avatar.PACK,
    vrc_world.PACK,
    vrc_account.PACK,
    vrc_call_api.PACK,
)


def default_registry(
    max_active: int = 3,
    max_chars: int = 1200,
    sticky_turns: int = 3,
) -> SkillRegistry:
    """构造默认技能注册表（内置块；每次新实例，不共享 sticky 状态）。"""
    return SkillRegistry(
        max_active=max_active,
        max_chars=max_chars,
        sticky_turns=sticky_turns,
    ).register_many(BUILTIN_PACKS)
