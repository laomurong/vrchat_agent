"""手势技能块（T02）：标准手势 16 种 + avatar 自定义 3 种 + 手势规则。

内容逐字搬自 prompts.py 的 _ACTION_KINDS 与 build_decision_instruction 规则。
"""

from __future__ import annotations

from ..registry import SkillPack

PACK = SkillPack(
    name="osc_gesture",
    domain="osc",
    triggers=(
        "手势", "比个耶", "比耶", "耶", "大拇指", "点赞", "摇滚", "比心",
        "peace", "rock", "fist", "thumb", "手势", "指", "挥手",
    ),
    description=(
        "gesture(手势: 两类——\n"
        "①avatar 自定义: point/thumbs_up/wave_hand；\n"
        "②标准手势(所有 avatar 通用，左右手各 8 种): \n"
        "left_neutral/left_fist/left_open/left_point/left_peace/left_rock/left_gun/left_thumb，\n"
        "right_neutral/right_fist/right_open/right_point/right_peace/right_rock/right_gun/right_thumb；\n"
        "如「比个耶」→ {\"kind\":\"gesture\",\"params\":{\"name\":\"right_peace\"}}、\n"
        "「竖大拇指」→ right_thumb；手势会保持 3 秒后自动放松)\n"
        "【手势规则】\n"
        "- 手势优先用标准手势（gesture:right_peace 等，所有 avatar 都支持）；\n"
        "  avatar 自定义手势（point/thumbs_up/wave_hand）只在明确需要时用。"
    ),
    priority=2,
    max_rounds=1,
)
