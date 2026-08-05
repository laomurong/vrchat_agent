"""空间导航技能块（T02）：nav goto / explore + 拓扑导航说明。

内容逐字搬自 prompts.py 的 _ACTION_KINDS（nav 段）。
"""

from __future__ import annotations

from ..registry import SkillPack

PACK = SkillPack(
    name="nav",
    domain="nav",
    triggers=(
        "导航", "探索", "探索一下", "到处走走", "看看那边", "去看看", "goto",
        "explore", "带我去", "怎么走", "去喷泉", "地标", "去那", "去那边",
    ),
    description=(
        "nav(空间导航(拓扑导航，VRChat OSC 无绝对坐标，只能相对转向+移动+视觉重定位)：\n"
        "goto=去往某处——params.target 填地标描述（如「去喷泉那里」→ "
        "{\"kind\":\"nav\",\"params\":{\"name\":\"goto\",\"target\":\"喷泉\"}}；\n"
        "不带 target 则返回最近已知位置)；\n"
        "explore=自主探索世界——向未访问方向移动并记录新地标，\n"
        "适合「探索一下」「到处走走」「看看那边有什么」→ "
        "{\"kind\":\"nav\",\"params\":{\"name\":\"explore\"}})"
    ),
    priority=5,
    max_rounds=3,
)
