"""VRChat 世界/实例技能块（T03）：搜世界/最近世界/当前实例。"""

from __future__ import annotations

from ..registry import SkillPack

PACK = SkillPack(
    name="vrc.world",
    domain="vrc",
    triggers=(
        "找世界", "搜索世界", "世界", "房间", "找房间", "搜一下", "world",
        "实例", "instance",
    ),
    description=(
        "【调用时 kind 固定为 'vrc'，本条目的动作用 params.name】\n"
        "VRChat 世界/房间/实例查询（联网 API，需已登录；约 1 次/60 秒全局限速，"
        "一次只输出一个 vrc 动作）：\n"
        "- 搜索世界 → {\"kind\":\"vrc\",\"params\":{\"name\":\"search_world\","
        "\"query\":\"japan\"}}（例：「搜一下日本刀相关世界」）\n"
        "- 查最近去过的世界 → {\"kind\":\"vrc\",\"params\":{\"name\":\"get_recent_worlds\"}}"
        "（例：「我最近去过哪些世界」）\n"
        "- 查当前世界实例信息 → {\"kind\":\"vrc\",\"params\":{\"name\":\"get_instance\","
        "\"world_id\":\"wrld_xxx\",\"instance_id\":\"xxx\"}}（例：「我现在在哪个实例」）"
    ),
    priority=22,
    max_rounds=1,
)
