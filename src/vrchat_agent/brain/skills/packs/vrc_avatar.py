"""VRChat 头像/用户技能块（T03）：换头像/查头像/搜用户/查资料。"""

from __future__ import annotations

from ..registry import SkillPack

PACK = SkillPack(
    name="vrc.avatar",
    domain="vrc",
    triggers=(
        "换头像", "头像", "换皮肤", "模型", "avatar", "搜用户", "搜索用户",
        "查资料", "用户信息",
    ),
    description=(
        "【调用时 kind 固定为 'vrc'，本条目的动作用 params.name】\n"
        "VRChat 头像/用户查询（联网 API，需已登录；约 1 次/60 秒全局限速，"
        "一次只输出一个 vrc 动作）：\n"
        "- 换头像 → {\"kind\":\"vrc\",\"params\":{\"name\":\"change_avatar\","
        "\"avatar_id\":\"avtr_xxx\"}}（例：「换成刚才那个头像」，avatar_id 必填）\n"
        "- 查当前使用头像 → {\"kind\":\"vrc\",\"params\":{\"name\":\"get_own_avatar\"}}"
        "（例：「我现在用的什么头像」）\n"
        "- 搜索用户 → {\"kind\":\"vrc\",\"params\":{\"name\":\"search_users\","
        "\"query\":\"xxx\"}}（例：「搜一下叫 xxx 的人」，返回 user_id 供加好友/邀请）\n"
        "- 查用户资料 → {\"kind\":\"vrc\",\"params\":{\"name\":\"get_user\","
        "\"user_id\":\"usr_xxx\"}}（例：「看看 xxx 的资料」）"
    ),
    priority=21,
    max_rounds=1,
)
