"""VRChat 账号/状态/通知/收藏技能块（T03）。"""

from __future__ import annotations

from ..registry import SkillPack

PACK = SkillPack(
    name="vrc.account",
    domain="vrc",
    triggers=(
        "状态", "在线状态", "忙碌", "离开", "改状态", "通知", "收藏", "账号",
        "我的信息", "status", "notification", "favorite", "account",
    ),
    description=(
        "VRChat 账号/状态/通知/收藏查询（联网 API，需已登录；约 1 次/60 秒全局限速，"
        "一次只输出一个 vrc 动作）：\n"
        "- 更新在线状态 → {\"kind\":\"vrc\",\"params\":{\"name\":\"set_status\","
        "\"status\":\"busy\"}}（例：「设置成忙碌」，status 取 active/join me/busy/ask me/offline）\n"
        "- 查当前用户信息 → {\"kind\":\"vrc\",\"params\":{\"name\":\"get_current_user\"}}"
        "（例：「我是谁」「我的账号信息」）\n"
        "- 查通知 → {\"kind\":\"vrc\",\"params\":{\"name\":\"get_notifications\"}}"
        "（例：「有没有新通知」）\n"
        "- 查收藏 → {\"kind\":\"vrc\",\"params\":{\"name\":\"get_favorites\"}}"
        "（例：「我的收藏」）\n"
        "⚠️ set_status 是写操作，默认开放但会记录审计日志。"
    ),
    priority=23,
    max_rounds=1,
)
