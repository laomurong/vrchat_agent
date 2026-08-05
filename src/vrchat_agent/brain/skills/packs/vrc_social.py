"""VRChat 社交技能块（T03）：查好友/加好友/接受申请/邀请（默认开放 + 审计）。"""

from __future__ import annotations

from ..registry import SkillPack

PACK = SkillPack(
    name="vrc.social",
    domain="vrc",
    triggers=(
        "好友", "朋友", "加好友", "添加好友", "邀请", "查好友", "好友列表",
        "好友申请", "friend", "invite", "加为好友",
    ),
    description=(
        "VRChat 社交操作（联网 API，需已登录；真实网络请求，约 1 次/60 秒全局限速，"
        "一次只输出一个 vrc 动作）：\n"
        "- 查好友列表 → {\"kind\":\"vrc\",\"params\":{\"name\":\"list_friends\"}}（例：「帮我查好友」）\n"
        "- 加好友 → {\"kind\":\"vrc\",\"params\":{\"name\":\"friend\",\"user_id\":\"usr_xxx\"}}"
        "（例：「把 xxx 加为好友」，user_id 可从用户搜索/好友列表获得）\n"
        "- 接受好友申请 → {\"kind\":\"vrc\",\"params\":{\"name\":\"accept_friend_request\","
        "\"notification_id\":\"ntf_xxx\"}}（例：「接受好友申请」）\n"
        "- 邀请玩家到当前世界 → {\"kind\":\"vrc\",\"params\":{\"name\":\"invite_user\","
        "\"user_id\":\"usr_xxx\"}}（例：「邀请 xxx 来我这里」）\n"
        "⚠️ 社交写操作（friend/accept_friend_request/invite_user）默认开放但会记录审计日志；"
        "只读操作（list_friends）随时可用。"
    ),
    priority=20,
    max_rounds=1,
)
