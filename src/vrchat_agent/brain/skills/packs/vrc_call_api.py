"""VRChat 通用 API 调用技能块（T03）：call_api 只读优先 + 安全策略说明。"""

from __future__ import annotations

from ..registry import SkillPack

PACK = SkillPack(
    name="vrc.call_api",
    domain="vrc",
    triggers=(
        "调用", "api", "接口", "查询接口", "call_api", "调用接口",
    ),
    description=(
        "【调用时 kind 固定为 'vrc'，本条目的动作用 params.name】\n"
        "通用 VRChat API 调用（联网 API，需已登录；约 1 次/60 秒全局限速，"
        "一次只输出一个 vrc 动作；**只读优先**）：\n"
        "- 通用调用 → {\"kind\":\"vrc\",\"params\":{\"name\":\"call_api\","
        "\"api\":\"users\",\"args\":{\"user_id\":\"usr_xxx\"}}}"
        "（例：「调用 users 接口查用户」）\n"
        "api 取 19 类句柄名：authentication/avatars/calendar/economy/favorites/files/"
        "friends/groups/instances/inventory/invite/jams/miscellaneous/notifications/"
        "playermoderation/prints/props/users/worlds；\n"
        "args 为该句柄 SDK 方法的参数（如 user_id/query/limit）。\n"
        "⚠️ 只读方法（GET）默认开放；写方法需命中白名单或开启 vrcapi.allow_destructive；"
        "危险操作（文件上传/付费/账号注册删除/群管理/内容发布/实例创建关闭/道具发布/印记上传）"
        "默认关闭不可调。"
    ),
    priority=24,
    max_rounds=1,
)
