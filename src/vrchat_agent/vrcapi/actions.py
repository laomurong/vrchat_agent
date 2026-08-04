"""VRChat API 动作封装：换头像 / 查好友 / 找世界 / 更新状态（任务①）。

VRCActionExecutor 是动作系统 vrc:* 的执行端：
- ActionMapper 把 ``vrc:change_avatar`` 等解析为 ``vrc_*`` 工具命令；
- MCPOSCActuator 识别 ``vrc_*`` 工具 → 委托本执行器 ``run()``；
- 每个方法经 VRCAPIClient.call_on() 限速 + asyncio.to_thread 执行同步 SDK
  调用，**不阻塞事件循环**（参考 speech/tts_cosyvoice.py 模式）；
- 返回统一 dict（供上层日志/Web 展示）；SDK 异常已由 client 包装为
  VRCAPIError，本层只做参数校验与结果整形。
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .client import VRCAPIClient, VRCAPIError

logger = logging.getLogger("vrchat_agent.vrcapi.actions")

#: VRChat 状态枚举（update_status 合法值，白名单校验）
VALID_STATUSES: tuple[str, ...] = ("active", "join me", "busy", "ask me", "offline")

#: 好友/世界列表默认分页大小
DEFAULT_FRIEND_LIMIT: int = 50
DEFAULT_WORLD_LIMIT: int = 20

#: 未填充占位符模式：映射模板 `{param}` 未被 LLM 参数替换时保留字面量
#: （如 `query: "{query}"`），executor 必须识别并视为缺失，绝不泄漏给 SDK。
_PLACEHOLDER_RE: re.Pattern[str] = re.compile(r"\{[^{}]*\}")


def _to_int(value: Any, default: int, field: str) -> int:
    """把参数安全转 int（非法/空回退默认；负数回退默认，防御性）。"""
    try:
        n = int(value)
    except (TypeError, ValueError):
        logger.warning("%s 参数非法（%r），回退默认 %d", field, value, default)
        return default
    return n if n >= 0 else default


class VRCActionExecutor:
    """VRChat API 四个能力的异步封装（动作系统 vrc:* 执行端）。"""

    def __init__(self, client: VRCAPIClient | None = None) -> None:
        self._client = client or VRCAPIClient()

    @property
    def client(self) -> VRCAPIClient:
        """底层 VRCAPIClient（懒加载登录 + 限速器）。"""
        return self._client

    # ------------------------------------------------------------------
    # 四个能力
    # ------------------------------------------------------------------
    async def change_avatar(self, avatar_id: str) -> dict:
        """换头像：调用 AvatarsApi.select_avatar(avatar_id)。

        SDK 1.20.7 起方法名为 select_avatar（旧 change_avatar 已移除），
        签名 ``select_avatar(avatar_id, **kwargs)`` 兼容位置传参。
        """
        avatar_id = str(avatar_id or "").strip()
        if not avatar_id:
            raise VRCAPIError("换头像缺少 avatar_id（LLM 应输出 params.avatar_id）")
        avatar = await self._client.call_on("avatars", "select_avatar", avatar_id)
        logger.info("VRChat 换头像完成: %s", avatar_id)
        return {"ok": True, "action": "change_avatar", "avatar_id": avatar_id, "result": avatar}

    async def list_friends(self, offset: int = 0, limit: int = DEFAULT_FRIEND_LIMIT) -> dict:
        """查好友：调用 FriendsApi.get_friends（含离线好友）。"""
        offset = _to_int(offset, 0, "offset")
        limit = _to_int(limit, DEFAULT_FRIEND_LIMIT, "limit")
        friends = await self._client.call_on(
            "friends", "get_friends", offset=offset, n=limit, offline=True
        )
        logger.info("VRChat 好友列表: %d 条", len(friends) if friends else 0)
        return {"ok": True, "action": "list_friends", "offset": offset, "limit": limit, "friends": friends}

    async def search_worlds(self, query: str = "", limit: int = DEFAULT_WORLD_LIMIT) -> dict:
        """找世界：调用 WorldsApi.search_worlds（按热度排序）。"""
        query = str(query or "").strip()
        limit = _to_int(limit, DEFAULT_WORLD_LIMIT, "limit")
        if _PLACEHOLDER_RE.search(query):
            # 未填充占位符：LLM 未给 query 时 ActionMapper 保留映射模板的
            # 字面量 {query}，绝不能把它当搜索词发给 SDK（会返回 0 结果）——
            # 视为缺失，回退热门列表（search=None）。
            logger.warning("search_worlds 收到未填充占位符 %r，回退热门列表", query)
            query = ""
        worlds = await self._client.call_on(
            "worlds",
            "search_worlds",
            search=query or None,
            n=limit,
            sort="popularity",
            order="descending",
        )
        logger.info("VRChat 世界搜索: %s → %d 条", query or "(热门)", len(worlds) if worlds else 0)
        return {"ok": True, "action": "search_worlds", "query": query, "limit": limit, "worlds": worlds}

    async def update_status(self, status: str = "active") -> dict:
        """更新状态：调用 UsersApi.update_user("me", UpdateUserRequest(status=...))。

        status 白名单：active / join me / busy / ask me / offline（非法回退 active）。
        """
        raw = str(status or "").strip().lower()
        if raw not in VALID_STATUSES:
            logger.warning("update_status 状态非法（%r），回退 active（可选: %s）", status, ", ".join(VALID_STATUSES))
            raw = "active"
        await self._client.ensure_authed()
        request_cls = self._client.update_user_request_cls
        request = request_cls(status=raw)
        # SDK 1.20.7 签名 update_user(self, user_id, **kwargs)，请求体只能
        # 以关键字 update_user_request=... 传入；位置传 request 会 TypeError。
        user = await self._client.call_on("users", "update_user", "me", update_user_request=request)
        logger.info("VRChat 状态更新完成: %s", raw)
        return {"ok": True, "action": "update_status", "status": raw, "result": user}

    # ------------------------------------------------------------------
    # 工具派发（动作系统 vrc_* 工具 → 上述能力）
    # ------------------------------------------------------------------
    async def run(self, tool: str, args: dict | None = None) -> dict:
        """按动作映射工具名执行（MCPOSCActuator 对 vrc_* 工具调用本方法）。

        :param tool: 映射表工具名，如 vrc_change_avatar / vrc_list_friends /
                     vrc_search_world / vrc_set_status
        :param args: 映射表 args（经 ActionMapper 填充 {param} 占位符）
        :returns: 统一 dict；未知工具 / 参数缺失抛 VRCAPIError
        """
        args = dict(args or {})
        if tool == "vrc_change_avatar":
            return await self.change_avatar(str(args.get("avatar_id", "") or ""))
        if tool == "vrc_list_friends":
            return await self.list_friends(
                offset=_to_int(args.get("offset"), 0, "offset"),
                limit=_to_int(args.get("limit"), DEFAULT_FRIEND_LIMIT, "limit"),
            )
        if tool == "vrc_search_world":
            return await self.search_worlds(
                query=str(args.get("query", "") or ""),
                limit=_to_int(args.get("limit"), DEFAULT_WORLD_LIMIT, "limit"),
            )
        if tool == "vrc_set_status":
            return await self.update_status(str(args.get("status", "active") or "active"))
        raise VRCAPIError(f"未知 vrc 工具: {tool}（映射表 tool 应为 vrc_change_avatar / vrc_list_friends / vrc_search_world / vrc_set_status）")
