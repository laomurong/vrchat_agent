"""VRChat API 动作封装：精选 16 动作 + call_api 通用层 + 审计日志（任务①+T03）。

VRCActionExecutor 是动作系统 vrc:* 的执行端：
- ActionMapper 把 ``vrc:change_avatar`` 等解析为 ``vrc_*`` 工具命令；
- MCPOSCActuator 识别 ``vrc_*`` 工具 → 委托本执行器 ``run()``；
- 每个方法经 VRCAPIClient.call_on() 限速 + asyncio.to_thread 执行同步 SDK
  调用，**不阻塞事件循环**（参考 speech/tts_cosyvoice.py 模式）；
- 返回统一 dict（供上层日志/Web 展示）；SDK 异常已由 client 包装为
  VRCAPIError，本层只做参数校验、安全策略与结果整形。

call_api 安全（用户已确认决策 3）：
- 只读优先（GET 默认开放）+ 写操作白名单 + ``vrcapi.allow_destructive`` 开关
  （默认 false）+ 危险操作（closed）关闭；
- 社交写操作（friend/accept_friend_request/invite_user/join_group/leave_group 等
  白名单写操作）默认开放但**必须记审计日志**（logger.info 含 user_id/action/时间）。
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any

from .api_catalog import APICatalog, CatalogEntry, SecurityPolicy
from .client import VRCAPIClient, VRCAPIError

logger = logging.getLogger("vrchat_agent.vrcapi.actions")

#: VRChat 状态枚举（update_status 合法值，白名单校验）
VALID_STATUSES: tuple[str, ...] = ("active", "join me", "busy", "ask me", "offline")

#: 好友/世界列表默认分页大小
DEFAULT_FRIEND_LIMIT: int = 50
DEFAULT_WORLD_LIMIT: int = 20
DEFAULT_LIST_LIMIT: int = 20

#: 通用 call_api 结果序列化时列表截断条数
TRUNCATE_MAX: int = 20

#: 未填充占位符模式：映射模板 `{param}` 未被 LLM 参数替换时保留字面量
#: （如 `query: "{query}"`），executor 必须识别并视为缺失，绝不泄漏给 SDK。
_PLACEHOLDER_RE: re.Pattern[str] = re.compile(r"\{[^{}]*\}")

#: 标准 vrc_* 工具名全集（run() 派发白名单；未知工具返回清晰错误而非抛异常，
#: BugFix：让 LLM 下一轮能纠正，不因单个未知动作导致整轮失败）。
_VRC_TOOLS: frozenset[str] = frozenset(
    {
        "vrc_change_avatar",
        "vrc_list_friends",
        "vrc_search_world",
        "vrc_set_status",
        "vrc_get_current_user",
        "vrc_get_own_avatar",
        "vrc_search_users",
        "vrc_get_user",
        "vrc_friend",
        "vrc_accept_friend_request",
        "vrc_invite_user",
        "vrc_get_notifications",
        "vrc_get_favorites",
        "vrc_get_recent_worlds",
        "vrc_get_instance",
        "vrc_call_api",
    }
)

#: 工具/动作名别名 → 标准工具（用户实测 LLM 爱写 friends_online/friends 等）。
#: 兼容三种形态：标准工具名 / 别名 / 裸精选动作名（list_friends）。
_VRC_TOOL_ALIASES: dict[str, str] = {
    "friends_online": "vrc_list_friends",
    "friends": "vrc_list_friends",
    "list_friends": "vrc_list_friends",
    "list_friend": "vrc_list_friends",
    "friend_list": "vrc_list_friends",
    "online_friends": "vrc_list_friends",
    "search_worlds": "vrc_search_world",
    "search_world": "vrc_search_world",
    "change_status": "vrc_set_status",
    "set_status": "vrc_set_status",
    "vrc_friends_online": "vrc_list_friends",
    "vrc_friends": "vrc_list_friends",
    "vrc_list_friend": "vrc_list_friends",
    "vrc_friend_list": "vrc_list_friends",
    "vrc_online_friends": "vrc_list_friends",
    "vrc_search_worlds": "vrc_search_world",
    "vrc_change_status": "vrc_set_status",
}

#: 错误提示用的可用工具名列表（精选动作 + call_api，供 LLM 下一轮纠正）。
_VRC_TOOLS_TEXT: str = (
    "vrc_change_avatar / vrc_list_friends / vrc_search_world / vrc_set_status / "
    "vrc_get_current_user / vrc_get_own_avatar / vrc_search_users / vrc_get_user / "
    "vrc_friend / vrc_accept_friend_request / vrc_invite_user / vrc_get_notifications / "
    "vrc_get_favorites / vrc_get_recent_worlds / vrc_get_instance / vrc_call_api"
)


def _resolve_vrc_tool(tool: str) -> str | None:
    """把工具名/动作名归一化为标准 ``vrc_*`` 工具名（未知返回 None）。

    兼容三种形态：
    1) 标准工具名（vrc_list_friends）→ 原样；
    2) 别名（vrc_friends_online / friends_online / friends）→ 归一化；
    3) 裸精选动作名（list_friends）→ 补 ``vrc_`` 前缀。
    """
    t = (tool or "").strip()
    if t in _VRC_TOOLS:
        return t
    if t in _VRC_TOOL_ALIASES:
        return _VRC_TOOL_ALIASES[t]
    if t and not t.startswith("vrc_"):
        cand = "vrc_" + t
        if cand in _VRC_TOOLS:
            return cand
    return None


def _to_int(value: Any, default: int, field: str) -> int:
    """把参数安全转 int（非法/空回退默认；负数回退默认，防御性）。"""
    try:
        n = int(value)
    except (TypeError, ValueError):
        logger.warning("%s 参数非法（%r），回退默认 %d", field, value, default)
        return default
    return n if n >= 0 else default


def _coerce_args_dict(value: Any) -> dict:
    """把 call_api 的 args 参数安全转 dict。

    映射表模板 ``args: "{args}"`` 会把 LLM 的 dict 参数转成字符串表示
    （ActionMapper 只做字符串替换），此处恢复为 dict；解析失败返回 {}。
    """
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        s = value.strip()
        if s.startswith(("{", "[")):
            try:
                import ast

                parsed = ast.literal_eval(s)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:  # noqa: BLE001, S110 - 宽松解析兜底，失败继续尝试 json
                pass
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:  # noqa: BLE001, S110 - 宽松解析兜底，失败返回 {}
                pass
    return {}


def _to_plain(value: Any, depth: int = 0) -> Any:
    """把 SDK 返回值序列化为纯 Python 结构（JSON 友好，供日志/Web）。

    - pydantic/模型对象 → to_dict() 递归；
    - list/tuple → 截断 TRUNCATE_MAX 条；
    - datetime/date → ISO 8601 字符串；
    - 其余基础类型原样；未知类型 str() 兜底。
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _to_plain(value.to_dict(), depth + 1)
        except Exception:  # noqa: BLE001 - 模型转 dict 失败降级为 str
            return str(value)
    if isinstance(value, dict):
        return {str(k): _to_plain(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        items = list(value)[:TRUNCATE_MAX]
        return [_to_plain(v, depth + 1) for v in items]
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


class VRCActionExecutor:
    """VRChat API 精选 16 动作 + call_api 通用层的异步封装（vrc:* 执行端）。"""

    def __init__(
        self,
        client: VRCAPIClient | None = None,
        catalog: APICatalog | None = None,
        allow_destructive: bool = False,
    ) -> None:
        self._client = client or VRCAPIClient()
        self._catalog = catalog or APICatalog()
        self._policy = SecurityPolicy(allow_destructive=allow_destructive)

    @property
    def client(self) -> VRCAPIClient:
        """底层 VRCAPIClient（懒加载登录 + 限速器）。"""
        return self._client

    @property
    def catalog(self) -> APICatalog:
        """274 方法目录（T03）。"""
        return self._catalog

    @property
    def policy(self) -> SecurityPolicy:
        """安全判定器（只读开放 / 白名单审计 / allow_destructive / closed）。"""
        return self._policy

    # ------------------------------------------------------------------
    # 审计日志（社交写操作默认开放 + 审计，用户已确认决策 1）
    # ------------------------------------------------------------------
    def _audit(self, entry: CatalogEntry, args: dict | None = None) -> None:
        """记录写操作审计日志（含 user_id/action/时间）。"""
        args = dict(args or {})
        target = (
            str(args.get("user_id") or args.get("notification_id") or args.get("avatar_id") or "?")
        )
        logger.info(
            "VRChat API 审计: action=%s.%s target=%s args=%s time=%s",
            entry.api,
            entry.method,
            target,
            json.dumps(args, ensure_ascii=False, default=str)[:300],
            datetime.now().astimezone().isoformat(timespec="seconds"),
            extra={
                "event": "vrcapi_audit",
                "vrc_action": f"{entry.api}.{entry.method}",
                "vrc_target": target,
            },
        )

    def _audit_call(self, api: str, method: str, args: dict | None = None) -> None:
        """按目录条目记录写操作审计日志（目录缺失静默跳过，不阻断调用）。

        精选动作与 call_api 走同一方法，保证同一写操作审计一致
        （如 change_avatar→select_avatar、update_status→update_user）。
        """
        entry = self._catalog.lookup(api, method)
        if entry is not None:
            self._audit(entry, args)

    # ------------------------------------------------------------------
    # 精选动作（16 个）
    # ------------------------------------------------------------------
    async def change_avatar(self, avatar_id: str) -> dict:
        """换头像：调用 AvatarsApi.select_avatar(avatar_id)。

        SDK 1.20.7 起方法名为 select_avatar（旧 change_avatar 已移除），
        签名 ``select_avatar(avatar_id, **kwargs)`` 兼容位置传参。
        写操作（select_avatar 在 AUDIT_WRITE_ACTIONS）→ 默认开放 + 审计。
        """
        avatar_id = str(avatar_id or "").strip()
        if not avatar_id or _PLACEHOLDER_RE.search(avatar_id):
            raise VRCAPIError("换头像缺少 avatar_id（LLM 应输出 params.avatar_id）")
        self._audit_call("avatars", "select_avatar", {"avatar_id": avatar_id})
        avatar = await self._client.call_on("avatars", "select_avatar", avatar_id)
        logger.info("VRChat 换头像完成: %s", avatar_id)
        return {"ok": True, "action": "change_avatar", "avatar_id": avatar_id, "result": _to_plain(avatar)}

    async def list_friends(self, offset: int = 0, limit: int = DEFAULT_FRIEND_LIMIT) -> dict:
        """查好友：调用 FriendsApi.get_friends（含离线好友）。"""
        offset = _to_int(offset, 0, "offset")
        limit = _to_int(limit, DEFAULT_FRIEND_LIMIT, "limit")
        friends = await self._client.call_on(
            "friends", "get_friends", offset=offset, n=limit, offline=True
        )
        logger.info("VRChat 好友列表: %d 条", len(friends) if friends else 0)
        return {"ok": True, "action": "list_friends", "offset": offset, "limit": limit, "friends": _to_plain(friends)}

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
        return {"ok": True, "action": "search_worlds", "query": query, "limit": limit, "worlds": _to_plain(worlds)}

    async def update_status(self, status: str = "active") -> dict:
        """更新状态：调用 UsersApi.update_user("me", UpdateUserRequest(status=...))。

        status 白名单：active / join me / busy / ask me / offline（非法回退 active）。
        写操作（update_user 在 AUDIT_WRITE_ACTIONS）→ 默认开放 + 审计。
        """
        raw = str(status or "").strip().lower()
        if raw not in VALID_STATUSES:
            logger.warning("update_status 状态非法（%r），回退 active（可选: %s）", status, ", ".join(VALID_STATUSES))
            raw = "active"
        self._audit_call("users", "update_user", {"user_id": "me", "status": raw})
        await self._client.ensure_authed()
        request_cls = self._client.update_user_request_cls
        request = request_cls(status=raw)
        # SDK 1.20.7 签名 update_user(self, user_id, **kwargs)，请求体只能
        # 以关键字 update_user_request=... 传入；位置传 request 会 TypeError。
        user = await self._client.call_on("users", "update_user", "me", update_user_request=request)
        logger.info("VRChat 状态更新完成: %s", raw)
        return {"ok": True, "action": "update_status", "status": raw, "result": _to_plain(user)}

    async def get_current_user(self) -> dict:
        """查当前用户：AuthenticationApi.get_current_user()。"""
        user = await self._client.call_on("authentication", "get_current_user")
        return {"ok": True, "action": "get_current_user", "result": _to_plain(user)}

    async def get_own_avatar(self) -> dict:
        """查当前使用头像：UsersApi.get_user("me")（CurrentUser.currentAvatar）。"""
        user = await self._client.call_on("users", "get_user", "me")
        return {"ok": True, "action": "get_own_avatar", "result": _to_plain(user)}

    async def search_users(self, query: str = "", limit: int = DEFAULT_LIST_LIMIT) -> dict:
        """搜索用户：UsersApi.search_users(search, n)。"""
        query = str(query or "").strip()
        limit = _to_int(limit, DEFAULT_LIST_LIMIT, "limit")
        if _PLACEHOLDER_RE.search(query):
            logger.warning("search_users 收到未填充占位符 %r，视为空查询", query)
            query = ""
        users = await self._client.call_on(
            "users", "search_users", search=query or None, n=limit
        )
        logger.info("VRChat 用户搜索: %s → %d 条", query or "(全部)", len(users) if users else 0)
        return {"ok": True, "action": "search_users", "query": query, "limit": limit, "users": _to_plain(users)}

    async def get_user(self, user_id: str) -> dict:
        """查用户资料：UsersApi.get_user(user_id)。"""
        user_id = str(user_id or "").strip()
        if not user_id or _PLACEHOLDER_RE.search(user_id):
            raise VRCAPIError("查用户资料缺少 user_id")
        user = await self._client.call_on("users", "get_user", user_id)
        return {"ok": True, "action": "get_user", "user_id": user_id, "result": _to_plain(user)}

    async def friend(self, user_id: str) -> dict:
        """加好友：FriendsApi.friend(user_id)（写操作，默认开放 + 审计）。"""
        user_id = str(user_id or "").strip()
        if not user_id or _PLACEHOLDER_RE.search(user_id):
            raise VRCAPIError("加好友缺少 user_id（可先 search_users 获取）")
        self._audit_call("friends", "friend", {"user_id": user_id})
        result = await self._client.call_on("friends", "friend", user_id)
        logger.info("VRChat 加好友完成: %s", user_id)
        return {"ok": True, "action": "friend", "user_id": user_id, "result": _to_plain(result)}

    async def accept_friend_request(self, notification_id: str) -> dict:
        """接受好友申请：NotificationsApi.accept_friend_request(notification_id)（写操作+审计）。"""
        notification_id = str(notification_id or "").strip()
        if not notification_id or _PLACEHOLDER_RE.search(notification_id):
            raise VRCAPIError("接受好友申请缺少 notification_id（可先 get_notifications 获取）")
        self._audit_call("notifications", "accept_friend_request", {"notification_id": notification_id})
        result = await self._client.call_on("notifications", "accept_friend_request", notification_id)
        logger.info("VRChat 接受好友申请完成: %s", notification_id)
        return {"ok": True, "action": "accept_friend_request", "notification_id": notification_id, "result": _to_plain(result)}

    async def invite_user(self, user_id: str) -> dict:
        """邀请玩家到当前世界：InviteApi.invite_user(user_id)（写操作+审计）。"""
        user_id = str(user_id or "").strip()
        if not user_id or _PLACEHOLDER_RE.search(user_id):
            raise VRCAPIError("邀请玩家缺少 user_id")
        self._audit_call("invite", "invite_user", {"user_id": user_id})
        result = await self._client.call_on("invite", "invite_user", user_id)
        logger.info("VRChat 邀请玩家完成: %s", user_id)
        return {"ok": True, "action": "invite_user", "user_id": user_id, "result": _to_plain(result)}

    async def get_notifications(self, limit: int = DEFAULT_LIST_LIMIT) -> dict:
        """查通知：NotificationsApi.get_notifications(n)。"""
        limit = _to_int(limit, DEFAULT_LIST_LIMIT, "limit")
        notifications = await self._client.call_on("notifications", "get_notifications", n=limit)
        return {"ok": True, "action": "get_notifications", "limit": limit, "notifications": _to_plain(notifications)}

    async def get_favorites(self, limit: int = DEFAULT_LIST_LIMIT) -> dict:
        """查收藏：FavoritesApi.get_favorites(n)。"""
        limit = _to_int(limit, DEFAULT_LIST_LIMIT, "limit")
        favorites = await self._client.call_on("favorites", "get_favorites", n=limit)
        return {"ok": True, "action": "get_favorites", "limit": limit, "favorites": _to_plain(favorites)}

    async def get_recent_worlds(self, limit: int = DEFAULT_WORLD_LIMIT) -> dict:
        """查最近去过的世界：WorldsApi.get_recent_worlds(n)。"""
        limit = _to_int(limit, DEFAULT_WORLD_LIMIT, "limit")
        worlds = await self._client.call_on("worlds", "get_recent_worlds", n=limit)
        return {"ok": True, "action": "get_recent_worlds", "limit": limit, "worlds": _to_plain(worlds)}

    async def get_instance(self, world_id: str = "", instance_id: str = "") -> dict:
        """查世界实例：InstancesApi.get_instance(world_id, instance_id)。"""
        world_id = str(world_id or "").strip()
        instance_id = str(instance_id or "").strip()
        if not world_id or not instance_id or _PLACEHOLDER_RE.search(world_id) or _PLACEHOLDER_RE.search(instance_id):
            raise VRCAPIError("查实例缺少 world_id/instance_id（形如 wrld_xxx / 12345~private(usr_xxx)~region(us)）")
        instance = await self._client.call_on("instances", "get_instance", world_id, instance_id)
        return {"ok": True, "action": "get_instance", "world_id": world_id, "instance_id": instance_id, "result": _to_plain(instance)}

    # ------------------------------------------------------------------
    # call_api 通用层（vrchatapi 全量接入的通用通道，T03）
    # ------------------------------------------------------------------
    async def call_api(self, api: str, args: dict | None = None) -> dict:
        """通用 VRChat API 调用：目录校验 → 安全策略 → call_on → 序列化信封。

        :param api: 19 类句柄名（authentication/avatars/.../worlds）
        :param args: SDK 方法参数 dict；其中 ``method`` 为必填方法名，
            其余为该方法参数（必须命中目录 params 白名单）
        :returns: ``{"ok": True, "api", "method", "args", "result"}`` 信封；
            拒绝/未知方法抛 VRCAPIError
        """
        api = str(api or "").strip()
        args = dict(args or {})
        method = str(args.pop("method", "") or "").strip()
        if not method:
            raise VRCAPIError("call_api 缺少 method（args.method 指定 SDK 方法名）")
        entry = self._catalog.validate(api, method, args)
        allow, reason = self._policy.resolve(entry)
        if not allow:
            raise VRCAPIError(reason)
        if self._policy.is_audited(entry):
            self._audit(entry, args)
        result = await self._client.call_on(api, method, **args)
        plain = _to_plain(result)
        logger.info(
            "VRChat call_api: %s.%s args=%s",
            api,
            method,
            json.dumps(args, ensure_ascii=False, default=str)[:200],
            extra={"event": "vrcapi_call_api", "vrc_api": api, "vrc_method": method},
        )
        return {"ok": True, "api": api, "method": method, "args": args, "result": plain}

    # ------------------------------------------------------------------
    # 工具派发（动作系统 vrc_* 工具 → 精选动作 / call_api）
    # ------------------------------------------------------------------
    async def run(self, tool: str, args: dict | None = None) -> dict:
        """按动作映射工具名执行（MCPOSCActuator 对 vrc_* 工具调用本方法）。

        :param tool: 映射表工具名，如 vrc_change_avatar / vrc_list_friends /
                     vrc_search_world / vrc_set_status / vrc_call_api；
                     兼容常见别名（friends_online/friends→vrc_list_friends，
                     BugFix：用户实测 LLM 爱这么写）与裸精选动作名
                     （list_friends→vrc_list_friends）
        :param args: 映射表 args（经 ActionMapper 填充 {param} 占位符）
        :returns: 统一 dict；未知工具返回 ``{"ok": False, "message": ...}``
                  （含可用动作列表，供 LLM 下一轮纠正；**不抛异常**导致整轮失败）
        """
        args = dict(args or {})
        resolved = _resolve_vrc_tool(tool)
        if resolved is None:
            logger.warning("未知 vrc 动作/工具: %s", tool)
            return {
                "ok": False,
                "action": str(tool or ""),
                "message": f"未知 vrc 动作 {tool}，可用：{_VRC_TOOLS_TEXT}",
            }
        tool = resolved
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
        if tool == "vrc_get_current_user":
            return await self.get_current_user()
        if tool == "vrc_get_own_avatar":
            return await self.get_own_avatar()
        if tool == "vrc_search_users":
            return await self.search_users(
                query=str(args.get("query", "") or ""),
                limit=_to_int(args.get("limit"), DEFAULT_LIST_LIMIT, "limit"),
            )
        if tool == "vrc_get_user":
            return await self.get_user(str(args.get("user_id", "") or ""))
        if tool == "vrc_friend":
            return await self.friend(str(args.get("user_id", "") or ""))
        if tool == "vrc_accept_friend_request":
            return await self.accept_friend_request(str(args.get("notification_id", "") or ""))
        if tool == "vrc_invite_user":
            return await self.invite_user(str(args.get("user_id", "") or ""))
        if tool == "vrc_get_notifications":
            return await self.get_notifications(
                limit=_to_int(args.get("limit"), DEFAULT_LIST_LIMIT, "limit")
            )
        if tool == "vrc_get_favorites":
            return await self.get_favorites(
                limit=_to_int(args.get("limit"), DEFAULT_LIST_LIMIT, "limit")
            )
        if tool == "vrc_get_recent_worlds":
            return await self.get_recent_worlds(
                limit=_to_int(args.get("limit"), DEFAULT_WORLD_LIMIT, "limit")
            )
        if tool == "vrc_get_instance":
            return await self.get_instance(
                world_id=str(args.get("world_id", "") or ""),
                instance_id=str(args.get("instance_id", "") or ""),
            )
        if tool == "vrc_call_api":
            return await self.call_api(
                api=str(args.get("api", "") or ""),
                args=_coerce_args_dict(args.get("args")),
            )
        # 防御：resolved 必在 _VRC_TOOLS 且上方全部分支已覆盖；万一漏配
        # （新增工具未加分支）返回清晰错误而非抛异常，避免整轮失败。
        logger.error("vrc 工具未派发（内部漏配分支）: %s", tool)
        return {
            "ok": False,
            "action": tool,
            "message": f"vrc 工具 {tool} 未派发（内部配置缺失），可用：{_VRC_TOOLS_TEXT}",
        }
