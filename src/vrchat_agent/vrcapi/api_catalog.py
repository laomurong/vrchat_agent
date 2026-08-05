"""VRChat API 目录：vrchatapi 274 方法的安全目录（T03）。

数据源：``src/vrchat_agent/vrcapi/catalog.yaml``（机器可读，由
``scripts/gen_vrcapi_catalog.py`` 从**已安装 SDK** 生成，防 mock/文档漂移）；
运行时只读 YAML，不 import vrchatapi（测试离线可跑）。

安全分级（用户已确认的决策 3）：
- ``open``：只读（GET）默认开放；
- ``whitelist``：写操作白名单（社交/状态/收藏等）默认开放 + **审计日志**；
- ``closed``：危险操作（文件上传/付费/账号注册删除/群管理/内容发布/实例创建
  关闭/道具发布/印记上传等）默认**不可调**；
- 运行期总开关 ``vrcapi.allow_destructive``（默认 false）：为 true 时
  whitelist 写操作放行，closed 仍拒绝。

``SecurityPolicy`` 是纯函数式判定器（可单测）：
``resolve(entry, allow_destructive) -> allow / deny_reason``。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .client import VRCAPIError

logger = logging.getLogger("vrchat_agent.vrcapi.api_catalog")

#: 目录 YAML 默认路径（相对本文件）
_CATALOG_PATH = Path(__file__).resolve().parent / "catalog.yaml"

#: 安全级别
SECURITY_OPEN = "open"
SECURITY_WHITELIST = "whitelist"
SECURITY_CLOSED = "closed"

#: 社交写操作（默认开放 + 审计日志；用户已确认决策 1/3）——与生成脚本白名单保持一致
AUDIT_WRITE_ACTIONS: frozenset[str] = frozenset({
    # friends
    "friend", "unfriend", "boop", "delete_friend_request",
    # notifications（好友申请确认 / 通知清理）
    "accept_friend_request", "acknowledge_notification_v2", "delete_notification",
    "delete_notification_v2", "clear_notifications", "delete_all_notification_v2s",
    # invite
    "invite_user", "invite_user_with_photo", "request_invite", "invite_myself_to",
    # 账号状态 / 头像 / 收藏 / 群社交
    "update_user", "add_tags", "remove_tags",
    "select_avatar", "add_favorite", "delete_favorite", "clear_favorite_group",
    "join_group", "leave_group", "cancel_group_request",
    "unmoderate_user",
})


@dataclass(frozen=True)
class CatalogEntry:
    """目录条目（api + method 的唯一记录）。"""

    api: str
    method: str
    verb: str
    path: str
    params: tuple[str, ...]
    read_only: bool
    security: str
    desc: str = ""


class APICatalog:
    """VRchatapi 方法目录：lookup / validate / security。

    - ``lookup(api, method)``：返回 CatalogEntry（不存在返回 None）；
    - ``validate(api, method, args)``：校验方法存在 + 参数在白名单内；
    - ``security(api, method)``：安全级别；
    - ``all_methods()``：全量 (api, method) 列表。
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else _CATALOG_PATH
        self._entries: dict[tuple[str, str], CatalogEntry] = {}
        self._load()

    def _load(self) -> None:
        import yaml

        if not self._path.exists():
            raise VRCAPIError(f"API 目录不存在: {self._path}（请运行 scripts/gen_vrcapi_catalog.py 生成）")
        try:
            with open(self._path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            raise VRCAPIError(f"API 目录 YAML 解析失败: {exc}") from exc
        apis = data.get("apis", data) if isinstance(data, dict) else {}
        for api_name, api_info in (apis or {}).items():
            methods = api_info.get("methods", {}) if isinstance(api_info, dict) else {}
            for method_name, item in methods.items():
                if not isinstance(item, dict):
                    continue
                self._entries[(str(api_name), str(method_name))] = CatalogEntry(
                    api=str(api_name),
                    method=str(method_name),
                    verb=str(item.get("verb", "GET")).upper(),
                    path=str(item.get("path", "")),
                    params=tuple(str(p) for p in item.get("params", [])),
                    read_only=bool(item.get("read_only", False)),
                    security=str(item.get("security", SECURITY_OPEN)),
                    desc=str(item.get("desc", "")),
                )
        logger.debug("API 目录加载完成: %d 条（%s）", len(self._entries), self._path)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def lookup(self, api: str, method: str) -> CatalogEntry | None:
        """按 api 句柄 + SDK 方法名查目录条目（不存在返回 None）。"""
        key = (str(api or "").strip().lower(), str(method or "").strip())
        return self._entries.get(key)

    def security(self, api: str, method: str) -> str:
        """安全级别（open / whitelist / closed）；未知方法返回 closed（保守拒绝）。"""
        entry = self.lookup(api, method)
        if entry is None:
            return SECURITY_CLOSED
        return entry.security

    def validate(self, api: str, method: str, args: dict | None) -> CatalogEntry:
        """校验方法存在 + 参数合法（未知方法/未知参数抛 VRCAPIError）。"""
        entry = self.lookup(api, method)
        if entry is None:
            raise VRCAPIError(
                f"VRChat API 目录中不存在方法: {api}.{method}（19 类句柄见 docs/vrcapi_catalog.md）"
            )
        args = dict(args or {})
        known = set(entry.params)
        unknown = [k for k in args if k not in known]
        if unknown:
            raise VRCAPIError(
                f"VRChat API 方法 {api}.{method} 收到未知参数: {unknown}（合法参数: {sorted(known)}）"
            )
        return entry

    def all_methods(self) -> list[tuple[str, str]]:
        """全量 (api, method) 列表（稳定排序）。"""
        return sorted(self._entries)

    def methods_of(self, api: str) -> list[CatalogEntry]:
        """某句柄下的全部条目（按方法名排序）。"""
        out = [e for (a, _), e in self._entries.items() if a == api]
        out.sort(key=lambda e: e.method)
        return out

    @property
    def api_names(self) -> tuple[str, ...]:
        """19 类句柄名（稳定排序）。"""
        return tuple(sorted({a for a, _ in self._entries}))

    def __len__(self) -> int:
        return len(self._entries)


class SecurityPolicy:
    """安全判定器：只读 open；写白名单默认开放；其余写需 allow_destructive；危险 closed。

    判定规则（用户已确认决策 3）：
    1. 只读（GET）→ 允许；
    2. 命中社交白名单（AUDIT_WRITE_ACTIONS）→ 允许 + 审计；
    3. 其余写操作 → 仅当 allow_destructive=true 时允许 + 审计；
    4. 危险操作（closed）→ 一律拒绝（即使 allow_destructive=true）。

    纯函数（无状态），便于单元测试。
    """

    def __init__(self, allow_destructive: bool = False) -> None:
        self._allow_destructive: bool = bool(allow_destructive)

    @property
    def allow_destructive(self) -> bool:
        """vrcapi.allow_destructive 开关（默认 false）。"""
        return self._allow_destructive

    def resolve(self, entry: CatalogEntry) -> tuple[bool, str]:
        """判定是否允许调用。

        :returns: (允许, 拒绝原因字符串；允许时原因为空串)
        """
        if entry.read_only or entry.security == SECURITY_OPEN:
            return True, ""
        if entry.security == SECURITY_CLOSED:
            return False, (
                f"{entry.api}.{entry.method} 是危险操作（closed）："
                "文件上传/付费/账号注册删除/群管理/内容发布/实例创建关闭/道具发布/印记上传等，默认不可调"
            )
        # whitelist 写操作：命中社交白名单默认开放；其余写操作需 allow_destructive
        if entry.method in AUDIT_WRITE_ACTIONS:
            return True, ""
        if self._allow_destructive:
            return True, ""
        return False, (
            f"{entry.api}.{entry.method} 是写操作且不在白名单内，"
            "需开启 vrcapi.allow_destructive 或命中白名单"
        )

    def is_audited(self, entry: CatalogEntry) -> bool:
        """是否需记审计日志：允许的写操作（open 只读除外，closed 不会被允许）。"""
        if entry.read_only or entry.security == SECURITY_OPEN:
            return False
        return entry.security != SECURITY_CLOSED
