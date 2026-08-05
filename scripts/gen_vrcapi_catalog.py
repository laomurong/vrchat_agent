"""生成 VRChat API 目录（src/vrchat_agent/vrcapi/catalog.yaml）。

数据源：**已安装的 vrchatapi SDK**（解析每个 API 类的 *_with_http_info 源码，
提取 REST path / HTTP verb / 参数名），保证目录与真实 SDK 零漂移——
mock/文档里写的方法不存在时本脚本生成不出该条目。

用法：
    conda activate vrchat-agent
    python scripts/gen_vrcapi_catalog.py            # 输出到 src/vrchat_agent/vrcapi/catalog.yaml

安全分级（用户已确认的决策 3）：
- read_only：HTTP GET → security=open（只读默认开放）
- 写操作（非 GET）：
    - 命中 SOCIAL_WHITELIST（好友/邀请/通知/收藏/状态/头像等社交写操作）
      → security=whitelist（默认开放 + 审计日志）
    - 命中 CLOSED_PATTERNS（文件上传/付费/账号注册删除/群管理/内容发布/
      实例创建关闭/道具发布/印记上传等危险操作）→ security=closed（默认不可调）
    - 其余写操作 → security=whitelist（保守：可调但需 allow_destructive 或审计）
- 默认 allow_destructive=false（运行期开关在 vrcapi.allow_destructive）。
"""

from __future__ import annotations

import importlib
import inspect
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "src" / "vrchat_agent" / "vrcapi" / "catalog.yaml"

# 社交写操作白名单（默认开放 + 审计日志；用户已确认的决策 1/3）
SOCIAL_WHITELIST: frozenset[str] = frozenset({
    # friends
    "friend", "unfriend", "boop", "delete_friend_request",
    # notifications（好友申请确认 / 通知清理）
    "accept_friend_request", "acknowledge_notification_v2", "delete_notification",
    "delete_notification_v2", "clear_notifications", "delete_all_notification_v2s",
    # invite（邀请 / 请求邀请）
    "invite_user", "invite_user_with_photo", "request_invite", "invite_myself_to",
    "get_invite_message", "get_invite_messages",
    # 账号状态（自身）
    "update_user", "add_tags", "remove_tags",
    # 头像（换头像 / 收藏）
    "select_avatar", "add_favorite", "delete_favorite", "clear_favorite_group",
    # 群（加入/离开/取消请求 —— 社交动作；其余群管理 closed）
    "join_group", "leave_group", "cancel_group_request",
    # 玩家面板（可逆的轻量操作）
    "unmoderate_user",
})

# 危险操作（closed，默认不可调）
CLOSED_PATTERNS: tuple[re.Pattern[str], ...] = (
    # 文件上传/下载管理
    re.compile(r"^(create_file|create_file_version|finish_file_data_upload|start_file_data_upload|delete_file|delete_file_version)"),
    # 内容上传（图标/图片/画廊图/印记）
    re.compile(r"^upload_"),
    # 付费 / 经济
    re.compile(r"^(buy|purchase|subscribe|unsubscribe|redeem|redeem_code|steam|oculus|license)"),
    # 账号注册 / 删除 / 认证敏感（含 2FA 管理）
    re.compile(r"^(register|delete_user|verify|confirm_email|check_user_exists)"),
    re.compile(r"^(disable2_fa|enable2_fa|cancel_pending2_fa|resend_email_confirmation)"),
    # 群管理（创建/删除/改/封禁/角色/帖子/图片/踢人/转移/审核等全部写操作）
    re.compile(r"^(create_group|delete_group|update_group|ban_group_member|unban_group_member|add_group_member_role|remove_group_member_role|add_group_gallery_image|delete_group_gallery_image|add_group_post|delete_group_post|update_group_post|block_group|unblock_group|transfer_group|create_group_invite|delete_group_invite|create_group_announcement|delete_group_announcement|edit_group_member|create_group_calendar_event|delete_group_calendar_event|update_group_calendar_event|follow_group_calendar_event|unfollow_group_calendar_event|kick_group_member|initiate_or_accept_group_transfer|cancel_group_transfer|respond_group_join_request)"),
    # 内容发布（世界/头像/道具）
    re.compile(r"^(create_world|update_world|delete_world|create_avatar|delete_avatar|delete_impostor|enqueue_impostor|create_prop|update_prop|delete_prop|publish_prop)"),
    # 实例创建/关闭
    re.compile(r"^(create_instance|close_instance)"),
    # 印记上传/编辑/删除
    re.compile(r"^(upload_print|edit_print|delete_print)"),
    # 全局玩家封禁（不可逆，重）
    re.compile(r"^(moderate_user|clear_all_player_moderations)"),
    # 库存消费/装备（影响账号资产）
    re.compile(r"^(consume_own_inventory_item|delete_own_inventory_item|equip_own_inventory_item)"),
)

# 全部 API 模块 → 句柄名（vrchatapi.api.<module> → VRCAPIClient.<handle>_api）
API_HANDLES: tuple[tuple[str, str], ...] = (
    ("authentication_api", "authentication"),
    ("avatars_api", "avatars"),
    ("calendar_api", "calendar"),
    ("economy_api", "economy"),
    ("favorites_api", "favorites"),
    ("files_api", "files"),
    ("friends_api", "friends"),
    ("groups_api", "groups"),
    ("instances_api", "instances"),
    ("inventory_api", "inventory"),
    ("invite_api", "invite"),
    ("jams_api", "jams"),
    ("miscellaneous_api", "miscellaneous"),
    ("notifications_api", "notifications"),
    ("playermoderation_api", "playermoderation"),
    ("prints_api", "prints"),
    ("props_api", "props"),
    ("users_api", "users"),
    ("worlds_api", "worlds"),
)


def _class_name(module_name: str) -> str:
    """authentication_api → AuthenticationApi。"""
    parts = module_name.split("_")
    if parts[-1] == "api":
        parts = parts[:-1]
    return "".join(p.capitalize() for p in parts) + "Api"


#: SDK 生成代码的内部参数（非业务参数，目录不收录）
_INTERNAL_PARAMS: frozenset[str] = frozenset({
    "async_req", "_return_http_data_only", "_preload_content",
    "_request_timeout", "_request_auth", "_content_type", "_headers",
})


def _extract_method(api_cls: type, method: str) -> dict | None:
    """解析 *_with_http_info 源码，提取 path/verb/params。

    - params 取 SDK 源码 ``all_params = [...]`` 列表（调用方 kwargs 名，过滤
      内部参数 async_req/_headers 等）；
    - 极少数无 all_params 的旧式方法回退解析 query/path_params.append。
    """
    try:
        src = inspect.getsource(api_cls)
    except (OSError, TypeError):
        return None
    idx = src.find(f"def {method}_with_http_info")
    if idx < 0:
        return None
    # 截断到下一个方法（避免把后续方法的参数误收进来）
    nxt = src.find("\n    def ", idx + 1)
    chunk = src[idx : nxt if nxt > 0 else idx + 30000]
    m_url = re.search(r"call_api\(\s*'([^']*)'\s*,\s*'([A-Z]+)'", chunk)
    if not m_url:
        return None
    path, verb = m_url.group(1), m_url.group(2)

    params: list[str] = []
    am = re.search(r"all_params\s*=\s*\[(.*?)\]", chunk, re.DOTALL)
    if am:
        for m in re.finditer(r"'(\w+)'", am.group(1)):
            if m.group(1) not in _INTERNAL_PARAMS and m.group(1) not in params:
                params.append(m.group(1))
    else:
        # 回退：query/path_params.append(('name', ...))
        for m in re.finditer(r"query_params\.append\(\s*\(\s*'(\w+)'", chunk):
            if m.group(1) not in params:
                params.append(m.group(1))
        for m in re.finditer(r"path_params\.append\(\s*\(\s*'(\w+)'", chunk):
            if m.group(1) not in params:
                params.append(m.group(1))
    return {"path": path, "verb": verb, "params": params}


def _security(api: str, method: str, verb: str) -> str:
    """安全分级：read_only 开放；写操作白名单/危险/其他。"""
    if verb == "GET":
        return "open"
    if method in SOCIAL_WHITELIST:
        return "whitelist"
    if any(p.search(method) for p in CLOSED_PATTERNS):
        return "closed"
    # 其余写操作保守归 whitelist（可调但需 allow_destructive/审计）
    return "whitelist"


def generate() -> dict:

    apis: dict[str, dict] = {}
    total = 0
    for module_name, handle in API_HANDLES:
        mod = importlib.import_module(f"vrchatapi.api.{module_name}")
        cls = getattr(mod, _class_name(module_name))
        methods: dict[str, dict] = {}
        for name in sorted(dir(cls)):
            if name.startswith("_") or name.endswith("_with_http_info"):
                continue
            member = getattr(cls, name)
            if not inspect.isfunction(member):
                continue
            info = _extract_method(cls, name)
            if info is None:
                continue
            verb = info["verb"]
            security = _security(handle, name, verb)
            desc = ""
            if member.__doc__:
                first = member.__doc__.strip().splitlines()[0].strip()
                desc = re.sub(r"\s*#\s*noqa:\s*E501\s*$", "", first)
            methods[name] = {
                "verb": verb,
                "path": info["path"],
                "params": info["params"],
                "read_only": verb == "GET",
                "security": security,
                "desc": desc,
            }
            total += 1
        apis[handle] = {"methods": methods}
    return {
        "version": 1,
        "generated_by": "scripts/gen_vrcapi_catalog.py",
        "note": "由已安装 vrchatapi SDK 自动生成；改 SDK 版本后请重新运行生成脚本。",
        "apis": apis,
    }


def main() -> int:
    data = generate()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)
    n = sum(len(v["methods"]) for v in data["apis"].values())
    print(f"OK 已生成 {OUT_PATH}（{n} 个方法，{len(data['apis'])} 个 API 类）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
