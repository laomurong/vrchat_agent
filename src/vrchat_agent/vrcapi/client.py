"""VRChat API 集成：懒加载登录 + 2FA 处理 + 全局限速器（任务①）。

设计要点：
- **懒加载**：构造 VRCAPIClient 不触发任何登录/网络/import SDK；首次真正调用
  API（``call_on`` → ``ensure_authed``）时才读环境变量、配置 SDK 并登录。
  未配置 / 未使用时，import 与进程启动均无副作用。
- **凭据只从环境变量读取**（默认 ``VRC_USERNAME`` / ``VRC_PASSWORD`` /
  ``VRC_API_KEY``），绝不硬编码、绝不落盘明文；缺凭据在登录期抛清晰
  ``VRCAPIError``（提示设置环境变量）。
- **2FA**：登录返回 ``requiresTwoFactorAuth`` 时，通过 ``two_factor_code_provider``
  回调获取验证码（默认 None → 抛清晰错误提示配置）；优先走 TOTP 验证
  （``verify2_fa``），邮箱验证码走 ``verify2_fa_email_code``（SDK 1.20.7
  已移除旧版 ``verify2fa_totp`` / ``verify2fa(code, method)``）。
- **全局限速器**：VRChat API 对单账号请求频率敏感（官方约 60s/请求），本模块用
  ``RateLimiter`` 对**所有** API 请求做全局串行节流——任意时刻最多一个请求
  在途，两次请求开始时间间隔 ≥ ``rate_limit_seconds``（默认 60s）。
- **不阻塞事件循环**：vrchatapi SDK 是同步阻塞库，所有 SDK 调用经
  ``asyncio.to_thread`` 放入线程池执行（参考 speech/tts_cosyvoice.py 模式）。
- **异常包装**：所有外部 API 异常统一包装为 ``VRCAPIError``，不裸抛。
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import logging
import os
import tempfile
import time
from collections.abc import Callable
from http.cookiejar import Cookie
from pathlib import Path
from typing import Any

logger = logging.getLogger("vrchat_agent.vrcapi.client")

#: VRChat API 官方 REST 端点
DEFAULT_API_HOST: str = "https://api.vrchat.cloud/api/1"

#: 默认凭据环境变量名（key 走 .env，绝不写入 config.yaml / 代码）
DEFAULT_USERNAME_ENV: str = "VRC_USERNAME"
DEFAULT_PASSWORD_ENV: str = "VRC_PASSWORD"
DEFAULT_API_KEY_ENV: str = "VRC_API_KEY"

#: 默认请求最小间隔（秒）——VRChat API 官方对单账号请求频率敏感
DEFAULT_RATE_LIMIT_SECONDS: float = 60.0

#: 会话（auth cookie）持久化文件路径（相对项目根；登录成功原子写，重启免登录复用）
DEFAULT_AUTH_CACHE_PATH: str = "data/vrchat_auth.json"

#: 会话缓存文件结构版本（升级时兼容旧文件）
_AUTH_CACHE_VERSION: int = 1

#: 元数据缺失时的回退版本号（与 pyproject.toml / __init__.__version__ 一致）
_FALLBACK_VERSION: str = "0.1.0"


def _build_default_user_agent() -> str:
    """构造默认 User-Agent：应用名 + 版本 + 联系方式（VRChat WAF 403 修复）。

    VRChat 的 Cloudflare WAF 要求请求 UA 必须包含应用名、版本与联系方式，
    否则返回 403 Forbidden（waf_code 13799）。版本优先读包元数据
    ``vrchat-agent``（``importlib.metadata.version``），缺失时回退
    ``_FALLBACK_VERSION``，避免元数据未注册导致构建失败。
    """
    try:
        version = importlib.metadata.version("vrchat-agent")
    except Exception:  # noqa: BLE001 - 元数据缺失/损坏均回退常量版本，保证 UA 总能构造
        version = _FALLBACK_VERSION
    return (
        f"VRChatAgent/{version} "
        "(contact: laomurong/vrchat_agent GitHub; "
        "https://github.com/laomurong/vrchat_agent)"
    )


#: 默认 User-Agent（VRChat WAF 强制要求：应用名 + 版本 + 联系方式）
DEFAULT_USER_AGENT: str = _build_default_user_agent()

#: 19 类 API 句柄 → (SDK 模块名, SDK API 类名)
#: 登录成功后为每类创建 handle，存 ``_handles`` 字典；旧 4 属性（avatars/friends/
#: worlds/users）保留兼容。
HANDLE_MODULES: tuple[tuple[str, str, str], ...] = (
    ("authentication", "authentication_api", "AuthenticationApi"),
    ("avatars", "avatars_api", "AvatarsApi"),
    ("calendar", "calendar_api", "CalendarApi"),
    ("economy", "economy_api", "EconomyApi"),
    ("favorites", "favorites_api", "FavoritesApi"),
    ("files", "files_api", "FilesApi"),
    ("friends", "friends_api", "FriendsApi"),
    ("groups", "groups_api", "GroupsApi"),
    ("instances", "instances_api", "InstancesApi"),
    ("inventory", "inventory_api", "InventoryApi"),
    ("invite", "invite_api", "InviteApi"),
    ("jams", "jams_api", "JamsApi"),
    ("miscellaneous", "miscellaneous_api", "MiscellaneousApi"),
    ("notifications", "notifications_api", "NotificationsApi"),
    ("playermoderation", "playermoderation_api", "PlayermoderationApi"),
    ("prints", "prints_api", "PrintsApi"),
    ("props", "props_api", "PropsApi"),
    ("users", "users_api", "UsersApi"),
    ("worlds", "worlds_api", "WorldsApi"),
)


class VRCAPIError(RuntimeError):
    """VRChat API 集成错误（所有外部异常统一包装为该类型）。"""


class RateLimiter:
    """全局限速器：串行化 + 最小间隔（VRChat API ≤ 1 次/60 秒）。

    - 任意时刻最多一个请求在途（``asyncio.Lock`` 串行，全局共享同一实例）；
    - 两次请求**开始**时间间隔 ≥ ``min_interval_seconds``；
    - ``clock`` 可注入（测试用假时钟），默认 ``time.monotonic``。
    """

    def __init__(self, min_interval_seconds: float = DEFAULT_RATE_LIMIT_SECONDS, clock: Callable[[], float] | None = None) -> None:
        self._min_interval: float = max(0.0, float(min_interval_seconds))
        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic
        self._last_ts: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def min_interval_seconds(self) -> float:
        """配置的最小请求间隔（秒）。"""
        return self._min_interval

    def _wait_seconds(self) -> float:
        """距离上次请求还需等待的秒数（≤0 表示立即可发）。"""
        return max(0.0, self._last_ts + self._min_interval - self._clock())

    async def acquire(self) -> None:
        """获取发送许可（阻塞直到满足"串行 + 最小间隔"约束）。"""
        async with self._lock:
            wait = self._wait_seconds()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_ts = self._clock()


class VRCAPIClient:
    """VRChat API 客户端：懒加载登录 + 2FA + 全局限速 + 线程池执行。

    用法（动作执行端）：
        client = VRCAPIClient()
        result = await client.call_on("avatars", "select_avatar", avatar_id)
    未登录时首次 ``call_on`` 自动完成登录（含 2FA），之后缓存 SDK 句柄。
    """

    def __init__(
        self,
        username_env: str = DEFAULT_USERNAME_ENV,
        password_env: str = DEFAULT_PASSWORD_ENV,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        rate_limit_seconds: float = DEFAULT_RATE_LIMIT_SECONDS,
        two_factor_code_provider: Callable[[list[str]], str] | None = None,
        clock: Callable[[], float] | None = None,
        auth_cache_path: str | Path | None = None,
    ) -> None:
        self._username_env: str = (username_env or DEFAULT_USERNAME_ENV).strip() or DEFAULT_USERNAME_ENV
        self._password_env: str = (password_env or DEFAULT_PASSWORD_ENV).strip() or DEFAULT_PASSWORD_ENV
        self._api_key_env: str = (api_key_env or DEFAULT_API_KEY_ENV).strip() or DEFAULT_API_KEY_ENV
        self._two_factor_code_provider: Callable[[list[str]], str] | None = two_factor_code_provider
        self._limiter = RateLimiter(rate_limit_seconds, clock=clock)
        self._auth_lock = asyncio.Lock()
        self._authed: bool = False
        # P2-2：会话持久化路径（None = 禁用；main.py 传 data/vrchat_auth.json）
        self._auth_cache_path: Path | None = Path(auth_cache_path) if auth_cache_path else None
        # Web 登录状态（GET /api/vrc/status 用）
        self._login_in_progress: bool = False
        self._login_error: str | None = None

        # SDK 句柄（登录成功后填充；登录前为 None → 属性访问抛清晰 VRCAPIError）
        self._api_client: Any = None
        self._handles: dict[str, Any] = {}
        self._auth_api: Any = None
        self._avatars_api: Any = None
        self._friends_api: Any = None
        self._worlds_api: Any = None
        self._users_api: Any = None
        self._update_user_request_cls: Any = None
        self._current_user: Any = None

    # ------------------------------------------------------------------
    # 公开状态 / 句柄
    # ------------------------------------------------------------------
    @property
    def is_authed(self) -> bool:
        """是否已完成登录（懒加载：构造后为 False，首次调用后为 True）。"""
        return self._authed

    @property
    def login_in_progress(self) -> bool:
        """是否有登录流程正在进行（含 2FA 等待；Web 登录状态用）。"""
        return self._login_in_progress

    @property
    def login_error(self) -> str | None:
        """最近一次登录失败的错误信息（未失败为 None）。"""
        return self._login_error

    @property
    def display_name(self) -> str:
        """当前登录用户的 displayName（未登录返回空串）。

        兼容两种 current_user 形态：真实 SDK 返回模型对象（.displayName 属性）、
        测试替身返回 dict（.get("displayName")）。
        """
        if self._current_user is None:
            return ""
        if isinstance(self._current_user, dict):
            return str(self._current_user.get("displayName", "") or "")
        return str(getattr(self._current_user, "displayName", "") or "")

    @property
    def auth_cache_path(self) -> Path | None:
        """会话缓存文件路径（None = 未启用持久化）。"""
        return self._auth_cache_path

    @property
    def rate_limiter(self) -> RateLimiter:
        """全局限速器实例（测试可读取间隔/等待时间）。"""
        return self._limiter

    @property
    def current_user(self) -> Any:
        """登录返回的当前用户对象（未登录访问抛 VRCAPIError）。"""
        if self._current_user is None:
            raise VRCAPIError("VRChat 未登录（请先调用 ensure_authed）")
        return self._current_user

    @property
    def handles(self) -> dict[str, Any]:
        """19 类 API 句柄字典（登录后填充；未登录为空字典）。"""
        return dict(self._handles)

    def get_handle(self, api_name: str) -> Any:
        """按句柄名取 API 实例（authentication/avatars/.../worlds，19 类）。

        未登录或句柄不存在 → 抛清晰 VRCAPIError。
        """
        if not self._authed or not self._handles:
            raise VRCAPIError("VRChat 未登录（请先调用 ensure_authed）")
        handle = self._handles.get(str(api_name or "").strip())
        if handle is None:
            raise VRCAPIError(
                f"VRChat API 句柄不存在: {api_name}（可用: {sorted(self._handles)}）"
            )
        return handle

    @property
    def avatars_api(self) -> Any:
        if self._avatars_api is None:
            raise VRCAPIError("VRChat 未登录（请先调用 ensure_authed）")
        return self._avatars_api

    @property
    def friends_api(self) -> Any:
        if self._friends_api is None:
            raise VRCAPIError("VRChat 未登录（请先调用 ensure_authed）")
        return self._friends_api

    @property
    def worlds_api(self) -> Any:
        if self._worlds_api is None:
            raise VRCAPIError("VRChat 未登录（请先调用 ensure_authed）")
        return self._worlds_api

    @property
    def users_api(self) -> Any:
        if self._users_api is None:
            raise VRCAPIError("VRChat 未登录（请先调用 ensure_authed）")
        return self._users_api

    @property
    def update_user_request_cls(self) -> Any:
        """vrchatapi 的 UpdateUserRequest 模型类（构造状态更新请求用）。"""
        if self._update_user_request_cls is None:
            raise VRCAPIError("VRChat 未登录（请先调用 ensure_authed）")
        return self._update_user_request_cls

    # ------------------------------------------------------------------
    # 登录（懒加载）
    # ------------------------------------------------------------------
    async def ensure_authed(self) -> None:
        """确保已登录（首次调用触发登录 + 2FA；已登录直接返回）。

        登录流程：优先尝试用持久化 auth cookie 恢复会话（``_try_restore_session``，
        免登录）；缓存缺失/失效（401 等）则清除缓存、用用户名密码重新登录，
        成功后把 cookie 持久化（``_persist_session``）。
        """
        async with self._auth_lock:
            if self._authed:
                return
            self._login_in_progress = True
            self._login_error = None
            try:
                await asyncio.to_thread(self._login_sync)
            except VRCAPIError as exc:
                self._login_error = str(exc)
                raise
            except Exception as exc:
                wrapped = VRCAPIError(f"VRChat 登录失败: {exc}")
                self._login_error = str(wrapped)
                raise wrapped from exc
            finally:
                self._login_in_progress = False
            self._authed = True
            logger.info("VRChat 登录完成")

    async def logout(self) -> None:
        """登出：调用 VRChat logout API 使服务端会话失效，清内存状态 + 删缓存。

        幂等：未登录时也清缓存文件。关闭进程不主动登出（保留 cookie 复用）。
        """
        async with self._auth_lock:
            if self._authed:
                try:
                    await asyncio.to_thread(self._logout_sync)
                except Exception as exc:
                    raise VRCAPIError(f"VRChat 登出失败: {exc}") from exc
                finally:
                    self._reset_session_state()
            self._clear_cache()
            self._login_error = None
            self._login_in_progress = False
            logger.info("VRChat 已登出（持久化会话已清除）")

    def _logout_sync(self) -> None:
        """同步调用 SDK logout（使服务端 auth cookie 失效）。"""
        if self._auth_api is not None:
            self._auth_api.logout()

    def _reset_session_state(self) -> None:
        """清空内存中的登录状态与 SDK 句柄（登出用；保留限速器）。"""
        self._authed = False
        self._api_client = None
        self._handles = {}
        self._auth_api = None
        self._avatars_api = None
        self._friends_api = None
        self._worlds_api = None
        self._users_api = None
        self._update_user_request_cls = None
        self._current_user = None

    async def call_on(self, api_name: str, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """限速 + 串行 + 线程池执行一次 SDK 调用（自动确保登录）。

        :param api_name: SDK API 句柄名（19 类：authentication/avatars/calendar/
            economy/favorites/files/friends/groups/instances/inventory/invite/jams/
            miscellaneous/notifications/playermoderation/prints/props/users/worlds）
        :param method_name: 句柄上的同步方法名（如 select_avatar / get_friends）
        :returns: SDK 方法返回值；异常统一包装为 VRCAPIError
        """
        await self.ensure_authed()
        handle = self.get_handle(api_name)
        method = getattr(handle, method_name, None)
        if method is None or not callable(method):
            raise VRCAPIError(
                f"VRChat API 句柄 {api_name} 无此方法: {method_name}"
                f"（可用方法见 docs/vrcapi_catalog.md）"
            )
        await self._limiter.acquire()
        try:
            return await asyncio.to_thread(method, *args, **kwargs)
        except VRCAPIError:
            raise
        except Exception as exc:
            raise VRCAPIError(f"VRChat API 调用失败 {api_name}.{method_name}: {exc}") from exc

    # ------------------------------------------------------------------
    # 同步登录流程（供 asyncio.to_thread 调用，不阻塞事件循环）
    # ------------------------------------------------------------------
    def _login_sync(self) -> None:
        """同步登录：优先恢复持久化会话 → 否则读凭据登录 → 处理 2FA → 持久化 cookie。

        顺序：
        1. ``_try_restore_session``：缓存文件存在且 cookie 有效（get_current_user
           返回 200）→ 直接建句柄，免登录免 2FA（Web 一次验证长期复用）；
        2. 缓存缺失/失效（401 等）→ ``_clear_cache`` 后走用户名密码登录（含 2FA）；
        3. 登录成功 → ``_persist_session`` 原子写 auth cookie 到缓存文件。
        """
        sdk = self._import_sdk()
        modules = sdk["modules"]
        ApiException = sdk["ApiException"]
        UpdateUserRequest = sdk["UpdateUserRequest"]
        TwoFactorAuthCode = sdk["TwoFactorAuthCode"]
        TwoFactorEmailCode = sdk["TwoFactorEmailCode"]

        # P2-2 步骤 1：尝试用持久化 cookie 恢复会话（免登录免 2FA）
        if self._try_restore_session(sdk):
            return

        username = os.environ.get(self._username_env, "").strip()
        password = os.environ.get(self._password_env, "").strip()
        api_key = os.environ.get(self._api_key_env, "").strip()

        if not username or not password:
            raise VRCAPIError(
                f"VRChat API 未配置凭据：请在 .env 设置环境变量 {self._username_env} 与 "
                f"{self._password_env}（用户名密码登录；或设置 {self._api_key_env}）"
            )

        configuration = sdk["vrchatapi"].Configuration(
            host=DEFAULT_API_HOST,
            username=username,
            password=password,
        )
        if api_key:
            # 备用：显式配置 API key（SDK 认证头 X-Api-Key）
            configuration.api_key["auth"] = api_key

        api_client = sdk["vrchatapi"].ApiClient(configuration)
        # VRChat WAF 403 修复：SDK 1.20.7 的 Configuration 构造不接受 user_agent
        # 参数，UA 需设置在 ApiClient 层（默认 'vrchatapi-py' 不含应用信息被 WAF 拦截）。
        api_client.user_agent = DEFAULT_USER_AGENT

        handles = self._build_handles(modules, api_client)
        auth_api = handles["authentication"]
        try:
            current_user = auth_api.get_current_user()
        except ApiException as exc:
            if not self._is_two_factor_required(exc):
                raise
            code = self._resolve_2fa_code(exc)
            self._verify_2fa(auth_api, code, exc, TwoFactorAuthCode, TwoFactorEmailCode)
            current_user = auth_api.get_current_user()

        self._install_session(api_client, handles, current_user, UpdateUserRequest, username)
        # P2-2 步骤 3：登录成功 → 持久化 auth cookie（重启免登录复用）
        self._persist_session()
        logger.info(
            "VRChat 登录成功: %s（%d 类 API 句柄可用）",
            getattr(current_user, "displayName", None) or username,
            len(handles),
        )

    def _build_handles(self, modules: dict[str, Any], api_client: Any) -> dict[str, Any]:
        """按 HANDLE_MODULES 创建 19 类 API 句柄（缺模块跳过；authentication 必存在）。

        登录与会话恢复共用；authentication 句柄缺失 → 抛 VRCAPIError（SDK 不完整）。
        """
        def _make_handle(handle: str, module_name: str, class_name: str) -> Any:
            mod = modules.get(handle)
            if mod is None:
                return None
            api_cls = getattr(mod, class_name, None)
            if api_cls is None:
                return None
            return api_cls(api_client)

        auth_api = _make_handle("authentication", "authentication_api", "AuthenticationApi")
        if auth_api is None:
            raise VRCAPIError("vrchatapi 缺少 AuthenticationApi（SDK 安装不完整）")
        handles: dict[str, Any] = {"authentication": auth_api}
        for handle, module_name, class_name in HANDLE_MODULES:
            if handle == "authentication":
                continue
            h = _make_handle(handle, module_name, class_name)
            if h is not None:
                handles[handle] = h
        return handles

    def _install_session(
        self,
        api_client: Any,
        handles: dict[str, Any],
        current_user: Any,
        update_user_request_cls: Any,
        username: str,
    ) -> None:
        """把登录/恢复得到的 SDK 句柄与当前用户安装到客户端（两条路径共用）。"""
        self._api_client = api_client
        self._handles = handles
        self._auth_api = handles.get("authentication")
        self._avatars_api = handles.get("avatars")
        self._friends_api = handles.get("friends")
        self._worlds_api = handles.get("worlds")
        self._users_api = handles.get("users")
        self._update_user_request_cls = update_user_request_cls
        self._current_user = current_user

    # ------------------------------------------------------------------
    # 会话持久化（P2-2：auth cookie 复用，Web 一次验证长期免登录）
    # ------------------------------------------------------------------
    def _try_restore_session(self, sdk: dict) -> bool:
        """尝试用持久化 auth cookie 恢复会话；成功返回 True（免登录免 2FA）。

        - 缓存缺失/损坏 → 返回 False（走用户名密码登录）；
        - cookie 无效（get_current_user 抛 401 等）→ 清缓存 + 返回 False；
        - 恢复成功 → 安装句柄 + 记录当前用户。
        """
        cache_path = self._auth_cache_path
        if cache_path is None:
            return False
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        entries = data.get("cookies") if isinstance(data, dict) else None
        if not entries:
            return False
        try:
            modules = sdk["modules"]
            configuration = sdk["vrchatapi"].Configuration(host=DEFAULT_API_HOST)
            api_client = sdk["vrchatapi"].ApiClient(configuration)
            api_client.user_agent = DEFAULT_USER_AGENT
            self._restore_cookie_jar(api_client, entries)
            handles = self._build_handles(modules, api_client)
            current_user = handles["authentication"].get_current_user()
        except Exception as exc:  # noqa: BLE001 - 任何恢复失败都回退用户名密码登录
            logger.warning("持久化会话不可用（%s），清除缓存后重新登录", exc)
            self._clear_cache()
            return False
        self._install_session(api_client, handles, current_user, sdk["UpdateUserRequest"], "")
        logger.info(
            "VRChat 会话恢复（cookie 复用，免登录）: %s（%d 类 API 句柄可用）",
            getattr(current_user, "displayName", None) or "?",
            len(handles),
        )
        return True

    def _persist_session(self) -> None:
        """把当前 SDK 会话的 auth cookie 原子写入缓存文件（mkstemp + os.replace）。"""
        cache_path = self._auth_cache_path
        api_client = self._api_client
        if cache_path is None or api_client is None:
            return
        entries = self._extract_cookie_entries(api_client)
        if not entries:
            logger.debug("登录成功但无 cookie 可持久化（跳过写缓存）")
            return
        payload = {
            "version": _AUTH_CACHE_VERSION,
            "saved_at": round(time.time(), 3),
            "cookies": entries,
        }
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".vrchat_auth.", suffix=".tmp", dir=str(cache_path.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False, indent=2)
                os.replace(tmp, cache_path)
            finally:
                try:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
                except OSError:
                    pass
            logger.info("VRChat 登录会话已持久化: %s", cache_path)
        except OSError as exc:
            logger.warning("VRChat 会话持久化失败（不影响本次登录）: %s", exc)

    def _clear_cache(self) -> None:
        """删除会话缓存文件（cookie 失效/登出时调用；幂等）。"""
        cache_path = self._auth_cache_path
        if cache_path is None:
            return
        try:
            if cache_path.exists():
                cache_path.unlink()
                logger.info("VRChat 会话缓存已清除: %s", cache_path)
        except OSError:
            pass

    @staticmethod
    def _extract_cookie_entries(api_client: Any) -> list[dict[str, Any]]:
        """从 SDK ApiClient 提取可持久化的 cookie 条目（真实 SDK：rest_client.cookie_jar）。

        优先遍历 ``rest_client.cookie_jar``（http.cookiejar.CookieJar，登录后由
        SDK 自动维护）；兜底 ``api_client.cookie`` 原始字符串（如 "auth=xxx"）。
        """
        entries: list[dict[str, Any]] = []
        rest = getattr(api_client, "rest_client", None)
        jar = getattr(rest, "cookie_jar", None)
        if jar is not None:
            try:
                for c in jar:
                    entries.append(
                        {
                            "version": getattr(c, "version", 0),
                            "name": getattr(c, "name", ""),
                            "value": getattr(c, "value", ""),
                            "port": getattr(c, "port", None),
                            "port_specified": bool(getattr(c, "port_specified", False)),
                            "domain": getattr(c, "domain", ""),
                            "domain_specified": bool(getattr(c, "domain_specified", False)),
                            "domain_initial_dot": bool(getattr(c, "domain_initial_dot", False)),
                            "path": getattr(c, "path", "/") or "/",
                            "path_specified": bool(getattr(c, "path_specified", True)),
                            "secure": bool(getattr(c, "secure", False)),
                            "expires": getattr(c, "expires", None),
                            "discard": bool(getattr(c, "discard", False)),
                            "comment": getattr(c, "comment", None),
                            "comment_url": getattr(c, "comment_url", None),
                            "rest": dict(getattr(c, "_rest", {}) or getattr(c, "rest", {}) or {}),
                            "rfc2109": bool(getattr(c, "rfc2109", False)),
                        }
                    )
            except Exception:
                logger.debug("提取 cookie_jar 失败（忽略）", exc_info=True)
        if not entries:
            raw = getattr(api_client, "cookie", None)
            if isinstance(raw, str) and raw.strip():
                name, _, value = raw.partition("=")
                entries.append(
                    {
                        "version": 0,
                        "name": (name.strip() or "auth"),
                        "value": value.strip(),
                        "port": None,
                        "port_specified": False,
                        "domain": "",
                        "domain_specified": False,
                        "domain_initial_dot": False,
                        "path": "/",
                        "path_specified": True,
                        "secure": False,
                        "expires": None,
                        "discard": False,
                        "comment": None,
                        "comment_url": None,
                        "rest": {},
                        "rfc2109": False,
                    }
                )
        return entries

    def _restore_cookie_jar(self, api_client: Any, entries: list[dict[str, Any]]) -> None:
        """把持久化 cookie 恢复到 SDK ApiClient（cookie_jar + api_client.cookie 双保险）。

        - 带 domain 的条目：重建 http.cookiejar.Cookie 写入 ``rest_client.cookie_jar``
          （SDK 每次请求经 add_cookie_header 自动携带）；
        - 无 domain 的条目（原始字符串兜底）：设置 ``api_client.cookie``（__call_api
          每次请求带 Cookie 头）。
        """
        rest = getattr(api_client, "rest_client", None)
        jar = getattr(rest, "cookie_jar", None)
        auth_value: str | None = None
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "") or "")
            value = str(entry.get("value", "") or "")
            if not name:
                continue
            if name == "auth" and value:
                auth_value = value
            domain = str(entry.get("domain", "") or "")
            if jar is not None and domain:
                try:
                    cookie = Cookie(
                        version=int(entry.get("version", 0) or 0),
                        name=name,
                        value=value,
                        port=entry.get("port"),
                        port_specified=bool(entry.get("port_specified", False)),
                        domain=domain,
                        domain_specified=bool(entry.get("domain_specified", False)),
                        domain_initial_dot=bool(entry.get("domain_initial_dot", False)),
                        path=str(entry.get("path", "/") or "/"),
                        path_specified=bool(entry.get("path_specified", True)),
                        secure=bool(entry.get("secure", False)),
                        expires=entry.get("expires"),
                        discard=bool(entry.get("discard", False)),
                        comment=entry.get("comment"),
                        comment_url=entry.get("comment_url"),
                        rest=dict(entry.get("rest", {}) or {}),
                        rfc2109=bool(entry.get("rfc2109", False)),
                    )
                    jar.set_cookie(cookie)
                except Exception:
                    logger.debug("恢复 cookie %s 失败（忽略）", name, exc_info=True)
            elif hasattr(api_client, "cookie"):
                # 无 domain（原始字符串兜底）：走 api_client.cookie 通道
                try:
                    api_client.cookie = f"{name}={value}"
                except Exception:
                    pass
        # 双保险：auth cookie 同时设置到 api_client.cookie（即使 jar 已恢复也带上）
        if auth_value and hasattr(api_client, "cookie"):
            try:
                api_client.cookie = f"auth={auth_value}"
            except Exception:
                pass

    @staticmethod
    def _import_sdk() -> dict:
        """惰性 import vrchatapi（缺包抛 VRCAPIError，提示安装）。

        返回 dict：vrchatapi 包 / 可用 API 模块映射 / 异常与模型类。
        对缺少的模块容错（测试替身只注入 5 类模块时其余句柄不可用，不阻断登录）。
        """
        try:
            import vrchatapi
            import vrchatapi.api as api_pkg
            from vrchatapi.exceptions import ApiException
            from vrchatapi.models.two_factor_auth_code import TwoFactorAuthCode
            from vrchatapi.models.two_factor_email_code import TwoFactorEmailCode
            from vrchatapi.models.update_user_request import UpdateUserRequest
        except ImportError as exc:
            raise VRCAPIError("vrchatapi 未安装（pip install vrchatapi）") from exc
        modules: dict[str, Any] = {}
        for handle, module_name, _ in HANDLE_MODULES:
            try:
                modules[handle] = getattr(api_pkg, module_name)
            except AttributeError:
                logger.warning("vrchatapi 缺少 API 模块 %s（句柄 %s 不可用）", module_name, handle)
        return {
            "vrchatapi": vrchatapi,
            "modules": modules,
            "ApiException": ApiException,
            "UpdateUserRequest": UpdateUserRequest,
            "TwoFactorAuthCode": TwoFactorAuthCode,
            "TwoFactorEmailCode": TwoFactorEmailCode,
        }

    # ------------------------------------------------------------------
    # 2FA 处理
    # ------------------------------------------------------------------
    @staticmethod
    def _is_two_factor_required(exc: Exception) -> bool:
        """判断登录异常是否为"需要两步验证"（SDK 对 2FA 返回 status=200 异常）。"""
        status = getattr(exc, "status", None)
        body = getattr(exc, "body", None) or ""
        return status == 200 and "requiresTwoFactorAuth" in str(body)

    @staticmethod
    def _extract_2fa_methods(exc: Exception) -> list[str]:
        """从登录异常 body 提取可用 2FA 方式（如 ["emailOtp", "totp"]）。"""
        body = getattr(exc, "body", None) or ""
        try:
            data = json.loads(body) if isinstance(body, str) else body
        except (TypeError, ValueError):
            return []
        if not isinstance(data, dict):
            return []
        err = data.get("error")
        if not isinstance(err, dict):
            return []
        methods = err.get("requiresTwoFactorAuth")
        if isinstance(methods, list):
            return [str(m) for m in methods if str(m).strip()]
        return []

    def _resolve_2fa_code(self, exc: Exception) -> str:
        """通过 two_factor_code_provider 回调获取 2FA 验证码。"""
        methods = self._extract_2fa_methods(exc)
        if self._two_factor_code_provider is None:
            raise VRCAPIError(
                "VRChat 账号需要两步验证（2FA），但未配置 two_factor_code_provider 回调；"
                f"可用方式: {methods or ['totp', 'emailOtp']}；请提供回调或改用 API key 登录"
            )
        try:
            code = self._two_factor_code_provider(methods)
        except Exception as exc2:
            raise VRCAPIError(f"2FA 验证码回调执行失败: {exc2}") from exc2
        code = str(code or "").strip()
        if not code:
            raise VRCAPIError("2FA 验证码为空（two_factor_code_provider 返回空）")
        return code

    def _verify_2fa(
        self,
        auth_api: Any,
        code: str,
        exc: Exception,
        two_factor_auth_code_cls: Any,
        two_factor_email_code_cls: Any,
    ) -> None:
        """提交 2FA 验证码：优先 TOTP，否则按返回方式列表尝试。

        SDK 1.20.7 的方法与入参模型（旧版 verify2fa_totp / verify2fa 已移除）：
        - TOTP：``verify2_fa(TwoFactorAuthCode(code=...))``；
        - 邮箱验证码：``verify2_fa_email_code(TwoFactorEmailCode(code=...))``。
        """
        methods = self._extract_2fa_methods(exc)
        if "emailOtp" in methods and "totp" not in methods:
            auth_api.verify2_fa_email_code(two_factor_email_code_cls(code=code))
        else:
            # TOTP 优先（含"方式未知"兜底，最常见的 2FA 类型）
            auth_api.verify2_fa(two_factor_auth_code_cls(code=code))
