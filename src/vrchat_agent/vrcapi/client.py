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
import json
import logging
import os
import time
from collections.abc import Callable
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
    ) -> None:
        self._username_env: str = (username_env or DEFAULT_USERNAME_ENV).strip() or DEFAULT_USERNAME_ENV
        self._password_env: str = (password_env or DEFAULT_PASSWORD_ENV).strip() or DEFAULT_PASSWORD_ENV
        self._api_key_env: str = (api_key_env or DEFAULT_API_KEY_ENV).strip() or DEFAULT_API_KEY_ENV
        self._two_factor_code_provider: Callable[[list[str]], str] | None = two_factor_code_provider
        self._limiter = RateLimiter(rate_limit_seconds, clock=clock)
        self._auth_lock = asyncio.Lock()
        self._authed: bool = False

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
        """确保已登录（首次调用触发登录 + 2FA；已登录直接返回）。"""
        async with self._auth_lock:
            if self._authed:
                return
            try:
                await asyncio.to_thread(self._login_sync)
            except VRCAPIError:
                raise
            except Exception as exc:
                raise VRCAPIError(f"VRChat 登录失败: {exc}") from exc
            self._authed = True
            logger.info("VRChat 登录完成")

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
        """同步登录：读凭据 → 配置 SDK → 登录 → 处理 2FA → 缓存全部 API 句柄。"""
        sdk = self._import_sdk()
        modules = sdk["modules"]
        ApiException = sdk["ApiException"]
        UpdateUserRequest = sdk["UpdateUserRequest"]
        TwoFactorAuthCode = sdk["TwoFactorAuthCode"]
        TwoFactorEmailCode = sdk["TwoFactorEmailCode"]

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
        try:
            current_user = auth_api.get_current_user()
        except ApiException as exc:
            if not self._is_two_factor_required(exc):
                raise
            code = self._resolve_2fa_code(exc)
            self._verify_2fa(auth_api, code, exc, TwoFactorAuthCode, TwoFactorEmailCode)
            current_user = auth_api.get_current_user()

        # 19 类句柄全部创建（缺模块的跳过；真实 SDK 全量）
        handles: dict[str, Any] = {"authentication": auth_api}
        for handle, module_name, class_name in HANDLE_MODULES:
            if handle == "authentication":
                continue
            h = _make_handle(handle, module_name, class_name)
            if h is not None:
                handles[handle] = h

        self._api_client = api_client
        self._handles = handles
        self._auth_api = auth_api
        self._avatars_api = handles.get("avatars")
        self._friends_api = handles.get("friends")
        self._worlds_api = handles.get("worlds")
        self._users_api = handles.get("users")
        self._update_user_request_cls = UpdateUserRequest
        self._current_user = current_user
        logger.info(
            "VRChat 登录成功: %s（%d 类 API 句柄可用）",
            getattr(current_user, "displayName", None) or username,
            len(handles),
        )

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
