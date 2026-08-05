"""2FA 验证码服务：Web 等待输入 + .env 兜底（P2-1 落地）。

用户实测：VRChat 账号开启两步验证后，登录返回 ``requiresTwoFactorAuth``，
但 main.py 装配 ``VRCApiClient`` 时没有传 ``two_factor_code_provider`` 回调 →
报错 "账号需要两步验证（2FA），但未配置 two_factor_code_provider 回调"。
本模块提供该回调的标准实现，两种取码通道：

1. **Web 等待输入（主通道）**：登录遇 2FA 时，把「需要验证码 + 可用方式
   （totp/emailOtp）」状态推给 Web 控制台（经 ``status_callback`` 发 SSE），
   然后**阻塞等待**用户提交验证码（带超时，默认 120s；超时报
   「2FA 验证码等待超时」）。
2. **.env 兜底**：回调先检查环境变量 ``VRC_2FA_CODE``（非空则直接用——
   适合邮箱一次性验证码，改 .env 后重启即可），未配置才走 Web 等待。

线程模型（关键约束）：
- VRChat 登录跑在 executor 线程（``VRCAPIClient.ensure_authed`` 用
  ``asyncio.to_thread``），等待**不能**用 asyncio 锁死事件循环；
- 本服务用 ``threading.Event`` + ``threading.Lock`` 保护的共享变量实现
  跨线程通知：executor 线程 ``Event.wait(timeout)`` 阻塞等待；Web 端
  （事件循环线程）提交验证码时 ``Event.set()`` 唤醒回调，登录继续。
- 状态变更经 ``status_callback``（可选）同步给 Web：main.py 装配时把它
  接到 ``EventBus.publish_threadsafe`` 发 ``vrc_2fa_status`` 事件，SSE 转发。
  回调始终在**锁外**调用（避免回调内再访问本服务时死锁）。

并发语义：
- 同一时刻只允许一个等待中的 2FA 请求（重复触发时旧等待先被取消）；
- ``submit`` 在无等待时返回 False（Web 端返回 409 清晰错误）；
- ``cancel`` 供服务关闭/测试使用（WebServer.shutdown 时取消挂起等待）。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("vrchat_agent.vrcapi.two_factor")

#: .env 兜底环境变量名（非空即直接作为 2FA 验证码使用）
DEFAULT_2FA_CODE_ENV: str = "VRC_2FA_CODE"

#: Web 等待验证码的默认超时（秒）
DEFAULT_2FA_TIMEOUT_S: float = 120.0


class TwoFactorCodeError(RuntimeError):
    """2FA 验证码获取失败（超时/为空/被取消等），由 client 包装为 VRCAPIError。"""


class TwoFactorCodeService:
    """2FA 验证码服务：``two_factor_code_provider(methods) -> str`` 的标准实现。

    用法（main.py 装配）：
        service = TwoFactorCodeService(
            timeout_s=config.vrcapi.two_factor_timeout_s,
            status_callback=lambda payload: bus.publish_threadsafe(
                Event("vrc_2fa_status", payload)),
        )
        client = VRCAPIClient(..., two_factor_code_provider=service.provide)

    ``provide`` 是 VRCAPIClient 要求的回调签名（``Callable[[list[str]], str]``），
    在登录 executor 线程内同步调用；Web 端经 ``submit`` 跨线程提交验证码。
    """

    def __init__(
        self,
        timeout_s: float = DEFAULT_2FA_TIMEOUT_S,
        code_env: str = DEFAULT_2FA_CODE_ENV,
        status_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._timeout_s: float = max(0.0, float(timeout_s))
        self._code_env: str = (code_env or DEFAULT_2FA_CODE_ENV).strip() or DEFAULT_2FA_CODE_ENV
        self._status_callback: Callable[[dict[str, Any]], None] | None = status_callback

        # 跨线程共享状态（登录 executor 线程读，Web 事件循环线程写）
        self._lock = threading.Lock()
        self._evt = threading.Event()
        self._waiting: bool = False
        self._methods: list[str] = []
        self._wait_started: float = 0.0
        self._code: str | None = None
        self._error: str | None = None

    # ------------------------------------------------------------------
    # 回调入口（VRCAPIClient 调用，登录 executor 线程内同步阻塞）
    # ------------------------------------------------------------------
    def provide(self, methods: list[str]) -> str:
        """two_factor_code_provider 回调：取 2FA 验证码。

        优先 .env 兜底（``VRC_2FA_CODE`` 非空直接用）；否则进入 Web 等待：
        推送「需要 2FA」状态 → ``threading.Event.wait(timeout)`` 阻塞等待
        用户提交 → 返回验证码。超时/空码/被取消抛 ``TwoFactorCodeError``。
        """
        env_code = os.environ.get(self._code_env, "").strip()
        if env_code:
            logger.info("使用环境变量 %s 提供的 2FA 验证码（跳过 Web 等待）", self._code_env)
            return env_code
        return self._wait_for_web_code(methods)

    # ------------------------------------------------------------------
    # Web 等待（跨线程通知核心）
    # ------------------------------------------------------------------
    def _wait_for_web_code(self, methods: list[str]) -> str:
        """置等待状态 → 阻塞等待提交 → 返回验证码（超时抛 TwoFactorCodeError）。"""
        with self._lock:
            if self._waiting:
                # 并发保护：同一时刻只允许一个等待中的 2FA 请求，旧等待先取消
                self._error = "检测到新的 2FA 请求，本次等待被取消"
                self._evt.set()
            self._code = None
            self._error = None
            self._waiting = True
            self._methods = [str(m) for m in (methods or []) if str(m).strip()]
            self._wait_started = time.monotonic()
            self._evt.clear()
            payload = self._snapshot_locked()
        self._emit_payload(payload)
        try:
            ok = self._evt.wait(timeout=self._timeout_s)
            with self._lock:
                err = self._error
                code = (self._code or "").strip()
                still_waiting = self._waiting
            if err:
                raise TwoFactorCodeError(err)
            if not ok or not still_waiting:
                raise TwoFactorCodeError(
                    "2FA 验证码等待超时"
                    f"（{self._timeout_s:.0f}s；请在 Web 控制台输入 TOTP / 邮箱验证码）"
                )
            if not code:
                raise TwoFactorCodeError("2FA 验证码为空（Web 提交了空验证码）")
            logger.info(
                "2FA 验证码已通过 Web 控制台提交（可用方式: %s）",
                self._methods or ["totp", "emailOtp"],
            )
            return code
        finally:
            with self._lock:
                self._waiting = False
                self._methods = []
                self._code = None
                self._error = None
                payload = self._snapshot_locked()
            self._emit_payload(payload)

    # ------------------------------------------------------------------
    # Web 端 API（事件循环线程调用）
    # ------------------------------------------------------------------
    def submit(self, code: str) -> bool:
        """提交验证码给等待中的回调（线程安全）。

        :param code: 用户输入的验证码（TOTP / 邮箱验证码）
        :returns: True=已通知等待中的回调；False=当前没有等待中的请求
        :raises ValueError: 验证码为空
        """
        code = str(code or "").strip()
        if not code:
            raise ValueError("验证码不能为空")
        with self._lock:
            if not self._waiting:
                return False
            self._code = code
            self._error = None
            self._evt.set()
            payload = self._snapshot_locked()
        self._emit_payload(payload)
        return True

    def cancel(self, reason: str = "2FA 验证码等待被取消") -> None:
        """取消当前等待（服务关闭/测试用）；无等待时无操作。"""
        with self._lock:
            if not self._waiting:
                return
            self._error = reason
            self._evt.set()
            payload = self._snapshot_locked()
        self._emit_payload(payload)

    # ------------------------------------------------------------------
    # 状态查询（GET /api/vrc/2fa/status 与 SSE 快照共用）
    # ------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        """当前 2FA 等待状态快照（供 Web 轮询 / SSE 事件）。"""
        with self._lock:
            return self._snapshot_locked()

    @property
    def waiting(self) -> bool:
        """当前是否有等待中的 2FA 请求（Web 端轮询用）。"""
        with self._lock:
            return self._waiting

    @property
    def timeout_s(self) -> float:
        """配置的 Web 等待超时（秒）。"""
        return self._timeout_s

    # ------------------------------------------------------------------
    # 内部（_snapshot_locked 必须在持锁状态下调用；回调在锁外执行）
    # ------------------------------------------------------------------
    def _snapshot_locked(self) -> dict[str, Any]:
        """构造状态快照（调用方必须已持有 self._lock）。"""
        elapsed = round(time.monotonic() - self._wait_started, 1) if self._waiting else 0.0
        return {
            "waiting": self._waiting,
            "methods": list(self._methods),
            "timeout_s": self._timeout_s,
            "wait_elapsed_s": elapsed,
            "wait_remaining_s": round(max(0.0, self._timeout_s - elapsed), 1) if self._waiting else 0.0,
            "code_env_configured": bool(os.environ.get(self._code_env, "").strip()),
            "submitted": self._code is not None,
        }

    def _emit_payload(self, payload: dict[str, Any]) -> None:
        """触发状态回调（锁外调用；失败仅记日志，不影响登录流程）。"""
        cb = self._status_callback
        if cb is None:
            return
        try:
            cb(payload)
        except Exception:  # noqa: BLE001 - 状态通知失败不影响登录流程
            logger.exception("2FA 状态回调异常（忽略）")
