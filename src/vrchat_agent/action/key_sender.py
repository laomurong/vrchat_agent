"""向 VRChat 窗口注入真实键盘按键（Win32 PostMessage）。

【为什么需要这个模块】
VRChat 官方 OSC 输入白名单（``/input/*``）**从未开放** ``Crouch`` 与 ``Prone`` ——
桌面客户端的 C（蹲下）/ Z（趴下）只是客户端本地键位绑定，不存在对应 OSC 地址。
向 ``/input/Crouch`` 发 UDP 包不会报任何错（UDP 无回执、桥接也认为发送成功），
但 VRChat 端会静默丢弃，表现为"全链路通、动作就是不生效"——这正是
`pose:crouch` / `pose:prone` 长期失效的根因。

因此本模块绕开 OSC，用 Win32 ``PostMessage`` 把 ``WM_KEYDOWN`` / ``WM_KEYUP``
直接投递到 VRChat 窗口的消息队列，模拟"按下 → 保持 duration 秒 → 抬起"。

【设计约束】
- 纯 Python（``ctypes`` + user32），不引入新的 Node / 第三方依赖；
- 非 Windows 平台、找不到 VRChat 窗口、按键名非法时一律 ``logger.warning``
  并返回 ``False``，**绝不抛异常**——按键注入失败不应打断整条动作链；
- ``PostMessage`` / ``find_vrchat_window`` 均为**模块级符号**，调用点通过模块
  全局名解析，便于测试用替身注入并断言 KEYDOWN → sleep → KEYUP 时序。

VK 速查：C = 0x43（蹲下）、Z = 0x5A（趴下）。
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import sys
import time

logger = logging.getLogger("vrchat_agent.action.key_sender")

# ---------------------------------------------------------------------------
# Win32 常量与平台探测
# ---------------------------------------------------------------------------
WM_KEYDOWN: int = 0x0100
WM_KEYUP: int = 0x0101

VK_C: int = 0x43
VK_Z: int = 0x5A

#: 默认按住时长（秒）。C 为"按住蹲下、松开站起"，Z 为切换型（短按即可）。
DEFAULT_HOLD_S: float = 0.6

#: VRChat（Unity 桌面客户端）主窗口类名。
VRCHAT_WINDOW_CLASS: str = "UnityWndClass"
#: VRChat 主窗口标题关键字（小写比较）。
VRCHAT_TITLE_KEYWORD: str = "vrchat"

IS_WINDOWS: bool = sys.platform.startswith("win")

if IS_WINDOWS:  # pragma: no cover - 平台分支，非 Windows CI 不覆盖
    from ctypes import wintypes as _wintypes

    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, _wintypes.HWND, _wintypes.LPARAM)
else:  # pragma: no cover - 非 Windows 上仅提供降级占位，保证 import 不炸
    _wintypes = None
    _user32 = None
    _WNDENUMPROC = None


# ---------------------------------------------------------------------------
# 按键名 → 虚拟键码
# ---------------------------------------------------------------------------
#: 具名功能键（大小写不敏感）。字母 a-z / 数字 0-9 由 resolve_vk 直接换算。
NAMED_KEYS: dict[str, int] = {
    "space": 0x20,
    "enter": 0x0D,
    "return": 0x0D,
    "esc": 0x1B,
    "escape": 0x1B,
    "tab": 0x09,
    "shift": 0x10,
    "ctrl": 0x11,
    "control": 0x11,
    "alt": 0x12,
    "up": 0x26,
    "down": 0x28,
    "left": 0x25,
    "right": 0x27,
}


def resolve_vk(key: str | int) -> int:
    """把按键标识解析为 Win32 虚拟键码（VK）。

    支持三种形态：
    - ``int``：直接视为 VK（如 ``0x43``）；
    - 单字符 ``str``：字母 a-z / 数字 0-9（大小写不敏感，VK 即大写 ASCII）；
    - 具名键 ``str``：见 :data:`NAMED_KEYS`（如 ``"space"``）。

    Args:
        key: 按键标识。

    Returns:
        虚拟键码整数。

    Raises:
        ValueError: 无法解析的按键标识（调用方负责降级为 warning）。
    """
    if isinstance(key, bool):  # bool 是 int 子类，显式拒绝避免 True → VK 1
        raise ValueError(f"非法按键标识: {key!r}")
    if isinstance(key, int):
        return int(key)
    if not isinstance(key, str):
        raise ValueError(f"非法按键标识: {key!r}")
    name = key.strip().lower()
    if not name:
        raise ValueError("按键标识为空")
    if name in NAMED_KEYS:
        return NAMED_KEYS[name]
    if len(name) == 1 and (name.isalpha() or name.isdigit()):
        return ord(name.upper())
    raise ValueError(f"未知按键名: {key!r}")


# ---------------------------------------------------------------------------
# Win32 薄封装（模块级函数，便于测试替身注入）
# ---------------------------------------------------------------------------
def PostMessage(hwnd: int, msg: int, wparam: int, lparam: int) -> bool:  # noqa: N802 - 对齐 Win32 命名
    """向窗口消息队列投递一条消息（不等待处理结果）。

    Args:
        hwnd: 目标窗口句柄。
        msg: 消息类型（``WM_KEYDOWN`` / ``WM_KEYUP``）。
        wparam: 消息参数一（此处为虚拟键码）。
        lparam: 消息参数二（此处固定 0）。

    Returns:
        投递成功返回 True；非 Windows 平台或 Win32 调用失败返回 False。
    """
    if _user32 is None:
        logger.warning("非 Windows 平台，按键注入不可用（msg=0x%04X vk=0x%02X）", msg, wparam)
        return False
    try:
        ok = _user32.PostMessageW(
            _wintypes.HWND(hwnd),
            _wintypes.UINT(msg),
            _wintypes.WPARAM(wparam),
            _wintypes.LPARAM(lparam),
        )
    except Exception as exc:  # pragma: no cover - ctypes 层异常极少见
        logger.warning("PostMessage 调用异常: %s", exc)
        return False
    if not ok:
        logger.warning(
            "PostMessage 投递失败（hwnd=%s msg=0x%04X vk=0x%02X err=%s）",
            hwnd,
            msg,
            wparam,
            ctypes.get_last_error(),
        )
        return False
    return True


def _window_text(hwnd) -> str:
    """读取窗口标题（失败返回空串）。"""
    if _user32 is None:  # pragma: no cover - 非 Windows
        return ""
    buf = ctypes.create_unicode_buffer(512)
    _user32.GetWindowTextW(hwnd, buf, 512)
    return buf.value or ""


def _window_class(hwnd) -> str:
    """读取窗口类名（失败返回空串）。"""
    if _user32 is None:  # pragma: no cover - 非 Windows
        return ""
    buf = ctypes.create_unicode_buffer(256)
    _user32.GetClassNameW(hwnd, buf, 256)
    return buf.value or ""


def find_vrchat_window() -> int | None:
    """枚举顶层窗口，找出 VRChat 主窗口句柄。

    优先级：标题含 ``VRChat`` 的可见窗口 > 类名为 ``UnityWndClass`` 的可见窗口
    （后者是兜底：VRChat 是 Unity 客户端，但同类窗口也可能属于其他 Unity 程序）。

    Returns:
        找到返回 HWND 整数；非 Windows 平台或未找到返回 None。
    """
    if _user32 is None or _WNDENUMPROC is None:
        return None
    by_title: list[int] = []
    by_class: list[int] = []

    @_WNDENUMPROC
    def _callback(hwnd, _lparam) -> bool:
        try:
            if not _user32.IsWindowVisible(hwnd):
                return True
            title = _window_text(hwnd)
            if VRCHAT_TITLE_KEYWORD in title.lower():
                by_title.append(int(hwnd))
                return False  # 标题命中即可停止枚举
            if _window_class(hwnd) == VRCHAT_WINDOW_CLASS:
                by_class.append(int(hwnd))
        except Exception:  # pragma: no cover - 单个窗口读取失败不应中断枚举
            logger.debug("枚举窗口时读取失败，跳过", exc_info=True)
        return True

    try:
        _user32.EnumWindows(_callback, 0)
    except Exception as exc:  # pragma: no cover - ctypes 层异常极少见
        logger.warning("枚举窗口失败: %s", exc)
        return None
    if by_title:
        return by_title[0]
    if by_class:
        return by_class[0]
    return None


# ---------------------------------------------------------------------------
# 门面
# ---------------------------------------------------------------------------
class KeySender:
    """向 VRChat 窗口注入按键的门面（无状态，全部为类方法/静态方法）。

    典型用法（由 :class:`~vrchat_agent.action.mcp_actuator.MCPOSCActuator`
    在遇到 ``tool: send_key`` 的映射命令时调用）::

        await KeySender.tap_async("c", 0.6)   # 蹲下
        await KeySender.tap_async("z", 0.6)   # 趴下
    """

    @staticmethod
    def find_window() -> int | None:
        """查找 VRChat 窗口句柄（薄转发，便于替身覆盖）。"""
        return find_vrchat_window()

    @staticmethod
    async def tap_async(key: str | int, duration: float = DEFAULT_HOLD_S) -> bool:
        """异步"按下 → 保持 duration 秒 → 抬起"。

        Args:
            key: 按键标识（``"c"`` / ``"z"`` / VK 整数，见 :func:`resolve_vk`）。
            duration: 按住时长（秒，负值按 0 处理）。

        Returns:
            全流程成功返回 True；任一环节失败（平台不支持 / 找不到窗口 /
            按键名非法 / 投递失败）返回 False，且只记 warning 不抛异常。
        """
        try:
            vk = resolve_vk(key)
        except ValueError as exc:
            logger.warning("按键注入跳过：%s", exc)
            return False
        hwnd = find_vrchat_window()
        if hwnd is None:
            logger.warning("未找到 VRChat 窗口，按键注入跳过（key=%s）", key)
            return False
        hold = max(0.0, float(duration))
        if not PostMessage(hwnd, WM_KEYDOWN, vk, 0):
            return False
        try:
            await asyncio.sleep(hold)
        finally:
            # 无论 sleep 是否被取消，都必须把键抬起来——否则 VRChat 里会一直保持按下。
            PostMessage(hwnd, WM_KEYUP, vk, 0)
        return True

    @staticmethod
    def tap(key: str | int, duration: float = DEFAULT_HOLD_S) -> bool:
        """同步版 :meth:`tap_async`（供非 asyncio 上下文/脚本调用）。

        Args:
            key: 按键标识。
            duration: 按住时长（秒）。

        Returns:
            与 :meth:`tap_async` 语义一致。
        """
        try:
            vk = resolve_vk(key)
        except ValueError as exc:
            logger.warning("按键注入跳过：%s", exc)
            return False
        hwnd = find_vrchat_window()
        if hwnd is None:
            logger.warning("未找到 VRChat 窗口，按键注入跳过（key=%s）", key)
            return False
        hold = max(0.0, float(duration))
        if not PostMessage(hwnd, WM_KEYDOWN, vk, 0):
            return False
        try:
            time.sleep(hold)
        finally:
            PostMessage(hwnd, WM_KEYUP, vk, 0)
        return True
