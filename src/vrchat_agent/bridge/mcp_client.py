"""D 桥接底座：MCP stdio 客户端（拉起 vrchat-mcp-osc Node 子进程）。

共享知识约定（架构文档第 8 节第 8 条）：
- MCPClient 以 stdio 子进程方式拉起 vrchat-mcp-osc；
- 启动失败或中途退出 → AgentLoop 降级为"仅语音无动作"并 ERROR 告警，不崩溃。

BugFix（降级不崩溃契约，工程师寇豆码）：
旧实现 `read, write = await stdio_client(params).__aenter__()` 只取 __aenter__ 的
返回值，**没有持有 stdio_client 返回的异步上下文管理器（async generator）引用**。
CPython 在临时 async generator 被 GC 时会通过事件循环调度 `aclose()`（表现为一个
名为 `async_generator_athrow` 的独立 task），它在**别的 task** 里退出 anyio 的
cancel scope → `RuntimeError("Attempted to exit cancel scope in a different task
than it was entered in")` → "Task exception was never retrieved" → 最终取消主循环
gather（实机崩溃根因，已用 mcp 2.0 + anyio 4.14 复现）。

修复要点：
1. `self._stdio_cm = stdio_client(params)` 持有引用，enter/exit 严格在同一 task 配对；
2. 失败清理在 start() 的 except 内同一 task 完成（不进 stop() 延迟清理）；
3. 不手动 aclose read/write stream（mcp 2.0 stdio_client 的 __aexit__ 会关流并终止
   子进程），避免双重关闭；
4. 任何清理异常吞掉并记日志，绝不外抛成未检索异常；清理失败时保留 cm 引用防 GC 二次触发。

BugFix P3（错误信息为空 → 清晰可操作，工程师寇豆码）：
- 根因：server.js 在连接 MCP stdio transport 前先 spawn relay 子进程（execPath 为
  裸命令名 'node'，靠 PATH 解析）并连接 WebSocket（失败重试 3 次 × 2s）；父进程
  PATH 不含 node 目录时 relay 假失败 → wsClient 重试 ~5.5s → Python 端
  `wait_for(initialize, 5s)` 超时 → asyncio.TimeoutError 的 str() 为空 → 空错误信息。
- 修复：① 自动把 node.exe 目录注入子进程 PATH（root cause）；② initialize 用
  更宽超时 max(timeout, 15s)；③ 失败信息按异常类型给出可操作中文提示，并附
  子进程 stderr 末尾诊断（临时文件承载，满足 anyio 对真实 fd 的要求）。
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import shutil
import tempfile
from typing import Any

logger = logging.getLogger("vrchat_agent.action.bridge")

# 初始化握手超时下限：server.js 在连接 MCP stdio transport 之前会先启动 relay
# 子进程并连接 WebSocket（失败时最多重试 3 次 × 2s），过短的超时会误报超时。
_INIT_TIMEOUT_FLOOR = 15.0

# 桥接子进程 stderr 诊断缓冲上限（字节）：仅保留末尾，用于失败时给用户可读线索。
_STDERR_TAIL_LIMIT = 4096


class _ErrlogFile:
    """子进程 stderr 重定向文件（有界 tail 读取）。

    - anyio.open_process(stderr=...) 需要真实文件描述符（fileno），纯 Python 缓冲
      对象会抛 "fileno" 错误，故用临时文件承载；
    - buffering=0：子进程直接写 OS fd，绕过 Python 缓冲，读回时所见即所得；
    - tail() 只读末尾若干字节，失败时拼进错误信息，避免内存膨胀。
    """

    def __init__(self, limit: int = _STDERR_TAIL_LIMIT) -> None:
        self._limit = limit
        self._fh = tempfile.TemporaryFile(mode="w+b", buffering=0)

    def fileno(self) -> int:
        return self._fh.fileno()

    def write(self, s) -> int:
        # anyio 通常直接写 fd，不经过 write；保留兜底以兼容其他调用路径。
        data = s.encode("utf-8", errors="replace") if isinstance(s, str) else bytes(s)
        try:
            self._fh.write(data)
        except (OSError, ValueError):
            pass
        return len(data)

    def tail(self, max_chars: int = 2000) -> str:
        try:
            self._fh.flush()
            size = os.fstat(self._fh.fileno()).st_size
            read_len = min(size, max_chars * 4)  # 多读一点，避免截断多字节字符
            self._fh.seek(max(0, size - read_len))
            raw = self._fh.read()
            return raw.decode("utf-8", errors="replace")[-max_chars:]
        except (OSError, ValueError):
            return ""

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


class MCPError(RuntimeError):
    """MCP 桥接错误。"""


class MCPClient:
    """MCP stdio 客户端：启动 / 调用工具 / 停止。"""

    def __init__(self, command: str, args: list[str] | None = None, timeout: float = 5.0) -> None:
        self._command = command
        self._args = args or []
        self._timeout = timeout
        self._session = None
        self._read = None
        self._write = None
        # BugFix：持有 stdio_client 异步上下文管理器引用（防 GC 跨 task aclose）。
        self._stdio_cm = None
        self._cm_entered = False
        self._enter_task = None
        self._started = False
        # BugFix P3：初始化握手用更宽超时（relay/WebSocket 冷启动可能 >5s），
        # 工具调用仍用配置的 timeout。见 _INIT_TIMEOUT_FLOOR。
        self._init_timeout = max(float(timeout), _INIT_TIMEOUT_FLOOR)
        self._errlog = _ErrlogFile()

    async def start(self) -> None:
        """启动 Node 桥接子进程并初始化 MCP 会话。"""
        # BugFix P1：先做命令可达性预检，给出友好报错（再 import mcp SDK，
        # 让"命令找不到"的错误优先于 SDK 缺失的错误，便于用户定位）。
        command = self._command
        if not self._is_path_like(command) and shutil.which(command) is None:
            raise MCPError(
                f"启动 MCP 桥接失败（{command}）: 未找到可执行文件。"
                f"请确认 Node 已安装并加入系统 PATH，或在 config.yaml 的 "
                f"action.bridge.command 配置 node.exe 绝对路径"
            )

        # BugFix P3：每次 start 使用全新的 stderr 诊断文件（上次失败/停止后
        # cleanup 已关闭旧句柄；幂等重试必须换新文件，否则子进程写入已关闭 fd）。
        self._errlog.close()
        self._errlog = _ErrlogFile()

        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise MCPError("mcp SDK 未安装（pip install mcp）") from exc

        params = StdioServerParameters(command=command, args=self._args, env=self._child_env(command))
        # BugFix：必须持有 stdio_client 的上下文管理器引用，并在同一 task 内完成
        # enter/exit 配对。旧代码丢弃临时 generator，GC 会以独立 task 触发 aclose，
        # 跨 task 退出 anyio cancel scope → RuntimeError → 未检索异常 → 主循环崩溃。
        #
        # BugFix P3：尽量把子进程 stderr 收集到有界缓冲（errlog），失败时拼进错误
        # 信息；对不支持 errlog 参数的 stdio_client（测试替身/旧版 SDK）回退不传。
        try:
            sig = inspect.signature(stdio_client)
            if "errlog" in sig.parameters:
                self._stdio_cm = stdio_client(params, errlog=self._errlog)
            else:
                self._stdio_cm = stdio_client(params)
        except (TypeError, ValueError):
            self._stdio_cm = stdio_client(params)
        self._enter_task = asyncio.current_task()
        try:
            read, write = await self._stdio_cm.__aenter__()
            self._cm_entered = True
            self._read, self._write = read, write
            self._session = await ClientSession(read, write).__aenter__()
            await asyncio.wait_for(self._session.initialize(), timeout=self._init_timeout)
            self._started = True
            logger.info("MCP 桥接已启动: %s %s", command, " ".join(self._args))
        except Exception as exc:
            # 同一 task 内完成清理：不跨 task 退出 cancel scope、不双重关闭、
            # 绝不留未检索异常；随后才抛出 MCPError 让上层降级。
            # 先快照 stderr 诊断（cleanup 会关闭 errlog 文件句柄）。
            tail = self._errlog.tail(max_chars=2000)
            await self._cleanup()
            raise MCPError(
                f"启动 MCP 桥接失败（{command}）: {self._format_failure(exc, tail=tail)}"
            ) from exc

    @staticmethod
    def _is_path_like(command: str) -> bool:
        """command 是否已是路径形式（含分隔符或绝对路径）。

        绝对路径（如 Windows 的 C:/... 或 Unix 的 /usr/bin/node）直接交给子进程，
        由 OS 报具体错误；仅对裸命令名（如 node）做 PATH 探测。
        """
        return bool(command) and (
            os.sep in command
            or (os.altsep is not None and os.altsep in command)
            or os.path.isabs(command)
        )

    def _child_env(self, command: str) -> dict[str, str] | None:
        """为子进程构造环境变量（BugFix P3 根因修复）。

        server.js 内部会用裸命令名 `spawn('node', [relayServerPath])` 启动 relay
        子进程（relay-server-manager.js execPath='node'，靠 PATH 解析）。当本进程用
        node.exe 绝对路径拉起 server.js 时，父进程 PATH 很可能不含 node 目录
        （node 位于 .workbuddy 托管目录）→ relay spawn ENOENT 静默失败 →
        wsClient 对 8765 重试 3 次 ≈5.5s → MCP initialize 超时（空错误信息）。

        修复：command 为路径形式时，把其所在目录注入子进程 PATH（仅当缺失时），
        使 server.js 内部的 relay 子进程能解析到同一 node。
        """
        if not self._is_path_like(command):
            return None
        node_dir = os.path.dirname(command)
        if not node_dir:
            return None
        cur_path = os.environ.get("PATH", "")
        if any(
            os.path.normcase(p) == os.path.normcase(node_dir)
            for p in cur_path.split(os.pathsep)
            if p
        ):
            return None  # 已在 PATH 中，无需覆盖
        return {"PATH": node_dir + os.pathsep + cur_path}

    def _format_failure(self, exc: Exception, tail: str = "") -> str:
        """把启动异常翻译成用户可操作的中文提示（不再出现空错误信息）。"""
        # BugFix P3：asyncio.TimeoutError（Python 3.11 起即内置 TimeoutError）的
        # str() 为空 → 旧日志出现 `...node.exe）: ` 后面无内容。必须显式分类。
        if isinstance(exc, TimeoutError):
            detail = (
                "与 VRChat OSC 桥接进程握手超时（initialize 未在限定时间内完成，"
                f"上限 {self._init_timeout:.0f}s）。\n"
                "  请依次排查：\n"
                "  1) VRChat 未开启 OSC：确认 VRChat 已启动，菜单 → OSC → 勾选 Enable（并保持在线）；\n"
                "  2) 桥接进程初始化过慢：可调大 config.yaml action.bridge.timeout（如 15.0）后重启；\n"
                "  3) 端口 8765/9000/9001 被占用或被防火墙拦截。"
            )
        elif isinstance(exc, (ConnectionError, EOFError, BrokenPipeError)):
            detail = (
                "桥接子进程启动后连接中断/提前退出（Connection closed）。\n"
                "  请依次排查：\n"
                "  1) 端口 8765（relay WebSocket）或 9000/9001（OSC）被其他程序占用；\n"
                "  2) server.js 或 relay 依赖缺失（可先手动执行 command + args 看输出）；\n"
                "  3) VRChat 未开启 OSC：确认菜单 → OSC → 勾选 Enable 后重试。"
            )
        else:
            raw = str(exc).strip()
            detail = raw if raw else f"{type(exc).__name__}（异常详情为空）"
        tail = (tail or self._errlog.tail(max_chars=2000)).strip()
        if tail:
            detail += f"\n  桥接进程输出（末尾）:\n    {tail}"
        return detail

    async def _cleanup(self) -> None:
        """关闭已打开的资源（session 先、stdio_cm 后；均在调用方 task 内执行）。

        - stdio_cm.__aexit__ 是权威清理（关流 + 终止子进程），不再手动 aclose stream；
        - 清理异常一律吞掉并记日志（绝不外抛成未检索异常）；
        - 若 cm 退出失败（如跨 task 调 stop），保留引用，避免 suspended generator 被
          GC 触发跨 task aclose 的未检索异常；进程退出时自然回收。
        """
        session, self._session = self._session, None
        if session is not None:
            try:
                await session.__aexit__(None, None, None)
            except Exception:
                logger.debug("MCP session 清理失败（忽略）", exc_info=True)

        if self._cm_entered:
            cm = self._stdio_cm
            self._cm_entered = False  # 幂等：后续 stop() 不再重复退出（双重关闭防护）
            try:
                await cm.__aexit__(None, None, None)
                self._stdio_cm = None
                self._enter_task = None
            except Exception:
                # 跨 task 或传输层退出异常：进程已尽力回收，保留引用防 GC 二次触发。
                logger.warning("MCP stdio 退出异常（已降级处理，保留引用防 GC）", exc_info=True)
        else:
            # 从未进入 cm（如 spawn 失败）：generator 已结束，无 cancel scope，可安全释放。
            self._stdio_cm = None
            self._enter_task = None

        self._read = None
        self._write = None
        # BugFix P3：关闭 stderr 诊断文件（子进程已随 cm 退出；TemporaryFile.close 幂等）。
        self._errlog.close()

    async def call_tool(self, name: str, args: dict | None = None) -> Any:
        """调用桥接暴露的 OSC 工具。"""
        if not self._started or self._session is None:
            raise MCPError("MCP 桥接未启动")
        try:
            result = await asyncio.wait_for(
                self._session.call_tool(name, arguments=args or {}), timeout=self._timeout
            )
            return result
        except asyncio.TimeoutError:
            raise MCPError(f"工具调用超时: {name}") from None
        except Exception as exc:
            raise MCPError(f"工具调用失败 {name}: {exc}") from exc

    async def list_tools(self) -> list[str]:
        """列出桥接实际暴露的 MCP 工具名（供冒烟校验/映射表对齐排查）。"""
        if not self._started or self._session is None:
            raise MCPError("MCP 桥接未启动")
        try:
            result = await asyncio.wait_for(
                self._session.list_tools(), timeout=self._timeout
            )
            return [t.name for t in result.tools]
        except asyncio.TimeoutError:
            raise MCPError("工具列表获取超时") from None
        except Exception as exc:
            raise MCPError(f"工具列表获取失败: {exc}") from exc

    async def stop(self) -> None:
        """关闭 MCP 会话与子进程（与 start() 同一 task 调用；幂等）。"""
        self._started = False
        await self._cleanup()

    @property
    def started(self) -> bool:
        return self._started
