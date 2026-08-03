"""D 动作控制：MCPOSCActuator（vrchat-mcp-osc 客户端实现，D-01~D-03）。

通过 MCPClient（stdio）调用 OSC 桥接工具；ActionMapper 负责语义→工具映射。

自动复位（问题 1：动作执行后不会自动停/复位）：
- server.js 语义确认：set_emote_parameter / set_avatar_parameter 写入 VRChat
  参数后是**持续状态**（无自动复原），而 move_avatar / look_direction 内部
  "按住 duration 秒自动松开"、jump 瞬时——因此只需对 emote/gesture 这类
  "持续型动作"做延迟自动复位；
- 实现：execute() 成功执行后，若映射条目配置了 reset 规格，则以 asyncio 后台
  任务延迟 delay 秒调用复位工具（复位 tool/args 从映射表读取，灵活可配）；
- 连续触发同一动作 → 取消旧复位任务、重建（复位计时重置）；
- 急停 reset() → 取消所有未执行复位任务 + 执行映射表急停复位命令
  （legacy reset 工具未在 server.js 注册，改用映射表 emergency_reset，
   不再 WARNING 刷屏）；
- shutdown() → Agent 关闭时执行完整急停复位（F-01）：仅取消残留复位任务
  会让"取消"被误当成"已复位"——若 Ctrl-C 落在动作保持窗口内，VRChat 里的
  模型会永远保持该动作/按键。因此关闭路径必须真正把复位命令发出去；
- 急停复位整体时间预算（Y-01）：急停复位是十几条**串行**网络命令，每条虽有
  bridge.timeout 兜底，但缺少整体预算——桥接"假死"（进程活着却不响应 OSC）
  时逐条超时会线性累加，Ctrl-C 最坏要等 N×timeout 才退出。故在命令循环外再
  套一层 `_reset_budget` 总预算，超时即截断并 WARNING，绝不 raise、绝不卡死
  （复位命令全是绝对赋值 0、幂等，截断远优于卡死）。
"""

from __future__ import annotations

import asyncio
import logging

from ..brain.schema import Action
from ..bridge.mcp_client import MCPClient
from .action_mapper import ActionMapper
from .action_schema import OSCCommand, ResetSpec
from .actuator import Actuator, ActuatorError
from .key_sender import DEFAULT_HOLD_S, KeySender

logger = logging.getLogger("vrchat_agent.action.actuator")

#: 伪工具名：映射表里 `tool: send_key` 的命令**不走 MCP**，改由 KeySender 直接向
#: VRChat 窗口注入 Win32 按键（VRChat 官方 OSC 白名单没有 Crouch/Prone，见
#: action/key_sender.py 模块文档）。
KEY_TOOL: str = "send_key"

# 急停复位命令序列的**整体**发送时间预算（秒，Y-01）。
# 正常路径实测约 0.3s 发完全部命令，余量充足；桥接假死时用于截断线性累加的
# 逐条超时，保证 Ctrl-C / 急停能在预算内返回。
DEFAULT_RESET_BUDGET: float = 5.0


class MCPOSCActuator(Actuator):
    """MCP OSC 动作执行器（含持续型动作自动复位，问题 1）。"""

    def __init__(
        self,
        client: MCPClient,
        mapper: ActionMapper | None = None,
        auto_reset: bool = True,
        reset_budget: float = DEFAULT_RESET_BUDGET,
    ) -> None:
        self._client = client
        self._mapper = mapper or ActionMapper()
        self._auto_reset = auto_reset
        # 急停复位命令循环的整体时间预算（Y-01）；<=0 视为不限预算。
        self._reset_budget: float = reset_budget
        # 自动复位后台任务表：key=映射查找键 kind:name（F-03）。
        # 用 kind 做键会让"同 kind 不同通道"互相取消（如 gesture:peace_left 与
        # gesture:peace_right），先触发的那只手永远不复位；改用 resolve_key 分槽。
        self._reset_tasks: dict[str, asyncio.Task] = {}

    @staticmethod
    def _extract_tool_error(result) -> str:
        """从工具调用结果提取错误文本（server.js 失败时返回 isError=true 而非抛异常）。

        mcp SDK 的 CallToolResult 有 isError/content 字段；测试替身可能是 dict。
        兼容两种形态：返回可读错误文本，无错误返回空串。
        """
        if result is None:
            return ""
        contents = None
        if hasattr(result, "isError") and result.isError:  # mcp SDK CallToolResult
            contents = getattr(result, "content", None) or []
        elif isinstance(result, dict):
            if result.get("isError"):
                contents = result.get("content") or []
            else:
                err = result.get("error")
                return str(err) if err else ""
        if contents is None:
            return ""
        texts: list[str] = []
        for c in contents:
            if isinstance(c, dict):
                texts.append(c.get("text", ""))
            elif hasattr(c, "text"):
                texts.append(str(c.text))
            else:
                texts.append(str(c))
        return " | ".join(t for t in texts if t).strip()

    async def _call_tool_safe(self, tool: str, args: dict) -> None:
        """调用 OSC 工具并记录错误（复位/急停路径：异常与 isError 均不抛出）。"""
        try:
            result = await self._client.call_tool(tool, args)
        except Exception as exc:
            logger.warning("OSC 工具调用失败 %s: %s", tool, exc)
            return
        err_text = self._extract_tool_error(result)
        if err_text:
            logger.warning("OSC 工具调用返回错误: %s -> %s", tool, err_text)
        else:
            logger.debug("OSC 工具调用成功: %s -> %s", tool, result)

    async def execute(self, action: Action) -> None:
        """执行一条语义动作：映射 → 顺序调用 OSC 工具 → 排自动复位。

        映射/调用任何异常统一包装为 ActuatorError（不抛裸 KeyError 等，BUG-4）；
        失败先 WARNING 明确记录工具名（P1：错误可见性，工具名不存在/调用失败时
        不再静默）。
        """
        try:
            commands = self._mapper.map(action)
        except Exception as exc:
            logger.warning("动作映射失败 %s: %s", action.kind, exc)
            raise ActuatorError(f"动作映射失败 {action.kind}: {exc}") from exc
        for cmd in commands:
            if cmd.tool == KEY_TOOL:
                # Bug B：crouch/prone 没有官方 OSC 地址 → 走 Win32 键盘注入，
                # 不调用 MCP（send_key 不是 server.js 注册的工具）。
                await self._send_key(cmd)
                continue
            try:
                result = await self._client.call_tool(cmd.tool, cmd.args)
            except Exception as exc:
                logger.warning("OSC 工具调用失败 %s: %s", cmd.tool, exc)
                raise ActuatorError(f"OSC 工具调用失败 {cmd.tool}: {exc}") from exc
            err_text = self._extract_tool_error(result)
            if err_text:
                # 工具存在但执行失败（server.js 以 isError=true 返回错误文本），
                # 此前被当作"调用成功"静默忽略 → 用户看到模型没动却无日志。P1 修复。
                logger.warning(
                    "OSC 工具调用返回错误（工具存在但执行失败）: %s -> %s",
                    cmd.tool,
                    err_text,
                )
            else:
                logger.debug("OSC 工具调用成功: %s -> %s", cmd.tool, result)

        # 问题 1：持续型动作（emote/gesture 等映射条目配置了 reset）→ 自动复位
        if self._auto_reset:
            try:
                spec = self._mapper.get_reset_spec(action)
            except Exception as exc:
                logger.warning("读取动作复位规格失败 %s: %s", action.kind, exc)
                spec = None
            if spec is not None and spec.tools:
                # F-03：按 kind:name 分槽，避免同 kind 不同通道互相取消复位。
                self._schedule_reset(self._mapper.resolve_key(action), spec)

    async def _send_key(self, cmd: OSCCommand) -> None:
        """执行一条键盘注入命令（``tool: send_key``）。

        args 契约：``{"key": "c", "duration": 0.6}``。
        缺 key 视为映射表配置错误 → 抛 ActuatorError（配置问题必须暴露）；
        注入失败（VRChat 未启动 / 窗口找不到）只 WARNING 不抛——运行环境问题
        不应让整条动作链失败，但必须可见（避免重蹈 OSC 静默丢包的覆辙）。
        """
        args = cmd.args or {}
        key = args.get("key")
        if not key:
            logger.warning("send_key 映射缺少 key 参数: %s", args)
            raise ActuatorError(f"send_key 映射缺少 key 参数: {args}")
        try:
            duration = float(args.get("duration", DEFAULT_HOLD_S))
        except (TypeError, ValueError):
            logger.warning("send_key duration 非法（%r），回退默认 %.2fs", args.get("duration"), DEFAULT_HOLD_S)
            duration = DEFAULT_HOLD_S
        ok = await KeySender.tap_async(key, duration)
        if ok:
            logger.debug("键盘注入成功: key=%s duration=%.2fs", key, duration)
        else:
            logger.warning(
                "键盘注入未生效（VRChat 窗口未找到 / 投递失败）: key=%s duration=%.2fs",
                key,
                duration,
            )

    # ------------------------------------------------------------------
    # 自动复位调度（问题 1）
    # ------------------------------------------------------------------
    def _schedule_reset(self, key: str, spec: ResetSpec) -> None:
        """排/重置一个延迟复位任务（同 key 旧任务先取消 → 复位计时重置）。"""
        old = self._reset_tasks.get(key)
        if old is not None and not old.done():
            old.cancel()
        task = asyncio.create_task(self._run_reset(key, spec))
        self._reset_tasks[key] = task
        logger.debug("已排动作自动复位: %s (delay=%.1fs)", key, spec.delay)

    async def _run_reset(self, key: str, spec: ResetSpec) -> None:
        """延迟 delay 秒后调用复位工具；被取消/被新任务替代时静默退出。"""
        try:
            await asyncio.sleep(spec.delay)
            for cmd in spec.tools:
                await self._call_tool_safe(cmd.tool, cmd.args)
            logger.info("动作自动复位完成: %s", key)
        except asyncio.CancelledError:
            logger.debug("动作自动复位已取消: %s", key)
            raise
        finally:
            # 仅当表中仍指向自己时移除（若被新任务覆盖，新任务已在表中）。
            if self._reset_tasks.get(key) is asyncio.current_task():
                self._reset_tasks.pop(key, None)

    async def _cancel_pending_resets(self) -> None:
        """取消全部未执行复位任务（急停 / 关闭时调用）。"""
        tasks = list(self._reset_tasks.values())
        self._reset_tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _drain_reset_commands(self, commands: list[OSCCommand]) -> None:
        """串行发出全部急停复位命令，整体不超过 `_reset_budget` 秒（Y-01）。

        超预算时截断并 WARNING，**不抛异常**：调用方（急停 / shutdown）无论
        如何都要能继续往下走。`_reset_budget <= 0` 表示不限预算（旧行为）。
        """

        async def _drain() -> None:
            for cmd in commands:
                await self._call_tool_safe(cmd.tool, cmd.args)

        if self._reset_budget <= 0:
            await _drain()
            logger.info("动作已复位（%d 条急停复位命令）", len(commands))
            return
        try:
            await asyncio.wait_for(_drain(), timeout=self._reset_budget)
        except asyncio.TimeoutError:
            logger.warning("急停复位超预算被截断（桥接可能假死），已尽力发出部分复位")
        else:
            logger.info("动作已复位（%d 条急停复位命令）", len(commands))

    async def reset(self) -> None:
        """复位动作/表情（急停时调用）。

        问题 1 修复：
        - 先取消所有未执行复位任务（急停立即复位，不再等延迟）；
        - 执行映射表急停复位命令（emergency_reset / 收集 reset 条目），
          替代 server.js **未注册**的 legacy reset 工具（不再 WARNING 刷屏）；
        - 两者都没有时静默跳过（DEBUG 记录，不刷 WARNING）。

        Y-01：命令序列整体受 `_reset_budget` 约束。单条命令的超时降级仍由
        `_call_tool_safe`（内含 bridge.timeout）负责，二者是叠加的两层保护：
        单条超时防单点卡死，整体预算防"逐条超时线性累加"。超预算只截断 +
        WARNING，不抛异常——复位命令是幂等的绝对赋值 0，发出多少算多少。
        """
        await self._cancel_pending_resets()
        try:
            commands = self._mapper.get_emergency_reset_commands()
        except Exception as exc:
            logger.warning("读取急停复位命令失败: %s", exc)
            commands = []
        if commands:
            await self._drain_reset_commands(commands)
            return
        # 降级：legacy reset 工具存在时使用（旧映射/旧桥接），失败静默。
        try:
            await self._client.call_tool("reset", {})
            logger.info("动作已复位（legacy reset）")
        except Exception:
            logger.debug("无急停复位命令且 legacy reset 不可用，跳过复位")

    async def shutdown(self) -> None:
        """关闭执行器：执行完整急停复位（F-01），不残留后台任务。

        旧实现只调用 `_cancel_pending_resets()`——取消挂起任务 ≠ 已复位：
        Ctrl-C 若落在动作保持窗口内（emote/gesture/抓握尚未到 delay），
        复位命令根本没发出去，VRChat 里的模型会一直保持该状态。
        `reset()` 内部已包含 `_cancel_pending_resets()`，因此这里直接复用，
        既取消残留任务又真正把复位命令发给桥接。

        Y-01：`reset()` 的命令循环受 `_reset_budget` 整体预算约束，因此本方法
        有确定性的时间上界（约 `_reset_budget` 秒），桥接假死也不会拖住退出。
        """
        await self.reset()
        logger.debug("MCPOSCActuator 已关闭（已执行急停复位，无残留复位任务）")
