"""D 动作控制：ActionMapper（语义动作 → OSC 工具调用映射，D-04）。

映射文件：mappings/avatar_default.yaml，格式：
    actions:
      emote:wave:
        tools:
          - tool: emote
            args: {name: Wave}
        # 可选：自动复位（问题 1）——持续型动作执行后延迟复位
        reset:
          delay: 3.0
          tool: set_emote_parameter
          args: {value: 0}
      move:forward:
        tools:
          - tool: move
            args: {direction: forward, duration: 0.8}
      chatbox:reply:
        tools:
          - tool: send_message
            args: {message: "{message}", send_immediately: true}
    # 可选：急停复位命令（问题 1）——急停时统一复位全部持续型动作；
    # 未配置时回退为"收集所有 reset 条目"。
    emergency_reset:
      tools:
        - tool: set_emote_parameter
          args: {value: 0}

查找键优先级：kind:name（params.name）→ kind（兜底）。
args 模板支持 {param} 占位符，用 action.params 动态填充（如 chatbox:reply 的 {message}）。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

from ..brain.schema import Action
from .action_schema import OSCCommand, ResetSpec

logger = logging.getLogger("vrchat_agent.action.mapper")

# VRChat chatbox 有长度限制：超出 140 字符截断并补省略号（任务 2）。
_CHATBOX_MAX_LEN = 140


def _parse_repeat(value) -> int:
    """解析条目 repeat 字段（默认 1；非法/<=1 一律视为 1，防御性）。

    背景（OSC 转向精度）：vrchat-mcp-osc 的 look_direction/move_avatar 是
    **输入级驱动**（模拟按住 LookLeft/MoveForward 等 /input/* 按钮 duration 秒），
    没有角度参数。单次 duration 受桥接 tryButtonLook 公式限制（实际按住 =
    (duration + 2) * 100 ms），要转更大角度可配置 `repeat: N`，由 map() 把整个
    tools 列表展开为 N 份顺序调用（每次"按住→松开"累计转角，间隙自动复位，
    比盲目加大 duration 更可控）。
    """
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 1
    return n if n >= 1 else 1


def _parse_reset_spec(reset_raw) -> ResetSpec:
    """把映射条目 reset 字段解析为 ResetSpec。

    支持两种形态：
    - 单工具：{delay: 3.0, tool: set_emote_parameter, args: {value: 0}}
    - 多工具：{delay: 3.0, tools: [{tool: ..., args: {...}}, ...]}
    """
    if not isinstance(reset_raw, dict):
        return ResetSpec(delay=3.0, tools=[])
    delay = float(reset_raw.get("delay", 3.0))
    tools: list[OSCCommand] = []
    raw_tools = reset_raw.get("tools")
    if isinstance(raw_tools, list):
        for item in raw_tools:
            if isinstance(item, dict) and item.get("tool"):
                tools.append(OSCCommand(tool=item["tool"], args=dict(item.get("args", {}))))
    elif reset_raw.get("tool"):
        tools.append(OSCCommand(tool=reset_raw["tool"], args=dict(reset_raw.get("args", {}))))
    return ResetSpec(delay=delay, tools=tools)


class ActionMapper:
    """语义动作 → OSCCommand 列表。"""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path: Path | None = None
        self._mapping: dict = {}
        self._raw: dict = {}
        if path:
            self.load(path)

    def load(self, path: str | Path) -> None:
        """加载映射 YAML。"""
        self._path = Path(path)
        with open(self._path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        self._raw = data
        self._mapping = data.get("actions", data) or {}
        logger.info("动作映射已加载: %s (%d 项)", self._path, len(self._mapping))

    def reload(self) -> None:
        """热重载映射文件（T05：config 热重载联动）。"""
        if self._path:
            self.load(self._path)

    def map(self, action: Action) -> list[OSCCommand]:
        """把语义动作映射为 OSCCommand 列表（找不到返回空列表）。

        查找键优先级：kind:name（params.name）→ kind（兜底）→ kind:kind（如 jump→jump:jump，
        修复 LLM 输出 {"kind": "jump", "params": {}} 时"未找到动作映射: jump"）。

        支持条目级 `repeat: N`：把该条目的 tools 列表整体展开为 N 份顺序调用
        （如 look:turn_left 配 repeat: 2 → 连续调用 look_direction 两次，累计转角；
        用于绕开桥接单次 duration 的角度上限，见 _parse_repeat 注释）。
        """
        entry = self._resolve_entry(action)
        if entry is None:
            logger.warning(
                "未找到动作映射: %s（可在映射表增加 key=%s 或 %s 条目，"
                "如 jump:jump；LLM 动作名请与 prompts 动作枚举对齐）",
                self.resolve_key(action),
                self.resolve_key(action),
                action.kind,
            )
            return []
        repeat = 1
        if isinstance(entry, dict) and "tools" in entry:
            repeat = _parse_repeat(entry.get("repeat"))
            entry = entry["tools"]
        if not isinstance(entry, list):
            entry = [entry]
        commands: list[OSCCommand] = []
        for item in entry:
            if not isinstance(item, dict):
                logger.warning("动作映射条目格式错误（跳过）: %r", item)
                continue
            tool = item.get("tool")
            if not tool:
                # 损坏条目：缺少 tool 键 → 跳过并告警，绝不抛裸 KeyError（BUG-4）
                logger.warning("动作映射条目缺少 tool 键（跳过）: %r", item)
                continue
            args = self._fill_args(item.get("args", {}), action.params)
            args = self._sanitize_args(tool, args)
            commands.append(OSCCommand(tool=tool, args=args))
        if repeat > 1 and commands:
            # 展开为 N 份独立命令（深拷贝 args，避免多份共享同一可变 dict）
            commands = [
                OSCCommand(tool=cmd.tool, args=dict(cmd.args))
                for _ in range(repeat)
                for cmd in commands
            ]
        return commands

    def _resolve_entry(self, action: Action):
        """按查找键优先级返回动作条目（无则 None）。"""
        key = self.resolve_key(action)
        entry = self._mapping.get(key)
        if entry is None:
            entry = self._mapping.get(action.kind)
        if entry is None:
            # kind:kind 兜底：jump 语义默认即 jump 自己（LLM 常省略 params.name）
            entry = self._mapping.get(f"{action.kind}:{action.kind}")
        return entry

    def get_reset_spec(self, action: Action) -> ResetSpec | None:
        """返回动作条目的自动复位规格（问题 1），未配置返回 None。

        复位规格从映射条目 reset 字段读取（如 emote:wave 的
        reset: {delay: 3.0, tool: set_emote_parameter, args: {value: 0}}），
        由执行器在动作执行成功后延迟 delay 秒自动调用复位工具。
        """
        entry = self._resolve_entry(action)
        if not isinstance(entry, dict):
            return None
        reset_raw = entry.get("reset")
        if not reset_raw:
            return None
        try:
            return _parse_reset_spec(reset_raw)
        except Exception as exc:
            logger.warning("动作复位规格解析失败 %s: %s", action.kind, exc)
            return None

    def mapped_parameter_names(self) -> list[str]:
        """映射表中 `set_avatar_parameter` 工具引用的参数名列表（启动自动校准比对用）。

        只收集 set_avatar_parameter 的参数名（OSC avatar 参数映射的核心载体）；
        其它工具（move/look_direction/send_message/emote 等）不引用 avatar 参数，
        不在校准范围内。返回去重保序列表。
        """
        names: list[str] = []
        seen: set[str] = set()
        for entry in self._mapping.values():
            if not isinstance(entry, dict):
                continue
            tools = entry.get("tools") if isinstance(entry.get("tools"), list) else [entry]
            for item in tools:
                if not isinstance(item, dict) or item.get("tool") != "set_avatar_parameter":
                    continue
                args = item.get("args") or {}
                pn = args.get("parameter_name")
                if pn and str(pn) not in seen:
                    seen.add(str(pn))
                    names.append(str(pn))
        return names

    def get_emergency_reset_commands(self) -> list[OSCCommand]:
        """急停复位命令（问题 1：替代未注册的 legacy reset 工具）。

        优先级：
        1. 映射表顶层 emergency_reset.tools（显式配置）；
        2. 兜底：遍历所有含 reset 的条目，收集复位命令并去重
           （tool + 序列化 args 指纹相同视为同一条）。
        两者都没有 → 返回空列表（调用方降级为 legacy reset 或跳过）。
        """
        er = (self._raw or {}).get("emergency_reset")
        if isinstance(er, dict) and isinstance(er.get("tools"), list):
            commands: list[OSCCommand] = []
            for item in er["tools"]:
                if isinstance(item, dict) and item.get("tool"):
                    commands.append(OSCCommand(tool=item["tool"], args=dict(item.get("args", {}))))
            if commands:
                return commands
        seen: set[tuple[str, str]] = set()
        collected: list[OSCCommand] = []
        for entry in self._mapping.values():
            if not isinstance(entry, dict):
                continue
            reset_raw = entry.get("reset")
            if not reset_raw:
                continue
            try:
                spec = _parse_reset_spec(reset_raw)
            except Exception:
                continue
            for cmd in spec.tools:
                fingerprint = (
                    cmd.tool,
                    json.dumps(cmd.args, sort_keys=True, ensure_ascii=False),
                )
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    collected.append(cmd)
        return collected

    def resolve_key(self, action: Action) -> str:
        """动作查找键：`kind:name`（有 name 时）否则 `kind`。

        公开接口（F-03）：执行器需用同一套键给自动复位任务分槽，
        否则"同 kind 不同通道"（如左右手比耶）会互相取消复位任务。
        """
        name = (
            action.params.get("name")
            or action.params.get("emote")
            or action.params.get("gesture")
        )
        if name:
            return f"{action.kind}:{name}"
        return action.kind

    @staticmethod
    def _fill_args(args: dict, params: dict) -> dict:
        """用动作参数填充 args 模板中的 {param} 占位符（如 {message}）。

        未提供的占位符保持原样（留给工具侧报错，避免静默丢失必填字段）。
        返回新 dict，不修改映射模板（防止多次 map 共享可变引用）。
        """
        if not params:
            return dict(args)
        filled: dict = {}
        for k, v in args.items():
            if isinstance(v, str):
                s = v
                for pname, pval in params.items():
                    s = s.replace("{" + pname + "}", str(pval))
                filled[k] = s
            else:
                filled[k] = v
        return filled

    @staticmethod
    def _sanitize_args(tool: str, args: dict) -> dict:
        """工具级参数清洗：send_message 的 message 超长截断（VRChat chatbox 限制）。"""
        if tool == "send_message" and isinstance(args.get("message"), str):
            msg = args["message"].strip()
            if len(msg) > _CHATBOX_MAX_LEN:
                args = dict(args)
                args["message"] = msg[:_CHATBOX_MAX_LEN].rstrip() + "…"
        return args
