"""B 大脑：Decision / Action pydantic 模型 + LLM 输出 JSON 校验（B-02/B-03）。

共享知识约定（架构文档第 8 节第 4 条）：
- 决策输出必须为 JSON，由 schema.Decision 校验；
- 解析失败自动降级为"不回应"并告警。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger("vrchat_agent.brain.schema")

# 动作类型（D 动作控制层可执行的动作种类）
# ⚠️ 这是**封闭白名单**：pydantic 校验时 kind 不在其中会让整条 Decision 校验失败，
#    进而被 parse_json 降级为 respond=False（用户的话被静默吞掉）。因此每次在
#    mappings/*.yaml 新增 `kind:name` 条目，都必须同步把新 kind 加进这里，
#    并在 brain/prompts.py 的 _ACTION_KINDS 里向 LLM 说明。
# - input：/input/* 交互按钮（run/use_right/grab_left/sit_stand/afk/panic/quick_menu 等）
# - avatar：写 avatar 自身状态（mute 等）
# - vrc：VRChat API 集成（任务①）——换头像/查好友/找世界/更新状态，
#   params.name 取 change_avatar/list_friends/search_world/set_status，
#   由 mappings/avatar_default.yaml 的 vrc:* 条目映射到 vrc_* 工具，
#   经 action/mcp_actuator.py 委托 vrcapi/actions.py 执行（异步 + 限速）。
# - nav：空间导航（任务④）——拓扑导航（VRChat OSC 无绝对坐标，只能
#   节点=视觉地标指纹 + 边=相对动作序列），由 AgentLoop._execute_actions
#   拦截交给 nav.NavController 执行（不走 actuator/MCP）。
#   params.name 取 goto（去往某处，params.target 填地标描述）/
#   explore（自主探索世界）；mappings/avatar_default.yaml 的 nav:* 条目
#   仅作文档占位（伪工具 nav_goto/nav_explore，正常不流入 actuator）。
ActionKind = Literal[
    "emote", "gesture", "move", "jump", "look", "pose", "chatbox", "input", "avatar", "vrc", "nav"
]

#: VRChat 技能块名（prompt 注入的技能块标识）。LLM 常把技能块名（vrc.social 等）
#: 误写成 kind 或 params.name，导致 pydantic literal_error 拒绝**整条** Decision
#: → 降级不回应（用户实测：kind="vrc.account" 好友查询 3 次全失败）。
#: 这些名字**不是** ActionKind，只是技能块标识。
_VRC_SKILL_NAMES: frozenset[str] = frozenset(
    {"vrc.account", "vrc.social", "vrc.avatar", "vrc.world", "vrc.call_api"}
)

#: 技能块名 → 精选动作名（params.name 误写技能块名时归一化，如
#: {"kind":"vrc","params":{"name":"vrc.social"}} → name="list_friends"）。
_VRC_SKILL_TO_ACTION: dict[str, str] = {
    "vrc.social": "list_friends",  # 社交最常见查询：好友列表
    "vrc.account": "get_current_user",  # 账号最常见查询：当前用户信息
    "vrc.avatar": "get_own_avatar",  # 头像最常见查询：当前使用头像
    "vrc.world": "search_world",  # 世界最常见查询：搜索世界
    "vrc.call_api": "call_api",  # 通用层
}


def _normalize_kind(kind: Any) -> Any:
    """把 LLM 误写的 ``vrc.*`` 技能块名归一化为 ``vrc``（其余原样返回）。

    BugFix（用户实测）：LLM 输出 ``{"kind":"vrc.account",...}`` 被 ActionKind
    白名单拒绝 → 整条 Decision 降级不回应。只归一化 ``vrc.`` 前缀（技能块名），
    不放宽白名单——真正非法的 kind（如 "foo"）仍被 pydantic 拒绝。
    """
    if isinstance(kind, str):
        k = kind.strip()
        if k.startswith("vrc."):
            return "vrc"
    return kind


def _normalize_params_name(name: Any) -> Any:
    """把 ``params.name`` 误写的技能块名归一化为对应精选动作名（其余原样返回）。

    只归一化已知技能块名；LLM 的其它别名（如 friends_online / friends）保持原值，
    由 ActionMapper / VRCActionExecutor 层容错归一化（executor 层见
    vrcapi/actions.py 的 _VRC_TOOL_ALIASES）。
    """
    if isinstance(name, str):
        n = name.strip()
        if n in _VRC_SKILL_TO_ACTION:
            return _VRC_SKILL_TO_ACTION[n]
    return name


class Action(BaseModel):
    """一条动作指令。"""

    kind: ActionKind
    params: dict = Field(default_factory=dict)
    priority: int = 0  # 数字越大越先执行

    @model_validator(mode="before")
    @classmethod
    def _normalize_vrc_kind(cls, data: Any) -> Any:
        """宽容归一化钩子：``vrc.*`` 技能块名误写为 kind/name 时自动修正。

        背景（用户实测 BugFix）：LLM 把技能块名（vrc.account/vrc.social/
        vrc.avatar/vrc.world/vrc.call_api）当成 kind 值输出 → pydantic
        literal_error 拒绝**整条** Decision → respond=False 降级不回应
        （连回复文本都丢）。修复：
        - kind 为 ``vrc.*`` 前缀 → 归一化为 ``vrc``（params.name 保持 LLM 意图）；
        - params.name 为技能块名 → 归一化为对应精选动作名；
        - 真正非法的 kind（如 "foo"）**仍被拒绝**——只做 ``vrc.*`` 前缀归一化，
          不放宽成"任意字符串都接受"。
        """
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if "kind" in out:
            out["kind"] = _normalize_kind(out["kind"])
        params = out.get("params")
        if isinstance(params, dict):
            p = dict(params)
            if "name" in p:
                p["name"] = _normalize_params_name(p["name"])
            out["params"] = p
        return out


class Decision(BaseModel):
    """LLM 结构化决策。"""

    respond: bool
    reply_text: str | None = None
    emotion: str = "calm"  # calm / happy / sad / angry / ...
    actions: list[Action] = Field(default_factory=list)

    @classmethod
    def parse_json(cls, raw: str) -> "Decision":
        """解析 LLM 输出的 JSON 字符串为 Decision。

        兼容三种形态：
        1) 顶层即决策对象：{"respond": true, ...}
        2) 包裹形态：{"decision": {...}}
        3) 宽松形态（本地小模型常见）：```json 代码围栏 / 前后夹带解释文字 /
           JSON5 尾逗号——先经 _extract_json_object 归一化再解析。

        解析失败时的降级策略：
        - 纯文本回退：原始文本非空且不含 {（LLM 直接返回自然语言，如
          "好的，帮你把视角往上转～"），降级为 respond=True 的纯文本回复，
          WARNING 告警（event="decision_plain_text_fallback"），避免用户的话被静默吞掉。
        - JSON 格式错误：原始文本含 { 但解析失败（截断/乱码/部分 JSON），
          降级为 Decision(respond=False) 并 ERROR 告警（event="decision_parse_failed"）。
        - 空字符串/纯空白：降级为 Decision(respond=False)。
        """
        try:
            candidate = _extract_json_object(raw)
            data = json.loads(candidate)
            if isinstance(data, dict) and "decision" in data and isinstance(data["decision"], dict):
                data = data["decision"]
            if not isinstance(data, dict) or "respond" not in data:
                raise ValueError("缺少 respond 字段")
            decision = cls.model_validate(data)
            logger.info("决策解析成功: respond=%s actions=%d", decision.respond, len(decision.actions))
            return decision
        except Exception as exc:
            # 纯文本回退：LLM 返回不含 { 的非空文本（如"好的，帮你转～"），
            # 直接作为回复文本降级，避免用户的话被静默吞掉。
            stripped = (raw or "").strip()
            if stripped and "{" not in (raw or ""):
                logger.warning(
                    "LLM 输出非 JSON（纯文本），降级为纯文本回复: %s | 原文: %r",
                    exc,
                    (raw or "")[:500],
                    extra={"event": "decision_plain_text_fallback"},
                )
                return cls(respond=True, reply_text=stripped, emotion="neutral", actions=[])
            logger.error(
                "决策 JSON 解析失败（降级为不回应）: %s | 原文: %r",
                exc,
                (raw or "")[:500],
                extra={"event": "decision_parse_failed"},
            )
            return cls(respond=False, reply_text=None, actions=[])


def decision_schema_markdown() -> str:
    """返回决策 JSON 的 Schema 说明（供系统提示词使用）。"""
    return """{
  "respond": true,                      // 是否回应（bool，必填）
  "reply_text": "当然可以！",            // 回应文本（respond=true 时必填）
  "emotion": "happy",                   // 语气：calm/happy/sad/angry/surprised/neutral
  "actions": [                          // 动作列表（可为空数组）
    {"kind": "emote", "params": {"name": "wave"}, "priority": 1}
  ]
}"""


def _extract_json_object(raw: str) -> str:
    """从 LLM 输出中提取最外层 JSON 对象文本（宽松解析，P0 修复）。

    实机发现：qwen3.5-9b 等本地模型经常用 ```json ... ``` 代码围栏包裹输出，
    导致 json.loads 直接失败（Expecting value: line 1 column 1）→ 决策降级为
    "不回应"（日志中"决策 JSON 解析失败"与 respond=False 一一对应）。本函数
    依次处理：
    1) 剥离 Markdown 代码围栏（```json ... ``` 或 ``` ... ```）；
    2) 取第一个 { 到最后一个 }（容忍前后夹带解释文字）；
    3) 剥离 JSON5 尾逗号（{"a": 1,} / [1, 2,]）。

    提取失败抛 ValueError（原因清晰，供调用方记录）。
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("LLM 输出为空")
    # 1) 剥离 Markdown 代码围栏（token 替换，只剥前后缀、不删整行——兼容
    #    单行围栏形态 ```json {"respond": true, ...}```，QA 回归锁定）
    text = re.sub(r"^```[A-Za-z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # 2) 取第一个 { 到最后一个 }（容忍前后解释文字）
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("未找到 JSON 对象（缺少 { 或 }）")
    candidate = text[start : end + 1]
    # 3) 剥离尾逗号（JSON5 兼容：} 或 ] 前不允许逗号）
    return _strip_trailing_commas(candidate)


def _strip_trailing_commas(s: str) -> str:
    """剥离 JSON 中的尾逗号（{"a":1,} → {"a":1}；[1,2,] → [1,2]）。

    逐字符扫描，跳过字符串字面量（含转义），仅在深度>0 的容器内把 `,` 后紧跟
    `}` 或 `]` 的逗号删除。不依赖 json5 三方库。
    """
    out: list[str] = []
    in_str = False
    esc = False
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == ",":
            # 前瞻：跳过空白，若下一个是 } 或 ] 则丢弃该逗号（尾逗号）
            j = i + 1
            while j < n and s[j] in " \t\r\n":
                j += 1
            if j < n and s[j] in "}]":
                i = j
                continue
            out.append(ch)
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)
