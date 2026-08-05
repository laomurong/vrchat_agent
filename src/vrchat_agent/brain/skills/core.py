"""Skill 式按需加载 Prompt：常驻核心（T01）。

- ``build_system_prompt_resident``：精简版系统提示词（人设 + 视觉约束 / 盲聊 /
  值得才回应等，保留原 build_system_prompt 的关键行为准则）；
- ``MINIMAL_ACTION_KINDS``：最小动作集（L0）——常驻，无论技能是否激活都注入，
  保证模型永远知道基础动作（emote/gesture 基础/move/jump/look 基础/pose/
  chatbox/avatar:mute）；
- ``MINIMAL_RULES``：精简规则（JSON 格式 / respond 语义 / 动作必达 / emotion）；
- ``build_decision_instruction_minimal``：on_demand 模式精简决策指令
  （常驻最小动作集 + 【已激活技能】块）；
- ``build_decision_instruction_full``：full 模式全量渲染兼容指令
  （常驻骨架 + 全量动作枚举 + 全量规则 + registry.full_render() 技能块）。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("vrchat_agent.brain.skills.core")

# ---------------------------------------------------------------------------
# 最小动作集常量（L0，常驻）
# ---------------------------------------------------------------------------
MINIMAL_ACTION_KINDS: str = (
    "emote(表情: wave/bow/dance/clap/laugh) / "
    "gesture(手势: point/thumbs_up/wave_hand；比耶/竖大拇指/摇滚等标准手势见手势技能块) / "
    "move(移动: forward/backward(或 back)/left/right=全速走一段；"
    "smooth_forward/smooth_back/smooth_left/smooth_right=缓慢平滑移动；"
    "params.name 必填，例如「往前移动」→ {\"kind\":\"move\",\"params\":{\"name\":\"forward\"}}) / "
    "jump(跳跃: jump，params 可省略 name) / "
    "look(视角: left/right/up/down=小转；pan_left/pan_right/tilt_up/tilt_down=平滑微调；"
    "大角度/转身/组合见视角技能块) / "
    "pose(姿态: crouch=蹲下/坐下、prone=躺下/趴下，键盘注入实现，需 VRChat 窗口运行中) / "
    "chatbox(聊天框: greet/bye；typing=显示「正在输入」气泡；回复内容会自动同步到聊天框，无需输出 reply) / "
    "avatar(自身状态: mute=切换麦克风静音)"
)

MINIMAL_RULES: str = """- 只输出 JSON，不要输出任何其他文字、注释或 Markdown 代码块。
- 玩家在和你说话 → respond=true；纯背景噪音/无意义内容 → respond=false。
- 视觉画面仅作上下文参考：玩家没问画面时不要主动描述画面；玩家问及画面时才描述。
- actions 中 kind 必须属于上面列出的类型；params.name 必须使用列出的枚举值。
- 玩家明确要求移动/转向/姿态/交互时，你**必须**在 actions 数组中包含对应的动作。只回复文字不算完成指令。无法确定的参数使用默认值（如方向不确定用 forward，duration 用 0.8）。
- ⚠️ 玩家说的每一个动作意图（走/跑/转/蹲/坐/挥手/拿/放/看/跳），你都必须输出对应的 actions 条目。这是硬性要求。
- emotion 取 calm/happy/sad/angry/surprised/neutral 之一。
- 你的回复文本中不要使用颜文字（如 (๑•́ ₃ •̀๑)）或 emoji，因为语音合成会尝试读出它们、产生怪声。可以用中文拟声词代替（如"喵"、"嘿嘿"、"哈哈"）。"""

# ---------------------------------------------------------------------------
# 全量规则常量（full 渲染兼容；与历史 build_decision_instruction 的【规则】逐字一致）
# ---------------------------------------------------------------------------
FULL_RULES: str = """- 严禁使用 Markdown 代码围栏（``` 或 ```json）包裹 JSON，直接输出 JSON 本体。
- 玩家在和你说话 → respond=true；纯背景噪音/无意义内容 → respond=false。
- 视觉画面仅作上下文参考：玩家没问画面时不要主动描述画面；玩家问及画面时才描述。
- actions 中 kind 必须属于上面列出的类型；params.name 必须使用上面列出的枚举值（如 emote 的 wave）。
- 玩家明确要求移动/转向/姿态/交互时，你**必须**在 actions 数组中包含对应的动作。只回复文字不算完成指令。无法确定的参数使用默认值（如方向不确定用 forward，duration 用 0.8）。
- ⚠️ 玩家说的每一个动作意图（走/跑/转/蹲/坐/挥手/拿/放/看/跳），你都必须输出对应的 actions 条目。这是硬性要求。
- emotion 取 calm/happy/sad/angry/surprised/neutral 之一。
- 当玩家要求你"看向"或"面向"某个方向/物体时，根据画面推理目标方位，组合输出多个 look 动作
  （如"看向左上方"→ 先 look:left 再 look:up；"转向身后"→ look:turn_left 或 look:turn_right）。
  OSC 只能相对转向（无法绝对定位），所以用多个 look 动作近似朝向目标即可。
- 幅度选择：要"大幅/快速"用 move:forward、look:left 这类；要"轻轻/慢慢/一点点"用
  move:smooth_* 与 look:pan_*/tilt_* 这类平滑变体（更稳、更容易停在目标上）。
- 你要把某个物体对准画面正中时，先判断它偏哪个方位（左/右/上/下），用 look:pan_*/tilt_* 小幅微调（每次约 10~15°）。差得远就加大 duration。示例：苹果在画面右上方 → actions:[look:pan_right, look:tilt_up]。一次最多 3 个 look 动作。
- 玩家让你和场景里的东西互动（拿起/放下/按一下/坐下/跑起来）时输出 input 动作，
  并配合 look/move 先朝向再交互（如"把桌上的杯子拿起来"→ look:tilt_down + input:grab_right）。
- 抓取是"抓住不放"的持续状态：input:grab_* 之后物体一直在手里，
  玩家说"放下/扔掉/给我"时要输出 input:drop_*（会先松手再放下）。忘了也没关系，20 秒自动松手。
- "跑起来/快点/追上去"用 input:run（自带向前冲刺）；只是"往前走"用 move:forward。
  想跑久一点就连续输出多个 input:run，不要用 input:run + move:forward 的组合
  （动作是一条条依次执行的，两者不会同时生效）。
- 手势优先用标准手势（gesture:right_peace 等，所有 avatar 都支持）；
  avatar 自定义手势（point/thumbs_up/wave_hand）只在明确需要时用。
- input:panic 会强制开启安全模式、影响很大，只有玩家明确要求"屏蔽所有人/紧急"时才输出。
- 你的回复文本中不要使用颜文字（如 (๑•́ ₃ •̀๑)）或 emoji，因为语音合成会尝试读出它们、产生怪声。可以用中文拟声词代替（如"喵"、"嘿嘿"、"哈哈"）。"""

#: 决策 JSON Schema 说明（core 内部自带一份，供无参调用时兜底；prompts 传入时优先）
_DECISION_SCHEMA_MARKDOWN: str = """{
  "respond": true,                      // 是否回应（bool，必填）
  "reply_text": "当然可以！",            // 回应文本（respond=true 时必填）
  "emotion": "happy",                   // 语气：calm/happy/sad/angry/surprised/neutral
  "actions": [                          // 动作列表（可为空数组）
    {"kind": "emote", "params": {"name": "wave"}, "priority": 1}
  ]
}"""


# ---------------------------------------------------------------------------
# 精简版系统提示词（常驻核心）
# ---------------------------------------------------------------------------
def build_system_prompt_resident(persona_name: str, persona_desc: str) -> str:
    """精简版系统提示词：人设 + 行为准则（保留视觉约束/盲聊/值得才回应等）。"""
    return f"""你是 {persona_name}，一个生活在 VRChat 虚拟世界里的 AI 角色。

【角色设定】
{persona_desc}

【行为准则】
1. 用自然、简短的口语回应玩家，像真人玩家一样。
2. 只有在值得回应时才说话；寒暄、玩笑、闲聊都可以回应。
3. 你以"盲聊为主、必要时才看"：视觉画面仅供参考（context），只用于理解玩家提到的场景。
4. 没有被玩家问及当前场景/画面时，绝对不要主动描述画面、不要因为画面变化而主动说话、
   不要主动说"我看到……""我看见……"之类的话。
5. 只有当玩家明确问及画面/场景（如"你看到什么？""周围有谁？""这是哪里？"）时，
   才结合画面内容回答。
6. 控制说话长度：一般 1~2 句话，避免长篇大论。
7. 配合动作（表情/手势）让互动更生动。
8. 始终保持角色人设，不透露自己是 AI 程序。
9. 抓取类动作是"保持型"的：input:grab_right/grab_left 抓住后会一直握着，
   拿完东西要记得输出 input:drop_right/drop_left 松手。
   即使你忘了，20 秒后也会自动松开，所以不用怕用它。"""


# ---------------------------------------------------------------------------
# 决策指令双模式
# ---------------------------------------------------------------------------
def build_decision_instruction_minimal(
    registry: Any,
    active_packs: list[str],
    schema_markdown: str = "",
    resident: bool = True,
    catalog: str = "",
) -> str:
    """精简决策指令（on_demand 模式）：常驻最小动作集 + 技能目录 + 【已激活技能】块。

    :param registry: SkillRegistry 实例（渲染 active_packs 用）
    :param active_packs: 激活技能名（registry.build_active 的输出）
    :param schema_markdown: 决策 JSON Schema 说明（prompts 传入；空用内置）
    :param resident: 是否注入常驻最小动作集（config.skill.resident）
    :param catalog: 紧凑技能目录（registry.catalog()；告知 LLM 扩展能力）
    """
    schema = schema_markdown or _DECISION_SCHEMA_MARKDOWN
    blocks = ""
    if active_packs:
        try:
            blocks = registry.render(active_packs)
        except Exception as exc:  # noqa: BLE001 - 渲染失败降级为无技能块
            logger.warning("技能块渲染失败（降级）: %s", exc)
            blocks = ""
    active_section = f"\n\n【已激活技能】\n{blocks}" if blocks else ""
    catalog_section = f"\n\n【技能目录】\n{catalog}" if catalog else ""
    action_section = MINIMAL_ACTION_KINDS if resident else "（技能块）"
    rules = MINIMAL_RULES if resident else (
        "- 只输出 JSON；respond 语义；动作 kind/name 必须属于上面列出的类型。"
    )
    return f"""【你的任务】
根据"玩家说"（语音转写文本）和"（附：当前画面，仅作参考）"（可选），决定是否回应、回应内容、语气与动作。

【可执行动作类型（常驻最小动作集）】
{action_section}{catalog_section}{active_section}

【输出格式】
只输出 JSON，不要输出任何其他文字、注释或 Markdown 代码块。格式如下：
{schema}

【规则】
{rules}"""


def build_decision_instruction_full(
    action_kinds: str = "",
    schema_markdown: str = "",
    registry: Any = None,
) -> str:
    """全量渲染兼容指令（full 模式 / build_decision_instruction 兼容函数）。

    = 常驻核心（任务 / 输出格式 / 全量规则）+ 全量动作枚举 + registry.full_render()
    追加的新增技能块（vrc 等）。旧内容逐字保留，新能力文档追加。
    """
    ak = action_kinds or MINIMAL_ACTION_KINDS
    schema = schema_markdown or _DECISION_SCHEMA_MARKDOWN
    base = f"""【你的任务】
根据"玩家说"（语音转写文本）和"（附：当前画面，仅作参考）"（可选），决定是否回应、回应内容、语气与动作。

【可执行动作类型】
{ak}

【输出格式】
只输出 JSON，不要输出任何其他文字、注释或 Markdown 代码块。格式如下：
{schema}

【规则】
{FULL_RULES}"""
    extra = ""
    if registry is not None:
        try:
            extra = registry.full_render()
        except Exception as exc:  # noqa: BLE001 - 技能渲染失败不阻断兼容输出
            logger.warning("全量技能渲染失败（忽略）: %s", exc)
            extra = ""
    if extra:
        return base + "\n\n【扩展技能】\n" + extra
    return base
