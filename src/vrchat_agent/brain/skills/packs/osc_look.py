"""视角技能块（T02）：全部 look 变体 + 组合引导 + 幅度规则。

内容逐字搬自 prompts.py 的 _ACTION_KINDS 与 build_decision_instruction 规则
（含大角度变体/转身/转圈/平滑微调与"一次最多 3 个 look"约束）。
"""

from __future__ import annotations

from ..registry import SkillPack

PACK = SkillPack(
    name="osc_look",
    domain="osc",
    triggers=(
        "看", "视角", "转向", "转身", "回头", "抬头", "低头", "左转", "右转",
        "转一圈", "朝向", "面向", "看向", "look", "turn", "pan", "tilt",
    ),
    description=(
        "look(视角: left/right=左右小转；left_big/right_big=左右大转；\n"
        "turn_left/turn_right=转身约180°；turn_around_left/turn_around_right=转一整圈约360°；\n"
        "up/down=上下小转(抬头/低头)；up_big/down_big=上下大转；\n"
        "pan_left/pan_right=平滑左右转、tilt_up/tilt_down=平滑上下转（转得更稳更精确，\n"
        "适合「稍微往左看一点」这类微调；pan_left/pan_right/tilt_up/tilt_down 也适合精准居中微调（每次约 10~15°））；\n"
        "可组合多个 look 动作来朝向不同方向，例如先左转再抬头看向左上方的物体)\n"
        "【视角规则】\n"
        "- 当玩家要求你\"看向\"或\"面向\"某个方向/物体时，根据画面推理目标方位，组合输出多个 look 动作\n"
        "  （如\"看向左上方\"→ 先 look:left 再 look:up；\"转向身后\"→ look:turn_left 或 look:turn_right）。\n"
        "  OSC 只能相对转向（无法绝对定位），所以用多个 look 动作近似朝向目标即可。\n"
        "- 幅度选择：要\"大幅/快速\"用 move:forward、look:left 这类；要\"轻轻/慢慢/一点点\"用\n"
        "  move:smooth_* 与 look:pan_*/tilt_* 这类平滑变体（更稳、更容易停在目标上）。\n"
        "- 你要把某个物体对准画面正中时，先判断它偏哪个方位（左/右/上/下），用 look:pan_*/tilt_* "
        "小幅微调（每次约 10~15°）。差得远就加大 duration。示例：苹果在画面右上方 → "
        "actions:[look:pan_right, look:tilt_up]。一次最多 3 个 look 动作。"
    ),
    priority=4,
    max_rounds=1,
)
