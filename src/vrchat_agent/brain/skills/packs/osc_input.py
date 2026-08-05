"""交互输入技能块（T02）：run / use / grab / drop / sit / afk / panic + 交互规则。

内容逐字搬自 prompts.py 的 _ACTION_KINDS 与 build_decision_instruction 规则
（含抓取保持型规则与 panic 限制）。
"""

from __future__ import annotations

from ..registry import SkillPack

PACK = SkillPack(
    name="osc_input",
    domain="osc",
    triggers=(
        "抓住", "抓", "拿起", "放下", "松手", "扔", "跑", "冲刺", "坐下",
        "起身", "挂机", "afk", "快捷菜单", "菜单", "屏蔽", "panic", "交互",
        "按钮", "按一下", "使用", "grab", "drop", "run", "sit",
    ),
    description=(
        "input(与世界交互: run=向前冲刺一小段（Run 是修饰键，会边跑边前进，\n"
        "不是原地按一下；想跑得更远就连续输出多个 input:run）；\n"
        "use_right/use_left=使用或按下面前的东西；\n"
        "grab_right/grab_left=用右手/左手抓住物体，\n"
        "⚠️抓住后会一直握着不放，直到你输出 drop_right/drop_left，或超过 20 秒自动松手；\n"
        "drop_right/drop_left=松手并把手里的东西放下（同时解除握持状态）；\n"
        "sit_stand=在椅子上坐下或起身；afk=切换挂机状态；quick_menu=打开快捷菜单；\n"
        "panic=紧急屏蔽周围所有人（会强制开启安全模式，只在玩家明确要求时用）；\n"
        "params.name 必填，例如「抓住那个杯子」→ "
        "{\"kind\":\"input\",\"params\":{\"name\":\"grab_right\"}})\n"
        "【交互规则】\n"
        "- 玩家让你和场景里的东西互动（拿起/放下/按一下/坐下/跑起来）时输出 input 动作，\n"
        "  并配合 look/move 先朝向再交互（如\"把桌上的杯子拿起来\"→ look:tilt_down + input:grab_right）。\n"
        "- 抓取是\"抓住不放\"的持续状态：input:grab_* 之后物体一直在手里，\n"
        "  玩家说\"放下/扔掉/给我\"时要输出 input:drop_*（会先松手再放下）。忘了也没关系，20 秒自动松手。\n"
        "- \"跑起来/快点/追上去\"用 input:run（自带向前冲刺）；只是\"往前走\"用 move:forward。\n"
        "  想跑久一点就连续输出多个 input:run，不要用 input:run + move:forward 的组合\n"
        "  （动作是一条条依次执行的，两者不会同时生效）。\n"
        "- input:panic 会强制开启安全模式、影响很大，只有玩家明确要求\"屏蔽所有人/紧急\"时才输出。"
    ),
    priority=3,
    max_rounds=3,
)
