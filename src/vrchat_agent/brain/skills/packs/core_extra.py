"""核心动作技能块（T02）：emote / move / jump / pose / chatbox / avatar + 通用规则。

内容逐字搬自 prompts.py 的 _ACTION_KINDS 与 build_decision_instruction 规则，
供 on_demand 模式在玩家提到移动/姿态/表情等基础动作时按需注入完整细节。
"""

from __future__ import annotations

from ..registry import SkillPack

PACK = SkillPack(
    name="core_extra",
    domain="core",
    triggers=(
        "移动", "走", "前进", "后退", "向前", "向后", "往前走", "往左走", "往右走",
        "蹲", "躺", "趴", "坐", "跳", "跳跃", "表情", "挥手", "跳舞", "鼓掌",
        "大笑", "笑", "鞠躬", "静音", "mute", "move", "forward", "backward",
        "crouch", "prone", "jump",
    ),
    description=(
        "emote(表情: wave/bow/dance/clap/laugh) / \n"
        "move(移动: forward/backward(或 back)/left/right=按住方向键走一段（全速）；\n"
        "smooth_forward/smooth_back/smooth_left/smooth_right=缓慢平滑移动（适合「慢慢走过来」「靠近一点」）；\n"
        "params.name 必填，方向枚举值见括号内，例如「往前移动」→ "
        "{\"kind\":\"move\",\"params\":{\"name\":\"forward\"}}、\n"
        "「向左走」→ {\"kind\":\"move\",\"params\":{\"name\":\"left\"}}；\n"
        "玩家说'向前/后/左/右走X步'时，必须输出对应 move 动作，duration 用 0.3~0.8s 对应一步) / \n"
        "jump(跳跃: jump，params 可省略 name) / \n"
        "pose(姿态: crouch=蹲下/坐下、prone=躺下/趴下；\n"
        "这两个动作 VRChat 的 OSC 不支持，本项目改用键盘注入实现——\n"
        "直接向 VRChat 窗口按下 C 键（蹲下）/ Z 键（躺下）约 0.6 秒后松开，\n"
        "因此需要 VRChat 桌面客户端窗口处于运行状态；\n"
        "params.name 必填，例如「蹲下」→ {\"kind\":\"pose\",\"params\":{\"name\":\"crouch\"}}、\n"
        "「躺下」→ {\"kind\":\"pose\",\"params\":{\"name\":\"prone\"}}) / \n"
        "chatbox(聊天框: greet/bye；typing=显示「正在输入」气泡；\n"
        "回复内容会自动同步到聊天框，无需输出 reply) / \n"
        "avatar(自身状态: mute=切换麦克风静音)\n"
        "【通用规则】\n"
        "- 玩家明确要求移动/转向/姿态/交互时，你**必须**在 actions 数组中包含对应的动作。"
        "只回复文字不算完成指令。无法确定的参数使用默认值（如方向不确定用 forward，duration 用 0.8）。\n"
        "- ⚠️ 玩家说的每一个动作意图（走/跑/转/蹲/坐/挥手/拿/放/看/跳），你都必须输出对应的 actions 条目。"
        "这是硬性要求。\n"
        "- emotion 取 calm/happy/sad/angry/surprised/neutral 之一。"
    ),
    priority=1,
    max_rounds=0,
)
