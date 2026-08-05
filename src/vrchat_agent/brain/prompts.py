"""B 大脑：系统提示词模板（角色人设 + 决策指令 + 记忆提取，B-05）。

- build_system_prompt：角色人设（persona）
- build_decision_instruction：输出结构化决策 JSON 的指令
- build_memory_extract_prompt：从最近对话提取值得长期记住的事实（标准版记忆系统）
"""

from __future__ import annotations

from .schema import decision_schema_markdown

# 支持的动作语义（与 mappings/avatar_default.yaml 的 key 一一对应；
# 决策 actions[].kind 与 params.name 必须落在下列枚举内，否则动作会被映射表忽略）
# 注意：chatbox:reply 由主循环在每次 AI 回复时自动附加（agent.chatbox_sync），
# LLM 无需也不应输出 reply（避免与自动发送重复）；这里只列 greet/bye 等显式语义。
_ACTION_KINDS = (
    "emote(表情: wave/bow/dance/clap/laugh) / "
    "gesture(手势: 两类——"
    "①avatar 自定义: point/thumbs_up/wave_hand；"
    "②标准手势(所有 avatar 通用，左右手各 8 种): "
    "left_neutral/left_fist/left_open/left_point/left_peace/left_rock/left_gun/left_thumb，"
    "right_neutral/right_fist/right_open/right_point/right_peace/right_rock/right_gun/right_thumb；"
    "如「比个耶」→ {\"kind\":\"gesture\",\"params\":{\"name\":\"right_peace\"}}、"
    "「竖大拇指」→ right_thumb；手势会保持 3 秒后自动放松) / "
    "move(移动: forward/backward(或 back)/left/right=按住方向键走一段（全速）；"
    "smooth_forward/smooth_back/smooth_left/smooth_right=缓慢平滑移动（适合「慢慢走过来」「靠近一点」）；"
    "params.name 必填，方向枚举值见括号内，例如「往前移动」→ "
    "{\"kind\":\"move\",\"params\":{\"name\":\"forward\"}}、"
    "「向左走」→ {\"kind\":\"move\",\"params\":{\"name\":\"left\"}}；"
    "玩家说'向前/后/左/右走X步'时，必须输出对应 move 动作，duration 用 0.3~0.8s 对应一步) / "
    "jump(跳跃: jump，params 可省略 name) / "
    "look(视角: left/right=左右小转；left_big/right_big=左右大转；"
    "turn_left/turn_right=转身约180°；turn_around_left/turn_around_right=转一整圈约360°；"
    "up/down=上下小转(抬头/低头)；up_big/down_big=上下大转；"
    "pan_left/pan_right=平滑左右转、tilt_up/tilt_down=平滑上下转（转得更稳更精确，"
    "适合「稍微往左看一点」这类微调；pan_left/pan_right/tilt_up/tilt_down 也适合精准居中微调（每次约 10~15°））；"
    "可组合多个 look 动作来朝向不同方向，例如先左转再抬头看向左上方的物体) / "
    "pose(姿态: crouch=蹲下/坐下、prone=躺下/趴下；"
    "这两个动作 VRChat 的 OSC 不支持，本项目改用键盘注入实现——"
    "直接向 VRChat 窗口按下 C 键（蹲下）/ Z 键（躺下）约 0.6 秒后松开，"
    "因此需要 VRChat 桌面客户端窗口处于运行状态；"
    "params.name 必填，例如「蹲下」→ {\"kind\":\"pose\",\"params\":{\"name\":\"crouch\"}}、"
    "「躺下」→ {\"kind\":\"pose\",\"params\":{\"name\":\"prone\"}}) / "
    "input(与世界交互: run=向前冲刺一小段（Run 是修饰键，会边跑边前进，"
    "不是原地按一下；想跑得更远就连续输出多个 input:run）；"
    "use_right/use_left=使用或按下面前的东西；"
    "grab_right/grab_left=用右手/左手抓住物体，"
    "⚠️抓住后会一直握着不放，直到你输出 drop_right/drop_left，或超过 20 秒自动松手；"
    "drop_right/drop_left=松手并把手里的东西放下（同时解除握持状态）；"
    "sit_stand=在椅子上坐下或起身；afk=切换挂机状态；quick_menu=打开快捷菜单；"
    "panic=紧急屏蔽周围所有人（会强制开启安全模式，只在玩家明确要求时用）；"
    "params.name 必填，例如「抓住那个杯子」→ "
    "{\"kind\":\"input\",\"params\":{\"name\":\"grab_right\"}}) / "
    "chatbox(聊天框: greet/bye；typing=显示「正在输入」气泡；"
    "回复内容会自动同步到聊天框，无需输出 reply) / "
    "avatar(自身状态: mute=切换麦克风静音) / "
    "vrc(VRChat API 集成（联网操作，需账号已配置且已登录，执行较慢约几十秒级）："
    "change_avatar=换头像（params 需带 avatar_id，形如 avtr_xxx）；"
    "list_friends=查好友列表；"
    "search_world=搜索世界/房间（params 可带 query 关键词，如「搜一下日本刀相关世界」→ "
    "{\"kind\":\"vrc\",\"params\":{\"name\":\"search_world\",\"query\":\"japan\"}}）；"
    "set_status=更新在线状态（params.status 取 active/join me/busy/ask me/offline，"
    "如「设置成忙碌」→ {\"kind\":\"vrc\",\"params\":{\"name\":\"set_status\",\"status\":\"busy\"}}）；"
    "⚠️ vrc 动作是真实网络请求且受 VRChat API 限速（约 1 次/分钟），只在玩家明确要求"
    "「换头像/查好友/找世界/改状态」时输出，且一次只输出一个 vrc 动作) / "
    "nav(空间导航(拓扑导航，VRChat OSC 无绝对坐标，只能相对转向+移动+视觉重定位)："
    "goto=去往某处——params.target 填地标描述（如「去喷泉那里」→ "
    "{\"kind\":\"nav\",\"params\":{\"name\":\"goto\",\"target\":\"喷泉\"}}；"
    "不带 target 则返回最近已知位置)；"
    "explore=自主探索世界——向未访问方向移动并记录新地标，"
    "适合「探索一下」「到处走走」「看看那边有什么」→ "
    "{\"kind\":\"nav\",\"params\":{\"name\":\"explore\"}})"
)


def build_system_prompt(persona_name: str, persona_desc: str) -> str:
    """构造系统提示词：人设 + 行为准则。"""
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


def build_decision_instruction() -> str:
    """构造决策输出指令（全量渲染兼容函数，输出与历史逐字一致）。

    实现（T02）：常驻核心（任务 / 输出格式 / 全量规则，逐字保留）由
    skills.core 提供；动作枚举用模块级 ``_ACTION_KINDS``（历史逐字保留）；
    新增技能块（vrc 等）经默认 SkillRegistry.full_render() 追加。旧内容
    一条不少，新能力文档追加，测试断言（oscexpand / nav / look_turn /
    osc_fix / oscmap）全部继续通过。
    """
    from .skills.core import build_decision_instruction_full
    from .skills.packs import default_registry

    return build_decision_instruction_full(
        action_kinds=_ACTION_KINDS,
        schema_markdown=decision_schema_markdown(),
        registry=default_registry(),
    )


def build_calibration_prompt(
    target_desc: str,
    image_width: int = 0,
    image_height: int = 0,
    fov_horizontal: int = 90,
) -> str:
    """构造视觉校准 prompt：仅判断目标物体是否居中+偏移方向+角度。

    Token 预算 <200，LLM 仅输出严格 JSON，无其他文字。
    image_width/image_height: 截图实际像素尺寸（0=未知，不注入尺寸行）。
    fov_horizontal: 水平视野角度（默认 90°），配合尺寸让 VLM 精确计算偏移度数。
    """
    lines = [
        f'视觉校准器。目标：「{target_desc}」。',
        '看画面判断目标是否在正中央。',
    ]
    if image_width > 0 and image_height > 0:
        lines.append(
            f'画面尺寸: {image_width}x{image_height} 像素, 水平视野(FOV): {fov_horizontal}°'
        )
    lines.extend([
        '只输出JSON（无其他文字）：',
        '{{"centered": bool, "offset_h": "left"|"right"|"none", "offset_v": "up"|"down"|"none", "degrees": int}}',
        '- centered: 目标在画面正中→true',
        '- offset_h: 目标相对画面中心的水平偏移方向',
        '- offset_v: 目标相对画面中心的垂直偏移方向',
        '- degrees: 大致偏移角度（整数，5的倍数 5/10/15/20/25/30/35/40/45）',
    ])
    return '\n'.join(lines)


def build_memory_extract_prompt() -> str:
    """构造记忆提取提示词：从最近对话提取值得长期记住的事实（标准版）。

    输出严格 JSON：``{"memories": ["...", "..."]}``。要求简短、独立、可复用；
    空结果输出 ``{"memories": []}``。Brain 在每 extract_interval 轮决策后以
    低优先级异步调用本 prompt（失败静默降级，不阻塞主循环）。
    """
    return """【记忆提取任务】从下面的对话中提取玩家值得被长期记住的事实、喜好与偏好，
供后续跨会话使用。例如：玩家喜欢安静的音乐；玩家养了一只叫布丁的猫；
玩家是一名程序员；玩家讨厌香菜。

【要求】
1. 只提取稳定、长期有效的信息；不要提取一次性寒暄、情绪化发言或临时话题。
2. 每条记忆用一句简短、独立、明确的陈述（主语尽量用"玩家"），无需重复"玩家说"。
3. 输出严格 JSON，不要输出任何其他文字或 Markdown 代码块：
{"memories": ["记忆1", "记忆2", ...]}
4. 没有值得记住的内容时输出 {"memories": []}。"""
