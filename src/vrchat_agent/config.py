"""配置体系：YAML 加载 + pydantic 校验 + 热重载（T01 / T05 增强）。

共享知识约定（架构文档第 8 节第 3 条）：
- config.yaml 顶层节：asr / audio / brain / speech / action / vision / agent / dashboard / emergency / logging
- API keys 不写入 yaml，统一从 .env 读取（LLM_API_KEY / TTS_API_KEY）
- Config.reload() 对支持热更新的节（阈值/采样帧率/映射表路径/人设）生效
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

logger = logging.getLogger("vrchat_agent.config")


class ConfigError(RuntimeError):
    """配置加载/校验错误。"""


# ---------------------------------------------------------------------------
# 各节配置模型（与 config.yaml 顶层节一一对应）
# ---------------------------------------------------------------------------
class AsrConfig(BaseModel):
    # ASR 引擎：whisper（faster-whisper，默认，原路径不变）| dolphin（sherpa-onnx Dolphin small CTC）。
    engine: str = "whisper"
    model_size: str = "medium"
    # 本地模型目录：
    # - engine=whisper：faster-whisper 模型目录（含 model.bin），优先于 model_size 加载；
    #   None 或目录不可用时回退 model_size 在线下载。
    # - engine=dolphin：Dolphin 模型目录（内含 model.onnx + tokens.txt）。
    model_path: str | None = None
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "zh"
    # Dolphin 引擎解码方式：greedy_search | beam_search（beam 可减少叠字）；whisper 忽略。
    decoding_method: str = "greedy_search"
    vad_mode: int = 2
    min_speech_ms: int = 300
    silence_ms: int = 800
    # A-16：max_segment_s 为 float（如 0.3s 用于短语音段）——此前声明为 int，
    # 0.3 会被 pydantic 拒绝；SpeechSegmenter 内部已按 float 处理（_recompute_frames
    # 用 float×1000/chunk_ms 计算帧数），热重载处也改 float() 保持一致。
    max_segment_s: float = 15.0


class AudioInputConfig(BaseModel):
    # x-widget: Web 控制台把设备字段渲染为系统设备下拉框（input 类）。
    # device 必须为非可选 str（默认空串 = 系统默认设备）：若声明为 str|None，
    # pydantic 会生成 anyOf:[{type:string},{type:null}] 而无顶层 type:string，
    # 导致前端 renderField 的 `type === "string" && x-widget` 分支永不命中
    # （下拉框不渲染，QA 回归 BUG）。
    device: str = Field(
        default="",
        json_schema_extra={"x-widget": "audio-device-select", "x-device-kind": "input"},
    )
    sample_rate: int = 16000
    chunk_ms: int = 30
    # RMS 静音预筛阈值（归一化 0~1，参考项目 vrc_chat_translator/voice.py 默认 0.005）：
    # 音频块 RMS 低于该值直接视为静音（不进 webrtcvad，避免底噪误触发"幻听"）。
    silence_threshold: float = 0.005
    # ASR 误识别过滤（A-05）：本地 Whisper 会把环境音/视频字幕/BGM 等误识别为
    # "用户说过的话"，这些文本不应进入 LLM 流程。UtteranceFilter 在转写成功后、
    # publish utterance_ready 之前判定；以下均为热重载字段（保存后即时生效）。
    filter_enabled: bool = True                       # 总开关
    # 子串匹配屏蔽词（命中即跳过）：长词（≥2 字）用包含匹配，覆盖
    # "字幕志愿者 李宗盛" 这类带前后缀的变体。
    filter_blocklist: list[str] = Field(
        default_factory=lambda: ["字幕志愿者", "谢谢观看", "字幕", "BGM", "背景音乐"]
    )
    # 精确匹配屏蔽词（仅单字/短回应词）：text.strip() == 词 才命中，
    # 避免 "你好" 被 "好" 误滤；要滤单字请加到这里而不是靠 min_length。
    filter_exact_words: list[str] = Field(default_factory=lambda: ["好", "嗯", "啊", "哦"])
    # 最短保留长度：text 去空白后长度 < 此值跳过（默认 1 = 仅滤空/纯空白）。
    filter_min_length: int = 1


class AudioOutputConfig(BaseModel):
    # x-widget: Web 控制台把设备字段渲染为系统设备下拉框（output 类）
    device: str = Field(
        default="",
        json_schema_extra={"x-widget": "audio-device-select", "x-device-kind": "output"},
    )
    sample_rate: int = 16000


class AudioConfig(BaseModel):
    input: AudioInputConfig = Field(default_factory=AudioInputConfig)
    output: AudioOutputConfig = Field(default_factory=AudioOutputConfig)


class LLMConfig(BaseModel):
    provider: str = "openai"          # openai / ollama / vllm（OpenAI 兼容端点）
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "LLM_API_KEY"
    api_key: str | None = None        # 运行时由 .env 解析，不写入 yaml
    model: str = "gpt-4o-mini"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)  # 合法范围 [0,2]（M4 表单校验）
    max_tokens: int = 512
    # mimo-v2.5 为推理模型：思考+生成耗时波动大，默认超时放宽至 45s
    timeout: float = 45.0
    vision_enabled: bool = True


class PersonaConfig(BaseModel):
    name: str = "Vivi"
    system_prompt: str = "你是一个住在 VRChat 虚拟世界里的活泼少女 Vivi，喜欢交朋友、爱开玩笑，说话简短自然。"


class ContextConfig(BaseModel):
    max_turns: int = 10


class SemanticMemoryConfig(BaseModel):
    """语义检索（embedding）配置：本地 sentence-transformers 模型（任务②）。

    - enabled：总开关（false 时检索完全走关键词匹配）；
    - model_name：embedding 模型名（多语言默认
      paraphrase-multilingual-MiniLM-L12-v2），模型权重放 model/ 目录不入 git；
    - model_dir：模型根目录（相对项目根；实际模型目录为 model_dir/model_name，
      含 config.json 才算可用）；目录不存在 → 首次使用时 WARNING 并自动降级
      关键词匹配（**不自动联网下载**）；
    - top_k：语义检索返回条数上限（查询时截断）；
    - device：推理设备（cpu / cuda），本地小模型默认 cpu 即可。
    属装配期读取（改后需重启）；纯本地推理，无 API key（不写明文）。
    """

    enabled: bool = True
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    model_dir: str = "model"
    top_k: int = 5
    device: str = "cpu"


class MemoryConfig(BaseModel):
    """长期记忆（标准版）配置：本地持久化事实/喜好，跨会话生效。

    - enabled：总开关（false 时 Brain 不注入不提取，旧行为不回归）；
    - db_path：SQLite 记忆库路径（相对项目根；装配期读取，改后需重启）；
    - max_items：记忆条数上限，超出自动删最久未用（装配期读取，改后需重启）；
    - inject_limit：每次注入 LLM 的记忆条数（热重载）；
    - auto_extract：对话后自动提取长期事实（消耗少量 LLM token，可关；热重载）；
    - extract_interval：每 N 轮决策提取一次（省 token；热重载）；
    - extract_max_items：每次提取最多写入条数（热重载）；
    - semantic：语义检索层（任务②，懒加载 embedding；模型缺失自动降级关键词，
      装配期读取，改后需重启）。
    """

    enabled: bool = True
    db_path: str = "data/memory.db"
    max_items: int = 200
    inject_limit: int = 5
    auto_extract: bool = True
    extract_interval: int = 3
    extract_max_items: int = 5
    semantic: SemanticMemoryConfig = Field(default_factory=SemanticMemoryConfig)


class BrainConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    persona: PersonaConfig = Field(default_factory=PersonaConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)  # 标准版长期记忆
    # 视觉接入模式（P1 视觉行为反馈）：
    #   reference_only（默认）——画面照常传入 LLM 作上下文，靠 prompt 约束"不主动描述"；
    #   keyword_filter ——仅当玩家输入含视觉关键词时才采样并传入画面（更彻底但灵活性低）。
    vision_mode: Literal["reference_only", "keyword_filter"] = "reference_only"


class CloneTTSConfig(BaseModel):
    """克隆语音 TTS 配置（本地模型服务，如 CosyVoice / GPT-SoVITS）。

    provider=clone 时 LocalCloneTTSEngine 通过 HTTP 调本地推理服务拿音频：
    - base_url：本地服务地址（CosyVoice 默认 http://127.0.0.1:8000）；
    - method：post（CosyVoice 等 JSON 服务）/ get（GPT-SoVITS /tts 风格，参数走 query）；
    - ref_audio_path：克隆参考音频路径（服务端可访问的路径或留空用服务默认音色）；
    - ref_text：参考音频对应的文本（few-shot/zero-shot 克隆，留空则由服务端决定）；
    - lang：合成语言（zh/en/ja/ko 等，CosyVoice 服务端做归一化；GPT-SoVITS 兼容）。
    """

    base_url: str = "http://127.0.0.1:9880"
    method: str = Field(
        default="get",
        json_schema_extra={"enum": ["get", "post"]},
    )
    ref_audio_path: str = ""
    ref_text: str = ""
    lang: str = "zh"
    timeout: float = 30.0


class MimoTTSConfig(BaseModel):
    """小米 MiMo 克隆语音 TTS 配置（云服务，chat.completions + audio 参数）。

    MiMo 克隆 TTS 与 OpenAI 兼容 audio/speech 端点完全不同：走
    chat.completions，messages=[user 空, assistant=要合成文本]，audio 参数携带
    参考音频（零样本声音克隆），响应 message.audio.data 为 base64 编码 wav。
    - base_url：MiMo TTS 服务地址（默认 https://api.xiaomimimo.com/v1）；
    - api_key_env：密钥环境变量名（key 走 .env，不写入 config.yaml）；
    - model：克隆 TTS 模型名（默认 mimo-v2.5-tts-voiceclone）；
    - ref_audio_path：克隆参考音频路径（wav，3~10s 干净人声；缺失/读失败会在
      synthesize 时给出含路径的清晰 TTSError）；
    - timeout：单次合成超时（秒，默认 60，首句克隆较慢）。
    """

    base_url: str = "https://api.xiaomimimo.com/v1"
    api_key_env: str = "MIMO_TTS_API_KEY"
    api_key: str | None = None        # 运行时由 .env 解析，不写入 yaml
    model: str = "mimo-v2.5-tts-voiceclone"
    # A-18：默认不再硬编码本机绝对路径（留空由装配期/合成期给出清晰提示"请配置"）。
    # config.yaml 已覆盖该项则不受影响；MimoTTSEngine 合成时对空值抛含字段名的
    # TTSError（触发 FallbackTTSEngine 回退 Edge，不阻断说话）。
    ref_audio_path: str = ""
    timeout: float = 60.0


class CosyVoiceTTSConfig(BaseModel):
    """阿里云 DashScope CosyVoice 云端 TTS 配置（CosyVoice v3.5-plus，音色克隆）。

    - api_key_env：密钥环境变量名（key 走 .env，**绝不写明文**，默认 DASHSCOPE_API_KEY）；
    - model：合成模型（默认 cosyvoice-v3.5-plus；克隆音色时必须与克隆所用模型一致）；
    - voice_id：克隆音色 ID（默认空）。空串时合成会报错提示克隆；可由 Web 控制台
      「音色克隆」创建并设为当前音色后写回本字段（热重载，下次合成即复用）；
    - timeout：单次合成超时（秒，默认 60）。
    """

    api_key_env: str = "DASHSCOPE_API_KEY"
    api_key: str | None = None        # 运行时由 .env 解析，不写入 yaml
    model: str = "cosyvoice-v3.5-plus"
    voice_id: str = ""
    timeout: float = 60.0


class TTSConfig(BaseModel):
    provider: str = Field(
        default="cloud",  # cloud / edge / clone / mimo / cosyvoice（Web 下拉框枚举）
        json_schema_extra={"enum": ["cloud", "edge", "clone", "mimo", "cosyvoice"]},
    )
    api_key_env: str = "TTS_API_KEY"
    api_key: str | None = None        # 运行时由 .env 解析
    # BugFix P0（TTS 主备音色不兼容）：voice 为云端（OpenAI 兼容）音色名；
    # edge_voice 为 Edge-TTS 微软音色名（如 zh-CN-XiaoxiaoNeural）。两者分离，
    # 避免云端配置 voice=alloy 时回退 Edge 收到无效音色导致主备均失败无声。
    voice: str = "alloy"
    edge_voice: str = "zh-CN-XiaoxiaoNeural"
    rate: int = 0
    volume: int = 0
    timeout: float = 10.0
    emotion_voice: dict[str, str] = Field(default_factory=dict)
    # 克隆语音（本地模型）配置：provider=clone 时使用（需求 1 方案变更）
    tts_clone: CloneTTSConfig = Field(default_factory=CloneTTSConfig)
    # 小米 MiMo 克隆语音（云服务）配置：provider=mimo 时使用（MiMo 接入增量）
    tts_mimo: MimoTTSConfig = Field(default_factory=MimoTTSConfig)
    # 阿里云 DashScope CosyVoice（云端克隆 TTS）配置：provider=cosyvoice 时使用
    tts_cosyvoice: CosyVoiceTTSConfig = Field(default_factory=CosyVoiceTTSConfig)


class SpeechConfig(BaseModel):
    tts: TTSConfig = Field(default_factory=TTSConfig)
    output: AudioOutputConfig = Field(default_factory=AudioOutputConfig)


class ActionMappingConfig(BaseModel):
    path: str = "mappings/avatar_default.yaml"


class BridgeConfig(BaseModel):
    # command/args 直接以 node + 绝对路径拉起 OSC 桥接（vrchat-mcp-osc 社区包装包）：
    # 包 bin 名为 iflow-mcp_krekun-vrchat-mcp-osc（非 vrchat-mcp-osc），且 stdio 子进程
    # 不依赖 cwd，故用 node 直接执行 server.js 最可靠；postinstall 已修复包内路径缺陷。
    # A-18：默认不再硬编码本机绝对路径（command/args 留空）。config.yaml 已覆盖则
    # 不受影响；未配置时 MCPClient.start() 会抛清晰 MCPError 提示"请配置
    # action.bridge.command"，main 装配层捕获后降级为仅语音无动作（NoopActuator）。
    command: str = ""
    args: list[str] = Field(default_factory=list)
    timeout: float = 5.0


class ActionConfig(BaseModel):
    mapping: ActionMappingConfig = Field(default_factory=ActionMappingConfig)
    bridge: BridgeConfig = Field(default_factory=BridgeConfig)


class SensorConfig(BaseModel):
    backend: str = "obs_websocket"    # obs_websocket / opencv / window_capture
    obs_host: str = "localhost"
    obs_port: int = 4455
    obs_password: str = ""
    source_name: str = ""
    camera_device: int = 0
    # window_capture 后端：按窗口标题关键词匹配目标窗口（如 "VRChat"）
    window_title_keyword: str = "VRChat"


class SamplingConfig(BaseModel):
    fps: float = 0.5
    max_size: int = 768
    jpeg_quality: int = 80
    event_trigger: bool = True
    change_threshold: float = 0.12


class VisionConfig(BaseModel):
    sensor: SensorConfig = Field(default_factory=SensorConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)


class LoopConfig(BaseModel):
    think_timeout: float = 20.0
    tts_timeout: float = 10.0
    action_timeout: float = 5.0
    latency_budget_s: float = 8.0


class InterruptConfig(BaseModel):
    enabled: bool = True


class DegradationConfig(BaseModel):
    message: str = "网络不佳，请稍后再试"
    beep: bool = True


class UtteranceQueueConfig(BaseModel):
    """对话队列 + 批量合并（需求 4）：模型忙时排队，≥max_batch 条合并批量处理。

    - enabled：总开关（热重载：AgentLoop 每次事件读取，保存即生效）；
    - max_batch：合并阈值——队列中待处理对话 ≥ max_batch 时，取前 max_batch 条
      拼接为一条喂 LLM（综合回应），否则逐条处理（热重载）。
    """

    enabled: bool = True
    max_batch: int = 3


class IdleInstinctConfig(BaseModel):
    """空闲本能微动作循环配置（Idle Instinct Loop）。

    IDLE 状态下周期性发送微小 OSC 动作（视角微抖、小步移动等），
    模拟人类"活着"的感觉——不调用 LLM，纯随机触发。

    所有字段均为热重载（保存 config 后 run_loop 下次 tick 读取最新值）。
    """

    enabled: bool = Field(default=True, title="启用本能循环")
    # 每次微动作间隔（秒），实际在 [min, max] 之间随机
    interval_min_sec: float = Field(default=0.3, ge=0.1, le=5.0, title="最小间隔秒")
    interval_max_sec: float = Field(default=0.7, ge=0.1, le=5.0, title="最大间隔秒")
    # 视角微抖角度范围（度），近似值
    jitter_min_deg: float = Field(default=1.0, ge=0.1, le=10.0, title="最小抖动度数")
    jitter_max_deg: float = Field(default=3.0, ge=0.1, le=10.0, title="最大抖动度数")
    # 过冲修正概率（叠在 jitter 上）
    look_overshoot_prob: float = Field(default=0.10, ge=0.0, le=1.0, title="过冲修正概率")
    # 停顿概率（本 tick 什么都不做）
    hesitate_prob: float = Field(default=0.30, ge=0.0, le=1.0, title="停顿概率")
    # 小步移动概率
    small_step_move_prob: float = Field(default=0.15, ge=0.0, le=1.0, title="小步移动概率")
    # 小步移动 duration 范围（秒），极短，几乎看不出来
    move_duration_min_sec: float = Field(default=0.08, ge=0.01, le=1.0, title="移动最短秒")
    move_duration_max_sec: float = Field(default=0.15, ge=0.01, le=1.0, title="移动最长秒")


class AgentConfig(BaseModel):
    loop: LoopConfig = Field(default_factory=LoopConfig)
    interrupt: InterruptConfig = Field(default_factory=InterruptConfig)
    degradation: DegradationConfig = Field(default_factory=DegradationConfig)
    # 回复时自动同步发送 VRChat 聊天框（chatbox:reply → send_message）
    # 默认开；false 则 AI 回复不自动发聊天框（仅保留 LLM 显式输出的 greet/bye 等）。
    chatbox_sync: bool = True
    # 启动自检（需求 2）：启动后异步后台执行功能自检（LLM/TTS/视觉/OSC 配置可达性）
    # + OSC 自动校准（枚举 avatar 参数与映射表比对，只读不写）。默认开，可关；
    # 属装配期读取（改后需重启进程生效）。
    startup_check: bool = True
    # 对话队列 + 批量合并（需求 4）
    utterance_queue: UtteranceQueueConfig = Field(default_factory=UtteranceQueueConfig)
    # 空闲本能微动作循环
    idle_instinct: IdleInstinctConfig = Field(default_factory=IdleInstinctConfig)


class DashboardConfig(BaseModel):
    enabled: bool = True
    terminal: bool = True
    refresh_ms: int = 500
    obs_overlay: bool = False


class EmergencyConfig(BaseModel):
    hotkey: str = "ctrl+alt+x"
    flag_file: str = "logs/emergency_stop.flag"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str = "logs/agent.jsonl"


class WebConfig(BaseModel):
    """Web 控制台配置（M4 增量，W-01）。"""

    enabled: bool = True
    host: str = "127.0.0.1"      # 仅本机可访问；开放局域网需配鉴权（P2）
    port: int = 0                # 0 = OS 随机分配空闲端口；可填固定端口（如 8080）
    auto_open_browser: bool = True  # 启动后自动打开默认浏览器


class VRCApiConfig(BaseModel):
    """VRChat API 集成配置（vrcapi 节，任务①：换头像/查好友/找世界/更新状态）。

    - 凭据一律走环境变量（默认 VRC_USERNAME / VRC_PASSWORD / VRC_API_KEY），
      **绝不写明文**到 config.yaml / 代码；VRCAPIClient 懒加载——首次使用时
      才读环境变量并登录，未配置/未使用不触发任何登录；
    - enabled：总开关（false 时动作映射 vrc:* 仍可解析，执行时报清晰错误）；
    - rate_limit_seconds：VRChat API 请求最小间隔（官方对单账号请求频率敏感，
      默认 60 秒/请求，全局串行节流）。
    """

    enabled: bool = True
    username_env: str = "VRC_USERNAME"
    password_env: str = "VRC_PASSWORD"
    api_key_env: str = "VRC_API_KEY"
    rate_limit_seconds: float = 60.0


class NavConfig(BaseModel):
    """空间导航配置（nav 节，任务④：拓扑导航）。

    核心约束：VRChat OSC 无绝对坐标 → 只能拓扑导航（节点=视觉地标指纹，
    边=相对动作序列），每步重定位防累积误差。

    - enabled：总开关（false 时 NavController 空转，所有接口安全降级）；
    - db_path：拓扑图 SQLite 文件（相对项目根；data/ 已 gitignore，不入 git）；
    - fingerprint_size：指纹灰度池化边长（默认 32，特征向量 = size²+12 维）；
    - similarity_threshold：指纹匹配/回环阈值（[0,1]，越大越严格）；
    - explore_interval_s：空闲探索最小间隔（秒，节流防频繁挪动）；
    - max_relocalize_attempts：每步重定位最大重试次数（超限放弃该边）。
    属装配期读取（改后需重启）；无明文 key。
    """

    enabled: bool = False
    db_path: str = "data/nav_topology.db"
    fingerprint_size: int = 32
    similarity_threshold: float = 0.85
    explore_interval_s: float = 30.0
    max_relocalize_attempts: int = 3


class AppConfig(BaseModel):
    """全量配置模型（顶层）。"""

    asr: AsrConfig = Field(default_factory=AsrConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    brain: BrainConfig = Field(default_factory=BrainConfig)
    speech: SpeechConfig = Field(default_factory=SpeechConfig)
    action: ActionConfig = Field(default_factory=ActionConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    emergency: EmergencyConfig = Field(default_factory=EmergencyConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    vrcapi: VRCApiConfig = Field(default_factory=VRCApiConfig)
    nav: NavConfig = Field(default_factory=NavConfig)


# ---------------------------------------------------------------------------
# Config 门面：加载 / 校验 / 热重载 / 按节访问
# ---------------------------------------------------------------------------
class Config:
    """配置门面：持有 AppConfig 实例，属性访问自动转发（支持热重载后仍指向最新）。"""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        env_file = self._path.parent / ".env"
        if env_file.exists():
            load_dotenv(env_file, override=False)
        self._app: AppConfig = self._parse()

    # ---- 加载 ----
    @classmethod
    def load(cls, path: str | Path) -> Config:
        """从 YAML 文件加载配置。"""
        return cls(path)

    def _parse(self) -> AppConfig:
        data = self._read_yaml()
        self._apply_env(data)
        try:
            return AppConfig.model_validate(data)
        except Exception as exc:  # pydantic ValidationError
            raise ConfigError(f"config.yaml 校验失败: {exc}") from exc

    def _read_yaml(self) -> dict[str, Any]:
        if not self._path.exists():
            raise ConfigError(f"配置文件不存在: {self._path}")
        try:
            with open(self._path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"配置文件 YAML 解析失败: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"配置文件格式错误（应为 YAML 映射）: {self._path}")
        return data

    @staticmethod
    def _apply_env(data: dict[str, Any]) -> None:
        """把 .env 中的 API keys / 覆盖项合并进配置字典。"""
        llm = data.setdefault("brain", {}).setdefault("llm", {})
        llm["api_key"] = os.getenv(llm.get("api_key_env", "LLM_API_KEY")) or llm.get("api_key")
        if os.getenv("LLM_BASE_URL"):
            llm["base_url"] = os.getenv("LLM_BASE_URL")
        if os.getenv("LLM_MODEL"):
            llm["model"] = os.getenv("LLM_MODEL")

        tts = data.setdefault("speech", {}).setdefault("tts", {})
        tts["api_key"] = os.getenv(tts.get("api_key_env", "TTS_API_KEY")) or tts.get("api_key")

        # MiMo 克隆 TTS key 同样走 .env（MIMO_TTS_API_KEY），注入配置供引擎运行时读取
        mimo = tts.setdefault("tts_mimo", {})
        mimo["api_key"] = os.getenv(mimo.get("api_key_env", "MIMO_TTS_API_KEY")) or mimo.get("api_key")

        # CosyVoice（DashScope）key 走 .env（DASHSCOPE_API_KEY），注入配置供运行时读取
        cosy = tts.setdefault("tts_cosyvoice", {})
        cosy["api_key"] = os.getenv(cosy.get("api_key_env", "DASHSCOPE_API_KEY")) or cosy.get("api_key")

        sensor = data.setdefault("vision", {}).setdefault("sensor", {})
        if os.getenv("OBS_PASSWORD"):
            sensor["obs_password"] = os.getenv("OBS_PASSWORD")

    # ---- 热重载 ----
    def reload(self) -> None:
        """重新读取并校验 config.yaml，替换内部 AppConfig。

        由于模块均通过本对象属性访问配置，替换后即时生效（热更新）。
        注意：仅对支持热更新的节（阈值/采样帧率/映射表路径/人设）生效；
        设备名等变更建议重启进程。
        """
        self._app = self._parse()
        logger.info("配置已热重载: %s", self._path)

    # ---- 访问 ----
    def get(self, section: str) -> Any:
        """按节名返回配置。

        支持点号路径：config.get("brain.llm.timeout")；
        也支持整节：config.get("brain") 返回 dict。
        """
        node: Any = self._app.model_dump()
        for part in section.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return None
        return node

    @property
    def path(self) -> Path:
        return self._path

    # 属性转发到 AppConfig（热重载后自动指向最新）
    def __getattr__(self, name: str) -> Any:
        app = object.__getattribute__(self, "_app")
        if hasattr(app, name):
            return getattr(app, name)
        raise AttributeError(f"Config 无此属性: {name}")
