"""主入口：装配全部组件并启动 AgentLoop（T01 骨架 → T04 完整）。

装配顺序：
1. 配置 + 日志
2. 事件总线（绑定事件循环）
3. 感知层：AudioInput → VAD 切分 → ASR → utterance_ready（AudioPipeline）
   自抑制：SelfSuppression 订阅 tts 事件 → AudioInput 静音窗口
4. 视觉：OBS WebSocket 首选 / OpenCV 回退；低频采样
5. 大脑：LLMClient + ConversationContext + Brain
6. 语音输出：TTSEngine（cloud/edge，默认 cloud）+ AudioOutput
7. 动作：ActionMapper + MCPClient + MCPOSCActuator（失败降级 NoopActuator）
8. 编排：AgentLoop + EmergencyStop + Dashboard

运行：
    python -m vrchat_agent.main          # 需 PYTHONPATH=src 或 pip install -e .
    powershell scripts/run.ps1           # 一键启动 [Win]
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

from .config import Config

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 模块级 logger：build_tts_engine 等装配辅助函数可被独立调用/测试，
# 不依赖 build_and_run 内部先执行 setup_logging。
logger = logging.getLogger("vrchat_agent")

# 模块级：最近一次进程级重启拉起的新进程引用（A-21）。仅保存引用供日志/诊断
# （如崩溃报告可查 PID），**不管理生命周期**——restart 语义下该子进程是新的
# agent 实例，旧进程退出后不得 terminate 它；新实例自行管理自身生命周期。
_restart_proc: subprocess.Popen | None = None


def _spawn_restart(config: Config) -> subprocess.Popen | None:
    """进程级重启：拉起新进程（继承 cwd/env），旧进程随即退出。

    方案选择（需求 1，回传说明）：
    - 采用 **Popen 进程级重启** 而非进程内重启（Config.reload() + 重新 build）：
      ① 进程内重启需重建全部模块（asyncio 任务/事件总线/MCP 子进程/声卡/uvicorn
         socket），装配复杂且易残留状态；② 进程级重启天然重新加载配置，无需处理
         热重载边界；③ Windows 下 os.execv 会**替换当前进程映像且不执行 Python
         层清理**（MCP 子进程/声卡/uvicorn socket 无法优雅释放），不可靠。
    - 时序：本函数在 build_and_run 的 finally（全部资源释放完成）之后调用——
      新进程启动时旧端口已释放，无需端口冲突重试；新进程继承 cwd/PYTHONPATH/env，
      配置路径经 --config 显式传入（不依赖环境变量残留）。
    - A-21：Popen 引用保存到模块级 _restart_proc 供日志/诊断（PID 可查），
      但不 terminate——新进程是新的 agent 实例，旧进程退出后不得杀它。
    """
    global _restart_proc
    try:
        env = dict(os.environ)
        src_dir = str(PROJECT_ROOT / "src")
        parts = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
        if src_dir not in parts:
            parts.insert(0, src_dir)
        env["PYTHONPATH"] = os.pathsep.join(parts)
        cmd = [sys.executable, "-m", "vrchat_agent.main", "--config", str(config.path)]
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env)
        _restart_proc = proc
        logger.info("已拉起新进程（pid=%s），旧进程即将退出", proc.pid, extra={"event": "restart"})
        return proc
    except Exception as exc:
        _restart_proc = None
        logger.error("重启拉起新进程失败: %s", exc, extra={"event": "restart_failed"})
        return None


def build_tts_engine(config: Config):
    """按配置装配 TTS 引擎（含 cloud/clone/mimo 失败自动回退 edge 的兜底，任务 1）。

    策略：
    - provider=edge            → 直接用 EdgeTTSEngine
    - provider=clone           → FallbackTTSEngine(primary=LocalClone, fallback=Edge)：
                                克隆语音本地服务（GPT-SoVITS/CosyVoice）失败
                                （服务不可达/超时/返回非音频）自动回退 Edge，保证
                                "克隆服务没起也不阻断说话"；未配置 base_url → 直接
                                降级 EdgeTTSEngine（INFO 日志说明，行为一致）。
    - provider=mimo            → FallbackTTSEngine(primary=MimoTTSEngine, fallback=Edge)：
                                小米 MiMo 克隆语音（云服务，chat.completions + audio
                                参数）失败（无 key/参考音频缺失/认证/超时/网络/返回
                                非音频）自动回退 Edge 中文女声，保证"克隆服务挂了也
                                不阻断说话"。无 key 不在装配期降级（与 cloud 不同），
                                由引擎合成时抛清晰 TTSError → Fallback 回退 Edge。
    - provider=cosyvoice       → 缺 DASHSCOPE_API_KEY：直接降级 EdgeTTSEngine（INFO
                                日志说明，与 cloud 无 key 降级一致）；有 key 则
                                FallbackTTSEngine(primary=CosyVoice, fallback=Edge)：
                                阿里云 DashScope CosyVoice v3.5-plus（云端克隆 TTS）
                                失败（合成超时/认证/网络/voice_id 未配置）自动回退
                                Edge，保证"DashScope 挂了也不阻断说话"。
    - provider=cloud 且有 key  → FallbackTTSEngine(primary=Cloud, fallback=Edge)：
                                云端失败（认证/端点不支持/解码失败）自动回退本地 Edge，
                                保证"cloud 挂了也不阻断说话"。
    - provider=cloud 无 key    → 直接降级 EdgeTTSEngine（INFO 日志说明，行为一致）。

    热重载（A1）：voice/rate/volume/emotion_voice/tts_clone.*/tts_mimo.*/tts_cosyvoice.*
    由引擎在合成时从 config 读取，引擎构造时传入 config 引用（provider 切换属"需重启"
    字段，A2）。
    """
    from .speech.tts import FallbackTTSEngine, TTSError
    from .speech.tts_cloud import CloudTTSEngine
    from .speech.tts_cosyvoice import CosyVoiceTTSEngine
    from .speech.tts_edge import EdgeTTSEngine
    from .speech.tts_local_clone import LocalCloneTTSEngine
    from .speech.tts_mimo import MimoTTSEngine

    provider = config.speech.tts.provider
    timeout = config.speech.tts.timeout
    if provider == "edge":
        return EdgeTTSEngine(
            voice=config.speech.tts.voice,
            rate=f"{config.speech.tts.rate:+d}%",
            timeout=timeout,
            config=config,
        )
    if provider == "cosyvoice":
        # 缺 DASHSCOPE_API_KEY：构造期即抛 TTSError → 降级 Edge（行为与 cloud 无 key 一致）
        try:
            cosyvoice = CosyVoiceTTSEngine(
                api_key_env=config.speech.tts.tts_cosyvoice.api_key_env,
                model=config.speech.tts.tts_cosyvoice.model,
                voice_id=config.speech.tts.tts_cosyvoice.voice_id,
                timeout=config.speech.tts.tts_cosyvoice.timeout,
                config=config,
            )
        except TTSError as exc:
            logger.info(
                "CosyVoice TTS 不可用（%s），使用 Edge 本地合成（provider=cosyvoice 已自动降级）",
                exc,
            )
            return EdgeTTSEngine(voice="zh-CN-XiaoxiaoNeural", timeout=timeout, config=config)
        edge = EdgeTTSEngine(voice="zh-CN-XiaoxiaoNeural", timeout=timeout, config=config)
        logger.info("TTS 引擎: cosyvoice（阿里云 DashScope CosyVoice，失败自动回退 Edge 本地合成）")
        return FallbackTTSEngine(primary=cosyvoice, fallback=edge)
    if provider == "mimo":
        mimo_cfg = config.speech.tts.tts_mimo
        mimo = MimoTTSEngine(
            base_url=mimo_cfg.base_url,
            api_key=mimo_cfg.api_key,
            model=mimo_cfg.model,
            ref_audio_path=mimo_cfg.ref_audio_path,
            timeout=mimo_cfg.timeout,
            config=config,
        )
        edge = EdgeTTSEngine(voice="zh-CN-XiaoxiaoNeural", timeout=timeout, config=config)
        logger.info("TTS 引擎: mimo（MiMo 克隆语音，失败自动回退 Edge 本地合成）")
        return FallbackTTSEngine(primary=mimo, fallback=edge)
    if provider == "clone":
        clone_cfg = config.speech.tts.tts_clone
        base_url = (clone_cfg.base_url or "").strip()
        if not base_url:
            logger.info(
                "未配置 speech.tts.tts_clone.base_url，使用 Edge 本地合成（provider=clone 已自动降级）"
            )
            return EdgeTTSEngine(voice="zh-CN-XiaoxiaoNeural", timeout=timeout, config=config)
        clone = LocalCloneTTSEngine(
            base_url=base_url,
            method=clone_cfg.method,
            ref_audio_path=clone_cfg.ref_audio_path,
            ref_text=clone_cfg.ref_text,
            timeout=clone_cfg.timeout,
            config=config,
        )
        edge = EdgeTTSEngine(voice="zh-CN-XiaoxiaoNeural", timeout=timeout, config=config)
        logger.info("TTS 引擎: clone（本地克隆语音，失败自动回退 Edge 本地合成）")
        return FallbackTTSEngine(primary=clone, fallback=edge)
    # provider == "cloud"
    if not config.speech.tts.api_key:
        # 行为与日志一致：无 key 必须真正落到 edge，而不是带着空 key 创建 Cloud 引擎
        logger.info("无 TTS_API_KEY，使用 Edge 本地合成（provider=cloud 已自动降级）")
        return EdgeTTSEngine(voice="zh-CN-XiaoxiaoNeural", timeout=timeout, config=config)
    cloud = CloudTTSEngine(
        base_url=config.brain.llm.base_url,
        api_key=config.speech.tts.api_key,
        voice=config.speech.tts.voice,
        timeout=timeout,
        emotion_voice=config.speech.tts.emotion_voice,
        config=config,
    )
    edge = EdgeTTSEngine(voice="zh-CN-XiaoxiaoNeural", timeout=timeout, config=config)
    logger.info("TTS 引擎: cloud（失败自动回退 Edge 本地合成）")
    return FallbackTTSEngine(primary=cloud, fallback=edge)


def _load_config(path: str | Path | None = None) -> Config:
    """加载配置（默认 config.yaml；可用 VRC_AGENT_CONFIG 覆盖）。"""
    p = Path(path) if path else Path(os.environ.get("VRC_AGENT_CONFIG", PROJECT_ROOT / "config.yaml"))
    return Config.load(p)


async def build_and_run(config: Config) -> None:
    """装配全部组件并启动（供 main() 调用，也便于测试）。"""
    from .logging_setup import UsageTracker, get_logger, setup_logging

    setup_logging(config.logging.level, config.logging.file)
    logger = get_logger("vrchat_agent")

    from .events import EventBus

    bus = EventBus()
    loop = asyncio.get_running_loop()
    bus.bind(loop)
    usage = UsageTracker()

    # ------------------------------------------------------------------
    # 3. 感知层（A）
    # ------------------------------------------------------------------
    from .audio.asr import create_asr
    from .audio.audio_input import AudioInput
    from .audio.filter import UtteranceFilter
    from .audio.pipeline import AudioPipeline
    from .audio.self_suppression import SelfSuppression
    from .audio.vad import VAD, SpeechSegmenter

    audio_input = AudioInput(
        device=config.audio.input.device,
        sample_rate=config.audio.input.sample_rate,
        chunk_ms=config.audio.input.chunk_ms,
    )
    vad = VAD(mode=config.asr.vad_mode, sample_rate=config.audio.input.sample_rate, config=config)
    segmenter = SpeechSegmenter(
        vad=vad,
        min_speech_ms=config.asr.min_speech_ms,
        silence_ms=config.asr.silence_ms,
        max_segment_s=config.asr.max_segment_s,
        sample_rate=config.audio.input.sample_rate,
        chunk_ms=config.audio.input.chunk_ms,
        silence_threshold=config.audio.input.silence_threshold,
        config=config,
    )
    # 按 config.asr.engine 装配 ASR：whisper（默认，原路径）| dolphin（sherpa-onnx）
    asr = create_asr(config)
    suppression = SelfSuppression(bus=bus, audio_input=audio_input)
    suppression.attach()
    # ASR 误识别过滤（A-05）：传 config 引用，每次转写后从 audio.input.filter_*
    # 读取（Web 保存后热重载即时生效）。
    utterance_filter = UtteranceFilter(config=config)
    pipeline = AudioPipeline(audio_input, segmenter, asr, bus, suppression, filter=utterance_filter)

    # ------------------------------------------------------------------
    # 4. 视觉（E）
    # ------------------------------------------------------------------
    from .vision import create_vision_sensor
    from .vision.event_detector import VisualEventDetector
    from .vision.sampler import ImageSampler

    sensor = None
    try:
        sensor = create_vision_sensor(config.vision.sensor)
        await sensor.start()
    except Exception as exc:
        logger.error("视觉传感器启动失败（视觉降级为不可用）: %s", exc, extra={"event": "vision_failed"})
        sensor = None

    sampler = ImageSampler(
        fps=config.vision.sampling.fps,
        max_size=config.vision.sampling.max_size,
        jpeg_quality=config.vision.sampling.jpeg_quality,
        event_trigger=config.vision.sampling.event_trigger,
        config=config,
    )
    # 事件检测器始终创建（无资源开销）：event_trigger 热重载开启时立即可用，
    # 阈值 change_threshold 每次判定从 config 读取。
    detector = VisualEventDetector(
        threshold=config.vision.sampling.change_threshold, config=config
    )

    # ------------------------------------------------------------------
    # 5. 大脑（B）
    # ------------------------------------------------------------------
    from .brain.brain import Brain
    from .brain.context import ConversationContext
    from .brain.llm_client import LLMClient
    from .brain.memory import MemoryStore

    if not config.brain.llm.api_key:
        logger.warning("未配置 LLM_API_KEY：将使用空 key（仅本地 Ollama/vLLM 可工作）")
    llm = LLMClient(
        base_url=config.brain.llm.base_url,
        api_key=config.brain.llm.api_key,
        model=config.brain.llm.model,
        timeout=config.brain.llm.timeout,
        max_tokens=config.brain.llm.max_tokens,
        config=config,
    )
    context = ConversationContext(max_turns=config.brain.context.max_turns)
    # 长期记忆（标准版）：SQLite 本地持久化，重启不丢、跨会话生效。
    # 始终创建（Web 可查看/编辑/删除）；enabled=false 时 Brain 不注入不提取。
    db_path = Path(config.brain.memory.db_path)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path
    memory_store = MemoryStore(db_path, max_items=config.brain.memory.max_items)
    logger.info(
        "长期记忆库: %s（%s，上限 %d 条）",
        memory_store.db_path,
        "启用" if config.brain.memory.enabled else "禁用",
        config.brain.memory.max_items,
    )
    brain = Brain(
        llm=llm,
        context=context,
        persona_name=config.brain.persona.name,
        persona_desc=config.brain.persona.system_prompt,
        vision_enabled=config.brain.llm.vision_enabled,
        config=config,
        memory=memory_store,
    )

    # ------------------------------------------------------------------
    # 6. 语音输出（C）——统一走 build_tts_engine（cloud 失败自动回退 edge，任务 1）
    # ------------------------------------------------------------------
    from .speech.audio_output import AudioOutput

    tts = build_tts_engine(config)
    audio_out = AudioOutput(
        device=config.speech.output.device, sample_rate=config.speech.output.sample_rate
    )

    # ------------------------------------------------------------------
    # 7. 动作（D）
    # ------------------------------------------------------------------
    from .action.action_mapper import ActionMapper
    from .action.actuator import NoopActuator
    from .action.mcp_actuator import MCPOSCActuator
    from .bridge.mcp_client import MCPClient

    mapper = ActionMapper(config.action.mapping.path)
    mcp = MCPClient(
        command=config.action.bridge.command,
        args=config.action.bridge.args,
        timeout=config.action.bridge.timeout,
    )
    # 任务①：VRChat API 执行器（懒加载——构造不登录，首次 vrc 动作才登录；
    # vrcapi 未启用/未配置时保持 None，vrc 动作执行时报清晰错误）
    vrc_executor = None
    try:
        if config.vrcapi.enabled:
            from .vrcapi.actions import VRCActionExecutor
            from .vrcapi.client import VRCAPIClient

            vrc_client = VRCAPIClient(
                username_env=config.vrcapi.username_env,
                password_env=config.vrcapi.password_env,
                api_key_env=config.vrcapi.api_key_env,
                rate_limit_seconds=config.vrcapi.rate_limit_seconds,
            )
            vrc_executor = VRCActionExecutor(vrc_client)
            logger.info("VRChat API 执行器已就绪（vrc 动作可用，懒加载登录）")
    except Exception as exc:  # noqa: BLE001 - 初始化失败仅降级（vrc 动作不可用），不阻断启动
        logger.warning("VRChat API 执行器初始化失败（vrc 动作将不可用）: %s", exc)
    try:
        await mcp.start()
        actuator = MCPOSCActuator(mcp, mapper, vrc_executor=vrc_executor)
        logger.info("OSC 桥接已就绪（动作可用）")
    except Exception as exc:
        logger.error("OSC 桥接启动失败，降级为仅语音无动作: %s", exc, extra={"event": "bridge_failed"})
        actuator = NoopActuator()

    # ------------------------------------------------------------------
    # 8. 编排（F）+ 急停 + 仪表盘（G）
    # ------------------------------------------------------------------
    from .core.agent_loop import AgentLoop
    from .core.dashboard import Dashboard
    from .core.emergency import EmergencyStop

    agent_loop = AgentLoop(
        bus=bus,
        brain=brain,
        tts=tts,
        audio_out=audio_out,
        actuator=actuator,
        vision=sensor,
        sampler=sampler,
        config=config,
    )
    emergency = EmergencyStop(
        hotkey=config.emergency.hotkey,
        bus=bus,
        flag_file=PROJECT_ROOT / config.emergency.flag_file,
    )
    dashboard = Dashboard(bus, config, usage) if config.dashboard.enabled else None

    # ------------------------------------------------------------------
    # Web 控制台（M4 增量，W-01/W-02）：同进程内嵌 uvicorn task
    # 随机端口 + 自动开浏览器由 WebServer 内部完成；随 gather 统一管理
    # ------------------------------------------------------------------
    web_server = None
    if config.web.enabled:
        from .speech.voice_clone import VoiceCloneManager
        from .web import WebServer

        # 音色克隆管理器：始终创建（读取 data/voices.json，供 Web 展示/克隆/设为当前
        # 音色）；provider=cosyvoice 时 set_voice_id 会写回 config，引擎热重载即复用。
        voice_clone_manager = VoiceCloneManager(config=config)

        web_server = WebServer(bus, config, usage, PROJECT_ROOT, dashboard=dashboard,
                               control=agent_loop, memory=memory_store,
                               voice_clone=voice_clone_manager)

    # ------------------------------------------------------------------
    # 启动
    # ------------------------------------------------------------------
    tasks = [
        asyncio.create_task(agent_loop.run(), name="agent_loop"),
        asyncio.create_task(pipeline.run(), name="audio_pipeline"),
        asyncio.create_task(emergency.arm(), name="emergency"),
    ]
    # 视觉采样循环始终启动（热重载 A1：event_trigger 运行中可开/关；关闭时循环仅休眠）
    if sensor is not None:
        tasks.append(asyncio.create_task(sampler.run(sensor, bus, detector), name="vision_sampler"))
    if dashboard is not None:
        tasks.append(asyncio.create_task(dashboard.run(), name="dashboard"))
    if web_server is not None:
        tasks.append(asyncio.create_task(web_server.run(), name="web"))

    # 需求 1：主循环正常停止（shutdown/restart）→ 取消其余长驻任务，保证 gather
    # 返回并走 finally 收尾（restart 时在 finally 拉起新进程）。各组件任务内部已
    # 做异常隔离（web 崩溃自动重启一次语义不变），此处只响应"主循环主动停止"。
    agent_loop_task = tasks[0]

    def _on_agent_loop_stopped(done_task: asyncio.Task) -> None:
        if done_task.cancelled() or done_task.exception() is not None:
            return
        for other in tasks[1:]:
            if not other.done():
                other.cancel()

    agent_loop_task.add_done_callback(_on_agent_loop_stopped)

    # 需求 2：启动自检 + 自动校准（异步后台执行，不阻塞装配/主循环；异常隔离）
    if config.agent.startup_check:

        async def _startup_check_worker() -> None:
            from .core.startup_check import run_startup_checks

            try:
                report = await run_startup_checks(
                    config,
                    bus=bus,
                    mcp=mcp if mcp.started else None,
                    vision_sensor=sensor,
                    mapper=mapper,
                    sampler=sampler,
                )
            except Exception as exc:
                logger.error("启动自检异常（忽略，不阻塞主循环）: %s", exc, extra={"event": "startup_check_failed"})
                return
            if dashboard is not None:
                dashboard.startup_report = report
            checks = report.get("checks", {}) or {}
            summary = " ".join(
                f"{name}={'OK' if (c or {}).get('ok') else 'FAIL'}"
                for name, c in checks.items()
            )
            logger.info("启动自检报告: %s", summary, extra={"event": "startup_check_done", "checks": checks})

        asyncio.create_task(_startup_check_worker(), name="startup_check")

    logger.info(
        "VRChat Agent 已启动: LLM=%s TTS=%s 视觉=%s 急停=%s",
        config.brain.llm.model,
        config.speech.tts.provider,
        config.vision.sensor.backend,
        config.emergency.hotkey,
        extra={"event": "agent_started"},
    )
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logger.info("Agent 正在关闭...")
        # 不 re-raise：Ctrl+C / shutdown / restart 均为正常退出路径，
        # 取消剩余任务并走 finally 完成资源收尾后进程自然退出。
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        if web_server is not None:
            web_server.shutdown()
        await emergency.disarm()
        # 问题 1：关闭前先取消 actuator 残留的自动复位后台任务（不残留后台任务），
        # 再停 MCP 会话（复位任务依赖 client.call_tool）。
        try:
            if hasattr(actuator, "shutdown"):
                await actuator.shutdown()
        except Exception:
            pass
        # A-13：关停前等待最近一次异步记忆提取任务完成（超时 3s，异常静默），
        # 避免 LLM/写库中途被进程退出打断造成半写入。
        try:
            await brain.wait_extract(timeout=3)
        except Exception:
            pass
        try:
            await mcp.stop()
        except Exception:
            pass
        if sensor is not None:
            try:
                await sensor.stop()
            except Exception:
                pass
        # 需求 1：进程级重启——全部资源释放完成后拉起新进程（端口已释放，无需
        # 冲突重试；新进程重新加载配置）。失败仅记日志，不影响本进程正常退出。
        if getattr(agent_loop, "restart_requested", False):
            _spawn_restart(config)


async def main(config_path: str | Path | None = None) -> None:
    """入口协程。"""
    config = _load_config(config_path)
    await build_and_run(config)


def main_entry() -> None:
    """控制台脚本入口（支持 --config 可选参数，供进程级重启复用）。"""
    import argparse

    parser = argparse.ArgumentParser(description="VRChat Agent")
    parser.add_argument(
        "--config",
        default=None,
        help="config.yaml 路径（默认 VRC_AGENT_CONFIG 环境变量或项目根 config.yaml）",
    )
    args = parser.parse_args()
    try:
        asyncio.run(main(args.config))
    except KeyboardInterrupt:
        print("\n[退出] Ctrl+C 收到，Agent 已停止。")


if __name__ == "__main__":
    main_entry()
