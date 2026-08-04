"""Web 控制台 FastAPI 应用 + 内嵌 uvicorn 生命周期（M4 增量，W-01/W-02）。

要点（架构 1.2）：
- `serve(sockets=[sock])` 复用已绑定 socket，消除"探测-关闭-重绑"竞态；
- `log_config=None` + `access_log=False`：不污染现有 logging_setup；
- `lifespan="off"`：依赖（bus/config/usage/status store）经 app.state 注入；
- `_EmbeddedServer.capture_signals()` 覆盖为 nullcontext()：不让 uvicorn 接管
  Ctrl+C，主循环 gather 才能收到 CancelledError 优雅退出；
- Web task 崩溃仅记录日志 + 自动重启一次，不冒泡到 gather（W-02 验收②③）。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import socket
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from ..audio.device_utils import DeviceResolveError, resolve_output_device_index
from ..config import Config
from ..events import Event, EventBus, EventType
from ..logging_setup import UsageTracker
from . import schema as schema_mod
from .audio_devices import list_audio_devices
from .schema import ConfigValidationError, ConfigWriteError
from .status import WebStatusStore

logger = logging.getLogger("vrchat_agent.web.server")


def _play_test_tone_blocking(
    device: str | None,
    duration: float = 1.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
) -> dict[str, Any]:
    """同步播放一段正弦波测试音到指定输出设备（测试设备是否出声）。

    功能：Web「🔊 测试发声」按钮后端。刻意用 440Hz 正弦波（不依赖 TTS 引擎/
    网络/云端 key），保证设备链路自测稳定、可复现。
    - device: 设备 index 字符串 / 设备名 / 空串（= 系统默认），复用
      audio.device_utils.resolve_output_device_index 统一解析语义；
    - 播放阻塞（sd.OutputStream.write），调用方须放 executor/独立线程；
    - 失败返回 {"ok": false, "error": <清晰原因>}（设备不可用/采样率不支持时
      含 VB-Audio 安装提示），成功返回 {"ok": true, "message": ...}。
    """
    try:
        import sounddevice as sd
    except ImportError:
        return {"ok": False, "error": "sounddevice 未安装（pip install sounddevice）"}
    try:
        resolved = resolve_output_device_index(device)
    except DeviceResolveError as exc:
        return {"ok": False, "error": str(exc)}

    # 生成 440Hz 正弦波测试音（0.3 振幅防削波；float32 与 AudioOutput 契约一致）
    n = max(1, int(sample_rate * duration))
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    tone = (np.sin(2.0 * np.pi * frequency * t) * 0.3).astype(np.float32)

    try:
        with sd.OutputStream(
            device=resolved, samplerate=sample_rate, channels=1, dtype="float32"
        ) as stream:
            stream.write(tone)
    except Exception as exc:
        err = str(exc) or type(exc).__name__
        lowered = err.lower()
        # 设备不可用 / 采样率不支持 → 给出可定位的清晰提示（含 VB-Audio 安装指引）
        if (
            "invalid sample rate" in lowered
            or "error opening" in lowered
            or "no output device" in lowered
            or "cannot open device" in lowered
        ):
            hint = (
                "；提示：设备不可用或采样率不支持。若目标是 VB-Audio 虚拟声卡"
                "（TTS → VRChat 麦克风），请先安装 VB-Audio Virtual Cable；"
                "若设备为 WASAPI 仅支持 48k，请改用支持 16k 播放的设备"
                "（如 7.1ch DirectSound）"
            )
            return {"ok": False, "error": f"设备不可用: {err}{hint}"}
        return {"ok": False, "error": f"播放失败: {err}"}

    device_label = device if device not in (None, "") else "系统默认设备"
    return {
        "ok": True,
        "message": f"已在「{device_label}」播放 {frequency}Hz 测试音（{duration:.1f}s）",
    }


class _EmbeddedServer(uvicorn.Server):
    """覆盖 capture_signals：不让 uvicorn 接管主进程 Ctrl+C（版本无关的稳健做法）。"""

    def capture_signals(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()


def _bind_socket(host: str, port: int) -> tuple[socket.socket, int]:
    """绑定监听 socket；port=0 → OS 随机分配；固定端口被占 → 回退随机（W-01 验收①）。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError:
        if port == 0:
            sock.close()
            raise
        sock.bind((host, 0))
    return sock, sock.getsockname()[1]


def _sse(event: str, data: Any) -> str:
    """格式化 SSE 帧（data 为 JSON，ensure_ascii=False 保持中文可读）。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _error(status: int, code: str, detail: Any) -> JSONResponse:
    """统一错误响应：{"ok": false, "error": ..., "detail": ...}。"""
    return JSONResponse(status_code=status, content={"ok": False, "error": code, "detail": detail})


def create_app(
    bus: EventBus,
    config: Config,
    usage: UsageTracker,
    project_root: Path | str,
    dashboard=None,
    heartbeat_s: float = 15.0,
    control=None,
    shutdown_delay_s: float = 0.3,
    memory=None,
    voice_clone=None,
) -> FastAPI:
    """构建 FastAPI 应用（依赖经 app.state 注入；SSE 心跳间隔可注入便于测试）。

    control：AgentLoop（或实现 pause/resume/stop/control_status 的控制对象）。
    为 None 时控制端点返回 503（未接线），不影响读/配类端点。
    memory：MemoryStore（长期记忆，标准版）。为 None 时记忆端点返回
    503/空列表（未接线），不影响其它端点。
    voice_clone：VoiceCloneManager（DashScope CosyVoice 音色克隆）。为 None 时
    音色克隆端点返回 503/空列表（未接线），不影响其它端点。
    shutdown_delay_s：/api/control/shutdown 先返回响应、再延迟停止主循环的秒数
    （保证 HTTP 响应能先 flush 出去，进程再优雅退出；测试可注入 0）。
    """
    store = WebStatusStore(bus, usage)
    root = Path(project_root)
    static_dir = Path(__file__).resolve().parent / "static"

    app = FastAPI(title="VRChat Agent Web Console", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.bus = bus
    app.state.config = config
    app.state.usage = usage
    app.state.store = store
    app.state.project_root = root
    app.state.dashboard = dashboard
    app.state.heartbeat_s = heartbeat_s
    app.state.control = control
    app.state.shutdown_delay_s = shutdown_delay_s
    app.state.memory = memory
    app.state.voice_clone = voice_clone

    # ------------------------------------------------------------------
    # GET / —— 单页 UI（index.html）
    # ------------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        try:
            return FileResponse(static_dir / "index.html")
        except Exception:
            logger.exception("GET / 异常", extra={"event": "web_error"})
            return _error(500, "internal_error", "页面加载失败")

    # ------------------------------------------------------------------
    # GET /api/status —— 当前快照（含用量/uptime/时间线）
    # ------------------------------------------------------------------
    @app.get("/api/status")
    async def api_status() -> JSONResponse:
        try:
            return JSONResponse(content=store.snapshot())
        except Exception:
            logger.exception("GET /api/status 异常", extra={"event": "web_error"})
            return _error(500, "internal_error", "状态获取失败")

    # ------------------------------------------------------------------
    # GET /api/stream —— SSE：先发 snapshot，再转发实时事件，15s 心跳
    # ------------------------------------------------------------------
    @app.get("/api/stream")
    async def api_stream() -> StreamingResponse:
        queue = store.register_queue()
        heartbeat = float(app.state.heartbeat_s)

        async def event_generator() -> Any:
            try:
                # 连接建立先发快照（含 UsageTracker.summary()），保证刷新即见当前状态
                yield _sse("snapshot", store.snapshot())
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=heartbeat)
                    except TimeoutError:
                        yield ": ping\n\n"  # 心跳注释行，防代理断链
                        continue
                    yield _sse(event.type, event.payload)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("SSE 生成器异常", extra={"event": "web_error"})
            finally:
                store.remove_queue(queue)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    # ------------------------------------------------------------------
    # GET /api/audio/devices —— 枚举系统音频设备（输入/输出），供设备下拉框
    # 无可用声卡 / sounddevice 缺失 / CI 环境 → 空列表，前端回退编辑框
    # ------------------------------------------------------------------
    @app.get("/api/audio/devices")
    async def api_audio_devices() -> JSONResponse:
        try:
            return JSONResponse(content=list_audio_devices())
        except Exception:
            logger.exception("GET /api/audio/devices 异常", extra={"event": "web_error"})
            return _error(500, "internal_error", "音频设备枚举失败")

    # ------------------------------------------------------------------
    # POST /api/audio/test_output —— 测试设备发声（功能需求）
    # body 可选：{device: <index 或 name>, duration: 1.0}；device 缺省用当前
    # config.speech.output.device。播放 440Hz 正弦波（不依赖 TTS 引擎/网络），
    # 用 executor 线程执行避免阻塞事件循环；设备不可用/采样率不支持返回清晰错误
    # （含 VB-Audio 安装提示）。
    # ------------------------------------------------------------------
    @app.post("/api/audio/test_output")
    async def api_audio_test_output(request: Request) -> JSONResponse:
        body: dict[str, Any] = {}
        if request.headers.get("content-type", "").startswith("application/json"):
            try:
                parsed = await request.json()
                if isinstance(parsed, dict):
                    body = parsed
            except Exception:
                body = {}
        # 缺省用当前配置输出设备（空串 = 系统默认设备）
        device = body.get("device")
        if device is None or str(device).strip() == "":
            device = config.speech.output.device
        try:
            duration = float(body.get("duration", 1.0))
        except (TypeError, ValueError):
            duration = 1.0
        duration = max(0.1, min(duration, 5.0))  # 安全上限：防误填超长播放
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None, _play_test_tone_blocking, device, duration
            )
        except Exception:
            logger.exception("POST /api/audio/test_output 异常", extra={"event": "web_error"})
            return _error(500, "audio_test_failed", "测试音播放失败（内部异常）")
        if result.get("ok"):
            return JSONResponse(
                content={
                    "ok": True,
                    "device": str(device),
                    "message": result.get("message", "已播放测试音"),
                }
            )
        return _error(500, "audio_test_failed", result.get("error", "测试音播放失败"))

    # ------------------------------------------------------------------
    # GET /api/config —— 全量配置（敏感字段掩码），不含 .env keys 明文
    # ------------------------------------------------------------------
    @app.get("/api/config")
    async def api_config() -> JSONResponse:
        try:
            data = schema_mod.mask_config(config.model_dump())
            return JSONResponse(content={"config": data, "path": str(config.path), "reloadable": True})
        except Exception:
            logger.exception("GET /api/config 异常", extra={"event": "web_error"})
            return _error(500, "internal_error", "配置读取失败")

    # ------------------------------------------------------------------
    # GET /api/config/schema —— JSON Schema + 热重载/需重启/敏感元数据
    # ------------------------------------------------------------------
    @app.get("/api/config/schema")
    async def api_config_schema() -> JSONResponse:
        try:
            return JSONResponse(content=schema_mod.schema_payload())
        except Exception:
            logger.exception("GET /api/config/schema 异常", extra={"event": "web_error"})
            return _error(500, "internal_error", "Schema 生成失败")

    # ------------------------------------------------------------------
    # PUT /api/config —— 校验 → 备份 → 原子写盘 → reload（400 非法值不写盘）
    # 返回 restart_required_changed：本次保存中"需重启"字段实际变更的子集（A3）
    # ------------------------------------------------------------------
    def _dotted_get(data: dict[str, Any], path: str) -> Any:
        node: Any = data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node

    @app.put("/api/config")
    async def api_config_put(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return _error(400, "bad_request", "请求体必须是 JSON")
        if not isinstance(body, dict):
            return _error(400, "bad_request", "请求体必须是 JSON 对象")
        old_data = config.model_dump()
        try:
            result = schema_mod.save_config(config, body)
        except ConfigValidationError as exc:
            return _error(400, "validation_error", exc.detail)
        except ConfigWriteError as exc:
            return _error(500, "write_error", str(exc))
        except Exception:
            logger.exception("PUT /api/config 异常", extra={"event": "web_error"})
            return _error(500, "internal_error", "配置保存失败")
        # 热重载 A1：logging.level 即时应用（不重建 handlers，仅改根 logger 级别）
        try:
            from ..logging_setup import apply_logging_level

            apply_logging_level(config.logging.level)
        except Exception:
            logger.debug("应用日志级别失败（忽略）", exc_info=True)
        new_data = config.model_dump()
        changed = [
            p
            for p in result.restart_required
            if _dotted_get(old_data, p) != _dotted_get(new_data, p)
        ]
        return JSONResponse(
            content={
                "ok": result.ok,
                "reloaded": result.reloaded,
                "restart_required": result.restart_required,
                "restart_required_changed": changed,
                "path": result.path,
            }
        )

    # ------------------------------------------------------------------
    # POST /api/emergency（P1 预留，接口已设计）：发 EMERGENCY_STOP 事件
    # ------------------------------------------------------------------
    @app.post("/api/emergency")
    @app.post("/api/emergency_stop")
    async def api_emergency() -> JSONResponse:
        try:
            bus.publish_threadsafe(Event(EventType.EMERGENCY_STOP.value, {"source": "web"}))
            return JSONResponse(content={"ok": True, "event": EventType.EMERGENCY_STOP.value})
        except Exception:
            logger.exception("POST /api/emergency 异常", extra={"event": "web_error"})
            return _error(500, "internal_error", "急停发送失败")

    # ------------------------------------------------------------------
    # POST/GET /api/control/* —— 操控端点（任务 2：B1 控制 API）
    # 与主进程同进程共享实例：control 即 AgentLoop（或等价控制对象）。
    # 安全（B4）：仅本机可访问（host 默认 127.0.0.1）；操作异步执行并返回结果。
    # ------------------------------------------------------------------
    def _control() -> Any:
        ctrl = app.state.control
        if ctrl is None:
            raise LookupError("control 未接线（AgentLoop 未注入 WebServer）")
        return ctrl

    def _no_control() -> JSONResponse:
        return _error(503, "control_unavailable", "操控功能未接线（control 未注入）")

    @app.get("/api/control/status")
    async def api_control_status() -> JSONResponse:
        try:
            ctrl = _control()
        except LookupError:
            return _no_control()
        try:
            return JSONResponse(content=ctrl.control_status())
        except Exception:
            logger.exception("GET /api/control/status 异常", extra={"event": "web_error"})
            return _error(500, "internal_error", "控制状态获取失败")

    @app.post("/api/control/emergency_stop")
    async def api_control_emergency_stop() -> JSONResponse:
        try:
            _control()
        except LookupError:
            return _no_control()
        try:
            bus.publish_threadsafe(Event(EventType.EMERGENCY_STOP.value, {"source": "web"}))
            return JSONResponse(content={"ok": True, "action": "emergency_stop", "event": EventType.EMERGENCY_STOP.value})
        except Exception:
            logger.exception("POST /api/control/emergency_stop 异常", extra={"event": "web_error"})
            return _error(500, "internal_error", "急停发送失败")

    @app.post("/api/control/pause")
    async def api_control_pause() -> JSONResponse:
        try:
            ctrl = _control()
        except LookupError:
            return _no_control()
        try:
            ctrl.pause()
            return JSONResponse(content={"ok": True, "action": "pause", "paused": bool(ctrl.paused)})
        except Exception:
            logger.exception("POST /api/control/pause 异常", extra={"event": "web_error"})
            return _error(500, "internal_error", "暂停失败")

    @app.post("/api/control/resume")
    async def api_control_resume() -> JSONResponse:
        try:
            ctrl = _control()
        except LookupError:
            return _no_control()
        try:
            ctrl.resume()
            return JSONResponse(content={"ok": True, "action": "resume", "paused": bool(ctrl.paused)})
        except Exception:
            logger.exception("POST /api/control/resume 异常", extra={"event": "web_error"})
            return _error(500, "internal_error", "恢复失败")

    @app.post("/api/control/restart")
    async def api_control_restart() -> JSONResponse:
        """进程级重启 Agent（需求 1）：先返回响应，再延迟触发 control.restart()。

        时序：响应先 flush → 延迟后 ctrl.restart()（置位标志 + stop 主循环）→
        main.build_and_run finally 释放全部资源（web/急停/动作/MCP/视觉）→
        检测 restart_requested 后 subprocess.Popen 拉起新进程（重新加载配置）。
        新进程在旧端口释放后才启动，天然避免端口冲突。
        """
        try:
            ctrl = _control()
        except LookupError:
            return _no_control()
        try:
            delay = float(app.state.shutdown_delay_s)

            async def _delayed_restart() -> None:
                await asyncio.sleep(delay)
                try:
                    ctrl.restart()
                except Exception:
                    logger.exception("control.restart() 失败", extra={"event": "web_error"})

            asyncio.create_task(_delayed_restart())
            return JSONResponse(content={"ok": True, "action": "restart", "delay_s": delay})
        except Exception:
            logger.exception("POST /api/control/restart 异常", extra={"event": "web_error"})
            return _error(500, "internal_error", "重启失败")

    @app.post("/api/control/skip")
    async def api_control_skip() -> JSONResponse:
        """P2-3：跳过当前播放，自动取输出队列下一条。"""
        try:
            ctrl = _control()
        except LookupError:
            return _no_control()
        try:
            await ctrl.skip()
            qs = ctrl.output_slot.queue_size() if ctrl.output_slot else 0
            return JSONResponse(
                content={
                    "ok": True,
                    "action": "skip",
                    "queue_size": qs,
                    "message": f"已跳过当前播放，队列剩余 {qs} 条",
                }
            )
        except Exception:
            logger.exception("POST /api/control/skip 异常", extra={"event": "web_error"})
            return _error(500, "internal_error", "跳过失败")

    @app.post("/api/control/shutdown")
    async def api_control_shutdown() -> JSONResponse:
        """优雅关闭 Agent：先返回响应，再延迟触发主循环 stop()（走现有优雅关闭流程）。

        延迟保证 HTTP 响应有足够时间 flush；主循环 stop() 后 gather 返回，
        main.build_and_run 的 finally 依次关闭 web/emergency/actuator/mcp/sensor。
        """
        try:
            ctrl = _control()
        except LookupError:
            return _no_control()
        try:
            delay = float(app.state.shutdown_delay_s)

            async def _delayed_stop() -> None:
                await asyncio.sleep(delay)
                try:
                    ctrl.stop()
                except Exception:
                    logger.exception("control.stop() 失败", extra={"event": "web_error"})

            asyncio.create_task(_delayed_stop())
            return JSONResponse(content={"ok": True, "action": "shutdown", "delay_s": delay})
        except Exception:
            logger.exception("POST /api/control/shutdown 异常", extra={"event": "web_error"})
            return _error(500, "internal_error", "关闭失败")

    # ------------------------------------------------------------------
    # 长期记忆（标准版）：GET/POST/PUT/DELETE /api/memory
    # 无 MemoryStore 注入（未接线）时：GET 返回空列表 + disabled；写操作 503。
    # 鉴权同现状：host 默认 127.0.0.1 本机访问无鉴权。
    # ------------------------------------------------------------------
    def _memory_store() -> Any:
        mem = app.state.memory
        if mem is None:
            raise LookupError("memory 未接线（MemoryStore 未注入 WebServer）")
        return mem

    def _memory_config_snapshot() -> dict[str, Any]:
        """当前记忆配置快照（enabled/auto_extract/max_items，供前端开关回显）。"""
        try:
            cfg = config.brain.memory
            return {
                "enabled": bool(getattr(cfg, "enabled", True)),
                "auto_extract": bool(getattr(cfg, "auto_extract", True)),
                "max_items": int(getattr(cfg, "max_items", 200)),
            }
        except Exception:
            return {"enabled": True, "auto_extract": True, "max_items": 200}

    @app.get("/api/memory")
    async def api_memory_list(kind: str | None = None) -> JSONResponse:
        try:
            store = _memory_store()
        except LookupError:
            # 未接线：前端仍可展示空列表与"记忆功能未启用"状态
            return JSONResponse(
                content={"ok": True, "memories": [], "total": 0,
                         "enabled": False, "auto_extract": False, "max_items": 0}
            )
        try:
            items = store.list(kind=kind)
            return JSONResponse(content={"ok": True, "memories": items, "total": len(items), **_memory_config_snapshot()})
        except Exception:
            logger.exception("GET /api/memory 异常", extra={"event": "web_error"})
            return _error(500, "internal_error", "记忆列表获取失败")

    @app.post("/api/memory")
    async def api_memory_add(request: Request) -> JSONResponse:
        try:
            store = _memory_store()
        except LookupError:
            return _error(503, "memory_unavailable", "记忆功能未接线（MemoryStore 未注入）")
        try:
            body = await request.json()
        except Exception:
            return _error(400, "bad_request", "请求体必须是 JSON")
        if not isinstance(body, dict):
            return _error(400, "bad_request", "请求体必须是 JSON 对象")
        content = str(body.get("content", "")).strip()
        kind = str(body.get("kind", "fact")).strip() or "fact"
        if not content:
            return _error(400, "validation_error", "content 不能为空")
        try:
            mem_id = store.add(content, kind=kind)
        except Exception:
            logger.exception("POST /api/memory 异常", extra={"event": "web_error"})
            return _error(500, "internal_error", "记忆新增失败")
        if mem_id is None:
            return _error(500, "internal_error", "记忆写入失败（磁盘不可用）")
        return JSONResponse(content={"ok": True, "id": mem_id})

    @app.put("/api/memory/{mem_id}")
    async def api_memory_update(mem_id: int, request: Request) -> JSONResponse:
        try:
            store = _memory_store()
        except LookupError:
            return _error(503, "memory_unavailable", "记忆功能未接线（MemoryStore 未注入）")
        try:
            body = await request.json()
        except Exception:
            return _error(400, "bad_request", "请求体必须是 JSON")
        if not isinstance(body, dict):
            return _error(400, "bad_request", "请求体必须是 JSON 对象")
        content = str(body.get("content", "")).strip()
        if not content:
            return _error(400, "validation_error", "content 不能为空")
        kind = body.get("kind")
        try:
            ok = store.update(mem_id, content, kind=kind)
        except Exception:
            logger.exception("PUT /api/memory 异常", extra={"event": "web_error"})
            return _error(500, "internal_error", "记忆更新失败")
        if not ok:
            return _error(404, "not_found", f"记忆不存在: {mem_id}")
        return JSONResponse(content={"ok": True, "id": mem_id})

    @app.delete("/api/memory/{mem_id}")
    async def api_memory_delete(mem_id: int) -> JSONResponse:
        try:
            store = _memory_store()
        except LookupError:
            return _error(503, "memory_unavailable", "记忆功能未接线（MemoryStore 未注入）")
        try:
            ok = store.delete(mem_id)
        except Exception:
            logger.exception("DELETE /api/memory 异常", extra={"event": "web_error"})
            return _error(500, "internal_error", "记忆删除失败")
        if not ok:
            return _error(404, "not_found", f"记忆不存在: {mem_id}")
        return JSONResponse(content={"ok": True, "id": mem_id})

    # ------------------------------------------------------------------
    # 音色克隆（DashScope CosyVoice）：POST 创建 / GET 查询 / GET 列表 / POST 选中
    # 未注入 VoiceCloneManager 时：GET 返回空列表；克隆/选中返回 503。
    # 鉴权同现状：host 默认 127.0.0.1 本机访问无鉴权。
    # ------------------------------------------------------------------
    def _voice_clone() -> Any:
        vc = app.state.voice_clone
        if vc is None:
            raise LookupError("voice_clone 未接线（VoiceCloneManager 未注入 WebServer）")
        return vc

    @app.post("/api/voice/clone")
    async def api_voice_clone(request: Request) -> JSONResponse:
        """用公网可访问的参考音频 URL 创建克隆音色（阻塞轮询到就绪）。

        body：{audio_url, prefix?, target_model?} → 200 {ok, voice_id, status:"OK"}；
        失败（UNDEPLOYED/超时/无 key）→ 4xx/5xx 且 detail 含可定位原因。
        克隆超时后仍可用 GET /api/voice/clone/{voice_id} 查询实际状态。
        """
        try:
            vc = _voice_clone()
        except LookupError:
            return _error(503, "voice_clone_unavailable", "音色克隆未接线（VoiceCloneManager 未注入）")
        try:
            body = await request.json()
        except Exception:
            return _error(400, "bad_request", "请求体必须是 JSON")
        if not isinstance(body, dict):
            return _error(400, "bad_request", "请求体必须是 JSON 对象")
        audio_url = str(body.get("audio_url", "")).strip()
        prefix = str(body.get("prefix", "myvoice")).strip() or "myvoice"
        target_model = body.get("target_model")
        if not audio_url:
            return _error(400, "validation_error", "缺少参考音频 URL（body.audio_url，需公网可访问）")
        try:
            voice_id = await vc.create_voice(
                target_model=target_model, prefix=prefix, audio_url=audio_url
            )
        except Exception as exc:
            logger.exception("POST /api/voice/clone 失败", extra={"event": "web_error"})
            return _error(500, "voice_clone_failed", str(exc))
        return JSONResponse(content={"ok": True, "voice_id": voice_id, "status": "OK"})

    @app.get("/api/voice/clone/{voice_id}")
    async def api_voice_clone_status(voice_id: str) -> JSONResponse:
        """查询单个克隆音色状态（前端轮询用）：{ok, voice_id, status}。"""
        try:
            vc = _voice_clone()
        except LookupError:
            return _error(503, "voice_clone_unavailable", "音色克隆未接线（VoiceCloneManager 未注入）")
        try:
            result = await vc.query_voice_status(voice_id)
        except Exception:
            logger.exception("GET /api/voice/clone/{id} 失败", extra={"event": "web_error"})
            return _error(500, "voice_clone_query_failed", "克隆状态查询失败")
        return JSONResponse(content={"ok": True, **result})

    @app.get("/api/voice")
    async def api_voice_list() -> JSONResponse:
        """返回当前 voice_id + 已保存音色列表（供前端回显「设为当前音色」）。"""
        try:
            vc = _voice_clone()
        except LookupError:
            return JSONResponse(content={"ok": True, "current_voice_id": "", "voices": []})
        try:
            return JSONResponse(
                content={
                    "ok": True,
                    "current_voice_id": vc.get_voice_id(),
                    "voices": vc.list_voices(),
                }
            )
        except Exception:
            logger.exception("GET /api/voice 失败", extra={"event": "web_error"})
            return _error(500, "internal_error", "音色列表获取失败")

    @app.post("/api/voice/select")
    async def api_voice_select(request: Request) -> JSONResponse:
        """设为当前音色：更新 voices.json 并写回 config（供下次合成复用）。"""
        try:
            vc = _voice_clone()
        except LookupError:
            return _error(503, "voice_clone_unavailable", "音色克隆未接线（VoiceCloneManager 未注入）")
        try:
            body = await request.json()
        except Exception:
            return _error(400, "bad_request", "请求体必须是 JSON")
        if not isinstance(body, dict):
            return _error(400, "bad_request", "请求体必须是 JSON 对象")
        voice_id = str(body.get("voice_id", "")).strip()
        if not voice_id:
            return _error(400, "validation_error", "缺少 voice_id（body.voice_id）")
        try:
            vc.set_voice_id(voice_id)
        except Exception:
            logger.exception("POST /api/voice/select 失败", extra={"event": "web_error"})
            return _error(500, "voice_select_failed", "设为当前音色失败")
        return JSONResponse(content={"ok": True, "voice_id": voice_id, "current_voice_id": vc.get_voice_id()})

    return app


class WebServer:
    """内嵌 Web 服务生命周期：随机端口 + uvicorn task + 异常隔离 + 自动开浏览器。"""

    def __init__(
        self,
        bus: EventBus,
        config: Config,
        usage: UsageTracker,
        project_root: Path | str,
        dashboard=None,
        heartbeat_s: float = 15.0,
        control=None,
        shutdown_delay_s: float = 0.3,
        memory=None,
        voice_clone=None,
    ) -> None:
        self._bus = bus
        self._config = config
        self._usage = usage
        self._project_root = Path(project_root)
        self._dashboard = dashboard
        self._heartbeat_s = heartbeat_s
        self._control = control
        self._shutdown_delay_s = shutdown_delay_s
        self._memory = memory
        self._voice_clone = voice_clone
        self._app = create_app(
            bus,
            config,
            usage,
            self._project_root,
            dashboard=dashboard,
            heartbeat_s=heartbeat_s,
            control=control,
            shutdown_delay_s=shutdown_delay_s,
            memory=memory,
            voice_clone=voice_clone,
        )
        self.port: int = 0
        self._server: _EmbeddedServer | None = None
        self._sock: socket.socket | None = None

    @property
    def url(self) -> str:
        """实际监听地址（启动后有效）。"""
        return f"http://{self._config.web.host}:{self.port}"

    async def run(self) -> None:
        """运行 uvicorn（阻塞直到 shutdown 或崩溃重启一次；异常不冒泡）。"""
        host = self._config.web.host
        port_cfg = self._config.web.port
        try:
            sock, port = _bind_socket(host, port_cfg)
        except OSError:
            logger.error("Web 端口绑定失败（host=%s port=%s）", host, port_cfg, extra={"event": "web_error"})
            return
        self._sock = sock
        self.port = port

        logger.info("Web 控制台: %s", self.url, extra={"event": "web_started", "url": self.url})
        print(f"[Web 控制台] {self.url}", flush=True)
        if self._config.web.auto_open_browser:
            self._open_browser(self.url)

        # 正常启动（若崩溃：自动重启一次，复用同 socket；socket 已关闭则回退随机端口）
        try:
            await self._serve(sock)
        except Exception as exc:
            logger.error("Web server 崩溃: %s，自动重启一次", exc, extra={"event": "web_error"})
            if sock.fileno() == -1:
                try:
                    sock, port = _bind_socket(host, 0)
                except OSError:
                    logger.error("Web 重启端口绑定失败", extra={"event": "web_error"})
                    return
                self._sock = sock
                self.port = port
                logger.info("Web 控制台重启: %s", self.url, extra={"event": "web_started", "url": self.url})
                print(f"[Web 控制台] {self.url}", flush=True)
            try:
                await self._serve(sock)
            except Exception as exc2:
                logger.error("Web server 二次崩溃，放弃重启；主循环继续: %s", exc2, extra={"event": "web_error"})

    async def _serve(self, sock: socket.socket) -> None:
        """创建 _EmbeddedServer 并 serve（sockets 复用已绑定 sock）。"""
        server = _EmbeddedServer(
            uvicorn.Config(
                self._app,
                host=self._config.web.host,
                port=self.port,
                log_config=None,
                access_log=False,
                lifespan="off",
            )
        )
        self._server = server
        await server.serve(sockets=[sock])

    def shutdown(self) -> None:
        """请求优雅关闭 uvicorn server（由 main finally 调用）。"""
        server = self._server
        if server is not None:
            server.should_exit = True

    def _open_browser(self, url: str) -> None:
        """自动打开默认浏览器（失败仅记录，不阻断启动）。"""
        try:
            webbrowser.open(url)
        except Exception:
            logger.warning("自动打开浏览器失败: %s", url, extra={"event": "web_warn"})
