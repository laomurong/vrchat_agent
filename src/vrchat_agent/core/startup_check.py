"""启动自检 + 自动校准（需求 2）：启动后异步后台执行，不阻塞装配与主循环。

设计原则：
- 全部异步 + 异常隔离：每一项检查失败只记日志，绝不让自检异常冒泡到
  build_and_run（不阻塞启动、不影响主循环）；
- 功能自检务实：LLM 只做配置完整性检查（不真发请求、不消耗 token），TTS 只
  记录引擎类型，视觉检查 window_capture 能否找到目标窗口，OSC 在桥接已启动时
  调用 get_avatar_name / 枚举参数；
- 自动校准只读不写：枚举当前 avatar 参数与映射表比对，发现缺失/变化记 WARNING
  日志（提示可运行 scripts/calibrate_actions.py），**不自动改写映射表**（避免
  破坏用户配置）；输出"校准报告"到日志与 Dashboard 自检区。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from ..config import Config

logger = logging.getLogger("vrchat_agent.check")


# ---------------------------------------------------------------------------
# 轻量结果解析（与 scripts/calibrate_actions.py 同款容错，独立实现避免脚本依赖）
# ---------------------------------------------------------------------------
def _extract_texts(result) -> list[str]:
    """从 MCP CallToolResult / dict 提取文本内容（容错）。"""
    if result is None:
        return []
    texts: list[str] = []
    if isinstance(result, dict):
        content = result.get("content") or []
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict):
                    texts.append(str(c.get("text", "")))
                elif hasattr(c, "text"):
                    texts.append(str(c.text))
                else:
                    texts.append(str(c))
        elif isinstance(content, str):
            texts.append(content)
        err = result.get("error")
        if err:
            texts.append(str(err))
    elif hasattr(result, "content"):
        content = getattr(result, "content", None) or []
        for c in content:
            texts.append(getattr(c, "text", str(c)))
    else:
        texts.append(str(result))
    return [t for t in texts if t.strip()]


def _walk_parameter_names(node, out: list[str]) -> None:
    """递归提取参数名：兼容 list[str] / list[dict] / dict{parameters: ...} 等形态。"""
    if isinstance(node, dict):
        for key in ("name", "parameter_name", "parameterName", "parameter"):
            if key in node and isinstance(node[key], str) and node[key].strip():
                out.append(node[key])
        for value in node.values():
            _walk_parameter_names(value, out)
    elif isinstance(node, list):
        for item in node:
            _walk_parameter_names(item, out)
    elif isinstance(node, str):
        s = node.strip()
        if s:
            out.append(s)


def _extract_parameters(result) -> list[str]:
    """从 get_avatar_parameters 结果提取参数名（容错：JSON 文本 / 裸文本 / 嵌套）。"""
    raw_names: list[str] = []
    for text in _extract_texts(result):
        data = None
        try:
            data = json.loads(text)
        except (TypeError, ValueError):
            data = text
        _walk_parameter_names(data, raw_names)
    seen: set[str] = set()
    names: list[str] = []
    for n in raw_names:
        if n not in seen:
            seen.add(n)
            names.append(n)
    return names


# ---------------------------------------------------------------------------
# 单项检查（每一项独立 try/except，异常隔离）
# ---------------------------------------------------------------------------
def _check_llm(config: Config) -> dict[str, Any]:
    """LLM 功能自检：仅配置完整性（不真发请求、不消耗 token）。"""
    llm = config.brain.llm
    ok = bool((llm.api_key or "").strip()) or bool((llm.base_url or "").strip())
    detail = f"provider={llm.provider} model={llm.model} base_url={llm.base_url or '-'}"
    if not ok:
        detail += "（未配置 api_key 且 base_url 为空，仅本地 Ollama/vLLM 可工作）"
    return {"ok": ok, "detail": detail}


def _check_tts(config: Config) -> dict[str, Any]:
    """TTS 功能自检：记录引擎类型与 key 配置（合成时自动回退 Edge，不预探测）。"""
    tts = config.speech.tts
    provider = tts.provider
    has_key = bool((tts.api_key or "").strip())
    detail = f"provider={provider} key={'已配置' if has_key else '未配置（失败自动回退 Edge 本地合成）'}"
    if provider == "clone":
        base_url = (tts.tts_clone.base_url or "").strip()
        detail += f" clone_base_url={base_url or '-'}"
    elif provider == "mimo":
        base_url = (tts.tts_mimo.base_url or "").strip()
        detail += f" mimo_base_url={base_url or '-'}"
    return {"ok": True, "detail": detail}


def _check_vision(config: Config, vision_sensor=None) -> dict[str, Any]:
    """视觉功能自检：window_capture 是否找到目标窗口（其余后端仅记录配置）。"""
    sensor_cfg = config.vision.sensor
    backend = sensor_cfg.backend
    if vision_sensor is None:
        return {
            "ok": False,
            "detail": f"backend={backend}（传感器未装配/启动失败，视觉降级为不可用）",
        }
    if backend == "window_capture":
        try:
            from ..vision.window_sensor import _find_window

            found = _find_window(sensor_cfg.window_title_keyword)
            if found is None:
                return {
                    "ok": False,
                    "detail": (
                        f"backend=window_capture 未找到标题包含「{sensor_cfg.window_title_keyword}」"
                        "的窗口（请确认 VRChat 已启动且窗口可见）"
                    ),
                }
            return {"ok": True, "detail": f"backend=window_capture 已找到窗口: {found[1]}"}
        except Exception as exc:
            return {"ok": False, "detail": f"backend=window_capture 检查失败: {exc}"}
    return {"ok": True, "detail": f"backend={backend}（已装配，启动时连接）"}


async def _check_osc(config: Config, mcp=None) -> dict[str, Any]:
    """OSC 桥接功能自检：桥接已启动则尝试 get_avatar_name。"""
    if mcp is None or not getattr(mcp, "started", False):
        return {"ok": False, "detail": "OSC 桥接未就绪（降级为仅语音无动作）"}
    try:
        result = await mcp.call_tool("get_avatar_name", {})
        avatar_text = " | ".join(_extract_texts(result)) or "（空返回）"
        return {"ok": True, "detail": f"桥接已就绪，avatar={avatar_text}"}
    except Exception as exc:
        return {"ok": False, "detail": f"桥接已启动但 get_avatar_name 失败: {exc}"}


async def _check_osc_calibration(config: Config, mcp=None, mapper=None) -> dict[str, Any]:
    """自动校准：枚举当前 avatar 参数 → 与映射表比对（只读不写）。

    - 桥接未就绪 / 无映射表 → 跳过（ok=False，detail 说明原因，不视为错误）；
    - 枚举失败 → 记 WARNING（不阻塞）；
    - 映射表参数在当前 avatar 缺失 → 记 WARNING，提示运行 calibrate_actions.py；
    - 全部一致 → INFO。
    """
    if mcp is None or not getattr(mcp, "started", False):
        return {"ok": False, "detail": "OSC 桥接未就绪，跳过自动校准"}
    avatar_text = ""
    try:
        avatar_result = await mcp.call_tool("get_avatar_name", {})
        avatar_text = " | ".join(_extract_texts(avatar_result)) or "（未获取）"
    except Exception as exc:
        avatar_text = f"（获取 avatar 名失败: {exc}）"
    try:
        params_result = await mcp.call_tool("get_avatar_parameters", {})
    except Exception as exc:
        logger.warning("自动校准：get_avatar_parameters 失败（跳过）: %s", exc, extra={"event": "calibration_failed"})
        return {"ok": False, "detail": f"get_avatar_parameters 失败: {exc}", "avatar": avatar_text}
    param_names = _extract_parameters(params_result)
    mapped = mapper.mapped_parameter_names() if mapper is not None else []
    missing = [p for p in mapped if p not in param_names]
    report: dict[str, Any] = {
        "ok": True,
        "detail": f"avatar={avatar_text}，枚举 {len(param_names)} 个参数，映射表引用 {len(mapped)} 个",
        "avatar": avatar_text,
        "parameters": param_names,
        "mapped": mapped,
        "missing": missing,
    }
    if missing:
        logger.warning(
            "自动校准：映射表 %d 个参数在当前 avatar 中缺失 %d 个: %s"
            "（如需更新映射请运行 scripts/calibrate_actions.py）",
            len(mapped),
            len(missing),
            missing,
            extra={"event": "calibration_warning", "missing": missing},
        )
    else:
        logger.info("自动校准：当前 avatar 参数与映射表一致（%d 个）", len(mapped))
    return report


async def _check_window_capture_consistency(
    config: Config, vision_sensor=None, sampler=None
) -> dict[str, Any]:
    """窗口截图尺寸一致性检查：对比 VRChat 窗口 client rect 与截图实际尺寸。

    差异 >10% 时 WARNING（提示校准可能不准），不阻断启动。
    """
    sensor_cfg = config.vision.sensor
    if sensor_cfg.backend != "window_capture":
        return {"ok": True, "detail": f"backend={sensor_cfg.backend}（非 window_capture，跳过尺寸一致性检查）"}
    if vision_sensor is None:
        return {"ok": False, "detail": "视觉传感器未装配，跳过尺寸一致性检查"}
    if sampler is None:
        return {"ok": False, "detail": "采样器未装配，跳过尺寸一致性检查"}

    try:
        import base64
        import io

        from PIL import Image

        from ..vision.window_sensor import _find_window, _get_window_rect

        # 获取 VRChat 窗口 rect
        found = _find_window(sensor_cfg.window_title_keyword)
        if found is None:
            return {"ok": False, "detail": f"未找到「{sensor_cfg.window_title_keyword}」窗口"}
        hwnd, title = found
        rect = _get_window_rect(hwnd)
        if rect is None:
            return {"ok": False, "detail": f"无法获取窗口「{title}」的屏幕矩形"}

        win_w = rect[2] - rect[0]
        win_h = rect[3] - rect[1]

        # 拍一张截图，获取实际图片尺寸
        b64 = await sampler.sample(vision_sensor)
        if b64 is None:
            return {"ok": False, "detail": f"窗口「{title}」截图失败，窗口尺寸={win_w}x{win_h}"}

        try:
            img_data = base64.b64decode(b64)
            img_w, img_h = Image.open(io.BytesIO(img_data)).size
        except Exception:
            return {"ok": False, "detail": f"截图解码失败，窗口尺寸={win_w}x{win_h}"}

        # 比较尺寸：差异 >10% 则 WARNING
        dw = abs(win_w - img_w) / max(win_w, 1)
        dh = abs(win_h - img_h) / max(win_h, 1)

        if dw > 0.10 or dh > 0.10:
            extra = {"event": "window_screenshot_mismatch",
                     "window": f"{win_w}x{win_h}",
                     "screenshot": f"{img_w}x{img_h}"}
            logger.warning(
                "截图尺寸(%dx%d)与游戏窗口(%dx%d)不一致，视觉校准可能不准。请保持窗口与截图区域一致。",
                img_w, img_h, win_w, win_h,
                extra=extra,
            )
            return {
                "ok": False,
                "detail": (
                    f"窗口「{title}」尺寸={win_w}x{win_h}，截图尺寸={img_w}x{img_h}，"
                    f"差异>10%"
                ),
            }

        logger.info(
            "窗口截图尺寸一致：窗口=%dx%d 截图=%dx%d",
            win_w, win_h, img_w, img_h,
            extra={"event": "window_screenshot_ok"},
        )
        return {
            "ok": True,
            "detail": f"窗口「{title}」尺寸={win_w}x{win_h}，截图尺寸={img_w}x{img_h}，一致",
        }
    except Exception as exc:
        return {"ok": False, "detail": f"尺寸一致性检查异常: {exc}"}


# ---------------------------------------------------------------------------
# 总入口
# ---------------------------------------------------------------------------
async def run_startup_checks(
    config: Config,
    bus=None,
    mcp=None,
    vision_sensor=None,
    mapper=None,
    sampler=None,
) -> dict[str, Any]:
    """执行启动自检 + 自动校准（异常隔离：单项失败只记日志，整体不抛异常）。

    返回报告 dict：
        {"started_at": ts, "checks": {llm/tts/vision/osc/calibration: {ok, detail, ...}}}

    bus 预留（后续可发布 startup_check 事件）；当前报告经返回值交给调用方
    （main 写入 Dashboard.startup_report 并记 INFO 日志）。
    """
    checks: dict[str, Any] = {}
    # LLM：同步纯配置检查，包一层异步兼容（与其它检查一致的调用形态）
    try:
        checks["llm"] = _check_llm(config)
    except Exception as exc:
        logger.warning("自检 llm 失败（忽略）: %s", exc, extra={"event": "startup_check_failed"})
        checks["llm"] = {"ok": False, "detail": f"检查异常: {exc}"}
    try:
        checks["tts"] = _check_tts(config)
    except Exception as exc:
        logger.warning("自检 tts 失败（忽略）: %s", exc, extra={"event": "startup_check_failed"})
        checks["tts"] = {"ok": False, "detail": f"检查异常: {exc}"}
    try:
        checks["vision"] = _check_vision(config, vision_sensor=vision_sensor)
    except Exception as exc:
        logger.warning("自检 vision 失败（忽略）: %s", exc, extra={"event": "startup_check_failed"})
        checks["vision"] = {"ok": False, "detail": f"检查异常: {exc}"}
    try:
        checks["osc"] = await _check_osc(config, mcp=mcp)
    except Exception as exc:
        logger.warning("自检 osc 失败（忽略）: %s", exc, extra={"event": "startup_check_failed"})
        checks["osc"] = {"ok": False, "detail": f"检查异常: {exc}"}
    try:
        checks["calibration"] = await _check_osc_calibration(config, mcp=mcp, mapper=mapper)
    except Exception as exc:
        logger.warning("自检 calibration 失败（忽略）: %s", exc, extra={"event": "startup_check_failed"})
        checks["calibration"] = {"ok": False, "detail": f"检查异常: {exc}"}
    # 窗口截图尺寸一致性检查（仅 window_capture 后端生效）
    try:
        checks["screenshot_size"] = await _check_window_capture_consistency(
            config, vision_sensor=vision_sensor, sampler=sampler
        )
    except Exception as exc:
        logger.warning("自检 screenshot_size 失败（忽略）: %s", exc, extra={"event": "startup_check_failed"})
        checks["screenshot_size"] = {"ok": False, "detail": f"检查异常: {exc}"}
    return {"started_at": time.time(), "checks": checks}
