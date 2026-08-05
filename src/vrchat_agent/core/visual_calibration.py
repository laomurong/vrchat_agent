"""多步静默视觉校准循环：截图→LLM判断→微调，将目标物体对准屏幕正中。

校准期间不发 TTS、不回复文字、不触发自抑制窗口（静默）。
通过 asyncio task cancel 支持被新语音打断。

P1-1 收敛性增强：
1. **末次动作后确认截图**：每次微调后（含最后一次）追加确认截图 + LLM 判断——
   居中则停（记录"确认居中"），否则在预算内继续；
2. **步长按迭代衰减**：第 1 次微调用满力度（LLM 返回的 degrees），后续每次
   按 ``_STEP_DECAY`` 衰减（0.6^n），避免过冲、加速收敛；
3. **同方向连续微调上限**：同一水平/垂直方向连续出现超过 ``max_same_direction``
   次（默认 3）→ 判定振荡/无法收敛，停止该轴继续调整（防左→右→左死循环）；
4. **degrees 有界**：LLM 返回的 degrees 钳制到 [0, 60]，杜绝无界值导致转幅失控。
"""

from __future__ import annotations

import asyncio
import json
import logging

from ..brain.schema import Action, _extract_json_object
from ..brain.prompts import build_calibration_prompt

logger = logging.getLogger("vrchat_agent.loop.calibration")

# P1-1：步长按迭代衰减系数（第 1 次 1.0，之后每次 ×0.6）
_STEP_DECAY = 0.6
# P1-1：degrees 钳制上限（LLM 返回值过大时封顶，防转幅失控）
_MAX_DEGREES = 60.0
# P1-1：同方向连续微调上限默认值（超过判定振荡）
_DEFAULT_MAX_SAME_DIRECTION = 3


async def _ask_llm(target_desc: str, messages, sampler, vision, llm_client) -> dict | None:
    """截图 → LLM 判断 → 解析；失败返回 None（调用方决定收尾）。"""
    try:
        vision_b64 = await sampler.sample(vision)
    except Exception as exc:
        logger.warning("校准截图失败: %s", exc)
        return None
    if vision_b64 is None:
        logger.warning("校准截图为空")
        return None
    try:
        raw = await llm_client.chat_with_image(messages, vision_b64, temperature=0.0)
    except Exception as exc:
        logger.warning("校准 LLM 调用失败: %s", exc)
        return None
    try:
        obj = _extract_json_object(raw)
        data = json.loads(obj)
        return {
            "centered": bool(data.get("centered", False)),
            "offset_h": str(data.get("offset_h", "none")).strip().lower(),
            "offset_v": str(data.get("offset_v", "none")).strip().lower(),
            # P1-1：degrees 钳制到 [0, 60]（LLM 可能返回无界/负值）
            "degrees": float(max(0.0, min(_MAX_DEGREES, float(data.get("degrees", 10))))),
        }
    except Exception as exc:
        logger.warning("校准结果解析失败: %s, raw=%r", exc, (raw or "")[:200])
        return None


async def _visual_calibration_loop(
    target_desc: str,
    vision,
    sampler,
    llm_client,
    actuator,
    max_iterations: int = 3,
    max_same_direction: int = _DEFAULT_MAX_SAME_DIRECTION,
) -> None:
    """多步静默视觉校准循环。

    每次迭代：截图 → LLM 判断偏移 → 若未居中则构造 look:pan_*/tilt_* 动作微调；
    微调后追加确认截图（P1-1：居中则停，否则继续）；达到 max_iterations 上限或
    同方向连续微调超限时停止。被 asyncio.CancelledError 取消时自动退出，
    不抛异常（由外层 _run_calibration 消化）。

    Args:
        target_desc: 目标物体描述（从用户原始语音文本提取）。
        vision: VisionSensor 实例。
        sampler: ImageSampler 实例。
        llm_client: LLMClient 实例（需支持 chat_with_image 多模态调用）。
        actuator: Actuator 实例。
        max_iterations: 最大迭代次数（默认 3）。
        max_same_direction: 同方向连续微调上限（默认 3，防振荡）。
    """
    import base64 as _b64
    import io
    from PIL import Image

    prompt = build_calibration_prompt(target_desc)
    messages = [{"role": "user", "content": prompt}]

    # 1.5) 首次迭代前：从截图提取实际尺寸，注入 prompt（VLM 据此精确计算偏移角度）
    #      ——在第一次 LLM 调用前完成（与既有行为一致）。
    try:
        first_b64 = await sampler.sample(vision)
        if first_b64:
            img_data = _b64.b64decode(first_b64)
            w, h = Image.open(io.BytesIO(img_data)).size
        else:
            w, h = 0, 0
    except Exception:
        w, h = 0, 0
    prompt = build_calibration_prompt(target_desc, image_width=w, image_height=h)
    messages[0] = {"role": "user", "content": prompt}

    # P1-1：同方向连续计数（水平/垂直各一个）
    same_h_count = 0
    same_v_count = 0
    last_offset_h: str | None = None
    last_offset_v: str | None = None

    for i in range(max_iterations):
        logger.info("校准迭代 %d/%d：目标「%s」", i + 1, max_iterations, target_desc)

        # 1) 截图 + LLM 判断偏移
        data = await _ask_llm(target_desc, messages, sampler, vision, llm_client)
        if data is None:
            return
        centered = data["centered"]
        offset_h = data["offset_h"]
        offset_v = data["offset_v"]
        degrees = data["degrees"]

        # 2) 判断：已居中则停止
        if centered:
            logger.info("校准完成：目标「%s」已居中（第 %d 次）", target_desc, i + 1)
            return

        # 3) P1-1：同方向连续微调计数——超限判定振荡，停止该轴
        if offset_h == last_offset_h and offset_h in ("left", "right"):
            same_h_count += 1
        else:
            same_h_count = 1
        if offset_v == last_offset_v and offset_v in ("up", "down"):
            same_v_count += 1
        else:
            same_v_count = 1
        last_offset_h = offset_h
        last_offset_v = offset_v
        if same_h_count > max_same_direction or same_v_count > max_same_direction:
            logger.info(
                "校准停止：同方向连续微调 %d 次（水平 %s=%d，垂直 %s=%d），判定振荡无法收敛",
                max_same_direction, offset_h, same_h_count, offset_v, same_v_count,
            )
            return

        # 4) P1-1：步长按迭代衰减（第 1 次 1.0，后续 ×0.6^iter）
        decay = _STEP_DECAY ** i
        deg = max(0.0, float(degrees))
        axis_value = min(0.3 + deg * 0.03, 1.0) * decay          # 偏移越大转幅越大（封顶 1.0）
        axis_duration = (0.4 + min(deg, 30.0) * 0.01) * decay    # 偏移越大按住越久（封顶 ~0.7s）
        actions: list[Action] = []
        if offset_h == "left":
            actions.append(Action(
                kind="look",
                params={"name": "pan_left", "value": round(axis_value, 3), "duration": round(axis_duration, 3)},
                priority=5,
            ))
        elif offset_h == "right":
            actions.append(Action(
                kind="look",
                params={"name": "pan_right", "value": round(axis_value, 3), "duration": round(axis_duration, 3)},
                priority=5,
            ))

        if offset_v == "up":
            actions.append(Action(
                kind="look",
                params={"name": "tilt_down", "value": round(axis_value, 3), "duration": round(axis_duration, 3)},
                priority=5,
            ))
        elif offset_v == "down":
            actions.append(Action(
                kind="look",
                params={"name": "tilt_up", "value": round(axis_value, 3), "duration": round(axis_duration, 3)},
                priority=5,
            ))

        if not actions:
            logger.info("校准完成：无明确偏移方向（第 %d 次）", i + 1)
            return

        logger.info(
            "校准微调: offset_h=%s offset_v=%s degrees=%d decay=%.3f value=%.3f duration=%.3fs actions=%s",
            offset_h, offset_v, degrees, decay, axis_value, axis_duration,
            [a.params.get("name") for a in actions],
        )

        # 5) 执行微调（顺序执行，保证水平/垂直不冲突）。
        #    A-15：优先走 actuator._call_tool_safe("look_axis", ...) 直接下发按 degrees
        #    缩放后的 value/duration（MCPOSCActuator 提供）；NoopActuator（桥接降级）
        #    无此方法 → 回退 execute(语义动作，固定步长)，保证两型执行器都不崩溃。
        try:
            look_caller = getattr(actuator, "_call_tool_safe", None)
            for action in actions:
                if look_caller is not None:
                    name = action.params.get("name")
                    # 与 mappings/avatar_default.yaml 的 look:pan_*/tilt_* 符号约定一致：
                    # pan_left / tilt_down 为负，pan_right / tilt_up 为正。
                    sign = -1.0 if name in ("pan_left", "tilt_down") else 1.0
                    axis = "horizontal" if name in ("pan_left", "pan_right") else "vertical"
                    await look_caller("look_axis", {
                        "axis": axis,
                        "value": round(sign * axis_value, 3),
                        "duration": round(axis_duration, 3),
                    })
                else:
                    await actuator.execute(action)
        except Exception as exc:
            logger.warning("校准动作执行失败: %s", exc)
            return

        # 让画面稳定后再进入下一次迭代 / 确认
        await asyncio.sleep(0.8)

        # 6) P1-1：微调后确认截图——居中则停；否则预算内继续（末次动作后不追加动作）
        confirm = await _ask_llm(target_desc, messages, sampler, vision, llm_client)
        if confirm is None:
            return
        if confirm["centered"]:
            logger.info("校准完成：目标「%s」微调后确认居中（第 %d 次后）", target_desc, i + 1)
            return
        logger.debug("校准确认未居中（第 %d 次微调后），继续", i + 1)

    logger.info("校准循环达到上限 %d 次，停止", max_iterations)
