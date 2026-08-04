"""多步静默视觉校准循环：截图→LLM判断→微调，将目标物体对准屏幕正中。

校准期间不发 TTS、不回复文字、不触发自抑制窗口（静默）。
通过 asyncio task cancel 支持被新语音打断。
"""

from __future__ import annotations

import asyncio
import json
import logging

from ..brain.schema import Action, _extract_json_object
from ..brain.prompts import build_calibration_prompt

logger = logging.getLogger("vrchat_agent.loop.calibration")


async def _visual_calibration_loop(
    target_desc: str,
    vision,
    sampler,
    llm_client,
    actuator,
    max_iterations: int = 3,
) -> None:
    """多步静默视觉校准循环。

    每次迭代：截图 → LLM 判断偏移 → 若未居中则构造 look:pan_*/tilt_* 动作微调。
    居中或达到 max_iterations 上限时停止。
    被 asyncio.CancelledError 取消时自动退出，不抛异常（由外层 _run_calibration 消化）。

    Args:
        target_desc: 目标物体描述（从用户原始语音文本提取）。
        vision: VisionSensor 实例。
        sampler: ImageSampler 实例。
        llm_client: LLMClient 实例（需支持 chat_with_image 多模态调用）。
        actuator: Actuator 实例。
        max_iterations: 最大迭代次数（默认 3）。
    """
    import base64 as _b64
    import io
    from PIL import Image

    prompt = build_calibration_prompt(target_desc)
    messages = [{"role": "user", "content": prompt}]

    for i in range(max_iterations):
        logger.info("校准迭代 %d/%d：目标「%s」", i + 1, max_iterations, target_desc)

        # 1) 截图
        try:
            vision_b64 = await sampler.sample(vision)
        except Exception as exc:
            logger.warning("校准截图失败: %s", exc)
            return

        if vision_b64 is None:
            logger.warning("校准截图为空")
            return

        # 1.5) 第一次迭代：从截图提取实际尺寸，注入 prompt（VLM 据此精确计算偏移角度）
        if i == 0:
            try:
                img_data = _b64.b64decode(vision_b64)
                w, h = Image.open(io.BytesIO(img_data)).size
            except Exception:
                w, h = 0, 0
            prompt = build_calibration_prompt(target_desc, image_width=w, image_height=h)
            messages[0] = {"role": "user", "content": prompt}

        # 2) LLM 判断偏移（temperature=0 保证确定性）
        try:
            raw = await llm_client.chat_with_image(messages, vision_b64, temperature=0.0)
        except Exception as exc:
            logger.warning("校准 LLM 调用失败: %s", exc)
            return

        # 3) 解析 JSON
        try:
            obj = _extract_json_object(raw)
            data = json.loads(obj)
            centered = bool(data.get("centered", False))
            offset_h = str(data.get("offset_h", "none")).strip().lower()
            offset_v = str(data.get("offset_v", "none")).strip().lower()
            degrees = int(data.get("degrees", 10))
        except Exception as exc:
            logger.warning("校准结果解析失败: %s, raw=%r", exc, (raw or "")[:200])
            return

        # 4) 判断：已居中则停止
        if centered:
            logger.info("校准完成：目标「%s」已居中（第 %d 次）", target_desc, i + 1)
            return

        # 5) 构造微调动作（look:pan_*/tilt_*，按 LLM 返回偏移角度缩放 value/duration）
        # A-15：此前 degrees 解析后丢弃、只用固定步长，收敛慢。现将 degrees 映射到
        # look_axis 的 value/duration（direction 由 offset 决定），按实际偏移量调整
        # 转幅，收敛更快；max_iterations 仍兜底。
        deg = max(0.0, float(degrees))
        axis_value = min(0.3 + deg * 0.03, 1.0)          # 偏移越大转幅越大（封顶 1.0）
        axis_duration = 0.4 + min(deg, 30.0) * 0.01      # 偏移越大按住越久（封顶 ~0.7s）
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
            "校准微调: offset_h=%s offset_v=%s degrees=%d value=%.3f duration=%.3fs actions=%s",
            offset_h,
            offset_v,
            degrees,
            axis_value,
            axis_duration,
            [a.params.get("name") for a in actions],
        )

        # 6) 执行微调（顺序执行，保证水平/垂直不冲突）。
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

        # 让画面稳定后再进入下一次迭代
        await asyncio.sleep(0.8)

    logger.info("校准循环达到上限 %d 次，停止", max_iterations)
