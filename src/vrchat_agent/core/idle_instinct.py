"""空闲本能微动作循环（Idle Instinct Loop）。

IDLE 状态下周期性发送微小 OSC 动作，模拟人类"活着"的感觉——
不调用 LLM，纯随机触发。灵感来自 Pomelo32141/VRchat_api_agent 的 idle loop。

微动作策略（每个 tick 独立随机）：
  - 视角微抖（look jitter）：50%，随机方向 left/right/up/down，1~3°
  - 过冲修正（overshoot）：10%（叠在 jitter 上），先转多再拉回
  - 停顿（hesitate）：30%，本 tick 什么都不做
  - 小步移动（small step move）：15%，极短移动 0.08~0.15s
  - 等待：其余，什么都不做

防重复机制：对每轮动作生成签名，与最近 5 次比对，重复则跳过；
连续 3 次跳过则强制变异为反方向动作。

安全约束：每 tick 最多 2 个动作；所有动作通过 actuator._call_tool_safe 执行。
"""

from __future__ import annotations

import asyncio
import logging
import random

from ..config import IdleInstinctConfig

logger = logging.getLogger("vrchat_agent.idle_instinct")

# ---- 视角微抖所用的 look_axis 工具名 ----
_LOOK_AXIS_TOOL = "look_axis"
# ---- 小步移动所用的 move_axis 工具名 ----
_MOVE_AXIS_TOOL = "move_axis"

# 微抖动典型 duration（秒）：极短，肉眼几乎不可见
_JITTER_DURATION = 0.05

# 过冲的额外角度范围（度）
_OVERSHOOT_EXTRA_MIN = 2.0
_OVERSHOOT_EXTRA_MAX = 3.0
# 拉回的角度范围（度）
_PULLBACK_MIN = 1.0
_PULLBACK_MAX = 2.0

# 过冲与拉回的 duration（秒）
_OVERSHOOT_DURATION = 0.06
_PULLBACK_DURATION = 0.04

# 小步移动的基础 value（绝对值）
_MOVE_BASE_VALUE = 0.4

# 签名历史长度
_SIGNATURE_HISTORY = 5
# 连续跳过阈值（超过后强制变异）
_MAX_CONSECUTIVE_SKIPS = 3

# 各动作独立触发概率（非 hesitate 时才走这些分支）
_JITTER_PROB = 0.50
_OVERSHOOT_PROB = 0.10  # 叠在 jitter 上
_HESITATE_PROB = 0.30
_SMALL_MOVE_PROB = 0.15


class IdleInstinct:
    """空闲本能微动作循环。

    IDLE 状态下周期性发送微小 OSC 动作（视角微抖、小步移动等），
    模拟人类"活着"的感觉——不调用 LLM，纯随机触发。

    用法::

        instinct = IdleInstinct(actuator, mapper, config)
        await instinct.start()   # 进入 IDLE 后启动
        await instinct.stop()    # 新 utterance / 急停时打断
    """

    def __init__(self, actuator, action_mapper, config: IdleInstinctConfig) -> None:
        """初始化空闲本能。

        Args:
            actuator: MCPOSCActuator 实例（需有 _call_tool_safe 方法）。
            action_mapper: ActionMapper 实例（保留以备后续扩展）。
            config: 空闲本能配置。
        """
        self._actuator = actuator
        self._mapper = action_mapper
        self._cfg = config
        # 取消事件：新 utterance 来 / 急停时 set，循环在下次检查点退出
        self._cancel_event = asyncio.Event()
        # 后台循环任务引用
        self._loop_task: asyncio.Task | None = None
        # 防重复签名队列（最近 SIGNATURE_HISTORY 次 tick 的签名）
        self._recent_signatures: list[str] = []
        # 连续跳过计数
        self._consecutive_skips: int = 0

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """启动本能循环（_maybe_idle 进入 IDLE 后调用）。

        如果已有循环在跑，先取消旧的再启动新的（幂等）。
        enabled=False 时直接跳过。
        """
        await self.stop()
        if not self._cfg.enabled:
            logger.debug("空闲本能已禁用（config agent.idle_instinct.enabled=false），跳过启动")
            return
        self._cancel_event.clear()
        self._loop_task = asyncio.create_task(self.run_loop(), name="idle_instinct_loop")
        logger.info("空闲本能循环已启动（interval=%.1f~%.1fs）",
                     self._cfg.interval_min_sec, self._cfg.interval_max_sec)

    async def stop(self) -> None:
        """停止本能循环（新 utterance 来 / 急停时调用）。"""
        self._cancel_event.set()
        task = self._loop_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._loop_task = None
        self._recent_signatures.clear()
        self._consecutive_skips = 0
        logger.debug("空闲本能循环已停止")

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    async def run_loop(self) -> None:
        """主循环：每隔 interval_min~max 秒随机触发一波微动作。

        在每次 tick 之前检查 cancel_event —— 被打断立即退出。
        """
        while not self._cancel_event.is_set():
            # 随机等待间隔
            interval = random.uniform(
                self._cfg.interval_min_sec, self._cfg.interval_max_sec
            )
            try:
                await asyncio.wait_for(self._cancel_event.wait(), timeout=interval)
                # cancel_event 被 set → 退出
                return
            except asyncio.TimeoutError:
                pass  # 正常超时，执行一次 tick

            if self._cancel_event.is_set():
                return

            await self._tick()

    # ------------------------------------------------------------------
    # 单次 tick
    # ------------------------------------------------------------------

    async def _tick(self) -> None:
        """单次 tick：根据概率决定执行什么微动作。"""
        actions: list[tuple[str, dict]] = []  # (tool_name, args)

        # ---- 停顿（30%）：本 tick 什么都不做 ----
        if random.random() < self._cfg.hesitate_prob:
            logger.debug("空闲本能 tick：停顿（hesitate）")
            return

        # ---- 视角微抖（50%）----
        jitter_fired = random.random() < _JITTER_PROB
        if jitter_fired:
            # 随机方向：left/right/up/down
            direction = random.choice(["left", "right", "up", "down"])
            # 过冲检查（10%，叠在 jitter 上）
            overshoot = random.random() < self._cfg.look_overshoot_prob

            if direction in ("left", "right"):
                axis = "horizontal"
                sign = -1 if direction == "left" else 1
            else:
                axis = "vertical"
                sign = 1 if direction == "up" else -1

            # 基础 jitter 角度 → look_axis value
            deg = random.uniform(self._cfg.jitter_min_deg, self._cfg.jitter_max_deg)
            base_value = self._deg_to_axis_value(deg) * sign

            if overshoot:
                # 过冲：先转多 2~3°，再拉回
                overshoot_deg = deg + random.uniform(_OVERSHOOT_EXTRA_MIN, _OVERSHOOT_EXTRA_MAX)
                overshoot_value = self._deg_to_axis_value(overshoot_deg) * sign
                actions.append((
                    _LOOK_AXIS_TOOL,
                    {"axis": axis, "value": round(overshoot_value, 4), "duration": _OVERSHOOT_DURATION},
                ))
                # 拉回（反方向小幅度）
                pullback_value = self._deg_to_axis_value(
                    random.uniform(_PULLBACK_MIN, _PULLBACK_MAX)
                ) * (-sign)
                actions.append((
                    _LOOK_AXIS_TOOL,
                    {"axis": axis, "value": round(pullback_value, 4), "duration": _PULLBACK_DURATION},
                ))
                logger.debug("空闲本能 tick：过冲抖动 %s overshoot=%.1f° pullback", direction, overshoot_deg)
            else:
                actions.append((
                    _LOOK_AXIS_TOOL,
                    {"axis": axis, "value": round(base_value, 4), "duration": _JITTER_DURATION},
                ))
                logger.debug("空闲本能 tick：微抖 %s %.1f°", direction, deg)

        # ---- 小步移动（15%）----
        if random.random() < self._cfg.small_step_move_prob:
            move_dir = random.choice(["forward", "backward", "left", "right"])
            if move_dir in ("forward", "backward"):
                axis = "vertical"
                value = _MOVE_BASE_VALUE if move_dir == "forward" else -_MOVE_BASE_VALUE
            else:
                axis = "horizontal"
                value = -_MOVE_BASE_VALUE if move_dir == "left" else _MOVE_BASE_VALUE
            duration = random.uniform(
                self._cfg.move_duration_min_sec, self._cfg.move_duration_max_sec
            )
            actions.append((
                _MOVE_AXIS_TOOL,
                {"axis": axis, "value": value, "duration": round(duration, 3)},
            ))
            logger.debug("空闲本能 tick：小步移动 %s %.3fs", move_dir, duration)

        # ---- 安全约束：每 tick 最多 2 个动作 ----
        if len(actions) > 2:
            actions = actions[:2]

        if not actions:
            return

        # ---- 防重复签名 ----
        signature = self._build_signature(actions)
        if self._is_duplicate(signature):
            self._consecutive_skips += 1
            if self._consecutive_skips >= _MAX_CONSECUTIVE_SKIPS:
                # 强制变异：反方向替代
                actions = self._force_alternative(actions)
                self._consecutive_skips = 0
                signature = self._build_signature(actions)
                logger.debug("空闲本能：连续 %d 次重复，强制变异 → %s",
                             _MAX_CONSECUTIVE_SKIPS, signature)
            else:
                logger.debug("空闲本能：跳过重复签名 %s（连续 %d 次）",
                             signature, self._consecutive_skips)
                return
        else:
            self._consecutive_skips = 0

        # 记录签名
        self._recent_signatures.append(signature)
        if len(self._recent_signatures) > _SIGNATURE_HISTORY:
            self._recent_signatures.pop(0)

        # ---- 执行动作（通过 actuator._call_tool_safe 安全调用）----
        for tool, args in actions:
            if self._cancel_event.is_set():
                return
            await self._actuator._call_tool_safe(tool, args)

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _build_signature(actions: list[tuple[str, dict]]) -> str:
        """生成动作签名：tool_name 用 '-' 连接。"""
        return "-".join(tool for tool, _ in actions)

    def _is_duplicate(self, signature: str) -> bool:
        """签名是否在最近 SIGNATURE_HISTORY 次中出现过。"""
        return signature in self._recent_signatures

    @staticmethod
    def _force_alternative(
        actions: list[tuple[str, dict]],
    ) -> list[tuple[str, dict]]:
        """强制生成备选动作：将所有轴类工具调用的 value 取反。"""
        new_actions: list[tuple[str, dict]] = []
        for tool, args in actions:
            new_args = dict(args)
            if "value" in new_args:
                new_args["value"] = -new_args["value"]
            new_actions.append((tool, new_args))
        return new_actions

    @staticmethod
    def _deg_to_axis_value(deg: float) -> float:
        """将角度（°）近似转换为 look_axis 的 value 参数。

        基于实测估算：look_axis 内部走 VRChat 摄像机旋转（~130°/s 桌面默认），
        angle ≈ |value| × 130°/s × duration(s)。
        对于微抖动典型 duration=0.05s：value ≈ deg / (130 × 0.05) ≈ deg / 6.5。
        """
        return deg / 6.5
