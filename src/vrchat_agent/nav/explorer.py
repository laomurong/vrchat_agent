"""空间导航：自主探索器（NavExplorer）。

当 Agent 空闲且无目标时，向"未访问"方向移动并登记新节点——把陌生区域
逐步扩展进拓扑图。核心约束不变：无绝对坐标，只能相对动作序列 + 视觉指纹
（移动后截图，若与已知节点都不同 → 新节点；与已知节点相似 → 回环检测，
把"回到已知位置"也登记成边）。

优先级（需求）：nav 探索激活时，idle_instinct 的随机微动让位——由
AgentLoop._maybe_idle 判断：``NavController.maybe_explore_coro()`` 返回
探索协程时优先启动探索，否则才启动空闲本能。

方向策略：内置一组相对动作序列（左转+前进 / 右转+前进 / 直行 / 横移），
轮转选择并跳过最近一次用过的方向，近似"向未访问方向走"。

作者：寇豆码（Kou）· 工程师
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from .landmark import LandmarkFingerprinter
from .planner import CaptureFn, NavPlanner, RunStepFn
from .topology import TopologyStore

logger = logging.getLogger("vrchat_agent.nav.explorer")

# 候选探索方向（相对动作序列）：轮转选择，避免原地打转
# look_axis value/duration 与 mappings/avatar_default.yaml 的 look:pan_* 一致；
# move_axis 走 /input/Vertical|Horizontal 浮点轴，duration 后自动归零。
_DIRECTION_SEQS: list[list[dict]] = [
    # 左转 30° 再前进（探索左侧）
    [
        {"tool": "look_axis", "args": {"axis": "horizontal", "value": -0.6, "duration": 0.5}},
        {"tool": "move_axis", "args": {"axis": "vertical", "value": 0.8, "duration": 1.5}},
    ],
    # 右转 30° 再前进（探索右侧）
    [
        {"tool": "look_axis", "args": {"axis": "horizontal", "value": 0.6, "duration": 0.5}},
        {"tool": "move_axis", "args": {"axis": "vertical", "value": 0.8, "duration": 1.5}},
    ],
    # 直行
    [
        {"tool": "move_axis", "args": {"axis": "vertical", "value": 0.8, "duration": 1.5}},
    ],
    # 向右横移
    [
        {"tool": "move_axis", "args": {"axis": "horizontal", "value": 0.8, "duration": 1.0}},
    ],
    # 向左横移
    [
        {"tool": "move_axis", "args": {"axis": "horizontal", "value": -0.8, "duration": 1.0}},
    ],
]


@dataclass
class ExploreResult:
    """一次探索的结果。"""

    success: bool
    new_node_id: int | None = None          # 登记的新节点 id（未移动则不新登记）
    matched_node_id: int | None = None      # 回环命中的已知节点 id
    moved: bool = False                     # 是否执行了移动动作
    message: str = ""                       # 人类可读说明（日志/测试断言）


class NavExplorer:
    """自主探索器：空闲时向未访问方向移动并登记节点/边。"""

    def __init__(
        self,
        store: TopologyStore,
        planner: NavPlanner,
        fingerprinter: LandmarkFingerprinter,
        directions: list[list[dict]] | None = None,
    ) -> None:
        """初始化探索器。

        Args:
            store: 拓扑图存储。
            planner: 规划器（用于指纹匹配当前节点）。
            fingerprinter: 指纹器。
            directions: 候选方向动作序列（默认 _DIRECTION_SEQS）。
        """
        self._store = store
        self._planner = planner
        self._fpr = fingerprinter
        self._directions = directions if directions is not None else list(_DIRECTION_SEQS)
        # 最近一次使用的方向下标（初始 -1 → 第一次取 0）
        self._last_direction_index: int = -1

    def _next_action_seq(self) -> list[dict]:
        """选择下一个探索方向：轮转 + 跳过最近用过的方向。

        简单启发：把候选方向下标 +1 步进（模长度），若正好是最近用过的
        再 +1——近似"不重复同一方向"，不依赖任何外部状态。
        """
        n = len(self._directions)
        if n == 0:
            return []
        idx = (self._last_direction_index + 1) % n
        if n > 1 and idx == self._last_direction_index:
            idx = (idx + 1) % n
        self._last_direction_index = idx
        return [dict(step) for step in self._directions[idx]]

    async def explore_once(
        self,
        capture: CaptureFn,
        run_step: RunStepFn,
        cancel_event: asyncio.Event | None = None,
    ) -> ExploreResult:
        """执行一次探索：截图定位 → 向未访问方向移动 → 登记新节点或回环边。

        Args:
            capture: 异步截图 → 指纹（bytes | None）。
            run_step: 异步执行单步动作（{"tool", "args"}）。
            cancel_event: 可选取消事件（空闲探索被新语音/急停打断时置位）。

        Returns:
            ExploreResult（success / new_node_id / matched_node_id / moved / message）。
        """
        # 1) 截图定位当前节点
        fp = await capture()
        if fp is None:
            return ExploreResult(False, message="无法获取当前画面指纹")
        cur_id, _ = self._planner.match_node(fp)
        if cur_id is None:
            cur_id = self._store.add_node(fp, description=self._describe(cur_id))
            logger.info("探索：登记当前位置为新节点 id=%d", cur_id)

        # 2) 选择方向并执行
        action_seq = self._next_action_seq()
        if not action_seq:
            return ExploreResult(False, message="无可用探索方向")
        moved = False
        for step in action_seq:
            if cancel_event is not None and cancel_event.is_set():
                logger.info("探索被打断（cancel_event 置位）")
                break
            try:
                await run_step(step)
                moved = True
            except Exception as exc:  # noqa: BLE001 - 单步失败不阻断探索，记录后继续下一方向
                logger.warning("探索动作执行失败（跳过后续）: %s", exc)
                break

        # 3) 移动后重新截图 → 新节点 / 回环
        fp2 = await capture()
        if fp2 is None:
            return ExploreResult(False, message="移动后无法获取画面指纹")
        matched, sim = self._planner.match_node(fp2)
        if matched is not None and matched != cur_id:
            # 回环：回到了某个已知节点 → 登记边（探索闭环）
            self._store.add_edge(cur_id, matched, action_seq)
            return ExploreResult(True, new_node_id=None, matched_node_id=matched,
                                 moved=moved, message=f"回环：回到节点 {matched}（sim={sim:.3f}）")
        if matched == cur_id:
            # 移动后仍在本节点（画面没变/动作没生效）→ 登记自环边，不新登记
            self._store.add_edge(cur_id, cur_id, action_seq)
            return ExploreResult(True, new_node_id=None, matched_node_id=cur_id,
                                 moved=moved, message=f"回到原位（节点 {cur_id}）")
        # 新位置：登记新节点 + 边
        new_id = self._store.add_node(fp2, description=self._describe(cur_id))
        self._store.add_edge(cur_id, new_id, action_seq)
        return ExploreResult(True, new_node_id=new_id, matched_node_id=None,
                             moved=moved, message=f"新节点 {new_id}（自节点 {cur_id}）")

    def _describe(self, _cur_id: int | None) -> str:
        """为新节点生成默认描述（探索点序号）。"""
        return f"探索点{self._store.node_count() + 1}"
