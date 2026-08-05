"""空间导航：自主探索器（NavExplorer）。

当 Agent 空闲且无目标时，向"未访问"方向移动并登记新节点——把陌生区域
逐步扩展进拓扑图。核心约束不变：无绝对坐标，只能相对动作序列 + 视觉指纹
（移动后截图，若与已知节点都不同 → 新节点；与已知节点相似 → 回环检测，
把"回到已知位置"也登记成边）。

优先级（需求）：nav 探索激活时，idle_instinct 的随机微动让位——由
AgentLoop._maybe_idle 判断：``NavController.maybe_explore_coro()`` 返回
探索协程时优先启动探索，否则才启动空闲本能。

方向策略（P0-2 增强）：
- **按节点记录已试方向**：每个节点记住哪些方向已探索过（``_tried_directions``），
  轮转时跳过已试方向——避免在同一节点反复走同一方向导致地图不增长；
- **死胡同/自环抑制**：移动后画面无明显变化（撞墙/被挡）或连续 N 次回到原位
  （自环）→ 判定阻塞，不登记自环边、换方向或放弃探索；
- 候选方向内置一组相对动作序列（左转+前进 / 右转+前进 / 直行 / 横移）。

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

# P0-2：默认"画面无明显变化"判定阈值（移动前后指纹相似度高于此值 → 撞墙/被挡）
_NO_CHANGE_SIM_THRESHOLD = 0.95
# P0-2：默认连续自环（回到原位）上限——超过判定阻塞，放弃探索（不登记自环边）
_MAX_SELF_LOOPS = 3


@dataclass
class ExploreResult:
    """一次探索的结果。"""

    success: bool
    new_node_id: int | None = None          # 登记的新节点 id（未移动则不新登记）
    matched_node_id: int | None = None      # 回环命中的已知节点 id
    moved: bool = False                     # 是否执行了移动动作
    message: str = ""                       # 人类可读说明（日志/测试断言）


class NavExplorer:
    """自主探索器：空闲时向未访问方向移动并登记节点/边。

    P0-2 增强：按节点记录已试方向 + 死胡同/自环抑制（拓扑图不再因自环边
    空转而不增长）。
    """

    def __init__(
        self,
        store: TopologyStore,
        planner: NavPlanner,
        fingerprinter: LandmarkFingerprinter,
        directions: list[list[dict]] | None = None,
        max_self_loops: int = _MAX_SELF_LOOPS,
        no_change_sim_threshold: float = _NO_CHANGE_SIM_THRESHOLD,
    ) -> None:
        """初始化探索器。

        Args:
            store: 拓扑图存储。
            planner: 规划器（用于指纹匹配当前节点）。
            fingerprinter: 指纹器。
            directions: 候选方向动作序列（默认 _DIRECTION_SEQS）。
            max_self_loops: 连续自环上限（默认 3），超过判定阻塞。
            no_change_sim_threshold: 移动前后"画面无明显变化"的相似度阈值
                （默认 0.95），高于此值视为撞墙/被挡（不登记自环边）。
        """
        self._store = store
        self._planner = planner
        self._fpr = fingerprinter
        self._directions = directions if directions is not None else list(_DIRECTION_SEQS)
        self._max_self_loops = max(1, int(max_self_loops))
        self._no_change_sim_threshold = float(no_change_sim_threshold)
        # 最近一次使用的方向下标（初始 -1 → 第一次取 0）
        self._last_direction_index: int = -1
        # P0-2：节点 id → 已试方向下标集合（每个节点记住哪些方向探索过）
        self._tried_directions: dict[int, set[int]] = {}
        # P0-2：连续自环计数（回到原位/画面无变化时 +1，新节点/回环到其它节点时清零）
        self._consecutive_self_loops: int = 0

    # ------------------------------------------------------------------
    # 方向选择（P0-2：按节点记忆已试方向）
    # ------------------------------------------------------------------
    def _next_action_seq(self, cur_id: int | None = None) -> list[dict]:
        """选择下一个探索方向。

        - cur_id 提供时：优先取该节点"未试过"的方向（轮转推进，跳过已试）；
          全部已试 → 回退普通轮转（重试）。
        - cur_id 为 None（兼容旧测试/外部调用）：普通轮转 + 跳过最近一次用过的方向。

        Returns:
            动作序列（list[dict]）；无可用方向返回 []。
        """
        n = len(self._directions)
        if n == 0:
            return []
        idx = self._pick_direction_index(cur_id, n)
        self._last_direction_index = idx
        return [dict(step) for step in self._directions[idx]]

    def _pick_direction_index(self, cur_id: int | None, n: int) -> int:
        """计算方向下标：优先未试方向；否则普通轮转。"""
        if cur_id is not None:
            tried = self._tried_directions.get(cur_id, set())
            untried = [i for i in range(n) if i not in tried]
            if untried:
                # 从最近使用方向之后找第一个未试方向；找不到则取最小的未试方向
                for off in range(1, n + 1):
                    cand = (self._last_direction_index + off) % n
                    if cand in untried:
                        return cand
                return min(untried)
        idx = (self._last_direction_index + 1) % n
        if n > 1 and idx == self._last_direction_index:
            idx = (idx + 1) % n
        return idx

    def _record_tried(self, cur_id: int, direction_idx: int) -> None:
        """记录节点 cur_id 已探索过方向 direction_idx（防重复走同一方向）。"""
        self._tried_directions.setdefault(cur_id, set()).add(direction_idx)

    # ------------------------------------------------------------------
    # 探索执行
    # ------------------------------------------------------------------
    async def explore_once(
        self,
        capture: CaptureFn,
        run_step: RunStepFn,
        cancel_event: asyncio.Event | None = None,
    ) -> ExploreResult:
        """执行一次探索：截图定位 → 向未访问方向移动 → 登记新节点或回环边。

        P0-2 自环抑制：
        - 移动前后画面无明显变化（相似度 > no_change_sim_threshold）→ 撞墙/被挡，
          不登记自环边；
        - 连续 max_self_loops 次回到原位/自环 → 判定阻塞，放弃探索（success=False）。

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

        # 2) 选择方向并执行（跳过该节点已试方向）
        action_seq = self._next_action_seq(cur_id)
        if not action_seq:
            return ExploreResult(False, message="无可用探索方向")
        direction_idx = self._last_direction_index
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

        # 记录本节点已试方向（无论结果，防止下次重复）
        self._record_tried(cur_id, direction_idx)

        if matched is not None and matched != cur_id:
            # 回环：回到了某个已知节点 → 登记边（探索闭环，有进展 → 清零自环计数）
            self._consecutive_self_loops = 0
            self._store.add_edge(cur_id, matched, action_seq)
            return ExploreResult(True, new_node_id=None, matched_node_id=matched,
                                 moved=moved, message=f"回环：回到节点 {matched}（sim={sim:.3f}）")

        if matched == cur_id:
            # 回到原位 / 画面无变化：
            # P0-2 判定移动前后画面是否无明显变化（撞墙/被挡）——用当前帧与移动前
            # 指纹的相似度；无变化时不登记自环边。
            no_change = self._fpr.similarity(fp, fp2) >= self._no_change_sim_threshold
            self._consecutive_self_loops += 1
            if self._consecutive_self_loops >= self._max_self_loops:
                self._consecutive_self_loops = 0  # 复位，允许后续换方向再试
                logger.info("探索受阻：连续 %d 次回到原位（节点 %d），放弃本次探索",
                            self._max_self_loops, cur_id)
                return ExploreResult(False, new_node_id=None, matched_node_id=cur_id,
                                     moved=moved,
                                     message=f"探索受阻：连续 {self._max_self_loops} 次回到原位（节点 {cur_id}）")
            if no_change:
                # 撞墙/被挡：画面没变 → 不登记自环边（拓扑图不因空转增加脏边）
                logger.info("探索无进展（画面无变化，疑似撞墙/被挡），不登记自环边（节点 %d）", cur_id)
                return ExploreResult(True, new_node_id=None, matched_node_id=cur_id,
                                     moved=moved, message=f"回到原位（节点 {cur_id}），画面无变化不登记自环")
            # 有画面变化但仍匹配本节点（转了方向又回来）→ 不登记自环边
            return ExploreResult(True, new_node_id=None, matched_node_id=cur_id,
                                 moved=moved, message=f"回到原位（节点 {cur_id}），未登记自环")

        # 新位置：登记新节点 + 边（有进展 → 清零自环计数）
        self._consecutive_self_loops = 0
        new_id = self._store.add_node(fp2, description=self._describe(cur_id))
        self._store.add_edge(cur_id, new_id, action_seq)
        return ExploreResult(True, new_node_id=new_id, matched_node_id=None,
                             moved=moved, message=f"新节点 {new_id}（自节点 {cur_id}）")

    def _describe(self, _cur_id: int | None) -> str:
        """为新节点生成默认描述（探索点序号）。"""
        return f"探索点{self._store.node_count() + 1}"
