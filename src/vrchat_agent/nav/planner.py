"""空间导航：路径规划器（NavPlanner）。

在拓扑图上做**指纹匹配 + BFS 最短路径 + 每步重定位**：
1. 指纹匹配：当前截图指纹 → 拓扑图中最相似节点（相似度超阈值才认定"在此节点"，
   否则视为未知位置）；
2. BFS 路径规划：节点图上的无权最短路径（每步代价相同——相对动作序列）；
3. 每步重定位：沿边执行相对动作序列后，重新截图匹配——
   - 匹配到预期下一节点 → 继续；
   - 匹配到其它已知节点 → 就地重规划（修正偏差）；
   - 仍在本节点或未知 → 重试该边动作（最多 N 次），超限放弃。

截图与动作执行通过可注入协程注入（``capture`` / ``run_step``），
测试全部 mock，不真截图、不发 OSC。

作者：寇豆码（Kou）· 工程师
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from .landmark import LandmarkFingerprinter
from .topology import TopologyStore

logger = logging.getLogger("vrchat_agent.nav.planner")

# 类型别名：异步截图提供者（返回指纹 bytes，失败返回 None）
CaptureFn = Callable[[], Awaitable[bytes | None]]
# 类型别名：异步单步动作执行器（step: {"tool": ..., "args": {...}}）
RunStepFn = Callable[[dict], Awaitable[None]]

# P0-3：全局步数上限默认值（总执行步数超过即判定导航失败，防 A↔B 无限循环）
_DEFAULT_MAX_TOTAL_STEPS = 50
# P0-3：振荡检测窗口默认值（最近 N 个定位节点仅在 2 个节点间来回 → 判定振荡）
_DEFAULT_OSCILLATION_WINDOW = 6


@dataclass
class NavigateResult:
    """一次导航执行的结果。"""

    success: bool
    path: list[int] = field(default_factory=list)      # 途经节点 id 序列（含起终点）
    steps_taken: int = 0                                # 实际执行的动作步数
    retries: int = 0                                    # 重定位重试次数
    message: str = ""                                   # 人类可读说明（日志/测试断言）

    def __bool__(self) -> bool:
        return self.success


class NavPlanner:
    """拓扑导航规划器：匹配当前节点 + BFS 寻路 + 逐步重定位执行。"""

    def __init__(
        self,
        store: TopologyStore,
        fingerprinter: LandmarkFingerprinter,
        similarity_threshold: float = 0.85,
        max_relocalize_attempts: int = 3,
        max_total_steps: int = _DEFAULT_MAX_TOTAL_STEPS,
        oscillation_window: int = _DEFAULT_OSCILLATION_WINDOW,
    ) -> None:
        """初始化规划器。

        Args:
            store: 拓扑图存储。
            fingerprinter: 指纹器（截图 → 指纹 bytes，相似度比较）。
            similarity_threshold: 指纹匹配阈值（[0,1]，越大越严格）。
            max_relocalize_attempts: 每步重定位最大重试次数。
            max_total_steps: 全局总步数上限（P0-3，默认 50），超过即失败。
            oscillation_window: 振荡检测窗口（P0-3，默认 6）——最近 N 个定位节点
                只在 2 个节点间来回交替 → 判定振荡（A↔B 死循环），放弃导航。
        """
        self._store = store
        self._fpr = fingerprinter
        self._threshold = max(0.0, min(1.0, float(similarity_threshold)))
        self._max_retries = max(0, int(max_relocalize_attempts))
        self._max_total_steps = max(1, int(max_total_steps))
        self._osc_window = max(3, int(oscillation_window))

    # ------------------------------------------------------------------
    # 指纹匹配
    # ------------------------------------------------------------------
    def match_node(self, fingerprint: bytes) -> tuple[int | None, float]:
        """把当前帧指纹匹配到拓扑图中最相似的节点。

        Args:
            fingerprint: 当前帧指纹。

        Returns:
            (node_id, similarity)：相似度超阈值返回命中节点 id；
            否则返回 (None, 最高相似度)。
        """
        best_id: int | None = None
        best_sim = 0.0
        for node in self._store.all_nodes():
            sim = self._fpr.similarity(fingerprint, node["fingerprint"])
            if sim > best_sim:
                best_sim = sim
                best_id = node["id"]
        if best_id is not None and best_sim >= self._threshold:
            return best_id, best_sim
        return None, best_sim

    def find_node_by_description(self, target: str) -> int | None:
        """按描述查找节点（"去往喷泉"→ 描述含"喷泉"的节点）。

        精确匹配优先，其次子串匹配；找不到返回 None。
        """
        t = (target or "").strip()
        if not t:
            return None
        nodes = self._store.all_nodes()
        for node in nodes:
            if (node["description"] or "").strip() == t:
                return node["id"]
        for node in nodes:
            if t in (node["description"] or ""):
                return node["id"]
        return None

    # ------------------------------------------------------------------
    # BFS 最短路径
    # ------------------------------------------------------------------
    def plan_path(self, start_id: int, goal_id: int) -> list[int]:
        """BFS 求节点图无权最短路径（含起终点）。

        Args:
            start_id: 起点节点 id。
            goal_id: 终点节点 id。

        Returns:
            路径节点 id 列表（如 [1, 3, 5]）；不可达/起点即终点时：
            起点==终点返回 [start_id]，不可达返回 []。
        """
        start_id = int(start_id)
        goal_id = int(goal_id)
        if start_id == goal_id:
            return [start_id]
        prev: dict[int, int] = {start_id: -1}
        queue: deque[int] = deque([start_id])
        while queue:
            cur = queue.popleft()
            for nxt, _ in self._store.neighbors(cur):
                if nxt in prev:
                    continue
                prev[nxt] = cur
                if nxt == goal_id:
                    return self._reconstruct(prev, start_id, goal_id)
                queue.append(nxt)
        return []

    @staticmethod
    def _reconstruct(prev: dict[int, int], start_id: int, goal_id: int) -> list[int]:
        """从 prev 表回溯路径。"""
        path: list[int] = []
        cur = goal_id
        while cur != -1:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path

    def build_route(self, path: list[int]) -> list[dict]:
        """把节点路径展开为边动作序列（供执行/登记）。

        Args:
            path: 节点 id 序列（plan_path 的返回）。

        Returns:
            每段边的动作序列列表（扁平化）：[{"tool": ..., "args": ...}, ...]。
            缺边时跳过该段并 WARNING（拓扑不完整）。
        """
        route: list[dict] = []
        # zip 相邻节点对展开边动作序列（RUF007 建议 pairwise，此处 zip 更直观）
        for frm, to in zip(path[:-1], path[1:]):  # noqa: RUF007
            edge = self._store.get_edge(frm, to)
            if edge is None:
                logger.warning("路径缺少边 %d→%d（跳过）", frm, to)
                continue
            route.extend(edge["action_seq"])
        return route

    # ------------------------------------------------------------------
    # 每步重定位导航
    # ------------------------------------------------------------------
    async def navigate_to(
        self,
        capture: CaptureFn,
        run_step: RunStepFn,
        goal_id: int,
        start_id: int | None = None,
        max_retries: int | None = None,
    ) -> NavigateResult:
        """执行到目标节点的导航：逐边执行 + 每步重定位。

        流程：
        1. 截图 → 匹配当前节点（未提供 start_id 时；提供了则直接用，仍会截图核对）；
        2. BFS 规划到 goal 的路径；
        3. 沿路径逐边执行相对动作序列；每步后截图重定位：
           - 匹配到预期节点 → 前进；
           - 匹配到其它已知节点 → 以该节点为起点重规划（修正偏差）；
           - 未移动/未知 → 重试本边（最多 max_retries 次）。

        Args:
            capture: 异步截图 → 指纹（bytes | None）。
            run_step: 异步执行单步动作（{"tool", "args"}）。
            goal_id: 目标节点 id。
            start_id: 已知起点（可选；缺省时用截图匹配）。
            max_retries: 单边重试上限（缺省用构造期值）。

        Returns:
            NavigateResult（success / path / steps_taken / retries / message）。
        """
        max_retry = self._max_retries if max_retries is None else max(0, int(max_retries))

        # 1) 定位当前节点
        cur_id = start_id
        if cur_id is None:
            fp = await capture()
            if fp is None:
                return NavigateResult(False, message="无法获取当前画面指纹")
            cur_id, _ = self.match_node(fp)
            if cur_id is None:
                cur_id = self._store.add_node(fp, description="导航起点")

        goal_id = int(goal_id)
        # 2) BFS 规划
        path = self.plan_path(cur_id, goal_id)
        if not path:
            return NavigateResult(False, path=[cur_id], message=f"无可用路径（{cur_id} → {goal_id}）")
        if len(path) == 1:
            return NavigateResult(True, path=path, message="已在目标位置")

        steps_taken = 0
        retries = 0
        i = 0
        # P0-3：振荡检测——记录最近定位到的节点 id（含起点），用于识别 A↔B 来回
        recent_positions: deque[int | None] = deque(maxlen=self._osc_window)
        recent_positions.append(cur_id)
        # path 从 cur_id 开始；逐段推进（i 为当前段起点在 path 中的下标）
        while i < len(path) - 1:
            # P0-3：全局步数预算——总执行步数超过上限即失败（防 A↔B 无限循环）
            if steps_taken > self._max_total_steps:
                return NavigateResult(False, path=path, steps_taken=steps_taken,
                                      retries=retries,
                                      message=f"导航步数超过全局预算（>{self._max_total_steps} 步），放弃")
            frm, to = path[i], path[i + 1]
            edge = self._store.get_edge(frm, to)
            if edge is None:
                return NavigateResult(False, path=path, steps_taken=steps_taken,
                                      retries=retries, message=f"缺少边 {frm}→{to}")
            action_seq = edge["action_seq"]
            if not action_seq:
                logger.warning("边 %d→%d 动作为空（直接前移）", frm, to)

            # 3) 执行本边动作
            try:
                for step in action_seq:
                    await run_step(step)
                    steps_taken += 1
            except Exception as exc:  # noqa: BLE001 - 单步失败终止本边并返回失败（由调用方决定是否重试）
                return NavigateResult(False, path=path, steps_taken=steps_taken,
                                      retries=retries, message=f"执行边动作失败: {exc}")

            # 4) 每步重定位
            fp = await capture()
            if fp is None:
                return NavigateResult(False, path=path, steps_taken=steps_taken,
                                      retries=retries, message="执行后无法获取画面指纹")
            matched, sim = self.match_node(fp)
            # P0-3：记录本次定位节点，检测振荡（最近 N 步只在 2 个节点间来回交替）
            recent_positions.append(matched if matched is not None else frm)
            if self._detect_oscillation(recent_positions):
                return NavigateResult(False, path=path, steps_taken=steps_taken,
                                      retries=retries,
                                      message="检测到振荡（在 2 个节点间来回），放弃导航")
            if matched == to:
                # 到达预期下一节点 → 前进
                i += 1
                continue
            if matched is not None and matched != frm:
                # 偏离到其它已知节点（不是起点也不是终点）→ 就地重规划（修正偏差）
                logger.info("重定位偏差: 预期 %d，实际在 %d（sim=%.3f），重新规划", to, matched, sim)
                new_path = self.plan_path(matched, goal_id)
                if not new_path:
                    return NavigateResult(False, path=path, steps_taken=steps_taken,
                                          retries=retries, message=f"重规划失败（{matched} → {goal_id}）")
                path = new_path
                i = 0
                continue
            # 仍在起点（matched==frm）或未知位置 → 重试本边
            retries += 1
            if retries > max_retry:
                return NavigateResult(False, path=path, steps_taken=steps_taken,
                                      retries=retries, message=f"重定位失败（重试 {max_retry} 次仍在 {frm} 附近）")
            logger.info("重定位未前进: %d→%d sim=%.3f，重试第 %d/%d 次", frm, to, sim, retries, max_retry)

        return NavigateResult(True, path=path, steps_taken=steps_taken,
                              retries=retries, message=f"导航成功（途经 {len(path)} 个节点）")

    @staticmethod
    def _detect_oscillation(recent: deque[int | None]) -> bool:
        """检测最近 N 个定位节点是否在 2 个节点间来回交替（A↔B 振荡）。

        P0-3：窗口满且仅含 2 个互异节点、且相邻节点互不相同（纯交替）→ 振荡。
        起点未知（None）也参与判定（未知↔已知来回同样算振荡）。
        """
        if len(recent) < recent.maxlen:
            return False
        uniq = {x for x in recent}
        if len(uniq) != 2:
            return False
        items = list(recent)
        # 仅含 2 个互异节点且相邻互不相同 → 必然 A,B,A,B,... 交替
        return all(a != b for a, b in zip(items[:-1], items[1:]))
