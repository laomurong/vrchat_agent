"""空间导航包：拓扑导航（节点=视觉地标指纹，边=相对动作序列）。

VRChat OSC 没有绝对坐标——/input/* 只有相对转向，读不到自己/他人在世界里的
位置（调研确认）。因此空间导航只能做**拓扑导航**：

- ``landmark.LandmarkFingerprinter``：截图像素指纹（32×32 灰度均值池化 +
  颜色直方图），确定性、轻量，够区分不同场景；
- ``topology.TopologyStore``：SQLite 存拓扑图（节点表 + 边表，action_seq
  以 JSON 存储），懒加载、幂等写入；
- ``planner.NavPlanner``：指纹匹配当前节点 + BFS 最短路径 + 每步重定位
  （执行一步后重新截图匹配，偏离则重规划/重试）；
- ``explorer.NavExplorer``：空闲自主探索——向未访问方向移动并登记新节点，
  nav 探索激活时 idle_instinct 让位（由 AgentLoop 判断）。

``NavController`` 是本包对主循环的集成门面：懒装配（enabled=False 时不建
DB/不建组件，未配置不崩）、拦截 nav 动作（goto / explore）、提供空闲探索
协程（``maybe_explore_coro``）供 AgentLoop 与 idle_instinct 竞争优先级。

作者：寇豆码（Kou）· 工程师
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import numpy as np

from ..brain.schema import Action
from .explorer import ExploreResult, NavExplorer
from .landmark import LandmarkFingerprinter, NavError
from .planner import NavigateResult, NavPlanner
from .topology import TopologyStore

logger = logging.getLogger("vrchat_agent.nav")

# 空闲探索连续无进展上限：连续 N 次未登记新节点（画面无变化/自环/回环空转）
# → 本轮探索结束，防死循环（配合 explore_max_steps 双保险）。
_MAX_CONSECUTIVE_NO_PROGRESS = 3

__all__ = [
    "ExploreResult",
    "LandmarkFingerprinter",
    "NavController",
    "NavError",
    "NavExplorer",
    "NavPlanner",
    "NavigateResult",
    "TopologyStore",
]


class NavController:
    """空间导航集成门面：懒装配 + 动作拦截 + 空闲探索优先级。

    - 懒加载：构造只保存引用；首次实际使用（goto/explore）才创建
      TopologyStore / NavPlanner / NavExplorer；
    - 降级：config.nav.enabled=False 或未配置 → 所有接口安全空转
      （handle_action 消费 nav 动作但不执行，maybe_explore_coro 返回 None）；
    - 截图来源可注入：默认走 sampler.sample(sensor)（JPEG base64 → cv2 解码 →
      指纹）；测试注入 fingerprint_provider 即可全 mock。
    """

    def __init__(
        self,
        config=None,
        actuator=None,
        sampler=None,
        vision=None,
        store: TopologyStore | None = None,
        fingerprinter: LandmarkFingerprinter | None = None,
        fingerprint_provider: Callable[[], Awaitable[bytes | None]] | None = None,
        run_step: Callable[[dict], Awaitable[None]] | None = None,
    ) -> None:
        """初始化导航控制器。

        Args:
            config: Config 门面（读取 config.nav.*；None/未配置 → 禁用）。
            actuator: 动作执行器（需有 _call_tool_safe，用于默认 run_step）。
            sampler: ImageSampler（默认截图来源；可 None）。
            vision: VisionSensor（默认截图来源；可 None）。
            store: 预置 TopologyStore（测试注入；缺省懒创建）。
            fingerprinter: 预置指纹器（测试注入；缺省按配置创建）。
            fingerprint_provider: 异步截图→指纹（测试注入，优先于 sampler）。
            run_step: 异步单步动作执行器（测试注入，优先于 actuator）。
        """
        self._config = config
        self._actuator = actuator
        self._sampler = sampler
        self._vision = vision
        self._store = store
        self._fingerprinter = fingerprinter
        self._fingerprint_provider = fingerprint_provider
        self._run_step_override = run_step

        self._planner: NavPlanner | None = None
        self._explorer: NavExplorer | None = None
        self._ready = False
        self._closed = False

        # 空闲探索状态：_idle_exploring 防重入；_last_explore_ts 节流；
        # _idle_cancel 供新语音/急停打断进行中的探索
        self._idle_exploring = False
        self._last_explore_ts: float = 0.0
        self._idle_cancel = asyncio.Event()

    # ------------------------------------------------------------------
    # 配置访问（异常回退，未配置不崩）
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        """是否启用空间导航（config.nav.enabled）。"""
        try:
            return bool(self._config.nav.enabled)
        except Exception:  # noqa: BLE001 - 未配置/异常 → 视为未启用（降级不崩）
            return False

    def _cfg(self) -> Any:
        """当前 nav 配置节（异常时返回 None）。"""
        try:
            return self._config.nav
        except Exception:  # noqa: BLE001 - 未配置/异常 → 返回 None（降级不崩）
            return None

    # ------------------------------------------------------------------
    # 懒装配
    # ------------------------------------------------------------------
    def _ensure_ready(self) -> None:
        """首次使用时装配组件（幂等）。未启用 → 抛 NavError（调用方兜底）。"""
        if self._ready:
            return
        if self._closed:
            raise NavError("空间导航已关闭")
        if not self.enabled:
            raise NavError("空间导航未启用（config.nav.enabled=false）")
        cfg = self._cfg()
        if cfg is None:
            raise NavError("缺少 nav 配置节")

        if self._fingerprinter is None:
            self._fingerprinter = LandmarkFingerprinter(size=int(cfg.fingerprint_size))
        if self._store is None:
            db_path = Path(cfg.db_path)
            if not db_path.is_absolute():
                base = Path(self._config.path).parent if self._config is not None else Path(".")
                db_path = base / db_path
            self._store = TopologyStore(db_path)
        self._planner = NavPlanner(
            self._store,
            self._fingerprinter,
            similarity_threshold=float(cfg.similarity_threshold),
            max_relocalize_attempts=int(cfg.max_relocalize_attempts),
            max_total_steps=int(getattr(cfg, "max_total_steps", 50)),
            oscillation_window=6,
        )
        self._explorer = NavExplorer(
            self._store,
            self._planner,
            self._fingerprinter,
            max_self_loops=int(getattr(cfg, "max_self_loops", 3)),
        )
        self._ready = True
        logger.info("空间导航已就绪: db=%s threshold=%.3f", self._store.db_path,
                    float(cfg.similarity_threshold))

    # ------------------------------------------------------------------
    # 截图 / 动作执行（可注入；默认走 sampler/actuator）
    # ------------------------------------------------------------------
    async def _capture(self) -> bytes | None:
        """获取当前画面指纹；失败/无画面返回 None（降级不抛异常）。"""
        if self._fingerprint_provider is not None:
            try:
                return await self._fingerprint_provider()
            except Exception as exc:  # noqa: BLE001 - 注入提供者失败 → 降级为无画面
                logger.warning("注入截图提供者失败: %s", exc)
                return None
        if self._sampler is None or self._vision is None:
            return None
        try:
            b64 = await self._sampler.sample(self._vision)
            if not b64:
                return None
            import base64 as _b64

            import cv2

            buf = _b64.b64decode(b64)
            arr = np.frombuffer(buf, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return None
            return self._fingerprinter.fingerprint(img)
        except Exception as exc:  # noqa: BLE001 - 采样/解码失败 → 降级为无画面
            logger.warning("画面采样/指纹生成失败: %s", exc)
            return None

    async def _run_step(self, step: dict) -> None:
        """执行单步相对动作（{"tool", "args"}）。"""
        if self._run_step_override is not None:
            await self._run_step_override(step)
            return
        tool = step.get("tool")
        args = step.get("args", {}) or {}
        if self._actuator is None:
            return
        caller = getattr(self._actuator, "_call_tool_safe", None)
        if caller is not None:
            await caller(tool, args)
        else:
            logger.debug("执行器无 _call_tool_safe，跳过动作 %s", tool)

    # ------------------------------------------------------------------
    # 动作拦截（AgentLoop._execute_actions 调用）
    # ------------------------------------------------------------------
    async def handle_action(self, action: Action) -> bool:
        """处理一条 nav 动作；返回 True 表示已消费（不再下发 actuator）。

        - nav:goto（params.target 地标描述；缺省回最近已知节点）
        - nav:explore（立即执行一次探索）

        未启用时消费并 WARNING（避免 "未找到动作映射" 噪音与误执行）。
        """
        if action.kind != "nav":
            return False
        params = action.params or {}
        name = params.get("name")
        if not self.enabled:
            logger.warning("nav 动作被忽略：空间导航未启用（config.nav.enabled=false）")
            return True
        try:
            self._ensure_ready()
        except NavError as exc:
            logger.warning("nav 动作无法执行: %s", exc)
            return True
        if name == "goto":
            result = await self.goto(params.get("target"))
            logger.info("导航 goto 结果: %s（success=%s）", result.message, result.success)
            return True
        if name == "explore":
            result = await self.explore_once()
            logger.info("探索结果: %s（success=%s）", result.message, result.success)
            return True
        logger.warning("未知 nav 动作: %s", name)
        return True

    async def goto(self, target: str | None = None) -> NavigateResult:
        """去往目标地标（target 为节点描述；缺省回最近已知节点）。"""
        self._ensure_ready()
        fp = await self._capture()
        if fp is None:
            return NavigateResult(False, message="无法获取当前画面指纹")
        cur_id, _ = self._planner.match_node(fp)
        if cur_id is None:
            cur_id = self._store.add_node(fp, description="当前位置")

        if target:
            goal_id = self._planner.find_node_by_description(target)
            if goal_id is None:
                return NavigateResult(False, path=[cur_id], message=f"未找到地标节点：{target}")
        else:
            nodes = self._store.all_nodes()
            candidates = [n for n in nodes if n["id"] != cur_id]
            if not candidates:
                return NavigateResult(False, path=[cur_id], message="拓扑图中没有其它已知节点")
            goal_id = max(candidates, key=lambda n: float(n["created_at"]))["id"]

        if cur_id == goal_id:
            return NavigateResult(True, path=[cur_id], message="已在目标位置")
        # 传入 start_id=cur_id：避免 navigate_to 内部重复截图（goto 已定位过起点）
        return await self._planner.navigate_to(
            self._capture,
            self._run_step,
            goal_id,
            start_id=cur_id,
            max_retries=self._max_retries(),
        )

    async def explore_once(self) -> ExploreResult:
        """立即执行一次探索（供 nav:explore 动作与空闲探索共用）。"""
        self._ensure_ready()
        return await self._explorer.explore_once(self._capture, self._run_step)

    # ------------------------------------------------------------------
    # 空闲探索优先级（AgentLoop._maybe_idle 判断）
    # ------------------------------------------------------------------
    def maybe_explore_coro(self) -> Awaitable[None] | None:
        """判断是否应启动空闲探索；是则返回探索协程（否则 None）。

        AgentLoop 空闲时：本方法返回协程 → 优先启动探索（idle_instinct 让位）；
        返回 None → 回退启动 idle_instinct。

        条件：启用 + 未在探索（防重入）+ 距上次探索超过 explore_interval_s。
        """
        if not self.enabled or self._closed:
            return None
        if self._idle_exploring:
            return None
        # 首次空闲探索前完成懒装配（失败 → 放弃探索，回退 idle_instinct）
        try:
            self._ensure_ready()
        except NavError as exc:
            logger.warning("空闲探索不可用（%s），回退 idle_instinct", exc)
            return None
        try:
            interval = float(self._cfg().explore_interval_s)
        except Exception:  # noqa: BLE001 - 配置异常 → 回退默认间隔
            interval = 30.0
        if time.monotonic() - self._last_explore_ts < interval:
            return None
        self._idle_exploring = True
        self._idle_cancel.clear()
        return self.run_idle_explore()

    async def run_idle_explore(self) -> None:
        """空闲探索任务体（maybe_explore_coro 返回的协程）。

        P2-4 连续循环化：一步接一步探索登记新节点，每步检查——
        1. ``cancel_event`` 置位（新语音/急停打断）→ 立即 break；
        2. 达到 ``explore_max_steps``（默认 10）→ break；
        3. ``explore_once`` 返回 success=False（无可用方向/判定受阻）→ break；
        4. 连续 ``_MAX_CONSECUTIVE_NO_PROGRESS`` 次未登记新节点
           （画面无变化/自环/回环空转）→ break 防死循环。
        每步之间按 ``explore_step_gap_s``（默认 0.3s）短暂休息，让动作系统
        喘息、避免截图竞争（VRChat 是 3D 实时画面，连续移动时轻微视角变化
        不应被误判——由 explorer 的指纹匹配容忍度保证，循环不额外卡画面）。

        整轮循环视为一次探索会话：结束后（含被打断）才更新
        ``_last_explore_ts`` 与 ``_idle_exploring=False``，冷却只作用于会话之间。
        """
        try:
            if self._explorer is None:
                logger.warning("空闲探索跳过：组件未就绪")
                return
            cfg = self._cfg()
            try:
                max_steps = max(1, int(getattr(cfg, "explore_max_steps", 10)))
            except Exception:  # noqa: BLE001 - 配置异常 → 回退默认步数上限
                max_steps = 10
            try:
                step_gap = max(0.0, float(getattr(cfg, "explore_step_gap_s", 0.3)))
            except Exception:  # noqa: BLE001 - 配置异常 → 回退默认步间间隔
                step_gap = 0.3

            steps_done = 0
            consecutive_no_progress = 0
            while steps_done < max_steps:
                # 检查点 1：语音/急停打断（每步开始前）
                if self._idle_cancel.is_set():
                    logger.info("探索被语音/急停打断（第 %d 步前停止）", steps_done)
                    break
                result = await self._explorer.explore_once(
                    self._capture, self._run_step, cancel_event=self._idle_cancel
                )
                steps_done += 1
                logger.info("空闲探索 [%d/%d] 完成: %s（success=%s）",
                            steps_done, max_steps, result.message, result.success)
                # 检查点 2：探索途中被打断（explore_once 内部动作循环已让位）
                if self._idle_cancel.is_set():
                    logger.info("探索被语音/急停打断（第 %d 步后停止）", steps_done)
                    break
                # 检查点 3：无可用方向 / 判定受阻 → 已无路可走
                if not result.success:
                    logger.info("探索无可用方向或判定受阻（success=False），本轮探索结束")
                    break
                # 检查点 4：连续无进展（画面无变化/自环）→ 防死循环
                if result.new_node_id is not None:
                    consecutive_no_progress = 0
                else:
                    consecutive_no_progress += 1
                    if consecutive_no_progress >= _MAX_CONSECUTIVE_NO_PROGRESS:
                        logger.info("连续 %d 次无新节点（画面无变化/自环），本轮探索结束",
                                    _MAX_CONSECUTIVE_NO_PROGRESS)
                        break
                if step_gap > 0:
                    await asyncio.sleep(step_gap)
            else:
                # 检查点 5：自然跑满 explore_max_steps（while-else：未 break）
                logger.info("达到探索步数上限（%d 步），本轮探索结束", max_steps)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("空闲探索异常")
        finally:
            self._idle_exploring = False
            self._last_explore_ts = time.monotonic()

    def stop_idle(self) -> None:
        """打断进行中的空闲探索（新语音/急停时由 AgentLoop 调用）。"""
        if self._idle_exploring:
            self._idle_cancel.set()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def _max_retries(self) -> int:
        """读取单步重定位重试上限（异常回退 3）。"""
        try:
            return max(0, int(self._cfg().max_relocalize_attempts))
        except Exception:  # noqa: BLE001 - 配置异常 → 回退默认重试上限
            return 3

    def close(self) -> None:
        """关闭导航（释放 SQLite 连接；幂等）。"""
        self._closed = True
        if self._store is not None:
            try:
                self._store.close()
            except Exception:  # noqa: BLE001 - 关闭失败仅记日志，不阻断主循环收尾
                logger.debug("关闭拓扑图存储失败（忽略）")
