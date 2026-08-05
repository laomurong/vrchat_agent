"""Skill 式按需加载 Prompt：技能块注册表（T01）。

设计（架构师高见远 T01）：
- ``SkillPack``：一个技能块（动作枚举片段 / 技能说明 + 例句 + 触发关键词），
  frozen dataclass，YAML 与 Python 注册入口统一收敛到该模型；
- ``SkillRegistry``：注册 / 关键词匹配（中英）/ 依赖展开 / sticky 会话保持 /
  按需渲染与全量渲染；
- sticky 语义：技能命中后未来 ``max_rounds`` 轮继续保持激活（每轮递减），
  适合"我正在抓东西 / 我正在导航"这类跨轮上下文；
- 裁剪：非常驻激活技能按 ``max_active`` 上限（依赖不计入）；渲染按
  ``max_chars`` 上限（块级裁剪，不截断中间）。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from pathlib import Path

logger = logging.getLogger("vrchat_agent.brain.skills.registry")


@dataclass(frozen=True)
class SkillPack:
    """一个可注册的技能块。

    :param name: 技能唯一名（如 ``osc_input`` / ``vrc.social``）
    :param domain: 领域（osc / vrc / nav / core）
    :param triggers: 触发关键词（中英），玩家 utterance 包含任一即激活该技能
    :param description: LLM 可见内容（动作枚举片段 / 技能说明 + 例句）；
        支持 ``{placeholder}``（渲染时由调用方替换，默认保留字面量）
    :param deps: 依赖技能名（激活时一并激活，**不计入** max_active）
    :param priority: 渲染排序（小者先）；同 priority 按 name 稳定排序
    :param resident: 常驻（无论是否命中都渲染；如核心动作集）
    :param max_rounds: sticky 保持轮数（>0 时命中后未来 N 轮继续激活）
    """

    name: str
    domain: str = "core"
    triggers: tuple[str, ...] = ()
    description: str = ""
    deps: tuple[str, ...] = ()
    priority: int = 10
    resident: bool = False
    max_rounds: int = 0


#: SkillPack 可用的字段名（YAML 加载时白名单过滤，未知字段忽略）
_PACK_FIELDS: frozenset[str] = frozenset(f.name for f in fields(SkillPack))


class SkillRegistry:
    """技能注册表：注册 / 匹配 / sticky 保持 / 渲染。"""

    def __init__(
        self,
        max_active: int = 3,
        max_chars: int = 1200,
        sticky_turns: int = 3,
    ) -> None:
        self._packs: dict[str, SkillPack] = {}
        #: name → 剩余 sticky 轮数（>0 表示未来轮次继续保持激活）
        self._session_sticky: dict[str, int] = {}
        self._max_active: int = max(1, int(max_active))
        self._max_chars: int = max(0, int(max_chars))
        #: 全局 sticky 轮数（config.brain.skill.sticky_turns；未显式配置 max_rounds 的技能
        #: 用它作默认 sticky 轮数；0 = 全局关闭 sticky）
        self._sticky_turns: int = max(0, int(sticky_turns))

    # ------------------------------------------------------------------
    # 注册 / 查询
    # ------------------------------------------------------------------
    def register(self, pack: SkillPack) -> SkillRegistry:
        """注册一个技能块（同名覆盖）。"""
        if not pack.name or not str(pack.name).strip():
            raise ValueError("SkillPack.name 不能为空")
        self._packs[pack.name] = pack
        return self

    def register_many(self, packs: Iterable[SkillPack]) -> SkillRegistry:
        """批量注册。"""
        for pack in packs:
            self.register(pack)
        return self

    def load_yaml(self, path: str | Path) -> SkillRegistry:
        """从 YAML 加载技能块（YAML 为主，Python 注册入口兜底）。

        支持两种形态：
        1) ``skills: [{name, domain, triggers, description, ...}, ...]``；
        2) ``skills: {name: {domain, triggers, ...}, ...}``。
        """
        import yaml

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"技能 YAML 不存在: {p}")
        with open(p, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        skills = data.get("skills", data) if isinstance(data, dict) else {}
        if isinstance(skills, list):
            for item in skills:
                if not isinstance(item, dict):
                    continue
                self.register(SkillPack(**self._filter_fields(item)))
        elif isinstance(skills, dict):
            for name, item in skills.items():
                if not isinstance(item, dict):
                    continue
                merged = dict(item)
                merged.setdefault("name", str(name))
                self.register(SkillPack(**self._filter_fields(merged)))
        else:
            # TRY004: ValueError 语义正确（YAML 结构错误，非类型错误）
            raise ValueError(f"技能 YAML 格式错误（应为 skills 列表/映射）: {p}")  # noqa: TRY004
        logger.info("技能 YAML 加载完成: %s（%d 个技能块）", p, len(self._packs))
        return self

    @staticmethod
    def _filter_fields(item: dict) -> dict:
        """白名单过滤 YAML 字段（忽略未知字段，避免 pydantic/dataclass 报错）。"""
        out: dict = {}
        for key, value in item.items():
            if key not in _PACK_FIELDS:
                continue
            if key in ("triggers", "deps"):
                if isinstance(value, str):
                    value = tuple(v.strip() for v in value.split(",") if v.strip())
                elif isinstance(value, list):
                    value = tuple(str(v) for v in value)
            out[key] = value
        return out

    def get(self, name: str) -> SkillPack | None:
        """按名取技能块（不存在返回 None）。"""
        return self._packs.get(name)

    @property
    def packs(self) -> dict[str, SkillPack]:
        """全部技能块快照（name → SkillPack）。"""
        return dict(self._packs)

    @property
    def names(self) -> tuple[str, ...]:
        """技能名列表（稳定排序）。"""
        return tuple(sorted(self._packs))

    def __len__(self) -> int:
        return len(self._packs)

    # ------------------------------------------------------------------
    # 匹配 / 依赖展开
    # ------------------------------------------------------------------
    def match(self, text: str) -> list[str]:
        """关键词包含匹配（中英）：返回命中的技能名（含 deps 展开，去重，priority 排序）。

        :param text: 玩家 utterance（原样匹配；大小写不敏感）
        """
        return self._expand_deps(self._raw_match(text))

    def _raw_match(self, text: str) -> list[str]:
        """关键词包含匹配（不展开依赖；供 build_active 计算 max_active 裁剪）。"""
        low = (text or "").lower()
        return [
            name
            for name, pack in self._packs.items()
            if any(trig and str(trig).lower() in low for trig in pack.triggers)
        ]

    def _expand_deps(self, names: Iterable[str]) -> list[str]:
        """依赖展开 + 去重 + priority 稳定排序。"""
        seen: set[str] = set()
        out: list[str] = []
        queue = [n for n in names if n]
        while queue:
            name = queue.pop(0)
            if name in seen or name not in self._packs:
                continue
            seen.add(name)
            pack = self._packs[name]
            for dep in pack.deps:
                if dep not in seen:
                    queue.append(dep)
            out.append(name)
        out.sort(key=lambda n: (self._packs[n].priority, n))
        return out

    # ------------------------------------------------------------------
    # sticky 会话保持
    # ------------------------------------------------------------------
    def tick(self) -> None:
        """每轮调用：sticky 剩余轮数递减，归零移除。"""
        for name in list(self._session_sticky):
            remaining = self._session_sticky[name] - 1
            if remaining <= 0:
                self._session_sticky.pop(name, None)
            else:
                self._session_sticky[name] = remaining

    def _sticky_rounds(self, pack: SkillPack) -> int:
        """技能生效的 sticky 轮数。

        语义（T01 死配置修复）：``pack.max_rounds`` 优先（>0 时作为 per-skill
        覆盖值）；未显式配置（0）的技能使用**全局 ``sticky_turns``**（来自
        config.brain.skill.sticky_turns，默认 3）。返回 0 表示不 sticky。
        """
        if pack.max_rounds > 0:
            return pack.max_rounds
        return self._sticky_turns

    def _apply_sticky(self, names: Sequence[str]) -> None:
        """命中/forced 技能刷新 sticky 轮数（供未来轮次保持）。

        存储值为 ``有效轮数 + 1``：build_active 每轮先 tick() 递减 1，
        触发轮之后仍保持 ``有效轮数`` 轮激活（与"sticky 会话保持 N 轮"语义一致）。
        """
        for name in names:
            pack = self._packs.get(name)
            if pack is None:
                continue
            rounds = self._sticky_rounds(pack)
            if rounds > 0:
                self._session_sticky[name] = max(
                    self._session_sticky.get(name, 0), rounds + 1
                )

    # ------------------------------------------------------------------
    # 激活
    # ------------------------------------------------------------------
    def build_active(self, text: str, forced: Sequence[str] | None = None) -> list[str]:
        """当前 utterance 命中 + sticky 保持 + forced 并集；裁剪 max_active。

        :param text: 玩家 utterance
        :param forced: 强制激活技能名（如 nav 探索/导航时由 AgentLoop 传入）
        :returns: 激活技能名（含依赖与常驻；priority 排序）

        裁剪规则：**依赖不计入 max_active**——先对顶层（命中/forced/sticky，
        未展开依赖）裁剪，再展开依赖并连同常驻一起返回。
        """
        # 1) 每轮先递减上轮遗留 sticky
        self.tick()
        # 2) 匹配（不展开依赖，顶层参与裁剪）+ forced 并集
        raw = self._raw_match(text)
        forced_names = [n for n in (forced or []) if n in self._packs]
        merged = list(dict.fromkeys(raw + forced_names))
        # 3) 命中技能刷新 sticky
        self._apply_sticky(merged)
        # 4) 顶层 = 本次命中 ∪ 遗留 sticky（均未展开依赖）
        sticky = [n for n, r in self._session_sticky.items() if r > 0]
        top = list(dict.fromkeys(merged + sticky))
        resident = [n for n in self._packs if self._packs[n].resident]
        # 5) 顶层非常驻裁剪（依赖不计入 max_active）
        top_non_resident = [n for n in top if n not in resident]
        if len(top_non_resident) > self._max_active:
            top_non_resident = top_non_resident[: self._max_active]
        # 6) 依赖展开 + 常驻
        return self._expand_deps(top_non_resident + resident)

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------
    def render(self, packs: Sequence[str], max_chars: int | None = None) -> str:
        """按 priority 渲染指定技能块（块级裁剪，不截断中间）。

        :param packs: 技能名序列
        :param max_chars: 输出上限（None 用构造期 max_chars；<=0 不限）
        :returns: 渲染文本（块间空行分隔）；空技能返回空串
        """
        limit = self._max_chars if max_chars is None else max(0, int(max_chars))
        blocks: list[str] = []
        total = 0
        for name in self._expand_deps(packs):
            pack = self._packs.get(name)
            if pack is None or not pack.description:
                continue
            block = self._render_pack(pack)
            if limit > 0 and total + len(block) > limit:
                logger.debug("技能渲染达 max_chars 上限，跳过后续块: %s", name)
                continue
            blocks.append(block)
            total += len(block)
        if not blocks:
            return ""
        return "\n\n".join(blocks)

    def full_render(self, max_chars: int | None = None) -> str:
        """全量渲染：全部已注册技能块（mode=full 兼容；build_decision_instruction 追加）。

        :param max_chars: 输出上限（默认 0 = 不限；full 模式应展示全部技能）。
        """
        if max_chars is None:
            max_chars = 0
        return self.render(list(self._packs), max_chars=max_chars)

    def catalog(self, max_chars: int = 400) -> str:
        """紧凑技能目录（name + 摘要；供 on_demand 指令告知 LLM 扩展能力）。

        摘要取 description 首行（截断 60 字符）；常驻技能与空描述技能不收录。
        """
        parts: list[str] = []
        for name in sorted(self._packs):
            pack = self._packs[name]
            if pack.resident or not pack.description:
                continue
            summary = ""
            for line in pack.description.splitlines():
                line = line.strip().rstrip("：:。")
                if line:
                    summary = line[:60]
                    break
            parts.append(f"{name}（{summary}）")
        text = " / ".join(parts)
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars].rstrip() + "…"
        return text

    @staticmethod
    def _render_pack(pack: SkillPack) -> str:
        """单个技能块的渲染格式。

        标题用「技能块：」而非「技能：」——BugFix：LLM 会把注入的
        「技能：vrc.social」误当成 actions.kind 的合法值（vrc.social 不是
        ActionKind），导致 schema 拒绝整条决策。标题明确为"技能块"降低误用。
        """
        return f"【技能块：{pack.name}】\n{pack.description}"
