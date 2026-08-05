"""B 大脑：Brain 决策编排（B-01~B-05）。

decide(utterance, vision) → Decision：
1. 组装 messages（system 人设 + 决策指令 + 长期记忆 + 上下文 + 当前用户消息）
2. 调 LLMClient（文本 / 文本+视觉）
3. Decision.parse_json 校验（失败降级为不回应）
4. 若回应，写入对话上下文
5. 决策完成后低优先级异步提取长期记忆（MemoryStore，extract_interval 节流）
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

from .context import ConversationContext
from .llm_client import LLMClient, LLMError
from .memory import MemoryStore
from .prompts import build_decision_instruction, build_memory_extract_prompt, build_system_prompt
from .schema import Decision, _extract_json_object, _strip_trailing_commas, decision_schema_markdown
from .skills.core import (
    build_decision_instruction_minimal,
    build_system_prompt_resident,
)
from .skills.packs import default_registry
from .skills.registry import SkillRegistry

logger = logging.getLogger("vrchat_agent.brain")


class Brain:
    """决策编排：把"听到了什么 + 看到了什么 + 上下文"变成结构化 Decision。

    热重载（A1）：传入 config 引用后，decide() 每次从 ``config.brain.*`` 读取
    temperature / persona / vision_enabled / max_tokens / timeout，改配置无需重启；
    未传 config 时回退装配期缓存值（保持旧测试兼容）。
    """

    def __init__(
        self,
        llm: LLMClient,
        context: ConversationContext,
        persona_name: str = "Vivi",
        persona_desc: str = "",
        vision_enabled: bool = True,
        config=None,
        memory: MemoryStore | None = None,
        skills: SkillRegistry | None = None,
    ) -> None:
        self._llm = llm
        self._context = context
        self._persona_name = persona_name
        self._persona_desc = persona_desc
        self._vision_enabled = vision_enabled
        self._config = config
        self._memory = memory
        self._skills = skills
        self._last_raw: str = ""
        self._extract_counter: int = 0
        self._extract_task: asyncio.Task | None = None

    # ---- Skill 式按需加载 Prompt（T03/T04）----
    @property
    def skills(self) -> SkillRegistry:
        """技能注册表（默认内置注册表；每次构造独立实例，不共享 sticky 状态）。"""
        if self._skills is None:
            self._skills = default_registry()
        return self._skills

    def _skill_mode(self) -> str:
        """读取 skill.mode（on_demand/full），异常回退 on_demand。"""
        if self._config is not None:
            try:
                return str(self._config.brain.skill.mode or "on_demand")
            except Exception:
                pass
        return "on_demand"

    def _skill_resident(self) -> bool:
        """读取 skill.resident（是否注入常驻最小动作集），异常回退 True。"""
        if self._config is not None:
            try:
                return bool(self._config.brain.skill.resident)
            except Exception:
                pass
        return True

    def _build_active_skills(self, utterance: str, forced: list[str] | None) -> list[str]:
        """激活技能：forced 为空自动 build_active(utterance)；非空并集。"""
        try:
            return self.skills.build_active(utterance, forced=forced)
        except Exception as exc:
            logger.warning("技能激活失败（降级为空）: %s", exc)
            return []

    # ---- 热重载读取（异常回退装配期缓存值） ----
    def _persona(self) -> tuple[str, str]:
        if self._config is not None:
            try:
                return (
                    str(self._config.brain.persona.name or self._persona_name),
                    str(self._config.brain.persona.system_prompt or self._persona_desc),
                )
            except Exception:
                pass
        return self._persona_name, self._persona_desc

    def _is_vision_enabled(self) -> bool:
        """读取 vision_enabled（热重载），异常回退装配期缓存值。"""
        if self._config is not None:
            try:
                return bool(self._config.brain.llm.vision_enabled)
            except Exception:
                pass
        return self._vision_enabled

    def _temperature(self) -> float:
        if self._config is not None:
            try:
                return float(self._config.brain.llm.temperature)
            except Exception:
                pass
        try:
            return float(getattr(self._llm, "temperature", 0.7) or 0.7)
        except Exception:
            return 0.7

    # ---- 长期记忆（标准版）：配置读取 / 注入 / 异步提取 ----
    def _memory_config(self):
        """读取 memory 配置（热重载），异常回退 None（用默认值）。"""
        if self._config is not None:
            try:
                return self._config.brain.memory
            except Exception:
                pass
        return None

    def _memory_enabled(self) -> bool:
        """记忆总开关：未注入 MemoryStore 或 config.brain.memory.enabled=false → 关闭。"""
        if self._memory is None:
            return False
        cfg = self._memory_config()
        if cfg is not None:
            try:
                return bool(cfg.enabled)
            except Exception:
                pass
        return self._memory.enabled

    def _build_memory_block(self, utterance: str) -> str:
        """构造长期记忆注入块（[长期记忆] 节）；禁用/无存储/无记忆返回空串。

        检索顺序（任务②语义层）：embedding 余弦 top-k（MemoryStore.search 内部
        已做语义→关键词 LIKE 兜底）→ 仍为空则回退常用记忆 recall；
        任何失败静默降级，不阻塞主循环。
        """
        if not self._memory_enabled():
            return ""
        cfg = self._memory_config()
        try:
            limit = int(cfg.inject_limit) if cfg is not None else 5
        except Exception:
            limit = 5
        limit = max(1, min(limit, 20))
        memories: list[dict[str, Any]] = []
        query = (utterance or "").strip()
        try:
            if query:
                # 语义优先 + 关键词兜底（语义层未注入/模型缺失时自动回退关键词）
                memories = self._memory.search(query, top_k=limit)
        except Exception:
            memories = []
        if not memories:
            try:
                # 最终兜底：常用记忆（use_count 高 + 最近使用优先）
                memories = self._memory.recall(limit=limit)
            except Exception:
                memories = []
        lines = []
        ids: list[int] = []
        for m in memories:
            content = str(m.get("content", "")).strip()
            if content:
                lines.append(f"- {content}")
                mid = m.get("id")
                if mid is not None:
                    ids.append(int(mid))
        if not lines:
            return ""
        # 标记为已使用（提升 recall 优先级；失败静默）
        try:
            self._memory.mark_used(ids)
        except Exception:
            pass
        return (
            "【长期记忆】（玩家之前告诉过你的、值得记住的事实/喜好，回复时自然运用，"
            "不要生硬复述）\n" + "\n".join(lines)
        )

    async def decide(
        self,
        utterance: str,
        vision: str | None = None,
        forced_skills: list[str] | None = None,
    ) -> Decision:
        """根据玩家语音（与可选视觉画面）生成决策。

        :param utterance: 玩家语音转写文本
        :param vision: 可选视觉画面 base64
        :param forced_skills: 强制激活技能名（如 nav 探索/导航由 AgentLoop 传入；
            非空时与关键词命中并集）
        """
        persona_name, persona_desc = self._persona()
        mode = self._skill_mode()
        if mode == "full":
            # 回退旧全量行为（常驻 + 全量动作枚举 + 全量规则 + 全量技能块）
            system = build_system_prompt(persona_name, persona_desc)
            instruction = build_decision_instruction()
        else:
            # on_demand：人设 → 最小动作集 → 技能目录 → 【已激活技能】→ 记忆 → 历史 → 玩家说
            active = self._build_active_skills(utterance, forced_skills)
            instruction = build_decision_instruction_minimal(
                registry=self.skills,
                active_packs=active,
                schema_markdown=decision_schema_markdown(),
                resident=self._skill_resident(),
                catalog=self.skills.catalog(),
            )
            system = build_system_prompt_resident(persona_name, persona_desc)
        messages = [{"role": "system", "content": f"{system}\n\n{instruction}"}]
        # 长期记忆注入（标准版 + 任务②语义层）：按当前 utterance 语义检索相关记忆，
        # 以 [长期记忆] 节注入 system，跨会话生效；语义不可用自动回退关键词/常用记忆
        memory_block = self._build_memory_block(utterance)
        if memory_block:
            messages[0]["content"] += f"\n\n{memory_block}"
        messages.extend(self._context.to_messages())

        user_content = f"玩家说：{utterance}"
        if vision:
            # 视觉仅作参考（P1 用户反馈）：不被问及时不要主动描述画面。
            user_content += "\n（附：当前画面，仅作参考；未被问及时不要主动描述）"
        messages.append({"role": "user", "content": user_content})

        t0 = time.time()
        temperature = self._temperature()
        try:
            if vision and self._is_vision_enabled():
                raw = await self._llm.chat_with_image(messages, vision, temperature=temperature)
            else:
                raw = await self._llm.chat(messages, temperature=temperature)
            self._last_raw = raw
            decision = Decision.parse_json(raw)
        except LLMError:
            # 传输/超时错误上抛，由主循环统一降级（播报"网络不佳"提示音，F-04）
            raise
        except Exception as exc:
            logger.error("决策流程异常: %s", exc, extra={"event": "llm_failed"})
            decision = Decision(respond=False, reply_text=None, actions=[])

        logger.info(
            "决策完成: respond=%s emotion=%s actions=%d 耗时=%.2fs",
            decision.respond,
            decision.emotion,
            len(decision.actions),
            time.time() - t0,
            extra={"event": "decision_ready"},
        )
        if decision.respond and decision.reply_text:
            self._context.add("user", utterance)
            self._context.add("assistant", decision.reply_text)
        # 决策完成后低优先级异步提取长期记忆（节流 extract_interval 轮；失败静默）
        self._maybe_extract()
        return decision

    def _maybe_extract(self) -> None:
        """按节流规则触发异步记忆提取；禁用/未到轮次/无对话内容则跳过。

        提取用同一 LLMClient 额外消耗 token（config.memory.auto_extract 可关；
        extract_interval 每 N 轮一次）。所有异常在 _extract_memories 内消化，
        不阻塞主循环。
        """
        cfg = self._memory_config()
        auto_extract = True
        interval = 3
        if cfg is not None:
            try:
                auto_extract = bool(cfg.auto_extract)
            except Exception:
                auto_extract = True
            try:
                interval = max(1, int(cfg.extract_interval))
            except Exception:
                interval = 3
        if not self._memory_enabled() or not auto_extract:
            return
        self._extract_counter += 1
        if self._extract_counter % interval != 0:
            return
        history = self._context.to_messages()
        if not history:
            return  # 无实质对话内容，省一次 LLM 调用
        try:
            task = asyncio.create_task(self._extract_memories(history))
            self._extract_task = task
        except Exception as exc:
            logger.debug("记忆提取任务创建失败（忽略）: %s", exc)

    async def _extract_memories(self, history: list[dict]) -> None:
        """从最近对话提取值得长期记住的事实并写入 MemoryStore（失败静默降级）。"""
        try:
            system = build_memory_extract_prompt()
            recent = history[-6:] if len(history) > 6 else history  # 最近 3 轮
            messages = [{"role": "system", "content": system}, *recent]
            raw = await self._llm.chat(messages, temperature=0.3)
            items = self._parse_memories(raw)
            if not items:
                return  # 提取为空不写
            cfg = self._memory_config()
            try:
                max_items = int(cfg.extract_max_items) if cfg is not None else 5
            except Exception:
                max_items = 5
            max_items = max(1, min(max_items, 20))
            for content in items[:max_items]:
                self._memory.add(content, kind="fact")
            logger.info("记忆提取完成: 写入 %d 条", min(len(items), max_items), extra={"event": "memory_extracted"})
        except Exception as exc:
            logger.warning(
                "记忆提取失败（静默降级，不阻塞主循环）: %s", exc,
                extra={"event": "memory_extract_failed"},
            )

    @staticmethod
    def _parse_memories(raw: str) -> list[str]:
        """宽松解析提取结果：{"memories": [...]} / {"facts": [...]} / 裸数组。

        兼容 Markdown 围栏、前后夹带文字、JSON5 尾逗号；解析失败返回空列表。
        """
        if not raw or not raw.strip():
            return []
        data = None
        # 1) 对象形态：{"memories": [...]}（复用 schema 的宽松 JSON 提取）
        try:
            obj = _extract_json_object(raw)
            data = json.loads(obj)
        except Exception:
            data = None
        if data is None:
            # 2) 裸数组形态：["a", "b"]（剥围栏后取 [ ... ]）
            text = re.sub(r"^```[A-Za-z]*\s*", "", (raw or "").strip())
            text = re.sub(r"\s*```$", "", text)
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end > start:
                try:
                    data = json.loads(_strip_trailing_commas(text[start : end + 1]))
                except Exception:
                    data = None
        if isinstance(data, dict):
            for key in ("memories", "facts", "items"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
            else:
                return []
        if not isinstance(data, list):
            return []
        out: list[str] = []
        for item in data:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                s = str(item.get("content") or item.get("text") or "").strip()
                if s:
                    out.append(s)
        return out

    async def wait_extract(self, timeout: float = 5.0) -> None:
        """等待最近一次异步记忆提取完成（测试/关闭前汇合用；异常静默）。"""
        task = self._extract_task
        if task is None:
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except Exception:
            pass

    def reset_context(self) -> None:
        """清空对话上下文。"""
        self._context.clear()

    @property
    def context(self) -> ConversationContext:
        return self._context

    @property
    def last_raw(self) -> str:
        return self._last_raw
