"""ASR 误识别文本过滤（A-05）。

背景：本地 Whisper 有时把"没说过的话"识别出来（麦克风/内录拾取的环境音、
视频字幕、BGM 内容等），这些误识别文本不应进入 LLM 流程。

过滤规则（命中任一即跳过，不 publish utterance_ready）：
1. 空白/None；
2. 去空白后长度 < min_length（最短保留长度）；
3. 子串包含匹配 blocklist 任一词（长词/多字词用包含匹配，覆盖
   "字幕志愿者 李宗盛" 这类带前后缀的变体）；
4. 精确匹配 exact_words 任一词（仅单字/短回应词用精确匹配，
   避免 "你好" 被 "好" 误滤）。

热重载：与项目其它模块一致，构造时传入 config 引用，每次 should_skip
从 config.audio.input.filter_* 读取；读取异常回退缓存值（不中断 ASR 流程）。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger("vrchat_agent.audio.filter")


class UtteranceFilter:
    """ASR 转写文本过滤判定器（纯函数式，可独立单测）。

    用法（装配层）：
        filter = UtteranceFilter(config=config)          # 热重载（推荐）
        filter = UtteranceFilter(blocklist=[...], ...)   # 静态参数
    """

    def __init__(
        self,
        enabled: bool = True,
        blocklist: Sequence[str] = (),
        exact_words: Sequence[str] = (),
        min_length: int = 1,
        config: Any = None,
    ) -> None:
        self._enabled = bool(enabled)
        self._blocklist: list[str] = [w for w in blocklist if w]
        self._exact_words: list[str] = [w for w in exact_words if w]
        self._min_length = max(0, int(min_length))
        self._config = config  # 可选 Config 引用：提供时每次判定读最新值

    # ---- 内部：热重载读取 ----
    def _cfg(self) -> tuple[bool, list[str], list[str], int]:
        """从 config 读取当前过滤参数（异常回退缓存，绝不抛错）。"""
        if self._config is None:
            return self._enabled, self._blocklist, self._exact_words, self._min_length
        try:
            cfg = self._config.audio.input
            enabled = bool(getattr(cfg, "filter_enabled", self._enabled))
            blocklist = [w for w in (getattr(cfg, "filter_blocklist", None) or []) if w]
            exact_words = [w for w in (getattr(cfg, "filter_exact_words", None) or []) if w]
            min_length = max(0, int(getattr(cfg, "filter_min_length", self._min_length) or 0))
            return enabled, blocklist, exact_words, min_length
        except Exception:
            # 热重载期间配置访问/类型异常 → 回退缓存值，不阻断 ASR 流程
            logger.warning("UtteranceFilter 读取配置失败，使用缓存参数", exc_info=True)
            return self._enabled, self._blocklist, self._exact_words, self._min_length

    # ---- 判定 ----
    def should_skip(self, text: str) -> bool:
        """True=应跳过（误识别/噪音，不进入 LLM 流程）。

        注意：enabled=False（总开关关闭）时直接返回 False（完全不过滤，
        空文本由 pipeline 的 `if not text` 另行处理，保证旧行为不变）。
        """
        enabled, blocklist, exact_words, min_length = self._cfg()
        if not enabled:
            return False
        if text is None:
            return True
        stripped = str(text).strip()
        if not stripped:
            return True
        if len(stripped) < min_length:
            return True
        for word in blocklist:
            if word in stripped:
                return True
        for word in exact_words:
            if stripped == word:
                return True
        return False
