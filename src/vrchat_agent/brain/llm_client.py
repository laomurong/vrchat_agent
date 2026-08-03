"""B 大脑：LLMClient（OpenAI 兼容协议抽象，B-01/B-04）。

- 只依赖 base_url + api_key + model（共享知识约定第 4 条）；
- base_url 可显式覆盖 provider 默认值：云端 / 本地（LM Studio / Ollama / vLLM
  的 /v1 端点均可；base_url 直接传给 AsyncOpenAI，不走 provider 内置默认）；
- 无真实 api_key 场景：api_key 为空串/None 时用占位 "EMPTY"，本地服务
  （LM Studio / Ollama）不校验 key，因此占位值即可正常工作，不会报错；
- 视觉以 data:image/jpeg;base64 走同一 chat/completions 接口；
- 统一 asyncio.wait_for 超时，失败抛 LLMError（上层降级）。

推理模型适配（MiMo 实机实测，BugFix）：
- mimo-v2.5 是推理模型：思考过程在 message.reasoning_content，
  message.content 可能为空字符串（实测 max_tokens=100 时 content=''、
  completion_tokens 全被 reasoning 吃掉）→ Brain 决策解析失败"识别到但没回复"；
- 实测 extra_body={"thinking": {"type": "disabled"}} 可关闭思考
  （reasoning_len=0、content 直接非空、更快 1.65s vs 3.60s），
  因此对 model 以 "mimo" 开头的模型默认传该参数（首选，让 content 稳定非空）；
  其它 provider 不传，避免未知参数报错；
- 兜底：content 为空/纯空白时回退 reasoning_content（思考内容至少提供响应）；
  content 与 reasoning_content 均为空 → 明确抛 LLMError，不静默返回空串。

推理模型新参数适配（max_completion_tokens，实测结论）：
- MiMo（及 OpenAI 新版推理模型）建议/要求用 max_completion_tokens；
  旧模型 / LM Studio / Ollama / vLLM 用 max_tokens；传错参数名可能 400 或截断语义差异；
- 因此对 model 以 "mimo" 开头的请求改用 max_completion_tokens（替代 max_tokens），
  其它 provider 保持 max_tokens 不变——保证 LM Studio 兼容；
- 判定统一走 _is_mimo()（与 _extra_body 复用同一判定，避免分叉）。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger("vrchat_agent.brain.llm")


class LLMError(RuntimeError):
    """LLM 调用错误（超时/网络/协议/空内容）。"""


class LLMClient:
    """OpenAI 兼容协议 LLM 客户端（文本 + 视觉）。

    热重载（A1）：传入 config 引用后，max_tokens / timeout 在每次调用时从
    ``config.brain.llm.*`` 读取（装配期缓存作为回退），改配置无需重启。
    base_url / api_key / model 属"需重启"字段（A2），保持装配期固化。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout: float = 45.0,
        max_tokens: int = 512,
        config=None,
    ) -> None:
        self._base_url = base_url
        self._api_key = api_key or "EMPTY"  # 本地 Ollama/vLLM 可不校验 key
        self._model = model
        self._timeout = timeout
        self._max_tokens = max_tokens
        self._config = config
        self._client = None  # openai.AsyncOpenAI（惰性创建）

    def _ensure_client(self):
        """惰性创建 AsyncOpenAI 客户端（便于离线测试 mock）。"""
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(base_url=self._base_url, api_key=self._api_key)
        return self._client

    def _cfg_timeout(self) -> float:
        """运行时读取超时（热重载），异常回退装配期缓存值。"""
        if self._config is not None:
            try:
                return float(self._config.brain.llm.timeout)
            except Exception:
                pass
        return self._timeout

    def _cfg_max_tokens(self) -> int:
        """运行时读取 max_tokens（热重载），异常回退装配期缓存值。"""
        if self._config is not None:
            try:
                return int(self._config.brain.llm.max_tokens)
            except Exception:
                pass
        return self._max_tokens

    def _is_mimo(self) -> bool:
        """模型是否 MiMo 推理模型（model 以 "mimo" 开头，大小写不敏感）。

        统一判定点：_extra_body（thinking.disabled）与 chat() 的参数名分派
        （max_completion_tokens vs max_tokens）共用，避免两处分叉漂移。
        """
        return bool(self._model and str(self._model).lower().startswith("mimo"))

    def _extra_body(self) -> dict[str, Any] | None:
        """推理模型关闭思考参数（仅对 MiMo 模型传，避免其它 provider 报未知参数）。

        实测（真实 key，mimo-v2.5）：
        - baseline：reasoning_content 有内容（105 token），content 可能为空；
        - enable_thinking=False：reasoning 仍在（52 token），未真正关闭；
        - thinking={"type": "disabled"}：reasoning_len=0，content 直接非空且更快。
        因此对 model 以 "mimo" 开头的模型默认传 thinking.disabled。
        """
        if self._is_mimo():
            return {"thinking": {"type": "disabled"}}
        return None

    @staticmethod
    def _extract_content(message: Any) -> str:
        """提取回复文本：content 为空/纯空白时回退 reasoning_content（推理模型兜底）。

        两者皆空抛 LLMError（不静默返回空串，避免上层把空串当"无回复"解析失败）。
        """
        content = getattr(message, "content", None) or ""
        if not content.strip():
            content = getattr(message, "reasoning_content", None) or ""
        if not content.strip():
            raise LLMError("模型返回空内容（content 与 reasoning_content 均为空）")
        return content.strip()

    async def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        """纯文本对话：返回助手回复文本（推理模型 content 为空时回退 reasoning_content）。

        max_tokens 参数名分派：MiMo（推理模型新协议）用 max_completion_tokens；
        其它 provider（qwen/ollama/LM Studio 等）保持 max_tokens，保证兼容。
        """
        client = self._ensure_client()
        timeout = self._cfg_timeout()
        max_tokens = self._cfg_max_tokens()
        token_param = "max_completion_tokens" if self._is_mimo() else "max_tokens"
        create_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            token_param: max_tokens,
        }
        extra_body = self._extra_body()
        if extra_body is not None:
            create_kwargs["extra_body"] = extra_body
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(**create_kwargs),
                timeout=timeout,
            )
            return self._extract_content(resp.choices[0].message)
        except asyncio.TimeoutError:
            raise LLMError(f"LLM 调用超时（>{timeout}s）") from None
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"LLM 调用失败: {exc}") from exc

    async def chat_with_image(
        self, messages: list[dict], image_b64: str, temperature: float = 0.7
    ) -> str:
        """多模态对话：把最后一条 user 消息替换为 文本 + 图片 组合内容。"""
        if not messages:
            messages = [{"role": "user", "content": "请看这张图并简要描述。"}]
        base = list(messages)
        last = base[-1]
        user_content: Any = [
            {"type": "text", "text": last.get("content", "") if last.get("role") == "user" else "请看这张图并简要描述。"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]
        if last.get("role") == "user":
            base[-1] = {"role": "user", "content": user_content}
        else:
            base.append({"role": "user", "content": user_content})
        return await self.chat(base, temperature)
