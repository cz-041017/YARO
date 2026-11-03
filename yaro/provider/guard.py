"""出网守卫：把 egress 策略包在 LangChain 模型外面。

``EgressPolicy.filter_messages`` 是给 OpenAI 格式 dict 消息用的；LangGraph 里流转的是
``BaseMessage``。本模块补一层适配：模型每次被调用前，先过端点闸、再对 BaseMessage 做
文本/路径/密钥脱敏，并兜底剥离不该出网的图片。

职责边界（务必记住）：
- **节点**负责"要不要给模型看图"（用 ``policy.allows_images(sensitivity)`` 决定附哪几张）。
- **守卫**负责"无论如何都不许漏出去"：``off``/``text`` 档下直接剥掉所有图片（防线冗余，
  节点漏判也不会泄露）；所有文本过 ``redact``（擦绝对路径 + 密钥）。
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from ..egress import EgressPolicy


class GuardedModel:
    """包一层 egress 的模型。用法与 ``BaseChatModel`` 相同（invoke / ainvoke / bind_tools）。"""

    def __init__(self, model: BaseChatModel, *, policy: EgressPolicy, base_url: str) -> None:
        self._model = model
        self._policy = policy
        self._base_url = base_url

    # ---- 属性透传，让调用方可以当普通模型用 -------------------------------- #

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "GuardedModel":
        return GuardedModel(
            self._model.bind_tools(tools, **kwargs),
            policy=self._policy,
            base_url=self._base_url,
        )

    # ---- 出网前处理 --------------------------------------------------------- #

    def _prepare(self, messages: Sequence[BaseMessage]) -> list[BaseMessage]:
        self._policy.check_endpoint(self._base_url)
        return [_redact_message(self._policy, msg) for msg in messages]

    def invoke(self, messages: Sequence[BaseMessage], **kwargs: Any) -> Any:
        return self._model.invoke(self._prepare(messages), **kwargs)

    async def ainvoke(self, messages: Sequence[BaseMessage], **kwargs: Any) -> Any:
        return await self._model.ainvoke(self._prepare(messages), **kwargs)


def _redact_message(policy: EgressPolicy, msg: BaseMessage) -> BaseMessage:
    """返回**新**消息（原消息不变，会话账本不能被就地改写）。"""
    updates: dict[str, Any] = {"content": _redact_content(policy, msg.content)}

    if getattr(msg, "additional_kwargs", None):
        # reasoning_content 就住在这里（DeepSeek thinking 模式），必须一起擦。
        updates["additional_kwargs"] = policy.redact_any(msg.additional_kwargs)

    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        updates["tool_calls"] = policy.redact_any(
            [_dump_tool_call(tc) for tc in tool_calls]
        )

    return msg.model_copy(update=updates)


def _dump_tool_call(tc: Any) -> Any:
    if hasattr(tc, "model_dump"):
        return tc.model_dump()
    return tc


def _redact_content(policy: EgressPolicy, content: Any) -> Any:
    if isinstance(content, str):
        return policy.redact(content)
    if isinstance(content, list):
        keep_images = policy.image_mode != "none"
        out: list[Any] = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type")
                if btype in ("image_url", "image", "input_image"):
                    if keep_images:
                        out.append(block)  # 是否发图已由节点按敏感度决定，这里只做"全禁"兜底
                    continue
                if "text" in block:
                    out.append({**block, "text": policy.redact(block["text"])})
                else:
                    out.append(policy.redact_any(block))
            else:
                out.append(block)
        return out
    return content


__all__ = ["GuardedModel"]
