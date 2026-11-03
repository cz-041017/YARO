"""DemoLLM 的 LangChain 适配器（让演示模式能进 ``build_agent`` 图）。

真实模型路径走 :class:`~yaro.provider.deepseek.ChatDeepSeek`（有 ``bind_tools``/``ainvoke``），
演示模式之前用 :class:`~yaro.llm.demo.DemoLLM` 的 ``complete``（旧 ``loop.agent`` 接口，无
``bind_tools``/``ainvoke``）。本类把 DemoLLM 的确定性剧本包成 LangChain 兼容接口：

- ``bind_tools`` 是 no-op（剧本不依赖工具列表，按"已发生的工具轮次"推进）；
- ``ainvoke`` 把 LangChain 消息转回 OpenAI 形状喂给 ``DemoLLM.complete``，再把
  ``AssistantTurn`` 转成 ``AIMessage``（reasoning_content 进 additional_kwargs、
  用量进 ``response_metadata["deepseek_usage"]``，与真模型路径一致，``usage_from_messages``
  能正确拆分 hit/miss）。

已知限制：演示模式不压缩（剧本不会到 200K token，``egress=None`` fail-closed 即可），
且若被强制压缩，摘要请求会返回剧本下一轮而非真实摘要——属演示模式可接受范围。
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from ..llm.deepseek import AssistantTurn
from ..llm.demo import DemoLLM


def _messages_to_openai(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """LangChain 消息 → OpenAI 形状（喂给 DemoLLM.complete）。"""
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            item: dict[str, Any] = {"role": "assistant", "content": m.content}
            rc = (m.additional_kwargs or {}).get("reasoning_content")
            if rc:
                item["reasoning_content"] = rc
            if m.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": tc.get("name", ""),
                            "arguments": json.dumps(tc.get("args") or {}, ensure_ascii=False),
                        },
                    }
                    for tc in m.tool_calls
                ]
            out.append(item)
        elif isinstance(m, ToolMessage):
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
        else:  # 兜底
            out.append({"role": "user", "content": m.content})
    return out


def _turn_to_aimessage(turn: AssistantTurn) -> AIMessage:
    """AssistantTurn → AIMessage（工具调用 + 用量按真模型路径编码）。"""
    kwargs: dict[str, Any] = {}
    if turn.reasoning_content:
        kwargs["additional_kwargs"] = {"reasoning_content": turn.reasoning_content}
    if turn.tool_calls:
        kwargs["tool_calls"] = [
            {
                "name": tc["function"]["name"],
                "args": json.loads(tc["function"].get("arguments") or "{}"),
                "id": tc.get("id", ""),
                "type": "tool_call",
            }
            for tc in turn.tool_calls
        ]
    kwargs["response_metadata"] = {
        "deepseek_usage": {
            "prompt_cache_hit_tokens": turn.usage.hit,
            "prompt_cache_miss_tokens": turn.usage.miss,
            "prompt_tokens": turn.usage.input_total,
            "completion_tokens": turn.usage.out,
        }
    }
    return AIMessage(content=turn.content, **kwargs)


class DemoChatModel:
    """演示模式的 LangChain 兼容模型。``bind_tools`` 无副作用，``ainvoke`` 走剧本。"""

    model = "deepseek-flash"

    def __init__(self, *, delay_s: float = 0.0) -> None:
        self._demo = DemoLLM(delay_s=delay_s)

    def bind_tools(self, tools: Sequence[dict[str, Any]], **kwargs: Any) -> "DemoChatModel":
        # 剧本按"已发生的工具轮次"推进，不依赖工具列表；返回自身保持链式。
        return self

    async def ainvoke(self, messages: Sequence[BaseMessage], **kwargs: Any) -> AIMessage:
        turn = await self._demo.complete(_messages_to_openai(messages))
        return _turn_to_aimessage(turn)


__all__ = ["DemoChatModel", "_messages_to_openai", "_turn_to_aimessage"]
