"""图状态定义。

``messages`` 用 ``add_messages`` reducer：同 id 的消息去重、新消息追加（本图每轮产生
新 id 的 AIMessage / ToolMessage，故效果是追加；不做 tool_call→ToolMessage 替换）。

``context`` 是节点之间传递的中间产物（scan / audit / proposals / train 结果、以及
tools 节点落下的结构化 ``tool_results``），不进消息历史，避免把大 JSON 塞进上下文。

``loop`` / ``stop_reason`` / ``summary`` 是标量控制字段（last-write-wins，无 reducer）。
``summary`` 是压缩摘要（compact 节点写、agent 节点读），不进 messages 历史。
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    context: dict[str, Any]
    loop: int
    stop_reason: str
    error: str
    summary: str
    # 待注入下一轮的视觉证据图（tools 节点写、agent 节点读后清空）。
    # 只存 ArtifactRef 引用 + sensitivity，字节永不进状态（与旧 loop.agent 的
    # ``_last_tool_images`` 语义一致：注入一次即清空，避免历史膨胀）。
    pending_images: list[dict[str, Any]]


__all__ = ["GraphState"]
