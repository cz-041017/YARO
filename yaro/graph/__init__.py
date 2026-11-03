"""LangGraph 编排层。

YARO 用 LangGraph 做"状态中枢"：状态图 + checkpoint（会话树回滚）+ interrupt（HITL）。
模型绑定见 :mod:`yaro.provider`，出网脱敏在 :class:`yaro.provider.guard.GuardedModel`，
工具派发在 :mod:`yaro.capabilities.registry`。

- :func:`build_agent` 是"薄 agent"工具调用循环。
- :func:`build_distill_graph` 是蒸馏 + 主动学习闭环子图。
- :func:`run_distillation`（:mod:`yaro.graph.runtime`）把它与审计账本/审批/checkpointer 串起来。
- :mod:`yaro.graph.context` 提供上下文经济原语（token 估算 / 安全压缩 / 工具延迟加载）。
"""

from .agent import build_agent
from .context import (
    compaction_plan,
    estimate_tokens,
    find_safe_cut,
    make_tool_selector,
    remove_prefix,
    select_schemas,
    summarize_prefix,
    summary_message,
)
from .distill import DistillState, build_distill_graph
from .runtime import run_distillation
from .state import GraphState

__all__ = [
    "DistillState",
    "GraphState",
    "build_agent",
    "build_distill_graph",
    "compaction_plan",
    "estimate_tokens",
    "find_safe_cut",
    "make_tool_selector",
    "remove_prefix",
    "run_distillation",
    "select_schemas",
    "summarize_prefix",
    "summary_message",
]
