"""LangGraph agent 图的会话桥：SessionStore 事件 ↔ 图状态。

server.py 换引擎（旧 ``loop.agent.Agent`` → ``graph.agent.build_agent``）时，GUI 仍依赖
``SessionStore`` 的 user/assistant/tool/run_end 事件链做增量渲染 + rewind/fork/verify。
本模块做两件事：

1. 把 ``SessionStore.messages(branch)`` 的 OpenAI 形状消息还原成 LangChain 消息（图输入）；
2. 跑完图后，把图**新产生**的消息映射回 SessionStore 事件（assistant/tool/run_end）。

约定（调用方责任，与 ``build_agent`` docstring 的 H5 清单对齐）：

- **不脱敏**：调用方必须传 ``GuardedModel``（``yaro.provider.guard``）包 egress；
- **不挂 checkpointer**：一次 ``astream`` 增量跑到底（agent 图无 interrupt）；rewind/fork 仍由
  ``SessionStore.lineage/heads/rewind/fork`` 实现，每次运行前重新 reduce 历史；
- **不记 max_cost/max_wall 预算**：``build_agent`` 只有 ``max_iterations``；
- **视觉证据注入**：``artifacts`` 传 ArtifactStore 后，工具产出的诊断图会按 egress 策略
  注入下一轮 vision parts（与旧 loop.agent._request_messages 一致）；``artifacts=None`` 则跳过。

事件映射以旧 ``loop.agent.Agent`` 的 payload 形状为准（见 ``loop/agent.py:161-282``），
保证 GUI / 前端 / ``SessionStore.messages()`` reduce 不感知引擎已换。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from ..contracts import Usage
from ..egress import DERIVED, RAW
from ..provider.usage import usage_from_messages
from .agent import (
    DEFAULT_KEEP_TAIL_RATIO,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_VISION_IMAGES,
    build_agent,
)


def _openai_tool_calls_to_lc(tool_calls: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """OpenAI 形状 tool_calls → LangChain AIMessage.tool_calls 形状。"""
    out: list[dict[str, Any]] = []
    for tc in tool_calls or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments") or "{}"
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except (json.JSONDecodeError, ValueError):
                args = {}
        out.append(
            {
                "name": fn.get("name", ""),
                "args": args,
                "id": tc.get("id", ""),
                "type": "tool_call",
            }
        )
    return out


def _lc_tool_calls_to_openai(tool_calls: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """LangChain AIMessage.tool_calls → OpenAI 形状（会话持久化用）。"""
    out: list[dict[str, Any]] = []
    for tc in tool_calls or []:
        out.append(
            {
                "id": tc.get("id", ""),
                "type": "function",
                "function": {
                    "name": tc.get("name", ""),
                    "arguments": json.dumps(tc.get("args") or {}, ensure_ascii=False),
                },
            }
        )
    return out


def session_to_langchain(messages: Sequence[dict[str, Any]]) -> list[BaseMessage]:
    """``SessionStore.messages(branch)`` 的 OpenAI 形状 → LangChain 消息。

    assistant 的 ``reasoning_content`` 进 ``additional_kwargs``（不丢，压缩/再请求要用）；
    tool 转 ``ToolMessage(tool_call_id=...)``；user/nudge/note/compaction 摘要一律
    ``HumanMessage``。
    """
    out: list[BaseMessage] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "assistant":
            kwargs: dict[str, Any] = {}
            rc = m.get("reasoning_content")
            if rc:
                kwargs["additional_kwargs"] = {"reasoning_content": rc}
            tcs = _openai_tool_calls_to_lc(m.get("tool_calls"))
            if tcs:
                kwargs["tool_calls"] = tcs
            out.append(AIMessage(content=content, **kwargs))
        elif role == "tool":
            out.append(ToolMessage(content=content, tool_call_id=m.get("tool_call_id", "")))
        else:  # user / nudge / note / [CONTEXT COMPACTION] 摘要
            out.append(HumanMessage(content=content))
    return out


def _assistant_payload(msg: AIMessage) -> dict[str, Any]:
    """AIMessage → assistant 事件 payload（与旧 ``loop.agent`` 一致）。"""
    payload: dict[str, Any] = {
        "content": msg.content,
        "reasoning_content": (msg.additional_kwargs or {}).get("reasoning_content", ""),
    }
    if msg.tool_calls:
        payload["tool_calls"] = _lc_tool_calls_to_openai(msg.tool_calls)
    return payload


def _tool_payload(msg: ToolMessage, record: dict[str, Any] | None) -> dict[str, Any]:
    """ToolMessage + tools 节点 record → tool 事件 payload（与旧 ``loop.agent`` 一致）。"""
    record = record or {}
    evidence = record.get("evidence") or []
    artifacts = record.get("artifacts") or []
    sensitive = bool(record.get("sensitive"))
    return {
        "tool_call_id": msg.tool_call_id,
        "name": record.get("name") or msg.name or "",
        "content": msg.content,
        "summary": record.get("summary", ""),
        "ok": record.get("ok", False),
        "metrics": record.get("metrics") or {},
        "evidence": evidence,
        "evidence_kinds": [e.get("kind") for e in evidence],
        "artifacts": artifacts,
        "images": [
            {
                "uri": a.get("uri", ""),
                "sha256": a.get("sha256", ""),
                "media_type": a.get("media_type", ""),
                # raw = 用户原始输入图；derived = harness 渲染的诊断图（与旧 loop.agent 一致）。
                "sensitivity": RAW if sensitive else DERIVED,
            }
            for a in artifacts
            if (a.get("media_type") or "").startswith("image/")
        ],
        "degraded": record.get("degraded") or [],
        "hint": record.get("hint"),
    }


def _skipped_payload(msg: ToolMessage) -> dict[str, Any]:
    """finalize 节点补的 skipped ToolMessage → tool 事件（与旧 ``_skip_remaining`` 一致）。"""
    return {
        "tool_call_id": msg.tool_call_id,
        "name": msg.name or "",
        "content": msg.content,
        "summary": "未执行",
        "ok": False,
        "metrics": {},
        "evidence": [],
        "evidence_kinds": [],
        "artifacts": [],
        "images": [],
        "degraded": ["skipped"],
        "hint": "预算到达上限，这一批工具调用没有执行；如需继续请提高 max_iterations 或新开会话。",
    }


def _safe_cost(usage: Usage, model_name: str, peak: bool) -> float:
    """算成本；未知型号（ContractError）降级为 0，不炸 run_end（修 L1）。

    与 contracts.py "未知型号必须报错" 哲学的分歧是有意为之：成本只是展示，缺失不该吞掉
    已算好的 turn/usage；但仍记一条 warning，让运维能在日志里看到计价失败。
    """
    try:
        return float(usage.cost_usd(model_name, peak=peak))
    except Exception as exc:  # noqa: BLE001 - ContractError 等，成本缺失不该吞掉已算好的 turn/usage
        logger.warning("计价失败（model=%r, peak=%r），成本记 0.0：%s", model_name, peak, exc)
        return 0.0


async def run_agent(
    model,
    registry,
    store,
    task: str,
    *,
    system: str | None = None,
    branch: str = "main",
    run_id: str = "",
    max_iterations: int = 32,
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
    keep_tail_ratio: float = DEFAULT_KEEP_TAIL_RATIO,
    tool_selector: Callable[[Sequence[BaseMessage]], list[dict[str, Any]]] | None = None,
    egress: Any = None,
    skills: Any = None,
    stop_check: Callable[[], bool] | None = None,
    model_name: str = "deepseek-flash",
    peak: bool = False,
    artifacts: Any = None,
    max_vision_images: int = DEFAULT_MAX_VISION_IMAGES,
) -> dict[str, Any]:
    """跑一次 agent 图，把结果增量落成 SessionStore 事件，返回汇总 dict。

    事件顺序：先写 user（任务），之后每出现一条新消息立刻写 assistant/tool（``astream``
    增量），最后写 run_end。图中途抛异常时，已完成轮次的事件已落账，再写
    ``run_end(stop_reason="error")``（带已累计的 turn/usage），而不是向上抛。

    ``artifacts`` / ``max_vision_images``：把工具产出的诊断图注入下一轮 vision parts
    （与旧 loop.agent 一致）；``artifacts`` 为 None 时跳过注入。

    返回：``{run_id, stop_reason, turns, usage, cost_usd}``（usage/cost 均为 dict）。
    """
    prior = store.messages(branch)
    prior_lc = session_to_langchain(prior)
    task_msg = HumanMessage(content=task)
    initial = [*prior_lc, task_msg]

    # user 事件先写：即使随后被取消/报错，任务本身也在账本里。
    store.append("user", {"content": task, "run_id": run_id}, branch=branch)

    graph = build_agent(
        model,
        registry,
        system=system,
        max_iterations=max_iterations,
        max_tokens=max_tokens,
        keep_tail_ratio=keep_tail_ratio,
        tool_selector=tool_selector,
        egress=egress,
        skills=skills,
        stop_check=stop_check,
        artifacts=artifacts,
        max_vision_images=max_vision_images,
    )

    # 增量落账（修 H1）：用 astream(stream_mode="values") 按节点推进，每出现一条新消息
    # 立刻写进 SessionStore。图中途抛异常时，已完成轮次的事件已在账本里，不丢。
    # 用消息 id 去重（修 F1）：compact 节点用 RemoveMessage 删旧前缀，消息列表长度会
    # 变短，不能按 len(initial) 计数切片；新消息 id 全局唯一，按 id 判断最稳。
    # 注意：LangGraph 会在状态里给消息分配新 UUID，所以 written_ids 必须从**首个
    # yield 的状态**（输入状态）取，不能取 `initial` 对象自己的 id（全是 None）。
    written_ids: set[str] = set()
    new_messages: list[BaseMessage] = []
    final: dict[str, Any] | None = None
    tool_records: list[dict[str, Any]] = []
    tool_idx = 0
    first_yield = True

    try:
        async for state in graph.astream({"messages": initial}, stream_mode="values"):
            if first_yield:
                # 输入状态：把初始消息（LangGraph 已分配 UUID）记进去，后续按 id 跳过。
                written_ids = {m.id for m in (state.get("messages") or [])}
                first_yield = False
                continue
            final = state
            tool_records = list((state.get("context") or {}).get("tool_results") or [])
            for msg in state.get("messages") or []:
                if msg.id in written_ids:
                    continue
                written_ids.add(msg.id)
                new_messages.append(msg)
                if isinstance(msg, AIMessage):
                    store.append("assistant", _assistant_payload(msg), branch=branch)
                elif isinstance(msg, ToolMessage):
                    if tool_idx < len(tool_records):
                        store.append("tool", _tool_payload(msg, tool_records[tool_idx]), branch=branch)
                        tool_idx += 1
                    else:
                        # finalize 节点补的 skipped ToolMessage（未执行，无 record）。
                        store.append("tool", _skipped_payload(msg), branch=branch)
    except BaseException as exc:  # noqa: BLE001 - 引擎边界，记账而非炸掉线程
        # 已完成轮次的事件已增量写入；error run_end 带上已累计的 turn/usage（修 H1）。
        usage = usage_from_messages(new_messages)
        cost = _safe_cost(usage, model_name, peak)
        turns = sum(1 for m in new_messages if isinstance(m, AIMessage))
        store.append(
            "run_end",
            {
                "stop_reason": "error",
                "detail": f"{type(exc).__name__}: {exc}"[:4000],
                "turn": turns,
                "usage": usage.to_dict(),
                "cost_usd": round(cost, 6),
            },
            branch=branch,
        )
        return {
            "run_id": run_id,
            "stop_reason": "error",
            "turns": turns,
            "usage": usage.to_dict(),
            "cost_usd": round(cost, 6),
        }

    usage = usage_from_messages(new_messages)
    cost = _safe_cost(usage, model_name, peak)
    turns = int((final or {}).get("loop", 0))
    stop_reason = (final or {}).get("stop_reason") or "done"
    store.append(
        "run_end",
        {
            "stop_reason": stop_reason,
            "turn": turns,
            "usage": usage.to_dict(),
            "cost_usd": round(cost, 6),
        },
        branch=branch,
    )
    return {
        "run_id": run_id,
        "stop_reason": stop_reason,
        "turns": turns,
        "usage": usage.to_dict(),
        "cost_usd": round(cost, 6),
    }


__all__ = [
    "run_agent",
    "session_to_langchain",
    "_openai_tool_calls_to_lc",
    "_lc_tool_calls_to_openai",
]
