"""LangGraph 编排层：最小 agent 图（agent 节点 + tools 节点 + finalize 节点）。

用 LangGraph 的 StateGraph / messages / 条件边做编排（B 方案，不手写 while 循环），
但工具派发仍走 ``ToolRegistry.call`` —— 因为工具返回结构化 ``ToolResult``
（evidence / artifacts / metrics / degraded / sensitive），需要落进 ``state["context"]``
供 GUI 与审计账本用；stock ``ToolNode`` 只能把返回值字符串化成 ToolMessage。

**调用方责任清单（H5，接入 server.py 前必须逐条自证）**：

1. 传入 ``GuardedModel``（``yaro.provider.guard``）包 egress —— 否则工具原文 / 系统提示 /
   证据会不脱敏直接出网；本模块**不做**任何脱敏。
2. 自己挂 checkpointer（``langgraph.checkpoint.sqlite.SqliteSaver`` 等）做会话树回滚 /
   interrupt；本模块编译时不带 checkpointer。
3. 自己记账：模型 usage 在 AIMessage.response_metadata / usage_metadata 里，本模块不提取、
   ``max_cost`` 预算不在此生效。
4. 自己写审计账本（events.jsonl digest 链）与 GUI 事件；本模块只把结构化结果放进
   ``state["context"]["tool_results"]``，不落盘。

**返回的图只支持 async**：两个节点都是 ``async def``，调用方必须用
``graph.ainvoke(...)`` / ``graph.astream(...)``；同步 ``graph.invoke()`` 会抛
``TypeError: No synchronous function provided``（已实测，见 tests）。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph

from ..capabilities.evidence import render_vision_parts
from ..contracts import ArtifactRef, Evidence
from ..egress import DERIVED, RAW
from .context import (
    compaction_plan,
    estimate_tokens,
    find_safe_cut,
    remove_prefix,
    summarize_prefix,
    summary_message,
)
from .state import GraphState

DEFAULT_MAX_ITERATIONS = 32
DEFAULT_MAX_TOKENS = 200_000
DEFAULT_KEEP_TAIL_RATIO = 0.35
DEFAULT_MAX_IDENTICAL_CALLS = 2
DEFAULT_MAX_VISION_IMAGES = 2


def _tool_call_signature(tc: dict[str, Any]) -> tuple[str, str]:
    """单个 tool_call 的稳定签名（name + 规范化 args），用于检测重复调用。"""
    return (tc.get("name", ""), json.dumps(tc.get("args") or {}, sort_keys=True, ensure_ascii=False))


def _is_stuck(messages: list[BaseMessage], max_identical_calls: int | None) -> bool:
    """模型在**当前 user 轮**内连续 ``max_identical_calls`` 轮发出完全相同的一组 tool_calls 即视为卡死。

    只反向扫描最近一条 ``HumanMessage``（user 轮边界）之后带 tool_calls 的 AIMessage，
    遇 user 轮边界即停——否则上次会话 stuck 会污染后续全新任务（会话被永久 brick）。
    与旧 loop.agent 的 per-Agent ``_call_counts``（每次 run 重置）语义一致：护栏不跨 user 轮泄漏。
    """
    if not max_identical_calls or max_identical_calls <= 0:
        return False
    batches: list[tuple[tuple[str, str], ...]] = []
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            # user 轮边界：只比较本轮连续调用，不扫上一轮 / 上次会话的历史。
            break
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            batches.append(tuple(sorted(_tool_call_signature(tc) for tc in m.tool_calls)))
            if len(batches) == max_identical_calls:
                break
    if len(batches) < max_identical_calls:
        return False
    return all(b == batches[0] for b in batches)


def _evidence_to_dict(evidence: Evidence) -> dict[str, Any]:
    """把证据压成可 JSON 序列化的形式，供 GUI / 审计读取。

    图片只保留 ArtifactRef 形状（sha256 + media_type），字节永不进事件；表格最多保留
    20 行、dict 最多 40 键 —— 事件是账本，不是数据仓库。与旧 loop.agent 的
    ``evidence_to_dict`` 等价。
    """
    payload = evidence.payload
    if isinstance(payload, ArtifactRef):
        form: Any = payload.to_dict()
    elif isinstance(payload, (str, int, float, bool)) or payload is None:
        form = payload
    elif isinstance(payload, list):
        form = [row if isinstance(row, (str, int, float)) else dict(row) for row in payload[:20]]
    elif isinstance(payload, dict):
        form = {key: payload[key] for key in list(payload)[:40]}
    else:
        form = str(payload)[:1200]
    return {"kind": evidence.kind, "caption": evidence.caption, "payload": form}


def _images_from_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从 tools 节点的结构化结果里抽出图像引用，映射成旧 loop.agent 的 ``images`` 形状。

    ``sensitivity`` 与旧行为一致：``result.sensitive`` 为 True 的产物图标 ``raw``
    （用户原始输入图），否则 ``derived``（harness 渲染的诊断图）。egress 据此决定
    哪些图能出网，原图在 derived 档会被剥离。
    """
    images: list[dict[str, Any]] = []
    for rec in records:
        sensitivity = RAW if rec.get("sensitive") else DERIVED
        for a in rec.get("artifacts") or []:
            if (a.get("media_type") or "").startswith("image/"):
                images.append(
                    {
                        "uri": a.get("uri", ""),
                        "sha256": a.get("sha256", ""),
                        "media_type": a.get("media_type", ""),
                        "size": a.get("size", 0),
                        "sensitivity": sensitivity,
                    }
                )
    return images


def _vision_message(
    pending: list[dict[str, Any]],
    *,
    egress: Any,
    artifacts: Any,
    max_vision_images: int,
) -> HumanMessage | None:
    """把待注入的视觉证据渲染成一条 user 消息（含图片 parts），无图/禁用则返回 None。

    与旧 ``loop.agent._request_messages`` 的注入逻辑一致：
    1. 按 ``egress.allows_images(sensitivity)`` 过滤（原图在 derived 档被剥离）；
    2. 剥离后用一条文本 part 明确告诉模型"图被剥了"，别让它猜图像内容；
    3. ``render_vision_parts`` 运行时现读现编码 base64，**绝不把字节写进状态/事件**。
    """
    if not pending or max_vision_images <= 0 or artifacts is None:
        return None
    if egress is None:
        # fail-closed：没有出网策略就不注入任何图（与压缩节点 context.summarize_prefix
        # 的 egress 必填守卫一致）。调用方要注入视觉证据必须显式传 egress，否则原图
        # 可能未经允许就发给模型。
        return None
    allowed = [img for img in pending if egress.allows_images(img.get("sensitivity", RAW))]
    dropped = len(pending) - len(allowed)
    parts = render_vision_parts(allowed[:max_vision_images], artifacts, max_images=max_vision_images)
    if dropped:
        parts = list(parts) + [
            {
                "type": "text",
                "text": (
                    f"[隐私策略] egress={getattr(egress, 'mode', '?')}：已剥离 {dropped} 张原始证据图，未出网。"
                    "不要猜测图像内容，改用数值证据，或请人类调整出网策略。"
                ),
            }
        ]
    if not parts:
        return None
    return HumanMessage(content=parts)


def _make_route(max_iterations: int, max_tokens: int | None, keep_tail_ratio: float, egress: Any) -> Callable[[GraphState], str]:
    """条件边：stop_reason → end；超预算 → finalize；上下文超阈值 → compact；否则看 tool_calls。"""

    def route(state: GraphState) -> str:
        if state.get("stop_reason"):
            return "end"
        if int(state.get("loop", 0)) >= max_iterations:
            return "finalize"
        messages = state.get("messages") or []
        last = messages[-1] if messages else None
        # 只有模型还要继续（带 tool_calls）时才考虑压缩；模型已结束（无 tools）直接 end，
        # 避免"已结束但超阈值"被误判成反复 compact → agent 死循环。
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            # 上下文经济：超阈值且存在安全切点 → 先压缩再执行工具（compact → tools）。
            # egress 缺失时不压缩（fail-closed：绝不把未脱敏历史发给摘要模型）。
            if (
                egress is not None
                and max_tokens is not None
                and estimate_tokens(messages) > max_tokens
                and find_safe_cut(messages, keep_tail_ratio=keep_tail_ratio) is not None
            ):
                return "compact"
            return "tools"
        return "end"

    return route


def _make_agent_node(model, registry, *, system: str | None, max_iterations: int, tool_selector, stop_check=None, max_identical_calls: int | None = DEFAULT_MAX_IDENTICAL_CALLS, artifacts=None, max_vision_images: int = DEFAULT_MAX_VISION_IMAGES, egress: Any = None):
    """agent 节点：协作式取消（GUI 停止按钮）→ 卡死护栏 → 预算检查前置 → 选工具 → 绑定 → 调模型。

    ``tool_selector`` 每轮按当前消息重选 schema（工具延迟加载，对应 context.select_schemas）；
    为 None 时退化为一次性绑全量 ``registry.schemas()``。
    ``stop_check`` 可选 ``() -> bool``：在**每次模型调用前**检查，返回 True 则本轮停止
    （协作式取消，不打断进行中的调用，与旧 loop.agent 的 stop_check 语义一致）。
    ``max_identical_calls``：模型在**当前 user 轮**内连续这么多轮发出同一组 tool_calls 则停
    （stuck 护栏，只比较本轮、不跨 user 轮泄漏）；传 None/<=0 禁用。
    ``artifacts`` / ``max_vision_images`` / ``egress``：把上一轮工具产出的诊断图注入本轮
    vision parts（与旧 loop.agent._request_messages 一致）；artifacts 为 None 时跳过注入。
    """

    async def agent(state: GraphState) -> dict[str, Any]:
        if stop_check is not None and stop_check():
            return {"stop_reason": "stopped"}
        if _is_stuck(list(state.get("messages") or []), max_identical_calls):
            return {"stop_reason": "stuck"}
        if int(state.get("loop", 0)) >= max_iterations:
            return {"stop_reason": "max_iterations"}
        messages: list[BaseMessage] = list(state.get("messages") or [])
        # system 恒为第一段、内容不变 → 与 DeepSeek 前缀缓存锚点一致，不持久化进 state。
        if system and (not messages or not isinstance(messages[0], SystemMessage)):
            messages = [SystemMessage(content=system), *messages]
        # 压缩摘要（如果有）以 HumanMessage 落地（历史前缀，非模型新输出）——
        # 摘要源自模型对用户历史的提取，必须放 user 角色（不可信历史），不能放
        # system（高权限，会被摘要里的对抗内容放大注入）；位置在 system 之后、
        # 其余消息之前，无 system 时放首位，绝不插到会话中间。
        summary = state.get("summary") or ""
        if summary:
            sm = summary_message(summary)
            if messages and isinstance(messages[0], SystemMessage):
                messages = [messages[0], sm, *messages[1:]]
            else:
                messages = [sm, *messages]
        # 视觉证据注入（旧 loop.agent._request_messages 的 vision parts）：把上一轮工具
        # 产出的诊断图以 user 消息拼进本轮请求，模型能看图而非只看文本摘要。只注入
        # 本地请求（不进 state["messages"]，避免历史膨胀 + 破坏 tool_call 配对）。
        pending = state.get("pending_images") or []
        vision = _vision_message(pending, egress=egress, artifacts=artifacts, max_vision_images=max_vision_images)
        if vision is not None:
            messages.append(vision)
        schemas = tool_selector(messages) if tool_selector else registry.schemas()
        bound = model.bind_tools(schemas)
        response = await bound.ainvoke(messages)
        return {
            "messages": [response],
            "loop": int(state.get("loop", 0)) + 1,
            "pending_images": [],  # 注入一次即清空
        }

    return agent


def _make_tools_node(registry):
    """tools 节点：逐个执行 tool_calls，把结构化结果收进 context["tool_results"]。

    顺序执行（不用 asyncio.gather）——DeepSeek 要求 tool 结果与 tool_calls 顺序一一对应。
    """

    async def tools(state: GraphState) -> dict[str, Any]:
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        out_msgs: list[ToolMessage] = []
        records: list[dict[str, Any]] = []
        for tc in tool_calls:
            name = tc["name"]
            args = tc.get("args") or {}
            result = await registry.call(name, args)
            out_msgs.append(
                ToolMessage(content=result.to_text(), tool_call_id=tc["id"], name=name)
            )
            records.append(
                {
                    "name": name,
                    "arguments": args,
                    "ok": result.ok,
                    "summary": result.summary,
                    "sensitive": result.sensitive,
                    "degraded": result.degraded,
                    "hint": result.hint,
                    "metrics": result.metrics,
                    "evidence": [_evidence_to_dict(e) for e in result.evidence],
                    "artifacts": [a.to_dict() for a in result.artifacts],
                }
            )
        ctx = dict(state.get("context") or {})
        ctx["tool_results"] = list(ctx.get("tool_results") or []) + records
        # 本轮工具产出的诊断图（ArtifactRef 引用 + sensitivity），留给 agent 节点注入
        # 下一轮 vision parts（只存引用，字节不进状态，与旧 loop.agent 一致）。
        return {
            "messages": out_msgs,
            "context": ctx,
            "pending_images": _images_from_records(records),
        }

    return tools


def _make_finalize_node():
    """finalize 节点：超预算时给未执行的 tool_calls 补 skipped ToolMessage。

    否则最后一条 AIMessage 带 tool_calls 却无对应 ToolMessage → 续跑 DeepSeek 400
    （每个 tool_call 都必须有对应 tool 结果）。
    """

    async def finalize(state: GraphState) -> dict[str, Any]:
        messages = state.get("messages") or []
        last = messages[-1] if messages else None
        skipped: list[ToolMessage] = []
        if isinstance(last, AIMessage):
            for tc in getattr(last, "tool_calls", None) or []:
                skipped.append(
                    ToolMessage(
                        content="[预算到达上限，本工具未执行]",
                        tool_call_id=tc["id"],
                        name=tc["name"],
                    )
                )
        return {"messages": skipped, "stop_reason": "max_iterations"}

    return finalize


def _make_compact_node(model, *, egress, max_tokens: int | None, keep_tail_ratio: float):
    """compact 节点：上下文超阈值时，把较早的 prefix 压成摘要、删掉旧消息。

    摘要放 ``state["summary"]``（agent 节点每次调用时把它拼在 system 之后），
    prefix 用 ``RemoveMessage`` 从消息历史里移除——tail（近期消息）原样保留。
    """

    async def compact(state: GraphState) -> dict[str, Any]:
        if egress is None:
            return {}  # fail-closed：没有 egress 就不压缩（路由已保证不会走到这）
        messages = state.get("messages") or []
        plan = compaction_plan(messages, max_tokens=max_tokens, keep_tail_ratio=keep_tail_ratio)
        if plan is None:
            return {}
        prefix, _tail = plan
        prior = state.get("summary") or ""
        new_summary = await asyncio.to_thread(
            summarize_prefix, model, prefix, egress=egress, prior_summary=prior
        )
        return {"summary": new_summary, "messages": remove_prefix(prefix)}

    return compact


def build_agent(
    model,
    registry,
    *,
    system: str | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tool_selector: Callable[[Sequence[BaseMessage]], list[dict[str, Any]]] | None = None,
    egress: Any = None,
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
    keep_tail_ratio: float = DEFAULT_KEEP_TAIL_RATIO,
    skills: Any = None,
    stop_check: Callable[[], bool] | None = None,
    max_identical_calls: int | None = DEFAULT_MAX_IDENTICAL_CALLS,
    artifacts: Any = None,
    max_vision_images: int = DEFAULT_MAX_VISION_IMAGES,
):
    """组装最小 agent 图。

    ``model`` 需支持 ``bind_tools`` 与 ``ainvoke``（ChatDeepSeek / GuardedModel 均满足）；
    ``registry`` 提供 ``schemas()`` 与 ``async call(name, args)``。
    ``tool_selector`` 可选：``(messages) -> schemas``，每轮按消息选工具再 bind（延迟加载）；
    不传则一次性绑全量 ``registry.schemas()``。
    ``egress`` 用于压缩摘要前脱敏（隐私，压缩前必填）；不传 ``egress`` 则不压缩（fail-closed）。
    ``max_tokens`` / ``keep_tail_ratio`` 控制上下文压缩触发阈值与保留比例。
    ``skills`` 可选：``yaro.skills.SkillRegistry``，其 ``render_index()`` 目录拼进 system 提示
    （渐进披露：正文按需读 ``.agents/skills/``），本模块不做正文加载。
    ``stop_check`` 可选：每轮模型调用前查一次，True 则停（协作式取消，GUI 停止按钮用）。
    ``max_identical_calls`` 可选：**当前 user 轮**内连续这么多轮发出同一组 tool_calls 则停
    （stuck 护栏，防烧轮）；传 ``None`` / ``<=0`` 禁用护栏。只比较本轮，不跨 user 轮泄漏。
    ``artifacts`` / ``max_vision_images``：把工具产出的诊断图注入下一轮 vision parts
    （与旧 loop.agent 一致）；``artifacts`` 为 None 时跳过注入。

    返回的图只支持 ``ainvoke`` / ``astream``（节点为 async，同步 ``invoke`` 会 TypeError）。
    """
    system_text = system or ""
    if skills is not None and len(skills) > 0:
        index = skills.render_index()
        if index:
            system_text = f"{system_text}\n\n可用技能（正文按需读 .agents/skills/）：\n{index}".strip()
    graph = StateGraph(GraphState)
    graph.add_node(
        "agent",
        _make_agent_node(
            model,
            registry,
            system=system_text or None,
            max_iterations=max_iterations,
            tool_selector=tool_selector,
            stop_check=stop_check,
            max_identical_calls=max_identical_calls,
            artifacts=artifacts,
            max_vision_images=max_vision_images,
            egress=egress,
        ),
    )
    graph.add_node("tools", _make_tools_node(registry))
    graph.add_node("finalize", _make_finalize_node())
    graph.add_node(
        "compact",
        _make_compact_node(model, egress=egress, max_tokens=max_tokens, keep_tail_ratio=keep_tail_ratio),
    )
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        _make_route(max_iterations, max_tokens, keep_tail_ratio, egress),
        {"tools": "tools", "finalize": "finalize", "compact": "compact", "end": END},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("compact", "tools")
    graph.add_edge("finalize", END)
    return graph.compile()
