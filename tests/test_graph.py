"""graph 编排层（LangGraph）的离线单测。

用脚本化假模型驱动一轮工具调用，证明 agent ↔ tools 能转起来，不发任何网络请求。
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from yaro.capabilities.registry import ToolRegistry
from yaro.contracts import ToolResult, ToolSpec
from yaro.graph import build_agent
from yaro.graph.agent import _is_stuck


class _EchoTool:
    spec = ToolSpec(
        name="echo",
        description="echo back the text",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )

    async def run(self, *, text: str) -> ToolResult:
        return ToolResult(ok=True, summary=f"echo:{text}", metrics={"len": float(len(text))})


class _SecondTool:
    spec = ToolSpec(
        name="second",
        description="the second tool",
        parameters={"type": "object", "properties": {}, "required": []},
    )

    async def run(self) -> ToolResult:
        return ToolResult(ok=True, summary="second")


class _FakeEgress:
    def redact(self, text: str) -> str:
        return text


class _ScriptedModel:
    """按顺序返回 AIMessage；最后一次无 tool_calls。"""

    def __init__(self, responses: list[AIMessage]):
        self._responses = list(responses)
        self.calls = 0
        self._bound: object | None = None

    def bind_tools(self, tools):
        self._bound = tools
        return self

    def invoke(self, messages):
        """压缩摘要用的同步入口（compact 节点调用）。"""
        self.summary_calls = getattr(self, "summary_calls", 0) + 1
        return "这是压缩后的历史摘要"

    async def ainvoke(self, messages, **kwargs):
        self.last_messages = list(messages)
        i = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[i]


def _tool_call_echo(text: str, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "echo", "args": {"text": text}, "id": call_id}],
    )


def test_is_stuck_same_turn_vs_user_boundary() -> None:
    """_is_stuck 只比较当前 user 轮：反向扫描遇 HumanMessage（轮边界）即停，跨轮不泄漏。"""
    tc = lambda text, cid: _tool_call_echo(text, cid)  # noqa: E731
    tool = lambda cid: ToolMessage(content="ok", tool_call_id=cid)  # noqa: E731

    # 同一轮连续两次相同调用 → stuck
    same = [HumanMessage(content="hi"), tc("x", "c1"), tool("c1"), tc("x", "c2")]
    assert _is_stuck(same, 2) is True

    # 同一轮但 args 不同 → 不 stuck
    diff = [HumanMessage(content="hi"), tc("x", "c1"), tool("c1"), tc("y", "c2")]
    assert _is_stuck(diff, 2) is False

    # 跨 user 轮：上一轮两次相同调用不应泄漏到新一轮（F 回归，护栏遇 HumanMessage 即 break）
    cross = [
        HumanMessage(content="task1"),
        tc("x", "c1"), tool("c1"), tc("x", "c2"), tool("c2"),
        HumanMessage(content="task2"),
    ]
    assert _is_stuck(cross, 2) is False

    # 禁用护栏（None / 0）
    assert _is_stuck(same, None) is False
    assert _is_stuck(same, 0) is False


def test_agent_runs_one_tool_roundtrip() -> None:
    model = _ScriptedModel(
        [_tool_call_echo("hi", "c1"), AIMessage(content="done")]
    )
    registry = ToolRegistry([_EchoTool()])
    graph = build_agent(model, registry, system="S", max_iterations=8)

    result = asyncio.run(graph.ainvoke({"messages": [HumanMessage(content="say hi")]}))

    messages = result["messages"]
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage) and messages[1].tool_calls
    assert isinstance(messages[2], ToolMessage)
    assert messages[2].content.startswith("OK: echo:hi")
    assert isinstance(messages[3], AIMessage) and messages[3].content == "done"

    ctx = result["context"]
    assert result["loop"] == 2
    rec = ctx["tool_results"][0]
    assert rec["name"] == "echo"
    assert rec["ok"] is True
    assert rec["metrics"] == {"len": 2.0}


def test_agent_stops_when_no_tools() -> None:
    model = _ScriptedModel([AIMessage(content="no tools needed")])
    registry = ToolRegistry([_EchoTool()])
    graph = build_agent(model, registry, max_iterations=8)

    result = asyncio.run(graph.ainvoke({"messages": [HumanMessage(content="hi")]}))

    assert len(result["messages"]) == 2  # human + assistant
    assert result["loop"] == 1
    assert not result.get("context", {}).get("tool_results")


def test_agent_hits_max_iterations() -> None:
    # 每轮不同 args，避免触发 stuck 护栏（stuck 只针对连续相同调用）。
    forever = [_tool_call_echo(f"x{i}", f"c{i}") for i in range(100)]
    model = _ScriptedModel(forever)
    registry = ToolRegistry([_EchoTool()])
    graph = build_agent(model, registry, max_iterations=3)

    result = asyncio.run(graph.ainvoke({"messages": [HumanMessage(content="hi")]}))

    assert result["loop"] == 3
    assert result["stop_reason"] == "max_iterations"
    # 3 次模型调用 + 3 个 ToolMessage（前 2 个真实执行、第 3 个是 skipped）：
    # 每个 AIMessage 的 tool_call 都有对应 ToolMessage，不悬空。
    messages = result["messages"]
    assert len(messages) == 1 + 3 + 3  # human + 3 assistant + 3 tool
    assert isinstance(messages[-1], ToolMessage)
    assert "预算到达上限" in messages[-1].content
    assert len(result["context"]["tool_results"]) == 2


def test_no_dangling_tool_calls_at_budget() -> None:
    """H1 回归：达到上限后不能留下带 tool_calls 却无 ToolMessage 的悬空 assistant。"""
    forever = [_tool_call_echo("x", f"c{i}") for i in range(100)]
    model = _ScriptedModel(forever)
    registry = ToolRegistry([_EchoTool()])
    graph = build_agent(model, registry, max_iterations=2)

    result = asyncio.run(graph.ainvoke({"messages": [HumanMessage(content="hi")]}))

    messages = result["messages"]
    ids_with_result = {
        m.tool_call_id for m in messages if isinstance(m, ToolMessage)
    }
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls or []:
                assert tc["id"] in ids_with_result, "悬空 tool_call，续跑会 400"


def test_max_iterations_zero_does_not_call_model() -> None:
    """L2 回归：max_iterations=0 时不应调用模型。"""
    model = _ScriptedModel([AIMessage(content="should not be returned")])
    registry = ToolRegistry([_EchoTool()])
    graph = build_agent(model, registry, max_iterations=0)

    result = asyncio.run(graph.ainvoke({"messages": [HumanMessage(content="hi")]}))

    assert result["stop_reason"] == "max_iterations"
    assert model.calls == 0
    assert len(result["messages"]) == 1  # 只有 human


def test_agent_uses_tool_selector_deferred_loading() -> None:
    """工具延迟加载：传 tool_selector 后，模型每轮只 bind 选中的子集，不绑全量。"""
    model = _ScriptedModel([_tool_call_echo("hi", "c1"), AIMessage(content="done")])
    registry = ToolRegistry([_EchoTool(), _SecondTool()])
    all_schemas = registry.schemas()
    echo_schema = [s for s in all_schemas if s["function"]["name"] == "echo"]

    def selector(messages):
        return echo_schema

    graph = build_agent(
        model, registry, system="S", max_iterations=8, tool_selector=selector
    )
    asyncio.run(graph.ainvoke({"messages": [HumanMessage(content="say hi")]}))

    bound_names = [s["function"]["name"] for s in model._bound]
    assert bound_names == ["echo"]


def test_agent_compacts_history_when_over_budget() -> None:
    """上下文超阈值时：compact 节点把前缀压成摘要、删旧消息，循环仍能结束。"""
    long_task = "长" * 300  # ~100 tokens
    responses = [_tool_call_echo(f"x{i}", f"c{i}") for i in range(6)] + [AIMessage(content="done")]
    model = _ScriptedModel(responses)
    registry = ToolRegistry([_EchoTool()])
    graph = build_agent(
        model, registry, system="S", max_iterations=20, max_tokens=50, egress=_FakeEgress()
    )

    result = asyncio.run(graph.ainvoke({"messages": [HumanMessage(content=long_task)]}))

    assert result.get("summary"), "超阈值应触发压缩并生成摘要"
    assert model.summary_calls >= 1
    # 原任务消息被压缩删掉，最终消息里不应再有那条长 human
    assert not any(isinstance(m, HumanMessage) and "长" in str(m.content) for m in result["messages"])
    # H1 回归：摘要以 HumanMessage 落地（不可信历史），不是 SystemMessage（高权限角色）
    summary_msgs = [
        m for m in model.last_messages
        if isinstance(m, HumanMessage) and "[历史摘要]" in str(m.content)
    ]
    assert summary_msgs, "摘要应以 HumanMessage 落地"
    system_msgs = [m for m in model.last_messages if isinstance(m, SystemMessage)]
    assert len(system_msgs) == 1, "只允许一条真实 system，摘要不得再占 system 角色"


def test_agent_injects_skill_index_into_system() -> None:
    """Skills 渐进披露：skill 目录拼进 system 提示（正文按需读，不进上下文）。"""
    class _FakeSkills:
        def __len__(self) -> int:
            return 1

        def render_index(self, max_chars: int = 1500) -> str:
            return "- demo_skill: 演示技能"

    model = _ScriptedModel([AIMessage(content="done")])
    registry = ToolRegistry([_EchoTool()])
    graph = build_agent(model, registry, system="S", max_iterations=8, skills=_FakeSkills())
    asyncio.run(graph.ainvoke({"messages": [HumanMessage(content="hi")]}))

    first = model.last_messages[0]
    assert isinstance(first, SystemMessage)
    assert first.content.startswith("S")  # 原 system 仍是前缀（缓存锚点）
    assert "demo_skill" in first.content


def test_agent_does_not_compact_without_egress() -> None:
    """F 回归：没有 egress 时不压缩（fail-closed），绝不把未脱敏历史发给摘要模型。"""
    long_task = "长" * 300
    responses = [_tool_call_echo("x", f"c{i}") for i in range(3)] + [AIMessage(content="done")]
    model = _ScriptedModel(responses)
    registry = ToolRegistry([_EchoTool()])
    graph = build_agent(model, registry, system="S", max_iterations=20, max_tokens=50)
    result = asyncio.run(graph.ainvoke({"messages": [HumanMessage(content=long_task)]}))
    assert not result.get("summary")
    assert getattr(model, "summary_calls", 0) == 0
