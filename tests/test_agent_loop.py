"""循环层的单元测试：用假模型驱动，不触碰真实 API。

这里覆盖三条最重要的停条件：正常结束、卡死、工具连续失败。
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from yaro.capabilities.registry import ToolRegistry
from yaro.contracts import ToolResult, ToolSpec, Usage
from yaro.llm.deepseek import AssistantTurn
from yaro.loop.agent import Agent, Budget
from yaro.memory.artifacts import ArtifactStore
from yaro.memory.session import SessionStore


class FakeLLM:
    """按脚本返回固定回复；用完脚本后返回"完成"。"""

    model = "deepseek-flash"

    def __init__(self, script: list[dict[str, Any]]) -> None:
        self.script = list(script)
        self.calls = 0

    async def complete(self, messages: list[dict[str, Any]], tools: Any = None, **kwargs: Any) -> AssistantTurn:
        self.calls += 1
        item = self.script.pop(0) if self.script else {"content": "完成"}
        return AssistantTurn(
            content=item.get("content", ""),
            reasoning_content=item.get("reasoning_content", ""),
            tool_calls=item.get("tool_calls", []),
            usage=Usage(hit=1_000, miss=100, out=50, calls=1),
            model=self.model,
        )


class EchoTool:
    spec = ToolSpec(
        name="echo",
        description="回显给定文本",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    )

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    async def run(self, **kwargs: Any) -> ToolResult:
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        return ToolResult(ok=True, summary=f"echo {kwargs.get('text')}")


def _tool_call(name: str = "echo", arguments: str = '{"text":"hi"}', call_id: str = "c1") -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _build(tmp_path, script, *, tool=None, budget: Budget | None = None):
    session = SessionStore.create(tmp_path / "sessions", "s1")
    artifacts = ArtifactStore(tmp_path / "artifacts")
    registry = ToolRegistry([tool or EchoTool()])
    agent = Agent(
        llm=FakeLLM(script),
        registry=registry,
        session=session,
        artifacts=artifacts,
        budget=budget or Budget(),
    )
    return agent, session


def test_agent_runs_tool_then_finishes(tmp_path) -> None:
    script = [
        {"content": "", "reasoning_content": "先调工具", "tool_calls": [_tool_call()]},
        {"content": "完成"},
    ]
    agent, session = _build(tmp_path, script)
    result = asyncio.run(agent.run("任务"))

    assert result.stop_reason == "done"
    assert result.turns == 2
    assert [event.type for event in session.all_events()] == ["user", "assistant", "tool", "assistant", "run_end"]
    # reasoning_content 必须落库，否则下一轮带 tools 的请求会 400
    assistant_events = [event for event in session.all_events() if event.type == "assistant"]
    assert assistant_events[0].payload["reasoning_content"] == "先调工具"
    assert "echo hi" in session.all_events()[2].payload["content"]
    assert session.verify()["ok"]


def test_stuck_detection_stops_repeated_identical_calls(tmp_path) -> None:
    script = [{"content": "", "tool_calls": [_tool_call()]} for _ in range(5)]
    agent, session = _build(tmp_path, script, budget=Budget(max_identical_calls=2))
    result = asyncio.run(agent.run("重复调用任务"))
    assert result.stop_reason == "stuck"
    assert any(event.payload.get("stop_reason") == "stuck" for event in session.all_events() if event.type == "run_end")


def test_consecutive_tool_errors_stop(tmp_path) -> None:
    script = [{"content": "", "tool_calls": [_tool_call()]} for _ in range(3)]
    agent, _ = _build(tmp_path, script, tool=EchoTool(fail=True), budget=Budget(max_consecutive_tool_errors=2))
    result = asyncio.run(agent.run("工具一直失败"))
    assert result.stop_reason == "tool_errors"


def test_cost_budget_stops(tmp_path) -> None:
    script = [{"content": "", "tool_calls": [_tool_call()]} for _ in range(5)]
    agent, _ = _build(tmp_path, script, budget=Budget(max_cost_usd=1e-9))
    result = asyncio.run(agent.run("很贵"))
    assert result.stop_reason == "max_cost"
    assert result.usage.calls >= 1


def test_turn_budget_stops(tmp_path) -> None:
    script = [{"content": "", "tool_calls": [_tool_call()]} for _ in range(5)]
    agent, _ = _build(tmp_path, script, budget=Budget(max_turns=1))
    result = asyncio.run(agent.run("只给一轮"))
    assert result.stop_reason == "max_turns"
    assert result.turns == 1


def test_registry_unknown_tool_is_tool_result(tmp_path) -> None:
    registry = ToolRegistry([EchoTool()])

    async def call() -> ToolResult:
        return await registry.call("nope", "{}")

    result = asyncio.run(call())
    assert result.ok is False
    assert "未知工具" in result.summary


def test_registry_bad_json_gives_schema_hint() -> None:
    registry = ToolRegistry([EchoTool()])

    async def call() -> ToolResult:
        return await registry.call("echo", "{not json")

    result = asyncio.run(call())
    assert result.ok is False
    assert result.hint is not None and "schema" in result.hint


def test_registry_swallows_tool_exception() -> None:
    registry = ToolRegistry([EchoTool(fail=True)])

    async def call() -> ToolResult:
        return await registry.call("echo", '{"text":"x"}')

    result = asyncio.run(call())
    assert result.ok is False
    assert "RuntimeError" in result.summary
    assert result.degraded == ["RuntimeError"]


def test_budget_defaults_are_sane() -> None:
    budget = Budget()
    assert budget.max_turns == 50
    assert budget.max_cost_usd == pytest.approx(0.5)
    assert budget.max_identical_calls >= 2
