"""graph.runner 会话桥的离线单测：SessionStore 事件 ↔ LangGraph 图状态。

证明旧 loop.agent 的事件形状被新图引擎原样保留（GUI / rewind / fork / verify 不感知）。
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from yaro.capabilities.registry import ToolRegistry
from yaro.contracts import ArtifactRef, Evidence, ToolResult, ToolSpec, Usage
from yaro.graph.runner import _safe_cost, run_agent, session_to_langchain
from yaro.memory.session import SessionStore


class _FakeEgress:
    """压缩摘要路径需要的 egress 桩（summarize_prefix 会调 redact）。"""

    def redact(self, text: str) -> str:
        return text


class _RichTool:
    """带 evidence/artifacts/degraded/hint/sensitive 全字段的工具（快照断言用）。"""

    spec = ToolSpec(
        name="rich",
        description="rich result",
        parameters={"type": "object", "properties": {}},
    )

    async def run(self, **kw) -> ToolResult:
        return ToolResult(
            ok=False,
            summary="partial",
            metrics={"m": 1.0},
            evidence=[Evidence(kind="table", payload=[{"a": 1}], caption="cap")],
            artifacts=[ArtifactRef(uri="out/a.png", sha256="deadbeef", media_type="image/png", size=5)],
            degraded=["cpu"],
            hint="hint text",
            sensitive=True,
        )


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


class _ScriptedModel:
    """按顺序返回 AIMessage；可选地在指定轮抛异常（模拟第 N 轮模型/网络故障）。"""

    def __init__(
        self,
        responses: list[AIMessage],
        *,
        error: str | None = None,
        fail_on_call: int | None = None,
    ):
        self._responses = list(responses)
        self.calls = 0
        self._bound: object | None = None
        self.error = error
        self.fail_on_call = fail_on_call

    def bind_tools(self, tools):
        self._bound = tools
        return self

    def invoke(self, messages):
        return "压缩摘要"

    async def ainvoke(self, messages, **kwargs):
        self.last_messages = list(messages)
        i = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        # error 只是异常消息；fail_on_call 是 0 基（fail_on_call=1 表示第 2 次调用抛）；
        # None 表示每次都抛。
        should_raise = (
            self.fail_on_call is not None and self.calls - 1 == self.fail_on_call
        ) or (self.fail_on_call is None and self.error is not None)
        if should_raise:
            raise RuntimeError(self.error or "boom")
        return self._responses[i]


def _tool_call(text: str, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "echo", "args": {"text": text}, "id": call_id}])


def _usage_msg(hit: int, miss: int, out: int) -> AIMessage:
    return AIMessage(
        content="done",
        response_metadata={
            "deepseek_usage": {
                "prompt_cache_hit_tokens": hit,
                "prompt_cache_miss_tokens": miss,
                "prompt_tokens": hit + miss,
                "completion_tokens": out,
            }
        },
    )


def _events(store: SessionStore) -> list[tuple[str, dict]]:
    return [(e.type, e.payload) for e in store.all_events()]


# --------------------------------------------------------------------------- #
# session_to_langchain
# --------------------------------------------------------------------------- #


def test_session_to_langchain_roundtrips() -> None:
    openai = [
        {"role": "user", "content": "say hi"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "thinking",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": '{"text": "hi"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "OK: echo:hi"},
        {"role": "assistant", "content": "done", "reasoning_content": "final"},
    ]
    msgs = session_to_langchain(openai)

    assert isinstance(msgs[0], HumanMessage) and msgs[0].content == "say hi"
    assert isinstance(msgs[1], AIMessage)
    assert msgs[1].additional_kwargs == {"reasoning_content": "thinking"}
    assert msgs[1].tool_calls == [
        {"name": "echo", "args": {"text": "hi"}, "id": "c1", "type": "tool_call"}
    ]
    assert isinstance(msgs[2], ToolMessage) and msgs[2].tool_call_id == "c1"
    assert msgs[2].content == "OK: echo:hi"
    assert isinstance(msgs[3], AIMessage) and msgs[3].content == "done"


# --------------------------------------------------------------------------- #
# run_agent
# --------------------------------------------------------------------------- #


def test_run_agent_writes_user_assistant_tool_run_end(tmp_path) -> None:
    model = _ScriptedModel([_tool_call("hi", "c1"), AIMessage(content="done")])
    registry = ToolRegistry([_EchoTool()])
    store = SessionStore.create(tmp_path, session_id="s1")

    out = asyncio.run(
        run_agent(model, registry, store, "say hi", system="S", max_iterations=8)
    )

    types = [t for t, _ in _events(store)]
    assert types == ["user", "assistant", "tool", "assistant", "run_end"]

    user = _events(store)[0][1]
    assert user["content"] == "say hi"
    assert user["run_id"] == ""

    tool = _events(store)[2][1]
    assert tool["tool_call_id"] == "c1"
    assert tool["name"] == "echo"
    assert tool["ok"] is True
    assert tool["summary"] == "echo:hi"
    assert tool["metrics"] == {"len": 2.0}

    run_end = _events(store)[-1][1]
    assert run_end["stop_reason"] == "done"
    assert run_end["turn"] == 2

    assert out["stop_reason"] == "done"
    assert out["turns"] == 2


def test_run_agent_extracts_deepseek_usage(tmp_path) -> None:
    """usage 从 response_metadata["deepseek_usage"] 拆分 hit/miss（缓存价差记账）。"""
    model = _ScriptedModel([_usage_msg(hit=100, miss=50, out=20)])
    registry = ToolRegistry([_EchoTool()])
    store = SessionStore.create(tmp_path, session_id="s1")

    out = asyncio.run(run_agent(model, registry, store, "hi", model_name="deepseek-flash"))

    usage = out["usage"]
    assert usage["hit"] == 100
    assert usage["miss"] == 50
    assert usage["out"] == 20
    assert out["cost_usd"] > 0.0


def test_run_agent_stop_check_cancels_before_model_call(tmp_path) -> None:
    model = _ScriptedModel([_tool_call("hi", "c1"), AIMessage(content="done")])
    registry = ToolRegistry([_EchoTool()])
    store = SessionStore.create(tmp_path, session_id="s1")

    out = asyncio.run(
        run_agent(model, registry, store, "hi", stop_check=lambda: True)
    )

    assert model.calls == 0, "stop_check 应在模型调用前生效"
    assert out["stop_reason"] == "stopped"
    types = [t for t, _ in _events(store)]
    assert types == ["user", "run_end"]
    assert _events(store)[-1][1]["stop_reason"] == "stopped"


def test_run_agent_max_iterations_writes_skipped_tool(tmp_path) -> None:
    forever = [_tool_call("x", f"c{i}") for i in range(100)]
    model = _ScriptedModel(forever)
    registry = ToolRegistry([_EchoTool()])
    store = SessionStore.create(tmp_path, session_id="s1")

    out = asyncio.run(run_agent(model, registry, store, "hi", max_iterations=2))

    assert out["stop_reason"] == "max_iterations"
    tools = [p for t, p in _events(store) if t == "tool"]
    # 2 次模型调用 → 1 个真实执行 + 1 个 finalize 补的 skipped。
    assert len(tools) == 2
    assert tools[0]["ok"] is True
    assert tools[-1]["ok"] is False
    assert tools[-1]["summary"] == "未执行"
    assert tools[-1]["degraded"] == ["skipped"]


def test_run_agent_error_writes_run_end_not_raise(tmp_path) -> None:
    model = _ScriptedModel([AIMessage(content="x")], error="boom")
    registry = ToolRegistry([_EchoTool()])
    store = SessionStore.create(tmp_path, session_id="s1")

    out = asyncio.run(run_agent(model, registry, store, "hi"))

    assert out["stop_reason"] == "error"
    types = [t for t, _ in _events(store)]
    assert types == ["user", "run_end"]
    run_end = _events(store)[-1][1]
    assert run_end["stop_reason"] == "error"
    assert "boom" in run_end["detail"]


def test_run_agent_compaction_keeps_events_in_order(tmp_path) -> None:
    """F1 回归：compaction（RemoveMessage 删旧前缀）不丢/不错位事件，无孤儿 tool。

    需 >=4 条消息才会触发 find_safe_cut（context.py:98），所以让模型连跑 2 轮工具，
    在第 2 轮之后触发压缩，再收尾。
    """
    model = _ScriptedModel(
        [_tool_call("x", "c1"), _tool_call("y", "c2"), AIMessage(content="done")]
    )
    registry = ToolRegistry([_EchoTool()])
    store = SessionStore.create(tmp_path, session_id="s1")

    out = asyncio.run(
        run_agent(model, registry, store, "hi", egress=_FakeEgress(), max_tokens=1)
    )

    types = [t for t, _ in _events(store)]
    # 关键：每个 assistant 都在它自己的 tool 之前（无孤儿 tool 打头、无错位）。
    assert types == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
        "run_end",
    ], types
    assert out["stop_reason"] == "done"
    assert out["turns"] == 3


def test_run_agent_midrun_error_preserves_prior_events(tmp_path) -> None:
    """H1 回归：第 2 轮抛错时，第 1 轮已完成的事件已落账（增量），run_end 带 turn>=1。"""
    model = _ScriptedModel(
        [_tool_call("hi", "c1"), AIMessage(content="done")],
        fail_on_call=1,
        error="boom",
    )
    registry = ToolRegistry([_EchoTool()])
    store = SessionStore.create(tmp_path, session_id="s1")

    out = asyncio.run(run_agent(model, registry, store, "hi"))

    types = [t for t, _ in _events(store)]
    assert types == ["user", "assistant", "tool", "run_end"], types
    run_end = _events(store)[-1][1]
    assert run_end["stop_reason"] == "error"
    assert run_end["turn"] == 1  # 第 1 轮 assistant 已记，不归零
    assert out["turns"] == 1


def test_run_agent_stuck_guard_stops_repeated_calls(tmp_path) -> None:
    """M1 回归：模型连续两轮发出同一组 tool_calls → stuck，不再烧轮。"""
    model = _ScriptedModel(
        [_tool_call("x", "c1"), _tool_call("x", "c2"), AIMessage(content="done")]
    )
    registry = ToolRegistry([_EchoTool()])
    store = SessionStore.create(tmp_path, session_id="s1")

    out = asyncio.run(run_agent(model, registry, store, "hi", max_iterations=8))

    assert model.calls == 2, "第 3 轮应被 stuck 护栏挡住，不再调模型"
    assert out["stop_reason"] == "stuck"
    tools = [p for t, p in _events(store) if t == "tool"]
    assert len(tools) == 2
    assert all(t["ok"] is True for t in tools)  # 两次都真实执行，无 skipped
    assert _events(store)[-1][1]["stop_reason"] == "stuck"


def test_run_agent_stuck_does_not_leak_across_runs(tmp_path) -> None:
    """F 回归：一次 stuck 后，同一会话的新任务不被历史里的重复调用 brick。

    旧实现 _is_stuck 扫完整会话历史（无 user 轮边界），run1 的两次相同调用会让 run2
    立即 stuck 且模型根本没被调用（calls=0/turns=0）。护栏必须只比较当前 user 轮。
    """
    store = SessionStore.create(tmp_path, session_id="s1")

    # run1：两次相同调用 → stuck。
    model1 = _ScriptedModel(
        [_tool_call("x", "c1"), _tool_call("x", "c2"), AIMessage(content="done")]
    )
    out1 = asyncio.run(
        run_agent(model1, ToolRegistry([_EchoTool()]), store, "task1", max_iterations=8)
    )
    assert out1["stop_reason"] == "stuck"
    assert model1.calls == 2

    # run2：全新任务、全新模型实例、同一 store——必须正常调用模型，不继承 run1 的 stuck。
    model2 = _ScriptedModel([_tool_call("hi", "c3"), AIMessage(content="done")])
    out2 = asyncio.run(
        run_agent(model2, ToolRegistry([_EchoTool()]), store, "task2", max_iterations=8)
    )
    assert model2.calls >= 1, "护栏不应跨 user 轮泄漏，新任务必须能调模型"
    assert out2["stop_reason"] == "done"


def test_safe_cost_unknown_model_returns_zero() -> None:
    """_safe_cost：未知型号（ContractError）降级 0.0 不抛，已知型号正常计价。"""
    usage = Usage(hit=100, miss=50, out=20)
    assert _safe_cost(usage, "unknown-model-xyz", peak=False) == 0.0
    assert _safe_cost(usage, "deepseek-flash", peak=False) > 0.0


def test_run_agent_tool_event_full_field_snapshot(tmp_path) -> None:
    """L3 回归：tool 事件逐字段快照（evidence/artifacts/images/degraded/hint/sensitive）。"""
    model = _ScriptedModel(
        [AIMessage(content="", tool_calls=[{"name": "rich", "args": {}, "id": "c1"}]), AIMessage(content="done")]
    )
    registry = ToolRegistry([_RichTool()])
    store = SessionStore.create(tmp_path, session_id="s1")

    asyncio.run(run_agent(model, registry, store, "hi"))

    tool = [p for t, p in _events(store) if t == "tool"][0]
    assert tool["tool_call_id"] == "c1"
    assert tool["name"] == "rich"
    assert tool["ok"] is False
    assert tool["summary"] == "partial"
    assert tool["metrics"] == {"m": 1.0}
    assert tool["evidence"] == [{"kind": "table", "caption": "cap", "payload": [{"a": 1}]}]
    assert tool["evidence_kinds"] == ["table"]
    assert tool["artifacts"] == [
        {"uri": "out/a.png", "sha256": "deadbeef", "media_type": "image/png", "size": 5}
    ]
    # 图片块：sensitive=True → RAW；media_type 是 image/* 才进 images。
    assert tool["images"] == [
        {"uri": "out/a.png", "sha256": "deadbeef", "media_type": "image/png", "sensitivity": "raw"}
    ]
    assert tool["degraded"] == ["cpu"]
    assert tool["hint"] == "hint text"
