"""provider.demo（DemoLLM 的 LangChain 适配器）离线单测。"""

from __future__ import annotations

import asyncio
import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from yaro.contracts import Usage
from yaro.llm.deepseek import AssistantTurn
from yaro.provider.demo import DemoChatModel, _messages_to_openai, _turn_to_aimessage


def test_messages_to_openai_shapes() -> None:
    msgs = [
        SystemMessage(content="S"),
        HumanMessage(content="hi"),
        AIMessage(
            content="",
            additional_kwargs={"reasoning_content": "thinking"},
            tool_calls=[{"name": "echo", "args": {"text": "hi"}, "id": "c1"}],
        ),
        ToolMessage(content="OK", tool_call_id="c1"),
    ]
    out = _messages_to_openai(msgs)

    assert out[0] == {"role": "system", "content": "S"}
    assert out[1] == {"role": "user", "content": "hi"}
    assert out[2]["role"] == "assistant"
    assert out[2]["reasoning_content"] == "thinking"
    assert out[2]["tool_calls"][0]["id"] == "c1"
    assert out[2]["tool_calls"][0]["function"]["name"] == "echo"
    assert json.loads(out[2]["tool_calls"][0]["function"]["arguments"]) == {"text": "hi"}
    assert out[3] == {"role": "tool", "tool_call_id": "c1", "content": "OK"}


def test_turn_to_aimessage_encodes_usage_and_tool_calls() -> None:
    turn = AssistantTurn(
        content="done",
        reasoning_content="r",
        tool_calls=[
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "echo", "arguments": '{"text": "x"}'},
            }
        ],
        usage=Usage(hit=100, miss=50, out=20, calls=1),
    )
    msg = _turn_to_aimessage(turn)

    assert isinstance(msg, AIMessage)
    assert msg.content == "done"
    assert msg.additional_kwargs["reasoning_content"] == "r"
    assert msg.tool_calls[0]["name"] == "echo"
    assert msg.tool_calls[0]["args"] == {"text": "x"}
    du = msg.response_metadata["deepseek_usage"]
    assert du["prompt_cache_hit_tokens"] == 100
    assert du["prompt_cache_miss_tokens"] == 50
    assert du["prompt_tokens"] == 150
    assert du["completion_tokens"] == 20


def test_demo_chat_model_bind_and_invoke() -> None:
    model = DemoChatModel(delay_s=0.0)
    assert model.bind_tools([{"function": {"name": "echo"}}]) is model

    msg = asyncio.run(model.ainvoke([HumanMessage(content="分析这个数据集")]))

    assert isinstance(msg, AIMessage)
    # 剧本第一轮 = list_files，必带 tool_calls。
    assert msg.tool_calls, "demo 第一轮应发出工具调用"
    du = msg.response_metadata["deepseek_usage"]
    assert du["prompt_cache_hit_tokens"] > 0
    assert du["prompt_cache_miss_tokens"] >= 0
    assert du["completion_tokens"] > 0
