"""provider 层（DeepSeek 适配器）的离线单测。

不发起任何网络请求：只构造 ``_get_request_payload`` 的输出并断言 wire 层正确性。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from yaro.provider import ChatDeepSeek, build_model, deepseek_extra_body


def _model(*, thinking: bool = True) -> ChatDeepSeek:
    return ChatDeepSeek(
        model="deepseek-flash",
        api_key="sk-test",
        base_url="https://api.deepseek.com",
        extra_body=deepseek_extra_body(thinking=thinking, effort="high"),
    )


def test_deepseek_extra_body() -> None:
    assert deepseek_extra_body(thinking=True, effort="high") == {
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
    }
    assert deepseek_extra_body(thinking=False) == {"thinking": {"type": "disabled"}}
    assert deepseek_extra_body(thinking=True, effort=None) == {
        "thinking": {"type": "enabled"}
    }


def test_reasoning_content_roundtrip() -> None:
    model = _model(thinking=True)
    msgs = [
        SystemMessage(content="S"),
        HumanMessage(content="hi"),
        AIMessage(
            content="",
            tool_calls=[{"name": "f", "args": {}, "id": "c1"}],
            additional_kwargs={"reasoning_content": "我想用工具"},
        ),
        ToolMessage(content="result", tool_call_id="c1"),
    ]
    payload = model._get_request_payload(msgs)
    out = payload["messages"]
    assistant = next(m for m in out if m.get("role") == "assistant")
    assert assistant["reasoning_content"] == "我想用工具"
    assert assistant.get("tool_calls") is not None


def test_reasoning_content_captured_from_response() -> None:
    """F1 回归：响应解析层必须把 reasoning_content 抄进 additional_kwargs。

    没有这个，`test_reasoning_content_roundtrip` 是假绿（手工塞了 additional_kwargs，
    生产里 ChatOpenAI 根本不会填它）。
    """
    model = _model(thinking=True)
    response = {
        "id": "chatcmpl-x",
        "model": "deepseek-flash",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "我先看看数据",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "dataset_scan", "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    result = model._create_chat_result(response)
    msg = result.generations[0].message
    assert isinstance(msg, AIMessage)
    assert msg.additional_kwargs.get("reasoning_content") == "我先看看数据"
    assert msg.tool_calls


def test_reasoning_content_echoed_without_tool_calls() -> None:
    """H2 回归：无 tool_calls 的 assistant 消息也必须回传 reasoning_content（续跑才不 400）。"""
    model = _model(thinking=True)
    msgs = [
        HumanMessage(content="hi"),
        AIMessage(content="我先想想", additional_kwargs={"reasoning_content": "无工具的思考"}),
        HumanMessage(content="继续"),
    ]
    payload = model._get_request_payload(msgs)
    assistant = next(m for m in payload["messages"] if m.get("role") == "assistant")
    assert assistant.get("reasoning_content") == "无工具的思考"


def test_reasoning_content_empty_string_fallback() -> None:
    model = _model(thinking=True)
    msgs = [AIMessage(content="hi")]
    payload = model._get_request_payload(msgs)
    assistant = payload["messages"][0]
    assert assistant.get("reasoning_content") == ""


def test_no_reasoning_when_not_thinking() -> None:
    model = _model(thinking=False)
    msgs = [
        AIMessage(
            content="",
            tool_calls=[{"name": "f", "args": {}, "id": "c1"}],
            additional_kwargs={"reasoning_content": "x"},
        )
    ]
    payload = model._get_request_payload(msgs)
    assistant = next(m for m in payload["messages"] if m.get("role") == "assistant")
    assert "reasoning_content" not in assistant


def test_tool_list_content_stringified() -> None:
    model = _model(thinking=True)
    msgs = [
        HumanMessage(content="hi"),
        AIMessage(
            content="",
            tool_calls=[{"name": "f", "args": {}, "id": "c1"}],
            additional_kwargs={"reasoning_content": ""},
        ),
        ToolMessage(content=[{"type": "text", "text": "a"}], tool_call_id="c1"),
    ]
    payload = model._get_request_payload(msgs)
    tool = next(m for m in payload["messages"] if m.get("role") == "tool")
    assert isinstance(tool["content"], str)
    assert json.loads(tool["content"]) == [{"type": "text", "text": "a"}]


def test_vision_in_non_user_raises() -> None:
    model = _model()
    msgs = [
        SystemMessage(
            content=[{"type": "image_url", "image_url": {"url": "http://x/1.png"}}]
        )
    ]
    with pytest.raises(ValueError):
        model._get_request_payload(msgs)


def test_build_model() -> None:
    s = SimpleNamespace(
        model="deepseek-flash",
        model_plan="deepseek-v4-pro",
        api_key="sk-x",
        base_url="https://api.deepseek.com",
    )
    m = build_model(s)
    assert m.model_name == "deepseek-flash"
    assert m.extra_body["thinking"]["type"] == "enabled"
    assert m.extra_body.get("reasoning_effort") == "high"
