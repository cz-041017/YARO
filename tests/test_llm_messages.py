"""传输层编码规则的单元测试（不需要真实 API Key）。

这里守的是 DeepSeek 的四个 400 陷阱里最贵的一个：带 tools 时必须回传 reasoning_content。
"""

from __future__ import annotations

import pytest

from yaro.contracts import ContractError
from yaro.llm.deepseek import assert_vision_placement, is_peak, to_api_messages


def test_missing_reasoning_content_with_tools_raises() -> None:
    messages = [{"role": "assistant", "content": "我在想", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "x", "arguments": "{}"}}
    ]}]
    with pytest.raises(ContractError, match="reasoning_content"):
        to_api_messages(messages, has_tools=True)


def test_without_tools_missing_reasoning_is_tolerated() -> None:
    messages = [{"role": "assistant", "content": "纯文本回答"}]
    assert to_api_messages(messages, has_tools=False) == [{"role": "assistant", "content": "纯文本回答"}]


def test_reasoning_content_is_passed_back_verbatim() -> None:
    messages = [
        {"role": "user", "content": "任务"},
        {"role": "assistant", "content": "", "reasoning_content": "先看证据", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "ground", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "OK: 2 个实例"},
    ]
    encoded = to_api_messages(messages, has_tools=True)
    assert encoded[1]["reasoning_content"] == "先看证据"
    assert encoded[1]["tool_calls"][0]["id"] == "c1"
    assert encoded[2]["tool_call_id"] == "c1"


def test_tool_message_requires_call_id() -> None:
    with pytest.raises(ContractError, match="tool_call_id"):
        to_api_messages([{"role": "tool", "content": "结果"}], has_tools=True)


def test_unknown_role_rejected() -> None:
    with pytest.raises(ContractError, match="role"):
        to_api_messages([{"role": "robot", "content": "hi"}], has_tools=False)


def test_vision_only_in_user_messages() -> None:
    image_part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}
    with pytest.raises(ContractError, match="图片只允许"):
        assert_vision_placement([{"role": "system", "content": [image_part]}])
    # user 里带图是合法的
    assert_vision_placement([{"role": "user", "content": [image_part, {"type": "text", "text": "看图"}]}])


def test_is_peak_window() -> None:
    from datetime import datetime, timezone

    # 2026-11-02 是周一；UTC 02:00 属于峰时，UTC 23:00 属于谷时
    monday_peak = datetime(2026, 11, 2, 2, 0, tzinfo=timezone.utc)
    monday_offpeak = datetime(2026, 11, 2, 23, 0, tzinfo=timezone.utc)
    saturday = datetime(2026, 11, 7, 2, 0, tzinfo=timezone.utc)
    assert is_peak(monday_peak) is True
    assert is_peak(monday_offpeak) is False
    assert is_peak(saturday) is False
