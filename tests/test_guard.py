"""GuardedModel（出网守卫）的离线单测。不发起网络请求。"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from yaro.contracts import ContractError
from yaro.egress import build_policy
from yaro.provider.guard import GuardedModel


class _RecordingModel:
    """记录最近一次调用消息的假模型（不实现完整 BaseChatModel 接口）。"""

    def __init__(self) -> None:
        self.last: list[Any] = []

    def invoke(self, messages: Any, **kwargs: Any) -> str:
        self.last = list(messages)
        return "ok"

    async def ainvoke(self, messages: Any, **kwargs: Any) -> str:
        self.last = list(messages)
        return "ok"

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "_RecordingModel":
        return self


def _policy(mode: str, *, allow_public: bool = False) -> Any:
    return build_policy(
        mode,
        allow_public=allow_public,
        redact_paths=("C:\\work\\secret", "C:\\home\\u"),
        secrets=("sk-super-secret-123456",),
    )


def test_text_redacts_paths_and_secrets() -> None:
    m = _RecordingModel()
    g = GuardedModel(m, policy=_policy("text"), base_url="http://127.0.0.1:8000/v1")
    g.invoke(
        [
            HumanMessage(content="读 C:\\work\\secret\\a.png，用 sk-super-secret-123456"),
            AIMessage(
                content="好",
                additional_kwargs={"reasoning_content": "路径 C:\\home\\u\\x.txt"},
            ),
        ]
    )
    assert "C:\\work\\secret" not in m.last[0].content
    assert "sk-super-secret-123456" not in m.last[0].content
    assert "<workspace>" in m.last[0].content
    assert "C:\\home\\u" not in m.last[1].additional_kwargs["reasoning_content"]


def test_public_endpoint_requires_allow_public() -> None:
    g = GuardedModel(_RecordingModel(), policy=_policy("text"), base_url="https://api.deepseek.com")
    with pytest.raises(ContractError):
        g.invoke([HumanMessage("hi")])


def test_off_mode_always_refused() -> None:
    g = GuardedModel(_RecordingModel(), policy=_policy("off"), base_url="http://127.0.0.1:8000/v1")
    with pytest.raises(ContractError):
        g.invoke([HumanMessage("hi")])


def test_text_mode_strips_images() -> None:
    m = _RecordingModel()
    g = GuardedModel(m, policy=_policy("text"), base_url="http://127.0.0.1:8000/v1")
    g.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": "看这张图"},
                    {"type": "image_url", "image_url": {"url": "http://x/1.png"}},
                ]
            )
        ]
    )
    kinds = [b.get("type") for b in m.last[0].content if isinstance(b, dict)]
    assert "image_url" not in kinds


def test_full_mode_keeps_images() -> None:
    m = _RecordingModel()
    g = GuardedModel(m, policy=_policy("full"), base_url="http://127.0.0.1:8000/v1")
    g.invoke(
        [HumanMessage(content=[{"type": "image_url", "image_url": {"url": "http://x/1.png"}}])]
    )
    kinds = [b.get("type") for b in m.last[0].content if isinstance(b, dict)]
    assert "image_url" in kinds


def test_tool_call_args_redacted() -> None:
    m = _RecordingModel()
    g = GuardedModel(m, policy=_policy("text"), base_url="http://127.0.0.1:8000/v1")
    g.invoke(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "list_files", "args": {"path": "C:\\work\\secret"}, "id": "c1"}],
            )
        ]
    )
    dumped = m.last[0].tool_calls[0]
    args = dumped.get("args") if isinstance(dumped, dict) else getattr(dumped, "args", {})
    assert "C:\\work\\secret" not in str(args)
