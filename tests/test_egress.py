"""出网策略测试：默认拒绝、显式授权、图像敏感度、脱敏与深拷贝。"""

from __future__ import annotations

import pytest

from yaro.contracts import ContractError
from yaro.egress import (
    DERIVED,
    MODE_DERIVED,
    MODE_FULL,
    MODE_OFF,
    MODE_TEXT,
    RAW,
    EgressPolicy,
    build_policy,
    host_of,
    is_private_host,
    normalize_mode,
    summarize_policy,
)


def test_normalize_mode_defaults_to_off_and_rejects_unknown():
    assert normalize_mode(None) == MODE_OFF
    assert normalize_mode("") == MODE_OFF
    assert normalize_mode("TEXT") == MODE_TEXT
    with pytest.raises(ContractError):
        normalize_mode("public")


def test_host_of_handles_urls_and_bare_hosts():
    assert host_of("https://api.deepseek.com") == "api.deepseek.com"
    assert host_of("http://127.0.0.1:8000/v1") == "127.0.0.1"
    assert host_of("api.deepseek.com") == "api.deepseek.com"
    assert host_of("") == ""


def test_private_hosts():
    for host in ("localhost", "127.0.0.1", "::1", "10.1.2.3", "192.168.1.5", "172.16.0.9", "box.local"):
        assert is_private_host(host) is True, host
    for host in ("api.deepseek.com", "8.8.8.8", ""):
        assert is_private_host(host) is False, host


def test_off_mode_refuses_even_local_endpoint():
    policy = EgressPolicy(mode=MODE_OFF)
    with pytest.raises(ContractError) as exc:
        policy.check_endpoint("http://127.0.0.1:8000/v1")
    assert "off" in str(exc.value)


def test_text_mode_allows_local_but_not_public_without_consent():
    policy = EgressPolicy(mode=MODE_TEXT)
    policy.check_endpoint("http://127.0.0.1:8000/v1")  # 不抛
    with pytest.raises(ContractError) as exc:
        policy.check_endpoint("https://api.deepseek.com")
    assert "公网" in str(exc.value)
    EgressPolicy(mode=MODE_TEXT, allow_public=True).check_endpoint("https://api.deepseek.com")


def test_image_sensitivity_by_mode():
    off = EgressPolicy(mode=MODE_OFF)
    text = EgressPolicy(mode=MODE_TEXT)
    derived = EgressPolicy(mode=MODE_DERIVED)
    full = EgressPolicy(mode=MODE_FULL)
    for policy in (off, text):
        assert policy.allows_images(DERIVED) is False
        assert policy.allows_images(RAW) is False
    assert derived.allows_images(DERIVED) is True
    assert derived.allows_images(RAW) is False
    assert full.allows_images(RAW) is True
    assert full.allows_images(DERIVED) is True


def test_redact_paths_and_secrets():
    policy = build_policy(
        MODE_TEXT,
        redact_paths=("C:\\work\\project", "C:\\Users\\dev\\.yaro"),
        secrets=("sk-abcdef1234567890",),
    )
    text = "open C:\\work\\project\\a.png with key sk-abcdef1234567890 and Bearer abcdefghijklmnop"
    out = policy.redact(text)
    assert "C:\\work\\project" not in out
    assert "sk-abcdef1234567890" not in out
    assert "<workspace>" in out


def test_filter_messages_copies_and_strips_images():
    policy = EgressPolicy(mode=MODE_TEXT, secrets=("sk-1234567890abcdef",))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "看一下 C:\\ws\\a.png"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
            ],
        },
        {
            "role": "assistant",
            "content": "ok",
            "reasoning_content": "key=sk-1234567890abcdef",
            "tool_calls": [{"function": {"name": "f", "arguments": '{"p": "C:\\\\ws\\\\a.png"}'}}],
        },
    ]
    filtered = policy.filter_messages(messages)
    # 原消息不被就地改写（会话账本不能被出网过滤污染）
    assert messages[0]["content"][1]["type"] == "image_url"
    assert filtered[0]["content"] == [{"type": "text", "text": "看一下 C:\\ws\\a.png"}]
    assert "sk-1234567890abcdef" not in filtered[1]["reasoning_content"]


def test_build_policy_sorts_longer_paths_first():
    policy = build_policy(MODE_TEXT, redact_paths=("C:\\ws", "C:\\ws\\deep\\project"))
    assert policy.redactions[0][0] == "C:\\ws\\deep\\project"
    assert policy.redactions[1][0] == "C:\\ws"


def test_describe_and_summary_are_human_readable():
    policy = EgressPolicy(mode=MODE_DERIVED)
    described = policy.describe()
    assert described["mode"] == MODE_DERIVED
    assert described["images"] == DERIVED
    summary = summarize_policy(policy, "http://127.0.0.1:8000/v1")
    assert "egress=derived" in summary
    assert "本机" in summary
