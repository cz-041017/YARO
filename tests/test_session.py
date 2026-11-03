"""事件溯源会话树的单元测试（核心不变式）。"""

from __future__ import annotations

import json

import pytest

from yaro.contracts import ContractError
from yaro.memory.session import GENESIS, SessionStore


def _store(tmp_path) -> SessionStore:
    return SessionStore.create(tmp_path / "sessions", "s1")


def test_append_links_digests(tmp_path) -> None:
    store = _store(tmp_path)
    first = store.append("user", {"content": "标定一下"})
    second = store.append("assistant", {"content": "好", "reasoning_content": "想"})
    assert first.prev_digest == GENESIS
    assert second.prev_digest == first.digest
    assert second.parent == first.event_id
    assert store.head("main") == second.event_id
    assert store.verify()["ok"]


def test_messages_reduce_includes_reasoning_content(tmp_path) -> None:
    store = _store(tmp_path)
    store.append("user", {"content": "跑一遍"})
    store.append("assistant", {"content": "", "reasoning_content": "先看图", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "ground", "arguments": "{}"}}
    ]})
    store.append("tool", {"tool_call_id": "c1", "content": "OK: 3 个实例"})
    messages = store.messages("main")
    assert [message["role"] for message in messages] == ["user", "assistant", "tool"]
    assert messages[1]["reasoning_content"] == "先看图"
    assert messages[2]["tool_call_id"] == "c1"


def test_rewind_moves_head_without_deleting(tmp_path) -> None:
    store = _store(tmp_path)
    store.append("user", {"content": "第一轮"})
    second = store.append("assistant", {"content": "回复1"})
    store.append("user", {"content": "第二轮"})
    store.rewind(second.event_id)
    assert store.head("main") == second.event_id
    assert len(store.all_events()) == 3  # 事件没被删
    assert len(store.messages("main")) == 2


def test_fork_from_history(tmp_path) -> None:
    store = _store(tmp_path)
    base = store.append("user", {"content": "起点"})
    store.append("assistant", {"content": "A"})
    store.fork(base.event_id, "experiment")
    assert store.head("experiment") == base.event_id
    store.append("assistant", {"content": "B"}, branch="experiment")
    assert store.head("main") != store.head("experiment")
    assert [message["content"] for message in store.messages("experiment")] == ["起点", "B"]


def test_rewind_rejects_foreign_event(tmp_path) -> None:
    store = _store(tmp_path)
    store.append("user", {"content": "x"})
    other = store.append("user", {"content": "unreachable-after-rewind"})
    store.rewind(store.lineage()[0].event_id)
    # other 现在不在 main 的历史上
    with pytest.raises(ContractError):
        store.rewind(other.event_id)


def test_tamper_detection(tmp_path) -> None:
    store = _store(tmp_path)
    store.append("user", {"content": "原始内容"})
    path = store.events_path
    data = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    data["payload"]["content"] = "被篡改"
    path.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")

    reopened = SessionStore.open(tmp_path / "sessions", "s1")
    report = reopened.verify()
    assert report["ok"] is False
    assert any("digest" in problem for problem in report["problems"])


def test_compaction_requires_turn_boundary(tmp_path) -> None:
    store = _store(tmp_path)
    store.append("user", {"content": "任务"})
    assistant = store.append("assistant", {"content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "run", "arguments": "{}"}}
    ]})
    store.append("tool", {"tool_call_id": "c1", "content": "结果"})
    with pytest.raises(ContractError):
        store.append_compaction("摘要", assistant.event_id)


def test_compaction_collapses_history(tmp_path) -> None:
    store = _store(tmp_path)
    store.append("user", {"content": "第一阶段"})
    store.append("assistant", {"content": "阶段一结论：RMS=0.42 超标"})
    end = store.append("run_end", {"stop_reason": "done"})
    store.append("user", {"content": "第二阶段"})

    store.append_compaction("阶段一：RMS 0.42 未达标，已补拍边缘视角", end.event_id)
    messages = store.messages("main")
    assert messages[0]["content"].startswith("[CONTEXT COMPACTION]")
    assert any("第二阶段" in str(message.get("content")) for message in messages)
    # 压缩后链仍然完整，且旧事件还在
    assert store.verify()["ok"]
    assert len(store.all_events()) == 5


def test_genesis_constant() -> None:
    assert GENESIS == "0" * 64
