"""goal 模块（持久目标状态机）的离线单测。"""

from __future__ import annotations

from pathlib import Path

import pytest

from yaro.goal import (
    STATUS_ACTIVE,
    STATUS_BLOCKED,
    STATUS_COMPLETED,
    STATUS_PAUSED,
    Goal,
    GoalError,
    GoalStore,
)


def test_create_and_get(tmp_path: Path) -> None:
    store = GoalStore(tmp_path)
    g = store.create("把 demo 数据集蒸馏成小模型")
    assert g.status == STATUS_ACTIVE
    assert g.revision == 1
    assert g.objective == "把 demo 数据集蒸馏成小模型"
    assert store.get(g.goal_id) == g


def test_create_rejects_empty(tmp_path: Path) -> None:
    store = GoalStore(tmp_path)
    with pytest.raises(GoalError):
        store.create("   ")


def test_complete(tmp_path: Path) -> None:
    store = GoalStore(tmp_path)
    g = store.create("x")
    g2 = store.update(g.goal_id, g.revision, action="complete")
    assert g2.status == STATUS_COMPLETED
    assert g2.revision == 2


def test_complete_wrong_status(tmp_path: Path) -> None:
    store = GoalStore(tmp_path)
    g = store.create("x")
    store.update(g.goal_id, g.revision, action="complete")
    with pytest.raises(GoalError):
        store.update(g.goal_id, 2, action="complete")


def test_blocked_requires_reason(tmp_path: Path) -> None:
    store = GoalStore(tmp_path)
    g = store.create("x")
    with pytest.raises(GoalError):
        store.update(g.goal_id, g.revision, action="blocked", blocked_reason="")
    g2 = store.update(g.goal_id, g.revision, action="blocked", blocked_reason="缺 CUDA")
    assert g2.status == STATUS_BLOCKED
    assert g2.blocked_reason == "缺 CUDA"


def test_pause_resume(tmp_path: Path) -> None:
    store = GoalStore(tmp_path)
    g = store.create("x")
    p = store.update(g.goal_id, g.revision, action="pause")
    assert p.status == STATUS_PAUSED
    r = store.update(g.goal_id, p.revision, action="resume")
    assert r.status == STATUS_ACTIVE
    with pytest.raises(GoalError):
        store.update(g.goal_id, r.revision, action="resume")  # 已 active，不能 resume


def test_edit(tmp_path: Path) -> None:
    store = GoalStore(tmp_path)
    g = store.create("x", max_rounds=10)
    g2 = store.update(g.goal_id, g.revision, action="edit", objective="y", max_rounds=5)
    assert g2.objective == "y"
    assert g2.max_rounds == 5


def test_revision_cas(tmp_path: Path) -> None:
    store = GoalStore(tmp_path)
    g = store.create("x")
    store.update(g.goal_id, g.revision, action="edit", objective="y")
    with pytest.raises(GoalError):
        store.update(g.goal_id, g.revision, action="edit", objective="z")  # 旧 revision


def test_unknown_action_and_missing_goal(tmp_path: Path) -> None:
    store = GoalStore(tmp_path)
    g = store.create("x")
    with pytest.raises(GoalError):
        store.update(g.goal_id, g.revision, action="nope")
    with pytest.raises(GoalError):
        store.update("deadbeef", 1, action="edit", objective="z")


def test_list_skips_corrupt(tmp_path: Path) -> None:
    store = GoalStore(tmp_path)
    store.create("x")
    (tmp_path / "bad.json").write_text("{ not valid json", encoding="utf-8")
    assert len(store.list()) == 1


def test_bad_goal_id_rejected(tmp_path: Path) -> None:
    store = GoalStore(tmp_path)
    with pytest.raises(GoalError):
        store.get("../../etc/passwd")


def test_windows_reserved_name_rejected(tmp_path: Path) -> None:
    store = GoalStore(tmp_path)
    with pytest.raises(GoalError):
        store.get("CON")


def test_blocked_from_completed_rejected(tmp_path: Path) -> None:
    store = GoalStore(tmp_path)
    g = store.create("x")
    g = store.update(g.goal_id, g.revision, action="complete")
    with pytest.raises(GoalError):
        store.update(g.goal_id, g.revision, action="blocked", blocked_reason="r")


def test_pause_from_completed_rejected(tmp_path: Path) -> None:
    store = GoalStore(tmp_path)
    g = store.create("x")
    g = store.update(g.goal_id, g.revision, action="complete")
    with pytest.raises(GoalError):
        store.update(g.goal_id, g.revision, action="pause")


def test_edit_from_completed_rejected(tmp_path: Path) -> None:
    store = GoalStore(tmp_path)
    g = store.create("x")
    g = store.update(g.goal_id, g.revision, action="complete")
    with pytest.raises(GoalError):
        store.update(g.goal_id, g.revision, action="edit", objective="y")


def test_blocked_reason_cleared_on_resume(tmp_path: Path) -> None:
    store = GoalStore(tmp_path)
    g = store.create("x")
    b = store.update(g.goal_id, g.revision, action="blocked", blocked_reason="缺 CUDA")
    assert b.blocked_reason == "缺 CUDA"
    r = store.update(b.goal_id, b.revision, action="resume")
    assert r.status == STATUS_ACTIVE
    assert r.blocked_reason == ""


def test_edit_max_rounds_sentinel(tmp_path: Path) -> None:
    store = GoalStore(tmp_path)
    g = store.create("x", max_rounds=10)
    # 不传 max_rounds = 不改
    g = store.update(g.goal_id, g.revision, action="edit", objective="y")
    assert g.max_rounds == 10
    # 显式 None = 清除上限
    g = store.update(g.goal_id, g.revision, action="edit", max_rounds=None)
    assert g.max_rounds is None


def test_create_rejects_bad_max_rounds(tmp_path: Path) -> None:
    store = GoalStore(tmp_path)
    with pytest.raises(GoalError):
        store.create("x", max_rounds=0)
    with pytest.raises(GoalError):
        store.create("x", max_rounds=-3)


def test_from_dict_validation() -> None:
    with pytest.raises(GoalError):
        Goal.from_dict({"goal_id": "x", "objective": "o", "status": "nope"})
    with pytest.raises(GoalError):
        Goal.from_dict({"goal_id": "x", "objective": "o", "revision": 0})
    with pytest.raises(GoalError):
        Goal.from_dict({"goal_id": "x", "objective": "o", "revision": -1})
    with pytest.raises(GoalError):
        Goal.from_dict({"goal_id": "x", "objective": "o", "max_rounds": -2})
    with pytest.raises(GoalError):
        Goal.from_dict({"goal_id": "x", "objective": "o", "max_rounds": "ten"})
    with pytest.raises(GoalError):
        Goal.from_dict([])  # 非对象


def test_list_skips_directory_named_json(tmp_path: Path) -> None:
    store = GoalStore(tmp_path)
    store.create("x")
    (tmp_path / "dir.json").mkdir()
    assert len(store.list()) == 1


def test_get_corrupt_returns_none(tmp_path: Path) -> None:
    store = GoalStore(tmp_path)
    g = store.create("x")
    (tmp_path / f"{g.goal_id}.json").write_text("{ bad", encoding="utf-8")
    assert store.get(g.goal_id) is None
