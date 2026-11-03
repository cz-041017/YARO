"""垂直切片运行时 run_distillation 的离线端到端单测。

用 fake model + fake vision + 真 MemorySaver + 真 AuditLedger 驱动，覆盖：
- 单轮跑完 + 审计链完整
- interrupt/resume（review + approve 两种 kind）
- 审批 deny / ask / allow 三条路径
"""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from yaro.approval import ApprovalPolicy, ApprovalRule
from yaro.audit import AuditLedger
from yaro.goal import GoalStore
from yaro.graph.runtime import run_distillation


class FakeVision:
    def __init__(self) -> None:
        self.prelabel_calls = 0
        self.train_calls = 0
        self.apply_calls = 0
        self.last_overwrite = None

    def prelabel(self, dataset: str, **kw: Any) -> dict[str, Any]:
        self.prelabel_calls += 1
        return {"warnings": []}

    def review_queue(self, dataset: str) -> dict[str, Any]:
        return {"pending": ["a.txt", "b.txt"], "count": 2}

    def train(self, dataset: str, **kw: Any) -> dict[str, Any]:
        self.train_calls += 1
        return {"ok": True, "best_map50": 0.5, "epochs_done": 10, "findings": []}

    def apply_proposals(self, dataset: str, **kw: Any) -> dict[str, Any]:
        self.apply_calls += 1
        self.last_overwrite = kw.get("overwrite")
        return {"applied": 2, "kept": 0, "overwrite": bool(kw.get("overwrite"))}

    def select_hard(self, dataset: str, **kw: Any) -> dict[str, Any]:
        # L1 修复：完整相对路径，与真实 select_hard_examples 契约一致。
        return {"hard": ["images/a.jpg", "images/b.jpg"], "count": 2, "scores": {}, "ok": True, "errors": []}


class FakeModel:
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls = 0

    def invoke(self, messages: Any) -> str:
        r = self._replies[min(self.calls, len(self._replies) - 1)]
        self.calls += 1
        return r


def _resolve(item: Any) -> dict[str, Any]:
    """按 interrupt 的 kind 决定 resume 值。"""
    kind = item.get("kind") if isinstance(item, dict) else None
    if kind == "review":
        return {"approve": True}
    if kind == "approve":
        return {"approve": True}
    return {"approve": False}


def _run_to_end(kw: dict[str, Any], resolve=None) -> dict[str, Any]:
    resolve = resolve or _resolve
    out = run_distillation(**kw)
    guard = 0
    while out["interrupted"] is not None:
        guard += 1
        if guard >= 20:
            pytest.fail("resume 循环超过 20 步，疑似无限循环")
        value = resolve(out["interrupted"])
        out = run_distillation(resume=value, thread_id=out["thread_id"], **kw)
    return out


def _kw(tmp_path, model, vision, approval=None) -> dict[str, Any]:
    return dict(
        dataset="demo/dataset",
        model=model,
        vision=vision,
        approval=approval,
        checkpointer=MemorySaver(),
        audit=AuditLedger.create(tmp_path / "events.jsonl"),
    )


def test_requires_audit(tmp_path) -> None:
    vision = FakeVision()
    model = FakeModel(["停止"])
    with pytest.raises(TypeError, match="audit"):
        run_distillation(dataset="d", model=model, vision=vision, checkpointer=MemorySaver())


def test_single_round_end_to_end(tmp_path) -> None:
    vision = FakeVision()
    model = FakeModel(["停止：已达标"])
    out = _run_to_end(_kw(tmp_path, model, vision))

    assert out["interrupted"] is None
    state = out["state"]
    assert state["stop_reason"] == "done"
    assert state["iteration"] == 1
    assert vision.train_calls == 1
    assert out["audit"]["ok"] is True
    # label + review + train + analyze，至少 4 条审计事件
    assert out["audit"]["events"] >= 4


def test_loop_then_stop(tmp_path) -> None:
    vision = FakeVision()
    model = FakeModel(["继续：降 lr", "停止"])
    out = _run_to_end(_kw(tmp_path, model, vision))

    assert out["interrupted"] is None
    assert out["state"]["iteration"] == 2
    assert vision.train_calls == 2
    assert out["audit"]["ok"] is True


def test_approval_deny_blocks_training(tmp_path) -> None:
    vision = FakeVision()
    model = FakeModel(["停止"])
    policy = ApprovalPolicy(mode="default", rules=(ApprovalRule("train_run", "deny"),))
    out = _run_to_end(_kw(tmp_path, model, vision, approval=policy))

    assert out["interrupted"] is None
    assert vision.train_calls == 0
    assert out["state"]["metrics"]["ok"] is False
    assert out["state"]["metrics"].get("denied") is True


def test_approval_ask_interrupts_then_allows(tmp_path) -> None:
    vision = FakeVision()
    model = FakeModel(["停止"])

    def resolve_ask(item: Any) -> dict[str, Any]:
        kind = item.get("kind") if isinstance(item, dict) else None
        if kind == "approve":
            return {"approve": True}
        return {"approve": True}

    out = _run_to_end(
        _kw(tmp_path, model, vision, approval=ApprovalPolicy(mode="default")),
        resolve=resolve_ask,
    )

    # default 模式下 train_run 在 ask 列表里 → 会先 interrupt(approve) 再放行
    assert out["interrupted"] is None
    assert vision.train_calls == 1
    assert out["audit"]["ok"] is True


def test_approval_ask_reject_blocks(tmp_path) -> None:
    vision = FakeVision()
    model = FakeModel(["停止"])

    def resolve_reject(item: Any) -> dict[str, Any]:
        kind = item.get("kind") if isinstance(item, dict) else None
        if kind == "approve":
            return {"approve": False}
        return {"approve": True}

    out = _run_to_end(
        _kw(tmp_path, model, vision, approval=ApprovalPolicy(mode="default")),
        resolve=resolve_reject,
    )

    assert out["interrupted"] is None
    assert vision.train_calls == 0
    assert out["state"]["metrics"]["ok"] is False


def test_audit_chain_valid_across_resume(tmp_path) -> None:
    vision = FakeVision()
    model = FakeModel(["继续", "停止"])
    kw = _kw(tmp_path, model, vision)
    out = _run_to_end(kw)

    assert out["interrupted"] is None
    # 跨多次 resume 后，重新打开账本 verify 仍绿（digest 链没断）
    reopened = AuditLedger.open(kw["audit"].path)
    assert reopened.verify()["ok"] is True
    assert len(reopened) == out["audit"]["events"]


def test_review_approve_lands_proposals(tmp_path) -> None:
    """H1 修复：review 通过后必须把 proposals 落地（apply_proposals 被调用）。"""
    vision = FakeVision()
    model = FakeModel(["停止"])
    out = _run_to_end(_kw(tmp_path, model, vision))

    assert out["interrupted"] is None
    assert vision.apply_calls == 1
    assert out["state"]["applied"]["applied"] == 2
    assert vision.last_overwrite is False  # _resolve 的 review 没带 overwrite → 默认 False


def test_review_reject_does_not_apply(tmp_path) -> None:
    vision = FakeVision()
    model = FakeModel(["停止"])

    def resolve_reject(item: Any) -> dict[str, Any]:
        if isinstance(item, dict) and item.get("kind") == "review":
            return {"approve": False}
        return {"approve": True}

    out = _run_to_end(_kw(tmp_path, model, vision), resolve=resolve_reject)

    assert out["interrupted"] is None
    assert vision.apply_calls == 0  # 审核没过 → 不落地 proposals
    # train 仍会跑（在旧 labels 上），当前图语义是「每轮都训练，只有 approve 才换新标签」


def test_apply_failure_surfaces(tmp_path) -> None:
    """review 批准了但 apply_proposals 落地失败 → 返回的 audit.ok 必须 False（fail-closed）。"""
    vision = FakeVision()

    def fail_apply(dataset: str, **kw: Any) -> dict[str, Any]:
        return {"error": "permission denied"}

    vision.apply_proposals = fail_apply  # type: ignore[method-assign]
    model = FakeModel(["停止"])
    out = _run_to_end(_kw(tmp_path, model, vision))

    assert out["interrupted"] is None
    assert out["audit"]["ok"] is False
    assert out["audit"]["apply_errors"]
    assert "permission denied" in out["audit"]["apply_errors"][0]["error"]


def test_audit_write_failure_surfaces(tmp_path, monkeypatch) -> None:
    """F1 修复：审计 append 失败必须让返回的 audit.ok=False 并带 write_errors。"""
    vision = FakeVision()
    model = FakeModel(["停止"])
    ledger = AuditLedger.create(tmp_path / "events.jsonl")

    def boom(type: str, payload: dict[str, Any] | None = None) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr(ledger, "append", boom)
    out = _run_to_end(
        dict(dataset="demo/dataset", model=model, vision=vision, checkpointer=MemorySaver(), audit=ledger)
    )

    assert out["interrupted"] is None
    assert out["audit"]["ok"] is False
    assert out["audit"]["write_errors"]
    assert any(e["type"] in ("label", "review", "train", "analyze") for e in out["audit"]["write_errors"])


def test_sqlite_saver_smoke(tmp_path) -> None:
    """F2 修复：同步 invoke 走 SqliteSaver 的同步 get_tuple/put/put_writes，不该崩。"""
    with SqliteSaver.from_conn_string(str(tmp_path / "ckpt.sqlite")) as checkpointer:
        vision = FakeVision()
        model = FakeModel(["停止"])
        out = _run_to_end(
            dict(
                dataset="demo/dataset",
                model=model,
                vision=vision,
                checkpointer=checkpointer,
                audit=AuditLedger.create(tmp_path / "events.jsonl"),
            )
        )

    assert out["interrupted"] is None
    assert out["state"]["stop_reason"] == "done"
    assert out["audit"]["ok"] is True


def test_goal_completed_on_done(tmp_path) -> None:
    """循环以 done 收尾时，目标应被标记 completed（可寻址记忆：目标反映真实结局）。"""
    store = GoalStore(tmp_path / "goals")
    goal = store.create("把 mAP50 提到 0.9")
    vision = FakeVision()
    model = FakeModel(["停止"])
    kw = _kw(tmp_path, model, vision)
    kw.update(goal_store=store, goal_id=goal.goal_id, goal=goal)
    out = _run_to_end(kw)

    assert out["interrupted"] is None
    assert out["goal_status"] == "completed"
    assert store.get(goal.goal_id).status == "completed"


def test_goal_stays_active_on_max_iterations(tmp_path) -> None:
    """触顶（max_iterations）不算收敛，目标保持 active 待续跑，不能误标 completed。"""
    store = GoalStore(tmp_path / "goals")
    goal = store.create("无限逼近 0.95")
    vision = FakeVision()
    model = FakeModel(["继续"] * 10)
    kw = _kw(tmp_path, model, vision)
    kw.update(goal_store=store, goal_id=goal.goal_id, goal=goal, max_iterations=2)
    out = _run_to_end(kw)

    assert out["interrupted"] is None
    assert out["goal_status"] == "active"
    assert store.get(goal.goal_id).status == "active"


def test_goal_missing_reported(tmp_path) -> None:
    """goal_id 不存在时返回 goal_status="missing"，不抛异常。"""
    store = GoalStore(tmp_path / "goals")
    vision = FakeVision()
    model = FakeModel(["停止"])
    kw = _kw(tmp_path, model, vision)
    kw.update(goal_store=store, goal_id="does-not-exist")
    out = _run_to_end(kw)

    assert out["interrupted"] is None
    assert out["goal_status"] == "missing"


def test_goal_writeback_error_surfaces_but_keeps_audit(tmp_path) -> None:
    """goal 写回失败（如版本冲突）→ goal_status 记 error，但不翻转 audit.ok。"""

    class BrokenStore:
        def get(self, goal_id: str):
            raise RuntimeError("io error")

    store = BrokenStore()
    vision = FakeVision()
    model = FakeModel(["停止"])
    kw = _kw(tmp_path, model, vision)
    kw.update(goal_store=store, goal_id="g-any")
    out = _run_to_end(kw)

    assert out["interrupted"] is None
    assert out["goal_status"].startswith("error:")
    assert out["audit"]["ok"] is True  # 蒸馏自身的审计仍绿，goal 簿记错误单独透出


def test_goal_not_completed_on_reject(tmp_path) -> None:
    """H1 修复（f2ffc731）：人类驳回 review(approve=False) 不是"目标达成"，不能写 completed。"""
    store = GoalStore(tmp_path / "goals")
    goal = store.create("把 mAP50 提到 0.9")
    vision = FakeVision()
    model = FakeModel(["停止"])
    kw = _kw(tmp_path, model, vision)
    kw.update(goal_store=store, goal_id=goal.goal_id, goal=goal)

    def resolve_reject(item: Any) -> dict[str, Any]:
        if isinstance(item, dict) and item.get("kind") == "review":
            return {"approve": False}
        return {"approve": True}

    out = _run_to_end(kw, resolve=resolve_reject)

    assert out["interrupted"] is None
    # 驳回：stop_reason=="done" 但 review.approve=False → 不算达成，保持 active
    assert out["goal_status"] == "active"
    assert store.get(goal.goal_id).status == "active"


def test_select_hard_failure_flips_audit_ok(tmp_path) -> None:
    """M2 修复（ce7492b4）：select_hard 失败(ok=False) 必须传导到 audit.ok，不能误报主动学习成功。"""
    vision = FakeVision()

    def fail_select_hard(dataset: str, **kw: Any) -> dict[str, Any]:
        return {"hard": [], "count": 0, "ok": False, "errors": ["no torch"]}

    vision.select_hard = fail_select_hard  # type: ignore[method-assign]
    model = FakeModel(["继续", "停止"])  # 第一轮"继续"触发 select_hard → 失败
    out = _run_to_end(_kw(tmp_path, model, vision))

    assert out["interrupted"] is None
    assert out["audit"]["ok"] is False
    assert out["audit"]["active_errors"] == ["no torch"]
